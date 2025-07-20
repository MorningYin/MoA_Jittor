# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

"""
LLaMA-Adapter 推理脚本
功能：使用训练好的LLaMA-Adapter模型进行文本生成和摘要任务
支持多种数据集格式：CovidET、QMSum、新闻摘要等
"""

import json
import os
import sys
import time

import fire
from tqdm import tqdm
import math
import jittor as jt
from argparse import Namespace


from Models.LLaMA_Adapter import LLaMA_adapter
from Models.ModelArgs import ModelArgs
from Utils.misc import init_distributed_mode

# 预定义的提示模板字典
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# 分段式提示模板，用于更精细的token控制
prompt_input = [(
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n"
                ),
                "\n\n### Input:\n",
                "\n\n### Response:"]

def load(llama_path, adapter_path: str, max_seq_len=None, max_batch_size: int = 32):
    """
    加载LLaMA-Adapter模型
    
    参数:
    - llama_path: LLaMA基础模型路径
    - adapter_path: Adapter权重文件路径
    - max_seq_len: 最大序列长度
    - max_batch_size: 最大批次大小
    
    返回: 加载好的模型
    """
    start_time = time.time()
    # device="cuda" if torch.cuda.is_available() else "cpu"
    # 加载LLaMA-Adapter权重和模型配置
    print(f'Loading LLaMA-Adapter from {adapter_path}')
    adapter_ckpt = jt.load(adapter_path)

    # 加载adapter参数配置
    with open(os.path.join(os.path.dirname(adapter_path), 'adapter_params.json'), 'r') as f:
        adapter_params = json.loads(f.read())
    
    # 如果指定了新的最大序列长度，则更新配置
    if max_seq_len and (max_seq_len > adapter_params['max_seq_len']):
        adapter_params['max_seq_len'] = max_seq_len

    adapter_params['max_batch_size'] = max_batch_size
    model_args: ModelArgs = ModelArgs(
        **adapter_params
    )

    llama_type = ''
    llama_ckpt_dir = os.path.join(llama_path, llama_type)
    llama_tokenzier_path = os.path.join(llama_path, 'tokenizer.model')
    model = LLaMA_adapter(model_args, llama_ckpt_dir, llama_tokenzier_path)

    # 加载adapter权重到模型
    load_result = model.load_state_dict(adapter_ckpt, strict=False)
    
    # # 统计可训练参数数量并保存
    # trainable_params_sum = 0
    # trainable_params_kv = []
    # for key, val in model.named_parameters():
    #     if val.requires_grad:
    #         trainable_params_kv.append((key, val.shape))
    #         trainable_params_sum += torch.numel(val)
    # trainable = {'trainable_params': trainable_params_sum,
    #              'trainable_params_kv': trainable_params_kv}
    # with open(os.path.join(os.path.dirname(adapter_path), 'trainable.json'), 'w') as f:
    #     f.write(json.dumps(trainable, ensure_ascii=False))

    # 检查是否有意外的键
    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    # return model.to(device)
    return model


def split_list(lst, size):
    """
    将列表按指定大小分割成批次
    """
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def main(
    ckpt_dir: str,          # LLaMA基础模型目录
    adapter_path: str,       # Adapter权重文件路径
    data_path: str,          # 输入数据文件路径
    save_path:str,           # 输出结果保存路径
    temperature: float = 0.1, # 生成温度参数
    top_p: float = 0.75,     # top-p采样参数
    max_seq_len = None,      # 最大序列长度
    max_gen_len: int = 128,  # 最大生成长度
    min_gen_len: int = 30,   # 最小生成长度
    max_batch_size: int = 32, # 最大批次大小
):
    """
    主函数：使用LLaMA-Adapter模型进行文本生成
    
    支持的数据集格式：
    - CovidET: 新冠疫情事件追踪数据集
    - QMSum: 会议摘要数据集
    - 新闻摘要数据集
    """
    # 设置模型并行环境
    local_rank, world_size = init_distributed_mode(Namespace())
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    # 加载模型
    model = load(ckpt_dir, adapter_path, max_seq_len=max_seq_len, max_batch_size=max_batch_size)
    model.eval()

    # 加载和处理数据
    ann = []
    if 'CovidET' in data_path or 'newts' in data_path or 'ma_news' in data_path:
        # 处理CovidET、新闻摘要等数据集
        with open(data_path, "r", encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            obj = json.loads(line)
            source = obj['article']           # 原文
            aspect_phrases = obj['phrases']   # 方面短语
            target = obj['abstract']          # 目标摘要
            data = {}
            data['instruction'] = f'Write a summary from {aspect_phrases} perspective'
            data['input'] = source
            data['output'] = target
            ann.append(data)
    elif 'QMSum' in data_path:
        # 处理QMSum会议摘要数据集
        with open(data_path, "r", encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            obj = json.loads(line)
            ann.append(obj)
    else:
        # 处理其他格式的数据集
        ann = json.load(open(data_path))
            
    print(f'local rank:{local_rank},  world size:{world_size}')

    # 数据分片处理（多GPU并行）
    local_num = math.ceil(len(ann)/world_size)
    local_ann = ann[local_rank*local_num:(local_rank+1)*local_num]
    batchs = split_list(local_ann, max_batch_size)
    print(f'local examples:{len(local_ann)}')
    # batchs = [ann[47:57]]

    # 获取生成参数
    # with open(os.path.join(os.path.dirname(adapter_path), 'generate_params.json'), 'r') as f:
    #     generate_params = json.loads(f.read())
    #     max_seq_len = generate_params['max_seq_len']
    max_seq_len = model.llama.params.max_seq_len
    
    # 创建输出目录
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 批量生成文本
    for batch in tqdm(batchs):
        prompts = []
        for x in batch:
            if x.get("input", "") == "":
                # 无输入的情况，使用简单提示模板
                prompt = PROMPT_DICT["prompt_no_input"].format_map(x)
                prompt = jt.array(model.tokenizer.encode(prompt, bos=True, eos=False), dtype=jt.int32)
            else:
                # 有输入的情况，使用分段式提示模板以控制token长度
                prompt0 = prompt_input[0]
                instruction = x['instruction']
                prompt1 = prompt_input[1]
                input = x['input']
                prompt2 = prompt_input[2]

                # 分别编码各个部分
                prompt0_token = model.tokenizer.encode(prompt0, bos=True, eos=False) # bos
                instruction_token = model.tokenizer.encode(instruction, bos=False, eos=False)
                prompt1_token = model.tokenizer.encode(prompt1, bos=False, eos=False)

                part1_token = prompt0_token + instruction_token + prompt1_token

                input_token = model.tokenizer.encode(input, bos=False, eos=False)
                prompt2_token = model.tokenizer.encode(prompt2, bos=False, eos=False)
                
                # 计算最大输入长度，确保有足够空间生成输出
                max_input_length = max_seq_len - (len(part1_token) + len(prompt2_token) + min_gen_len)

                # 截断输入文本
                input_token = input_token[:max_input_length]
                prompt = part1_token + input_token + prompt2_token

            prompts.append(prompt)
                
        # 使用模型生成文本
        results = model.generate(prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)

        # 保存生成结果
        with open(save_path, 'a', encoding='utf-8') as f:
            for i,result in enumerate(results):
                tmp_result = {}
                tmp_result['generate'] = result          # 生成的文本
                tmp_result['output'] = batch[i]['output'] # 标准答案
                tmp_result['input'] = batch[i]['input']   # 输入文本
                tmp_result['instruction'] = batch[i]['instruction'] # 指令
                tmp_result['answer'] = batch[i].get('answer','') # 答案（如果有）
                json_data = json.dumps(tmp_result, ensure_ascii=False)
                f.write(json_data + '\n')
                # print(result)
                # print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
