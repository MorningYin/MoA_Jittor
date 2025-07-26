# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

"""
LLaMA-Adapter 推理脚本
功能：使用训练好的LLaMA-Adapter模型进行文本生成和摘要任务
"""

import json
import os
import sys
import time
import fire
from tqdm import tqdm
import math
import jittor as jt

from Models.LLaMA_Adapter import LLaMA_adapter
from Models.ModelArgs import ModelArgs
from Utils.misc import init_distributed_mode

# 设置混合精度
jt.flags.auto_mixed_precision_level = 1

# 预定义的提示模板
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

# 分段式提示模板
prompt_input = [(
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
),
"\n\n### Input:\n",
"\n\n### Response:"]


def load(llama_path, adapter_path: str, max_seq_len=None, max_batch_size: int = 32):
    """加载LLaMA-Adapter模型"""
    start_time = time.time()
    print(f'Loading LLaMA-Adapter from {adapter_path}')
    
    # 加载适配器权重
    adapter_ckpt = jt.load(adapter_path)
    adapter_params = adapter_ckpt['args']
    
    # 更新序列长度配置
    if max_seq_len and (max_seq_len > adapter_params.max_seq_len):
        adapter_params.max_seq_len = max_seq_len
    adapter_params.max_batch_size = max_batch_size
    model_args: ModelArgs = adapter_params

    # 初始化模型
    llama_type = ''
    llama_ckpt_dir = os.path.join(llama_path, llama_type)
    llama_tokenzier_path = os.path.join(llama_path, 'tokenizer.model')
    model = LLaMA_adapter(model_args, llama_ckpt_dir, llama_tokenzier_path)

    # 加载适配器权重
    load_result = model.load_state_dict(adapter_ckpt['trainable_params'])
    
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model


def split_list(lst, size):
    """将列表按指定大小分割成批次"""
    return [lst[i:i+size] for i in range(0, len(lst), size)]


def main(
    ckpt_dir: str = '/HOME/thzskj_wfeng34/thzskj_wfeng34_1/HDD_POOL/Meta-Llama-3-8B-Instruct/original',
    adapter_path: str = '/HOME/thzskj_wfeng34/thzskj_wfeng34_1/HDD_POOL/MoA_Jittor/output/commonsense_15k/LoRA_0-32_r8_a8_Q,K,V,O_Prompt_0-32_len10_PAdapter_0-32_size16_swi_x1_lr1e-4_bs16_commonsense_15k_seed1234/checkpoint-0.pth',
    data_path: str = '/HOME/thzskj_wfeng34/thzskj_wfeng34_1/HDD_POOL/MoA_Jittor/Data/Dataset/commonsense_15k/test.json',
    save_path:str = '/HOME/thzskj_wfeng34/thzskj_wfeng34_1/HDD_POOL/MoA_Jittor/output/Test',
    temperature: float = 0.1,
    top_p: float = 0.75,
    max_seq_len = None,
    max_gen_len: int = 128,
    min_gen_len: int = 30,
    max_batch_size: int = 32,
):
    """主函数：使用LLaMA-Adapter模型进行文本生成"""

    # 设置CUDA
    jt.flags.use_cuda = 1

    # 分布式训练设置
    local_rank = 0
    world_size = 1
    
    if jt.in_mpi:
        local_rank = jt.rank
        world_size = jt.world_size
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

    # 加载模型
    model = load(ckpt_dir, adapter_path, max_seq_len=max_seq_len, max_batch_size=max_batch_size)
    model.float_auto()
    model.eval()

    # 加载和处理数据
    ann = []
    if 'CovidET' in data_path or 'newts' in data_path or 'ma_news' in data_path:
        # 处理CovidET、新闻摘要等数据集
        with open(data_path, "r", encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            obj = json.loads(line)
            source = obj['article']
            aspect_phrases = obj['phrases']
            target = obj['abstract']
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

    # 数据分片处理
    local_num = math.ceil(len(ann)/world_size)
    local_ann = ann[local_rank*local_num:(local_rank+1)*local_num]
    batchs = split_list(local_ann, max_batch_size)
    print(f'local examples:{len(local_ann)}')

    max_seq_len = model.llama.params.max_seq_len
    
    # 批量生成文本
    for batch in tqdm(batchs):
        prompts = []
        for x in batch:
            if x.get("input", "") == "":
                # 无输入的情况
                prompt = PROMPT_DICT["prompt_no_input"].format_map(x)
                prompt = jt.array(model.tokenizer.encode(prompt, bos=True, eos=False), dtype=jt.int32)
            else:
                # 有输入的情况，使用分段式提示模板
                prompt0 = prompt_input[0]
                instruction = x['instruction']
                prompt1 = prompt_input[1]
                input = x['input']
                prompt2 = prompt_input[2]

                # 编码各部分
                prompt0_token = model.tokenizer.encode(prompt0, bos=True, eos=False)
                instruction_token = model.tokenizer.encode(instruction, bos=False, eos=False)
                prompt1_token = model.tokenizer.encode(prompt1, bos=False, eos=False)

                part1_token = prompt0_token + instruction_token + prompt1_token

                input_token = model.tokenizer.encode(input, bos=False, eos=False)
                prompt2_token = model.tokenizer.encode(prompt2, bos=False, eos=False)
                
                # 计算最大输入长度
                max_input_length = max_seq_len - (len(part1_token) + len(prompt2_token) + min_gen_len)
                input_token = input_token[:max_input_length]
                prompt = part1_token + input_token + prompt2_token

            prompts.append(prompt)
                
        # 生成文本
        results = model.generate(prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)

        # 构建保存路径
        adapter_name = os.path.basename(os.path.dirname(adapter_path))
        data_name = os.path.basename(os.path.dirname(data_path))
        filename = f"{adapter_name}_{data_name}.txt"
        final_save_path = os.path.join(save_path, filename)
        
        # 创建输出目录
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)

        # 保存结果
        with open(final_save_path, "a", encoding="utf-8") as f:
            for i, result in enumerate(results):
                tmp = {
                    'generate': result,
                    'output': batch[i]['output'],
                    'input': batch[i]['input'],
                    'instruction': batch[i]['instruction'],
                    'answer': batch[i].get('answer', '')
                }
                f.write(json.dumps(tmp, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
