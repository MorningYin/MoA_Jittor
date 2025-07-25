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

from Models.LLaMA_Adapter import LLaMA_adapter
from Models.ModelArgs import ModelArgs
from Utils.misc import setup_for_distributed
from Data.MathDataset import MathDataset_test

jt.flags.auto_mixed_precision_level = 1


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


def load_data(data_path):
        
        ann = []
        with open(data_path, "r", encoding='utf8') as f:
            # 尝试加载为完整JSON数组
            try:
                data = json.load(f)
                if isinstance(data, list):
                    # 如果是JSON数组，直接使用
                    ann = data
                else:
                    # 如果是单个对象，包装成列表
                    ann = [data]
            except json.JSONDecodeError:
                # 如果失败，尝试按行读取JSONL格式
                f.seek(0)  # 重置文件指针
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:  # 跳过空行
                        obj = json.loads(line)
                        ann.append(obj)

        return ann

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

    adapter_params = adapter_ckpt['args']
    
    # 如果指定了新的最大序列长度，则更新配置
    if max_seq_len and (max_seq_len > adapter_params.max_seq_len):
        adapter_params.max_seq_len = max_seq_len

    adapter_params.max_batch_size = max_batch_size
    model_args: ModelArgs = adapter_params

    llama_type = ''
    llama_ckpt_dir = os.path.join(llama_path, llama_type)
    llama_tokenzier_path = os.path.join(llama_path, 'tokenizer.model')
    model = LLaMA_adapter(model_args, llama_ckpt_dir, llama_tokenzier_path)

    # 加载adapter权重到模型
    model.load_state_dict(adapter_ckpt['trainable_params'][1])
    
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
    # assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    # return model.to(device)
    return model


def split_list(lst, size):
    """
    将列表按指定大小分割成批次
    """
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def main(
    ckpt_dir: str = '/hy-tmp/LLaMA/original',          # LLaMA基础模型目录
    adapter_path: str = '/root/MoA_Jittor/Output/LoRA_0-32_r8_a8_Q_MoE_expert7_swi_x1_lr5e-5_bs8_AddSub_seed125/checkpoint-3.pth',       # Adapter权重文件路径
    data_path: str = '/root/MoA_Jittor/Data/Dataset/math_commonsense/AddSub/test.json',          # 输入数据文件路径
    save_path: str = '/root/MoA_Jittor/Test_seed125_MoE',           # 输出结果保存路径
    temperature: float = 0.1, # 生成温度参数
    top_p: float = 0.75,     # top-p采样参数
    max_seq_len: int = 300,  # 最大序列长度
    max_gen_len: int = 128,  # 最大生成长度
    min_gen_len: int = 64,   # 最小生成长度
    max_batch_size: int = 32, # 最大批次大小
    if_save_type: str = True, # 是否保存结果
):
    """
    主函数：使用LLaMA-Adapter模型进行文本生成
    
    支持的数据集格式：
    - CovidET: 新冠疫情事件追踪数据集
    - QMSum: 会议摘要数据集
    - 新闻摘要数据集
    """

    jt.flags.use_cuda = 1

    local_rank = 0
    world_size = 1
    
    if jt.in_mpi:
        # 设置模型并行环境
        local_rank = jt.rank
        world_size = jt.world_size
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

    # 加载模型
    model = load(ckpt_dir, adapter_path, max_seq_len=max_seq_len, max_batch_size=max_batch_size)
    model.float_auto()
    model.eval()

    math_prompt = [
        "You are a math tutor. Solve the following word problem step by step. First, carefully read and understand the problem. Then, break down the problem into smaller steps and solve each step logically. Show your work clearly and explain your reasoning. Finally, provide the final answer.\n\n",
        "### Problem:",
        "### Solution:",    
        "### Answer:"
    ]

    ann = load_data(data_path)

    # 数据分片处理（多GPU并行）
    local_num = math.ceil(len(ann)/world_size)
    local_ann = ann[local_rank*local_num:(local_rank+1)*local_num]
    batchs = split_list(local_ann, max_batch_size)
    print(f'local examples:{len(local_ann)}')

    max_seq_len = model.llama.params.max_seq_len
    
    # 初始化正确率统计
    correct_count = 0
    total_count = 0

    # 构建保存路径
    adapter_name = os.path.basename(os.path.dirname(adapter_path))
    data_name = os.path.basename(os.path.dirname(data_path))
    save_path = os.path.join(save_path, data_name)
    os.makedirs(save_path, exist_ok=True)
    filename = f"{adapter_name}_{data_name}.txt"
    final_save_path = os.path.join(save_path, filename)

    # 批量生成文本
    for i, batch in enumerate(tqdm(batchs)):
        prompts = []
        outputs = []
        answers = []
        for x in batch:
            prompt0 = math_prompt[0]
            prompt1 = math_prompt[1]
            instruction = x['instruction']
            prompt2 = math_prompt[2]

            # 分别编码各个部分
            prompt0_token = model.tokenizer.encode(prompt0, bos=True, eos=False) # bos
            prompt1_token = model.tokenizer.encode(prompt1, bos=False, eos=False)
            instruction_token = model.tokenizer.encode(instruction, bos=False, eos=False)
            prompt2_token = model.tokenizer.encode(prompt2, bos=False, eos=False)

            part1_token = prompt0_token + prompt1_token
            part2_token = prompt2_token

            # 计算最大输入长度，确保有足够空间生成输出
            max_input_length = max_seq_len - (len(part1_token) + len(part2_token) + min_gen_len)

            # 截断输入文本
            instruction_token = instruction_token[:max_input_length]
            prompt = part1_token + instruction_token + part2_token

            output = x['output']
            answer = x['answer']

            prompts.append(prompt)
            outputs.append(output)
            answers.append(answer)

        # 使用模型生成文本
        results = model.generate(prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, get_weights=(i == 0) and if_save_type, save_path=save_path)
        
        # 保存结果并对比答案
        with open(final_save_path, "a", encoding="utf-8") as f:
            for i, result in enumerate(results):
                # 提取答案
                answer_start = result.find("### Answer:")
                if answer_start != -1:
                    generated_answer = result[answer_start + len("### Answer:"):].strip()
                else:
                    generated_answer = ""  # 找不到则为空
                
                # 对比答案：尝试数值比较，如果失败则字符串比较
                try:
                    gen_val = float(generated_answer.strip())
                    true_val = float(answers[i].strip())
                    is_correct = abs(gen_val - true_val) < 1e-6  # 浮点数近似相等
                except ValueError:
                    # 非数字，回退到字符串匹配
                    is_correct = generated_answer.strip() == answers[i].strip()
                
                correct_count += int(is_correct)
                total_count += 1

                tmp = {
                    'generate': result,
                    'output': outputs[i],
                    'instruction': prompts[i],
                    'answer': answers[i],
                    'generated_answer': generated_answer,
                    'is_correct': is_correct
                }
                f.write(json.dumps(tmp, ensure_ascii=False) + "\n")
        
        # 在每个 batch 后同步（分布式环境）
        if jt.in_mpi:
            jt.sync_all()

    # 计算并保存正确率
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    with open(final_save_path, "a", encoding="utf-8") as f:
        f.write(f"\nAccuracy: {correct_count}/{total_count} = {accuracy:.4f}\n")

if __name__ == "__main__":
    fire.Fire(main)
