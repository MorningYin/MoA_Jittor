# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

"""
LLaMA-Adapter 推理脚本
功能：使用训练好的LLaMA-Adapter模型进行数学问题求解
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

# 设置混合精度
jt.flags.auto_mixed_precision_level = 1


def load_data(data_path):
    """加载数据文件"""
    ann = []
    with open(data_path, "r", encoding='utf8') as f:
        # 尝试加载为完整JSON数组
        try:
            data = json.load(f)
            if isinstance(data, list):
                ann = data
            else:
                ann = [data]
        except json.JSONDecodeError:
            # 按行读取JSONL格式
            f.seek(0)
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    ann.append(obj)
    return ann


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
    model.load_state_dict(adapter_ckpt['trainable_params'][1])
    
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model


def split_list(lst, size):
    """将列表按指定大小分割成批次"""
    return [lst[i:i+size] for i in range(0, len(lst), size)]


def main(
    ckpt_dir: str = '/hy-tmp/LLaMA/original',          # LLaMA基础模型目录
    adapter_path: str = '/root/MoA_Jittor/Output/LoRA_0-32_r8_a8_Q,K,V,O,FFN_UP_Prompt_0-32_len10_PAdapter_0-32_size16_swi_x1_lr5e-5_bs7_AddSub_seed125/checkpoint-4.pth',       # Adapter权重文件路径
    data_path: str = '/root/MoA_Jittor/Data/Dataset/math_commonsense/SVAMP/test.json',          # 输入数据文件路径
    save_path: str = '/root/MoA_Jittor/Test_seed125',           # 输出结果保存路径
    temperature: float = 0.1, # 生成温度参数
    top_p: float = 0.75,     # top-p采样参数
    max_seq_len: int = 300,  # 最大序列长度
    max_gen_len: int = 128,  # 最大生成长度
    min_gen_len: int = 64,   # 最小生成长度
    max_batch_size: int = 32, # 最大批次大小
    if_save_type: str = False, # 是否保存结果
):
    """主函数：使用LLaMA-Adapter模型进行数学问题求解"""

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

    # 数学问题提示模板
    math_prompt = [
        "You are a math tutor. Solve the following word problem step by step. First, carefully read and understand the problem. Then, break down the problem into smaller steps and solve each step logically. Show your work clearly and explain your reasoning. Finally, provide the final answer.\n\n",
        "### Problem:",
        "### Solution:",    
        "### Answer:"
    ]

    # 加载数据
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
        
        # 构建提示
        for x in batch:
            prompt0 = math_prompt[0]
            prompt1 = math_prompt[1]
            instruction = x['instruction']
            prompt2 = math_prompt[2]

            # 编码各个部分
            prompt0_token = model.tokenizer.encode(prompt0, bos=True, eos=False)
            prompt1_token = model.tokenizer.encode(prompt1, bos=False, eos=False)
            instruction_token = model.tokenizer.encode(instruction, bos=False, eos=False)
            prompt2_token = model.tokenizer.encode(prompt2, bos=False, eos=False)

            part1_token = prompt0_token + prompt1_token
            part2_token = prompt2_token

            # 计算最大输入长度
            max_input_length = max_seq_len - (len(part1_token) + len(part2_token) + min_gen_len)
            instruction_token = instruction_token[:max_input_length]
            prompt = part1_token + instruction_token + part2_token

            output = x['output']
            answer = x['answer']

            prompts.append(prompt)
            outputs.append(output)
            answers.append(answer)

        # 生成文本
        results = model.generate(prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, get_weights=(i == 0) and if_save_type, save_path=save_path)
        
        # 保存结果并评估
        with open(final_save_path, "a", encoding="utf-8") as f:
            for i, result in enumerate(results):
                # 提取答案
                answer_start = result.find("### Answer:")
                if answer_start != -1:
                    generated_answer = result[answer_start + len("### Answer:"):].strip()
                else:
                    generated_answer = ""
                
                # 答案对比
                try:
                    gen_val = float(generated_answer.strip())
                    true_val = float(answers[i].strip())
                    is_correct = abs(gen_val - true_val) < 1e-6
                except ValueError:
                    is_correct = generated_answer.strip() == answers[i].strip()
                
                correct_count += int(is_correct)
                total_count += 1

                # 保存结果
                tmp = {
                    'generate': result,
                    'output': outputs[i],
                    'instruction': prompts[i],
                    'answer': answers[i],
                    'generated_answer': generated_answer,
                    'is_correct': is_correct
                }
                f.write(json.dumps(tmp, ensure_ascii=False) + "\n")
        
        # 分布式同步
        if jt.in_mpi:
            jt.sync_all()

    # 计算并保存正确率
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    with open(final_save_path, "a", encoding="utf-8") as f:
        f.write(f"\nAccuracy: {correct_count}/{total_count} = {accuracy:.4f}\n")


if __name__ == "__main__":
    fire.Fire(main)
