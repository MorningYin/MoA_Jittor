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
from Utils.misc import init_distributed_mode

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
    model.load_state_dict(adapter_ckpt['trainable_params'])
    
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
    ckpt_dir: str = '/HOME/thzskj_wfeng34/thzskj_wfeng34_1/HDD_POOL/Meta-Llama-3-8B-Instruct/original',          # LLaMA基础模型目录
    adapter_path: str = '/HOME/thzskj_wfeng34/thzskj_wfeng34_1/HDD_POOL/MoA_Jittor/output/commonsense_15k/LoRA_0-32_r8_a8_Q,K,V,O_Prompt_0-32_len10_PAdapter_0-32_size16_swi_x1_lr1e-4_bs16_commonsense_15k_seed1234/checkpoint-0.pth',       # Adapter权重文件路径
    data_path: str = '/HOME/thzskj_wfeng34/thzskj_wfeng34_1/HDD_POOL/MoA_Jittor/Data/Dataset/commonsense_15k/test.json',          # 输入数据文件路径
    save_path:str = '/HOME/thzskj_wfeng34/thzskj_wfeng34_1/HDD_POOL/MoA_Jittor/output/Test',           # 输出结果保存路径
    temperature: float = 0.1, # 生成温度参数
    top_p: float = 0.75,     # top-p采样参数
    max_seq_len = 300,      # 最大序列长度
    max_gen_len: int = 128,  # 最大生成长度
    min_gen_len: int = 64,   # 最小生成长度
    max_batch_size: int = 32, # 最大批次大小
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

    test_dataset = MathDataset_test(data_path, 
        model.tokenizer, 
        max_seq_len=max_seq_len, 
        max_gen_len=max_gen_len, 
        min_gen_len=min_gen_len, 
        max_batch_size=max_batch_size, 
        batch_size=16)

    results = []
                
    for prompts, output, answer in test_dataset:
        # 使用模型生成文本
        result = model.generate(prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        results.extend(result)

    # 构建保存路径
    adapter_name = os.path.basename(os.path.dirname(adapter_path))
    data_name    = os.path.basename(os.path.dirname(data_path))
    filename = f"{adapter_name}_{data_name}.txt"
    final_save_path = os.path.join(save_path, filename)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)

    # 然后打开文件（如果不存在就新建，如果存在则追加）
    with open(final_save_path, "a", encoding="utf-8") as f:
        for i, result in enumerate(results):
            tmp = {
                'generate':      result,
                'output':        batch[i]['output'],
                'input':         batch[i]['input'],
                'instruction':   batch[i]['instruction'],
                'answer':        batch[i].get('answer', '')
            }
            f.write(json.dumps(tmp, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
