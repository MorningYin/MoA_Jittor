import jittor as jt
from jittor.dataset import Dataset
import json
from Models.Tokenizer import Tokenizer
import numpy as np
from typing import List


class MathDataset(Dataset):
    """微调数据集类，支持多种数据格式"""
    
    def __init__(self, 
                 data_paths: List[str], 
                 tokenizer_path: str, 
                 max_tokens: int = 512, 
                 partition: str = "train",
                 val_ratio: float = None,
                 **kwargs):

        self.kwargs = kwargs

        self.math_prompt = [
        "You are a math tutor. Solve the following word problem step by step. First, carefully read and understand the problem. Then, break down the problem into smaller steps and solve each step logically. Show your work clearly and explain your reasoning. Finally, provide the final answer.\n\n",
        "### Problem:",
        "### Solution:",    
        "### Answer:"
        ]

        self.ann = []
        for data_path in data_paths:
            self.ann.extend(self.load_data(data_path))

        # 2. 生成一个可复现的乱序索引
        num_samples = len(self.ann)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # 3. 按比例划分
        if val_ratio is not None:
            val_size = int(num_samples * val_ratio)
            if partition == "train":
                selected = indices[val_size:]   # 后面的给训练集
            else:
                selected = indices[:val_size]   # 前面的给验证集
        else:
            selected = indices

        # 4. 用索引子集重组 self.ann
        self.ann = [self.ann[i] for i in selected]
        
        self.max_tokens = max_tokens
        tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenizer = tokenizer

    def load_data(self, data_path):
        
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
    
    def __len__(self):
        return super().__len__()

    def datainit(self):
        """初始化数据集"""
        super().__init__(**self.kwargs)
        self.total_len = len(self.ann)

    def __getitem__(self, index):
        """获取单个样本"""
        ann = self.ann[index]

        prompt1 = self.math_prompt[0] + self.math_prompt[1]
        input = ann['question']
        prompt2 = self.math_prompt[2]
        output = ann['chain-of-thought'] + self.math_prompt[3] + str(ann['answer'])

        # 编码各部分
        part1_token = self.tokenizer.encode(prompt1, bos=True, eos=False)

        input_token = self.tokenizer.encode(input, bos=False, eos=False)
        prompt2_token = self.tokenizer.encode(prompt2, bos=False, eos=False)
        output_token = self.tokenizer.encode(output, bos=False, eos=True)
        
        # 计算最大输入长度，确保总长度不超过限制
        max_input_length = self.max_tokens - (len(part1_token) + len(prompt2_token) + len(output_token))
        input_token = input_token[:max_input_length]
        
        prompt_tokens = part1_token + input_token + prompt2_token
        example_tokens = prompt_tokens + output_token

        prompt = jt.array(prompt_tokens, dtype='int32')
        example = jt.array(example_tokens, dtype='int32')

        # 填充或截断到指定长度
        padding = self.max_tokens - example.shape[0]
        if padding > 0:
            pad_arr = jt.full((padding,), -1, dtype='int32')
            example = jt.concat([example, pad_arr], dim=0)
        elif padding < 0:
            example = example[-self.max_tokens:]

        # 创建标签，只对输出部分计算损失
        labels = example.clone()
        labels[: len(prompt)] = -1  # 提示部分不计算损失

        # 创建掩码
        example_mask = (example >= 0).astype(jt.float16)
        label_mask = (labels >= 0).astype(jt.float16)

        # 应用掩码
        example = jt.where(example_mask, example, jt.zeros_like(example))
        labels = jt.where(label_mask, labels, jt.zeros_like(labels))

        return example, labels, example_mask


class MathDataset_test(Dataset):
    """微调数据集类，支持多种数据格式"""
    
    def __init__(self, 
                data_paths: str, 
                tokenizer_path: str, 
                max_seq_len = 300,    
                max_gen_len: int = 128,
                min_gen_len: int = 64,   
                max_batch_size: int = 32,
                 **kwargs):

        self.kwargs = kwargs

        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.min_gen_len = min_gen_len
        self.max_batch_size = max_batch_size

        self.math_prompt = [
        "You are a math tutor. Solve the following word problem step by step. First, carefully read and understand the problem. Then, break down the problem into smaller steps and solve each step logically. Show your work clearly and explain your reasoning. Finally, provide the final answer.\n\n",
        "### Problem:",
        "### Solution:",    
        "### Answer:"
        ]

        self.ann = self.load_data(data_paths)
        tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenizer = tokenizer

    def load_data(self, data_path):
        
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
    
    def __len__(self):
        return super().__len__()

    def datainit(self):
        """初始化数据集"""
        super().__init__(**self.kwargs)
        self.total_len = len(self.ann)

    def __getitem__(self, index):
        """获取单个样本"""
        x = self.ann[index]

        prompt0 = self.math_prompt[0]
        prompt1 = self.math_prompt[1]
        instruction = x['instruction']
        prompt2 = self.math_prompt[2]
        prompt3 = self.math_prompt[3]

        # 分别编码各个部分
        prompt0_token = self.tokenizer.encode(prompt0, bos=True, eos=False) # bos
        prompt1_token = self.tokenizer.encode(prompt1, bos=False, eos=False)
        instruction_token = self.tokenizer.encode(instruction, bos=False, eos=False)
        prompt2_token = self.tokenizer.encode(prompt2, bos=False, eos=False)
        prompt3_token = self.tokenizer.encode(prompt3, bos=False, eos=False)

        part1_token = prompt0_token + prompt1_token
        part2_token = prompt2_token + prompt3_token

        # 计算最大输入长度，确保有足够空间生成输出
        max_input_length = self.max_seq_len - (len(part1_token) + len(part2_token) + self.min_gen_len)

        # 截断输入文本
        instruction_token = instruction_token[:max_input_length]
        prompt = part1_token + instruction_token + prompt2_token

        output = x['output']
        answer = x['answer']

        return prompt, output, answer

