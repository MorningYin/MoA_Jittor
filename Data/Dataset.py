import jittor as jt
from jittor.dataset import Dataset
import json
from Models.Tokenizer import Tokenizer
import random

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

# 分段提示模板
prompt_input = [(
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
),
"\n\n### Input:\n",
"\n\n### Response:"]


class FinetuneDataset(Dataset):
    """微调数据集类"""
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer_path: str, 
                 batch_size: int = 16,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 num_workers: int = 1,
                 max_tokens: int = 512, 
                 partition: str = "train"):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

        # 根据数据路径判断数据格式并加载
        if 'CovidET' in data_path or 'ma_news' in data_path or 'newts' in data_path:
            # CovidET格式：文章摘要任务
            ann = []
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
            self.ann = ann
        elif 'QMSum' in data_path:
            # QMSum格式：会议摘要任务
            ann = []
            with open(data_path, "r", encoding='utf8') as f:
                lines = f.readlines()
            for line in lines:
                obj = json.loads(line)
                ann.append(obj)
            self.ann = ann
        else:
            # Alpaca格式：标准指令微调格式
            self.ann = json.load(open(data_path))
        
        # 根据分区选择数据
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]  # 验证集取前200条
        
        self.max_tokens = max_tokens
        tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return super().__len__()

    def datainit(self):
        """初始化数据集"""
        super().__init__(batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last, num_workers=self.num_workers)
        self.total_len = len(self.ann)

    def __getitem__(self, index):
        """获取单个样本"""
        ann = self.ann[index]
        
        if ann.get("input", "") == "":
            # 无输入的情况
            prompt_str = PROMPT_DICT["prompt_no_input"].format_map(ann)
            example_str = prompt_str + ann["output"]

            prompt_tokens = self.tokenizer.encode(prompt_str, bos=True, eos=False)
            example_tokens = self.tokenizer.encode(example_str, bos=True, eos=True)

            prompt = jt.array(prompt_tokens, dtype='int32')
            example = jt.array(example_tokens, dtype='int32')
        else:
            # 有输入的情况
            prompt0 = prompt_input[0]
            instruction = ann['instruction']
            prompt1 = prompt_input[1]
            input = ann['input']
            prompt2 = prompt_input[2]
            output = ann['output']

            # 编码各部分
            prompt0_token = self.tokenizer.encode(prompt0, bos=True, eos=False)
            instruction_token = self.tokenizer.encode(instruction, bos=False, eos=False)
            prompt1_token = self.tokenizer.encode(prompt1, bos=False, eos=False)

            part1_token = prompt0_token + instruction_token + prompt1_token

            input_token = self.tokenizer.encode(input, bos=False, eos=False)
            prompt2_token = self.tokenizer.encode(prompt2, bos=False, eos=False)
            output_token = self.tokenizer.encode(output, bos=False, eos=True)
            
            # 计算最大输入长度
            max_input_length = self.max_tokens - (len(part1_token) + len(prompt2_token) + len(output_token))
            input_token = input_token[:max_input_length]
            
            prompt_tokens = part1_token + input_token + prompt2_token
            example_tokens = prompt_tokens + output_token

            prompt = jt.array(prompt_tokens, dtype='int32')
            example = jt.array(example_tokens, dtype='int32')

        # 填充或截断
        padding = self.max_tokens - example.shape[0]
        if padding > 0:
            pad_arr = jt.full((padding,), -1, dtype='int32')
            example = jt.concat([example, pad_arr], dim=0)
        elif padding < 0:
            example = example[-self.max_tokens:]

        # 创建标签和掩码
        labels = example.clone()
        labels[: len(prompt)] = -1

        example_mask = (example >= 0).astype(jt.float16)
        label_mask = (labels >= 0).astype(jt.float16)

        example = jt.where(example_mask, example, jt.zeros_like(example))
        labels = jt.where(label_mask, labels, jt.zeros_like(labels))

        return example, labels, example_mask


