import jittor as jt
from jittor.dataset import Dataset
import json
from Models.Tokenizer import Tokenizer
import copy
import os

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

prompt_input = [(
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n"
                ),
                "\n\n### Input:\n",
                "\n\n### Response:"]

class FinetuneDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer_path: str, 
                 batch_size: int = 16,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 num_workers: int = 0,
                 max_tokens: int = 512, 
                 partition: str = "train"):

        super().__init__(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

        
        # CovidET
        if 'CovidET' in data_path or 'ma_news' in data_path or 'newts' in data_path:
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
            ann = []
            with open(data_path, "r", encoding='utf8') as f:
                lines = f.readlines()
            for line in lines:
                obj = json.loads(line)
                ann.append(obj)
            self.ann = ann
        else:
        # alpaca
            self.ann = json.load(open(data_path))
        
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]
        
        self.max_tokens = max_tokens
        tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt_str = PROMPT_DICT["prompt_no_input"].format_map(ann)
            example_str = prompt_str + ann["output"]

            prompt_tokens = self.tokenizer.encode(prompt_str, bos=True, eos=False)
            example_tokens = self.tokenizer.encode(example_str, bos=True, eos=True)

            prompt = jt.array(prompt_tokens, dtype='int32')
            example = jt.array(example_tokens, dtype='int32')
        else:
            # prompt = PROMPT_DICT["prompt_input"].format_map(ann)
            prompt0 = prompt_input[0]
            instruction = ann['instruction']
            prompt1 = prompt_input[1]
            input = ann['input']
            prompt2 = prompt_input[2]
            output = ann['output']

            prompt0_token = self.tokenizer.encode(prompt0, bos=True, eos=False)  # bos
            instruction_token = self.tokenizer.encode(instruction, bos=False, eos=False)
            prompt1_token = self.tokenizer.encode(prompt1, bos=False, eos=False)

            part1_token = prompt0_token + instruction_token + prompt1_token

            input_token = self.tokenizer.encode(input, bos=False, eos=False)
            prompt2_token = self.tokenizer.encode(prompt2, bos=False, eos=False)
            output_token = self.tokenizer.encode(output, bos=False, eos=True)  # eos
            
            max_input_length = self.max_tokens - (len(part1_token) + len(prompt2_token) + len(output_token))

            input_token = input_token[:max_input_length]
            prompt_tokens = part1_token + input_token + prompt2_token
            example_tokens = prompt_tokens + output_token

            prompt = jt.array(prompt_tokens, dtype='int32')
            example = jt.array(example_tokens, dtype='int32')

        padding = self.max_tokens - example.shape[0]
        if padding > 0:
            pad_arr = jt.full((padding,), -1, dtype='int32')
            example = jt.concat([example, pad_arr], dim=0)
        elif padding < 0:
            example = example[-self.max_tokens:]

        labels = example.clone()
        labels[: len(prompt)] = -1  # loss only for labels

        example_mask = (example >= 0).astype(jt.float16)
        label_mask = (labels >= 0).astype(jt.float16)

        example = jt.where(example_mask, example, jt.zeros_like(example))
        labels = jt.where(label_mask, labels, jt.zeros_like(labels))

        return example, labels, example_mask