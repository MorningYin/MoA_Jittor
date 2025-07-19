import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe


logger = getLogger(__name__)


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]


class Tokenizer:
    """Tokenizing and encoding/decoding text using the Tiktoken tokenizer."""

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """初始化Tokenizer，使用Tiktoken分词器"""
        assert os.path.isfile(model_path), model_path

        # 加载分词模型
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        
        # 特殊Token定义
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        
        # 特殊Token映射
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        
        # 创建Tiktoken编码器
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Reloaded tiktoken model from {model_path}")

        # 词汇表信息设置
        self.n_words: int = self.model.n_vocab
        
        # 关键Token ID设置
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        
        # 停止Token设置
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """将字符串编码为Token ID列表"""
        assert type(s) is str

        # 长文本处理设置
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000
        MAX_NO_WHITESPACES_CHARS = 25_000

        # 文本分段处理
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        
        # 编码处理
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        
        # 添加特殊标记
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
            
        return t

    def decode(self, t: Sequence[int]) -> str:
        """将Token ID列表解码为字符串"""
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """分割字符串，确保每个子字符串包含不超过指定长度的连续空白字符或非空白字符"""
        # 初始化变量
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        # 遍历字符串进行分割
        for i in range(len(s)):
            is_now_space = s[i].isspace()

            # 字符类型变化检测
            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                # 长度限制检查
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        
        # 输出最后一个子字符串
        yield s[slice_start:]


class ChatFormat:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens
