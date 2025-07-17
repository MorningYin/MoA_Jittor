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
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        初始化Tokenizer，使用Tiktoken分词器
        
        该分词器基于Tiktoken实现，支持LLaMA模型的文本编码和解码。
        主要功能包括：
        1. 文本分词和编码
        2. Token ID解码为文本
        3. 特殊Token处理
        4. 长文本分段处理
        
        Args:
            model_path (str): Tiktoken模型文件路径
        """
        assert os.path.isfile(model_path), model_path

        # ==================== 加载分词模型 ====================
        mergeable_ranks = load_tiktoken_bpe(model_path)  # 加载BPE合并规则
        num_base_tokens = len(mergeable_ranks)  # 基础词汇表大小
        
        # ==================== 特殊Token定义 ====================
        # 定义LLaMA模型使用的特殊Token
        special_tokens = [
            "<|begin_of_text|>",      # 文本开始标记
            "<|end_of_text|>",        # 文本结束标记
            "<|reserved_special_token_0|>",  # 保留特殊Token 0
            "<|reserved_special_token_1|>",  # 保留特殊Token 1
            "<|reserved_special_token_2|>",  # 保留特殊Token 2
            "<|reserved_special_token_3|>",  # 保留特殊Token 3
            "<|start_header_id|>",    # 头部开始标记
            "<|end_header_id|>",      # 头部结束标记
            "<|reserved_special_token_4|>",  # 保留特殊Token 4
            "<|eot_id|>",             # 对话轮次结束标记
        ] + [
            # 动态生成剩余的保留特殊Token
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        
        # ==================== 特殊Token映射 ====================
        # 将特殊Token映射到Token ID（在基础词汇表之后）
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        
        # ==================== 创建Tiktoken编码器 ====================
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,        # 模型名称
            pat_str=self.pat_str,              # 分词模式字符串
            mergeable_ranks=mergeable_ranks,   # BPE合并规则
            special_tokens=self.special_tokens, # 特殊Token映射
        )
        logger.info(f"Reloaded tiktoken model from {model_path}")

        # ==================== 词汇表信息设置 ====================
        self.n_words: int = self.model.n_vocab  # 总词汇表大小
        
        # ==================== 关键Token ID设置 ====================
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]  # 文本开始Token ID
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]    # 文本结束Token ID
        self.pad_id: int = -1  # 填充Token ID（-1表示不使用）
        
        # ==================== 停止Token设置 ====================
        # 定义生成时的停止Token集合
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],  # 文本结束Token
            self.special_tokens["<|eot_id|>"],       # 对话轮次结束Token
        }
        
        # ==================== 日志输出 ====================
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
        """
        将字符串编码为Token ID列表
        
        该方法是分词器的核心功能，将输入文本转换为模型可以理解的数字序列。
        支持长文本自动分段处理，避免内存溢出问题。
        
        Args:
            s (str): 要编码的输入字符串
            bos (bool): 是否在序列开头添加开始标记
            eos (bool): 是否在序列末尾添加结束标记
            allowed_special ("all"|set[str]): 允许在字符串中出现的特殊Token
            disallowed_special ("all"|Collection[str]): 在字符串中出现时会报错的特殊Token
            
        Returns:
            list[int]: Token ID列表
            
        Note:
            - 默认情况下，disallowed_special=()会忽略所有特殊Token，将其作为普通文本处理
            - 设置allowed_special="all"会将所有特殊Token文本作为特殊Token处理
            - 支持长文本自动分段，避免Tiktoken的字符限制问题
        """
        assert type(s) is str

        # ==================== 长文本处理设置 ====================
        # Tiktoken分词器可以处理不超过400k字符的文本，避免pyo3_runtime.PanicException
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # 参考：https://github.com/openai/tiktoken/issues/195
        # 这里我们迭代处理子序列，如果超过连续非空白字符或空白字符的限制就分割
        MAX_NO_WHITESPACES_CHARS = 25_000

        # ==================== 文本分段处理 ====================
        # 将长文本分割为可处理的子序列
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)  # 按最大字符数分段
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        
        # ==================== 编码处理 ====================
        t: List[int] = []  # 存储编码结果
        for substr in substrs:
            # 对每个子序列进行编码
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        
        # ==================== 添加特殊标记 ====================
        if bos:
            t.insert(0, self.bos_id)  # 在开头添加开始标记
        if eos:
            t.append(self.eos_id)      # 在末尾添加结束标记
            
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        将Token ID列表解码为字符串
        
        该方法是编码的逆操作，将数字序列转换回可读的文本。
        
        Args:
            t (List[int]): 要解码的Token ID列表
            
        Returns:
            str: 解码后的字符串
            
        Note:
            - 类型转换在这里是安全的，Tiktoken不会对序列进行列表相关操作
        """
        # 类型转换在这里是安全的。Tiktoken不会对序列进行列表相关操作
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        分割字符串，确保每个子字符串包含不超过指定长度的连续空白字符或非空白字符
        
        这是一个辅助方法，用于处理超长文本，避免Tiktoken的字符限制问题。
        
        Args:
            s (str): 要分割的字符串
            max_consecutive_slice_len (int): 最大连续字符长度
            
        Yields:
            Iterator[str]: 分割后的子字符串迭代器
            
        Note:
            - 该方法确保每个子字符串要么全是空白字符，要么全是非空白字符
            - 当连续字符超过限制时，会在适当位置分割
            - 这种分割方式保持了文本的语义完整性
        """
        # ==================== 初始化变量 ====================
        current_slice_len = 0  # 当前连续字符长度
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False  # 当前字符类型（空白/非空白）
        slice_start = 0  # 当前子字符串的起始位置

        # ==================== 遍历字符串进行分割 ====================
        for i in range(len(s)):
            is_now_space = s[i].isspace()  # 当前字符是否为空白

            # ==================== 字符类型变化检测 ====================
            # 使用异或操作检测字符类型是否发生变化
            if current_slice_is_space ^ is_now_space:
                # 字符类型发生变化，重置计数器
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                # 字符类型相同，增加计数器
                current_slice_len += 1
                # ==================== 长度限制检查 ====================
                if current_slice_len > max_consecutive_slice_len:
                    # 超过长度限制，输出当前子字符串并重新开始
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        
        # ==================== 输出最后一个子字符串 ====================
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
