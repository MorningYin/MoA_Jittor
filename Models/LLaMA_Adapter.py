import os
import json
import time
from pathlib import Path

from .LLM import Transformer
from .Tokenizer import Tokenizer
from .ModelArgs import ModelArgs

import jittor as jt
import jittor.nn as nn

def to_float16(raw_sd):
    """将 state_dict 中的权重统一转换为 float16"""
    for k, v in raw_sd.items():
        raw_sd[k] = v.float16()


class LLaMA_adapter(nn.Module):

    def __init__(self, args, llama_ckpt_dir, llama_tokenizer):
        """初始化LLaMA适配器模型"""
        super().__init__()
        
        # 加载LLaMA模型配置
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())

        # 创建模型参数配置
        model_args: ModelArgs = ModelArgs(
            # 基础训练参数
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
            w_bias=args.w_bias,
            
            # LoRA参数
            lora_layers=args.lora_layers,
            lora_rank=args.lora_rank,
            lora_targets=args.lora_targets,
            lora_alpha=args.lora_alpha,
            expert_num=args.expert_num,
            swi_x=args.swi_x,
            hydra_moe=args.hydra_moe,
            
            # Parallel Adapter参数
            p_adapter_layers=args.p_adapter_layers,
            p_adapter_size=args.p_adapter_size,
            p_adapter_hydra=args.p_adapter_hydra,
            
            # Prompt Tuning参数
            prompt_layers=args.prompt_layers,
            prompt_len=args.prompt_len,
            expert_weight=args.expert_weight,
            
            # 预训练模型参数
            **params
        )
        self.model_args = model_args

        # 初始化分词器
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 创建Transformer模型
        assert model_args.vocab_size == self.tokenizer.n_words
        
        self.llama = Transformer(model_args)
        self.llama = self.llama.float16()
        
        print(f'================================================== 加载预训练权重 ====================================================')
        # 加载预训练权重
        for ckpt_path in sorted(Path(llama_ckpt_dir).glob("*.pth")):
            try:
                state_dict = jt.load(str(ckpt_path))
            except Exception as e:
                print(f"[Warning] jt.load 加载 {ckpt_path.name} 失败: {e}")
                continue

            to_float16(state_dict)
            self.llama.load_state_dict(state_dict)
            print(f"[Info] 加载 {ckpt_path.name} 完成")

        # 配置训练损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 设置可训练参数
        self.get_trainable_params()

        self.best_model_state_dict = ('Epoch_0_0', {name: param.data.copy() for name, param in self.named_parameters() if param.requires_grad})

        # # 打印可训练参数信息
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #        print(f"Trainable param: {name}, {param.shape}, {param.dtype}")

    def get_trainable_params(self):
        """设置模型的可训练参数"""
        # 冻结所有参数
        for name, para in self.named_parameters():
            para.stop_grad()

        # 选择性解冻参数
        for name, para in self.named_parameters():
            # 偏置项微调
            if name.startswith("llama."):
                if self.model_args.w_bias:
                    if 'norm' in name or 'bias' in name:
                        para.start_grad()
                        
                # 参数高效微调参数
                if 'lora' in name or 'prompt' in name or 'adapter' in name or 'router' in name:
                    para.start_grad()

    def execute(self, tokens, labels, prompt_mask):
        """模型前向传播（训练模式）"""
        _bsz, seqlen = tokens.shape

        # Token嵌入
        h = self.llama.tok_embeddings(tokens)
        
        # 位置编码设置
        freqs_cis = self.llama.freqs_cis[:seqlen]
        
        # 注意力掩码创建
        mask = jt.full((1, 1, seqlen, seqlen), float('-inf'), dtype=h.dtype)
        mask = jt.triu(mask, diagonal=1)
        
        # Transformer层前向传播
        for i, layer in enumerate(self.llama.layers):
            h = layer(h, 0, freqs_cis, mask)

        # 最终输出处理
        h = self.llama.norm(h)
        output = self.llama.output(h)
        
        # 序列对齐
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        # 损失计算
        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())

        return c_loss, c_loss

    def forward_inference(self, tokens, start_pos: int):
        """
        推理阶段的前向传播方法
        
        Args:
            tokens: 输入的token序列，形状为 [batch_size, seq_len]
            start_pos: 当前处理序列的起始位置，用于缓存机制
            
        Returns:
            output: 最后一个位置的logits输出，形状为 [batch_size, vocab_size]
        """
        with jt.no_grad():
            # ==================== 输入处理 ====================
            _bsz, seqlen = tokens.shape  # 获取批次大小和序列长度
            h = self.llama.tok_embeddings(tokens)  # 将tokens转换为嵌入向量
            
            # ==================== 位置编码处理 ====================
            # 获取旋转位置编码（RoPE）
            freqs_cis = self.llama.freqs_cis[start_pos : start_pos + seqlen]
            
            # ==================== 注意力掩码创建 ====================
            mask = None  # TODO: check mask for cache
            # 创建因果掩码，确保模型只能看到当前位置及之前的信息
            mask = jt.full((1, 1, seqlen, seqlen), float("-inf"), dtype=h.dtype)
            # 上三角掩码，从1开始，实现因果注意力（调整为start_pos）
            mask = jt.triu(mask, diagonal=start_pos + 1)
            
            # ==================== Transformer层前向传播 ====================
            # 逐层通过Transformer块，传入起始位置用于缓存
            for layer in self.llama.layers:
                h = layer(h, start_pos, freqs_cis, mask)

            # ==================== 最终输出处理 ====================
            h = self.llama.norm(h)  # 应用最终层归一化
            # 只取最后一个位置的输出用于生成下一个token
            output = self.llama.output(h[:, -1, :])

            return output.float32()  # 返回float32精度的logits

    def generate(self, prompts, max_gen_len: int = 256, temperature: float = 0.1, top_p: float = 0.75):
        """
        文本生成方法，使用自回归方式生成文本
        
        Args:
            prompts: 输入提示列表，每个元素是token序列
            max_gen_len: 最大生成长度
            temperature: 温度参数，控制生成的随机性（0为确定性生成）
            top_p: 核采样参数，控制词汇表截断范围
            
        Returns:
            decoded: 解码后的生成文本列表
        """
        with jt.no_grad():
            # ==================== 参数验证和初始化 ====================
            bsz = len(prompts)  # 批次大小
            params = self.llama.params
            assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)  # 验证批次大小

            # ==================== 序列长度计算 ====================
            min_prompt_size = min([len(t) for t in prompts])  # 最小提示长度
            max_prompt_size = max([len(t) for t in prompts])   # 最大提示长度
            total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)  # 总序列长度

            # ==================== Token序列初始化 ====================
            # 创建填充的token序列，初始化为pad_id
            tokens = jt.full((bsz, total_len), self.tokenizer.pad_id, dtype=jt.int64)

            # ==================== 填充输入提示 ====================
            # 将每个提示的token填充到序列中
            for k, t in enumerate(prompts):
                tokens[k, : len(t)] = jt.array(t, dtype=jt.int64)
            input_text_mask = tokens != self.tokenizer.pad_id  # 创建输入文本掩码
            start_pos = min_prompt_size  # 生成起始位置
            prev_pos = 0  # 前一个位置
            
            # ==================== 自回归生成循环 ====================
            for cur_pos in range(start_pos, total_len):
                # 使用混合精度进行推理（如果需要，调整amp_level）
                with jt.flag_scope(amp_level=1):
                    logits = self.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos)
                    
                # ==================== Token采样 ====================
                if temperature > 0:
                    # 使用温度采样
                    probs = jt.nn.softmax(logits / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)  # 核采样
                else:
                    # 确定性生成（贪婪解码）
                    next_token = jt.argmax(logits, dim=-1)
                next_token = next_token.reshape(-1)

                # ==================== Token更新和早停 ====================
                # 保持输入文本不变，只更新生成部分
                next_token = jt.ternary(
                    input_text_mask[:, cur_pos],
                    tokens[:, cur_pos],
                    next_token
                )
                tokens[:, cur_pos] = next_token
                
                # 技巧：当批次大小为1且遇到结束符时提前停止
                if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                    break
                prev_pos = cur_pos

            # ==================== 结果解码 ====================
            decoded = []
            for i, t in enumerate(tokens.numpy().tolist()):  # 转换为list
                # 截取到最大生成长度
                t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
                # 如果遇到结束符则截断
                try:
                    t = t[: t.index(self.tokenizer.eos_id)]
                except ValueError:
                    pass
                decoded.append(self.tokenizer.decode(t))  # 解码为文本

            return decoded