import os
import json
import time
from pathlib import Path

import torch

from .LLM import Transformer
from .Tokenizer import Tokenizer
from .ModelArgs import ModelArgs

import jittor as jt
import jittor.nn as nn

def sample_top_p(probs, p):
    """
    核采样函数，从概率分布中采样，只考虑累积概率达到p的top tokens
    """
    sorted_indices = jt.argsort(probs, descending=True)
    # 重新计算排序后的概率
    sorted_probs = jt.gather(probs, dim=-1, index=sorted_indices)
    cumulative_probs = jt.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
    sorted_indices_to_remove[..., 0] = False
    # 创建掩码
    indices_to_remove = jt.zeros_like(probs, dtype='bool')
    for i in range(sorted_indices.shape[-1]):
        mask = sorted_indices_to_remove[..., i:i+1]
        idx = sorted_indices[..., i:i+1]
        indices_to_remove = indices_to_remove.scatter(dim=-1, index=idx, src=mask)
    probs = jt.where(indices_to_remove, jt.zeros_like(probs), probs)
    return jt.multinomial(probs, num_samples=1)


def to_float16(raw_sd):
    """
    将 state_dict 中的权重统一转换为 float16（Jittor Var）。
    """
    # 原地将所有张量转为 Jittor Var(float16)
    for k, v in raw_sd.items():
        if isinstance(v, torch.Tensor):
            # 转到 CPU → float16 → numpy → Jittor Var
            arr = v.detach().cpu().to(torch.float16).numpy()
            raw_sd[k] = jt.array(arr)
        elif isinstance(v, jt.Var):
            raw_sd[k] = v.float16()
        # 其余类型保持不变


class LLaMA_adapter(nn.Module):

    def __init__(self, args, llama_ckpt_dir, llama_tokenizer):
        """
        初始化LLaMA适配器模型
        
        该方法是MoA（Mixture of Adapters）模型的核心初始化函数，负责：
        1. 加载预训练LLaMA模型配置
        2. 设置模型参数和精度
        3. 初始化分词器
        4. 创建Transformer模型
        5. 加载预训练权重
        6. 配置训练参数和损失函数
        7. 设置可训练参数
        
        Args:
            args: 训练参数对象，包含所有模型配置
            llama_ckpt_dir (str): LLaMA预训练模型检查点目录路径
            llama_tokenizer (str): LLaMA分词器模型文件路径
        """
        super().__init__()
        
        # ==================== 加载LLaMA模型配置 ====================
        # 从预训练模型目录加载params.json配置文件
        # 该文件包含模型的基础架构参数（维度、层数、头数等）
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        
        # ==================== 精度设置 ====================
        # -------------------- 精度设置 --------------------
        # Jittor 未直接提供 "is_bf16_supported"，这里采用运行时检测：
        # 若用户请求 bf16 且能够成功创建 bfloat16 Tensor，则认为支持。
        # if args.bf16:
        #     try:
        #         _ = jt.ones(1, dtype='bfloat16')  # 尝试创建 bf16 张量
        #         bf16 = True
        #     except Exception:
        #         bf16 = False
        #         print("------bfloat16 is not supported-----")
        # else:
        #     bf16 = False

        # ==================== 创建模型参数配置 ====================
        # 构建ModelArgs对象，合并用户参数和预训练模型参数
        model_args: ModelArgs = ModelArgs(
            # 基础训练参数
            max_seq_len=args.max_seq_len,        # 最大序列长度
            max_batch_size=args.max_batch_size,  # 最大批次大小（仅影响推理）
            w_bias=args.w_bias,                  # 是否微调偏置项
            
            # LoRA参数
            lora_layers=args.lora_layers,        # LoRA应用的层范围
            lora_rank=args.lora_rank,            # LoRA的秩
            lora_targets=args.lora_targets,      # LoRA目标模块
            lora_alpha=args.lora_alpha,          # LoRA缩放参数
            expert_num=args.expert_num,          # 专家数量
            swi_x=args.swi_x,                    # 适配器类型路由参数
            hydra_moe=args.hydra_moe,            # 是否启用Hydra MoE
            
            # Parallel Adapter参数
            p_adapter_layers=args.p_adapter_layers,  # Parallel Adapter层范围
            p_adapter_size=args.p_adapter_size,      # Parallel Adapter大小
            p_adapter_hydra=args.p_adapter_hydra,    # Parallel Adapter Hydra模式
            
            # Prompt Tuning参数
            prompt_layers=args.prompt_layers,    # Prompt Tuning层范围
            prompt_len=args.prompt_len,          # Prompt长度
            expert_weight=args.expert_weight,    # 专家权重设置
            
            # 预训练模型参数（从params.json加载）
            **params
        )
        self.model_args = model_args  # 保存模型参数配置

        # ==================== 初始化分词器 ====================
        # 创建Tokenizer实例，用于文本编码和解码
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # ==================== 创建Transformer模型 ====================
        # 验证词汇表大小一致性
        assert model_args.vocab_size == self.tokenizer.n_words
        
        # ==================== 设置计算设备 ====================
        # Jittor 自动选择设备，无需显式设置默认 device

        # ==================== 创建Transformer模型实例 ====================
        # 根据配置参数创建Transformer模型
        self.llama = Transformer(model_args)
        self.llama = self.llama.float16()

        # print(self.llama.children())

        #         # ==================== 设置模型精度 ====================
        # # 根据配置设置模型的默认数据类型
        # if model_args.bf16:
        #     self.llama.bfloat16()                      # 使用bfloat16
        #     print('-----bfloat16 for llama-----')
        # else:
        #     self.llama.float16()                       # 使用float16
        #     print('-------float16 for llama-----')

        for ckpt_path in sorted(Path(llama_ckpt_dir).glob("*.pth")):
            try:
                state_dict = torch.load(str(ckpt_path), map_location=torch.device('cpu'))
            except Exception as e:
                print(f"[Warning] jt.load 加载 {ckpt_path.name} 失败: {e}")
                continue

            to_float16(state_dict)

            # Jittor 的 load_state_dict 不返回 (missing, unexpected)，只会在内部打印失败项
            self.llama.load_state_dict(state_dict)
            print(f"[Info] 加载 {ckpt_path.name} 完成")

        # ==================== 配置训练损失函数 ====================
        # 使用交叉熵损失函数，忽略填充token（ID为0）
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # ==================== 设置可训练参数 ====================
        # 调用方法设置哪些参数需要梯度更新
        self.get_trainable_params()

        # ==================== 打印可训练参数信息 ====================
        # 遍历所有参数，打印需要梯度的参数信息
        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")

    def get_trainable_params(self):
        """
        设置模型的可训练参数
        
        该方法负责配置哪些模型参数需要梯度更新，实现参数高效微调：
        1. 首先冻结所有参数（requires_grad = False）
        2. 根据配置选择性地解冻特定参数
        3. 支持偏置项微调、LoRA、Prompt Tuning、Adapter等参数高效方法
        
        参数选择策略：
        - 如果启用偏置微调：解冻所有norm层和bias参数
        - 如果使用LoRA：解冻所有LoRA相关参数
        - 如果使用Prompt Tuning：解冻所有prompt相关参数
        - 如果使用Adapter：解冻所有adapter相关参数
        """
        # ==================== 冻结所有参数 ====================
        # 首先将所有参数设置为不可训练（冻结预训练权重）
        for name, para in self.named_parameters():
            para.stop_grad()  # 冻结梯度

        # ==================== 选择性解冻参数 ====================
        for name, para in self.named_parameters():
            # ==================== 偏置项微调 ====================
            # 如果启用偏置微调，解冻norm层和bias参数
            if name.startswith("llama."):
                if self.model_args.w_bias:
                    if 'norm' in name or 'bias' in name:
                        para.start_grad()  # 解冻
                        
                # ==================== 参数高效微调参数 ====================
                # 解冻所有LoRA、Prompt Tuning、Adapter相关参数
                if 'lora' in name or 'prompt' in name or 'adapter' in name:
                    para.stop_grad()

    def execute(self, tokens, labels, prompt_mask):
        """
        模型前向传播（训练模式）
        
        该方法执行模型的前向传播，用于训练阶段：
        1. 将输入tokens转换为嵌入表示
        2. 应用位置编码（RoPE）
        3. 通过Transformer层进行特征提取
        4. 计算损失函数
        
        Args:
            tokens (torch.Tensor): 输入的token序列，形状为 [batch_size, seq_len]
            labels (torch.Tensor): 标签序列，形状为 [batch_size, seq_len]
            prompt_mask (torch.Tensor): prompt掩码，用于区分prompt和生成部分
            
        Returns:
            tuple: (c_loss, c_loss) - 分类损失（这里返回两个相同的损失值）
        """
        # ==================== 获取输入维度信息 ====================
        _bsz, seqlen = tokens.shape  # 批次大小和序列长度

        # ==================== Token嵌入 ====================
        # 将token ID转换为嵌入向量
        h = self.llama.tok_embeddings(tokens)
        
        # ==================== 位置编码设置 ====================
        # 获取预计算的旋转位置编码（RoPE）
        freqs_cis = self.llama.freqs_cis[:seqlen]  # type: ignore[index]  # 截取到当前序列长度
        
        # ==================== 注意力掩码创建 ====================
        # 创建因果掩码（causal mask），确保模型只能看到当前位置及之前的信息
        mask = jt.full((1, 1, seqlen, seqlen), float('-inf'), dtype=h.dtype)
        mask = jt.triu(mask, diagonal=1)
        
        # ==================== Transformer层前向传播 ====================
        # 逐层通过Transformer块
        for layer in self.llama.layers:
            h = layer(h, 0, freqs_cis, mask)  # 0表示起始位置

        # ==================== 最终输出处理 ====================
        h = self.llama.norm(h)  # 应用最终层归一化
        output = self.llama.output(h)  # 线性投影到词汇表大小
        
        # ==================== 序列对齐 ====================
        # 将输出和标签对齐，用于计算损失
        output = output[:, :-1, :]  # 去掉最后一个token的输出
        labels = labels[:, 1:]       # 去掉第一个token的标签

        # ==================== 损失计算 ====================
        if labels.sum() == 0:
            # 如果没有有效标签，返回零损失
            c_loss = output.mean() * 0
        else:
            # 计算交叉熵损失
            # 将输出重塑为 [batch_size * seq_len, vocab_size]
            # 将标签重塑为 [batch_size * seq_len]
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())

        # ==================== 返回结果 ====================
        # 返回两个相同的损失值（为了兼容某些接口）
        return c_loss, c_loss


    @jt.no_grad()
    def generate_unified(
        self, 
        prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
        enable_time_stat: bool = False,
        enable_router_stat: bool = False,
        enable_router_case: bool = False
    ):
        """
        统一的文本生成方法，支持多种功能模式
        
        Args:
            prompts: 输入提示列表，每个元素是token序列
            max_gen_len: 最大生成长度
            temperature: 温度参数，控制生成的随机性（0为确定性生成）
            top_p: 核采样参数，控制词汇表截断范围
            enable_time_stat: 是否启用时间统计
            enable_router_stat: 是否启用路由器统计
            enable_router_case: 是否启用路由器案例分析（仅支持batch_size=1）
            
        Returns:
            decoded: 解码后的生成文本列表
            time_cost: 时间成本统计（仅在enable_time_stat=True时返回）
            layer_stat: 路由器统计信息（仅在enable_router_stat或enable_router_case=True时返回）
        """
        # ==================== 参数验证 ====================
        bsz = len(prompts)  # 批次大小
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        # 路由器案例分析仅支持batch_size=1
        if enable_router_case:
            assert bsz == 1, "路由器案例分析仅支持batch_size=1"
        
        # ==================== 序列长度计算 ====================
        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        # ==================== Token序列初始化 ====================
        tokens = jt.full((bsz, total_len), self.tokenizer.pad_id, dtype='int64')

        # ==================== 填充输入提示 ====================
        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = jt.array(t, dtype='int64')
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0

        # ==================== 统计信息初始化 ====================
        time_cost = None
        layer_stat = None
        
        if enable_time_stat:
            prefill_time = None
            end_time = None
            start_time = time.time()
            
        if enable_router_stat:
            layer_stat = {'token_num': 0}
            
        if enable_router_case:
            layer_stat = {}

        # ==================== 自回归生成循环 ====================
        for cur_pos in range(start_pos, total_len):
            # Jittor会自动处理精度，无需手动设置混合精度
            # 根据功能模式选择前向传播方法
            if enable_router_stat:
                logits, layer_stat = self.forward_inference_router_stat(tokens[:, prev_pos:cur_pos], prev_pos, layer_stat)
                layer_stat['token_num'] += bsz * (cur_pos - prev_pos)
            elif enable_router_case:
                logits, layer_stat = self.forward_inference_router_case(tokens[:, prev_pos:cur_pos], prev_pos, layer_stat)
            else:
                logits = self.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos)

            # ==================== 时间统计 ====================
            if enable_time_stat and cur_pos == start_pos:
                prefill_time = time.time()

            # ==================== Token采样 ====================
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            # ==================== Token更新和早停 ====================
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        # ==================== 时间成本计算 ====================
        if enable_time_stat and prefill_time is not None:
            end_time = time.time()
            all_cost = end_time - start_time
            inference_cost = end_time - prefill_time
            time_cost = {'all_cost': all_cost, 'inference_cost': inference_cost}

        # ==================== 结果解码 ====================
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        # ==================== 路由器案例分析处理 ====================
        if enable_router_case and layer_stat is not None:
            for i, t in enumerate(tokens.tolist()):
                t0 = t[:]
                t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
                try:
                    t0 = t0[: t0.index(self.tokenizer.eos_id)]
                    t = t[: t.index(self.tokenizer.eos_id)]
                except ValueError:
                    pass
                
                # 单词级别案例分析
                words = [self.tokenizer.decode([ti]) for ti in t0]
                if layer_stat and 0 in layer_stat and 'weights' in layer_stat[0]:
                    layer_stat['words'] = words
                    assert len(words) == layer_stat[0]['weights'].shape[1]

        # ==================== 返回结果 ====================
        if enable_time_stat and enable_router_stat:
            return decoded, time_cost, layer_stat
        elif enable_time_stat:
            return decoded, time_cost
        elif enable_router_stat or enable_router_case:
            return decoded, layer_stat
        else:
            return decoded

    # 为了保持向后兼容，保留原有的方法名作为包装器
    @torch.inference_mode()
    def generate(self, prompts, max_gen_len=256, temperature=0.1, top_p=0.75):
        """标准文本生成"""
        return self.generate_unified(prompts, max_gen_len, temperature, top_p)

    @torch.inference_mode()
    def generate_time(self, prompts, max_gen_len=256, temperature=0.1, top_p=0.75, time_gen=False):
        """带时间统计的文本生成"""
        if time_gen:
            return self.generate_unified(prompts, max_gen_len, temperature, top_p, enable_time_stat=True)
        else:
            return self.generate_unified(prompts, max_gen_len, temperature, top_p)

    @torch.inference_mode()
    def generate_router_stat(self, prompts, max_gen_len=256, temperature=0.1, top_p=0.75):
        """带路由器统计的文本生成"""
        return self.generate_unified(prompts, max_gen_len, temperature, top_p, enable_router_stat=True)

    @torch.inference_mode()
    def generate_router_case(self, prompts, max_gen_len=256, temperature=0.1, top_p=0.75):
        """带路由器案例分析的文本生成"""
        return self.generate_unified(prompts, max_gen_len, temperature, top_p, enable_router_case=True)
