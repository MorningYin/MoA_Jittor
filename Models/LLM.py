import jittor as jt
from jittor import nn
from typing import Optional, Tuple, cast
from .ModelArgs import ModelArgs
from .Blok import Attention, FeedForward, RMSNorm, PAdapterLayer, Router
from .utils import precompute_freqs_cis, inference_mode_jt


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, w_lora=False, w_prompt=False, w_padapter=False):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(args, w_lora=w_lora, w_prompt=w_prompt)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier, args=args,
            w_lora=w_lora
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.w_padapter = w_padapter
        if self.w_padapter:
            self.p_adapter = PAdapterLayer(self.dim, args.p_adapter_size, args.expert_num, args.p_adapter_hydra)
        
        # ==================== 统计各类适配器数量 ====================
        # adapter_type 表示当前层总共挂载了多少种“可切换”子模块
        # attention_type 仅统计会作用于 Attention 的适配器（Q/K/V/O/Prompt）
        # FFN_type       仅统计作用于 FFN 的适配器（FFN_*）
        self.adapter_type = 0
        self.attention_type = 0
        self.FFN_type = 0
        if w_lora:
            lora_targets = args.lora_targets.split(',')
            self.adapter_type += len(lora_targets)
            attention_targets = ['Q', 'K', 'V', 'O']
            FFN_targets = ['FFN_UP', 'FFN_GATE', 'FFN_DOWN']
            for x in lora_targets:
                if x in attention_targets:
                    self.attention_type += 1
                if x in FFN_targets:
                    self.FFN_type += 1
        if w_prompt:
            self.adapter_type += 1
            self.attention_type += 1
        if w_padapter:
            self.adapter_type += 1
        
        if args.swi_x == 0:
            # 直接用 Linear 做路由：输出维度 = adapter_type
            self.adapter_type_router = nn.Linear(args.dim, self.adapter_type)
        elif args.swi_x > 0:
            # 使用两层 MLP(或 SwiGLU) 做更复杂的路由
            # 隐藏层宽度 = adapter_type * swi_x
            self.adapter_type_router = Router(args.dim, self.adapter_type * args.swi_x, self.adapter_type)


    # Jittor 中子模块的前向接口用 execute
    def execute(self, x: jt.Var, start_pos: int, freqs_cis: jt.Var, mask: Optional[jt.Var]):

        # ==================== 计算类型权重 ====================
        # adapter_type_router 输出 logits，sigmoid/softmax 后得到 [bsz, seqlen, adapter_type]
        # 这里用 sigmoid，使每个适配器权重独立在 0~1 之间，可看作门控值
        # 若希望互斥选择，可改用 softmax
        # type_weights 的切片顺序： [Attention相关权重 | FFN相关权重 | P-Adapter权重]
        # 具体数量由 attention_type / FFN_type 决定
        type_weights = jt.sigmoid(self.adapter_type_router(x)).astype(x.dtype)   # [bsz, seqlen, adapter_type]  # type: ignore[arg-type]
        type_idx = 0
        h = x + self.attention(
            self.attention_norm(x),
            start_pos,
            freqs_cis,
            mask,
            type_weight=type_weights[:,:,type_idx:type_idx+self.attention_type]  # 分配给 Attention
        )
        # out = h + self.feed_forward.forward(self.ffn_norm(h))
        type_idx += self.attention_type
        residual = h
        h = self.ffn_norm(h)
        out = self.feed_forward(
            h,
            type_weight=type_weights[:,:,type_idx:type_idx+self.FFN_type]      # 分配给 FFN
        )
        type_idx += self.FFN_type
        if self.w_padapter:
            # P-Adapter 只有一个权重通道，取剩余的最后一维
            adapter_states = self.p_adapter(h, type_weight=type_weights[:,:,type_idx])
            out = out + adapter_states
        out = residual + out

        # 避免 float16 溢出（Jittor: jt.clamp）
        out = jt.clamp(out, -65500, 65500)
        return out.astype(x.dtype)

        # router 分布, 不统计时需要注释掉
        # batch sum
        # sum_weights = torch.sum(type_weights, (0,1)) # [adapter_type]
        # return out, sum_weights

        # router case, 不统计时需要注释掉
        # weights = type_weights # [bsz, seqlen, adapter_type]
        # return out, weights

class Transformer(nn.Module):
    # ==================== 架构核心：Transformer ====================
    def __init__(self, params: ModelArgs):
        """
        初始化 Transformer 主干模型。

        主要流程：
        1. 保存模型关键超参数（词表大小、层数等）。
        2. 创建词嵌入矩阵 `tok_embeddings`。
        3. 解析 LoRA、Parallel Adapter、Prompt Tuning 需要作用的层号，分别存入
           `lora_layers_id` / `p_adapter_layers_id` / `prompt_layers_id`。
        4. 根据层号循环构建 `TransformerBlock`，并按需为每层启用 LoRA、Adapter 或 Prompt。
        5. 构造输出层前的归一化 `norm` 以及最终线性投影 `output`。
        6. 预计算旋转位置编码用到的复数常数 `freqs_cis`，便于后续快速取用。
        """
        super().__init__()

        # -------- 保存关键超参数 --------
        # 这些属性在推理 / 训练及其它方法中都会频繁使用
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        # -------- 创建词嵌入 --------
        # 将形如 "0-8,12-16" 的字符串解析为具体层索引列表
        self.lora_layers_id = [x for span in params.lora_layers.split(',') for x in range(int(span.split('-')[0]), int(span.split('-')[1]))]
        print(f'lora_layers_id:{self.lora_layers_id}')

        self.p_adapter_layers_id = [x for span in params.p_adapter_layers.split(',') for x in range(int(span.split('-')[0]), int(span.split('-')[1]))]
        print(f'p_adapter_layers_id:{self.p_adapter_layers_id}')

        self.prompt_layers_id = [x for span in params.prompt_layers.split(',') for x in range(int(span.split('-')[0]), int(span.split('-')[1]))]
        print(f'prompt_layers_id:{self.prompt_layers_id}')

        # -------- 构建每层 TransformerBlock --------
        # 遍历全部层，根据是否位于指定列表决定启用 LoRA / Adapter / Prompt
        # 使用 Jittor 的 ModuleList 以便 load_state_dict 能正确解析路径
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            block = TransformerBlock(
                layer_id, params,
                w_lora   = (layer_id in self.lora_layers_id),
                w_prompt = (layer_id in self.prompt_layers_id),
                w_padapter = (layer_id in self.p_adapter_layers_id),
            )
            # 关键：把模块也绑定成属性，这样 Jittor 才能发现它
            setattr(self, f"layer_{layer_id}", block)
            self.layers.append(block)

        # -------- 输出层前归一化 & 线性投影 --------
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        # -------- 预计算旋转位置编码常数 --------
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            bool(params.use_scaled_rope),
        )

        self.freqs_cis = jt.float16(self.freqs_cis)


    @inference_mode_jt
    def execute(self, tokens: jt.Var, start_pos: int):
        shape = cast(Tuple[int, int], tokens.shape())  # type: ignore
        _bsz, seqlen = shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]  # type: ignore[index]

        mask = None
        if seqlen > 1:
            mask = jt.full((1, 1, seqlen, seqlen), jt.array(float('-inf')), dtype=h.dtype)
            mask = jt.triu(mask, diagonal=start_pos + 1)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.astype(jt.float32)