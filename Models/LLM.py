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
        
        # 统计各类适配器数量
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
            # 直接用 Linear 做路由
            self.adapter_type_router = nn.Linear(args.dim, self.adapter_type).float16()
        elif args.swi_x > 0:
            # 使用两层 MLP(或 SwiGLU) 做更复杂的路由
            self.adapter_type_router = Router(args.dim, self.adapter_type * args.swi_x, self.adapter_type)

    def execute(self, x: jt.Var, start_pos: int, freqs_cis: jt.Var, mask: Optional[jt.Var]):
        # 计算类型权重
        type_weights = jt.sigmoid(self.adapter_type_router(x)).astype(x.dtype)
        type_idx = 0
        h = x + self.attention(
            self.attention_norm(x),
            start_pos,
            freqs_cis,
            mask,
            type_weight=type_weights[:,:,type_idx:type_idx+self.attention_type]
        )
        type_idx += self.attention_type
        residual = h
        h = self.ffn_norm(h)
        out = self.feed_forward(
            h,
            type_weight=type_weights[:,:,type_idx:type_idx+self.FFN_type]
        )
        type_idx += self.FFN_type
        if self.w_padapter:
            adapter_states = self.p_adapter(h, type_weight=type_weights[:,:,type_idx])
            out = out + adapter_states
        out = residual + out

        # 避免 float16 溢出
        out = jt.clamp(out, -65500, 65500)
        return out.astype(x.dtype)

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """初始化 Transformer 主干模型"""
        super().__init__()

        # 保存关键超参数
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim, dtype='float16'
        )
        
        print(f'================================================== 微调模块细节 ====================================================')
        # 解析适配器层范围
        self.lora_layers_id = [x for span in params.lora_layers.split(',') for x in range(int(span.split('-')[0]), int(span.split('-')[1]))]
        print(f'lora_layers_id:{self.lora_layers_id}')

        self.p_adapter_layers_id = [x for span in params.p_adapter_layers.split(',') for x in range(int(span.split('-')[0]), int(span.split('-')[1]))]
        print(f'p_adapter_layers_id:{self.p_adapter_layers_id}')

        self.prompt_layers_id = [x for span in params.prompt_layers.split(',') for x in range(int(span.split('-')[0]), int(span.split('-')[1]))]
        print(f'prompt_layers_id:{self.prompt_layers_id}')

        # 构建每层 TransformerBlock
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            block = TransformerBlock(
                layer_id, params,
                w_lora   = (layer_id in self.lora_layers_id),
                w_prompt = (layer_id in self.prompt_layers_id),
                w_padapter = (layer_id in self.p_adapter_layers_id),
            )
            setattr(self, f"layer_{layer_id}", block)
            self.layers.append(block)

        # 输出层前归一化 & 线性投影
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        ).float16()

        # 预计算旋转位置编码常数
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            bool(params.use_scaled_rope),
        )

    @inference_mode_jt
    def execute(self, tokens: jt.Var, start_pos: int):
        shape = cast(Tuple[int, int], tokens.shape())
        _bsz, seqlen = shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = jt.full((1, 1, seqlen, seqlen), jt.array(float('-inf')), dtype=h.dtype)
            mask = jt.triu(mask, diagonal=start_pos + 1)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.astype(jt.float32)