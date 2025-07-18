import math
import jittor as jt
from jittor import nn
F = nn  # type: ignore
from typing import Optional
from .ModelArgs import ModelArgs
from .utils import apply_rotary_emb
from .utils import repeat_kv
from typing import cast


# Jittor 版 RMSNorm
class RMSNorm(nn.Module):
    """均方根归一化，相比LayerNorm更高效"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = jt.array(eps, dtype='float16')
        self.weight = nn.Parameter(jt.ones(dim, dtype='float16'))

    def _norm(self, x):
        """执行RMS归一化"""
        denom = jt.sqrt((x ** 2).mean(-1, keepdims=True) + self.eps)
        return x / denom

    def execute(self, x):
        """前向传播方法"""
        output = self._norm(x)
        return output * self.weight


class MOELoraLayer(nn.Module):
    """混合专家LoRA层，结合LoRA和MoE的适配器层"""
    
    def __init__(self, input_dim, output_dim, r, expert_num, lora_alpha:float=8, hydra=False):
        super().__init__()

        self.expert_num = expert_num
        self.hydra = hydra
        self.scaling = lora_alpha / r
        self.params_num = 0

        # 单专家模式（标准LoRA）
        if expert_num == 1:
            self.lora_A = nn.Linear(input_dim, r, bias=False).float16()
            self.lora_B = nn.Linear(r, output_dim, bias=False).float16()
            nn.init.constant_(self.lora_B.weight, 0.0)

        # 多专家模式（MoE LoRA）
        elif expert_num > 1:
            self.router = nn.Linear(input_dim, expert_num, bias=False).float16()
            
            # A矩阵设置
            if hydra:
                self.lora_A = nn.Linear(input_dim, r, bias=False).float16()
            else:
                self.lora_A_l = nn.ModuleList()
                for i in range(expert_num):
                    self.lora_A_l.append(nn.Linear(input_dim, r, bias=False).float16())
                
            # B矩阵设置
            self.lora_B_l = nn.ModuleList()
            for i in range(expert_num):
                self.lora_B_l.append(nn.Linear(r, output_dim, bias=False).float16())

            # 初始化B矩阵为零
            for linear in self.lora_B_l:
                nn.init.constant_(linear.weight, 0.0)
        else:
            raise Exception("The number of Experts is wrong")
    
    def params_count(self):
        """计算LoRA适配器的参数数量"""
        self.params_num = 0
        
        if self.expert_num == 1:
            self.params_num += int(self.lora_A.weight.numel())
            self.params_num += int(self.lora_B.weight.numel())

        elif self.expert_num > 1:
            self.params_num += int(self.router.weight.numel())
            
            if self.hydra:
                self.params_num += int(self.lora_A.weight.numel())
            else:
                for i in range(self.expert_num):
                    linear = cast(nn.Linear, self.lora_A_l[i])
                    self.params_num += int(linear.weight.numel())
                
            for i in range(self.expert_num):
                linear = cast(nn.Linear, self.lora_B_l[i])
                self.params_num += int(linear.weight.numel())
                
        return self.params_num

    def execute(self, x: jt.Var, type_weight: Optional[jt.Var]):
        """前向传播方法"""
        # 单专家模式（标准LoRA）
        if self.expert_num == 1:
            result = self.lora_B(self.lora_A(x)) * self.scaling
            result = jt.unsqueeze(type_weight, -1) * result
            return result
        
        # 多专家模式（MoE LoRA）
        route_weight = nn.softmax(self.router(x), dim=-1)
        route_weight = route_weight * jt.unsqueeze(type_weight, -1)

        result = None
        for i in range(self.expert_num):
            if self.hydra:
                b_layer = cast(nn.Linear, self.lora_B_l[i])
                tmp = jt.unsqueeze(route_weight[:,:,i], -1) * b_layer(self.lora_A(x)) * self.scaling
            else:
                a_layer = cast(nn.Linear, self.lora_A_l[i])
                b_layer = cast(nn.Linear, self.lora_B_l[i])
                tmp = jt.unsqueeze(route_weight[:,:,i], -1) * b_layer(a_layer(x)) * self.scaling
            
            if i == 0:
                result = tmp
            else:
                result = result + tmp
                
        return result
    

class PAdapterLayer(nn.Module):
    """并行适配器层，支持单专家和多专家模式"""
    
    def __init__(self, hidden_size, adapter_size, expert_num:int=1, hydra:bool=False):
        super(PAdapterLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.expert_num = expert_num
        self.hydra = hydra

        self.adapter_act_fn = nn.SiLU()

        # 单专家模式（标准Parallel Adapter）
        if expert_num == 1:
            self.down_proj = nn.Linear(hidden_size, adapter_size).float16()
            self.up_proj = nn.Linear(adapter_size, hidden_size).float16()
            
        # 多专家模式（MoE Parallel Adapter）
        elif expert_num > 1:
            self.router = nn.Linear(hidden_size, expert_num).float16()
            
            # 降维投影设置
            if hydra:
                self.down_proj = nn.Linear(hidden_size, adapter_size).float16()
            else:
                self.down_proj_l = nn.ModuleList()
                for i in range(expert_num):
                    self.down_proj_l.append(nn.Linear(hidden_size, adapter_size).float16())
                
            # 升维投影设置
            self.up_proj_l = nn.ModuleList()
            for i in range(expert_num):
                self.up_proj_l.append(nn.Linear(adapter_size, hidden_size).float16())
        else:
            raise Exception("The number of Experts is wrong")
        
        self.reset_parameters()

    def reset_parameters(self):
        """重置和初始化适配器参数"""
        if self.expert_num ==1:
            nn.init.xavier_uniform_(self.down_proj.weight, gain=1e-4)
            nn.init.xavier_uniform_(self.up_proj.weight, gain=1e-4)
            nn.init.constant_(self.down_proj.bias, 0.0)
            nn.init.constant_(self.up_proj.bias, 0.0)
            
        elif self.expert_num >1:
            if self.hydra:
                nn.init.xavier_uniform_(self.down_proj.weight, gain=1e-4)
                nn.init.constant_(self.down_proj.bias, 0.0)
            else:
                for i in range(self.expert_num):
                    nn.init.xavier_uniform_(self.down_proj_l[i].weight, gain=1e-4)
                    nn.init.constant_(self.down_proj_l[i].bias, 0.0)
                    
            for i in range(self.expert_num):
                nn.init.xavier_uniform_(self.up_proj_l[i].weight, gain=1e-4)
                nn.init.constant_(self.up_proj_l[i].bias, 0.0)

    def execute(self, x: jt.Var, type_weight: Optional[jt.Var]):
        """前向传播方法"""
        # 单专家模式（标准Parallel Adapter）
        if self.expert_num == 1:
            x = self.down_proj(x)
            x = self.adapter_act_fn(x)
            x = self.up_proj(x)
            x = x * jt.unsqueeze(type_weight, -1)
            return x 

        # 多专家模式（MoE Parallel Adapter）
        route_weight = nn.softmax(self.router(x), dim=-1)
        route_weight = route_weight * jt.unsqueeze(type_weight, -1)

        result = None
        for i in range(self.expert_num):
            if self.hydra:
                tmp = jt.unsqueeze(route_weight[:,:,i], -1) * self.up_proj_l[i](self.adapter_act_fn(self.down_proj(x)))
            else:
                tmp = jt.unsqueeze(route_weight[:,:,i], -1) * self.up_proj_l[i](self.adapter_act_fn(self.down_proj_l[i](x)))
            
            if i == 0:
                result = tmp
            else:
                result = result + tmp
                
        return result


class Router(nn.Module):
    """SwiGLU路由器，用于MoA架构中的适配器类型选择"""
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int 
    ):
        super().__init__()

        # SwiGLU结构组件
        self.w1 = nn.Linear(in_dim, hidden_dim).float16()
        self.w2 = nn.Linear(hidden_dim, out_dim).float16()
        self.w3 = nn.Linear(in_dim, hidden_dim).float16()
        
        # 偏置初始化
        nn.init.constant_(self.w1.bias, 0)
        nn.init.constant_(self.w2.bias, 0)
        nn.init.constant_(self.w3.bias, 0)
    
    def execute(self, x):
        """前向传播方法"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        args: ModelArgs,
        w_lora=False
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=args.w_bias).float16()
        self.w2 = nn.Linear(hidden_dim, dim, bias=args.w_bias).float16()
        self.w3 = nn.Linear(dim, hidden_dim, bias=args.w_bias).float16()
        if args.w_bias:
            nn.init.constant_(self.w1.bias, 0)
            nn.init.constant_(self.w2.bias, 0)
            nn.init.constant_(self.w3.bias, 0)
        self.w_lora = w_lora
        if self.w_lora:
            self.lora_targets = args.lora_targets.split(',')
            if 'FFN_UP' in self.lora_targets:
                self.lora_UP = MOELoraLayer(args.dim, hidden_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)

            if 'FFN_GATE' in self.lora_targets:
                self.lora_GATE = MOELoraLayer(args.dim, hidden_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)

            if 'FFN_DOWN' in self.lora_targets:
                self.lora_DOWN = MOELoraLayer(hidden_dim, args.dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)

    def execute(self, x: jt.Var, type_weight: Optional[jt.Var]):
        if self.w_lora:
            type_idx = 0
            if 'FFN_UP' in self.lora_targets:
                out = self.w3(x) + self.lora_UP(x, type_weight[:,:,type_idx])
                type_idx += 1
            else:
                out = self.w3(x)

            if 'FFN_GATE' in self.lora_targets:
                out = F.silu(self.w1(x) + self.lora_GATE(x, type_weight[:,:,type_idx])) * out 
                type_idx += 1
            else:
                out = F.silu(self.w1(x)) * out

            if 'FFN_DOWN' in self.lora_targets:
                out = self.w2(out) + self.lora_DOWN(out, type_weight[:,:,type_idx])
            else:
                out = self.w2(out)
            return out
        else:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """多头注意力机制，支持分组查询注意力、LoRA适配器和Prompt Tuning"""
    
    def __init__(self, args: ModelArgs, w_lora=False, w_prompt=False):
        super().__init__()
        self.args = args

        # 注意力头配置
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # 基础线性变换
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=args.w_bias).float16()
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False).float16()
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False).float16()
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=args.w_bias).float16()
        
        # 偏置初始化
        if args.w_bias:
            nn.init.constant_(self.wq.bias, 0)
            nn.init.constant_(self.wo.bias, 0)

        # LoRA适配器设置
        self.w_lora = w_lora
        if self.w_lora:
            self.lora_targets = args.lora_targets.split(',')
            if 'Q' in self.lora_targets:
                self.lora_Q = MOELoraLayer(args.dim, args.dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
            if 'K' in self.lora_targets:
                self.lora_K = MOELoraLayer(args.dim, args.n_kv_heads * self.head_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
            if 'V' in self.lora_targets:
                self.lora_V = MOELoraLayer(args.dim, args.n_kv_heads * self.head_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
            if 'O' in self.lora_targets:
                self.lora_O = MOELoraLayer(args.dim, args.dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
        
        # Prompt Tuning设置
        self.w_prompt = w_prompt
        if self.w_prompt:
            self.prompt = nn.Embedding(args.expert_num * args.prompt_len, args.dim).float16()
            self.prompt_gate = nn.Parameter(jt.zeros((1, self.n_local_heads, 1, 1), dtype='float16'))
            if self.args.expert_num >1:
                self.prompt_router = nn.Linear(args.dim, self.args.expert_num).float16()
                
        # KV缓存设置
        self.cache_k = None
        self.cache_v = None

    def train(self, mode: bool = True):
        """训练模式切换方法"""
        if mode:
            self.cache_k = None
            self.cache_v = None
        else:
            self.cache_k = jt.zeros((self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim))
            self.cache_v = jt.zeros((self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim))
        return super().train()

    def execute(self, x: jt.Var, start_pos: int, freqs_cis: jt.Var, mask: Optional[jt.Var], type_weight: Optional[jt.Var]):
        """前向传播方法"""
        bsz, seqlen, _ = x.shape
        
        # 基础线性变换
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # LoRA适配器应用
        type_idx = 0
        if self.w_lora:
            if 'Q' in self.lora_targets:
                xq = xq + self.lora_Q(x, type_weight[:,:,type_idx])
                type_idx += 1
            if 'K' in self.lora_targets:
                xk = xk + self.lora_K(x, type_weight[:,:,type_idx])
                type_idx += 1
            if 'V' in self.lora_targets:
                xv = xv + self.lora_V(x, type_weight[:,:,type_idx])
                type_idx += 1

        # 张量重塑
        xq = xq.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # KV缓存处理
        if not self.training:
            self.cache_k = self.cache_k
            self.cache_v = self.cache_v
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            assert start_pos==0
            keys = xk
            values = xv
        
        # 注意力计算
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = jt.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        scores = nn.softmax(scores, dim=-1)
        output = jt.matmul(scores, values)

        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)

        # Prompt Tuning处理
        if self.w_prompt:
            prompt = self.prompt.weight.reshape(self.args.expert_num, self.args.prompt_len, self.args.dim)
            prompt_k = self.wk(prompt).reshape(1, self.args.expert_num * self.args.prompt_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            prompt_v = self.wv(prompt).reshape(1, self.args.expert_num * self.args.prompt_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)

            prompt_k = repeat_kv(prompt_k, self.n_rep)
            prompt_v = repeat_kv(prompt_v, self.n_rep)

            prompt_k = prompt_k.transpose(1, 2)
            prompt_v = prompt_v.transpose(1, 2)

            xq = xq.astype(prompt_k.dtype)
            prompt_scores = jt.matmul(xq, prompt_k.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            softmax_output = nn.softmax(prompt_scores, dim=-1)
            prompt_gate = self.prompt_gate.astype(softmax_output.dtype)
            prompt_scores = prompt_gate * softmax_output

            prompt_scores = prompt_scores.reshape(bsz, self.n_local_heads, -1, self.args.expert_num, self.args.prompt_len).transpose(2,3)
            prompt_v = prompt_v.reshape(bsz, self.n_local_heads, self.args.expert_num, self.args.prompt_len, self.head_dim)
            
            prompt_scores = prompt_scores.astype(prompt_v.dtype)
            experts_output = jt.matmul(prompt_scores, prompt_v)
            experts_output = experts_output.permute(0,3,2,1,4).contiguous().reshape(bsz,seqlen,self.args.expert_num, -1)
            
            if self.args.expert_num >1:
                prompt_weight = nn.softmax(self.prompt_router(x), dim=-1)
                prompt_weight = prompt_weight * type_weight[:,:,type_idx].unsqueeze(-1)
                experts_output = jt.sum(prompt_weight.unsqueeze(-1) * experts_output, dim=2, keepdims=True)
            elif self.args.expert_num == 1:
                experts_output = experts_output * type_weight[:,:,type_idx].unsqueeze(-1).unsqueeze(-1)
            type_idx += 1
            
            experts_output = experts_output.squeeze(2)
            output = output + experts_output

        # 输出投影和LoRA
        if self.w_lora and 'O' in self.lora_targets:
            return self.wo(output) + self.lora_O(output, type_weight[:,:,type_idx])
        else:
            return self.wo(output)