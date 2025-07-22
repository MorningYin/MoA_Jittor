import math
import jittor as jt
from jittor import nn
F = nn  # type: ignore
from typing import Optional
from .ModelArgs import ModelArgs
from .utils import apply_rotary_emb, repeat_kv
from typing import cast
from jittor import init


class Module(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        ''' 将模块设置为训练模式，只对可训练参数操作梯度。 '''
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                v.is_train = True
                if k == 'attention':
                    v.clear_cache()
        self.dfs([], None, callback, None)

        # 备份存在时，只恢复可训练参数的梯度状态
        if hasattr(self, "backup_grad_state"):
            for p in self.parameters():
                pid = id(p)
                if pid in self._trainable_params and pid in self.backup_grad_state and self.backup_grad_state[pid]:
                    p.start_grad()
            # 可选：清理备份以避免重复使用
            del self.backup_grad_state
        return self

    def eval(self):
        ''' 将模块设置为评估模式，只对可训练参数操作梯度。 '''
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                v.is_train = False
                # 可选：如果有评估特定逻辑，如注意力缓存，添加这里
        self.dfs([], None, callback, None)

        # 只备份和停止可训练参数的梯度
        self.backup_grad_state = {}
        for p in self.parameters():
            pid = id(p)
            if pid in self._trainable_params:
                # 假设 not p.is_stop_grad() 表示 requires_grad=True
                self.backup_grad_state[pid] = not p.is_stop_grad()
                p.stop_grad()
        return self
    
    def init(self):
        ''' 将模块设置为评估eval模式。 '''
        def callback(parents, k, v, n):
            if isinstance(v, Module):
                v.init_weights()
        self.dfs([], None, callback, None)

    def init_weights(self):
        return
    

# Jittor 版 RMSNorm
class RMSNorm(Module):
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


class Gamma(Module):
    def __init__(self, input_dim:int, if_trainable:bool=False, gamma:float=0.5):
        super().__init__()
        self.gamma = jt.array(gamma)
        self.rate = jt.array([[[1.]]])

        if if_trainable:
            self.linear = nn.Linear(input_dim, 1, bias=False)
    
    def execute(self, x):

        if self.linear is not None:
            self.rate = jt.sigmoid(self.linear(x))
        
        return self.rate * self.gamma
        

class MOELoraLayer(Module):
    """混合专家LoRA层，结合LoRA和MoE的适配器层"""
    
    def __init__(self, input_dim, output_dim, r, expert_num, lora_alpha:float=8, hydra=False, sparse=False, if_trainable_gamma:bool=False, gamma:float=0.5):
        super().__init__()

        self.expert_num = expert_num
        self.hydra = hydra
        self.scaling = lora_alpha / r
        self.params_num = 0

        if sparse:
            if if_trainable_gamma:
                self.gamma = Gamma(input_dim, if_trainable=True, gamma=gamma)
            else:
                self.gamma = Gamma(input_dim, if_trainable=False, gamma=gamma)
        else:
            self.gamma = Gamma(input_dim, if_trainable=False, gamma=0)

        # 单专家模式（标准LoRA）
        if expert_num == 1:
            self.lora_A = nn.Linear(input_dim, r, bias=False)
            self.lora_B = nn.Linear(r, output_dim, bias=False)
            nn.init.constant_(self.lora_B.weight, 0.0)

        # 多专家模式（MoE LoRA）
        elif expert_num > 1:
            self.router = nn.Linear(input_dim, expert_num, bias=False)
            
            # A矩阵设置
            if hydra:
                self.lora_A = nn.Linear(input_dim, r, bias=False)
            else:
                self.lora_A_l = nn.ModuleList()
                for i in range(expert_num):
                    self.lora_A_l.append(nn.Linear(input_dim, r, bias=False))
                
            # B矩阵设置
            self.lora_B_l = nn.ModuleList()
            for i in range(expert_num):
                self.lora_B_l.append(nn.Linear(r, output_dim, bias=False))

            # 初始化B矩阵为零
            for linear in self.lora_B_l:
                nn.init.constant_(linear.weight, 0.0)
        else:
            raise Exception("The number of Experts is wrong")

    def init_weights(self):
        if self.expert_num == 1:
            self.lora_A.weight = init.invariant_uniform((self.lora_A.out_features, self.lora_A.in_features), "float32")
            nn.init.constant_(self.lora_B.weight, 0.0)
        elif self.expert_num > 1:
            if self.hydra:
                self.lora_A.weight = init.invariant_uniform((self.lora_A.out_features, self.lora_A.in_features), "float32")
            else:
                for i in range(self.expert_num):
                    self.lora_A_l[i].weight = init.invariant_uniform((self.lora_A_l[i].out_features, self.lora_A_l[i].in_features), "float32")
            
            for i in range(self.expert_num):
                nn.init.constant_(self.lora_B_l[i].weight, 0.0)
    
    def params_count(self):
        """计算LoRA适配器的参数数量"""
        self.params_num = 0

        if self.sparse and self.gamma is None:
            self.params_num += int(self.gamma.weight.numel())
        
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

        tokens_weight = self.gamma(x)
        mask = jt.unsqueeze(type_weight, -1) > tokens_weight

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
            
        result = result * mask
                
        if not self.is_train:
            return result, tokens_weight.clone()
        else:
            return result
    

class PAdapterLayer(Module):
    """并行适配器层，支持单专家和多专家模式"""
    
    def __init__(self, hidden_size, adapter_size, expert_num:int=1, hydra:bool=False, sparse:bool=False, if_trainable_gamma:bool=False, gamma:float=0.5):
        super(PAdapterLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.expert_num = expert_num
        self.hydra = hydra

        if sparse:
            if if_trainable_gamma:
                self.gamma = Gamma(hidden_size, if_trainable=True, gamma=gamma)
            else:
                self.gamma = Gamma(hidden_size, if_trainable=False, gamma=gamma)
        else:
            self.gamma = Gamma(hidden_size, if_trainable=False, gamma=0)

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
        
        self.init_weights()

    def init_weights(self):
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

        tokens_weight = self.gamma(x)
        mask = jt.unsqueeze(type_weight, -1) > tokens_weight

        # 单专家模式（标准Parallel Adapter）
        if self.expert_num == 1:
            x = self.down_proj(x)
            x = self.adapter_act_fn(x)
            x = self.up_proj(x)
            x = x * jt.unsqueeze(type_weight, -1)

            if not self.is_train:
                return x * mask, tokens_weight.clone()
            else:
                return x * mask

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
                
        result = result * mask
                
        if not self.is_train:
            return result, tokens_weight.clone()
        else:
            return result


class Router(Module):
    """SwiGLU路由器，用于MoA架构中的适配器类型选择"""
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int 
    ):
        super().__init__()

        # SwiGLU结构组件
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)
        self.w3 = nn.Linear(in_dim, hidden_dim)
        
        # 偏置初始化
        nn.init.constant_(self.w1.bias, 0)
        nn.init.constant_(self.w2.bias, 0)
        nn.init.constant_(self.w3.bias, 0)

    def init_weights(self):
        self.w1.weight = init.invariant_uniform((self.w1.out_features, self.w1.in_features), "float32")
        self.w2.weight = init.invariant_uniform((self.w2.out_features, self.w2.in_features), "float32")
        self.w3.weight = init.invariant_uniform((self.w3.out_features, self.w3.in_features), "float32")
        nn.init.constant_(self.w1.bias, 0)
        nn.init.constant_(self.w2.bias, 0)
        nn.init.constant_(self.w3.bias, 0)
    
    def execute(self, x):
        """前向传播方法"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class FeedForward(Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        args: ModelArgs,
        w_lora=False,
        sparse:bool=False,
        if_trainable_gamma:bool=False,
        gamma:float=0.5
    ):
        super().__init__()
        self.args = args
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=args.w_bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=args.w_bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=args.w_bias)
        if args.w_bias:
            nn.init.constant_(self.w1.bias, 0)
            nn.init.constant_(self.w2.bias, 0)
            nn.init.constant_(self.w3.bias, 0)
        self.w_lora = w_lora
        if self.w_lora:
            self.lora_targets = args.lora_targets.split(',')
            if 'FFN_UP' in self.lora_targets:
                self.lora_UP = MOELoraLayer(args.dim, hidden_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe, sparse, if_trainable_gamma, gamma)

            if 'FFN_GATE' in self.lora_targets:
                self.lora_GATE = MOELoraLayer(args.dim, hidden_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe, sparse, if_trainable_gamma, gamma)

            if 'FFN_DOWN' in self.lora_targets:
                self.lora_DOWN = MOELoraLayer(hidden_dim, args.dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe, sparse, if_trainable_gamma, gamma)

    def init_weights(self):
        if self.args.w_bias:
            nn.init.constant_(self.w1.bias, 0)
            nn.init.constant_(self.w2.bias, 0)
            nn.init.constant_(self.w3.bias, 0)

    def execute(self, x: jt.Var, type_weight: Optional[jt.Var]):
        tokens_weights = []

        if self.w_lora:
            type_idx = 0
            if 'FFN_UP' in self.lora_targets:
                if not self.is_train:
                    lora_out, tokens_weight = self.lora_UP(x, type_weight[:,:,type_idx])
                    tokens_weights.append(tokens_weight)
                    out = self.w3(x) + lora_out
                else:
                    out = self.w3(x) + self.lora_UP(x, type_weight[:,:,type_idx])
                type_idx += 1
            else:
                out = self.w3(x)

            if 'FFN_GATE' in self.lora_targets:
                if not self.is_train:
                    lora_out, tokens_weight = self.lora_GATE(x, type_weight[:,:,type_idx])
                    tokens_weights.append(tokens_weight)
                    out = F.silu(self.w1(x) + lora_out) * out
                else:
                    out = F.silu(self.w1(x) + self.lora_GATE(x, type_weight[:,:,type_idx])) * out
                type_idx += 1
            else:
                out = F.silu(self.w1(x)) * out

            if 'FFN_DOWN' in self.lora_targets:
                if not self.is_train:
                    lora_out, tokens_weight = self.lora_DOWN(out, type_weight[:,:,type_idx])
                    tokens_weights.append(tokens_weight)
                    out = self.w2(out) + lora_out
                else:
                    out = self.w2(out) + self.lora_DOWN(out, type_weight[:,:,type_idx])
            else:
                out = self.w2(out)
            
            if not self.is_train:
                return out, jt.stack(tokens_weights, dim=2)
            else:
                return out
        
        else:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(Module):
    """多头注意力机制，支持分组查询注意力、LoRA适配器和Prompt Tuning"""
    
    def __init__(self, args: ModelArgs, w_lora=False, w_prompt=False, sparse:bool=False, if_trainable_gamma:bool=False, gamma:float=0.5):
        super().__init__()
        self.args = args

        if sparse:
            if if_trainable_gamma:
                self.gamma = Gamma(args.dim, if_trainable=True, gamma=gamma)
            else:
                self.gamma = Gamma(args.dim, if_trainable=False, gamma=gamma)
        else:
            self.gamma = Gamma(args.dim, if_trainable=False, gamma=0)

        # 注意力头配置
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # 基础线性变换
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=args.w_bias)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=args.w_bias)
        
        # 偏置初始化
        if args.w_bias:
            nn.init.constant_(self.wq.bias, 0)
            nn.init.constant_(self.wo.bias, 0)

        # LoRA适配器设置
        self.w_lora = w_lora
        if self.w_lora:
            self.lora_targets = args.lora_targets.split(',')
            if 'Q' in self.lora_targets:
                self.lora_Q = MOELoraLayer(args.dim, args.dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe, sparse, if_trainable_gamma, gamma)
            if 'K' in self.lora_targets:
                self.lora_K = MOELoraLayer(args.dim, args.n_kv_heads * self.head_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe, sparse, if_trainable_gamma, gamma)
            if 'V' in self.lora_targets:
                self.lora_V = MOELoraLayer(args.dim, args.n_kv_heads * self.head_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe, sparse, if_trainable_gamma, gamma)
            if 'O' in self.lora_targets:
                self.lora_O = MOELoraLayer(args.dim, args.dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe, sparse, if_trainable_gamma, gamma)
        
        # Prompt Tuning设置
        self.w_prompt = w_prompt
        if self.w_prompt:
            self.prompt = nn.Embedding(args.expert_num * args.prompt_len, args.dim)
            self.prompt_gate = nn.Parameter(jt.zeros((1, self.n_local_heads, 1, 1)))
            if self.args.expert_num >1:
                self.prompt_router = nn.Linear(args.dim, self.args.expert_num)
                
        # KV缓存设置
        self.cache_k = None
        self.cache_v = None
    
    def init_weights(self):
        if self.w_prompt:
            self.prompt.weight = init.gauss([self.prompt.num_embeddings, self.prompt.embedding_dim], 'float32')
            self.prompt_gate = nn.Parameter(jt.zeros((1, self.n_local_heads, 1, 1)))
            if self.args.expert_num > 1:
                self.prompt_router.weight = init.invariant_uniform((self.prompt_router.out_features, self.prompt_router.in_features), "float32")

        if self.args.w_bias:
            nn.init.constant_(self.wq.bias, 0)
            nn.init.constant_(self.wo.bias, 0)


    def set_cache(self):
        self.cache_k = jt.zeros((self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim))
        self.cache_v = jt.zeros((self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim))
        # print(self.cache_k.shape, self.cache_v.shape)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None

    def execute(self, x: jt.Var, start_pos: int, freqs_cis: jt.Var, mask: Optional[jt.Var], type_weight: Optional[jt.Var]):
        """前向传播方法"""
        bsz, seqlen, _ = x.shape
        tokens_weights = []
        
        # 基础线性变换
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # LoRA适配器应用
        type_idx = 0
        if self.w_lora:
            if 'Q' in self.lora_targets:
                if not self.is_train:
                    out, tokens_weight = self.lora_Q(x, type_weight[:,:,type_idx])
                    tokens_weights.append(tokens_weight)
                    xq = xq + out
                else:
                    xq = xq + self.lora_Q(x, type_weight[:,:,type_idx])
                type_idx += 1
            if 'K' in self.lora_targets:
                if not self.is_train:
                    out, tokens_weight = self.lora_K(x, type_weight[:,:,type_idx])
                    tokens_weights.append(tokens_weight)
                    xk = xk + out
                type_idx += 1
            if 'V' in self.lora_targets:
                if not self.is_train:
                    out, tokens_weight = self.lora_V(x, type_weight[:,:,type_idx])
                    tokens_weights.append(tokens_weight)
                    xv = xv + out
                else:
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

            tokens_weight = self.gamma(x)
            mask = jt.unsqueeze(type_weight[:, :, type_idx], -1) > tokens_weight

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
            output = output + experts_output * mask
            tokens_weights.append(tokens_weight)

        # 输出投影和LoRA
        if self.w_lora and 'O' in self.lora_targets:
            if not self.is_train:
                out, tokens_weight = self.lora_O(output, type_weight[:,:,type_idx])
                tokens_weights.append(tokens_weight)
                output = self.wo(output) + out
            else:
                output = self.wo(output) + self.lora_O(output, type_weight[:,:,type_idx])
        else:
            output = self.wo(output)
        
        if not self.is_train and len(tokens_weights) > 0:
            return output, jt.stack(tokens_weights, dim=2)
        else:
            return output