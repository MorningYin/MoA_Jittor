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
    """
    均方根归一化（Root Mean Square Normalization）
    
    实现RMSNorm归一化层，相比LayerNorm更高效
    只对输入进行缩放，不进行平移，计算量更少
    
    Args:
        dim: 归一化的维度大小
        eps: 数值稳定性常数，防止除零错误
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # 数值稳定性常数
        # 可学习的缩放参数，初始化为1
        self.weight = nn.Parameter(jt.ones(dim))

    def _norm(self, x):
        """
        执行RMS归一化
        
        计算均方根，然后对输入进行缩放
        公式：x / sqrt(mean(x^2) + eps)
        
        Args:
            x: 输入张量
            
        Returns:
            normalized_x: 归一化后的张量
        """
        # 计算均方根并应用缩放
        denom = jt.sqrt((x ** 2).mean(-1, keepdims=True) + self.eps)
        return x / denom

    # Jittor 前向接口依旧使用 execute 方法名，保留 forward 兼容
    def execute(self, x):
        """
        前向传播方法
        
        先进行归一化，然后应用可学习的缩放参数
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, dim]
            
        Returns:
            output: 归一化后的输出，形状与输入相同
        """
        # 转换为float32进行归一化计算，然后恢复原始类型
        output = self._norm(x)
        # 应用可学习的缩放参数
        return output * self.weight


class MOELoraLayer(nn.Module):
    """
    混合专家LoRA层（Mixture of Experts LoRA Layer）
    
    结合了LoRA（Low-Rank Adaptation）和MoE（Mixture of Experts）的适配器层
    支持单专家模式（标准LoRA）和多专家模式（MoE LoRA）
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        r: LoRA的秩（rank），控制适配器的参数量
        expert_num: 专家数量，1表示单专家，>1表示多专家
        lora_alpha: LoRA的缩放参数，默认为8
        hydra: 是否使用Hydra模式（共享A矩阵）
    """
    def __init__(self, input_dim, output_dim, r, expert_num, lora_alpha:float=8, hydra=False):
        super().__init__()

        # ==================== 基本参数设置 ====================
        self.expert_num = expert_num  # 专家数量
        self.hydra = hydra  # 是否使用Hydra模式（共享A矩阵）
        self.scaling = lora_alpha / r  # LoRA缩放因子
        self.params_num = 0  # 参数数量统计

        # ==================== 单专家模式（标准LoRA） ====================
        if expert_num == 1:
            # 创建标准的LoRA A和B矩阵
            self.lora_A = nn.Linear(input_dim, r, bias=False)  # 降维矩阵A
            self.lora_B = nn.Linear(r, output_dim, bias=False)  # 升维矩阵B
            nn.init.constant_(self.lora_B.weight, 0.0)  # 初始化B矩阵为零

        # ==================== 多专家模式（MoE LoRA） ====================
        elif expert_num > 1: # moe
            # 创建路由器，用于选择专家
            self.router = nn.Linear(input_dim, expert_num, bias=False)
            
            # ==================== A矩阵设置 ====================
            if hydra:
                # Hydra模式：所有专家共享一个A矩阵
                self.lora_A = nn.Linear(input_dim, r, bias=False)
            else:
                # 标准模式：每个专家有独立的A矩阵
                self.lora_A_l = nn.ModuleList()
                for i in range(expert_num):
                    self.lora_A_l.append(nn.Linear(input_dim, r, bias=False))
                
            # ==================== B矩阵设置 ====================
            # 每个专家都有独立的B矩阵
            self.lora_B_l = nn.ModuleList()
            for i in range(expert_num):
                self.lora_B_l.append(nn.Linear(r, output_dim, bias=False))

            # ==================== 初始化B矩阵为零 ====================
            # 确保训练开始时LoRA适配器不影响原始模型输出
            for linear in self.lora_B_l:
                nn.init.constant_(linear.weight, 0.0)
        else:
            raise Exception("The number of Experts is wrong")
    
    def params_count(self):
        """
        计算LoRA适配器的参数数量
        
        Returns:
            self.params_num: 总参数数量
        """
        self.params_num = 0
        
        # ==================== 单专家模式参数统计 ====================
        if self.expert_num == 1:
            # 统计A矩阵和B矩阵的参数数量
            self.params_num += int(self.lora_A.weight.numel())  # A矩阵参数
            self.params_num += int(self.lora_B.weight.numel())  # B矩阵参数

        # ==================== 多专家模式参数统计 ====================
        elif self.expert_num > 1: # moe
            # 统计路由器参数
            self.params_num += int(self.router.weight.numel())  # 路由器参数
            
            # ==================== A矩阵参数统计 ====================
            if self.hydra:
                # Hydra模式：共享A矩阵，只统计一个A矩阵的参数
                self.params_num += int(self.lora_A.weight.numel())
            else:
                # 标准模式：每个专家有独立的A矩阵
                for i in range(self.expert_num):
                    linear = cast(nn.Linear, self.lora_A_l[i])
                    self.params_num += int(linear.weight.numel())
                
            # ==================== B矩阵参数统计 ====================
            # 每个专家都有独立的B矩阵
            for i in range(self.expert_num):
                linear = cast(nn.Linear, self.lora_B_l[i])
                self.params_num += int(linear.weight.numel())
                
        return self.params_num


    # Jittor 前向接口
    def execute(self, x: jt.Var, type_weight: Optional[jt.Var]):
        """
        前向传播方法
        
        根据专家数量执行不同的计算逻辑：
        - 单专家：标准LoRA计算
        - 多专家：MoE LoRA计算，包含路由器选择和专家组合
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            type_weight: 类型权重，用于控制适配器的激活强度，形状为 [batch_size, seq_len]
            
        Returns:
            result: 输出张量，形状为 [batch_size, seq_len, output_dim]
        """
        # ==================== 单专家模式（标准LoRA） ====================
        if self.expert_num == 1:
            # 标准LoRA计算：A矩阵降维 -> B矩阵升维 -> 缩放
            result = self.lora_B(self.lora_A(x)) * self.scaling
            
            # 应用类型权重控制适配器强度
            result = jt.unsqueeze(type_weight, -1) * result
            return result
        
        # ==================== 多专家模式（MoE LoRA） ====================
        # type_weight: [bsz, seqlen]
        
        # ==================== 路由器计算 ====================
        # 计算每个专家的路由权重，使用softmax确保权重和为1
        route_weight = nn.softmax(self.router(x), dim=-1) # [bsz, seqlen, expert_num]
        
        # ==================== 类型权重融合 ====================
        # 将类型权重与路由权重相乘，控制整体适配器强度
        route_weight = route_weight * jt.unsqueeze(type_weight, -1)

        # ==================== 专家组合计算 ====================
        result = None
        for i in range(self.expert_num):
            if self.hydra:
                # Hydra模式：所有专家共享A矩阵
                b_layer = cast(nn.Linear, self.lora_B_l[i])
                tmp = jt.unsqueeze(route_weight[:,:,i], -1) * b_layer(self.lora_A(x)) * self.scaling
            else:
                # 标准模式：每个专家有独立的A矩阵
                a_layer = cast(nn.Linear, self.lora_A_l[i])
                b_layer = cast(nn.Linear, self.lora_B_l[i])
                tmp = jt.unsqueeze(route_weight[:,:,i], -1) * b_layer(a_layer(x)) * self.scaling  # type: ignore[arg-type]
            
            # ==================== 累加专家输出 ====================
            # 将当前专家的输出累加到总结果中
            if i == 0:
                result = tmp
            else:
                result = result + tmp
                
        return result
    

class PAdapterLayer(nn.Module):
    """
    并行适配器层（Parallel Adapter Layer）
    
    实现与主模型并行的适配器结构，支持单专家和多专家模式
    通过降维-激活-升维的结构实现高效的参数适配
    
    Args:
        hidden_size: 隐藏层维度大小
        adapter_size: 适配器中间层维度大小
        expert_num: 专家数量，1表示单专家，>1表示多专家
        hydra: 是否使用Hydra模式（共享降维矩阵）
    """
    def __init__(self, hidden_size, adapter_size, expert_num:int=1, hydra:bool=False):
        super(PAdapterLayer, self).__init__()
        
        # ==================== 基本参数设置 ====================
        self.hidden_size = hidden_size    # 隐藏层维度
        self.adapter_size = adapter_size  # 适配器维度
        self.expert_num = expert_num      # 专家数量
        self.hydra = hydra               # 是否使用Hydra模式

        # ==================== 激活函数设置 ====================
        self.adapter_act_fn = nn.SiLU()  # 使用SiLU激活函数

        # ==================== 单专家模式（标准Parallel Adapter） ====================
        if expert_num == 1:
            # 标准的降维-升维结构
            self.down_proj = nn.Linear(hidden_size, adapter_size)  # 降维投影
            self.up_proj = nn.Linear(adapter_size, hidden_size)    # 升维投影
            
        # ==================== 多专家模式（MoE Parallel Adapter） ====================
        elif expert_num > 1: # moe
            # 创建路由器，用于选择专家
            self.router = nn.Linear(hidden_size, expert_num)
            
            # ==================== 降维投影设置 ====================
            if hydra:
                # Hydra模式：所有专家共享一个降维矩阵
                self.down_proj = nn.Linear(hidden_size, adapter_size)
            else:
                # 标准模式：每个专家有独立的降维矩阵
                self.down_proj_l = nn.ModuleList()
                for i in range(expert_num):
                    self.down_proj_l.append(nn.Linear(hidden_size, adapter_size))
                
            # ==================== 升维投影设置 ====================
            # 每个专家都有独立的升维矩阵
            self.up_proj_l = nn.ModuleList()
            for i in range(expert_num):
                self.up_proj_l.append(nn.Linear(adapter_size, hidden_size))
        else:
            raise Exception("The number of Experts is wrong")
        
        # ==================== 参数初始化 ====================
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置和初始化适配器参数
        
        使用Xavier均匀分布初始化权重，偏置初始化为0
        确保适配器在训练开始时不会对主模型产生过大影响
        """
        # ==================== 单专家模式参数初始化 ====================
        if self.expert_num ==1:
            # 使用Xavier均匀分布初始化权重，gain=1e-4确保小幅度初始化
            nn.init.xavier_uniform_(self.down_proj.weight, gain=1e-4)
            nn.init.xavier_uniform_(self.up_proj.weight, gain=1e-4)
            # 偏置初始化为0
            nn.init.constant_(self.down_proj.bias, 0.0)
            nn.init.constant_(self.up_proj.bias, 0.0)
            
        # ==================== 多专家模式参数初始化 ====================
        elif self.expert_num >1:
            # ==================== 降维投影参数初始化 ====================
            if self.hydra:
                # Hydra模式：初始化共享的降维矩阵
                nn.init.xavier_uniform_(self.down_proj.weight, gain=1e-4)
                nn.init.constant_(self.down_proj.bias, 0.0)
            else:
                # 标准模式：初始化每个专家的独立降维矩阵
                for i in range(self.expert_num):
                    nn.init.xavier_uniform_(self.down_proj_l[i].weight, gain=1e-4)
                    nn.init.constant_(self.down_proj_l[i].bias, 0.0)
                    
            # ==================== 升维投影参数初始化 ====================
            # 初始化每个专家的升维矩阵
            for i in range(self.expert_num):
                nn.init.xavier_uniform_(self.up_proj_l[i].weight, gain=1e-4)
                # nn.init.zeros_(self.up_proj_l[i].weight) # zero init like lora
                nn.init.constant_(self.up_proj_l[i].bias, 0.0)

    # Jittor 前向接口
    def execute(self, x: jt.Var, type_weight: Optional[jt.Var]):
        """
        前向传播方法
        
        根据专家数量执行不同的计算逻辑：
        - 单专家：标准Parallel Adapter计算
        - 多专家：MoE Parallel Adapter计算，包含路由器选择和专家组合
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_size]
            type_weight: 类型权重，用于控制适配器的激活强度，形状为 [batch_size, seq_len]
            
        Returns:
            result: 输出张量，形状为 [batch_size, seq_len, hidden_size]
        """
        # ==================== 单专家模式（标准Parallel Adapter） ====================
        if self.expert_num == 1:
            # 标准的降维-激活-升维计算流程
            x = self.down_proj(x)           # 降维投影
            x = self.adapter_act_fn(x)      # SiLU激活
            x = self.up_proj(x)             # 升维投影
            
            # 应用类型权重控制适配器强度
            x = x * jt.unsqueeze(type_weight, -1)
            return x 

        # ==================== 多专家模式（MoE Parallel Adapter） ====================
        # type_weight: [bsz, seqlen]
        
        # ==================== 路由器计算 ====================
        # 计算每个专家的路由权重，使用softmax确保权重和为1
        route_weight = nn.softmax(self.router(x), dim=-1)  # [bsz, seqlen, expert_num]
        
        # ==================== 类型权重融合 ====================
        # 将类型权重与路由权重相乘，控制整体适配器强度
        route_weight = route_weight * jt.unsqueeze(type_weight, -1)

        # ==================== 专家组合计算 ====================
        result = None
        for i in range(self.expert_num):
            if self.hydra:
                # Hydra模式：所有专家共享降维矩阵
                tmp = jt.unsqueeze(route_weight[:,:,i], -1) * self.up_proj_l[i](self.adapter_act_fn(self.down_proj(x)))
            else:
                # 标准模式：每个专家有独立的降维矩阵
                tmp = jt.unsqueeze(route_weight[:,:,i], -1) * self.up_proj_l[i](self.adapter_act_fn(self.down_proj_l[i](x)))
            
            # ==================== 累加专家输出 ====================
            # 将当前专家的输出累加到总结果中
            if i == 0:
                result = tmp
            else:
                result = result + tmp
                
        return result


class Router(nn.Module):
    """
    SwiGLU路由器（SwiGLU Router）
    
    实现基于SwiGLU激活函数的路由器，用于MoA架构中的适配器类型选择
    采用门控机制，通过两个并行的线性变换和SiLU激活实现动态路由
    
    Args:
        in_dim: 输入维度
        hidden_dim: 隐藏层维度
        out_dim: 输出维度（适配器类型数量）
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int 
    ):
        super().__init__()

        # ==================== SwiGLU结构组件 ====================
        # 第一个线性变换，用于门控机制
        self.w1 = nn.Linear(
            in_dim, hidden_dim
        )
        # 输出线性变换
        self.w2 = nn.Linear(
            hidden_dim, out_dim
        )
        # 第二个线性变换，用于门控机制
        self.w3 = nn.Linear(
            in_dim, hidden_dim
        )
        
        # ==================== 偏置初始化 ====================
        # 将所有偏置初始化为0
        nn.init.constant_(self.w1.bias, 0)
        nn.init.constant_(self.w2.bias, 0)
        nn.init.constant_(self.w3.bias, 0)
    
    def execute(self, x):
        """
        前向传播方法
        
        实现SwiGLU门控机制：gate(x) * linear(x)
        其中gate(x) = SiLU(w1(x))，linear(x) = w3(x)
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, in_dim]
            
        Returns:
            output: 路由输出，形状为 [batch_size, seq_len, out_dim]
        """
        # SwiGLU计算：SiLU(w1(x)) * w3(x) -> w2(...)
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
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=args.w_bias
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=args.w_bias
        )
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
            # if 'FFN_UP' in self.lora_targets:
            #     out = F.silu(self.w1(x)) * (self.w3(x) + self.lora_UP(x, type_weight[:,:,type_idx]))
            #     type_idx += 1
            # else:
            #     out = F.silu(self.w1(x)) * self.w3(x)
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
    """
    多头注意力机制（Multi-Head Attention）
    
    实现支持分组查询注意力（GQA）的多头注意力机制
    集成了LoRA适配器、Prompt Tuning和KV缓存功能
    支持Flash Attention 2加速和RoPE位置编码
    
    Args:
        args: 模型配置参数
        w_lora: 是否启用LoRA适配器
        w_prompt: 是否启用Prompt Tuning
    """
    def __init__(self, args: ModelArgs, w_lora=False, w_prompt=False):
        super().__init__()
        self.args = args

        # ==================== 注意力头配置 ====================
        # 分组查询注意力（GQA）配置
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads  # KV头数
        self.n_local_heads = args.n_heads                                              # 总头数
        self.n_local_kv_heads = self.n_kv_heads                                       # 本地KV头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads                      # 重复次数
        self.head_dim = args.dim // args.n_heads                                       # 每个头的维度

        # ==================== 基础线性变换 ====================
        # Query投影矩阵
        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.w_bias
        )
        # Key投影矩阵（无偏置）
        self.wk = nn.Linear(
            args.dim,
            args.n_kv_heads * self.head_dim,
            bias=False
        )
        # Value投影矩阵（无偏置）
        self.wv = nn.Linear(
            args.dim,
            args.n_kv_heads * self.head_dim,
            bias=False
        )
        # 输出投影矩阵
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=args.w_bias
        )
        
        # ==================== 偏置初始化 ====================
        if args.w_bias:
            nn.init.constant_(self.wq.bias, 0)
            nn.init.constant_(self.wo.bias, 0)

        # ==================== LoRA适配器设置 ====================
        self.w_lora = w_lora
        if self.w_lora:
            self.lora_targets = args.lora_targets.split(',')
            # Query LoRA适配器
            if 'Q' in self.lora_targets:
                self.lora_Q = MOELoraLayer(args.dim, args.dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
            # Key LoRA适配器
            if 'K' in self.lora_targets:
                self.lora_K = MOELoraLayer(args.dim, args.n_kv_heads * self.head_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
            # Value LoRA适配器
            if 'V' in self.lora_targets:
                self.lora_V = MOELoraLayer(args.dim, args.n_kv_heads * self.head_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
            # Output LoRA适配器
            if 'O' in self.lora_targets:
                self.lora_O = MOELoraLayer(args.dim, args.dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
            
            # ==================== 专家权重设置（注释掉） ====================
            # self.expert_weight = args.expert_weight
            # if self.expert_weight:
            #     type_param_num = []
            #     if 'Q' in self.lora_targets:
            #         type_param_num.append(self.lora_Q.params_count())
            #     if 'K' in self.lora_targets:
            #         type_param_num.append(self.lora_K.params_count())
            #     if 'V' in self.lora_targets:
            #         type_param_num.append(self.lora_V.params_count())
            #     if 'O' in self.lora_targets:
            #         type_param_num.append(self.lora_O.params_count())
            #     # weight according to param number
            #     with torch.no_grad():
            #         type_weight_param = torch.FloatTensor(type_param_num)
            #         self.type_weight_param = self.lora_type * nn.functional.softmax(type_weight_param, dim=-1, dtype=torch.float32)
        
        # ==================== Prompt Tuning设置 ====================
        self.w_prompt = w_prompt
        if self.w_prompt:
            # 可学习的prompt嵌入
            self.prompt = nn.Embedding(args.expert_num * args.prompt_len, args.dim)
            # Prompt门控参数
            self.prompt_gate = nn.Parameter(jt.zeros((1, self.n_local_heads, 1, 1)))
            # 多专家模式下的prompt路由器
            if self.args.expert_num >1:
                self.prompt_router = nn.Linear(args.dim, self.args.expert_num)
                
        # ==================== KV缓存设置 ====================
        self.cache_k = None  # Key缓存
        self.cache_v = None  # Value缓存

    def train(self, mode: bool = True):
        """
        训练模式切换方法
        
        在训练和推理模式之间切换时管理KV缓存
        - 训练模式：清空缓存
        - 推理模式：初始化缓存
        
        Args:
            mode: True表示训练模式，False表示推理模式
            
        Returns:
            self: 返回自身，支持链式调用
        """
        if mode:
            # ==================== 训练模式 ====================
            # 清空KV缓存
            self.cache_k = None
            self.cache_v = None
        else:
            # ==================== 推理模式 ====================
            # 初始化KV缓存，用于自回归生成
            self.cache_k = jt.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim)
            )
            self.cache_v = jt.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim)
            )
        return super().train()

    # Jittor 中使用 execute 作为前向
    def execute(self, x: jt.Var, start_pos: int, freqs_cis: jt.Var, mask: Optional[jt.Var], type_weight: Optional[jt.Var]):
        """
        前向传播方法
        
        实现多头注意力机制的前向传播，支持LoRA适配器、Prompt Tuning和KV缓存
        包含分组查询注意力（GQA）、RoPE位置编码和Flash Attention 2优化
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, dim]
            start_pos: 当前处理序列的起始位置，用于KV缓存
            freqs_cis: 预计算的RoPE频率张量
            mask: 注意力掩码，用于因果注意力
            type_weight: 类型权重，用于控制LoRA适配器的激活强度
            
        Returns:
            output: 注意力输出，形状为 [batch_size, seq_len, dim]
        """
        # ==================== 输入处理 ====================
        bsz, seqlen, _ = x.shape
        
        # ==================== 基础线性变换 ====================
        # 计算Query、Key、Value
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # ==================== LoRA适配器应用 ====================
        type_idx = 0
        if self.w_lora:
            # 应用Query LoRA适配器
            if 'Q' in self.lora_targets:
                xq = xq + self.lora_Q(x, type_weight[:,:,type_idx])
                type_idx += 1
            # 应用Key LoRA适配器
            if 'K' in self.lora_targets:
                xk = xk + self.lora_K(x, type_weight[:,:,type_idx])
                type_idx += 1
            # 应用Value LoRA适配器
            if 'V' in self.lora_targets:
                xv = xv + self.lora_V(x, type_weight[:,:,type_idx])
                type_idx += 1

        # ==================== 张量重塑 ====================
        # 将线性变换结果重塑为多头格式
        xq = xq.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # ==================== RoPE位置编码 ====================
        # 对Query和Key应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # ==================== KV缓存处理 ====================
        if not self.training:
            # ==================== 推理模式：使用KV缓存 ====================
            # 将缓存移动到正确的设备
            self.cache_k = self.cache_k
            self.cache_v = self.cache_v

            # 更新缓存中的当前序列部分
            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            # 获取完整的Key和Value（包括缓存的历史信息）
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            # ==================== 训练模式：不使用缓存 ====================
            assert start_pos==0  # 训练时起始位置必须为0
            keys = xk
            values = xv
        
        # ==================== 注意力计算 ====================
        # ==================== 标准注意力计算 ====================
        # 重复Key/Value头以匹配Query头数（分组查询注意力）
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        # ==================== 张量转置 ====================
        # 转置为注意力计算的标准格式
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # ==================== 注意力分数计算 ====================
        # 计算Query和Key的相似度分数
        scores = jt.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # ==================== 掩码应用 ====================
        if mask is not None:
            # 应用因果掩码，确保模型只能看到当前位置及之前的信息
            scores = scores + mask

        # ==================== Softmax和注意力输出 ====================
        # 应用softmax获得注意力权重
        scores = nn.softmax(scores, dim=-1)
        # 计算注意力输出
        output = jt.matmul(scores, values)

        # ==================== 输出重塑 ====================
        # 转置并重塑为最终输出格式
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)

        # ==================== Prompt Tuning处理 ====================
        if self.w_prompt:
            # ==================== Prompt嵌入处理 ====================
            # 重塑prompt权重为专家格式
            prompt = self.prompt.weight.reshape(self.args.expert_num, self.args.prompt_len, self.args.dim)
            # 计算prompt的Key和Value
            prompt_k = self.wk(prompt).reshape(1, self.args.expert_num * self.args.prompt_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            prompt_v = self.wv(prompt).reshape(1, self.args.expert_num * self.args.prompt_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)

            # ==================== 分组查询注意力处理 ====================
            # 重复prompt的KV以匹配Query头数
            prompt_k = repeat_kv(prompt_k, self.n_rep) # [bs, expert_num * prompt_len, n_local_heads, head_dim]
            prompt_v = repeat_kv(prompt_v, self.n_rep)

            # ==================== 张量转置 ====================
            prompt_k = prompt_k.transpose(1, 2)
            prompt_v = prompt_v.transpose(1, 2) # [bs, n_local_heads, expert_num * prompt_len, head_dim]

            # ==================== Prompt注意力计算 ====================
            # 计算序列与prompt之间的注意力分数
            prompt_scores = jt.matmul(xq, prompt_k.transpose(2, 3)) / math.sqrt(self.head_dim) # [bs, n_local_heads, seqlen, expert_num * prompt_len]
            
            # ==================== Prompt门控和Softmax ====================
            # 应用prompt门控和softmax
            # self.prompt_gate 默认是 float32，而 softmax 输出根据 AMP 可能是 float16；
            # 二者直接相乘会触发 dtype 不一致的编译错误。先将 prompt_gate 转成与 xq 一致的数据类型。
            prompt_scores = self.prompt_gate * nn.softmax(prompt_scores, dim=-1)

            # ==================== 专家输出计算 ====================
            # 重塑为专家格式
            prompt_scores = prompt_scores.reshape(bsz, self.n_local_heads, -1, self.args.expert_num, self.args.prompt_len).transpose(2,3)
            prompt_v = prompt_v.reshape(bsz, self.n_local_heads, self.args.expert_num, self.args.prompt_len, self.head_dim)
            
            # 计算专家输出
            experts_output = jt.matmul(prompt_scores, prompt_v) # [bsz, local_heads, expertnum, seqlen, head_dim]
            experts_output = experts_output.permute(0,3,2,1,4).contiguous().reshape(bsz,seqlen,self.args.expert_num, -1)
            
            # ==================== 多专家路由 ====================
            if self.args.expert_num >1:
                # 多专家模式：使用路由器选择专家
                prompt_weight = nn.softmax(self.prompt_router(x), dim=-1)
                prompt_weight = prompt_weight * type_weight[:,:,type_idx].unsqueeze(-1)
                experts_output = jt.sum(prompt_weight.unsqueeze(-1) * experts_output, dim=2, keepdims=True)
            elif self.args.expert_num == 1:
                # 单专家模式：直接应用类型权重
                experts_output = experts_output * type_weight[:,:,type_idx].unsqueeze(-1).unsqueeze(-1)
            type_idx += 1
            
            # ==================== 输出融合 ====================
            # 将prompt输出与主注意力输出相加
            experts_output = experts_output.squeeze(2)
            output = output + experts_output

        # ==================== 输出投影和LoRA ====================
        if self.w_lora and 'O' in self.lora_targets:
            # 应用Output LoRA适配器
            return self.wo(output) + self.lora_O(output, type_weight[:,:,type_idx])
        else:
            # 仅使用标准输出投影
            return self.wo(output)