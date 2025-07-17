from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    """
    LLaMA模型配置参数类
    
    该类定义了LLaMA模型的所有配置参数，包括：
    1. 基础模型架构参数（维度、层数、头数等）
    2. 训练和推理参数（批次大小、序列长度等）
    3. 参数高效微调参数（LoRA、Adapter、Prompt Tuning等）
    4. 混合专家系统参数（MoE）
    5. 优化和性能参数（Flash Attention、精度等）
    
    该配置类支持多种参数高效微调方法，实现MoA（Mixture of Adapters）的核心功能。
    """
    
    # ==================== 基础模型架构参数 ====================
    dim: int = 4096
    """
    模型的隐藏维度大小
    决定了模型的表示能力和参数量
    """
    
    n_layers: int = 32
    """
    模型的层数（Transformer块的数量）
    影响模型的深度和表达能力
    """
    
    n_heads: int = 32
    """
    注意力头的数量
    多头注意力机制中的头数，影响模型的并行处理能力
    """
    
    n_kv_heads: Optional[int] = None
    """
    Key和Value的注意力头数（用于分组查询注意力）
    如果为None，则使用n_heads的值
    用于减少内存使用和计算量
    """
    
    vocab_size: int = -1
    """
    词汇表大小
    -1表示从分词器自动获取
    """
    
    multiple_of: int = 256
    """
    SwiGLU隐藏层大小的倍数
    确保隐藏层大小是256的倍数，优化计算效率
    """
    
    ffn_dim_multiplier: Optional[float] = None
    """
    前馈网络维度乘数
    用于调整前馈网络的隐藏层大小
    """
    
    norm_eps: float = 1e-5
    """
    层归一化的epsilon值
    防止除零错误，提高数值稳定性
    """
    
    rope_theta: float = 500000
    """
    RoPE（旋转位置编码）的theta参数
    控制位置编码的旋转频率
    """
    
    use_scaled_rope: Optional[bool] = False
    """
    是否使用缩放版本的RoPE
    用于处理长序列的位置编码
    """

    # ==================== 性能和精度参数 ====================
    flash_attention2: bool = False
    """
    是否启用Flash Attention 2
    加速注意力计算，减少内存使用
    """
    
    bf16: bool = False
    """
    是否使用bfloat16精度
    在支持的GPU上使用混合精度训练
    """

    # ==================== 训练和推理参数 ====================
    max_batch_size: int = 32
    """
    最大批次大小
    推理时的批次大小限制
    """
    
    max_seq_len: int = 2048
    """
    最大序列长度
    输入文本的最大token数量
    """

    # ==================== 参数高效微调参数 ====================
    w_bias: bool = False
    """
    是否微调偏置项
    控制是否更新模型中的偏置参数
    """
    
    # ==================== LoRA参数 ====================
    lora_layers: str = '0-0'
    """
    LoRA应用的层范围
    格式：'0-8,24-32' 表示第0-8层和第24-32层应用LoRA
    """
    
    lora_rank: int = 8
    """
    LoRA的秩（rank）
    控制低秩分解的维度，影响参数量和表达能力
    """
    
    lora_targets: str = 'Q,K,V,O,FFN_UP,FFN_DOWN'
    """
    LoRA应用的目标模块
    可选：Q(Query), K(Key), V(Value), O(Output), FFN_UP, FFN_DOWN
    """
    
    lora_alpha: float = 32
    """
    LoRA的缩放参数alpha
    控制LoRA输出的缩放程度，通常设置为rank的倍数
    """
    
    hydra_moe: bool = False
    """
    是否启用Hydra MoE（多头专家混合）
    Hydra LoRA是一种非对称LoRA变体
    """

    # ==================== Parallel Adapter参数 ====================
    p_adapter_layers: str = '0-0'
    """
    Parallel Adapter应用的层范围
    格式：'0-8,24-32' 表示第0-8层和第24-32层应用Parallel Adapter
    """
    
    p_adapter_size: int = 16
    """
    Parallel Adapter的隐藏层大小
    控制adapter的参数量和表达能力
    """
    
    p_adapter_hydra: bool = False
    """
    Parallel Adapter是否使用Hydra模式
    启用非对称的adapter结构
    """

    # ==================== Prompt Tuning参数 ====================
    prompt_layers: str = '0-0'
    """
    Prompt Tuning应用的层范围
    格式：'0-8,24-32' 表示第0-8层和第24-32层应用Prompt Tuning
    """
    
    prompt_len: int = 10
    """
    Prompt的长度
    每个层添加的可学习prompt token数量
    """

    # ==================== 混合专家系统参数 ====================
    expert_num: int = 1
    """
    专家数量
    MoE（Mixture of Experts）中的专家个数
    1表示不使用MoE，使用单一专家
    """

    # ==================== 适配器类型路由参数 ====================
    swi_x: int = 0
    """
    适配器类型路由参数
    0表示使用普通Linear层
    非零值表示使用swi_x * adapter_type作为SwiGLU路由器的隐藏层大小
    """
    
    expert_weight: bool = False
    """
    是否根据专家参数数量设置专家权重
    在MoE中根据每个专家的参数量调整权重
    """