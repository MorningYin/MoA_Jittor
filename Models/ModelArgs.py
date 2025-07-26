from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    """LLaMA模型配置参数类"""
    
    # 基础模型架构参数
    dim: int = 4096
    """模型的隐藏维度大小"""
    
    n_layers: int = 32
    """模型的层数（Transformer块的数量）"""
    
    n_heads: int = 32
    """注意力头的数量"""
    
    n_kv_heads: Optional[int] = None
    """Key和Value的注意力头数（用于分组查询注意力）"""
    
    vocab_size: int = -1
    """词汇表大小，-1表示从分词器自动获取"""
    
    multiple_of: int = 256
    """SwiGLU隐藏层大小的倍数"""
    
    ffn_dim_multiplier: Optional[float] = None
    """前馈网络维度乘数"""
    
    norm_eps: float = 1e-5
    """层归一化的epsilon值"""
    
    rope_theta: float = 500000
    """RoPE（旋转位置编码）的theta参数"""
    
    use_scaled_rope: Optional[bool] = False
    """是否使用缩放版本的RoPE"""

    # 性能和精度参数
    flash_attention2: bool = False
    """是否启用Flash Attention 2"""
    
    bf16: bool = False
    """是否使用bfloat16精度"""

    # 训练和推理参数
    max_batch_size: int = 32
    """最大批次大小"""
    
    max_seq_len: int = 2048
    """最大序列长度"""

    # 参数高效微调参数
    w_bias: bool = False
    """是否微调偏置项"""
    
    # LoRA参数
    lora_layers: str = '0-0'
    """LoRA应用的层范围，格式：'0-8,24-32'"""
    
    lora_rank: int = 8
    """LoRA的秩（rank）"""
    
    lora_targets: str = 'Q,K,V,O,FFN_UP,FFN_DOWN'
    """LoRA应用的目标模块"""
    
    lora_alpha: float = 32
    """LoRA的缩放参数alpha"""
    
    hydra_moe: bool = False
    """是否启用Hydra MoE（多头专家混合）"""

    # Parallel Adapter参数
    p_adapter_layers: str = '0-0'
    """Parallel Adapter应用的层范围"""
    
    p_adapter_size: int = 16
    """Parallel Adapter的隐藏层大小"""
    
    p_adapter_hydra: bool = False
    """Parallel Adapter是否使用Hydra模式"""

    # Prompt Tuning参数
    prompt_layers: str = '0-0'
    """Prompt Tuning应用的层范围"""
    
    prompt_len: int = 10
    """Prompt的长度"""

    # 混合专家系统参数
    expert_num: int = 1
    """专家数量，1表示不使用MoE"""

    # 适配器类型路由参数
    swi_x: int = 0
    """适配器类型路由参数"""
    
    expert_weight: bool = False
    """是否根据专家参数数量设置专家权重"""

    # 稀疏路由参数
    sparse: bool = False
    """是否使用稀疏路由"""
    
    if_trainable_gamma: bool = False
    """阈值是否可训练"""
    
    gamma: float = 0.5
    """阈值"""