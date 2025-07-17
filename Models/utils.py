import jittor as jt
import math
from typing import Tuple
import functools


def apply_scaling(freqs: jt.Var):
    """
    应用频率缩放，用于处理长序列的位置编码
    
    通过网格搜索获得的最优参数，用于扩展RoPE的位置编码范围
    对不同频率范围采用不同的缩放策略，确保长序列的位置编码质量
    
    Args:
        freqs: 原始频率张量
        
    Returns:
        new_freqs: 缩放后的频率张量
    """
    # ==================== 缩放参数（通过网格搜索获得） ====================
    scale_factor = 8        # 缩放因子
    low_freq_factor = 1     # 低频因子
    high_freq_factor = 4    # 高频因子
    old_context_len = 8192  # 原始LLaMA3的上下文长度

    # ==================== 波长阈值计算 ====================
    low_freq_wavelen = old_context_len / low_freq_factor   # 低频波长阈值
    high_freq_wavelen = old_context_len / high_freq_factor # 高频波长阈值
    
    # ==================== 频率缩放处理 ====================
    new_freqs = []
    for freq in freqs.tolist():
        wavelen = 2 * math.pi / freq  # 计算当前频率对应的波长，保持与 freq 类型一致
        
        if wavelen < high_freq_wavelen:
            # 高频：保持原频率
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            # 低频：直接缩放
            new_freqs.append(freq / scale_factor)
        else:
            # 中频：平滑插值
            assert low_freq_wavelen != high_freq_wavelen
            # 计算平滑因子
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            # 线性插值
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
            
    new_freqs_jt = jt.array(new_freqs, dtype=freqs.dtype)
    return new_freqs_jt


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    use_scaled: bool = False,
):
    """
    预计算旋转位置编码（RoPE）的频率
    
    生成用于RoPE的复数频率张量，支持缩放版本用于长序列处理
    
    Args:
        dim: 嵌入维度
        end: 序列长度
        theta: 频率基数，默认为10000.0
        use_scaled: 是否使用缩放版本，用于长序列
        
    Returns:
        freqs_cis: 复数频率张量，形状为 [end, dim//2]
    """
    # ==================== 频率计算 ====================
    # 计算不同维度的频率：1 / (theta^(2i/dim))
    freqs = 1.0 / (
        theta ** (jt.arange(0, dim, 2)[: dim // 2].float32() / dim)
    )  # [dim//2]
    
    # ==================== 位置索引 ====================
    # 生成位置索引张量
    t = jt.arange(end).float32()
    
    # ==================== 频率缩放（可选） ====================
    if use_scaled:
        # 应用频率缩放，用于处理长序列
        freqs = apply_scaling(freqs)
        
    # ==================== 外积计算 ====================
    # 计算位置和频率的外积：t ⊗ freqs
    freqs = jt.outer(t, freqs)  # [end, dim//2]
    
    # ==================== 复数转换 ====================
    # 返回 cos、sin 实数对，形状 [end, dim//2, 2]
    cos = jt.cos(freqs)
    sin = jt.sin(freqs)
    freqs_cis = jt.stack([cos, sin], dim=-1)
    return freqs_cis


def apply_rotary_emb(
    xq: jt.Var,
    xk: jt.Var,
    freqs_cis: jt.Var,
) -> Tuple[jt.Var, jt.Var]:
    """
    应用旋转位置编码（RoPE）
    
    对Query和Key张量应用旋转位置编码，实现相对位置编码
    通过复数乘法实现旋转操作，保持注意力机制的相对位置信息
    
    Args:
        xq: Query张量，形状为 [batch_size, seq_len, n_heads, head_dim]
        xk: Key张量，形状为 [batch_size, seq_len, n_heads, head_dim]
        freqs_cis: 预计算的复数频率张量
        
    Returns:
        xq_out: 应用RoPE后的Query张量
        xk_out: 应用RoPE后的Key张量
    """
    # ==================== 使用实数实现 RoPE ====================
    # 假设 freqs_cis 由 cosθ, sinθ 组成，形状为 [seq_len, head_dim//2, 2]

    # 拆分 cos 与 sin，并调整为可广播形状 [1, seq_len, 1, head_dim//2]
    cos = freqs_cis[..., 0].unsqueeze(0).unsqueeze(2)
    sin = freqs_cis[..., 1].unsqueeze(0).unsqueeze(2)

    # 拆分偶数 / 奇数维度
    xq_even = xq[..., 0::2]
    xq_odd  = xq[..., 1::2]
    xk_even = xk[..., 0::2]
    xk_odd  = xk[..., 1::2]

    # 旋转
    xq_rot_even = xq_even * cos - xq_odd * sin
    xq_rot_odd  = xq_even * sin + xq_odd * cos
    xk_rot_even = xk_even * cos - xk_odd * sin
    xk_rot_odd  = xk_even * sin + xk_odd * cos

    # 重新交错拼接
    xq_out = jt.stack([xq_rot_even, xq_rot_odd], dim=-1).reshape(xq.shape)
    xk_out = jt.stack([xk_rot_even, xk_rot_odd], dim=-1).reshape(xk.shape)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def repeat_kv(x: jt.Var, n_rep: int) -> jt.Var:
    """
    重复Key和Value张量，用于分组查询注意力（GQA）
    
    当Query头数大于Key/Value头数时，需要重复KV张量以匹配维度
    实现分组查询注意力的关键操作
    
    Args:
        x: 输入张量，形状为 [batch_size, seq_len, n_kv_heads, head_dim]
        n_rep: 重复次数，通常为 n_heads // n_kv_heads
        
    Returns:
        repeated_x: 重复后的张量，形状为 [batch_size, seq_len, n_heads, head_dim]
    """
    # 获取张量的形状信息
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # ==================== 边界条件处理 ====================
    if n_rep == 1:
        # 如果不需要重复，直接返回原张量
        return x
        
    # ==================== 重复操作 ====================
    # 通过扩展和重塑实现重复：
    # 1. 添加新维度：[:, :, :, None, :]
    # 2. 扩展到目标形状：.expand(bs, slen, n_kv_heads, n_rep, head_dim)
    # 3. 重塑为最终形状：.reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    return (
        x.unsqueeze(3)  # [:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def inference_mode_jt(fn):
    """PyTorch inference_mode 的最小 Jittor 版"""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        was_training = getattr(self, "training", False)   # 记录原模式
        self.eval()                                       # 进入 eval
        with jt.no_grad():                                # 关闭梯度
            out = fn(self, *args, **kwargs)
        if was_training:
            self.train()                                  # 恢复训练模式
        return out
    return wrapper