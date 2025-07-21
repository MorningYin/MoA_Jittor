import jittor as jt
import math
from typing import Tuple


def apply_scaling(freqs: jt.Var):
    """应用频率缩放，用于处理长序列的位置编码"""
    # 缩放参数
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192

    # 波长阈值计算
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    
    # 频率缩放处理
    new_freqs = []
    for freq in freqs.tolist():
        wavelen = 2 * math.pi / freq
        
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
            
    new_freqs_jt = jt.array(new_freqs, dtype=freqs.dtype)
    return new_freqs_jt


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    use_scaled: bool = False,
):
    """预计算旋转位置编码（RoPE）的频率"""
    # 频率计算
    freqs = 1.0 / (
        theta ** (jt.arange(0, dim, 2)[: dim // 2].float16() / dim)
    )
    
    # 位置索引
    t = jt.arange(end).float16()
    
    # 频率缩放（可选）
    if use_scaled:
        freqs = apply_scaling(freqs)
        
    # 外积计算
    freqs = jt.outer(t, freqs)
    
    # 复数转换
    cos = jt.cos(freqs)
    sin = jt.sin(freqs)
    freqs_cis = jt.stack([cos, sin], dim=-1)
    return freqs_cis


def apply_rotary_emb(
    xq: jt.Var,
    xk: jt.Var,
    freqs_cis: jt.Var,
) -> Tuple[jt.Var, jt.Var]:
    """应用旋转位置编码（RoPE）"""
    # 拆分 cos 与 sin
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
    """重复Key和Value张量，用于分组查询注意力（GQA）"""
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # 边界条件处理
    if n_rep == 1:
        return x
        
    # 重复操作
    return (
        x.unsqueeze(3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def sample_top_p(probs, p):
    # 按概率从大到小排序
    probs_sort, probs_idx = jt.sort(probs, dim=-1, descending=True)
    # 累加概率
    probs_sum = jt.cumsum(probs_sort, dim=-1)
    # 生成mask，超过p的部分置零
    mask = probs_sum > p
    probs_sort = probs_sort * mask
    # 归一化
    probs_sort = probs_sort / (probs_sort.sum(dim=-1, keepdims=True) + 1e-8)
    # 多项式采样
    next_token = jt.multinomial(probs_sort, 1)
    # 还原到原始索引
    next_token = jt.gather(probs_idx, -1, next_token)
    return next_token