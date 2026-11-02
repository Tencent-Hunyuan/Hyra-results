"""In-place per-head QK RMSNorm + rotary embedding on the packed [T, 3, H, D] QKV buffer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope_kernel(qkv_ptr, pos_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, eps,
                         H: tl.constexpr, D: tl.constexpr, HB: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    pos = tl.load(pos_ptr + token).to(tl.int64)
    heads = tl.arange(0, HB)[:, None]
    half = tl.arange(0, D // 2)[None, :]
    mask = heads < H
    cos = tl.load(cos_ptr + pos * (D // 2) + half)
    sin = tl.load(sin_ptr + pos * (D // 2) + half)
    for which in tl.static_range(2):
        base = qkv_ptr + token * (3 * H * D) + which * (H * D) + heads * D + half
        a = tl.load(base, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(base + D // 2, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt((tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)) / D + eps)[:, None]
        if which == 0:
            wa = tl.load(qw_ptr + half)
            wb = tl.load(qw_ptr + D // 2 + half)
        else:
            wa = tl.load(kw_ptr + half)
            wb = tl.load(kw_ptr + D // 2 + half)
        a = a * rstd * wa
        b = b * rstd * wb
        tl.store(base, (a * cos - b * sin).to(tl.bfloat16), mask=mask)
        tl.store(base + D // 2, (b * cos + a * sin).to(tl.bfloat16), mask=mask)


def rope_tables(dim, theta, max_positions, device):
    """FP32 cos/sin [max_positions, dim // 2], computed exactly like the reference Rotary module."""
    inv = theta ** (-torch.arange(0, dim, 2, device=device).float() / dim)
    freq = torch.arange(max_positions, device=device).float()[:, None] * inv
    return freq.cos().contiguous(), freq.sin().contiguous()


def qk_norm_rope_(qkv, positions, q_weight, k_weight, cos, sin, eps):
    """qkv: [T, 3, H, D] BF16, normalized and rotated in place (q and k only)."""
    tokens, _, heads, dim = qkv.shape
    _qk_norm_rope_kernel[(tokens,)](qkv, positions, q_weight, k_weight, cos, sin, eps,
                                    H=heads, D=dim, HB=triton.next_power_of_2(heads), num_warps=4)
    return qkv
