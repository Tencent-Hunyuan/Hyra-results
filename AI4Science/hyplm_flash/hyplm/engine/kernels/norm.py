"""Fused residual add + RMSNorm, optionally emitting INT8 activations with per-128-column scales."""

import torch
import triton
import triton.language as tl

from .qstore import outputs, store_row


@triton.jit
def _add_rmsnorm_kernel(x_ptr, d_ptr, w_ptr, y_ptr, s_ptr, n_cols, eps, HAS_DELTA: tl.constexpr,
                        QUANT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_DELTA:
        x += tl.load(d_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(x_ptr + row * n_cols + cols, x, mask=mask)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    y = x * rstd * tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    store_row(y, y_ptr, s_ptr, row, cols, mask, n_cols, QUANT, BLOCK)


def add_rmsnorm(x, delta, weight, eps, out_dtype=torch.bfloat16, quant=False):
    """x (FP32 residual stream [T, N]) += delta in place; return RMSNorm(x) * weight.

    With quant=True the result is returned as (INT8 values, per-128-column scales), the activation format of
    int8_gemm.
    """
    rows, cols = x.shape
    y, s = outputs(rows, cols, x.device, quant, out_dtype)
    _add_rmsnorm_kernel[(rows,)](x, x if delta is None else delta, weight, y, s, cols, eps,
                                 HAS_DELTA=delta is not None, QUANT=quant, BLOCK=triton.next_power_of_2(cols),
                                 num_warps=8)
    return (y, s) if quant else y
