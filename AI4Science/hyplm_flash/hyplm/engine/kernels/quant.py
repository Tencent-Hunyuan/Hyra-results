"""INT8 quantization of BF16 activations with one scale per row and group of 128 columns."""

import triton
import triton.language as tl

from .qstore import outputs, store_row


@triton.jit
def _quant_kernel(x_ptr, y_ptr, s_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    store_row(x, y_ptr, s_ptr, row, cols, mask, n_cols, True, BLOCK)


def quantize_rows(x):
    """x: [T, N] contiguous; returns (INT8 [T, N], FP32 scales [T, N // 128])."""
    rows, cols = x.shape
    y, s = outputs(rows, cols, x.device, True)
    _quant_kernel[(rows,)](x, y, s, cols, BLOCK=triton.next_power_of_2(cols), num_warps=8)
    return y, s
