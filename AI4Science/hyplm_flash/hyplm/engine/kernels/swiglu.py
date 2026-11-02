"""SwiGLU on the fused [gate | up] projection, optionally emitting INT8 activations."""

import triton
import triton.language as tl

from .qstore import outputs, store_row


@triton.jit
def _swiglu_kernel(g_ptr, y_ptr, s_ptr, n_cols, QUANT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    gate = tl.load(g_ptr + row * 2 * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(g_ptr + row * 2 * n_cols + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    store_row(gate * tl.sigmoid(gate) * up, y_ptr, s_ptr, row, cols, mask, n_cols, QUANT, BLOCK)


def swiglu(gate_up, quant=False):
    """gate_up: [T, 2F] with columns [gate | up]; returns silu(gate) * up (BF16, or INT8 + scales)."""
    rows, width = gate_up.shape
    y, s = outputs(rows, width // 2, gate_up.device, quant)
    _swiglu_kernel[(rows,)](gate_up, y, s, width // 2, QUANT=quant, BLOCK=triton.next_power_of_2(width // 2),
                            num_warps=8)
    return (y, s) if quant else y
