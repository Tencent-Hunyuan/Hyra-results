"""Row epilogue shared by the RMSNorm, SwiGLU and quantization kernels: store a row as BF16, or as INT8 with
one FP32 scale per group of 128 columns."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

GROUP = 128


@triton.jit
def store_row(y, y_ptr, s_ptr, row, cols, mask, n_cols, QUANT: tl.constexpr, BLOCK: tl.constexpr):
    if QUANT:
        G: tl.constexpr = 128
        groups = tl.reshape(y, (BLOCK // G, G))
        scales = tl.maximum(tl.max(tl.abs(groups), axis=1), 1e-12) / 127.0
        q = tl.reshape(groups / scales[:, None], (BLOCK,))
        g = tl.arange(0, BLOCK // G)
        tl.store(s_ptr + row * (n_cols // G) + g, scales, mask=g < n_cols // G)
        q = tl.minimum(tl.maximum(libdevice.rint(q), -127.0), 127.0)
        tl.store(y_ptr + row * n_cols + cols, q.to(tl.int8), mask=mask)
    else:
        tl.store(y_ptr + row * n_cols + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def outputs(rows, cols, device, quant, dtype=torch.bfloat16):
    """Allocate (values, scales) for store_row; scales alias values when unquantized."""
    if not quant:
        y = torch.empty(rows, cols, device=device, dtype=dtype)
        return y, y
    return (torch.empty(rows, cols, device=device, dtype=torch.int8),
            torch.empty(rows, cols // GROUP, device=device, dtype=torch.float32))
