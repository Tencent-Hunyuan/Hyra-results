"""Process-wide compiler settings required by the largo engine.

IEEE FP32 matrix products in PyTorch (tf32x3 products are explicit in the
kernels), Triton kernels compiled without floating-point contraction (FMA
fusion), and the dynamic-shape torch.compile policy. Applied once, before
any largo compilation; the settings persist for the process.
"""
from __future__ import annotations

import warnings


def configure_largo_compiler():
    import torch
    from triton.backends.nvidia.compiler import CUDABackend
    from hyff.engines.capture import dynamic_shapes

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # IEEE products are intentional; silence inductor's suggestion to enable TF32.
    warnings.filterwarnings('ignore', message='TensorFloat32 tensor cores', category=UserWarning)
    if not getattr(CUDABackend, '_hyff_contraction_off', False):
        original = CUDABackend.parse_options

        def parse_options(self, options):
            requested = dict(options)
            requested['enable_fp_fusion'] = False
            result = original(self, requested)
            if result.enable_fp_fusion:
                raise RuntimeError('Triton floating-point contraction could not be disabled')
            return result

        CUDABackend.parse_options = parse_options
        CUDABackend._hyff_contraction_off = True
    dynamic_shapes.install()
