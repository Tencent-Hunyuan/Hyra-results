"""Native symmetric pair energy with the checkpoint's own element stride.

`hyff.kernels.pair_energy` binds `NE=17` through kernel defaults, so its
`pair_energy` would index an 83-element pair table with the wrong stride. The Triton
kernels themselves take `NE` as a compile-time constant; this module reuses them
unchanged and passes `NE = model.n_elements`. Nothing else differs.
"""
import torch

from hyff.kernels.pair_energy import _pair_backward, _pair_forward


class _GeneralPairEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, cell, neighbors, shifts, counts, elements, coefficients, cutoff, width, fma, n_elements):
        n = len(position)
        if coefficients.shape[2] != n_elements * n_elements:
            raise ValueError('Pair table rows do not match the element count')
        result = position.new_empty(n)
        _pair_forward[(n,)](position, cell, neighbors, shifts, counts, elements, coefficients, result,
                            neighbors.shape[1], coefficients.shape[0], cutoff, width, 32,
                            NC=coefficients.shape[1], NE=n_elements, num_warps=4, enable_fp_fusion=fma)
        ctx.save_for_backward(position, cell, neighbors, shifts, counts, elements, coefficients)
        ctx.settings = cutoff, width, fma, n_elements
        return result

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Fused pairs are inference-only; use the plain energy for force training')
        position, cell, neighbors, shifts, counts, elements, coefficients = ctx.saved_tensors
        cutoff, width, fma, n_elements = ctx.settings
        gp = torch.empty_like(position)
        _pair_backward[(len(position),)](position, cell, neighbors, shifts, counts, elements, coefficients,
                                         gradient.contiguous(), gp, neighbors.shape[1], coefficients.shape[0],
                                         cutoff, width, 32, NC=coefficients.shape[1], NE=n_elements,
                                         num_warps=4, enable_fp_fusion=fma)
        return gp, None, None, None, None, None, None, None, None, None, None


def pair_energy(position, plan, elements, table):
    """Drop-in replacement for hyff.kernels.pair_energy.pair_energy with NE = n_elements."""
    return _GeneralPairEnergy.apply(position, plan.cell, plan.neighbors, plan.shifts, plan.counts, elements,
                                    table.coefficients, plan.model.cutoff, plan.model.config.switch_width,
                                    plan.model.fused_message_settings[3], int(plan.model.n_elements))
