"""Capture-safe first-order autograd Function for the latent-charge pair energy.

forward(positions, charges, coupling, sigma, backend) -> coupling * E (shape [1]), where the
backend returns unscaled (phi, g) in FP32 (see hyff.kernels.long_range.dense). backward returns exact partials of
the evaluated energy:  dE/dx_i = c q_i g_i,  dE/dq_i = c phi_i,  dE/dc = E_unscaled.
sigma is a constant here (MD-only contract: no sigma gradient). No host synchronisation.
Saved state is O(N): charges, phi, g.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable


class LatentChargePair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, charges, coupling, sigma, backend):
        q = charges.detach().to(torch.float32)
        phi, g = backend.evaluate(positions.detach(), q, sigma.detach())
        e = 0.5 * torch.dot(q.double(), phi.double())
        ctx.save_for_backward(q, phi, g, coupling, e)
        ctx.position_dtype = positions.dtype
        return (coupling * e.to(coupling.dtype)).reshape(1)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        q, phi, g, coupling, e = ctx.saved_tensors
        gs = (grad.reshape(()) * coupling).to(torch.float32)
        dx = (gs * q[:, None] * g).to(ctx.position_dtype) if ctx.needs_input_grad[0] else None
        dq = gs * phi if ctx.needs_input_grad[1] else None
        dc = (grad.reshape(()) * e).to(coupling.dtype) if ctx.needs_input_grad[2] else None
        return dx, dq, dc, None, None


class DenseBackend:
    """Fused exact dense Triton evaluation; stateless, plan-free."""
    name = "dense_fused_triton_fp32"

    def __init__(self, block_i=64, block_j=64, num_warps=4):
        self.block_i, self.block_j, self.num_warps = block_i, block_j, num_warps

    def evaluate(self, positions, charges, sigma):
        from hyff.kernels.long_range.dense import dense_triton
        return dense_triton(positions, charges, sigma, block_i=self.block_i,
                            block_j=self.block_j, num_warps=self.num_warps)

    def needs_rebuild(self, positions):
        return False

    def describe(self):
        return dict(backend=self.name, block_i=self.block_i, block_j=self.block_j,
                    num_warps=self.num_warps)
