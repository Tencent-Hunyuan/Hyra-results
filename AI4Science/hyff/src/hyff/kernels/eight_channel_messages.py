"""Eight-channel message kernels and species-generic grouped experts for the presto engines.

Three pieces, all inference-only (frozen weights; the coordinate VJP is the exact adjoint of the
forward):

* `Scalar8Messages`: the scalar-eight outer operator A_i = .25 sum_j switch(r_ij) phi(r_ij) m_j,
  reading the current FP32 coordinates through the position-based `_geometry` of
  `hyff.kernels.fused_messages`. The table holds exactly eight quintic filters (no angular padding).
* `FirstMessages8`: the scalar-source first block (8 s + 8 v + 8 t channels) through the
  atom-tiled kernels `hyff.kernels.message_tiles._forward_atoms` /
  `hyff.kernels.fused_messages._backward_scalar`, launched with a channel tile equal to the eight
  channels (no masked lanes) and an explicit atom/neighbor/warp layout. `fused_messages.messages`
  only selects its tuned layout for C = 16 and otherwise runs one atom per program with a 16-lane
  channel tile, which is markedly slower for C = 8.
* `FrozenElementExpertMLP32`: species-generic grouped element experts (one expert per atom, any expert
  count) with an exact IEEE FP32 product option. A variant of `hyff.kernels.grouped_experts`
  (TF32x3 / FP16x3 only); the FP32 expert parameters stay the module's parameters,
  packed copies are non-persistent buffers.
"""
from __future__ import annotations

import torch
from torch import nn
import triton as tr
import triton.language as tl

from hyff.kernels.fused_linear import _accurate_dot
from hyff.kernels.fused_messages import _backward_scalar, _geometry, _poly, _record
from hyff.kernels.message_tiles import _forward_atoms

PRECISIONS = ('fp32', 'tf32x3', 'fp16x3')


# ---------------------------------------------------------------------------------------------
# Scalar-eight outer messages with position-based geometry.

@tr.jit
def _scalar_filter(Coeff, r, channel, mask, C: tl.constexpr, INTERVALS: tl.constexpr,
                   CUTOFF: tl.constexpr, WIDTH: tl.constexpr):
    HINV: tl.constexpr = INTERVALS / CUTOFF
    coordinate = tl.minimum(tl.maximum(r * HINV, 0.), INTERVALS)
    interval = tl.minimum(coordinate.to(tl.int32), INTERVALS - 1)
    local = coordinate - interval
    y = tl.minimum(tl.maximum((CUTOFF - r) / WIDTH, 0.), 1.)
    envelope = y * y * y * (10. - 15. * y + 6. * y * y) * mask * .25
    slope = -30. * y * y * (1. - y) * (1. - y) / WIDTH * mask * .25
    value, derivative = _poly(Coeff, interval, local, channel, C, C, HINV, 0, 6)
    return value * envelope, derivative * envelope + value * slope


@tr.jit
def scalar_forward(P, Cell, Neighbors, Shifts, Counts, Message, Coeff, Output,
                   N: tl.constexpr, K: tl.constexpr, C: tl.constexpr, INTERVALS: tl.constexpr,
                   CUTOFF: tl.constexpr, WIDTH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    raw = tl.program_id(0) * BM + tl.arange(0, BM)
    node = tl.minimum(raw, N - 1)
    channel = tl.arange(0, C)[None, None, :]
    count = tl.load(Counts + node)
    total = tl.full((BM, C), 0., tl.float32)
    for start in range(tl.cdiv(tl.max(count, 0), BN)):
        slot = start * BN + tl.arange(0, BN)
        j, active, r, _, _, _ = _geometry(P, Cell, Neighbors, Shifts, node[:, None], slot[None, :], K)
        active = active & (slot[None, :] < count[:, None])
        # Count-excluded capacity may hold stale indices; sanitize before table and source reads.
        j = tl.where(active, j, 0)
        r = tl.where(active, r, 1.)
        weight, _ = _scalar_filter(Coeff, r[:, :, None], channel, active[:, :, None], C, INTERVALS, CUTOFF, WIDTH)
        source = tl.load(Message + j[:, :, None] * C + channel)
        total += tl.sum(weight * source, 1)
    tl.store(Output + raw[:, None] * C + tl.arange(0, C)[None, :], total, raw[:, None] < N)


@tr.jit
def scalar_backward(P, Cell, Neighbors, Shifts, Counts, Message, Coeff, Gradient, GM, GP,
                    N: tl.constexpr, K: tl.constexpr, C: tl.constexpr, INTERVALS: tl.constexpr,
                    CUTOFF: tl.constexpr, WIDTH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    NEED_SOURCE: tl.constexpr):
    raw = tl.program_id(0) * BM + tl.arange(0, BM)
    node = tl.minimum(raw, N - 1)
    channel = tl.arange(0, C)[None, None, :]
    own_message = tl.load(Message + node[:, None, None] * C + channel)
    own_gradient = tl.load(Gradient + node[:, None, None] * C + channel)
    count = tl.load(Counts + node)
    total_m = tl.full((BM, C), 0., tl.float32)
    total_x, total_y, total_z = total_m, total_m, total_m
    for start in range(tl.cdiv(tl.max(count, 0), BN)):
        slot = start * BN + tl.arange(0, BN)
        j, active, r, u, v, w = _geometry(P, Cell, Neighbors, Shifts, node[:, None], slot[None, :], K)
        active = active & (slot[None, :] < count[:, None])
        j = tl.where(active, j, 0)
        r = tl.where(active, r, 1.)
        u, v, w = tl.where(active, u, 0.), tl.where(active, v, 0.), tl.where(active, w, 0.)
        weight, slope = _scalar_filter(Coeff, r[:, :, None], channel, active[:, :, None], C, INTERVALS, CUTOFF, WIDTH)
        other_message = tl.load(Message + j[:, :, None] * C + channel)
        other_gradient = tl.load(Gradient + j[:, :, None] * C + channel)
        radial = slope * (other_message * own_gradient + own_message * other_gradient)
        total_x -= tl.sum(radial * u[:, :, None], 1)
        total_y -= tl.sum(radial * v[:, :, None], 1)
        total_z -= tl.sum(radial * w[:, :, None], 1)
        if NEED_SOURCE:
            total_m += tl.sum(weight * other_gradient, 1)
    if NEED_SOURCE:
        tl.store(GM + raw[:, None] * C + tl.arange(0, C)[None, :], total_m, raw[:, None] < N)
    tl.store(GP + raw * 3, tl.sum(total_x, 1), raw < N)
    tl.store(GP + raw * 3 + 1, tl.sum(total_y, 1), raw < N)
    tl.store(GP + raw * 3 + 2, tl.sum(total_z, 1), raw < N)


def _check_layout(layout):
    if len(layout) != 3 or any(type(value) is not int or value < 1 or value & (value - 1) for value in layout):
        raise ValueError('Power-of-two atom/neighbor/warp layout required')
    return tuple(layout)


class Scalar8Messages(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, cell, neighbors, shifts, counts, message, coefficients, cutoff, width, forward_layout, backward_layout):
        if not p.is_cuda or p.dtype != torch.float32 or message.shape != (len(p), 8):
            raise ValueError('True scalar8 inference requires CUDA FP32 coordinates and an [N, 8] source')
        if coefficients.ndim != 3 or coefficients.shape[1:] != (6, 8):
            raise ValueError('Exactly eight quintic filters required; no angular padding')
        if any(value.requires_grad for value in (cell, coefficients)):
            raise ValueError('Frozen coordinate/source inference only')
        message = message.contiguous()
        n, k = neighbors.shape
        output = message.new_empty(n, 8)
        bm, bn, warps = forward_layout
        kernel = scalar_forward[(tr.cdiv(n, bm),)](p, cell, neighbors, shifts, counts, message, coefficients, output,
                                                   n, k, 8, len(coefficients), cutoff, width, bm, bn, num_warps=warps)
        _record(kernel, ('scalar8_forward', k, bm, bn, warps))
        ctx.save_for_backward(p, cell, neighbors, shifts, counts, message, coefficients)
        ctx.settings = (cutoff, width, backward_layout)
        return output

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Frozen first VJP only; train using the reference model')
        p, cell, neighbors, shifts, counts, message, coefficients = ctx.saved_tensors
        cutoff, width, (bm, bn, warps) = ctx.settings
        n, k = neighbors.shape
        source_needed = ctx.needs_input_grad[5]
        gm = torch.empty_like(message) if source_needed else p
        gp = torch.empty_like(p)
        kernel = scalar_backward[(tr.cdiv(n, bm),)](p, cell, neighbors, shifts, counts, message, coefficients,
                                                    gradient.contiguous(), gm, gp, n, k, 8, len(coefficients), cutoff, width,
                                                    bm, bn, source_needed, num_warps=warps)
        _record(kernel, ('scalar8_vjp', k, bm, bn, warps, source_needed))
        return gp, None, None, None, None, gm if source_needed else None, None, None, None, None, None


# ---------------------------------------------------------------------------------------------
# Scalar-source first block (8 s + 8 v + 8 t) on the shared message kernels with an eight-lane channel tile.

class FirstMessages8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, cell, neighbors, shifts, counts, message, coefficients, cutoff, width, forward_layout, backward_layout):
        n = len(p)
        if not p.is_cuda or p.dtype != torch.float32 or message.shape != (n, 24):
            raise ValueError('Scalar-source R8 inference requires CUDA FP32 coordinates and an [N, 24] source')
        if coefficients.ndim != 3 or coefficients.shape[1:] != (6, 24):
            raise ValueError('Expected the 3 x 8 quintic filter layout of the first block')
        if any(value.requires_grad for value in (cell, coefficients)):
            raise ValueError('Frozen coordinate/source inference only')
        message = message.contiguous()
        k = neighbors.shape[1]
        out_s, out_v, out_t = p.new_empty(n, 8), p.new_empty(n, 3, 8), p.new_empty(n, 5, 8)
        bm, bn, warps = forward_layout
        kernel = _forward_atoms[(tr.cdiv(n, bm), 1)](
            p, cell, neighbors, shifts, counts, message, p, p, coefficients, out_s, out_v, out_t,
            8, k, len(coefficients), cutoff, width, True, 8, bn, n, NC=6, BM=bm, num_warps=warps, enable_fp_fusion=True)
        _record(kernel, ('first8_forward', k, bm, bn, warps))
        ctx.save_for_backward(p, cell, neighbors, shifts, counts, message, coefficients)
        ctx.settings = (cutoff, width, backward_layout)
        ctx.set_materialize_grads(False)
        return out_s, out_v, out_t

    @staticmethod
    def backward(ctx, gs, gv, gt):
        if torch.is_grad_enabled():
            raise RuntimeError('Inference-only coordinate VJP; train using the reference model')
        p, cell, neighbors, shifts, counts, message, coefficients = ctx.saved_tensors
        cutoff, width, (bm, bn, warps) = ctx.settings
        n, k = neighbors.shape
        gradients = [g.contiguous() if g is not None else p.new_zeros(shape)
                     for g, shape in zip((gs, gv, gt), ((n, 8), (n, 3, 8), (n, 5, 8)))]
        node_needed = ctx.needs_input_grad[5]
        gm = torch.empty_like(message) if node_needed else p
        gp = p.new_empty(n, 3, 1)
        kernel = _backward_scalar[(tr.cdiv(n, bm), 1)](
            p, cell, neighbors, shifts, counts, message, coefficients, *gradients, gm, gp,
            8, k, len(coefficients), cutoff, width, node_needed, 8, bn, n, NC=6, BM=bm, num_warps=warps, enable_fp_fusion=True)
        _record(kernel, ('first8_vjp', k, bm, bn, warps, node_needed))
        return gp.squeeze(-1), None, None, None, None, gm if node_needed else None, None, None, None, None, None


# ---------------------------------------------------------------------------------------------
# Species-generic grouped element experts with an exact FP32 product option.

@tr.jit
def _dot(a, b, acc, PRECISION: tl.constexpr):
    if PRECISION == 'fp32':
        return tl.dot(a, b, acc, input_precision='ieee')
    else:
        return _accurate_dot(a, b, acc, PRECISION)


@tr.jit
def _grouped_product(A, W, Bias, Order, Starts, Tiles, Pre, Out, SavedPre,
                     M: tl.constexpr, E: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TRANSPOSE: tl.constexpr,
                     HAS_BIAS: tl.constexpr, SILU: tl.constexpr, SILU_GRAD: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BE: tl.constexpr, PRECISION: tl.constexpr):
    tile = tl.program_id(0)
    if tile < tl.load(Tiles + E):
        experts = tl.arange(0, BE)
        first_tile = tl.load(Tiles + experts, experts < E, 2147483647)
        expert = tl.max(tl.where((experts < E) & (first_tile <= tile), experts, -1), 0)
        start = tl.load(Starts + expert)
        stop = tl.load(Starts + expert + 1)
        local_tile = tile - tl.load(Tiles + expert)
        row = start + local_tile * BM + tl.arange(0, BM)
        valid = row < stop
        atom = tl.load(Order + row, valid, 0)
        column = tl.program_id(1) * BN + tl.arange(0, BN)
        base = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0., tl.float32)
        for block in range(tl.cdiv(K, BK)):
            k = block * BK + base
            a = tl.load(A + atom[:, None] * K + k[None, :], valid[:, None] & (k[None, :] < K), 0.)
            if TRANSPOSE:
                address = expert * K * N + k[:, None] * N + column[None, :]
            else:
                address = expert * N * K + column[None, :] * K + k[:, None]
            w = tl.load(W + address, (k[:, None] < K) & (column[None, :] < N), 0.)
            acc = _dot(a, w, acc, PRECISION)
        if HAS_BIAS:
            acc += tl.load(Bias + expert * N + column, column < N, 0.)[None, :]
        address = atom[:, None] * N + column[None, :]
        mask = valid[:, None] & (column[None, :] < N)
        if SILU:
            tl.store(SavedPre + address, acc, mask)
            acc = acc / (1. + tl.exp(-acc))
        if SILU_GRAD:
            pre = tl.load(Pre + address, mask, 0.)
            sigmoid = 1. / (1. + tl.exp(-pre))
            acc *= sigmoid + pre * sigmoid * (1. - sigmoid)
        tl.store(Out + address, acc, mask)


def _product(values, weight, order, starts, tiles, *, inputs, outputs, count, precision,
             transpose=False, bias=None, pre=None, silu=False, silu_grad=False):
    result = values.new_empty((len(order), outputs))
    saved_pre = torch.empty_like(result) if silu else result
    bm, bn, bk = 64, 32 if outputs <= 32 else 64, 16 if inputs == 16 else 32
    grid = (tr.cdiv(len(order), bm) + count, tr.cdiv(outputs, bn))
    _grouped_product[grid](values, weight, bias if bias is not None else result,
                           order, starts, tiles, pre if pre is not None else result, result, saved_pre,
                           len(order), count, outputs, inputs, transpose, bias is not None,
                           silu, silu_grad, bm, bn, bk, tr.next_power_of_2(count), precision,
                           num_warps=4, num_stages=3)
    return result, saved_pre


class _FrozenElement(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, order, starts, tiles, weight1, bias1, weight2, bias2, precision):
        values = values.contiguous()
        count, hidden, inputs = weight1.shape
        outputs = weight2.shape[1]
        activated, pre = _product(values, weight1, order, starts, tiles, inputs=inputs, outputs=hidden,
                                  count=count, bias=bias1, silu=True, precision=precision)
        result, _ = _product(activated, weight2, order, starts, tiles, inputs=hidden, outputs=outputs,
                             count=count, bias=bias2, precision=precision)
        ctx.save_for_backward(order, starts, tiles, pre, weight1, weight2)
        ctx.precision = precision
        return result

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Force training uses the plain differentiable experts')
        order, starts, tiles, pre, weight1, weight2 = ctx.saved_tensors
        count, hidden, inputs = weight1.shape
        outputs = weight2.shape[1]
        gradient = gradient.contiguous()
        hidden_gradient, _ = _product(gradient, weight2, order, starts, tiles, inputs=outputs, outputs=hidden,
                                      count=count, transpose=True, pre=pre, silu_grad=True, precision=ctx.precision)
        dx, _ = _product(hidden_gradient, weight1, order, starts, tiles, inputs=hidden, outputs=inputs,
                         count=count, transpose=True, precision=ctx.precision)
        return dx, None, None, None, None, None, None, None, None


class FrozenElementExpertMLP32(nn.Module):
    """One affine/SiLU/affine expert per species, executed grouped; any species count."""

    def __init__(self, reference, precision='fp32'):
        super().__init__()
        if precision not in PRECISIONS:
            raise ValueError(f'precision must be one of {PRECISIONS}')
        experts = getattr(reference, 'experts', None)
        if not isinstance(experts, nn.ModuleList) or not len(experts):
            raise TypeError('Expected an element-expert MLP with a ModuleList of experts')
        for expert in experts:
            if (not isinstance(expert, nn.Sequential) or len(expert) != 3 or not isinstance(expert[0], nn.Linear)
                    or not isinstance(expert[1], nn.SiLU) or not isinstance(expert[2], nn.Linear)
                    or expert[0].bias is None or expert[2].bias is None):
                raise TypeError('Each expert must be Linear/SiLU/Linear with biases')
        shapes = {(tuple(e[0].weight.shape), tuple(e[2].weight.shape)) for e in experts}
        if len(shapes) != 1:
            raise ValueError('All experts must share one width')
        if any(p.requires_grad for p in reference.parameters()):
            raise ValueError('Freeze experts before installing inference execution')
        # Keep the module's parameter objects and names; packed copies are non-persistent caches.
        self.experts = experts
        self.out_features = int(experts[0][2].weight.shape[0])
        if getattr(reference, 'out_features', self.out_features) != self.out_features:
            raise ValueError('Reference out_features differ from the last expert layer')
        self.partition = None
        self.precision = precision
        self.count = len(experts)
        for index, position in ((1, 0), (2, 2)):
            weight = torch.stack([expert[position].weight.detach() for expert in experts])
            bias = torch.stack([expert[position].bias.detach() for expert in experts])
            if weight.dtype != torch.float32 or bias.dtype != torch.float32:
                raise ValueError('The grouped path requires FP32 learned tensors')
            self.register_buffer(f'weight{index}', weight.contiguous(), persistent=False)
            self.register_buffer(f'bias{index}', bias.contiguous(), persistent=False)

    def forward(self, values):
        if not values.is_cuda or values.dtype != torch.float32:
            raise ValueError('Grouped frozen experts consume CUDA FP32 atom features')
        part = self.partition
        if (part is None or part.atoms != values.shape[0] or len(part.order) != part.atoms
                or len(part.starts) != self.count + 1 or len(part.tiles) != self.count + 1):
            raise ValueError('Bind a complete compact element schedule before force evaluation')
        return _FrozenElement.apply(values, part.order, part.starts, part.tiles,
                                    self.weight1, self.bias1, self.weight2, self.bias2, self.precision)


__all__ = ['Scalar8Messages', 'FirstMessages8', 'FrozenElementExpertMLP32', 'PRECISIONS', 'scalar_forward', 'scalar_backward']
