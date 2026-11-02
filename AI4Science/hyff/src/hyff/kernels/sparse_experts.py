"""Frozen sparse expert execution with device-side grouped matrix products.

Four assignments per atom are sorted by expert on the device. Exact group
boundaries and a compact tile schedule are recomputed on every call. All
buffers have geometry-independent sizes, no assignment is dropped, and no
host count transfer is needed during CUDA graph replay. Complexity is
O(N log N) for sorting and O(N) for fixed-width products. The router's input
derivatives flow through the explicit spline coefficients in both directions.

Only FP16-representable frozen weights (from an FP16-quantized checkpoint) are
accepted; FP32 activations/VJPs use the two-product FP16w2 arithmetic.
This experimental backend is opt-in and does not change core L2 execution.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import triton as tr
import triton.language as tl

from hyff.kernels.fused_linear import _accurate_dot
from hyff.models.experts import ElementExpertMLP, SplineExpertMLP


@tr.jit
def _route_boundaries(Indices, Order, Starts, Tiles,
                      M: tl.constexpr, E: tl.constexpr, BE: tl.constexpr,
                      BM: tl.constexpr, ITERATIONS: tl.constexpr):
    expert = tl.arange(0, BE)
    low = tl.full((BE,), 0, tl.int32)
    high = tl.full((BE,), M, tl.int32)
    for _ in range(ITERATIONS):
        middle = (low + high) // 2
        assignment = tl.load(Order + middle, middle < M, 0)
        key = tl.load(Indices + assignment, middle < M, E)
        advance = (key < expert) & (low < high)
        high = tl.where((~advance) & (low < high), middle, high)
        low = tl.where(advance, middle + 1, low)
    next_start = tl.gather(low, tl.minimum(expert + 1, BE - 1), 0)
    count = tl.where(expert < E, next_start - low, 0)
    tiles = tl.cdiv(count, BM)
    tile_start = tl.cumsum(tiles, 0) - tiles
    tl.store(Starts + expert, low, expert <= E)
    tl.store(Tiles + expert, tile_start, expert <= E)


@tr.jit
def _grouped_product(A, W, Bias, Order, Starts, Tiles, Coefficient, Pre,
                     Out, SavedPre, M: tl.constexpr, E: tl.constexpr,
                     N: tl.constexpr, K: tl.constexpr, INPUT_ATOM: tl.constexpr,
                     TRANSPOSE: tl.constexpr, HAS_BIAS: tl.constexpr,
                     SILU: tl.constexpr, SILU_GRAD: tl.constexpr,
                     WEIGHTED: tl.constexpr, BM: tl.constexpr,
                     BN: tl.constexpr, BK: tl.constexpr, BE: tl.constexpr,
                     ROUTES: tl.constexpr):
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
        assignment = tl.load(Order + row, valid, 0)
        input_row = assignment // ROUTES if INPUT_ATOM else assignment
        column = tl.program_id(1) * BN + tl.arange(0, BN)
        base = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0., tl.float32)
        for block in range(tl.cdiv(K, BK)):
            k = block * BK + base
            a = tl.load(A + input_row[:, None] * K + k[None, :],
                valid[:, None] & (k[None, :] < K), 0.)
            if TRANSPOSE:
                address = expert * K * N + k[:, None] * N + column[None, :]
            else:
                address = expert * N * K + column[None, :] * K + k[:, None]
            w = tl.load(W + address, (k[:, None] < K) & (column[None, :] < N), 0.)
            acc = _accurate_dot(a, w, acc, 'fp16w2')
        if HAS_BIAS:
            acc += tl.load(Bias + expert * N + column, column < N, 0.)[None, :]
        if WEIGHTED:
            acc *= tl.load(Coefficient + assignment, valid, 0.)[:, None]
        address = assignment[:, None] * N + column[None, :]
        mask = valid[:, None] & (column[None, :] < N)
        if SILU:
            tl.store(SavedPre + address, acc, mask)
            acc = acc / (1. + tl.exp(-acc))
        if SILU_GRAD:
            pre = tl.load(Pre + address, mask, 0.)
            sigmoid = 1. / (1. + tl.exp(-pre))
            acc *= sigmoid + pre * sigmoid * (1. - sigmoid)
        tl.store(Out + address, acc, mask)


def _layout(indices, count, block=64):
    order = torch.argsort(indices, stable=True)
    starts = torch.empty(count + 1, dtype=torch.int32, device=indices.device)
    tiles = torch.empty_like(starts)
    _route_boundaries[(1,)](indices, order, starts, tiles, len(indices), count,
        tr.next_power_of_2(count + 1), block, math.ceil(math.log2(len(indices) + 1)) + 1,
        num_warps=4)
    return order, starts, tiles


def _product(values, weight, order, starts, tiles, *, inputs, outputs,
             count, input_atom=False, transpose=False, bias=None, coefficient=None,
             pre=None, silu=False, silu_grad=False, routes=4):
    assignments = len(order)
    result = values.new_empty((assignments, outputs))
    saved_pre = torch.empty_like(result) if silu else result
    bm, bn, bk = 64, 32 if outputs <= 32 else 64, 16 if inputs == 16 else 32
    grid = (tr.cdiv(assignments, bm) + count, tr.cdiv(outputs, bn))
    _grouped_product[grid](values, weight, bias if bias is not None else result,
        order, starts, tiles, coefficient if coefficient is not None else result,
        pre if pre is not None else result, result, saved_pre, assignments,
        count, outputs, inputs, input_atom, transpose, bias is not None,
        silu, silu_grad, coefficient is not None, bm, bn, bk,
        tr.next_power_of_2(count), routes, num_warps=4, num_stages=3)
    return result, saved_pre


class _FrozenSpline(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, coefficients, weight1, bias1, weight2, bias2):
        values = values.contiguous()
        indices = indices.reshape(-1).contiguous()
        coefficients = coefficients.contiguous()
        count, hidden, inputs = weight1.shape
        outputs = weight2.shape[1]
        order, starts, tiles = _layout(indices, count)
        activated, pre = _product(values, weight1, order, starts, tiles,
            inputs=inputs, outputs=hidden, count=count, input_atom=True, bias=bias1, silu=True)
        parts, _ = _product(activated, weight2, order, starts, tiles,
            inputs=hidden, outputs=outputs, count=count, bias=bias2)
        result = (parts.reshape(-1, 4, outputs) * coefficients[..., None]).sum(1)
        ctx.save_for_backward(order, starts, tiles, coefficients, pre, parts, weight1, weight2)
        return result

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Force training needs the differentiable sparse reference')
        order, starts, tiles, coefficients, pre, parts, weight1, weight2 = ctx.saved_tensors
        count, hidden, inputs = weight1.shape
        outputs = weight2.shape[1]
        gradient = gradient.contiguous()
        hidden_gradient, _ = _product(gradient, weight2, order, starts, tiles,
            inputs=outputs, outputs=hidden, count=count, input_atom=True, transpose=True,
            coefficient=coefficients, pre=pre, silu_grad=True)
        input_parts, _ = _product(hidden_gradient, weight1, order, starts, tiles,
            inputs=hidden, outputs=inputs, count=count, transpose=True)
        dx = input_parts.reshape(-1, 4, inputs).sum(1)
        dc = (parts.reshape(-1, 4, outputs) * gradient[:, None, :]).sum(-1)
        return dx, None, dc, None, None, None, None


class _FrozenElement(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, order, starts, tiles, weight1, bias1, weight2, bias2):
        values = values.contiguous()
        count, hidden, inputs = weight1.shape
        outputs = weight2.shape[1]
        activated, pre = _product(values, weight1, order, starts, tiles,
            inputs=inputs, outputs=hidden, count=count, input_atom=True,
            bias=bias1, silu=True, routes=1)
        result, _ = _product(activated, weight2, order, starts, tiles,
            inputs=hidden, outputs=outputs, count=count, bias=bias2, routes=1)
        ctx.save_for_backward(order, starts, tiles, pre, weight1, weight2)
        return result

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Force training needs the differentiable element reference')
        order, starts, tiles, pre, weight1, weight2 = ctx.saved_tensors
        count, hidden, inputs = weight1.shape
        outputs = weight2.shape[1]
        gradient = gradient.contiguous()
        hidden_gradient, _ = _product(gradient, weight2, order, starts, tiles,
            inputs=outputs, outputs=hidden, count=count, input_atom=True,
            transpose=True, pre=pre, silu_grad=True, routes=1)
        dx, _ = _product(hidden_gradient, weight1, order, starts, tiles,
            inputs=hidden, outputs=inputs, count=count, transpose=True, routes=1)
        return dx, None, None, None, None, None, None, None


class _FrozenExpertBank(nn.Module):
    def __init__(self, reference):
        super().__init__()
        if any(p.requires_grad for p in reference.parameters()):
            raise ValueError('Sparse execution requires frozen expert and router weights')
        self.reference = reference
        self.count = len(reference.experts)
        self.out_features = reference.out_features
        for index, position in ((1, 0), (2, -1)):
            full = torch.stack([expert[position].weight.detach() for expert in reference.experts])
            half = full.half()
            if not torch.equal(full, half.float()):
                raise ValueError('Sparse FP16w2 execution requires FP16-quantized expert weights')
            self.register_buffer(f'weight{index}', half.contiguous(), persistent=False)
            self.register_buffer(f'bias{index}', torch.stack(
                [expert[position].bias.detach() for expert in reference.experts]).contiguous(), persistent=False)


class FrozenSplineExpertMLP(_FrozenExpertBank):
    def forward(self, values):
        if not values.is_cuda or values.dtype != torch.float32:
            raise ValueError('Frozen sparse execution requires CUDA FP32 atom features')
        _, indices, coefficients = self.reference.routes(values)
        return _FrozenSpline.apply(values, indices, coefficients,
            self.weight1, self.bias1, self.weight2, self.bias2)


class FrozenElementExpertMLP(_FrozenExpertBank):
    def __init__(self, reference):
        super().__init__(reference)
        self.partition = None

    def forward(self, values):
        if not values.is_cuda or values.dtype != torch.float32:
            raise ValueError('Frozen element execution requires CUDA FP32 atom features')
        if self.partition is None or self.partition.atoms != values.shape[0]:
            raise ValueError('Bind the exact element partition before frozen execution')
        return _FrozenElement.apply(values, self.partition.order, self.partition.starts,
            self.partition.tiles, self.weight1, self.bias1, self.weight2, self.bias2)


def enable_sparse_expert_nodes(model):
    """Call after ordinary fast/joined conversion and before node compilation."""
    if any(p.requires_grad for p in model.parameters()) or getattr(model, 'compiled_nodes', False):
        raise ValueError('Freeze the model and enable sparse nodes before compilation')
    if getattr(model, 'sparse_expert_nodes', False):
        raise ValueError('Sparse expert nodes are already enabled')
    if model.config.expert_routing == 'dense':
        model.sparse_expert_nodes = True
        return model
    for index in model.config.expert_blocks:
        block = model.blocks[index]
        if isinstance(block.update_mlp, SplineExpertMLP):
            block.update_mlp = FrozenSplineExpertMLP(block.update_mlp)
        elif isinstance(block.update_mlp, ElementExpertMLP):
            block.update_mlp = FrozenElementExpertMLP(block.update_mlp)
        else:
            raise TypeError('Expected an unfused sparse expert block')
    model.sparse_expert_nodes = True
    return model
