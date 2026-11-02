"""Frozen grouped element experts on the model's FP32 weight tensors.

Based on the element-only path of hyff.kernels.sparse_experts. Each atom
executes one expert. This implementation accepts generic species count/width
and uses TF32x3 or a three-product residual FP16 decomposition of BOTH FP32
operands. No checkpoint weight is rounded or replaced.
"""
import torch
from torch import nn
import triton as tr
import triton.language as tl

from hyff.kernels.fused_linear import _accurate_dot
from hyff.models.experts import ElementPartition


@torch.no_grad()
def bind_partition(numbers, atomic_numbers):
    """Setup-only exact routing schedule; no topology or coordinates involved."""
    if numbers.ndim != 1 or numbers.dtype != torch.int64:
        raise ValueError('Atomic numbers must be an int64 vector')
    species = tuple(int(z) for z in atomic_numbers)
    if len(set(species)) != len(species) or not species:
        raise ValueError('Distinct nonempty supported species required')
    cpu = numbers.detach().cpu()
    groups = [torch.nonzero(cpu == z, as_tuple=False).flatten() for z in species]
    counts = [len(group) for group in groups]
    if sum(counts) != len(cpu):
        raise ValueError('An atom has no declared element expert')
    starts, tiles = [0], [0]
    for count in counts:
        starts.append(starts[-1] + count)
        tiles.append(tiles[-1] + (count + 63) // 64)
    order = torch.cat(groups).to(device=numbers.device, dtype=torch.int64)
    views = tuple(order[starts[i]:starts[i+1]] for i in range(len(species)))
    return ElementPartition(views, len(cpu), order,
        torch.tensor(starts, dtype=torch.int32, device=numbers.device),
        torch.tensor(tiles, dtype=torch.int32, device=numbers.device))


@tr.jit
def _grouped_product(A, W, Bias, Order, Starts, Tiles, Pre, Out, SavedPre,
                     M: tl.constexpr, E: tl.constexpr, N: tl.constexpr,
                     K: tl.constexpr, TRANSPOSE: tl.constexpr,
                     HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
                     SILU_GRAD: tl.constexpr, BM: tl.constexpr,
                     BN: tl.constexpr, BK: tl.constexpr, BE: tl.constexpr,
                     PRECISION: tl.constexpr):
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
            a = tl.load(A + atom[:, None] * K + k[None, :],
                valid[:, None] & (k[None, :] < K), 0.)
            if TRANSPOSE:
                address = expert * K * N + k[:, None] * N + column[None, :]
            else:
                address = expert * N * K + column[None, :] * K + k[:, None]
            w = tl.load(W + address, (k[:, None] < K) & (column[None, :] < N), 0.)
            acc = _accurate_dot(a, w, acc, PRECISION)
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


def _product(values, weight, order, starts, tiles, *, inputs, outputs,
             count, precision, transpose=False, bias=None, pre=None,
             silu=False, silu_grad=False):
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
        activated, pre = _product(values, weight1, order, starts, tiles,
            inputs=inputs, outputs=hidden, count=count, bias=bias1,
            silu=True, precision=precision)
        result, _ = _product(activated, weight2, order, starts, tiles,
            inputs=hidden, outputs=outputs, count=count, bias=bias2, precision=precision)
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
        hidden_gradient, _ = _product(gradient, weight2, order, starts, tiles,
            inputs=outputs, outputs=hidden, count=count, transpose=True,
            pre=pre, silu_grad=True, precision=ctx.precision)
        dx, _ = _product(hidden_gradient, weight1, order, starts, tiles,
            inputs=hidden, outputs=inputs, count=count, transpose=True, precision=ctx.precision)
        return dx, None, None, None, None, None, None, None, None


class FrozenElementExpertMLP(nn.Module):
    def __init__(self, reference, precision='tf32x3'):
        super().__init__()
        if precision not in ('tf32x3', 'fp16x3'):
            raise ValueError('This same-weight backend accepts only TF32x3 or FP16x3')
        if any(p.requires_grad for p in reference.parameters()):
            raise ValueError('Freeze experts before installing inference execution')
        # Keep the module's parameter names and objects. Packed weights are
        # nonpersistent caches, never a substitute checkpoint representation.
        self.experts = reference.experts
        self.out_features = reference.out_features
        self.partition = None
        self.precision = precision
        self.count = len(self.experts)
        self._parameter_versions = tuple(p._version for p in self.parameters())
        for index, position in ((1, 0), (2, -1)):
            weight = torch.stack([expert[position].weight.detach() for expert in self.experts])
            bias = torch.stack([expert[position].bias.detach() for expert in self.experts])
            if weight.dtype != torch.float32 or bias.dtype != torch.float32:
                raise ValueError('The grouped path requires FP32 learned tensors')
            self.register_buffer(f'weight{index}', weight.contiguous(), persistent=False)
            self.register_buffer(f'bias{index}', bias.contiguous(), persistent=False)

    def forward(self, values):
        if not values.is_cuda or values.dtype != torch.float32:
            raise ValueError('Grouped frozen experts consume CUDA FP32 atom features')
        part = self.partition
        if (part is None or part.atoms != values.shape[0] or len(part.order) != part.atoms
                or len(part.starts) != self.count+1 or len(part.tiles) != self.count+1):
            raise ValueError('Bind a complete compact element schedule before force evaluation')
        return _FrozenElement.apply(values, part.order, part.starts, part.tiles,
            self.weight1, self.bias1, self.weight2, self.bias2, self.precision)


def enable_grouped_experts(model, precision='tf32x3'):
    """Replace only the element experts, after linear/fusion conversion."""
    if any(p.requires_grad for p in model.parameters()) or getattr(model, 'compiled_nodes', False):
        raise ValueError('Freeze parameters and install grouped experts before compilation')
    from hyff.models.element_model import ElementExpertMLP
    for index in model.config.expert_blocks:
        block = model.blocks[index]
        if not isinstance(block.update_mlp, ElementExpertMLP):
            raise TypeError('Expected the generic element-expert MLP')
        block.update_mlp = FrozenElementExpertMLP(block.update_mlp, precision)
    model.sparse_expert_nodes = True
    model.grouped_expert_precision = precision
    return model
