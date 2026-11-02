"""Fused two-layer grouped element experts (one launch per direction) for any species count.

Each program owns one atom tile of one expert (tile schedule from `schedule`), computes
pre = x W1^T + b1 (stored for the backward), h = SiLU(pre) and y = h W2^T + b2 without writing the
hidden activation. The backward recomputes SiLU' from the stored pre-activation and writes only dx.
Products are IEEE FP32 `tl.dot`s by default (TF32x3 / FP16x3 optional); the FP32 expert
parameters stay the module's parameters, the packed banks are non-persistent buffers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import triton as tr
import triton.language as tl

from hyff.kernels.fused_messages import _record

from hyff.kernels.eight_channel_messages import PRECISIONS, _dot


@torch.no_grad()
def schedule(numbers, atomic_numbers, block=64):
    """Setup-only compact routing schedule (atoms grouped by species, tiles of `block` rows per expert)."""
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
        tiles.append(tiles[-1] + (count + block - 1) // block)
    order = torch.cat(groups).to(device=numbers.device, dtype=torch.int64)
    views = tuple(order[starts[i]:starts[i + 1]] for i in range(len(species)))
    return Schedule(views, len(cpu), order,
                    torch.tensor(starts, dtype=torch.int32, device=numbers.device),
                    torch.tensor(tiles, dtype=torch.int32, device=numbers.device), int(block))


@dataclass(frozen=True)
class Schedule:
    """`ElementPartition` fields plus the tile size the `tiles` prefix was built for."""
    groups: tuple
    atoms: int
    order: torch.Tensor
    starts: torch.Tensor
    tiles: torch.Tensor
    block: int


@tr.jit
def _expert_forward(X, W1, B1, W2, B2, Order, Starts, Tiles, Pre, Out,
                    E: tl.constexpr, KIN: tl.constexpr, HID: tl.constexpr, N: tl.constexpr,
                    BM: tl.constexpr, BK: tl.constexpr, BE: tl.constexpr, PRECISION: tl.constexpr):
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
        hidden = tl.arange(0, HID)
        acc = tl.zeros((BM, HID), tl.float32)
        for block in range(tl.cdiv(KIN, BK)):
            k = block * BK + tl.arange(0, BK)
            a = tl.load(X + atom[:, None] * KIN + k[None, :], valid[:, None] & (k[None, :] < KIN), 0.)
            w = tl.load(W1 + expert * HID * KIN + hidden[None, :] * KIN + k[:, None], k[:, None] < KIN, 0.)
            acc = _dot(a, w, acc, PRECISION)
        pre = acc + tl.load(B1 + expert * HID + hidden)[None, :]
        tl.store(Pre + atom[:, None] * HID + hidden[None, :], pre, valid[:, None])
        h = pre / (1. + tl.exp(-pre))
        n = tl.arange(0, N)
        w2 = tl.load(W2 + expert * N * HID + n[None, :] * HID + hidden[:, None])
        out = _dot(h, w2, tl.zeros((BM, N), tl.float32), PRECISION) + tl.load(B2 + expert * N + n)[None, :]
        tl.store(Out + atom[:, None] * N + n[None, :], out, valid[:, None])


@tr.jit
def _expert_backward(G, W1, W2, Order, Starts, Tiles, Pre, GX,
                     E: tl.constexpr, KIN: tl.constexpr, HID: tl.constexpr, N: tl.constexpr,
                     BM: tl.constexpr, BC: tl.constexpr, BE: tl.constexpr, PRECISION: tl.constexpr):
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
        hidden = tl.arange(0, HID)
        n = tl.arange(0, N)
        g = tl.load(G + atom[:, None] * N + n[None, :], valid[:, None], 0.)
        w2 = tl.load(W2 + expert * N * HID + n[:, None] * HID + hidden[None, :])
        dh = _dot(g, w2, tl.zeros((BM, HID), tl.float32), PRECISION)
        pre = tl.load(Pre + atom[:, None] * HID + hidden[None, :], valid[:, None], 0.)
        sigmoid = 1. / (1. + tl.exp(-pre))
        dpre = dh * (sigmoid + pre * sigmoid * (1. - sigmoid))
        CHUNKS: tl.constexpr = (KIN + BC - 1) // BC
        for chunk in tl.static_range(CHUNKS):
            k = chunk * BC + tl.arange(0, BC)
            w1 = tl.load(W1 + expert * HID * KIN + hidden[:, None] * KIN + k[None, :], k[None, :] < KIN, 0.)
            dx = _dot(dpre, w1, tl.zeros((BM, BC), tl.float32), PRECISION)
            tl.store(GX + atom[:, None] * KIN + k[None, :], dx, valid[:, None] & (k[None, :] < KIN))


class _FusedExpert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, order, starts, tiles, weight1, bias1, weight2, bias2, precision, block, warps):
        values = values.contiguous()
        count, hidden, inputs = weight1.shape
        outputs = weight2.shape[1]
        atoms = len(order)
        pre = values.new_empty(atoms, hidden)
        out = values.new_empty(atoms, outputs)
        grid = (tr.cdiv(atoms, block) + count,)
        kernel = _expert_forward[grid](values, weight1, bias1, weight2, bias2, order, starts, tiles, pre, out,
                                       count, inputs, hidden, outputs, block, 16 if inputs == 16 else 32, tr.next_power_of_2(count), precision,
                                       num_warps=warps, num_stages=2)
        if not torch.compiler.is_compiling():   # the expert runs inside a torch.compile'd closure; kernel metadata is unavailable under tracing
            _record(kernel, ('fused_expert_forward', inputs, hidden, outputs, block, warps, precision))
        ctx.save_for_backward(order, starts, tiles, pre, weight1, weight2)
        ctx.settings = (precision, block, warps)
        return out

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Force training uses the plain differentiable experts')
        order, starts, tiles, pre, weight1, weight2 = ctx.saved_tensors
        precision, block, warps = ctx.settings
        count, hidden, inputs = weight1.shape
        outputs = weight2.shape[1]
        atoms = len(order)
        gx = gradient.new_empty(atoms, inputs)
        grid = (tr.cdiv(atoms, block) + count,)
        kernel = _expert_backward[grid](gradient.contiguous(), weight1, weight2, order, starts, tiles, pre, gx,
                                        count, inputs, hidden, outputs, block, 32 if inputs <= 32 else 64, tr.next_power_of_2(count), precision,
                                        num_warps=warps, num_stages=2)
        if not torch.compiler.is_compiling():
            _record(kernel, ('fused_expert_backward', inputs, hidden, outputs, block, warps, precision))
        return gx, None, None, None, None, None, None, None, None, None, None


class FusedElementExpertMLP32(nn.Module):
    """One Linear/SiLU/Linear expert per species, both layers in one launch per direction."""

    def __init__(self, reference, precision='fp32', block=64, warps=8):
        super().__init__()
        if precision not in PRECISIONS:
            raise ValueError(f'precision must be one of {PRECISIONS}')
        if block not in (16, 32, 64, 128) or warps not in (2, 4, 8):
            raise ValueError('Unsupported expert tile')
        experts = getattr(reference, 'experts', None)
        if not isinstance(experts, nn.ModuleList) or not len(experts):
            raise TypeError('Expected an element-expert MLP with a ModuleList of experts')
        for expert in experts:
            if (not isinstance(expert, nn.Sequential) or len(expert) != 3 or not isinstance(expert[0], nn.Linear)
                    or not isinstance(expert[1], nn.SiLU) or not isinstance(expert[2], nn.Linear)
                    or expert[0].bias is None or expert[2].bias is None):
                raise TypeError('Each expert must be Linear/SiLU/Linear with biases')
        if len({(tuple(e[0].weight.shape), tuple(e[2].weight.shape)) for e in experts}) != 1:
            raise ValueError('All experts must share one width')
        if any(p.requires_grad for p in reference.parameters()):
            raise ValueError('Freeze experts before installing inference execution')
        hidden, inputs = experts[0][0].weight.shape
        outputs = experts[0][2].weight.shape[0]
        if hidden & (hidden - 1) or outputs & (outputs - 1) or hidden < 16 or outputs < 16:
            raise ValueError('Fused experts need power-of-two hidden/output widths of at least 16')
        self.experts = experts
        self.out_features = int(outputs)
        self.partition = None
        self.precision, self.block, self.warps = precision, int(block), int(warps)
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
            raise ValueError('Fused experts consume CUDA FP32 atom features')
        part = self.partition
        if (part is None or part.atoms != values.shape[0] or len(part.order) != part.atoms or getattr(part, 'block', 64) != self.block
                or len(part.starts) != self.count + 1 or len(part.tiles) != self.count + 1):
            raise ValueError('Bind a complete compact element schedule with the module tile before force evaluation')
        return _FusedExpert.apply(values, part.order, part.starts, part.tiles, self.weight1, self.bias1, self.weight2, self.bias2,
                                  self.precision, self.block, self.warps)


__all__ = ["FusedElementExpertMLP32", "Schedule", "schedule"]
