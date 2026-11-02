"""C = 8 message path for full L2Interaction blocks and fp16w2-capable grouped experts.

* `L2Messages8`: the seven-path L2 messages (scalar, vector and tensor sources, edge width 8) of a
  non-first block through the atom-tiled kernels `hyff.kernels.message_tiles._forward_atoms`
  (forward) and `hyff.kernels.tensor_vjp._backward_tensor` (joint coordinate / source VJP),
  launched with an eight-lane channel tile and an explicit (atoms, neighbor slots, warps) layout.
  `hyff.kernels.fused_messages.messages` dispatches these tuned kernels only for C = 16 with the
  settings (16, 8, 4, True); for C = 8 it falls back to one atom per program with a 16-lane tile
  and half the lanes masked. Both kernels are constexpr in C and BC, so the eight-lane launch is
  an execution-layout change only (same table, same contractions, same directed-edge treatment
  as the C = 16 dispatch).
* `FrozenElementExperts`: species-generic grouped element experts (one expert per atom, any expert
  count and any width) on the `hyff.kernels.eight_channel_messages._grouped_product` kernel with
  IEEE FP32, TF32x3, FP16x3 or fp16w2 products. fp16w2 stores FP16 copies of the packed expert
  weights and requires every expert weight to be FP16-representable; it never rounds a model
  silently.
"""
from __future__ import annotations

import torch
from torch import nn
import triton as tr

from hyff.kernels.eight_channel_messages import FirstMessages8, _FrozenElement, _record
from hyff.kernels.message_tiles import _forward_atoms
from hyff.kernels.tensor_vjp import _backward_tensor

PRECISIONS = ('fp32', 'tf32x3', 'fp16x3', 'fp16w2')
C8 = 8


def check_layout(layout):
    layout = tuple(int(v) for v in layout)
    if len(layout) != 3 or any(v < 1 or v & (v - 1) for v in layout):
        raise ValueError('Power-of-two atom/neighbor/warp layout required')
    return layout


class L2Messages8(torch.autograd.Function):
    """Seven-path L2 messages at edge width 8 (non-first block) on the atom-tiled kernels."""

    @staticmethod
    def forward(ctx, p, cell, neighbors, shifts, counts, message, vector, tensor, coefficients, cutoff, width,
                forward_layout, backward_layout):
        n = len(p)
        if not p.is_cuda or p.dtype != torch.float32:
            raise ValueError('L2 C=8 inference requires CUDA FP32 coordinates')
        if message.shape != (n, 7 * C8) or vector.shape != (n, 3, C8) or tensor.shape != (n, 5, C8):
            raise ValueError('Expected an [N, 56] node message and [N, 3, 8] / [N, 5, 8] edge-width sources')
        if coefficients.ndim != 3 or coefficients.shape[1:] != (6, 7 * C8):
            raise ValueError('Expected the 7 x 8 quintic filter table of a cross-order block')
        if any(value.requires_grad for value in (cell, coefficients)):
            raise ValueError('Frozen coordinate/source inference only')
        message, vector, tensor = message.contiguous(), vector.contiguous(), tensor.contiguous()
        k = neighbors.shape[1]
        out_s, out_v, out_t = p.new_empty(n, C8), p.new_empty(n, 3, C8), p.new_empty(n, 5, C8)
        bm, bn, warps = forward_layout
        kernel = _forward_atoms[(tr.cdiv(n, bm), 1)](
            p, cell, neighbors, shifts, counts, message, vector, tensor, coefficients, out_s, out_v, out_t,
            C8, k, len(coefficients), cutoff, width, False, C8, bn, n, NC=6, BM=bm, num_warps=warps, enable_fp_fusion=True)
        _record(kernel, ('presto_c8_forward', k, bm, bn, warps))
        ctx.save_for_backward(p, cell, neighbors, shifts, counts, message, vector, tensor, coefficients)
        ctx.settings = (cutoff, width, backward_layout)
        ctx.set_materialize_grads(False)
        return out_s, out_v, out_t

    @staticmethod
    def backward(ctx, gs, gv, gt):
        if torch.is_grad_enabled():
            raise RuntimeError('Inference-only VJP; train using the reference model')
        p, cell, neighbors, shifts, counts, message, vector, tensor, coefficients = ctx.saved_tensors
        cutoff, width, (bm, bn, warps) = ctx.settings
        n, k = neighbors.shape
        gradients = [g.contiguous() if g is not None else p.new_zeros(shape)
                     for g, shape in zip((gs, gv, gt), ((n, C8), (n, 3, C8), (n, 5, C8)))]
        need_node = any(ctx.needs_input_grad[5:8])
        gm, gv_in, gt_in = ((torch.empty_like(message), torch.empty_like(vector), torch.empty_like(tensor))
                            if need_node else (p, p, p))
        gp = p.new_empty(n, 3, 1)
        kernel = _backward_tensor[(tr.cdiv(n, bm), 1)](
            p, cell, neighbors, shifts, counts, message, vector, tensor, coefficients, *gradients, gm, gv_in, gt_in, gp,
            C8, k, len(coefficients), cutoff, width, need_node, C8, bn, n, NC=6, BM=bm, JOINT=True,
            num_warps=warps, enable_fp_fusion=True)
        _record(kernel, ('presto_c8_vjp', k, bm, bn, warps, need_node))
        return (gp.squeeze(-1), None, None, None, None, gm if need_node else None, gv_in if need_node else None,
                gt_in if need_node else None, None, None, None, None, None)


def fp16_representable(tensor):
    return bool(torch.equal(tensor.detach().half().float(), tensor.detach()))


class FrozenElementExperts(nn.Module):
    """One affine/SiLU/affine expert per species, executed grouped; any species count and width.

    The expert parameters stay the module's parameters (same objects, same names); packed
    copies are non-persistent buffers (FP32, or FP16 for `fp16w2` after a representability check).
    """

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
        if len({(tuple(e[0].weight.shape), tuple(e[2].weight.shape)) for e in experts}) != 1:
            raise ValueError('All experts must share one width')
        if any(p.requires_grad for p in reference.parameters()):
            raise ValueError('Freeze experts before installing inference execution')
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
                raise ValueError('Grouped experts require FP32 learned tensors')
            if precision == 'fp16w2':
                if not fp16_representable(weight):
                    raise ValueError('fp16w2 requires FP16-representable expert weights; the export is not rounded')
                weight = weight.half()
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


__all__ = ['L2Messages8', 'FirstMessages8', 'FrozenElementExperts', 'PRECISIONS', 'check_layout', 'fp16_representable']
