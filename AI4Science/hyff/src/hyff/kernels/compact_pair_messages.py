"""R8 execution layout with all eight channel lanes active (edge width 8 models).

The standard pair dispatch uses BC16 for both R8 and R16. This layout runs the
same kernels with BC8/BM16 for forward/full adjoints and BC8/BN16 for frozen
scalar adjoints. Parameterization and radial coefficients are unchanged.
"""
from types import MethodType

import torch
import triton as tr

from hyff.kernels.fused_messages import _record
from hyff.models.l2_potential import L2Interaction

from hyff.kernels.pair_messages import _forward_pair, _backward_pair_scalar, _backward_pair_tensor
from hyff.engines.native.pair_radial_execution import FrozenPairPotential


class _CompactPairMessages(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, cell, neighbors, shifts, counts, elements,
                message, vector, tensor, coefficients, cutoff, width, first, settings):
        if position.dtype != torch.float32 or not position.is_cuda:
            raise ValueError('Symmetric fused inference requires CUDA FP32')
        n, c = vector.shape[0], vector.shape[-1]
        banks = coefficients.ndim == 4
        intervals = coefficients.shape[1] if banks else coefficients.shape[0]
        if (c != 8 or tuple(settings) != (16, 8, 4, True)
                or (banks and coefficients.shape[0] != 36)
                or coefficients.shape[-2] != 6
                or coefficients.shape[-1] != (3 if first else 7) * c):
            raise ValueError('Compact dispatch accepts only R8 quintic tables')
        if any(x.requires_grad for x in (cell, coefficients)):
            raise ValueError('Cell/table training is unsupported by the frozen force backend')
        message, vector, tensor = message.contiguous(), vector.contiguous(), tensor.contiguous()
        out_s = position.new_empty((n, c))
        out_v, out_t = torch.empty_like(vector), torch.empty_like(tensor)
        _, _, warps, fma = settings
        bc = 8
        kernel = _forward_pair[(tr.cdiv(n, 16), tr.cdiv(c, bc))](
            position, cell, neighbors, shifts, counts, message, vector, tensor, coefficients,
            elements, out_s, out_v, out_t, c, neighbors.shape[1], intervals, cutoff, width,
            first, bc, 1, n, NC=6, BM=16, BANKS=banks, num_warps=warps, enable_fp_fusion=fma)
        _record(kernel, f'compact_pair_forward_C{c}_K{neighbors.shape[1]}_first{first}_banks{banks}_atoms16')
        ctx.save_for_backward(position, cell, neighbors, shifts, counts, elements,
                              message, vector, tensor, coefficients)
        ctx.settings = (cutoff, width, first, settings, banks, intervals)
        return out_s, out_v, out_t

    @staticmethod
    def backward(ctx, gs, gv, gt):
        if torch.is_grad_enabled():
            raise RuntimeError('Pair fused messages are inference-only; use the reference for force training')
        position, cell, neighbors, shifts, counts, elements, message, vector, tensor, coefficients = ctx.saved_tensors
        cutoff, width, first, settings, banks, intervals = ctx.settings
        _, _, warps, fma = settings
        bc, bn = 8, 16
        n, c = vector.shape[0], vector.shape[-1]
        gp = position.new_empty((n, 3, tr.cdiv(c, bc)))
        gs = gs.contiguous() if gs is not None else position.new_zeros((n, c))
        gv = gv.contiguous() if gv is not None else torch.zeros_like(vector)
        gt = gt.contiguous() if gt is not None else torch.zeros_like(tensor)
        if first:
            need_message = ctx.needs_input_grad[6]
            gm = torch.empty_like(message) if need_message else position
            atom_tile = 8 if need_message else 1
            kernel = _backward_pair_scalar[(tr.cdiv(n, atom_tile), tr.cdiv(c, bc))](
                position, cell, neighbors, shifts, counts, message, coefficients, elements,
                gs, gv, gt, gm, gp, c, neighbors.shape[1], intervals, cutoff, width,
                need_message, bc, bn // atom_tile, n, NC=6, BM=atom_tile, BANKS=banks,
                num_warps=warps, enable_fp_fusion=fma)
            _record(kernel, f'compact_pair_scalar_vjp_C{c}_K{neighbors.shape[1]}_nodes{need_message}_banks{banks}_atoms{atom_tile}')
            return (gp.sum(-1), None, None, None, None, None,
                    gm if need_message else None,
                    torch.zeros_like(vector) if ctx.needs_input_grad[7] else None,
                    torch.zeros_like(tensor) if ctx.needs_input_grad[8] else None,
                    None, None, None, None, None)
        need_node = any(ctx.needs_input_grad[6:9])
        gm, gvi, gti = ((torch.empty_like(message), torch.empty_like(vector), torch.empty_like(tensor))
                        if need_node else (position, position, position))
        kernel = _backward_pair_tensor[(tr.cdiv(n, 16), tr.cdiv(c, bc))](
            position, cell, neighbors, shifts, counts, message, vector, tensor, coefficients,
            elements, gs, gv, gt, gm, gvi, gti, gp, c, neighbors.shape[1], intervals,
            cutoff, width, need_node, bc, 1, n, NC=6, BM=16, JOINT=True, BANKS=banks,
            num_warps=warps, enable_fp_fusion=fma)
        _record(kernel, f'compact_pair_tensor_vjp_C{c}_K{neighbors.shape[1]}_nodes{need_node}_banks{banks}_atoms16')
        return (gp.sum(-1), None, None, None, None, None, gm if need_node else None,
                gvi if need_node else None, gti if need_node else None,
                None, None, None, None, None)


def _compact_message_forward(self, scalar, vector, tensor, radial, direction, quadrupole,
                     source, destination, edge_envelope, radial_output=None,
                     csr=None, reference=None, node_message=None):
    if csr is None and (reference is None or not hasattr(self.radial_filter, 'scale')):
        return L2Interaction.forward(self, scalar, vector, tensor, radial, direction,
            quadrupole, source, destination, edge_envelope, radial_output, csr, reference, node_message)
    n_atoms, hidden = scalar.shape[0], self.message_hidden
    if self.tensor_bottleneck:
        if self.input_is_scalar:
            message_vector = vector.new_zeros((n_atoms, 3, hidden))
            message_tensor = tensor.new_zeros((n_atoms, 5, hidden))
        else:
            message_vector = self.edge_vector_in(vector)
            message_tensor = self.edge_tensor_in(tensor)
    else:
        message_vector, message_tensor = vector, tensor
    if node_message is None:
        node_message = self.scalar_message(self.scalar_norm(scalar))
    if csr is not None:
        positions, plan, table = csr
        values = messages(positions, plan.cell, plan.neighbors, plan.shifts, plan.counts,
            plan.elements, node_message, message_vector, message_tensor, table,
            self.cutoff, self.switch_width, self.input_is_scalar, plan.model.fused_message_settings)
    else:
        from hyff.kernels.pair_messages_reference import reference_messages
        displacement, valid, table, rbf, cutoff, width, chunk = reference
        values = reference_messages(displacement, source, destination, node_message,
            message_vector, message_tensor, valid, table, self.radial_filter, rbf,
            cutoff, width, self.input_is_scalar, chunk)
    return self._finish_messages(scalar, vector, tensor, *values)


def messages(position, cell, neighbors, shifts, counts, elements, message, vector, tensor,
             table, cutoff, width, first, settings=(16, 8, 4, True)):
    return _CompactPairMessages.apply(position, cell, neighbors, shifts, counts, elements,
        message, vector, tensor, table.coefficients, cutoff, width, first, settings)


class CompactPairPotential(FrozenPairPotential):
    def __init__(self, config):
        if config.edge_hidden != 8:
            raise ValueError('Compact pair dispatch accepts only R8')
        super().__init__(config)
        for block in self.blocks:
            block.forward = MethodType(_compact_message_forward, block)
