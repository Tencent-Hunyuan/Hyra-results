"""Bounded-memory verification of frozen conditioned messages with PyTorch.

Each backward chunk recomputes ordinary differentiable tensor operations.
The implementation shares no hand-derived coordinate adjoints with the GPU
backend. All edges and atom features are kept. This is inference-only.
"""
import torch

from hyff.models.l2_potential import (
    _quadrupole, _tensor_dot_direction, _vector_tensor_product, smooth_switch,
)


def edge_values(displacement, message, vector, tensor, valid, low, high,
                table, bank, rbf, cutoff, width, first):
    distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1e-9)
    direction = displacement / distance[:, None]
    quadrupole = _quadrupole(direction)
    if table is None:
        hidden = bank.base[0](rbf(distance))
        gain = 1. + bank.scale[low] + bank.scale[high]
        shift = bank.shift[low] + bank.shift[high]
        radial = bank.base[2](bank.base[1](hidden * gain + shift))
    else:
        coordinate = (distance * table.inverse_spacing).clamp(0., float(table.intervals))
        interval = coordinate.floor().long().clamp(max=table.intervals - 1)
        x = (coordinate - interval.to(coordinate.dtype))[:, None]
        pair = low * (15 - low) // 2 + high
        radial = table.coefficients[pair, interval, 5]
        for power in range(4, -1, -1):
            radial = radial * x + table.coefficients[pair, interval, power]
    gate = message * radial * (.25 * smooth_switch(distance, cutoff, width) * valid)[:, None]
    if first:
        scalar, vr, tr = gate.chunk(3, -1)
        return scalar, vr[:, None, :] * direction[:, :, None], tr[:, None, :] * quadrupole[:, :, None]
    scalar, vr, v, tv, tr, t, vt = gate.chunk(7, -1)
    tensor_vector, tensor_projection = _tensor_dot_direction(tensor, direction)
    vector_tensor, vector_projection = _vector_tensor_product(vector, direction)
    return (
        scalar + vt * vector_projection + tv * tensor_projection,
        v[:, None, :] * vector + vr[:, None, :] * direction[:, :, None] + tv[:, None, :] * tensor_vector,
        t[:, None, :] * tensor + tr[:, None, :] * quadrupole[:, :, None] + vt[:, None, :] * vector_tensor,
    )


class _ChunkedPairReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, displacement, source, destination, message, vector, tensor,
                valid, low, high, table, bank, rbf, cutoff, width, first, chunk):
        outputs = [torch.zeros_like(message[:, :vector.shape[-1]]),
                   torch.zeros_like(vector), torch.zeros_like(tensor)]
        for start in range(0, len(source), chunk):
            ids = slice(start, start + chunk)
            j = source[ids]
            values = edge_values(displacement[ids], message[j], vector[j], tensor[j],
                valid[ids], low[ids], high[ids], table, bank, rbf, cutoff, width, first)
            for output, value in zip(outputs, values):
                output.index_add_(0, destination[ids], value)
        ctx.save_for_backward(displacement, source, destination, message, vector, tensor, valid, low, high)
        ctx.settings = (table, bank, rbf, cutoff, width, first, chunk)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, gs, gv, gt):
        if torch.is_grad_enabled():
            raise RuntimeError('Chunked pair reference is for frozen verification, not force training')
        displacement, source, destination, message, vector, tensor, valid, low, high = ctx.saved_tensors
        table, bank, rbf, cutoff, width, first, chunk = ctx.settings
        gradients = [torch.zeros_like(message), torch.zeros_like(vector), torch.zeros_like(tensor)]
        gradient_displacement = torch.empty_like(displacement)
        for start in range(0, len(source), chunk):
            ids = slice(start, start + chunk)
            j, i = source[ids], destination[ids]
            with torch.enable_grad():
                inputs = [x.detach().requires_grad_(True)
                          for x in (displacement[ids], message[j], vector[j], tensor[j])]
                outputs = edge_values(*inputs, valid[ids], low[ids], high[ids],
                    table, bank, rbf, cutoff, width, first)
                loss = sum((out * gradient[i]).sum()
                           for out, gradient in zip(outputs, (gs, gv, gt)) if gradient is not None)
                derivatives = torch.autograd.grad(loss, inputs, allow_unused=True)
            gradient_displacement[ids] = derivatives[0]
            for target, value in zip(gradients, derivatives[1:]):
                if value is not None:
                    target.index_add_(0, j, value)
        return (gradient_displacement, None, None, *gradients, None, None, None,
                None, None, None, None, None, None, None)


def reference_messages(displacement, source, destination, message, vector, tensor,
                       valid, table, bank, rbf, cutoff, width, first, chunk):
    if chunk < 1 or bank.context is None:
        raise ValueError('Positive chunk size and current pair context required')
    return _ChunkedPairReference.apply(displacement, source, destination, message, vector, tensor,
        valid, bank.context.low, bank.context.high, table, bank, rbf, cutoff, width, first, chunk)
