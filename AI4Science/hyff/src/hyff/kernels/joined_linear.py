"""Join frozen node projections without changing the learned energy function."""
import torch
from torch import nn
from torch.nn import functional as F


class JoinedLinear(nn.Module):
    def __init__(self, weights, dim, fast=False, precision='tf32x3'):
        super().__init__()
        self.register_buffer('weight', torch.cat(weights, dim=dim).detach(), persistent=False)
        self.fast = fast
        self.precision = precision
        if precision=='fp16w2':
            quantized=self.weight.to(torch.float16)
            if torch.equal(quantized.float(),self.weight):
                self.register_buffer('quantized_weight',quantized,persistent=False)
            else:
                # A product of frozen quantized matrices is generally FP32.
                # Preserve that derived projection with the three-product
                # backend rather than changing the learned energy again.
                self.precision='fp16x3'

    def forward(self, values):
        if self.fast and min(self.weight.shape) >= 16:
            from hyff.kernels.fused_linear import _Linear
            weight=self.quantized_weight if self.precision=='fp16w2' else self.weight
            return _Linear.apply(values, weight, None, True, self.precision)
        return F.linear(values, self.weight)


def enable_joined_nodes(model, fold_first=False):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Joined projections require frozen model parameters')
    if getattr(model, 'compiled_nodes', False):
        raise ValueError('Join projections before compiling atom updates')
    fast = getattr(model, 'fast_nodes', False)
    precision = getattr(model, 'fast_node_precision', 'tf32x3')
    for block_index, block in enumerate(model.blocks):
        block.joined_vector = JoinedLinear([block.vector_u.weight, block.vector_v.weight], 0, fast, precision)
        block.joined_tensor = JoinedLinear([block.tensor_u.weight, block.tensor_v.weight], 0, fast, precision)
        if block.narrow_tensors:
            block.joined_invariants = JoinedLinear(
                [block.invariant_vector_out.weight, block.invariant_tensor_out.weight], 1, fast, precision)
        if fold_first and block_index == 0 and block.input_is_scalar:
            # Initial vector/tensor states are exactly zero. Compose the two
            # constant projections in FP64 once, then store their FP32 result.
            # Later scalar-source messages keep nonzero local tensor states
            # and therefore cannot use this first-block-only factorization.
            for order in ['vector', 'tensor']:
                u, v = getattr(block, order + '_u').weight, getattr(block, order + '_v').weight
                expansion = (getattr(block, 'edge_' + order + '_out').weight if block.tensor_bottleneck
                    else torch.eye(block.tensor_hidden, device=u.device, dtype=u.dtype))
                wide = expansion.detach().double()
                projection = getattr(block, order + '_projection_in', None)
                if projection is not None:
                    wide = projection.weight.detach().double() @ wide
                weights = [expansion, (u.detach().double() @ wide).to(u.dtype),
                           (v.detach().double() @ wide).to(v.dtype)]
                setattr(block, 'folded_first_' + order, JoinedLinear(weights, 0, fast, precision))
    # The first source message depends only on atomic number and frozen
    # parameters. Its lookup is valid at every geometry; no geometry is cached.
    with torch.no_grad():
        first = model.blocks[0]
        messages = first.scalar_message(first.scalar_norm(model.embedding.weight)).detach()
    model.register_buffer('frozen_first_messages', messages, persistent=False)
    model.joined_nodes = True
    model.folded_first_projections = fold_first
    return model
