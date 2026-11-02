"""Scalar-refresh model: two scalar-source passes without interlayer angular state.

Each layer creates vector/STF features (R8 messages, T16 local features) from
its current geometric density. Only the scalar residual and the scalar source
pass to the next layer; both stay differentiable. Every layer keeps its energy
readout.
"""
from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn

from hyff.models.family_reference import HyFFConfig, HyFFPotential
from hyff.models.l2_potential import L2Interaction


ARCHITECTURE = 'hyff_scalar_refresh'


@dataclass(frozen=True)
class ScalarRefreshConfig(HyFFConfig):
    refresh_angular: bool = True

    def __post_init__(self):
        super().__post_init__()
        required = dict(hidden=32, edge_hidden=8, tensor_hidden=16, interactions=2,
            cutoff=4.5, switch_width=1.0, n_rbf=32, expert_hidden=64,
            expert_count=8, expert_routing='element', radial_hidden=64,
            radial_conditioning='shared', atomic_refinements=0,
            higher_invariants=True, tensor_mixing_rank=None)
        for name, value in required.items():
            if getattr(self, name) != value:
                raise ValueError(f'The scalar-refresh model requires {name}={value!r}')
        if self.layer_cutoffs != (3.5, 4.5) or self.scalar_source_blocks != (1,):
            raise ValueError('Expected the two scalar-source 3.5/4.5 Angstrom passes')
        if type(self.refresh_angular) is not bool:
            raise ValueError('refresh_angular must be an explicit boolean')


def _select_outputs(layer, rows):
    result = nn.Linear(layer.in_features, len(rows), bias=layer.bias is not None,
                       device='meta')
    index = torch.tensor(rows, dtype=torch.long, device=layer.weight.device)
    result.weight = nn.Parameter(layer.weight.detach().index_select(0, index).clone())
    if layer.bias is not None:
        result.bias = nn.Parameter(layer.bias.detach().index_select(0, index).clone())
    return result


def _finish_fresh(self, scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t):
    if hasattr(self, 'folded_first_vector'):
        # This path already derives all angular values from this layer's new
        # density. It is valid in every refreshed layer, not only the first.
        return L2Interaction._finish_messages(self, scalar, vector, tensor,
                                               aggregate_s, aggregate_v, aggregate_t)
    if self.bottleneck:
        aggregate_s = self.edge_scalar_out(aggregate_s)
    if self.tensor_bottleneck:
        aggregate_v = self.edge_vector_out(aggregate_v)
        aggregate_t = self.edge_tensor_out(aggregate_t)
    # Deliberately do not add the incoming angular states. Both feature
    # calculations and their scalar outputs keep their full force derivatives.
    return self._node_update(scalar+aggregate_s, aggregate_v, aggregate_t)


class ScalarRefreshPotential(HyFFPotential):
    def __init__(self, config):
        if not isinstance(config, ScalarRefreshConfig):
            raise TypeError('Use ScalarRefreshConfig')
        # Construct the identical parent first, preserving common RNG draws.
        super().__init__(config)
        if config.refresh_angular:
            for block in self.blocks:
                if not block.input_is_scalar:
                    raise ValueError('The scalar-refresh model needs scalar-source transport')
                if not block.terminal:
                    s, t = block.scalar_hidden, block.tensor_hidden
                    # The parent outputs are S, Vdot, Vgate, Tdot, Tgate.
                    # Keep only the three rows used by the scalar closure.
                    rows = tuple(range(s+t)) + tuple(range(s+2*t, s+3*t))
                    for expert in block.update_mlp.experts:
                        expert[-1] = _select_outputs(expert[-1], rows)
                    block.update_mlp.out_features = len(rows)
                block.terminal = True
                block.local_cross = False
                for name in ('vector_tensor_gain', 'vector_vector_gain'):
                    if name in block._parameters:
                        delattr(block, name)
                block._finish_messages = MethodType(_finish_fresh, block)

    def expert_accounting(self):
        result = super().expert_accounting()
        result.update(architecture=ARCHITECTURE, refresh_angular=self.config.refresh_angular,
            interlayer_learned_state='scalar only' if self.config.refresh_angular else 'parent scalar/vector/STF',
            order_channels=[8,8,8], local_angular_channels=16,
            scalar_residual=True, all_layer_energy_readouts=True,
            topology_free=True, additional_geometric_passes=0,
            deployment_status='no native engine; eager reference execution')
        return result


def load_checkpoint(path, device='cpu'):
    raise ValueError('Checkpoint loading is not available here; use hyff.weights.load_model')
