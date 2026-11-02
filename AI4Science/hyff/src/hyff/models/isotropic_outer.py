"""Isotropic-outer model: the inner block keeps its angular closure, the outer block uses only 0->0 transport.

The scalar-refresh parent is constructed first. Selecting the kept rows
and columns consumes no new RNG draws. No fixed bonds or angular edge padding
enter the outer reference; all coordinate and mixed derivatives use autograd.
"""
from dataclasses import dataclass

import torch
from torch import nn

from hyff.models.scalar_refresh import ScalarRefreshConfig, ScalarRefreshPotential

ARCHITECTURE = 'hyff_isotropic_outer'


@dataclass(frozen=True)
class IsotropicOuterConfig(ScalarRefreshConfig):
    isotropic_outer: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.refresh_angular is not True or type(self.isotropic_outer) is not bool:
            raise ValueError('Require scalar-refresh inner block and explicit outer switch')


def select_linear(layer, outputs=None, inputs=None):
    outputs = layer.weight.shape[0] if outputs is None else outputs
    inputs = layer.weight.shape[1] if inputs is None else inputs
    selected = nn.Linear(inputs, outputs, bias=layer.bias is not None, device='meta')
    selected.weight = nn.Parameter(layer.weight.detach()[:outputs, :inputs].clone())
    if layer.bias is not None:
        selected.bias = nn.Parameter(layer.bias.detach()[:outputs].clone())
    return selected


class IsotropicOuterInteraction(nn.Module):
    def __init__(self, parent):
        super().__init__()
        if not parent.input_is_scalar or not parent.terminal or parent.message_hidden != 8:
            raise ValueError('Expected the terminal R8 scalar-source refresh block')
        for name in ('scalar_norm', 'update_norm', 'scalar_message', 'radial_filter',
                     'edge_scalar_out', 'update_mlp'):
            self.add_module(name, getattr(parent, name))
        self.scalar_message[-1] = select_linear(self.scalar_message[-1], outputs=8)
        self.radial_filter[-1] = select_linear(self.radial_filter[-1], outputs=8)
        for expert in self.update_mlp.experts:
            expert[0] = select_linear(expert[0], inputs=32)
            expert[-1] = select_linear(expert[-1], outputs=32)
        self.update_mlp.out_features = 32
        for name in ('cutoff', 'switch_width', 'message_hidden', 'scalar_hidden',
                     'tensor_hidden', 'message_scale'):
            setattr(self, name, getattr(parent, name))
        self.input_is_scalar = self.terminal = self.bottleneck = True
        self.cross_order = self.local_cross = self.higher_invariants = False
        self.tensor_bottleneck = False
        self.narrow_tensors = True
        self.tensor_mixing_rank = None
        self.isotropic_outer = True
        self.order_channels = (8, 0, 0)

    def _finish_scalar(self, scalar, aggregate_s):
        state = scalar + self.edge_scalar_out(aggregate_s)
        return state + self.update_mlp(self.update_norm(state))

    def _finish_messages(self, scalar, vector, tensor, aggregate_s, *unused):
        return self._finish_scalar(scalar, aggregate_s), vector, tensor

    def forward(self, scalar, vector, tensor, radial, direction, quadrupole,
                source, destination, edge_envelope, radial_output=None, csr=None,
                reference=None, node_message=None):
        if csr is not None or reference is not None:
            raise ValueError('The isotropic-outer block runs eagerly here; CSR execution is hyff.engines.native.fast_csr')
        if node_message is None:
            node_message = self.scalar_message(self.scalar_norm(scalar))
        radial_output = self.radial_filter(radial) if radial_output is None else radial_output
        messages = node_message[source] * radial_output * (self.message_scale * edge_envelope[:, None])
        aggregate_s = scalar.new_zeros((scalar.shape[0], 8)).index_add(0, destination, messages)
        # V/T are untouched API placeholders and do not enter any calculation.
        return self._finish_messages(scalar, vector, tensor, aggregate_s)


class IsotropicOuterPotential(ScalarRefreshPotential):
    def __init__(self, config):
        if not isinstance(config, IsotropicOuterConfig):
            raise TypeError('Use explicit IsotropicOuterConfig')
        super().__init__(config)
        if config.isotropic_outer:
            self.blocks[1] = IsotropicOuterInteraction(self.blocks[1])

    def forward(self, batch, **kwargs):
        if kwargs.get('csr') is not None or self.fused_messages or self.reference_chunk_size:
            raise ValueError('Use the isotropic-outer native engine (hyff.engines.native.fast_csr) for CSR execution')
        return super().forward(batch, **kwargs)

    def expert_accounting(self):
        result = super().expert_accounting()
        result.update(architecture=ARCHITECTURE, isotropic_outer=self.config.isotropic_outer,
            order_channels_per_block=[[8,8,8],[8,0,0]] if self.config.isotropic_outer else [[8,8,8],[8,8,8]],
            requires_covalent_topology=False, topology_free=True, scalar_residual=True,
            all_layer_energy_readouts=True, outer_angular_features=False if self.config.isotropic_outer else True,
            deployment_status='native engine: hyff.engines.native.fast_csr')
        return result


def load_checkpoint(path, device='cpu'):
    raise ValueError('Checkpoint loading is not available here; use hyff.weights.load_model')
