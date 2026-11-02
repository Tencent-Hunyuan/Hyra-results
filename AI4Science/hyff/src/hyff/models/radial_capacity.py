"""Decouple learned radial capacity from transported equivariant channels.

The added hidden units start with zero output columns, preserving the initial
function of the corresponding narrow model. This is a function-preserving
width expansion, not a new physical interaction. Frozen radial tables keep
the same dimensions when only this hidden width changes.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from hyff.models.pair_radials import PairRadialConfig, PairRadialPotential, SymmetricFiLMRadial


@dataclass(frozen=True)
class RadialCapacityConfig(PairRadialConfig):
    radial_hidden: int | None = None

    def __post_init__(self):
        super().__post_init__()
        initial = self.edge_hidden or self.hidden
        width = initial if self.radial_hidden is None else self.radial_hidden
        if type(width) is not int or width < initial:
            raise ValueError('Radial hidden width must be an integer at least the message width')
        object.__setattr__(self, 'radial_hidden', width)


def expand_network(network, width, generator):
    """Copy old channels and add trainable units without drawing global RNG."""
    first, activation, last = network
    if not isinstance(activation, nn.SiLU) or first.weight.shape[0] != last.weight.shape[1]:
        raise ValueError('Expected affine/SiLU/affine with one common hidden width')
    initial = first.weight.shape[0]
    if width == initial:
        return network
    if width < initial or first.weight.device.type != 'cpu':
        raise ValueError('Expansion is CPU initialization, before moving the model')
    extra = width - initial
    added_weight = first.weight.new_empty((extra, first.weight.shape[1]))
    # Same first-layer distribution as the existing Xavier initialization.
    bound = math.sqrt(6. / (first.weight.shape[1] + initial))
    added_weight.uniform_(-bound, bound, generator=generator)
    expanded = nn.Sequential(nn.Linear(first.weight.shape[1], width, device='meta'),
        nn.SiLU(), nn.Linear(width, last.weight.shape[0], device='meta'))
    expanded[0].weight = nn.Parameter(torch.cat((first.weight.detach(), added_weight), 0))
    expanded[0].bias = nn.Parameter(torch.cat((first.bias.detach(), first.bias.new_zeros(extra))))
    expanded[2].weight = nn.Parameter(torch.cat((last.weight.detach(),
        last.weight.new_zeros((last.weight.shape[0], extra))), 1))
    expanded[2].bias = nn.Parameter(last.bias.detach().clone())
    return expanded


def expand_radial_capacity(model):
    if not isinstance(model.config, RadialCapacityConfig):
        raise TypeError('Expected explicit RadialCapacityConfig')
    generator = torch.Generator(device='cpu')
    generator.set_state(torch.random.get_rng_state())
    for block in model.blocks:
        radial = block.radial_filter
        conditional = isinstance(radial, SymmetricFiLMRadial)
        base = radial.base if conditional else radial
        expanded = expand_network(base, model.config.radial_hidden, generator)
        if expanded is base:
            continue
        if conditional:
            widened = SymmetricFiLMRadial(expanded)
            with torch.no_grad():
                initial = radial.scale.shape[1]
                widened.scale[:, :initial].copy_(radial.scale)
                widened.shift[:, :initial].copy_(radial.shift)
            block.radial_filter = widened
        else:
            block.radial_filter = expanded


class RadialCapacityPotential(PairRadialPotential):
    def __init__(self, config):
        if not isinstance(config, RadialCapacityConfig):
            raise TypeError('Expected explicit RadialCapacityConfig')
        super().__init__(config)
        expand_radial_capacity(self)
