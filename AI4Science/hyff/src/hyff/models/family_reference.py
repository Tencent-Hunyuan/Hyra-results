"""Topology-free HyFF models with one or more geometric aggregations.

The single-aggregation member uses the same scalar/vector/STF algebra and
conservative energy as the other members. It has no chemical graph encoder.
All atom-only refinements are the equivariant polynomial updates of L2Potential;
they change neither the geometric cutoff nor the neighbor graph.
"""
from dataclasses import asdict, dataclass

import torch

from hyff.models.radial_capacity import RadialCapacityConfig, RadialCapacityPotential


ARCHITECTURE = 'hyff_family'


@dataclass(frozen=True)
class HyFFConfig(RadialCapacityConfig):
    def __post_init__(self):
        if type(self.interactions) is not int or self.interactions < 1:
            raise ValueError('HyFF needs at least one geometric aggregation')
        if self.interactions != 1:
            super().__post_init__()
            return
        if self.atomic_refinements or self.scalar_source_blocks:
            raise ValueError('A terminal single aggregation has no inter-layer refinements')
        if self.expert_blocks is not None and tuple(self.expert_blocks) != (0,):
            raise ValueError('The one-aggregation model has only expert block zero')
        radii = (float(self.cutoff),) if self.layer_cutoffs is None else tuple(self.layer_cutoffs)
        if radii != (float(self.cutoff),):
            raise ValueError('The single aggregation uses the declared physical cutoff')
        # Validate all shared widths/routing/radials through the multi-layer
        # configuration class, without changing that class.
        values = asdict(self)
        values.update(interactions=2, layer_cutoffs=(radii[0], radii[0]), expert_blocks=(0,))
        validated = RadialCapacityConfig(**values)
        for name, value in asdict(validated).items():
            if name not in ('interactions', 'layer_cutoffs'):
                object.__setattr__(self, name, value)
        object.__setattr__(self, 'layer_cutoffs', radii)


class HyFFPotential(RadialCapacityPotential):
    def __init__(self, config):
        if not isinstance(config, HyFFConfig):
            raise TypeError('An explicit HyFFConfig is required')
        super().__init__(config)

    def expert_accounting(self):
        result = super().expert_accounting()
        result.update(geometric_aggregations=len(self.blocks),
                      atom_only_updates=sum(len(group) for group in self.refinements),
                      requires_covalent_topology=False)
        return result


def load_checkpoint(path, device='cpu'):
    raise ValueError('Checkpoint loading is not available here; use hyff.weights.load_model')
