"""Fixed-capacity within-graph latent-charge pairs for native largo batching.

Uses the native charge-head products, charge conservation, screened kernel,
coupling and scale; only pair enumeration and reduction differ from the dense
term. Each real pair appears once in increasing source/destination order;
ignored dummy energies have no pair work. Capacity overflow raises, never truncates.
"""
from __future__ import annotations

import math
from types import MethodType

import numpy as np
import torch

from hyff.batch.pool import NativeBatchedPotential as NativePool


def pair_arrays(values, capacity):
    counts = values['counts'][:values['real_graphs']]
    required = sum(int(count) * (int(count) - 1) // 2 for count in counts)
    if type(capacity) is not int or capacity < 1 or required > capacity:
        raise ValueError('Fixed latent-charge pair capacity exceeded or invalid')
    dummy = len(values['numbers']) - 1
    source = np.full(capacity, dummy, np.int64)
    destination = np.full(capacity, dummy, np.int64)
    valid = np.zeros(capacity, np.float32)
    cursor = start = 0
    for count in counts:
        left, right = np.triu_indices(int(count), k=1)
        end = cursor + len(left)
        source[cursor:end], destination[cursor:end] = left + start, right + start
        valid[cursor:end] = 1.
        cursor, start = end, start + int(count)
    return source, destination, valid, required


def packed_latent_charge_forward(self, scalars, positions, atom_batch, total_charge):
    source, destination, valid = self.__dict__['_batch_latent_charge_pairs']
    with torch.autocast(device_type=positions.device.type, enabled=False):
        positions, atom_batch = positions.float(), atom_batch.long()
        # Keep native FastLinear/TF32x3 head after native bind_base conversion.
        charge = self.conserved_charges(scalars, atom_batch, total_charge)
        delta = positions[source] - positions[destination]
        kernel = self._screened_kernel(delta.square().sum(dim=-1))
        pair_energy = charge[source] * charge[destination] * kernel * valid
        row_energy = positions.new_zeros(len(positions)).index_add(0, source, pair_energy)
        energy = positions.new_zeros(total_charge.numel()).index_add(0, atom_batch, row_energy)
        return (self.coupling_ev_angstrom * self.scale) * energy


def install_packed_latent_charge(term):
    if '_batch_latent_charge_original_forward' in term.__dict__:
        raise ValueError('Packed latent-charge pairs are already installed')
    term.__dict__['_batch_latent_charge_original_forward'] = term.forward
    term.forward = MethodType(packed_latent_charge_forward, term)


class NativeBatchedPotential(NativePool):
    """Native largo batching with a fixed real-pair capacity per captured bucket.

    ``pair_capacity`` should bound max(sum(Ng*(Ng-1)/2)) over the scheduled
    microbatches. Omitting it sizes from the first call plus 5%; later overflow
    raises. Include pair planning/staging and every compile/capture in timing.
    No precision policy, learned tensor or electronic-embedding/latent-charge wrapper is replaced.
    """
    def __init__(self, model, atom_capacity, edge_capacity, graph_capacity, *,
                 pair_capacity=None, **native_options):
        native_options.setdefault('tier', 'largo')
        if native_options['tier'] not in {'largo', 'largo', 'largo'}:
            raise ValueError('Packed latent-charge batching applies to largo only')
        if pair_capacity is not None and (type(pair_capacity) is not int or pair_capacity < 1):
            raise ValueError('pair_capacity must be a positive integer')
        super().__init__(model, atom_capacity, edge_capacity, graph_capacity, **native_options)
        self.pair_capacity = pair_capacity
        install_packed_latent_charge(self.model.latent_charge_energy)

    def _stage_native(self, entry, values):
        super()._stage_native(entry, values)
        required = sum(int(count) * (int(count) - 1) // 2
                       for count in values['counts'][:values['real_graphs']])
        if self.pair_capacity is None:
            self.pair_capacity = max(256, ((math.ceil(required * 1.05) + 255) // 256) * 256)
        arrays = pair_arrays(values, self.pair_capacity)
        if 'latent_charge_pairs' not in entry:
            entry['latent_charge_pairs'] = tuple(torch.empty(self.pair_capacity, dtype=dtype,
                device=self.device) for dtype in (torch.int64, torch.int64, torch.float32))
            entry['latent_charge_pairs_host'] = tuple(torch.empty_like(value, device='cpu', pin_memory=True)
                                               for value in entry['latent_charge_pairs'])
        for target, host, array in zip(entry['latent_charge_pairs'], entry['latent_charge_pairs_host'], arrays[:3]):
            host.copy_(torch.from_numpy(array))
            target.copy_(host, non_blocking=True)
        entry['latent_charge_real_pairs'] = arrays[3]
        self.model.latent_charge_energy.__dict__['_batch_latent_charge_pairs'] = entry['latent_charge_pairs']

    @property
    def metadata(self):
        value = super().metadata
        value.update(execution_route='largo_native_batched_packed_latent_charges',
            native_default_route=False,
            long_range_scope='native charge conservation and screened kernel; real within-graph upper-triangle pairs only',
            latent_charge_pair_capacity=getattr(self, 'pair_capacity', None),
            latent_charge_pair_reduction='pair index_add to atom rows, then graph index_add; FP32 summation order differs from the dense term')
        return value
