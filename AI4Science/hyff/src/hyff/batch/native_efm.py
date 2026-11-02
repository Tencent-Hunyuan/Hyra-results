"""Native batching for andante and largo (edge-frame-mixing models).

Smaller models delegate to native_dense. andante keeps FP32 per-element
expert networks and stabilizers; its capture profiles have fixed padded
per-element capacities, and a new count profile costs another capture. largo
keeps grouped TF32x3 experts and graph-aware electronic-embedding and
latent-charge energy wrappers.
"""
from __future__ import annotations

import threading
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from hyff.batch.base import BatchedPotential, _inputs, _reference_products
from hyff.batch.native_presto import fast
from hyff.batch.native_dense import NativeBatchedPotential as DenseBatchedPotential, _full_forward, forward_fp64


def _xl_profile(numbers, real_atoms, species):
    """Power-of-two row capacity per real species; dummy rows need no update."""
    real = numbers[:real_atoms]
    unsupported = set(np.unique(real)).difference(species)
    if unsupported:
        raise ValueError('Unsupported atomic numbers in native expert routing')
    counts = tuple(int(np.count_nonzero(real == z)) for z in species)
    return tuple(1 << (count - 1).bit_length() if count else 0 for count in counts)


def _stage_andante_partition(batch, values, species, capacities, device):
    """Each unused expert row targets an ignored dummy atom, never a real row.

    The native expert's index_copy may overwrite that dummy repeatedly. No
    real output row has duplicate indices and dummy energies are masked out.
    The FP32 Sequential expert networks and the stabilizer order are not modified.
    """
    bound = batch.get('_xl_padded_partition')
    if bound is None:
        bound = SimpleNamespace(atoms=len(values['numbers']), groups=tuple(
            torch.empty(size, dtype=torch.int64, device=device) for size in capacities))
        batch['_xl_padded_partition'] = bound
    if tuple(len(group) for group in bound.groups) != tuple(capacities):
        raise ValueError('andante expert capacity profile differs from captured storage')
    real = values['numbers'][:values['real_atoms']]
    for z, capacity, target in zip(species, capacities, bound.groups):
        selected = np.flatnonzero(real == z)
        if len(selected) > capacity:
            raise ValueError('andante per-element expert row capacity exceeded')
        indices = np.full(capacity, len(values['numbers']) - 1, dtype=np.int64)
        indices[:len(selected)] = selected
        target.copy_(torch.from_numpy(indices), non_blocking=False)
    batch['_engine_partition'] = bound


def prepare_extended(model, tier, precision, compiled_nodes, edge_capacity,
                     neighbor_capacity, device):
    from hyff.engines.capture.base_port import bind_base
    if tier == 'largo':
        from hyff.engines.capture.compile_policy import configure_largo_compiler
        configure_largo_compiler()
        from hyff.engines.capture.efm_port_grouped import install_efm
    else:
        from hyff.engines.capture.efm_port import install_efm
    bind_base(model, global_context=None, reserve=1.5, precision=precision,
        grouped_experts=tier == 'largo', joined_projections=False, fold_first=False,
        frozen_layernorm=False, fused_features=False)
    # Preserve the base's initial_scalar binding, every stabilizer and all EFM
    # modules. Only the species-table first message becomes graph-conditioned.
    model.forward = MethodType(_full_forward, model)
    model._engine_expert_block = 64
    state = install_efm(model, slots=neighbor_capacity, edge_reserve=1.5,
                        compile_nodes=compiled_nodes)
    # A fixed all-edge upper bound avoids data-dependent EFM compaction growth
    # during capture. Exact counts are still checked after every replay.
    state.capacity = max(1024, ((edge_capacity + 1023) // 1024) * 1024)
    state.stats['initial_max_count'] = neighbor_capacity
    state.stats['initial_edges'] = None
    if tier == 'largo':
        from hyff.models.latent_charge_model import install_prepared_context
        install_prepared_context(model,
            {'positions': torch.empty((1, 3), device=device),
             'total_charge': 0., 'multiplicity': 1}, device)
    return state


class NativeBatchedPotential(DenseBatchedPotential):
    """Native batching with each tier's native products and state/energy wrappers.

    andante uses at most ``max_cached_profiles`` element-count capacity profiles.
    largo uses one capacity bucket; its latent-charge term has a quadratic padded
    atom-pair cost.
    """
    def __init__(self, model, atom_capacity, edge_capacity, graph_capacity, *,
                 device='cuda', backend='cuda_graph', mode='energy_forces', tier='presto',
                 precision=None, statistics=None, neighbor_capacity=128,
                 compiled_nodes=True, max_cached_profiles=16):
        tier = 'largo' if tier in {'largo', 'largo'} else tier
        precision = precision or ('fp32' if tier == 'andante' else 'tf32x3')
        if tier not in {'andante', 'largo'}:
            super().__init__(model, atom_capacity, edge_capacity, graph_capacity,
                device=device, backend=backend, mode=mode, tier=tier,
                precision=precision, statistics=statistics,
                neighbor_capacity=neighbor_capacity, compiled_nodes=compiled_nodes)
            return
        expected = 'fp32' if tier == 'andante' else 'tf32x3'
        if precision != expected or statistics not in {None, 'fp64'}:
            raise ValueError('andante requires fp32 and largo tf32x3 products; both use fp64 statistics')
        if torch.device(device).type != 'cuda':
            raise ValueError('The native prepared kernels require CUDA')
        if type(neighbor_capacity) is not int or neighbor_capacity < 8 or neighbor_capacity % 8:
            raise ValueError('neighbor_capacity must be a positive multiple of eight, at least eight')
        if type(max_cached_profiles) is not int or max_cached_profiles < 1:
            raise ValueError('max_cached_profiles must be a positive integer')
        BatchedPotential.__init__(self, model, atom_capacity, edge_capacity, graph_capacity,
            device=device, backend=backend, mode=mode, max_cached_signatures=max_cached_profiles)
        self.tier, self.precision, self.statistics = tier, precision, 'fp64'
        self.neighbor_capacity = neighbor_capacity
        self._fixed_entry, self._routing_entries, self._active_capacities = None, {}, None
        self._lock = threading.RLock()
        with _reference_products():
            self._native_efm_state = prepare_extended(self.model, tier, precision,
                compiled_nodes, edge_capacity, neighbor_capacity, self.device)

    def _stage_native(self, entry, values):
        if self.tier not in {'andante', 'largo'}:
            return super()._stage_native(entry, values)
        batch, host = entry['batch'], entry['host']
        entry['real_atoms'], entry['real_graphs'] = values['real_atoms'], values['real_graphs']
        arrays = dict(atomic_numbers=values['numbers'], ptr=values['ptr'],
            batch=np.repeat(np.arange(len(values['counts']), dtype=np.int64), values['counts']),
            total_charge=values['charge'], multiplicity=values['multiplicity'])
        mask = np.zeros(self.bucket.graph_capacity + 1, np.float32)
        mask[:values['real_graphs']] = 1.
        arrays['_batch_real_graph_mask'] = mask
        with torch.no_grad():
            self._native_efm_state.flags.zero_()
            for name, array in arrays.items():
                host[name].copy_(torch.from_numpy(array))
                batch[name].copy_(host[name], non_blocking=True)
            if self.tier == 'andante':
                _stage_andante_partition(batch, values, self.model.atomic_numbers,
                                    self._active_capacities, self.device)
            else:
                schedule = fast.schedule(torch.from_numpy(values['numbers']),
                    self.model.atomic_numbers, self.model._engine_expert_block)
                bound = batch['_engine_partition']
                for name in ('order', 'starts', 'tiles'):
                    getattr(bound, name).copy_(getattr(schedule, name), non_blocking=False)
            count = (batch['ptr'][1:] - batch['ptr'][:-1]).to(self.model.embedding.weight)
            features = [batch['total_charge'].to(count) / count]
            if self.model.condition_spin:
                features.append((batch['multiplicity'].to(count) - 1.) / count)
            elif np.any(values['multiplicity'] != 1):
                raise ValueError('A charge-only model has no trained nonsinglet response')
            state = F.linear(torch.stack(features, dim=-1), self.model.state_projection.weight,
                             self.model.state_projection.bias)
            table = self.model.embedding.weight[None, :, :] + state[:, None, :]
            first = self.model.blocks[0]
            normalized = forward_fp64(first.scalar_norm, table)
            source = first.scalar_message
            messages = F.linear(F.silu(F.linear(normalized, source[0].weight, source[0].bias)),
                                source[2].weight, source[2].bias)
            ids, z = batch['batch'], batch['atomic_numbers']
            batch['_engine_initial_scalar'].copy_(table[ids, z])
            batch['_engine_first_message_atoms'].copy_(messages[ids, z])
            elements = self.model.element_lut[z]
            batch['_engine_elements'].copy_(elements)
            batch['_engine_offset'].copy_(self.model.element_offset(z).squeeze(-1))
            if entry.get('prepared') is not None:
                entry['prepared'].elements.copy_(elements)
            if self.tier == 'largo':
                self._stage_main_context(entry, count)
        self._stage(entry, values)

    def _stage_main_context(self, entry, count):
        batch = entry['batch']
        ids, q, m = batch['batch'], batch['total_charge'], batch['multiplicity']
        if '_main_aee_totals' not in batch:
            batch['_main_aee_totals'] = torch.empty((len(q), 2), device=self.device)
            batch['_main_aee_node_state'] = torch.empty((len(ids), 3), device=self.device)
        totals = batch['_main_aee_totals']
        totals.copy_(torch.stack((q, m.to(q) - 1.), dim=-1))
        graph_state = torch.cat((totals, torch.log1p(count)[:, None]), dim=-1)
        node_state = batch['_main_aee_node_state']
        node_state.copy_(graph_state[ids])
        self.model.__dict__['_electronic_prepared_context'] = (ids, totals, node_state)
        self.model.__dict__['_latent_charge_prepared_metadata'] = (ids, q)

    def __call__(self, positions, atomic_numbers, offsets, total_charge=None, multiplicity=None):
        if self.tier not in {'andante', 'largo'}:
            return super().__call__(positions, atomic_numbers, offsets, total_charge, multiplicity)
        with self._lock:
            if self.tier == 'andante':
                values = _inputs(self.bucket, positions, atomic_numbers, offsets, total_charge, multiplicity)
                key = _xl_profile(values['numbers'], values['real_atoms'], self.model.atomic_numbers)
                self._active_capacities = key
                self._fixed_entry = self._routing_entries.get(key)
                if self._fixed_entry is None and len(self._routing_entries) >= self.max_cached_signatures:
                    del self._routing_entries[next(iter(self._routing_entries))]
            result = super().__call__(positions, atomic_numbers, offsets, total_charge, multiplicity)
            self._native_efm_state.check_overflow()
            if self.tier == 'andante':
                self._routing_entries[key] = self._fixed_entry
                self._entries = dict(self._routing_entries)
            result['metadata'] = self.metadata
            return result

    @property
    def metadata(self):
        value = super().metadata
        if self.tier in {'andante', 'largo'}:
            value.update(execution_route=self.tier + '_native_batched',
                native_math=('FP32 per-element experts/stabilizers/FP32 EFM' if self.tier == 'andante'
                    else 'native grouped TF32x3 experts/FP32 EFM/electronic-embedding and latent-charge wrappers'),
                model_statistics_policy='centered FP64 base norm forwards',
                signature_scope=('fixed padded per-element capacity profiles; unseen profiles capture'
                    if self.tier == 'andante' else 'one fixed capacity bucket with graph-aware electronic-embedding/latent-charge buffers'),
                setup_math='per-graph FP32 Q/N,(M-1)/N; centered FP64 setup norm; FP32 first-message products',
                capture_profiles=len(self._routing_entries) if self.tier == 'andante' else len(self._entries),
                long_range_scope='dense all-pairs latent-charge term with graph masking; quadratic in padded atom capacity')
        return value

    def clear(self):
        super().clear()
        if hasattr(self, '_routing_entries'):
            self._routing_entries.clear()
