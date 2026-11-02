"""Native batching entry point for all five models.

presto uses its own bucket kernels (native_presto), vivace and moderato the
dense native operators with force-time FP32 LayerNorm statistics (default) or
FP64 statistics (native_dense), andante and largo the edge-frame-mixing
operators (native_efm). Setup-time first messages keep centered FP64
normalization. andante can additionally fix per-element expert row capacities.
"""
from collections.abc import Mapping

import numpy as np

from hyff.batch.base import BatchedPotential, _inputs, _reference_products
from hyff.batch.native_dense import torch, MethodType, full, large, _initial_scalar, _full_forward
from hyff.batch.native_dense import NativeBatchedPotential as DenseBatchedPotential
from hyff.batch.native_efm import NativeBatchedPotential as EFMBatchedPotential
from hyff.kernels.layernorm_fp32 import enable_frozen_layernorm
from hyff.kernels.local_features_fp32 import enable_frozen_local_features


def normalize_expert_capacities(capacities, species):
    if capacities is None:
        return None
    if isinstance(capacities, Mapping):
        if set(capacities).difference(species):
            raise ValueError('Expert capacities include unsupported atomic numbers')
        result = tuple(capacities.get(z, 0) for z in species)
    else:
        result = tuple(capacities)
    if len(result) != len(species) or any(type(value) is not int or value < 0 for value in result):
        raise ValueError('Provide one nonnegative integer row capacity per model species')
    return result


def prepare_full(model, tier, precision='tf32x3', statistics='fp32', compiled_nodes=True):
    """Apply the native single-system setup to a freshly loaded model; no config or export file is read."""
    if tier not in {'vivace', 'moderato'} or precision != 'tf32x3' or statistics != 'fp32':
        raise ValueError('vivace/moderato native batching requires tf32x3 products and fp32 statistics')
    expected = {'vivace': (64, 32, 8, 3), 'moderato': (128, 64, 16, 4)}[tier]
    actual = (model.config.hidden, model.tensor_hidden, model.config.edge_hidden, len(model.blocks))
    if actual != expected:
        raise ValueError('Raw model widths differ from the declared native tier')
    model.eval().requires_grad_(False)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('Native operators require FP32 learned tensors')
    if model.radial_tables is not None or model.fused_messages or model.reference_chunk_size:
        raise ValueError('A fresh raw model is required')
    model._engine_global_context = None
    model._engine_expert_block = 64
    model.reference_chunk_size = 4096
    model.compile_radial_tables(200, order=5)
    model.fused_messages = model.fused = True
    model.fused_message_settings = (16, 8, 4, True)
    model.filter_skin_neighbors = True
    full.base.configure_neighbor_reserve(model, 1.25)
    if model.config.edge_hidden == 8:
        from hyff.kernels.compact_pair_messages import _compact_message_forward
        for block in model.blocks:
            block.forward = MethodType(_compact_message_forward, block)
    full.enable_joined_nodes(model, fold_first=True)
    delattr(model, 'frozen_first_messages')
    large._convert_selected(model, precision)
    full.enable_fast_nodes(model, fuse_activations=True, strided_inputs=True, precision=precision)
    for module in model.modules():
        if isinstance(module, full.JoinedLinear):
            module.fast, module.precision = True, precision
    enable_frozen_layernorm(model, atom_tile=8)
    enable_frozen_local_features(model, atom_tile=4)
    full.enable_grouped_experts(model, precision)
    model.initial_scalar = MethodType(_initial_scalar, model)
    model.forward = MethodType(_full_forward, model)
    if compiled_nodes:
        model.enable_compiled_nodes()
    return model


class NativeBatchedPotential(EFMBatchedPotential):
    """Native batched energies and forces for one model and one capacity bucket.

    Statistics default to fp32 for presto, vivace and moderato and fp64 for
    andante and largo. For andante, ``expert_capacities`` (a sequence in
    ``model.atomic_numbers`` order, or a mapping from atomic number to rows;
    missing entries mean zero) fixes the expert rows of one capture per
    geometric bucket. Unused rows target an ignored dummy atom. Capacity
    overflow raises before evaluation. Omitting it keeps the per-profile
    capture cache.
    """
    def __init__(self, model, atom_capacity, edge_capacity, graph_capacity, *,
                 device='cuda', backend='cuda_graph', mode='energy_forces', tier='presto',
                 precision=None, statistics=None, neighbor_capacity=128,
                 compiled_nodes=True, expert_capacities=None, max_cached_profiles=16):
        statistics = statistics if statistics is not None else ('fp32' if tier in {'presto', 'vivace', 'moderato'} else 'fp64')
        precision = precision or ('fp32' if tier == 'andante' else 'tf32x3')
        if tier not in {'vivace', 'moderato'} or statistics != 'fp32':
            super().__init__(model, atom_capacity, edge_capacity, graph_capacity,
                device=device, backend=backend, mode=mode, tier=tier,
                precision=precision, statistics=statistics,
                neighbor_capacity=neighbor_capacity, compiled_nodes=compiled_nodes,
                max_cached_profiles=max_cached_profiles)
            if expert_capacities is not None and self.tier != 'andante':
                raise ValueError('Fixed per-element capacities apply only to andante FP32 experts')
            self.fixed_expert_capacities = normalize_expert_capacities(
                expert_capacities, self.model.atomic_numbers)
            return
        if expert_capacities is not None:
            raise ValueError('Fixed per-element capacities apply only to andante FP32 experts')
        if torch.device(device).type != 'cuda':
            raise ValueError('The native prepared kernels require CUDA')
        if type(neighbor_capacity) is not int or neighbor_capacity < 8 or neighbor_capacity % 8:
            raise ValueError('neighbor_capacity must be a positive multiple of eight, at least eight')
        BatchedPotential.__init__(self, model, atom_capacity, edge_capacity, graph_capacity,
            device=device, backend=backend, mode=mode, max_cached_signatures=1)
        with _reference_products():
            prepare_full(self.model, tier, precision, statistics, compiled_nodes)
        self.tier, self.precision, self.statistics = tier, precision, statistics
        self.neighbor_capacity = neighbor_capacity
        self._fixed_entry, self.fixed_expert_capacities = None, None

    def __call__(self, positions, atomic_numbers, offsets, total_charge=None, multiplicity=None):
        if self.tier != 'andante' or self.fixed_expert_capacities is None:
            return super().__call__(positions, atomic_numbers, offsets, total_charge, multiplicity)
        with self._lock:
            values = _inputs(self.bucket, positions, atomic_numbers, offsets, total_charge, multiplicity)
            real = values['numbers'][:values['real_atoms']]
            if set(np.unique(real)).difference(self.model.atomic_numbers):
                raise ValueError('Unsupported atomic numbers in native expert routing')
            for z, capacity in zip(self.model.atomic_numbers, self.fixed_expert_capacities):
                if np.count_nonzero(real == z) > capacity:
                    raise ValueError('andante fixed per-element expert row capacity exceeded')
            self._active_capacities = self.fixed_expert_capacities
            # Skip only the profile-cache selection of the EFM class; the dense class delegates
            # to the fixed-entry implementation, which still stages the EFM inputs and runs the
            # same FP32/EFM evaluator.
            result = DenseBatchedPotential.__call__(self, positions, atomic_numbers, offsets,
                                                    total_charge, multiplicity)
            self._native_efm_state.check_overflow()
            result['metadata'] = self.metadata
            return result

    @property
    def metadata(self):
        value = super().metadata
        capacities = getattr(self, 'fixed_expert_capacities', None)
        if self.tier == 'andante' and capacities is not None:
            value.update(signature_scope='one fixed capacity bucket with fixed per-element FP32 expert rows',
                expert_capacities=dict(zip(self.model.atomic_numbers, capacities)),
                capture_profiles=len(self._entries))
        if self.tier in {'vivace', 'moderato'} and self.statistics == 'fp32':
            value.update(statistics='fp32',
                statistics_scope='force-time standalone and fused scalar LayerNorm forward and input VJP; '
                                 'setup first-message cache remains FP64')
        return value
