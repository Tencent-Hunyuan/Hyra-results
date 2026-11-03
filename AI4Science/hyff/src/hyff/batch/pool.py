"""Serialized native buckets sharing one transformed model, all five models.

Each bucket owns input/state/routing/CSR/capture storage. EFM's operator
functions, flags and read-only pad cache are shared. Its Python capacity/slot
configuration is selected under the common lock before a bucket call. Kernel
pointers and launch sizes are held by that bucket's CUDA graph.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

from hyff.batch.base import BatchBucket
from hyff.batch.native import NativeBatchedPotential as NativeEntry, normalize_expert_capacities


class NativeBatchedPotential(NativeEntry):
    """Native batching with serialized per-bucket EFM configuration selection."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._efm_bucket_capacity = _efm_capacity(self.bucket.edge_capacity)

    def __call__(self, *args, **kwargs):
        if self.tier not in {'andante', 'largo'}:
            return super().__call__(*args, **kwargs)
        with self._lock:
            state = self._native_efm_state
            # Setup-only scalar reads in native _csr_geometry stay disabled:
            # fixed CSR capacity is a checked upper bound for every radius.
            state.capacity = self._efm_bucket_capacity
            state.slots = self.neighbor_capacity
            state.stats['initial_max_count'] = self.neighbor_capacity
            return super().__call__(*args, **kwargs)


def _efm_capacity(edge_capacity):
    return max(1024, ((edge_capacity + 1023) // 1024) * 1024)


def fork_native_bucket(prepared, atom_capacity, edge_capacity, graph_capacity, *,
                       expert_capacities=None, neighbor_capacity=None,
                       mode=None, backend=None):
    """Fork an empty bucket; share preparation and call serialization only.

    Use this module's NativeBatchedPotential so every EFM call selects its
    own shape configuration. andante capacities default to the parent's vector;
    supply each bucket's scheduled maximum element counts when they differ.
    """
    if type(prepared) is not NativeBatchedPotential:
        raise TypeError('Expected this module\'s prepared native API')
    mode = prepared.mode if mode is None else mode
    backend = prepared.backend if backend is None else backend
    neighbor_capacity = prepared.neighbor_capacity if neighbor_capacity is None else neighbor_capacity
    if mode not in {'energy', 'energy_forces'} or backend not in {'eager', 'cuda_graph'}:
        raise ValueError('Invalid mode or backend')
    if type(neighbor_capacity) is not int or neighbor_capacity < 8 or neighbor_capacity % 8:
        raise ValueError('neighbor_capacity must be a positive multiple of eight, at least eight')
    if expert_capacities is not None and prepared.tier != 'andante':
        raise ValueError('Fixed expert capacities apply only to andante')
    bucket = BatchBucket(atom_capacity, edge_capacity, graph_capacity)
    capacities = (prepared.fixed_expert_capacities if expert_capacities is None else
                  normalize_expert_capacities(expert_capacities, prepared.model.atomic_numbers))
    result = copy.copy(prepared)
    result.bucket, result.mode, result.backend = bucket, mode, backend
    result.neighbor_capacity = neighbor_capacity
    result._entries, result._fixed_entry, result.last_call = {}, None, {}
    result._efm_bucket_capacity = _efm_capacity(edge_capacity)
    result.fixed_expert_capacities = capacities
    if hasattr(result, '_routing_entries'):
        result._routing_entries, result._active_capacities = {}, None
    return result


def _bucket_spec(spec):
    if isinstance(spec, Mapping):
        result = dict(spec)
    else:
        values = tuple(spec)
        if len(values) != 3:
            raise ValueError('A capacity tuple must have exactly three entries')
        result = dict(zip(('atom_capacity', 'edge_capacity', 'graph_capacity'), values))
    required = {'atom_capacity', 'edge_capacity', 'graph_capacity'}
    allowed = required | {'expert_capacities', 'neighbor_capacity', 'mode', 'backend'}
    if not required.issubset(result) or set(result).difference(allowed):
        raise ValueError('Bucket requires atom/edge/graph capacities and only supported overrides')
    return result


class NativeBucketPool:
    """Prepare once; include every bucket's first compile/capture in cold cost.

    ``buckets`` accepts (A,E,G) tuples or dictionaries with those capacity
    names and optional expert_capacities, neighbor_capacity, mode, backend.
    ``native_options`` contains tier/device/precision/statistics/compiled_nodes.
    Calls through all pool buckets share one lock and must be serialized.
    """
    def __init__(self, model, buckets, **native_options):
        specs = tuple(_bucket_spec(spec) for spec in buckets)
        if not specs:
            raise ValueError('At least one capacity bucket is required')
        first_options = {**native_options, **specs[0]}
        first = NativeBatchedPotential(model, **first_options)
        self.buckets = [first]
        self.buckets.extend(fork_native_bucket(first, **spec) for spec in specs[1:])

    @classmethod
    def from_weights(cls, weights, buckets, *, device='cuda', **native_options):
        from hyff.weights import load_model
        model, _ = load_model(weights, device)
        return cls(model, buckets, device=device, **native_options)

    def add_bucket(self, atom_capacity, edge_capacity, graph_capacity, **options):
        result = fork_native_bucket(self.buckets[0], atom_capacity, edge_capacity,
                                    graph_capacity, **options)
        self.buckets.append(result)
        return result

    def clear(self):
        for bucket in self.buckets:
            bucket.clear()
