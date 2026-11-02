"""Prepare-once capacity buckets for native largo batching with packed latent-charge pairs."""
from __future__ import annotations

import copy
from collections.abc import Mapping

from hyff.batch.base import BatchBucket
from hyff.batch.latent_charge_pairs import NativeBatchedPotential
from hyff.batch.pool import _bucket_spec, _efm_capacity


def fork_native_bucket(prepared, atom_capacity, edge_capacity, graph_capacity, *,
                       pair_capacity=None, neighbor_capacity=None, mode=None, backend=None):
    """Share transformed operators; own all captured buffers, including pairs."""
    if type(prepared) is not NativeBatchedPotential:
        raise TypeError('Expected the packed largo batching API')
    mode = prepared.mode if mode is None else mode
    backend = prepared.backend if backend is None else backend
    neighbor_capacity = prepared.neighbor_capacity if neighbor_capacity is None else neighbor_capacity
    pair_capacity = prepared.pair_capacity if pair_capacity is None else pair_capacity
    if mode not in {'energy', 'energy_forces'} or backend not in {'eager', 'cuda_graph'}:
        raise ValueError('Invalid mode or backend')
    if type(neighbor_capacity) is not int or neighbor_capacity < 8 or neighbor_capacity % 8:
        raise ValueError('neighbor_capacity must be a positive multiple of eight, at least eight')
    if pair_capacity is not None and (type(pair_capacity) is not int or pair_capacity < 1):
        raise ValueError('pair_capacity must be a positive integer')
    bucket = BatchBucket(atom_capacity, edge_capacity, graph_capacity)
    result = copy.copy(prepared)
    result.bucket, result.mode, result.backend = bucket, mode, backend
    result.neighbor_capacity, result.pair_capacity = neighbor_capacity, pair_capacity
    result._entries, result._fixed_entry, result.last_call = {}, None, {}
    result._routing_entries, result._active_capacities = {}, None
    result._efm_bucket_capacity = _efm_capacity(edge_capacity)
    return result


def _packed_spec(spec):
    if not isinstance(spec, Mapping):
        return _bucket_spec(spec)
    ordinary = dict(spec)
    pair = ordinary.pop('pair_capacity', None)
    result = _bucket_spec(ordinary)
    if 'expert_capacities' in result:
        raise ValueError('largo packed buckets do not use andante expert capacities')
    if pair is not None:
        result['pair_capacity'] = pair
    return result


class NativeBucketPool:
    """Accept (A,E,G) tuples or named bucket dictionaries plus pair_capacity.

    The first bucket prepares one largo model and installs packed latent-charge pairs.
    Forks share the model and serialization lock. Every pair/CSR/state/routing
    buffer and graph belongs to one entry; staging rebinds the electronic-embedding/latent-charge
    metadata and packed pair context before capture/eager/replay. New capacity
    shapes still incur first-use compilation/capture and belong in cold costs.
    """
    def __init__(self, model, buckets, **native_options):
        specs = tuple(_packed_spec(spec) for spec in buckets)
        if not specs:
            raise ValueError('At least one bucket is required')
        first = NativeBatchedPotential(model, **{**native_options, **specs[0]})
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
