"""Exact geometry capacities with bounded NumPy blocks and no geometry reuse.

All graphs are independently counted, including repeated input geometries.
Only molecule sizes determine the temporary packing groups. Inputs are the
validated flat collection accepted by ``batch_collection``.
"""
from __future__ import annotations

import math
import numpy as np


# Conservative live-array estimate for each distance block, in bytes. The
# actual pair arrays are two axiswise FP64 distance arrays and a bool. The
# squared distance is accumulated as (dx^2 + dy^2) + dz^2.
_BLOCK_BYTES = 96 * 1024 * 1024
_SIZE_QUANTUM = 16


def _geometry_counts(ptr, xyz, sizes, radius):
    edges = np.zeros(len(sizes), dtype=np.int64)
    degrees = np.zeros(len(sizes), dtype=np.int64)
    padded = ((sizes + _SIZE_QUANTUM - 1) // _SIZE_QUANTUM) * _SIZE_QUANTUM
    order = np.argsort(padded, kind='stable')
    ordered_sizes = padded[order]
    boundaries = np.r_[0, np.flatnonzero(ordered_sizes[1:] != ordered_sizes[:-1]) + 1, len(order)]
    radius2 = radius * radius
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        width = int(ordered_sizes[begin])
        # Row tiling also bounds a single unusually large molecule. Packing
        # and masks reserve 64 bytes per padded atom in this estimate.
        rows = min(256, width, max(1, (_BLOCK_BYTES // width - 64) // 64))
        block_graphs = max(1, _BLOCK_BYTES // (width * (rows * 64 + 64)))
        columns = np.arange(width)
        for first in range(int(begin), int(end), block_graphs):
            ids = order[first:min(first + block_graphs, int(end))]
            valid = columns[None, :] < sizes[ids, None]
            gather = ptr[ids, None] + columns[None, :]
            gather[~valid] = 0
            points = xyz[gather].astype(np.float64)
            block_edges = np.zeros(len(ids), dtype=np.int64)
            block_degrees = np.zeros(len(ids), dtype=np.int64)
            for left in range(0, width, rows):
                right = min(left + rows, width)
                delta = points[:, left:right, 0, None] - points[:, None, :, 0]
                np.square(delta, out=delta)
                scratch = np.empty_like(delta)
                for axis in (1, 2):
                    np.subtract(points[:, left:right, axis, None], points[:, None, :, axis], out=scratch)
                    np.square(scratch, out=scratch)
                    np.add(delta, scratch, out=delta)
                mask = delta <= radius2
                mask &= valid[:, left:right, None]
                mask &= valid[:, None, :]
                mask[:, np.arange(right - left), np.arange(left, right)] = False
                counts = np.count_nonzero(mask, axis=2)
                block_edges += counts.sum(axis=1)
                block_degrees = np.maximum(block_degrees, counts.max(axis=1))
            edges[ids] = block_edges
            degrees[ids] = block_degrees
    return edges, degrees


def _batch_species(ptr, z, sizes, batch):
    """One element histogram per scheduled batch, in the element order described below."""
    ids = np.asarray(batch, dtype=np.int64)
    lengths = sizes[ids]
    ends = np.cumsum(lengths)
    packed_starts = np.r_[0, ends[:-1]]
    flat = np.repeat(ptr[ids] - packed_starts, lengths) + np.arange(int(ends[-1]))
    elements = z[flat]
    totals = np.bincount(elements, minlength=119)
    # Elements are ordered by the first graph containing them, then by atomic
    # number: the order of a graph-by-graph scan of each graph's sorted species.
    first_graph = np.full(119, len(ids), dtype=np.int64)
    np.minimum.at(first_graph, elements, np.repeat(np.arange(len(ids)), lengths))
    present = np.flatnonzero(totals)
    ordered = present[np.lexsort((present, first_graph[present]))]
    return ((int(element), int(totals[element])) for element in ordered)


def geometry_capacities(values, plan, cutoff, *, edge_capacity=0, neighbor_capacity=0):
    """Return exact per-bucket capacity bounds, counted in vectorized blocks rather than per molecule."""
    if not math.isfinite(float(cutoff)) or cutoff <= 0:
        raise ValueError('Model cutoff must be finite and positive')
    if type(edge_capacity) is not int or edge_capacity < 0:
        raise ValueError('edge_capacity must be a nonnegative integer')
    if type(neighbor_capacity) is not int or neighbor_capacity < 0 or neighbor_capacity % 8:
        raise ValueError('neighbor_capacity must be zero or a positive multiple of eight')
    ptr, xyz, z = values['offsets'], values['positions'], values['numbers']
    sizes = ptr[1:] - ptr[:-1]
    edges, degrees = _geometry_counts(ptr, xyz, sizes, float(cutoff) + 1e-4)
    result = []
    for bucket, initial in enumerate(plan['buckets']):
        batches = [entry['indices'] for entry in plan['schedule'] if entry['bucket'] == bucket]
        max_edges = max(int(edges[np.asarray(batch, dtype=np.int64)].sum()) for batch in batches)
        max_degree = max(int(degrees[np.asarray(batch, dtype=np.int64)].max()) for batch in batches)
        if edge_capacity and edge_capacity < max_edges:
            raise ValueError('Explicit global edge capacity is below the geometry bound')
        if neighbor_capacity and neighbor_capacity < max_degree:
            raise ValueError('Explicit neighbor capacity is below the geometry bound')
        experts = {}
        max_pairs = 0
        for batch in batches:
            for element, number in _batch_species(ptr, z, sizes, batch):
                experts[element] = max(experts.get(element, 0), number)
            max_pairs = max(max_pairs, sum(int(sizes[i]) * (int(sizes[i]) - 1) // 2 for i in batch))
        result.append(dict(**initial,
            edge_capacity=edge_capacity or max(1024, ((math.ceil(max_edges * 1.10) + 1023) // 1024) * 1024),
            neighbor_capacity=neighbor_capacity or max(8, ((math.ceil(max_degree * 1.05) + 7) // 8) * 8),
            expert_capacities=experts, pair_capacity=max(1, math.ceil(max_pairs * 1.05)),
            maximum_directed_edges=max_edges, maximum_degree=max_degree, maximum_real_pairs=max_pairs))
    return result
