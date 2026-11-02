"""Per-graph FP32 GPU capacity counts, with no geometry cache or deduplication.

The cutoff+1e-3 margin is a conservative capacity choice. Whether the counts
bound the native CSR edges, and how long counting takes, depend on the GPU and
the inputs; check both on representative inputs.
Importing this module uses only NumPy; Torch/Triton load on an explicit call.
"""
import math

import numpy as np

from hyff.batch.geometry_vectorized import _batch_species

_KERNEL = None


def _load_kernel():
    global _KERNEL, tl
    if _KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def count_kernel(X, PTR, IDS, EDGES, DEGREES, RADIUS2, COLS: tl.constexpr):
            graph = tl.load(IDS + tl.program_id(0))
            start = tl.load(PTR + graph)
            size = tl.load(PTR + graph + 1) - start
            row = tl.program_id(1) * 8 + tl.arange(0, 8)
            valid_row = row < size
            rx = tl.load(X + 3 * (start + row), mask=valid_row, other=0)
            ry = tl.load(X + 3 * (start + row) + 1, mask=valid_row, other=0)
            rz = tl.load(X + 3 * (start + row) + 2, mask=valid_row, other=0)
            counts = tl.full((8,), 0, tl.int32)
            for base in range(0, tl.cdiv(size, COLS)):
                col = base * COLS + tl.arange(0, COLS)
                valid_col = col < size
                cx = tl.load(X + 3 * (start + col), mask=valid_col, other=0)
                cy = tl.load(X + 3 * (start + col) + 1, mask=valid_col, other=0)
                cz = tl.load(X + 3 * (start + col) + 2, mask=valid_col, other=0)
                dx, dy, dz = rx[:, None] - cx[None, :], ry[:, None] - cy[None, :], rz[:, None] - cz[None, :]
                distance2 = dx * dx + dy * dy + dz * dz
                keep = (distance2 <= RADIUS2) & valid_row[:, None] & valid_col[None, :] & (row[:, None] != col[None, :])
                counts += tl.sum(tl.cast(keep, tl.int32), axis=1)
            tl.atomic_add(EDGES + graph, tl.sum(tl.cast(counts, tl.int64), axis=0))
            tl.atomic_max(DEGREES + graph, tl.cast(tl.max(counts, axis=0), tl.int64))

        _KERNEL = count_kernel
    return _KERNEL


def _groups(sizes):
    """Bound columns at 512; a program loops column tiles for larger graphs."""
    widths = np.ones(len(sizes), dtype=np.int64)
    for width in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        widths[sizes > width // 2] = width
    order = np.argsort(widths, kind='stable')
    ordered = widths[order]
    boundaries = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, len(order)]
    groups = [(int(begin), int(end), int(ordered[begin]), int(sizes[order[begin:end]].max()))
              for begin, end in zip(boundaries[:-1], boundaries[1:])]
    return order, groups


def geometry_counts(values, cutoff, *, device='cuda'):
    """Return int64 directed-edge and maximum-degree arrays in input order.

    Every graph is counted independently, including exact repeats. FP32 pair
    distances include nonself pairs at or below (cutoff+1e-3)^2. This is neither
    a physical neighbor-list change nor a substitute for native CSR validation.
    No error triggers a CPU or other-backend retry.
    """
    if not math.isfinite(float(cutoff)) or cutoff <= 0:
        raise ValueError('Model cutoff must be finite and positive')
    ptr, xyz = np.asarray(values['offsets']), np.asarray(values['positions'])
    if (ptr.dtype != np.int64 or ptr.ndim != 1 or len(ptr) < 2 or ptr[0] != 0 or
            not np.all(ptr[1:] > ptr[:-1]) or xyz.dtype != np.float32 or
            xyz.shape != (int(ptr[-1]), 3) or not np.isfinite(xyz).all()):
        raise ValueError('GPU counting requires validated flat FP32 geometry and int64 offsets')
    with np.errstate(over='ignore', invalid='ignore'):
        radius = np.float32(float(cutoff) + 1e-3)
        radius2 = np.float32(radius * radius)
    if not np.isfinite(radius2) or radius2 <= 0:
        raise ValueError('Capacity radius must have a finite positive FP32 square')
    sizes = ptr[1:] - ptr[:-1]
    if int(sizes.max()) >= np.iinfo(np.int32).max:
        raise ValueError('Per-atom degree exceeds the kernel integer range')
    order, groups = _groups(sizes)
    import torch
    target = torch.device(device)
    if target.type != 'cuda':
        raise ValueError('GPU geometry counting requires a CUDA device')
    kernel = _load_kernel()
    points = torch.as_tensor(np.ascontiguousarray(xyz), device=target)
    offsets = torch.as_tensor(np.ascontiguousarray(ptr), device=target)
    ids = torch.as_tensor(order, device=target)
    edges = torch.zeros(len(sizes), dtype=torch.int64, device=target)
    degrees = torch.zeros_like(edges)
    for begin, end, width, largest in groups:
        kernel[(end - begin, (largest + 7) // 8)](
            points, offsets, ids[begin:end], edges, degrees, float(radius2),
            COLS=width, num_warps=4, enable_fp_fusion=False)
    # Both transfers synchronize their producer stream before exposing NumPy.
    return edges.cpu().numpy(), degrees.cpu().numpy()


def geometry_capacities(values, plan, cutoff, *, edge_capacity=0, neighbor_capacity=0, device='cuda'):
    """GPU graph counts with the vectorized CPU scheduling algebra."""
    if type(edge_capacity) is not int or edge_capacity < 0:
        raise ValueError('edge_capacity must be a nonnegative integer')
    if type(neighbor_capacity) is not int or neighbor_capacity < 0 or neighbor_capacity % 8:
        raise ValueError('neighbor_capacity must be zero or a positive multiple of eight')
    ptr, z = values['offsets'], values['numbers']
    sizes = ptr[1:] - ptr[:-1]
    edges, degrees = geometry_counts(values, cutoff, device=device)
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
