"""CUDA stable per-owned-row selection using the base-message geometry helper."""
import torch
import triton as tr
import triton.language as tl
from hyff.multigpu.base_messages import _owned_geometry


@tr.jit
def _select(P, Cell, Neighbors, Shifts, Counts, OwnedRows,
            OutNeighbors, OutShifts, OutCounts, Status,
            K: tl.constexpr, B: tl.constexpr, CUTOFF: tl.constexpr):
    node = tl.program_id(0)
    slot = tl.arange(0, B)
    skin_count = tl.load(Counts + node)
    neighbor, valid, distance, _, _, _ = _owned_geometry(
        P, Cell, Neighbors, Shifts, OwnedRows, node, slot, K)
    prefix = slot < skin_count
    malformed = (skin_count < 0) | (skin_count > K)
    malformed = malformed | (tl.sum(((slot < K) & (prefix != valid)).to(tl.int32), 0) > 0)
    malformed = malformed | (tl.sum((valid & prefix &
        ((distance != distance) | (distance == float("inf")))).to(tl.int32), 0) > 0)
    if malformed:
        tl.atomic_or(Status, malformed.to(tl.int64), sem="relaxed")
    keep = valid & prefix & (distance < CUTOFF)
    count = tl.sum(keep.to(tl.int32), 0)
    rank = tl.cumsum(keep.to(tl.int32), 0) - 1
    # Tail clears and kept scatter targets are disjoint, including count=0/K.
    tail = (slot < K) & (slot >= count)
    tl.store(OutNeighbors + node*K + slot, -1, tail)
    for axis in tl.static_range(3):
        shift = tl.load(Shifts + (node*K + slot)*3 + axis, slot < K, 0)
        tl.store(OutShifts + (node*K + slot)*3 + axis, 0, tail)
        tl.store(OutShifts + (node*K + rank)*3 + axis, shift, keep)
    tl.store(OutNeighbors + node*K + rank, neighbor, keep)
    tl.store(OutCounts + node, count)


@torch.no_grad()
def select_cuda(graph, positions, *, cutoff, fma):
    neighbors = torch.empty_like(graph.neighbors)
    shifts = torch.empty_like(graph.shifts)
    counts = torch.empty_like(graph.counts)
    status = torch.zeros(1, dtype=torch.int64, device=positions.device)
    if graph.n_local:
        _select[(graph.n_local,)](positions, graph.cell, graph.neighbors, graph.shifts,
            graph.counts, graph.local_rows, neighbors, shifts, counts, status,
            graph.capacity, tr.next_power_of_2(graph.capacity), cutoff,
            num_warps=4, enable_fp_fusion=fma)
    return neighbors, shifts, counts, status
