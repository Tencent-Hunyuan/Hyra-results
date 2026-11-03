"""Radius-view selection helpers for the native engines (two radius views in one launch)."""
from contextlib import contextmanager
from unittest.mock import patch

import triton as tr
import triton.language as tl

from hyff.kernels.layernorm import enable_frozen_layernorm
from hyff.kernels.fused_messages import PreparedL2Neighbors, _geometry


@tr.jit
def _select_two(P, Cell, Neighbors, Shifts, ON0, OS0, OC0, ON1, OS1, OC1,
                N: tl.constexpr, K: tl.constexpr, B: tl.constexpr,
                C0: tl.constexpr, C1: tl.constexpr, BM: tl.constexpr):
    node = tl.program_id(0)*BM + tl.arange(0, BM)
    safe = tl.minimum(node, N-1)
    slot = tl.arange(0, B)
    neighbor, valid, distance, _, _, _ = _geometry(
        P, Cell, Neighbors, Shifts, safe[:, None], slot[None, :], K)
    valid &= node[:, None] < N
    sx = tl.load(Shifts+(safe[:, None]*K+slot[None, :])*3, slot[None, :] < K, 0)
    sy = tl.load(Shifts+(safe[:, None]*K+slot[None, :])*3+1, slot[None, :] < K, 0)
    sz = tl.load(Shifts+(safe[:, None]*K+slot[None, :])*3+2, slot[None, :] < K, 0)
    for which in tl.static_range(2):
        if which == 0:
            ON, OS, OC, cutoff = ON0, OS0, OC0, C0
        else:
            ON, OS, OC, cutoff = ON1, OS1, OC1, C1
        keep = valid & (distance < cutoff)
        count = tl.sum(keep.to(tl.int32), 1)
        rank = tl.cumsum(keep.to(tl.int32), 1)-1
        tail = (node[:, None] < N) & (slot[None, :] < K) & (slot[None, :] >= count[:, None])
        tail_index = node[:, None]*K+slot[None, :]
        out_index = node[:, None]*K+rank
        tl.store(ON+tail_index, -1, tail)
        # Geometry masks every negative neighbor before using its displacement.
        # Clear neighbor sentinels, while leaving ignored tail shifts untouched.
        # This avoids writing three unused integers per padded slot, per radius.
        tl.store(ON+out_index, neighbor, keep)
        tl.store(OS+out_index*3, sx, keep)
        tl.store(OS+out_index*3+1, sy, keep)
        tl.store(OS+out_index*3+2, sz, keep)
        tl.store(OC+node, count, node < N)


_reference_refresh = PreparedL2Neighbors.refresh_subgraphs


def refresh_two(self, positions, active_depth, atom_tile=4):
    radii = set(self.model.layer_cutoffs[:active_depth])
    if self.filter_skin:
        radii.add(self.model.cutoff)
    radii = sorted(r for r in radii if r in self.radius_views)
    if len(radii) != 2:
        return _reference_refresh(self, positions, active_depth)
    first, second = (self.radius_views[r] for r in radii)
    _select_two[(tr.cdiv(self.n, atom_tile),)](
        positions, self.cell, self.neighbors, self.shifts,
        first.neighbors, first.shifts, first.counts,
        second.neighbors, second.shifts, second.counts,
        self.n, self.capacity, tr.next_power_of_2(self.capacity),
        radii[0], radii[1], atom_tile, num_warps=4)


def install_norm(model, tile=16):
    return enable_frozen_layernorm(model, atom_tile=tile)


@contextmanager
def radius_variant(tile=4):
    def refresh(self, positions, active_depth):
        return refresh_two(self, positions, active_depth, tile)
    with patch.object(PreparedL2Neighbors, 'refresh_subgraphs', refresh):
        yield
