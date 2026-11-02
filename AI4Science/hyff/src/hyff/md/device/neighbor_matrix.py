"""Fixed-buffer neighbor rebuilds for an existing prepared force graph.

This is an execution adapter for one simulation cell at a time (``set_cell`` moves it
to another), not a force model.
It builds the full directed cutoff-plus-skin list at the supplied coordinates.
All publication buffers keep their addresses. A caller MUST consume ``status``
on device before evaluating/committing a step: a truncated list is never valid.
The old prepared object's COO edges become stale metadata and must not be used
as a reference graph. Radius views are refreshed by the normal force graph.
"""
from __future__ import annotations

import importlib.util  # nvalchemiops 0.3.1 import requirement
import math

import numpy as np
import torch
import triton as tr
import triton.language as tl
import warp as wp

from nvalchemiops.neighbors.cell_list import (
    _cell_list_bin_atoms_overload, _cell_list_construct_bin_size_overload,
    _cell_list_count_atoms_per_bin_overload, query_cell_list)
from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list


OVERFLOW = 1
INVALID_NEIGHBOR = 2
CELL_CHANGED = 4
PBC_CHANGED = 8
INVALID_POSITION = 16


@tr.jit
def _scan_blocks(Values, Offsets, BlockSums, N: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0)
    index = block * B + tl.arange(0, B)
    value = tl.load(Values + index, index < N, 0)
    inclusive = tl.cumsum(value, 0)
    tl.store(Offsets + index, inclusive - value, index < N)
    tl.store(BlockSums + block, tl.sum(value, 0))


@tr.jit
def _add_block_offsets(Offsets, BlockOffsets, N: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0)
    index = block * B + tl.arange(0, B)
    value = tl.load(Offsets + index, index < N, 0)
    tl.store(Offsets + index, value + tl.load(BlockOffsets + block), index < N)


class FixedExclusiveScan:
    """Hierarchical integer scan with setup-only scratch allocation."""
    def __init__(self, values, offsets, block=1024):
        if (values.dtype != torch.int32 or values.ndim != 1 or values.numel() == 0
                or offsets.shape != values.shape or offsets.dtype != values.dtype):
            raise ValueError('Nonempty, equal-shaped int32 vectors required')
        self.block = block
        self.levels = []
        while True:
            sums = torch.empty(tr.cdiv(values.numel(), block), device=values.device, dtype=torch.int32)
            self.levels.append((values, offsets, sums))
            if sums.numel() == 1:
                break
            values, offsets = sums, torch.empty_like(sums)

    def __call__(self):
        for values, offsets, sums in self.levels:
            _scan_blocks[(sums.numel(),)](values, offsets, sums, values.numel(), self.block, num_warps=4)
        for level in range(len(self.levels)-2, -1, -1):
            values, offsets, sums = self.levels[level]
            block_offsets = self.levels[level+1][1]
            _add_block_offsets[(sums.numel(),)](offsets, block_offsets, values.numel(), self.block, num_warps=4)


@tr.jit
def _configuration_status(Cell, FixedCell, PBC, FixedPBC, Status):
    index = tl.arange(0, 16)
    current = tl.load(Cell + index, index < 9, 0.)
    fixed = tl.load(FixedCell + index, index < 9, 0.)
    changed_cell = tl.sum(((current != fixed) & (index < 9)).to(tl.int32), 0) > 0
    current_pbc = tl.load(PBC + index, index < 3, 0)
    fixed_pbc = tl.load(FixedPBC + index, index < 3, 0)
    changed_pbc = tl.sum(((current_pbc != fixed_pbc) & (index < 3)).to(tl.int32), 0) > 0
    tl.store(Status, changed_cell.to(tl.int32) * 4 + changed_pbc.to(tl.int32) * 8)
    tl.store(Status + 1, 0)
    tl.store(Status + 2, 0)
    tl.store(Status + 3, 0)


@tr.jit
def _prepare_positions(Position, SafePosition, InverseCell, Status,
                       N: tl.constexpr, LIMIT: tl.constexpr, B: tl.constexpr):
    atom = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(Position + atom * 3, atom < N, 0.)
    y = tl.load(Position + atom * 3 + 1, atom < N, 0.)
    z = tl.load(Position + atom * 3 + 2, atom < N, 0.)
    fx = x * tl.load(InverseCell) + y * tl.load(InverseCell + 3) + z * tl.load(InverseCell + 6)
    fy = x * tl.load(InverseCell + 1) + y * tl.load(InverseCell + 4) + z * tl.load(InverseCell + 7)
    fz = x * tl.load(InverseCell + 2) + y * tl.load(InverseCell + 5) + z * tl.load(InverseCell + 8)
    # This also rejects NaN/Inf, before Warp converts cell indices to int32.
    valid = (tl.abs(fx) < LIMIT) & (tl.abs(fy) < LIMIT) & (tl.abs(fz) < LIMIT)
    bad = tl.sum(((atom < N) & ~valid).to(tl.int32), 0)
    if bad:
        tl.atomic_or(Status, 16)
        tl.atomic_add(Status + 3, bad)
    tl.store(SafePosition + atom * 3, tl.where(valid, x, 0.), atom < N)
    tl.store(SafePosition + atom * 3 + 1, tl.where(valid, y, 0.), atom < N)
    tl.store(SafePosition + atom * 3 + 2, tl.where(valid, z, 0.), atom < N)


@tr.jit
def _publish_matrix(NativeNeighbors, NativeShifts, NativeCounts,
                    Neighbors, Shifts, Counts, Status,
                    N: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    atom = tl.program_id(0)
    slot = tl.arange(0, B)
    count = tl.load(NativeCounts + atom)
    bounded_count = tl.minimum(tl.maximum(count, 0), K)
    present = slot < bounded_count
    neighbor = tl.load(NativeNeighbors + atom * K + slot, (slot < K) & present, -1)
    sx = tl.load(NativeShifts + (atom * K + slot) * 3, (slot < K) & present, 0)
    sy = tl.load(NativeShifts + (atom * K + slot) * 3 + 1, (slot < K) & present, 0)
    sz = tl.load(NativeShifts + (atom * K + slot) * 3 + 2, (slot < K) & present, 0)
    valid = present & (neighbor >= 0) & (neighbor < N)
    valid = valid & ((neighbor != atom) | (sx != 0) | (sy != 0) | (sz != 0))
    invalid = (count < 0) | (tl.sum((present & ~valid).to(tl.int32), 0) > 0)
    overflow = count > K
    if overflow:
        tl.atomic_or(Status, 1)
        tl.atomic_add(Status + 2, 1)
    if invalid:
        tl.atomic_or(Status, 2)
        tl.atomic_add(Status + 3, 1)
    tl.atomic_max(Status + 1, count)
    # Every slot is overwritten, including stale tails. The force kernels
    # recognize -1, whereas the library's ordinary empty sentinel is N.
    tl.store(Neighbors + atom * K + slot, tl.where(valid, neighbor, -1), slot < K)
    tl.store(Shifts + (atom * K + slot) * 3, tl.where(valid, sx, 0), slot < K)
    tl.store(Shifts + (atom * K + slot) * 3 + 1, tl.where(valid, sy, 0), slot < K)
    tl.store(Shifts + (atom * K + slot) * 3 + 2, tl.where(valid, sz, 0), slot < K)
    tl.store(Counts + atom, bounded_count)


class NativeMatrixUpdater:
    """Build directly into fixed matrix storage without host count checks.

    Setup performs allocation and cell sizing. ``rebuild_gpu`` is capturable;
    warm it before capture. ``status`` is int32[4]: error bits, true maximum
    degree, overflowing atom count, and invalid atom count. The first element
    MUST equal zero before a trajectory force or step can be accepted. Target
    counts are clamped only for memory safety, not to permit truncation.

    The supplied plan must belong to one FP32 periodic sample. The bound
    cell/PBC buffers are checked on device against copies taken at construction
    or at the last ``set_cell``. No reference-coordinate/skin bookkeeping is
    updated here; that belongs to the checked integrator, after this operation
    has succeeded. ``cell_headroom`` > 1 sizes the bin storage for a cell grown
    by that factor in every length (constant-pressure MD).
    """

    def __init__(self, plan, skin=1.5, cell_headroom=1.0):
        self.plan = plan
        self.n, self.capacity = int(plan.n), int(plan.capacity)
        self.skin = float(skin)
        self.cutoff = float(plan.model.cutoff) + self.skin
        if self.n <= 0 or self.capacity <= 0 or not math.isfinite(self.skin) or self.skin <= 0:
            raise ValueError('Positive atom count, matrix capacity, and finite skin required')
        if len(plan.batch['ptr']) != 2 or 'pbc' not in plan.batch:
            raise ValueError('Native matrix updating requires one fixed-cell physical sample')
        if plan.cell.shape != (3, 3) or plan.cell.dtype != torch.float32 or not plan.cell.is_cuda:
            raise ValueError('A CUDA FP32 3x3 force cell is required')
        self.device = plan.cell.device
        self.current_cell = plan.cell
        self.current_pbc = plan.batch['pbc']
        if self.current_pbc.numel() != 3 or self.current_pbc.dtype != torch.bool:
            raise ValueError('One three-axis bool PBC tensor is required')
        for tensor in (plan.neighbors, plan.shifts, plan.counts, self.current_cell, self.current_pbc):
            if tensor.device != self.device or not tensor.is_contiguous():
                raise ValueError('All publication/configuration buffers must be contiguous on one device')
        if (plan.neighbors.shape != (self.n, self.capacity) or plan.neighbors.dtype != torch.int64
                or plan.shifts.shape != (self.n, self.capacity, 3) or plan.shifts.dtype != torch.int32
                or plan.counts.shape != (self.n,) or plan.counts.dtype != torch.int32):
            raise ValueError('Unexpected prepared matrix layout')
        self.fixed_cell = self.current_cell.detach().clone().contiguous()
        self.fixed_pbc = self.current_pbc.detach().reshape(3).clone().contiguous()
        if not bool(torch.isfinite(self.fixed_cell).all()) or float(torch.linalg.det(self.fixed_cell)) <= 0:
            raise ValueError('The native cell-list backend requires a finite positive-volume cell')
        self.inverse_cell = torch.linalg.inv(self.fixed_cell).contiguous()
        if not (math.isfinite(cell_headroom) and cell_headroom >= 1):
            raise ValueError('cell_headroom must be a finite number >= 1')
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)),
                             sync_enter=False, sync_exit=False):
            self.total_cells, search_radius = estimate_cell_list_sizes(
                (self.fixed_cell * cell_headroom).unsqueeze(0), self.fixed_pbc, self.cutoff,
                max_nbins=max(8192, self.n))
        self.search_radius = [int(v) for v in search_radius.cpu().tolist()]
        self.pbc_axes = [bool(v) for v in self.fixed_pbc.cpu().tolist()]
        self.storage = allocate_cell_list(self.n, self.total_cells, search_radius, self.device)
        self.scan = FixedExclusiveScan(self.storage[4], self.storage[5])
        self.positions = torch.empty((self.n, 3), device=self.device, dtype=torch.float32)
        self.native_neighbors = torch.empty((self.n, self.capacity), device=self.device, dtype=torch.int32)
        self.native_shifts = torch.empty_like(plan.shifts)
        self.native_counts = torch.empty_like(plan.counts)
        self.status = torch.zeros(4, device=self.device, dtype=torch.int32)
        self.initial_ptrs = self.buffer_pointers()
        self.fractional_limit = float(2**28 / max(1, self.total_cells))

        # Cache all Warp array views. A fixed scratch scan replaces the core
        # launcher's allocating CUB scan; all storage is owned by this object.
        self._wp_position = wp.from_torch(self.positions, dtype=wp.vec3)
        self._wp_cell = wp.from_torch(self.fixed_cell.unsqueeze(0), dtype=wp.mat33)
        self._wp_pbc = wp.from_torch(self.fixed_pbc, dtype=wp.bool)
        types = (wp.int32, wp.int32, wp.vec3i, wp.vec3i, wp.int32, wp.int32, wp.int32)
        self._wp_storage = tuple(wp.from_torch(t, dtype=d) for t, d in zip(self.storage, types))
        self._wp_neighbors = wp.from_torch(self.native_neighbors, dtype=wp.int32)
        self._wp_shifts = wp.from_torch(self.native_shifts, dtype=wp.vec3i)
        self._wp_counts = wp.from_torch(self.native_counts, dtype=wp.int32)
        self._construct_bins()

    def _construct_bins(self):
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)),
                             sync_enter=False, sync_exit=False):
            wp.launch(_cell_list_construct_bin_size_overload[wp.float32], dim=1,
                inputs=[self._wp_cell, self._wp_pbc, self._wp_storage[0],
                        wp.float32(self.cutoff), self.total_cells], device=str(self.device))

    @torch.no_grad()
    def set_cell(self, cell):
        """Adopt a new cell in place (same buffers, so captured rebuilds see it); call between launches.

        The caller copies the same values into the plan's cell. The bin grid is recomputed; the
        neighbour search radius allocated at construction must still cover the cutoff.
        """
        cell = np.asarray(cell, dtype=np.float32).reshape(3, 3)
        if not np.isfinite(cell).all() or float(np.linalg.det(cell.astype(np.float64))) <= 0:
            raise ValueError('The native cell-list backend requires a finite positive-volume cell')
        inverse = np.linalg.inv(cell.astype(np.float64))
        for axis in range(3):
            face = 1.0 / float(np.linalg.norm(inverse[:, axis]))
            bins = max(int(face / self.cutoff), 1)
            needed = 0 if bins == 1 and not self.pbc_axes[axis] else math.ceil(self.cutoff * bins / face)
            if needed > self.search_radius[axis]:
                raise ValueError(f'The cell became too small for the neighbour search set up for it '
                                 f'(axis {axis}: face distance {face:.3f} A, cutoff plus skin {self.cutoff:.3f} A)')
        host = torch.from_numpy(np.stack([cell, inverse.astype(np.float32)]))
        self.fixed_cell.copy_(host[0])
        self.inverse_cell.copy_(host[1])
        self._construct_bins()

    def buffer_pointers(self):
        return tuple(t.data_ptr() for t in (self.plan.neighbors, self.plan.shifts,
                                            self.plan.counts, self.plan.cell))

    @torch.no_grad()
    def rebuild_gpu(self, positions):
        if (positions.shape != (self.n, 3) or positions.dtype != torch.float32
                or positions.device != self.device or not positions.is_contiguous()):
            raise ValueError('Supply the actual contiguous FP32 force coordinates on the bound device')
        if self.buffer_pointers() != self.initial_ptrs:
            raise ValueError('Prepared matrix storage changed; construct a new updater')
        _configuration_status[(1,)](self.current_cell, self.fixed_cell,
            self.current_pbc, self.fixed_pbc, self.status, num_warps=1)
        _prepare_positions[(tr.cdiv(self.n, 128),)](positions, self.positions, self.inverse_cell,
            self.status, self.n, self.fractional_limit, 128, num_warps=4)
        self.native_counts.zero_()
        self.storage[4].zero_()
        cpd, search, periodic, mapping, counts, offsets, atoms = self._wp_storage
        # PyTorch owns cross-stream dependencies. Warp's implicit entry sync
        # would create an uncaptured dependency when called in a child graph.
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)),
                             sync_enter=False, sync_exit=False):
            wp.launch(_cell_list_count_atoms_per_bin_overload[wp.float32], dim=self.n,
                inputs=[self._wp_position, self._wp_cell, self._wp_pbc, cpd, counts, periodic],
                device=str(self.device))
            self.scan()
            self.storage[4].zero_()
            wp.launch(_cell_list_bin_atoms_overload[wp.float32], dim=self.n,
                inputs=[self._wp_position, self._wp_cell, self._wp_pbc, cpd, mapping, counts, offsets, atoms],
                device=str(self.device))
            query_cell_list(self._wp_position, self._wp_cell, self._wp_pbc, self.cutoff,
                cpd, search, periodic, mapping, counts, offsets, atoms, self._wp_neighbors,
                self._wp_shifts, self._wp_counts, wp.float32, str(self.device), half_fill=False)
        _publish_matrix[(self.n,)](self.native_neighbors, self.native_shifts, self.native_counts,
            self.plan.neighbors, self.plan.shifts, self.plan.counts, self.status,
            self.n, self.capacity, tr.next_power_of_2(self.capacity), num_warps=4)
        return self.status

    def checked_status(self):
        """Explicit host synchronization for setup/tests, never an MD hot path."""
        error, degree, overflow_atoms, invalid_atoms = self.status.cpu().tolist()
        result = dict(error_bits=error, true_max_degree=degree,
                      overflow_atoms=overflow_atoms, invalid_atoms=invalid_atoms)
        if error:
            raise RuntimeError(f'Invalid native neighbor matrix: {result}')
        return result

    def description(self):
        return dict(backend='fixed_native_matrix', n_atoms=self.n, capacity=self.capacity,
                    cutoff_plus_skin_A=self.cutoff, skin_A=self.skin, total_cells=self.total_cells,
                    error_bits=dict(overflow=OVERFLOW, invalid_neighbor=INVALID_NEIGHBOR,
                                    cell_changed=CELL_CHANGED, pbc_changed=PBC_CHANGED,
                                    invalid_position=INVALID_POSITION),
                    status_columns=['error_bits', 'true_max_degree', 'overflow_atoms', 'invalid_atoms'],
                    overflow_policy='reject before force/commit; safe counts do not authorize truncation',
                    cell_policy='device comparison with the cell copied at construction or set_cell',
                    reference_coordinates='caller owns and updates only after status[0] == 0',
                    construction='direct native cell list to fixed matrix; no COO or host count checks',
                    scan='preallocated hierarchical Triton exclusive scan',
                    scan_levels=len(self.scan.levels))
