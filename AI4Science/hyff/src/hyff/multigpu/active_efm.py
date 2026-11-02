"""Per-force active EFM views inside an immutable owned skin epoch.

Each selection has its own fixed allocation. The caller collectively checks
overflow, grows the selector's future allocation if needed, then reselects and
agrees before finalizing a logical whole-tile view or executing any block. A
selection owns its tensors through backward; no prepared topology or existing
selection buffer is rewritten. This path is deliberately outside CUDA capture.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import weakref

import torch

from hyff.multigpu.projected_efm import _require


def _stamp(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape),
            tuple(value.stride()), value.dtype, value.device, value.requires_grad)


def _gather(values, indices):
    sentinel = values.new_zeros((1, *values.shape[1:]))
    return torch.cat((values, sentinel), dim=0).index_select(0, indices)


@dataclass(frozen=True)
class ActiveEFMSelection:
    selector: object
    positions: torch.Tensor
    position_stamp: tuple
    indices: torch.Tensor
    source: torch.Tensor
    destination: torch.Tensor
    global_destination: torch.Tensor
    translation: torch.Tensor
    valid: torch.Tensor
    count: torch.Tensor
    overflow: torch.Tensor
    allocation_capacity: int
    capacity_generation: int
    logical_count: int | None = None
    buffer_stamps: tuple = ()
    scalar_stamp: tuple = ()

    def buffers(self):
        return (self.indices, self.source, self.destination, self.global_destination,
                self.translation, self.valid, self.count, self.overflow)

    def check(self, *, operator=None, positions=None, finalized=True):
        self.selector.assert_current()
        _require(_stamp(self.positions) == self.position_stamp and
                 (positions is None or positions is self.positions),
                 "EFM_ACTIVE_FRAME_CHANGED")
        _require(tuple(_stamp(value) for value in self.buffers()) == self.buffer_stamps,
                 "EFM_ACTIVE_SELECTION_CHANGED")
        _require(self.scalar_stamp == (id(self.selector), self.position_stamp,
                 self.allocation_capacity, self.capacity_generation, self.logical_count),
                 "EFM_ACTIVE_SELECTION_CHANGED")
        # A previous force may still own a live autograd tape after a later
        # force grows. Its private allocation remains valid, without acquiring
        # the larger capacity granted to newly selected tensors.
        _require(type(self.allocation_capacity) is int and
                 type(self.capacity_generation) is int and
                 0 <= self.capacity_generation <= self.selector.capacity_generation and
                 0 < self.allocation_capacity <= self.selector.capacity and
                 self.allocation_capacity % self.selector.tile_edges == 0,
                 "EFM_ACTIVE_SELECTION_CAPACITY_CHANGED")
        if self.logical_count is None:
            rows = self.allocation_capacity
        else:
            _require(type(self.logical_count) is int and
                     0 <= self.logical_count <= self.allocation_capacity,
                     "EFM_ACTIVE_SELECTION_ROWS_INVALID")
            rows = math.ceil(self.logical_count / self.selector.tile_edges) * self.selector.tile_edges
        _require(all(value.shape[0] == rows for value in
                 (self.indices, self.source, self.destination,
                  self.global_destination, self.translation, self.valid)),
                 "EFM_ACTIVE_SELECTION_ROWS_INVALID")
        if operator is not None:
            _require(getattr(operator, "_active_selector", None) is self.selector and
                     self.selector.accepts(operator), "EFM_ACTIVE_SELECTOR_MISMATCH")
        _require(not finalized or self.logical_count is not None,
                 "EFM_ACTIVE_CONTROL_NOT_FINALIZED")


def _seal(selection):
    return replace(selection, buffer_stamps=tuple(_stamp(value) for value in selection.buffers()),
                   scalar_stamp=(id(selection.selector), selection.position_stamp,
                       selection.allocation_capacity, selection.capacity_generation, selection.logical_count))


class PreparedActiveEFMSelector:
    """Immutable epoch topology plus per-call allocation/count telemetry."""

    def __init__(self, operator, positions, cell, radius, reserve_factor):
        operator._assert_current()
        meta = operator.metadata
        _require(type(radius) in (int, float) and math.isfinite(radius) and radius > 0 and
                 type(reserve_factor) in (int, float) and math.isfinite(reserve_factor) and
                 reserve_factor >= 1.0, "EFM_ACTIVE_OPTIONS_INVALID")
        _require(isinstance(cell, torch.Tensor) and cell.shape == (3, 3) and
                 cell.dtype == meta.dtype and cell.device == meta.device and not cell.requires_grad,
                 "EFM_ACTIVE_CELL_INVALID")
        self.metadata = meta
        self.radius = float(radius)
        self.reserve_factor = float(reserve_factor)
        self.tile_edges = meta.tile_edges
        self._metadata_identity = id(meta)
        self._metadata_stamps = tuple(_stamp(value) for value in meta.buffers())
        # Operators own the selector. Weak membership avoids retaining an old
        # GPU epoch in a reference cycle after neighbor replacement.
        self._operators = weakref.WeakSet()
        self.selection_calls = 0
        self.last_active_count = None
        self.last_logical_rows = None
        with torch.no_grad():
            shifts = torch.where(meta.valid[:, None], meta.unit_shifts, torch.zeros_like(meta.unit_shifts))
            self.translation = (shifts.to(meta.dtype) @ cell).detach().clone()
            self.padding = positions.new_tensor((1.0, 0.0, 0.0))
            self._geometry_stamps = (_stamp(self.translation), _stamp(self.padding))
            self.initial_active_count = int(self._mask(positions).sum())
        self.capacity = max(self.tile_edges,
            math.ceil(self.initial_active_count * self.reserve_factor / self.tile_edges) * self.tile_edges)
        self.initial_capacity = self.capacity
        self.capacity_generation = 0
        self._policy = (self.radius, self.reserve_factor, self.tile_edges, self.initial_capacity,
                        self.capacity, self.capacity_generation)
        self.bind(operator)

    def assert_current(self):
        _require(id(self.metadata) == self._metadata_identity and
                 (self.radius, self.reserve_factor, self.tile_edges, self.initial_capacity,
                  self.capacity, self.capacity_generation) == self._policy and
                 tuple(_stamp(value) for value in self.metadata.buffers()) == self._metadata_stamps and
                 (_stamp(self.translation), _stamp(self.padding)) == self._geometry_stamps,
                 "EFM_ACTIVE_EPOCH_CHANGED")

    def reserve_at_least(self, count):
        """Grow future selections only; caller must reselect and agree globally.

        This changes host capacity policy, never a topology tensor, operator,
        compiled factory or existing selection. The count is rank-local and
        comes from the caller's observed control vector for this skin epoch.
        """
        self.assert_current()
        _require(type(count) is int and 0 <= count <= self.metadata.valid_edges,
                 "EFM_ACTIVE_CAPACITY_COUNT_INVALID")
        if count <= self.capacity:
            return False
        reserved = count * self.reserve_factor
        _require(math.isfinite(reserved), "EFM_ACTIVE_CAPACITY_GROWTH_INVALID")
        capacity = math.ceil(reserved / self.tile_edges) * self.tile_edges
        _require(capacity >= count and capacity > self.capacity,
                 "EFM_ACTIVE_CAPACITY_GROWTH_INVALID")
        generation = self.capacity_generation + 1
        policy = (self.radius, self.reserve_factor, self.tile_edges, self.initial_capacity,
                  capacity, generation)
        self.capacity, self.capacity_generation, self._policy = capacity, generation, policy
        return True

    def bind(self, operator):
        """Preparation-only equality check for shared-radius block metadata."""
        self.assert_current()
        operator._assert_current()
        meta, reference = operator.metadata, self.metadata
        scalar = lambda item: (item.n_global, item.n_owned, item.tile_edges, item.epoch,
                               item.generation, item.dtype, item.device, item.edge_capacity)
        _require(scalar(meta) == scalar(reference) and all(torch.equal(a, b) for a, b in zip(
            (meta.source, meta.destination, meta.valid, meta.unit_shifts, meta.owned_global_indices),
            (reference.source, reference.destination, reference.valid,
             reference.unit_shifts, reference.owned_global_indices))),
            "EFM_ACTIVE_SHARED_TOPOLOGY_MISMATCH")
        current = getattr(operator, "_active_selector", None)
        _require(current is None or current is self, "EFM_ACTIVE_SELECTOR_ALREADY_BOUND")
        operator._active_selector = self
        self._operators.add(operator)

    def accepts(self, operator):
        return operator in self._operators

    def _mask(self, positions):
        meta = self.metadata
        _require(isinstance(positions, torch.Tensor) and positions.shape == (meta.n_global, 3) and
                 positions.dtype == meta.dtype and positions.device == meta.device,
                 "EFM_ACTIVE_POSITION_METADATA_MISMATCH")
        readable = positions if meta.n_global else positions.new_zeros((1, 3))
        displacement = (readable.index_select(0, meta.safe_source) -
                        readable.index_select(0, meta.global_destination) + self.translation)
        displacement = torch.where(meta.valid[:, None], displacement, self.padding)
        distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-9)
        return meta.valid & (distance < self.radius)

    def select(self, positions):
        self.assert_current()
        meta = self.metadata
        with torch.no_grad():
            mask = self._mask(positions)
            size = mask.numel()
            prefix = mask.to(torch.int64).cumsum(0)
            count = mask.sum(dtype=torch.int64)
            original = torch.arange(size, dtype=torch.int64, device=mask.device)
            destinations = torch.where(mask, prefix - 1, count + original - prefix)
            order = torch.full((max(size, self.capacity),), size,
                               dtype=torch.int64, device=mask.device)
            order.scatter_(0, destinations, original)
            valid = torch.arange(self.capacity, device=mask.device) < count
            indices = torch.where(valid, order[:self.capacity], size)
            selection = ActiveEFMSelection(self, positions, _stamp(positions), indices,
                _gather(meta.safe_source, indices), _gather(meta.destination, indices),
                _gather(meta.global_destination, indices), _gather(self.translation, indices),
                valid, count, count > self.capacity, self.capacity, self.capacity_generation)
        self.selection_calls += 1
        return _seal(selection)

    def public_metadata(self):
        return {"schema": "OWNED_ACTIVE_EFM_SELECTOR", "radius_A": self.radius,
                "epoch": self.metadata.epoch, "generation": self.metadata.generation,
                "skin_edge_capacity": self.metadata.edge_capacity, "reserved_capacity": self.capacity,
                "initial_reserved_capacity": self.initial_capacity,
                "capacity_generation": self.capacity_generation,
                "capacity_growth_count": self.capacity_generation,
                "initial_active_count": self.initial_active_count, "tile_edges": self.tile_edges,
                "reserve_factor": self.reserve_factor, "selection_calls": self.selection_calls,
                "last_active_count": self.last_active_count, "last_logical_rows": self.last_logical_rows,
                "predicate": "SERVICE_FP32_DISTANCE_STRICT_PHYSICAL_CUTOFF",
                "logical_rows_policy": "CEIL_ACTIVE_COUNT_TO_TILE_AFTER_COLLECTIVE_CONTROL_SYNC",
                "overflow_policy": "COLLECTIVE_GROW_RESELECT_BEFORE_BLOCKS_BOUNDED_RETRY",
                "growth_preserves_existing_selection_tensors": True,
                "per_force_host_control_sync": True,
                "cuda_capture_supported": False}


def prepare_owned_active_efm_selector(operator, positions, cell, *, radius, reserve_factor=1.05):
    return PreparedActiveEFMSelector(operator, positions, cell, radius, reserve_factor)


def active_selection_control(cache):
    """Device [overflow, local counts...] vector; caller reduces only [:1]."""
    _require(isinstance(cache, dict) and bool(cache), "EFM_ACTIVE_SELECTION_CACHE_EMPTY")
    selections = tuple(cache.values())
    for selection in selections:
        selection.check(finalized=False)
    overflow = torch.stack(tuple(value.overflow.to(torch.int64) for value in selections)).amax()
    return torch.stack((overflow, *(value.count for value in selections)))


def finalize_active_selections(cache, counts):
    """Finalize after the caller's all-rank overflow check and count read."""
    _require(isinstance(counts, (tuple, list)) and len(counts) == len(cache),
             "EFM_ACTIVE_CONTROL_COUNTS_INVALID")
    for (key, selection), count in zip(tuple(cache.items()), counts):
        selection.check(finalized=False)
        selector = selection.selector
        _require(type(count) is int and 0 <= count <= selection.allocation_capacity,
                 "EFM_ACTIVE_CAPACITY_OVERFLOW")
        rows = math.ceil(count / selector.tile_edges) * selector.tile_edges
        _require(rows <= selection.allocation_capacity and all(value.shape[0] >= rows for value in
                 (selection.indices, selection.source, selection.destination,
                  selection.global_destination, selection.translation, selection.valid)),
                 "EFM_ACTIVE_CAPACITY_OVERFLOW")
        finalized = replace(selection, indices=selection.indices[:rows], source=selection.source[:rows],
            destination=selection.destination[:rows], global_destination=selection.global_destination[:rows],
            translation=selection.translation[:rows], valid=selection.valid[:rows], logical_count=count)
        cache[key] = _seal(finalized)
        selector.last_active_count, selector.last_logical_rows = count, rows


__all__ = ["ActiveEFMSelection", "PreparedActiveEFMSelector", "prepare_owned_active_efm_selector",
           "active_selection_control", "finalize_active_selections"]
