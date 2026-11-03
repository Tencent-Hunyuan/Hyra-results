"""Force-local stable active row views; the shared skin graph is never modified.

The CPU path is an unfused FP32 reference; it does not show CUDA/FMA equivalence.
CUDA dispatch uses the base-message owned-geometry helper with its FMA setting.
All result buffers are fresh; autograd operators retain them through backward.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import torch


class ActiveOwnedGraphError(ValueError):
    pass


def _require(condition, code):
    if not condition:
        raise ActiveOwnedGraphError(code)


def _stamp(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape),
            tuple(value.stride()), value.dtype, value.device, value.requires_grad)


def _buffers(graph):
    return (graph.local_rows, graph.neighbors, graph.shifts, graph.counts, graph.cell)


def _graph_stamp(graph):
    return (id(graph), id(graph.metadata), graph.rank, graph.capacity,
            graph.n_global, graph.n_local, tuple(_stamp(v) for v in _buffers(graph)))


@dataclass(frozen=True)
class ActiveOwnedGraphSelection:
    graph: object
    skin_graph: object
    positions: torch.Tensor
    cutoff: float
    fma: bool
    generation: int
    status: torch.Tensor
    position_stamp: tuple
    skin_stamp: tuple
    graph_stamp: tuple
    status_stamp: tuple

    def check(self, *, positions=None):
        _require((positions is None or positions is self.positions) and
                 _stamp(self.positions) == self.position_stamp,
                 "ACTIVE_BASE_PAIR_FRAME_CHANGED")
        _require(_graph_stamp(self.skin_graph) == self.skin_stamp,
                 "ACTIVE_BASE_PAIR_SKIN_CHANGED")
        _require(_graph_stamp(self.graph) == self.graph_stamp and
                 _stamp(self.status) == self.status_stamp,
                 "ACTIVE_BASE_PAIR_SELECTION_CHANGED")
        return self


def _metadata(graph, positions, cutoff, fma, generation):
    n, k = graph.n_local, graph.capacity
    device = graph.cell.device
    _require(type(cutoff) in (int, float) and math.isfinite(cutoff) and cutoff > 0 and
             type(fma) is bool and type(generation) is int and generation >= 0,
             "ACTIVE_BASE_PAIR_POLICY_INVALID")
    _require(type(k) is int and k > 0 and graph.n_global > 0 and n >= 0,
             "ACTIVE_BASE_PAIR_GRAPH_SHAPE_INVALID")
    expected = ((graph.local_rows, (n,), torch.int64),
                (graph.neighbors, (n, k), torch.int64),
                (graph.shifts, (n, k, 3), torch.int32),
                (graph.counts, (n,), torch.int32),
                (graph.cell, (3, 3), torch.float32),
                (positions, (graph.n_global, 3), torch.float32))
    _require(all(isinstance(v, torch.Tensor) and tuple(v.shape) == shape and
                 v.dtype == dtype and v.device == device and v.is_contiguous()
                 for v, shape, dtype in expected), "ACTIVE_BASE_PAIR_TENSOR_METADATA_INVALID")
    _require(all(not value.requires_grad for value in _buffers(graph)),
             "ACTIVE_BASE_PAIR_FROZEN_SKIN_REQUIRED")
    _require(device.type in ("cpu", "cuda"), "ACTIVE_BASE_PAIR_DEVICE_UNSUPPORTED")


@torch.no_grad()
def _select_cpu(graph, positions, cutoff):
    """Stable CPU reference; every arithmetic operation rounds in FP32 without FMA."""
    n, k = graph.n_local, graph.capacity
    slot = torch.arange(k, device=positions.device).expand(n, k)
    prefix = slot < graph.counts[:, None]
    valid = (graph.neighbors >= 0) & (graph.neighbors < graph.n_global)
    source = graph.neighbors.clamp(0, graph.n_global - 1)
    delta = positions[source] - positions[graph.local_rows, None, :]
    # Match the expression order in _owned_geometry; no matmul/TF32 path.
    shifts = graph.shifts.to(torch.float32)
    for axis in range(3):
        delta = delta + shifts[:, :, axis, None] * graph.cell[axis]
    distance = torch.sqrt((delta[:, :, 0] * delta[:, :, 0] +
                           delta[:, :, 1] * delta[:, :, 1]) +
                          delta[:, :, 2] * delta[:, :, 2])
    malformed = ((graph.counts < 0) | (graph.counts > k)).any()
    malformed = malformed | (prefix != valid).any()
    malformed = malformed | (prefix & ~torch.isfinite(distance)).any()
    keep = prefix & valid & (distance < cutoff)
    ranks = keep.to(torch.int64).cumsum(1)
    counts = keep.sum(1, dtype=torch.int32)
    destinations = torch.where(keep, ranks - 1, counts[:, None] + slot - ranks)
    neighbors = torch.full_like(graph.neighbors, -1)
    neighbors.scatter_(1, destinations, torch.where(keep, graph.neighbors, -1))
    shifts_out = torch.zeros_like(graph.shifts)
    shifts_out.scatter_(1, destinations[:, :, None].expand(n, k, 3),
                        torch.where(keep[:, :, None], graph.shifts, 0))
    return neighbors, shifts_out, counts, malformed.to(torch.int64).reshape(1)


def select_active_owned_graph(graph, positions, *, cutoff, fma, generation=0):
    """Return a fresh immutable-by-contract graph; device status is checked by caller.

    Rows are owned destination atoms; neighbor IDs index global source atoms.
    Image displacement is P[source]-P[destination]+integer_shift @ cell.
    K, ownership, cell and the skin buffers are kept, never rewritten or truncated.
    CPU callers must request fma=False because the CPU reference has no fused
    multiply-add.
    """
    _metadata(graph, positions, cutoff, fma, generation)
    position_stamp, skin_stamp = _stamp(positions), _graph_stamp(graph)
    if positions.device.type == "cpu":
        _require(not fma, "ACTIVE_BASE_PAIR_CPU_FMA_UNSUPPORTED")
        buffers = _select_cpu(graph, positions, float(cutoff))
    else:
        from hyff.multigpu.active_base_pair_gpu import select_cuda
        buffers = select_cuda(graph, positions, cutoff=float(cutoff), fma=fma)
    neighbors, shifts, counts, status = buffers
    active = replace(graph, neighbors=neighbors, shifts=shifts, counts=counts)
    result = ActiveOwnedGraphSelection(active, graph, positions, float(cutoff), fma,
        generation, status, position_stamp, skin_stamp, _graph_stamp(active), _stamp(status))
    return result.check(positions=positions)


@dataclass(frozen=True)
class ActiveBasePairForce:
    layers: tuple
    pair_operator: object
    pair: object
    selections: tuple
    status: torch.Tensor
    numeric_status: torch.Tensor
    generation: int

    def check(self, positions):
        for value in self.selections:
            value.check(positions=positions)
        return self


def refresh_active_base_pair(layers, pair_operator, pair, positions, *, generation):
    """One timing boundary: allocation, selection, status and operator rebinding.

    Reuse is restricted to this call's dictionary. Every call allocates fresh
    result tensors, including when the same position object is evaluated again.
    The active-edge compaction control reads .status before any base/pair block executes.
    """
    cache = {}

    def operator(value):
        fma = value.settings[3] if hasattr(value, "settings") else value.fma
        key = (id(value.graph), float(value.cutoff), fma)
        if key not in cache:
            cache[key] = select_active_owned_graph(value.graph, positions,
                cutoff=value.cutoff, fma=fma, generation=generation)
        return replace(value, graph=cache[key].graph)

    refreshed = []
    for layer in layers:
        base = operator(layer.inputs.base_operator)
        refreshed.append(replace(layer, inputs=replace(layer.inputs, base_operator=base),
                                 base=replace(layer.base, operator=base)))
    pair_operator = operator(pair_operator)
    status = torch.stack(tuple(value.status for value in cache.values())).amax(0)
    return ActiveBasePairForce(tuple(refreshed), pair_operator,
        replace(pair, operator=pair_operator), tuple(cache.values()), status,
        torch.zeros(1, dtype=torch.int64, device=positions.device), generation).check(positions)


@torch.no_grad()
def guard_active_base_pair_numeric(status, values=(), *, cotangents=(), force=None):
    """Device-only nonfinite guard and one timing boundary for ALL guard work.

    Projected local source fields are checked on their owner. Tensor hooks
    check incoming local aggregate cotangents before the all-gather. The
    final collective output check rejects a flagged rank; no
    per-block host read or collective is added. Hooks return their input
    gradient by identity, and the successful final force is value-preserved.

    This covers explicit NaN/Inf operands, not universal finite-overflow IEEE
    equivalence of removed arithmetic. Frozen radial coefficients are already
    finite-checked at preparation and version-guarded by PreparedOwnedModel.
    """
    _require(isinstance(status, torch.Tensor) and status.shape == (1,) and
             status.dtype == torch.int64 and not status.requires_grad,
             "ACTIVE_BASE_PAIR_NUMERIC_STATUS_INVALID")
    for value in values:
        if value is not None:
            _require(isinstance(value, torch.Tensor) and value.device == status.device and
                     value.is_floating_point(), "ACTIVE_BASE_PAIR_NUMERIC_VALUE_INVALID")
            status.bitwise_or_((~torch.isfinite(value).all()).to(torch.int64))

    def observe(gradient):
        # The module-level lookup is intentional: a timing wrapper also sees
        # backward-time checks, outside the explicit base/pair VJP kernels.
        guard_active_base_pair_numeric(status, (gradient,))
        return gradient

    for value in cotangents:
        _require(isinstance(value, torch.Tensor) and value.device == status.device and
                 value.requires_grad, "ACTIVE_BASE_PAIR_COTANGENT_HOOK_INVALID")
        value.register_hook(observe)
    if force is None:
        return None
    _require(isinstance(force, torch.Tensor) and force.device == status.device and
             force.is_floating_point(), "ACTIVE_BASE_PAIR_NUMERIC_FORCE_INVALID")
    # Apply only after backward. The NONFINITE_OUTPUT check (step.py) and
    # its MAX agreement then reject every rank before accepting this force.
    return torch.where(status.reshape(()) == 0, force, torch.full_like(force, float("nan")))


__all__ = ["ActiveOwnedGraphError", "ActiveOwnedGraphSelection", "ActiveBasePairForce",
           "select_active_owned_graph", "refresh_active_base_pair", "guard_active_base_pair_numeric"]
