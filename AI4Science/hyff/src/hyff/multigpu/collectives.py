"""Prepared collectives over atom ownership for the owned-row multi-GPU model.

Control plane: construct immutable CPU metadata, prepare device index maps,
validate rank agreement, and prepare each typed exchange outside capture.
Tensor plane: padded owner-major AG, NCCL SUM-RS (gloo SUM-AR fallback), exact
global softmax/charge VJPs. No timed tensor path reads a tensor value on the host.

Only first derivatives for frozen-parameter inference are supported. Process
groups are created by the caller, and these functions do not order collectives
across independent autograd branches (ordered_collectives does). NCCL
collectives under CUDA graph capture are not tested.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch
import torch.distributed as dist


PARTIAL_SOURCE_SUM = "partial_source_sum_transpose"
DETACHED_SOURCE = "detached_source_cache"
DISJOINT_OUTPUT = "disjoint_output_adjoints_no_sum"
GLOBAL_SOFTMAX = "global_two_channel_softmax"
GLOBAL_CHARGE = "global_charge_projection"
_CONTRACTS = (PARTIAL_SOURCE_SUM, DETACHED_SOURCE, DISJOINT_OUTPUT, GLOBAL_SOFTMAX, GLOBAL_CHARGE)


class CollectiveContractError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise CollectiveContractError(code)


def _integers(values):
    if isinstance(values, torch.Tensor):
        _require(values.device.type == "cpu" and values.ndim == 1 and values.dtype in
                 (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
                 "CPU_INTEGER_METADATA_REQUIRED")
        values = values.tolist()
    values = tuple(values)
    _require(all(type(value) is int for value in values), "INTEGER_METADATA_REQUIRED")
    return values


@dataclass(frozen=True)
class OwnershipMetadata:
    global_ids: tuple[int, ...]
    owners: tuple[int, ...]
    indices_by_rank: tuple[tuple[int, ...], ...]
    counts: tuple[int, ...]
    canonical_slots: tuple[int, ...]
    world_size: int
    padded_rows_per_rank: int
    edge_counts: tuple[int, ...] | None = None
    edge_loads: tuple[int, ...] | None = None

    @property
    def n_global(self):
        return len(self.global_ids)


def make_ownership_metadata(global_ids, owners, *, world_size, edge_counts=None):
    """CPU-only control work; IDs/owners/edge counts enter in the same order."""
    _require(type(world_size) is int and world_size > 0, "POSITIVE_WORLD_SIZE_REQUIRED")
    ids, owner = _integers(global_ids), _integers(owners)
    _require(len(ids) == len(owner) and len(set(ids)) == len(ids), "UNIQUE_IDS_AND_OWNERS_REQUIRED")
    _require(all(0 <= value < world_size for value in owner), "OWNER_OUTSIDE_WORLD")
    edges = None if edge_counts is None else _integers(edge_counts)
    _require(edges is None or (len(edges) == len(ids) and all(value >= 0 for value in edges)),
             "NONNEGATIVE_EDGE_COUNTS_REQUIRED")
    order = sorted(range(len(ids)), key=ids.__getitem__)
    ids, owner = tuple(ids[i] for i in order), tuple(owner[i] for i in order)
    edges = None if edges is None else tuple(edges[i] for i in order)
    indices = tuple(tuple(i for i, value in enumerate(owner) if value == rank)
                    for rank in range(world_size))
    counts = tuple(len(rows) for rows in indices)
    maximum = max(1, max(counts))
    slots = [0] * len(ids)
    for rank, rows in enumerate(indices):
        for local, global_row in enumerate(rows):
            slots[global_row] = rank * maximum + local
    loads = None if edges is None else tuple(sum(edges[i] for i in rows) for rows in indices)
    return OwnershipMetadata(ids, owner, indices, counts, tuple(slots), world_size, maximum, edges, loads)


def edge_balanced_metadata(global_ids, edge_counts, *, world_size):
    """Outside-runtime greedy edge balancing; stable IDs break equal-load ties."""
    _require(type(world_size) is int and world_size > 0, "POSITIVE_WORLD_SIZE_REQUIRED")
    ids, edges = _integers(global_ids), _integers(edge_counts)
    _require(len(ids) == len(edges) and len(set(ids)) == len(ids) and all(v >= 0 for v in edges),
             "VALID_EDGE_BALANCE_INPUT_REQUIRED")
    loads, counts, owner = [0] * world_size, [0] * world_size, [0] * len(ids)
    for i in sorted(range(len(ids)), key=lambda i: (-edges[i], ids[i])):
        rank = min(range(world_size), key=lambda r: (loads[r], counts[r], r))
        owner[i] = rank
        loads[rank] += edges[i]
        counts[rank] += 1
    return make_ownership_metadata(ids, owner, world_size=world_size, edge_counts=edges)


def ownership_diagnostics(metadata):
    mean = metadata.n_global / metadata.world_size
    result = {
        "global_atoms": metadata.n_global, "world_size": metadata.world_size,
        "atom_counts": list(metadata.counts), "padded_rows_per_rank": metadata.padded_rows_per_rank,
        "padded_global_rows": metadata.world_size * metadata.padded_rows_per_rank,
        "max_mean_atom_count_ratio": max(metadata.counts) / mean if mean else 1.0,
    }
    if metadata.edge_loads is not None:
        mean_edges = sum(metadata.edge_loads) / metadata.world_size
        result.update(edge_loads=list(metadata.edge_loads), mean_edge_load=mean_edges,
                      max_edge_load=max(metadata.edge_loads),
                      max_mean_edge_load_ratio=max(metadata.edge_loads) / mean_edges if mean_edges else 1.0)
    return result


@dataclass(frozen=True)
class CollectiveRuntime:
    metadata: OwnershipMetadata
    rank: int
    backend: str
    group: Any
    device: torch.device
    local_indices: torch.Tensor
    global_to_local: torch.Tensor
    canonical_slots: torch.Tensor

    @property
    def n_local(self):
        return self.metadata.counts[self.rank]

    @property
    def n_global(self):
        return self.metadata.n_global

    @property
    def world_size(self):
        return self.metadata.world_size


def _outside_capture(device):
    if device.type == "cuda":
        _require(not torch.cuda.is_current_stream_capturing(), "PREPARATION_OUTSIDE_CAPTURE_REQUIRED")


def _agreement(runtime, payload):
    """Control-plane collective and host comparison; never call during timing."""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).digest()
    fingerprint = [int.from_bytes(digest[i:i + 8], "little", signed=True) for i in range(0, 32, 8)]
    signature = torch.tensor(fingerprint, dtype=torch.int64, device=runtime.device)
    if runtime.world_size > 1:
        copies = [torch.empty_like(signature) for _ in range(runtime.world_size)]
        dist.all_gather(copies, signature, group=runtime.group)
        _require(all(value.cpu().tolist() == fingerprint for value in copies), "RANK_CONTROL_SIGNATURE_MISMATCH")


def prepare_runtime(metadata, *, device, group=None):
    """Bind an existing group and immutable maps outside capture; no group init.

    CUDA callers must first select the intended local device themselves. All
    group members must invoke preparations in the same order. A plan is invalid
    after group destruction or ownership changes and must be rebuilt then.
    """
    _require(isinstance(metadata, OwnershipMetadata), "OWNERSHIP_METADATA_REQUIRED")
    rebuilt = make_ownership_metadata(metadata.global_ids, metadata.owners,
                                      world_size=metadata.world_size, edge_counts=metadata.edge_counts)
    _require(rebuilt == metadata, "OWNERSHIP_METADATA_INCONSISTENT")
    device = torch.device(device)
    _require(device.type in ("cpu", "cuda"), "CPU_OR_CUDA_DEVICE_REQUIRED")
    if device.type == "cpu":
        device = torch.device("cpu")
    if device.type == "cuda":
        _require(device.index is not None and device.index == torch.cuda.current_device(),
                 "EXPLICIT_CURRENT_CUDA_DEVICE_REQUIRED")
    _outside_capture(device)
    if dist.is_available() and dist.is_initialized():
        rank, size, backend = dist.get_rank(group), dist.get_world_size(group), str(dist.get_backend(group))
        _require((backend == "nccl" and device.type == "cuda") or
                 (backend == "gloo" and device.type == "cpu"), "BACKEND_DEVICE_MISMATCH")
    else:
        _require(group is None, "INITIALIZED_GROUP_REQUIRED")
        rank, size, backend = 0, 1, "single"
    _require(size == metadata.world_size, "OWNERSHIP_WORLD_MISMATCH")
    local = torch.tensor(metadata.indices_by_rank[rank], dtype=torch.int64, device=device)
    inverse = torch.full((metadata.n_global,), -1, dtype=torch.int64, device=device)
    inverse.index_copy_(0, local, torch.arange(local.numel(), dtype=torch.int64, device=device))
    slots = torch.tensor(metadata.canonical_slots, dtype=torch.int64, device=device)
    runtime = CollectiveRuntime(metadata, rank, backend, group, device, local, inverse, slots)
    _agreement(runtime, {"ids": metadata.global_ids, "owners": metadata.owners,
                         "world_size": size, "edges": metadata.edge_counts})
    return runtime


@dataclass(frozen=True)
class ExchangeSpec:
    runtime: CollectiveRuntime
    tail_shape: tuple[int, ...]
    dtype: torch.dtype
    contract: str
    fixed_total_charge: torch.Tensor | None = None


def prepare_exchange(runtime, tail_shape, dtype, contract, *, total_charge=None):
    """Validate payload agreement and fixed physical input outside capture.

    Use separate specs for separate graph/field exchanges. No combined base/EFM
    packet is implied. All ranks must prepare exactly the same ordered specs.
    """
    _require(isinstance(runtime, CollectiveRuntime), "PREPARED_RUNTIME_REQUIRED")
    _outside_capture(runtime.device)
    tail = tuple(tail_shape)
    _require(all(type(value) is int and value > 0 for value in tail), "POSITIVE_TRAILING_DIMENSIONS_REQUIRED")
    _require(dtype in (torch.float32, torch.float64), "FP32_OR_FP64_PAYLOAD_REQUIRED")
    _require(contract in _CONTRACTS, "EXPLICIT_ADJOINT_CONTRACT_REQUIRED")
    if contract == GLOBAL_SOFTMAX:
        _require(tail == (2,), "TWO_CHANNEL_SOFTMAX_REQUIRED")
    if contract == GLOBAL_CHARGE:
        _require(tail in ((), (1,)) and type(total_charge) in (int, float) and math.isfinite(total_charge),
                 "FIXED_SCALAR_TOTAL_CHARGE_REQUIRED")
        _require(abs(total_charge) <= torch.finfo(dtype).max, "TOTAL_CHARGE_OUTSIDE_DTYPE_RANGE")
        _require(runtime.n_global > 0 or total_charge == 0, "EMPTY_SYSTEM_NONZERO_CHARGE")
    else:
        _require(total_charge is None, "UNEXPECTED_TOTAL_CHARGE")
    _agreement(runtime, {"tail_shape": tail, "dtype": str(dtype), "contract": contract,
                         "total_charge": total_charge})
    target = (torch.tensor(float(total_charge), dtype=dtype, device=runtime.device)
              if contract == GLOBAL_CHARGE else None)
    return ExchangeSpec(runtime, tail, dtype, contract, target)


def _check_spec(spec, contract):
    _require(isinstance(spec, ExchangeSpec) and spec.contract == contract, "EXCHANGE_CONTRACT_MISMATCH")


def _check_tensor(value, spec, *, global_rows=False, differentiable=False):
    # All predicates inspect tensor metadata only. Finite-value/physical guards
    # belong to the caller's eager preparation or a separate device-side check.
    count = spec.runtime.n_global if global_rows else spec.runtime.n_local
    _require(isinstance(value, torch.Tensor) and value.device == spec.runtime.device and
             value.dtype == spec.dtype and tuple(value.shape) == (count, *spec.tail_shape),
             "PAYLOAD_METADATA_MISMATCH")
    if differentiable:
        _require(value.requires_grad, "OWNED_GRADIENT_DEPENDENCY_REQUIRED")


def _zeros(spec, count):
    return torch.zeros((count, *spec.tail_shape), dtype=spec.dtype, device=spec.runtime.device)


def _empty(spec, count):
    return torch.empty((count, *spec.tail_shape), dtype=spec.dtype, device=spec.runtime.device)


def _gather_tensor(owned, spec):
    runtime = spec.runtime
    maximum = runtime.metadata.padded_rows_per_rank
    send = _zeros(spec, maximum)
    send[:runtime.n_local].copy_(owned)
    if runtime.world_size == 1:
        rank_major = send
    else:
        rank_major = _empty(spec, runtime.world_size * maximum)
        if runtime.backend == "nccl":
            dist.all_gather_into_tensor(rank_major, send, group=runtime.group)
        else:
            # gloo is the CPU reference path; plain all_gather avoids relying on
            # gloo support for all_gather_into_tensor.
            _require(runtime.backend == "gloo", "GATHER_BACKEND_UNSUPPORTED")
            dist.all_gather(list(rank_major.split(maximum, dim=0)), send, group=runtime.group)
    return rank_major.index_select(0, runtime.canonical_slots)


def _sum_scatter_tensor(partial_global, spec):
    runtime = spec.runtime
    maximum = runtime.metadata.padded_rows_per_rank
    rank_major = _zeros(spec, runtime.world_size * maximum)
    rank_major.index_copy_(0, runtime.canonical_slots, partial_global)
    if runtime.world_size == 1:
        result = rank_major
    elif runtime.backend == "nccl":
        result = _empty(spec, maximum)
        dist.reduce_scatter_tensor(result, rank_major, op=dist.ReduceOp.SUM, group=runtime.group)
    else:
        # This fallback is deliberately restricted to the CPU gloo runtime.
        _require(runtime.backend == "gloo", "SUM_SCATTER_BACKEND_UNSUPPORTED")
        dist.all_reduce(rank_major, op=dist.ReduceOp.SUM, group=runtime.group)
        result = rank_major.narrow(0, runtime.rank * maximum, maximum)
    return result[:runtime.n_local].clone()


def _first_derivative_only():
    _require(not torch.is_grad_enabled(), "INFERENCE_FIRST_DERIVATIVES_ONLY")


class _GatherPartialSources(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owned, spec):
        ctx.spec = spec
        ctx.set_materialize_grads(False)
        return _gather_tensor(owned, spec)

    @staticmethod
    def backward(ctx, partial_global):
        _first_derivative_only()
        spec = ctx.spec
        if partial_global is None:
            partial_global = _zeros(spec, spec.runtime.n_global)
        return _sum_scatter_tensor(partial_global, spec), None


def gather_partial_sources(owned, spec):
    """Generic source gather; backward SUMs partial consumer VJPs at owners.

    Never put a complete-owned base VJP behind this transpose. Empty owners
    still require a differentiable owned input and an anchored output use.
    """
    _check_spec(spec, PARTIAL_SOURCE_SUM)
    _check_tensor(owned, spec, differentiable=True)
    return _GatherPartialSources.apply(owned, spec)


@torch.no_grad()
def sum_partial_source_adjoints(partial_global, spec):
    """Explicit detached SUM-RS for manually computed PARTIAL consumer VJPs."""
    _check_spec(spec, PARTIAL_SOURCE_SUM)
    _check_tensor(partial_global, spec, global_rows=True)
    return _sum_scatter_tensor(partial_global.detach(), spec)


@torch.no_grad()
def gather_detached_sources(owned, spec):
    """Data cache for complete-owned operators: no autograd transpose."""
    _check_spec(spec, DETACHED_SOURCE)
    _check_tensor(owned, spec)
    return _gather_tensor(owned.detach(), spec)


@torch.no_grad()
def gather_disjoint_output_adjoints(owned_adjoint, spec):
    """Gather each owned output row once, with NO SUM and no transpose.

    An unused output can supply None; it still contributes zero rows and enters
    the same collective. Complete-owned gradients return directly to local
    inputs after this gather, never through sum_partial_source_adjoints.
    """
    _check_spec(spec, DISJOINT_OUTPUT)
    if owned_adjoint is None:
        owned_adjoint = _zeros(spec, spec.runtime.n_local)
    _check_tensor(owned_adjoint, spec)
    return _gather_tensor(owned_adjoint.detach(), spec)


def _reduce(value, runtime, operation):
    if runtime.world_size > 1:
        dist.all_reduce(value, op=operation, group=runtime.group)
    return value


class _GlobalTwoChannelSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, spec):
        runtime = spec.runtime
        maximum = (logits.max(dim=0).values if runtime.n_local else
                   logits.new_full((2,), -torch.inf))
        _reduce(maximum, runtime, dist.ReduceOp.MAX)
        exponential = torch.exp(logits - maximum)
        denominator = _reduce(exponential.sum(dim=0), runtime, dist.ReduceOp.SUM)
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        probabilities = exponential / denominator
        ctx.spec = spec
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx, gradient):
        _first_derivative_only()
        (probabilities,) = ctx.saved_tensors
        contraction = _reduce((gradient * probabilities).sum(dim=0), ctx.spec.runtime, dist.ReduceOp.SUM)
        return probabilities * (gradient - contraction), None


def global_two_channel_softmax(owned_logits, spec):
    """Global normalization over one graph, with exact global contraction VJP."""
    _check_spec(spec, GLOBAL_SOFTMAX)
    _check_tensor(owned_logits, spec, differentiable=True)
    return _GlobalTwoChannelSoftmax.apply(owned_logits, spec)


class _GlobalChargeProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, spec):
        total = _reduce(raw.sum(), spec.runtime, dist.ReduceOp.SUM)
        ctx.spec = spec
        if spec.runtime.n_global == 0:
            return raw.clone()
        return raw + (spec.fixed_total_charge - total) / spec.runtime.n_global

    @staticmethod
    def backward(ctx, gradient):
        _first_derivative_only()
        spec = ctx.spec
        total = _reduce(gradient.sum(), spec.runtime, dist.ReduceOp.SUM)
        return (gradient.clone() if spec.runtime.n_global == 0 else
                gradient - total / spec.runtime.n_global), None


def project_global_charge(owned_raw_charge, spec):
    """Conserve one global graph's fixed charge; remove the GLOBAL VJP mean."""
    _check_spec(spec, GLOBAL_CHARGE)
    _check_tensor(owned_raw_charge, spec, differentiable=True)
    return _GlobalChargeProjection.apply(owned_raw_charge, spec)


__all__ = [
    "CollectiveContractError", "OwnershipMetadata", "CollectiveRuntime", "ExchangeSpec",
    "PARTIAL_SOURCE_SUM", "DETACHED_SOURCE", "DISJOINT_OUTPUT", "GLOBAL_SOFTMAX", "GLOBAL_CHARGE",
    "make_ownership_metadata", "edge_balanced_metadata", "ownership_diagnostics", "prepare_runtime",
    "prepare_exchange", "gather_partial_sources", "sum_partial_source_adjoints",
    "gather_detached_sources", "gather_disjoint_output_adjoints", "global_two_channel_softmax",
    "project_global_charge",
]
