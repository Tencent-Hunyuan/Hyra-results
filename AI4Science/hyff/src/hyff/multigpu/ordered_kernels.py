"""Control-ordered boundaries for the complete-owned base and pair VJPs.

The position source exchange has a SUM transpose for partial EFM/LR forces.
Base and pair kernels receive detached position/source caches and return
complete owned derivatives directly. Their aggregate cotangents use disjoint
all-gather, never SUM. The control token joins those backward collectives to
the same order as source packets, AEE, charge projection and LR.

This module adds no zero-valued arithmetic dependencies to physical tensors.
A constant first message remains constant, so its unused source VJP stays
omitted. CUDA graph capture of these boundaries is not tested here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from hyff.multigpu import collectives as raw
from hyff.multigpu import ordered_collectives as ordered
from hyff.multigpu import base_messages as base
from hyff.multigpu import pair_energy as pair


@dataclass(frozen=True)
class PositionExchange:
    owned: torch.Tensor
    global_positions: torch.Tensor
    detached_positions: torch.Tensor
    prepared: ordered.OrderedExchange


def exchange_positions(owned_positions, control, prepared):
    raw._require(isinstance(prepared, ordered.OrderedExchange) and
                 prepared.physical_gradient and prepared.spec.tail_shape == (3,) and
                 prepared.spec.dtype == torch.float32, "ORDERED_POSITION_EXCHANGE_REQUIRED")
    full, control = ordered.ordered_partial_gather(owned_positions, control, prepared)
    return PositionExchange(owned_positions, full, full.detach(), prepared), control


def _position_epoch(positions, graph):
    raw._require(isinstance(positions, PositionExchange), "POSITION_EXCHANGE_ORIGIN_REQUIRED")
    runtime = positions.prepared.spec.runtime
    raw._require(runtime.metadata == graph.metadata and runtime.rank == graph.rank and
                 runtime.device == graph.cell.device, "POSITION_EXCHANGE_EPOCH_MISMATCH")


def prepare_first_base_caches(operator):
    """One-time inert first-layer caches; they are never sent over the wire."""
    raw._require(isinstance(operator, base.PreparedBaseOperator) and operator.first,
                 "FIRST_BASE_OPERATOR_REQUIRED")
    raw._outside_capture(operator.graph.cell.device)
    return tuple(operator.graph.cell.new_zeros((operator.graph.n_global, components, 84))
                 for components in (3, 5))


def base_snapshot(positions, owned_fields, packet, packet_spec, operator, *, first_caches=None):
    """Build a base snapshot from the exchanged positions and packet.

    Packet unpacking produces strided fields. The base kernels need contiguous
    caches, so the fields are copied; the copies are part of the step cost.
    """
    _position_epoch(positions, operator.graph)
    raw._require(isinstance(packet, ordered.ProjectedPacket) and
                 isinstance(packet_spec, ordered.ProjectedPacketSpec) and
                 len(owned_fields) == 3 and len(packet.base) == (1 if operator.first else 3),
                 "BASE_PROJECTED_PACKET_REQUIRED")
    runtime = packet_spec.spec.runtime
    raw._require(runtime.metadata == operator.graph.metadata and runtime.rank == operator.graph.rank and
                 runtime.device == operator.graph.cell.device and
                 packet_spec.base_shapes == (((252,),) if operator.first else ((588,), (3,84), (5,84))),
                 "BASE_PACKET_EPOCH_OR_SHAPES_MISMATCH")
    owned = base.BaseInputs(positions.owned, *owned_fields)
    if operator.first:
        raw._require(isinstance(first_caches, tuple) and len(first_caches) == 2,
                     "PREPARED_INERT_FIRST_CACHES_REQUIRED")
        fields = (packet.base[0].contiguous(), *first_caches)
    else:
        raw._require(first_caches is None, "FIRST_CACHES_ONLY_FOR_FIRST_BLOCK")
        fields = tuple(value.contiguous() for value in packet.base)
    cache = base.BaseInputs(positions.detached_positions,
                           *fields)
    snapshot = base.SourceSnapshot(owned, cache, operator, "detached_collectives")
    base._check_snapshot(snapshot)
    return snapshot


def pair_snapshot(positions, operator):
    _position_epoch(positions, operator.graph)
    snapshot = pair.PairSnapshot(positions.owned, positions.detached_positions,
                                 operator, "detached_collectives")
    pair._snapshot_metadata(snapshot)
    return snapshot


@dataclass(frozen=True)
class OrderedBaseSpec:
    operator: base.PreparedBaseOperator
    physical_gradient_flags: tuple[bool, bool, bool, bool]
    label: str
    trace: ordered.CollectiveTrace | None = None


@dataclass(frozen=True)
class OrderedPairSpec:
    operator: pair.OwnedPairOperator
    label: str
    trace: ordered.CollectiveTrace | None = None


def prepare_ordered_base(operator, *, physical_gradient_flags, label, trace=None):
    flags = tuple(physical_gradient_flags)
    raw._require(isinstance(operator, base.PreparedBaseOperator) and
                 operator.adjoint_specs is not None and len(flags) == 4 and
                 flags[0] and all(type(flag) is bool for flag in flags),
                 "ORDERED_BASE_OPERATOR_AND_FLAGS_REQUIRED")
    spec = operator.adjoint_specs[0]
    raw._outside_capture(spec.runtime.device)
    ordered._label(label)
    ordered._trace_guard(trace, spec.runtime.device)
    raw._agreement(spec.runtime, {"ordered_base": label, "flags": flags, "first": operator.first})
    return OrderedBaseSpec(operator, flags, label, trace)


def prepare_ordered_pair(operator, *, label, trace=None):
    raw._require(isinstance(operator, pair.OwnedPairOperator) and operator.adjoint_spec is not None,
                 "ORDERED_PAIR_OPERATOR_REQUIRED")
    spec = operator.adjoint_spec
    raw._outside_capture(spec.runtime.device)
    ordered._label(label)
    ordered._trace_guard(trace, spec.runtime.device)
    raw._agreement(spec.runtime, {"ordered_pair": label, "physical_position_gradient": True})
    return OrderedPairSpec(operator, label, trace)


def _event(prepared, phase, operation, unused=()):
    op = prepared.operator
    spec = op.adjoint_specs[0] if isinstance(prepared, OrderedBaseSpec) else op.adjoint_spec
    ordered._trace(prepared.trace, phase, prepared.label, operation,
                   world_size=spec.runtime.world_size, unused=unused)


class _OrderedBase(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, message, vector, tensor, control, snapshot, prepared):
        ctx.set_materialize_grads(False)
        ctx.prepared, ctx.origin = prepared, snapshot.origin
        op, graph = snapshot.operator, snapshot.operator.graph
        ctx.save_for_backward(positions, message, vector, tensor, *snapshot.cache.tensors(),
                              graph.local_rows, graph.neighbors, graph.shifts, graph.counts,
                              graph.cell, op.coefficients)
        _event(prepared, "forward", "OWNED_BASE")
        return (*base.owned_base_forward(snapshot), control.clone())

    @staticmethod
    def backward(ctx, gs, gv, gt, _control_gradient):
        raw._first_derivative_only()
        saved, prepared = ctx.saved_tensors, ctx.prepared
        snapshot = base.SourceSnapshot(base.BaseInputs(*saved[:4]), base.BaseInputs(*saved[4:8]),
                                       prepared.operator, ctx.origin)
        global_adjoints = []
        for index, (gradient, spec) in enumerate(zip((gs, gv, gt), prepared.operator.adjoint_specs)):
            _event(prepared, "backward", "DISJOINT_ALL_GATHER_" + str(index),
                   (index,) if gradient is None else ())
            global_adjoints.append(raw.gather_disjoint_output_adjoints(gradient, spec))
        gradients = base.owned_base_vjp(snapshot, tuple(global_adjoints),
                                        needs=prepared.physical_gradient_flags)
        return (*gradients, ordered._zero_control(prepared.operator.adjoint_specs[0]), None, None)


def ordered_base_messages(snapshot, control, prepared):
    base._check_snapshot(snapshot)
    raw._require(isinstance(prepared, OrderedBaseSpec) and snapshot.operator is prepared.operator,
                 "ORDERED_BASE_OPERATOR_MISMATCH")
    raw._require(tuple(value.requires_grad for value in snapshot.owned.tensors()) ==
                 prepared.physical_gradient_flags, "BASE_PHYSICAL_GRADIENT_FLAGS_MISMATCH")
    ordered._control(control, prepared.operator.adjoint_specs[0])
    values = _OrderedBase.apply(*snapshot.owned.tensors(), control, snapshot, prepared)
    return tuple(values[:3]), values[3]


class _OrderedPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, control, snapshot, prepared):
        ctx.set_materialize_grads(False)
        ctx.prepared, ctx.origin = prepared, snapshot.origin
        op, graph = snapshot.operator, snapshot.operator.graph
        ctx.save_for_backward(positions, snapshot.global_positions, op.elements, op.coefficients,
                              graph.cell, graph.neighbors, graph.shifts, graph.counts, graph.local_rows)
        _event(prepared, "forward", "OWNED_PAIR")
        return pair.owned_pair_forward(snapshot), control.clone()

    @staticmethod
    def backward(ctx, local_cotangent, _control_gradient):
        raw._first_derivative_only()
        saved, prepared = ctx.saved_tensors, ctx.prepared
        snapshot = pair.PairSnapshot(saved[0], saved[1], prepared.operator, ctx.origin)
        _event(prepared, "backward", "DISJOINT_ALL_GATHER", (0,) if local_cotangent is None else ())
        global_cotangent = raw.gather_disjoint_output_adjoints(local_cotangent, prepared.operator.adjoint_spec)
        return (pair.owned_pair_vjp(snapshot, global_cotangent),
                ordered._zero_control(prepared.operator.adjoint_spec), None, None)


def ordered_pair_energy(snapshot, control, prepared):
    pair._snapshot_metadata(snapshot)
    raw._require(isinstance(prepared, OrderedPairSpec) and snapshot.operator is prepared.operator and
                 snapshot.owned_positions.requires_grad, "ORDERED_PAIR_OPERATOR_MISMATCH")
    ordered._control(control, prepared.operator.adjoint_spec)
    return _OrderedPair.apply(snapshot.owned_positions, control, snapshot, prepared)


__all__ = ["PositionExchange", "exchange_positions", "prepare_first_base_caches", "base_snapshot", "pair_snapshot",
           "OrderedBaseSpec", "OrderedPairSpec", "prepare_ordered_base", "prepare_ordered_pair",
           "ordered_base_messages", "ordered_pair_energy"]
