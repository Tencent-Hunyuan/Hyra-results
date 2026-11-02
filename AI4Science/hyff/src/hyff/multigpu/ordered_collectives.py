"""Explicit first-derivative collective ordering through control-only edges.

Preparation agrees physical-gradient flags across ranks outside capture. Start
the chain with ``seed_control(local_positions)`` and finish it with
``ordered_energy_anchor``. A free-standing scalar leaf is NOT a valid seed for
``autograd.grad(loss, local_positions)``: its ancestors can be pruned.

Physical projections never receive a numeric token. Detached base caches have
no transpose; the combined packet SUM-scatters only active EFM source fields.
The complete-owned base and pair backward join this chain through
ordered_kernels. Process groups come from the caller, only first derivatives
are supported, and CUDA graph capture of these collectives is not tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.distributed as dist

try:
    from hyff.multigpu import collectives as raw
except ImportError:
    from hyff.multigpu import collectives as raw


@dataclass
class CollectiveTrace:
    """Optional CPU-only test instrument; never enabled for CUDA capture."""
    events: list[dict] = field(default_factory=list)


def _trace(trace, phase, label, operation, *, world_size=1, unused=()):
    if trace is not None:
        trace.events.append({"phase": phase, "label": label, "operation": operation,
                             "world_size": world_size, "unused_physical_fields": list(unused)})


def _trace_guard(trace, device):
    raw._require(trace is None or (isinstance(trace, CollectiveTrace) and device.type == "cpu"),
                 "ORDER_TRACE_CPU_ONLY")


def _label(label):
    raw._require(type(label) is str and bool(label) and len(label) <= 96 and
                 all(c.isalnum() or c in "_-" for c in label), "ORDER_LABEL_INVALID")


def _control(control, spec):
    raw._require(isinstance(control, torch.Tensor) and control.shape == torch.Size([]) and
                 control.device == spec.runtime.device and control.dtype == spec.dtype and
                 control.requires_grad, "POSITION_CONNECTED_SCALAR_CONTROL_REQUIRED")


def _zero_control(spec):
    return torch.zeros((), dtype=spec.dtype, device=spec.runtime.device)


def _event(prepared, phase, operation, unused=()):
    _trace(prepared.trace, phase, prepared.label, operation,
           world_size=prepared.spec.runtime.world_size, unused=unused)


@dataclass(frozen=True)
class OrderedExchange:
    spec: raw.ExchangeSpec
    physical_gradient: bool
    label: str
    trace: CollectiveTrace | None = None


def prepare_ordered_exchange(spec, *, physical_gradient, label, trace=None):
    """Eager collective agreement; never derive flags from local empty rows."""
    raw._require(isinstance(spec, raw.ExchangeSpec) and spec.contract in
                 (raw.PARTIAL_SOURCE_SUM, raw.GLOBAL_SOFTMAX, raw.GLOBAL_CHARGE),
                 "ORDERED_EXCHANGE_CONTRACT_REQUIRED")
    raw._outside_capture(spec.runtime.device)
    raw._require(type(physical_gradient) is bool, "EXPLICIT_PHYSICAL_GRADIENT_FLAG_REQUIRED")
    _label(label)
    _trace_guard(trace, spec.runtime.device)
    raw._agreement(spec.runtime, {"ordered_exchange": label, "contract": spec.contract,
                                 "physical_gradient": physical_gradient,
                                 "tail_shape": spec.tail_shape, "dtype": str(spec.dtype)})
    return OrderedExchange(spec, physical_gradient, label, trace)


def _check_ordered(owned, control, prepared, contract):
    raw._require(isinstance(prepared, OrderedExchange), "PREPARED_ORDERED_EXCHANGE_REQUIRED")
    raw._check_spec(prepared.spec, contract)
    raw._check_tensor(owned, prepared.spec)
    raw._require(owned.requires_grad == prepared.physical_gradient,
                 "PHYSICAL_GRADIENT_FLAG_MISMATCH")
    _control(control, prepared.spec)


class _SeedControl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, value, trace):
        ctx.shape, ctx.dtype, ctx.device = tuple(positions.shape), positions.dtype, positions.device
        ctx.trace = trace
        _trace(trace, "forward", "seed", "POSITION_CONTROL_SEED")
        return positions.new_full((), value)

    @staticmethod
    def backward(ctx, _gradient):
        raw._first_derivative_only()
        _trace(ctx.trace, "backward", "seed", "ZERO_POSITION_COTANGENT")
        return torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device), None, None


def seed_control(local_positions, *, value=0.0, trace=None):
    """Connect control ancestry to the exact owned-position grad target.

    The numeric value is irrelevant and can differ by rank. No token is added
    to a physical tensor. The seed contributes exactly zero to dpositions.
    """
    raw._require(isinstance(local_positions, torch.Tensor) and local_positions.ndim == 2 and
                 local_positions.shape[1] == 3 and local_positions.requires_grad and
                 local_positions.dtype in (torch.float32, torch.float64) and torch.is_grad_enabled(),
                 "DIFFERENTIABLE_OWNED_POSITIONS_REQUIRED")
    raw._require(type(value) in (int, float) and math.isfinite(value), "FINITE_CONTROL_VALUE_REQUIRED")
    _trace_guard(trace, local_positions.device)
    return _SeedControl.apply(local_positions, value, trace)


class _OrderedPartialGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owned, control, prepared):
        ctx.prepared = prepared
        ctx.set_materialize_grads(False)
        _event(prepared, "forward", "ALL_GATHER")
        output = raw._gather_tensor(owned, prepared.spec)
        if not prepared.physical_gradient:
            ctx.mark_non_differentiable(output)
        return output, control.clone()

    @staticmethod
    def backward(ctx, gradient, _control_gradient):
        raw._first_derivative_only()
        prepared, owned_gradient = ctx.prepared, None
        if prepared.physical_gradient:
            unused = gradient is None
            if unused:
                gradient = raw._zeros(prepared.spec, prepared.spec.runtime.n_global)
            _event(prepared, "backward", "SUM_SCATTER", (0,) if unused else ())
            owned_gradient = raw._sum_scatter_tensor(gradient, prepared.spec)
        else:
            _event(prepared, "backward", "CONTROL_ONLY")
        return owned_gradient, _zero_control(prepared.spec), None


def ordered_partial_gather(owned, control, prepared):
    _check_ordered(owned, control, prepared, raw.PARTIAL_SOURCE_SUM)
    return _OrderedPartialGather.apply(owned, control, prepared)


class _OrderedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, control, prepared):
        spec, runtime = prepared.spec, prepared.spec.runtime
        maximum = logits.max(dim=0).values if runtime.n_local else logits.new_full((2,), -torch.inf)
        _event(prepared, "forward", "ALL_REDUCE_MAX")
        raw._reduce(maximum, runtime, dist.ReduceOp.MAX)
        exponential = torch.exp(logits - maximum)
        _event(prepared, "forward", "ALL_REDUCE_SUM")
        denominator = raw._reduce(exponential.sum(dim=0), runtime, dist.ReduceOp.SUM)
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        probabilities = exponential / denominator
        ctx.prepared = prepared
        ctx.set_materialize_grads(False)
        if prepared.physical_gradient:
            ctx.save_for_backward(probabilities)
        else:
            ctx.mark_non_differentiable(probabilities)
        return probabilities, control.clone()

    @staticmethod
    def backward(ctx, gradient, _control_gradient):
        raw._first_derivative_only()
        prepared, logits_gradient = ctx.prepared, None
        if prepared.physical_gradient:
            (probabilities,) = ctx.saved_tensors
            unused = gradient is None
            if unused:
                gradient = torch.zeros_like(probabilities)
            _event(prepared, "backward", "ALL_REDUCE_SUM", (0,) if unused else ())
            contraction = raw._reduce((gradient * probabilities).sum(dim=0),
                                      prepared.spec.runtime, dist.ReduceOp.SUM)
            logits_gradient = probabilities * (gradient - contraction)
        else:
            _event(prepared, "backward", "CONTROL_ONLY")
        return logits_gradient, _zero_control(prepared.spec), None


def ordered_global_softmax(owned_logits, control, prepared):
    _check_ordered(owned_logits, control, prepared, raw.GLOBAL_SOFTMAX)
    return _OrderedSoftmax.apply(owned_logits, control, prepared)


class _OrderedCharge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owned_raw, control, prepared):
        spec = prepared.spec
        _event(prepared, "forward", "ALL_REDUCE_SUM")
        total = raw._reduce(owned_raw.sum(), spec.runtime, dist.ReduceOp.SUM)
        output = (owned_raw.clone() if spec.runtime.n_global == 0 else
                  owned_raw + (spec.fixed_total_charge - total) / spec.runtime.n_global)
        ctx.prepared = prepared
        ctx.set_materialize_grads(False)
        if not prepared.physical_gradient:
            ctx.mark_non_differentiable(output)
        return output, control.clone()

    @staticmethod
    def backward(ctx, gradient, _control_gradient):
        raw._first_derivative_only()
        prepared, owned_gradient = ctx.prepared, None
        if prepared.physical_gradient:
            spec = prepared.spec
            unused = gradient is None
            if unused:
                gradient = raw._zeros(spec, spec.runtime.n_local)
            _event(prepared, "backward", "ALL_REDUCE_SUM", (0,) if unused else ())
            total = raw._reduce(gradient.sum(), spec.runtime, dist.ReduceOp.SUM)
            owned_gradient = gradient.clone() if spec.runtime.n_global == 0 else gradient - total/spec.runtime.n_global
        else:
            _event(prepared, "backward", "CONTROL_ONLY")
        return owned_gradient, _zero_control(prepared.spec), None


def ordered_global_charge(owned_raw, control, prepared):
    _check_ordered(owned_raw, control, prepared, raw.GLOBAL_CHARGE)
    return _OrderedCharge.apply(owned_raw, control, prepared)


@dataclass(frozen=True)
class ProjectedPacketSpec:
    spec: raw.ExchangeSpec
    reverse_spec: raw.ExchangeSpec | None
    base_shapes: tuple[tuple[int, ...], ...]
    efm_shapes: tuple[tuple[int, ...], ...]
    efm_gradient_flags: tuple[bool, ...]
    label: str
    trace: CollectiveTrace | None = None


@dataclass(frozen=True)
class ProjectedPacket:
    base: tuple[torch.Tensor, ...]
    efm: tuple[torch.Tensor, ...]
    control: torch.Tensor


def prepare_projected_packet(runtime, base_shapes, efm_shapes, dtype, *, efm_gradient_flags,
                             label, trace=None):
    """Agree a one-AG packet and an active-EFM-only reverse SUM-RS packet.

    Base fields always return detached caches. Their complete local VJP belongs
    to the dedicated base operator, never this source transpose.
    """
    raw._require(isinstance(runtime, raw.CollectiveRuntime), "PREPARED_RUNTIME_REQUIRED")
    raw._outside_capture(runtime.device)
    base, efm = tuple(map(tuple, base_shapes)), tuple(map(tuple, efm_shapes))
    flags = tuple(efm_gradient_flags)
    raw._require(bool(base or efm) and all(all(type(v) is int and v > 0 for v in shape)
                                        for shape in base + efm), "PACKET_SHAPES_INVALID")
    raw._require(len(flags) == len(efm) and all(type(flag) is bool for flag in flags),
                 "EXPLICIT_EFM_GRADIENT_FLAGS_REQUIRED")
    _label(label)
    _trace_guard(trace, runtime.device)
    raw._agreement(runtime, {"projected_packet": label, "base_shapes": base, "efm_shapes": efm,
                             "efm_gradient_flags": flags, "dtype": str(dtype)})
    width = sum(math.prod(shape) for shape in base + efm)
    reverse_width = sum(math.prod(shape) for shape, flag in zip(efm, flags) if flag)
    forward = raw.prepare_exchange(runtime, (width,), dtype, raw.DETACHED_SOURCE)
    reverse = (raw.prepare_exchange(runtime, (reverse_width,), dtype, raw.PARTIAL_SOURCE_SUM)
               if reverse_width else None)
    return ProjectedPacketSpec(forward, reverse, base, efm, flags, label, trace)


def _pack(fields, shapes, rows):
    return torch.cat([value.reshape(rows, math.prod(shape)) for value, shape in zip(fields, shapes)], dim=1)


def _unpack(packet, shapes, rows):
    fields, offset = [], 0
    for shape in shapes:
        width = math.prod(shape)
        fields.append(packet[:, offset:offset+width].reshape(rows, *shape))
        offset += width
    return tuple(fields)


class _OrderedProjectedPacket(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *arguments):
        control, prepared = arguments[-2:]
        fields = arguments[:-2]
        shapes = prepared.base_shapes + prepared.efm_shapes
        runtime = prepared.spec.runtime
        _event(prepared, "forward", "ALL_GATHER")
        packet = raw._gather_tensor(_pack(fields, shapes, runtime.n_local), prepared.spec)
        outputs = _unpack(packet, shapes, runtime.n_global)
        flags = (False,)*len(prepared.base_shapes) + prepared.efm_gradient_flags
        detached = tuple(value for value, flag in zip(outputs, flags) if not flag)
        if detached:
            ctx.mark_non_differentiable(*detached)
        ctx.prepared = prepared
        ctx.set_materialize_grads(False)
        return (*outputs, control.clone())

    @staticmethod
    def backward(ctx, *gradients):
        raw._first_derivative_only()
        prepared = ctx.prepared
        base_count, runtime = len(prepared.base_shapes), prepared.spec.runtime
        efm_gradients = gradients[base_count:-1]
        owned = [None]*len(prepared.efm_shapes)
        if prepared.reverse_spec is not None:
            active_fields, active_shapes, active_indices, unused = [], [], [], []
            for i, (gradient, shape, flag) in enumerate(zip(efm_gradients, prepared.efm_shapes,
                                                           prepared.efm_gradient_flags)):
                if flag:
                    if gradient is None:
                        unused.append(i)
                        gradient = torch.zeros((runtime.n_global, *shape), dtype=prepared.spec.dtype,
                                               device=runtime.device)
                    active_fields.append(gradient)
                    active_shapes.append(shape)
                    active_indices.append(i)
            _event(prepared, "backward", "EFM_ONLY_SUM_SCATTER", unused)
            local = raw._sum_scatter_tensor(_pack(active_fields, active_shapes, runtime.n_global),
                                           prepared.reverse_spec)
            for i, value in zip(active_indices, _unpack(local, active_shapes, runtime.n_local)):
                owned[i] = value
        else:
            _event(prepared, "backward", "CONTROL_ONLY")
        return (*([None]*base_count), *owned, _zero_control(prepared.spec), None)


def ordered_projected_packet(base_fields, efm_fields, control, prepared):
    raw._require(isinstance(prepared, ProjectedPacketSpec), "PREPARED_PROJECTED_PACKET_REQUIRED")
    base_fields, efm_fields = tuple(base_fields), tuple(efm_fields)
    raw._require(len(base_fields) == len(prepared.base_shapes) and len(efm_fields) == len(prepared.efm_shapes),
                 "PACKET_FIELD_COUNT_MISMATCH")
    runtime = prepared.spec.runtime
    for value, shape in zip(base_fields+efm_fields, prepared.base_shapes+prepared.efm_shapes):
        raw._require(isinstance(value, torch.Tensor) and value.device == runtime.device and
                     value.dtype == prepared.spec.dtype and tuple(value.shape) == (runtime.n_local, *shape),
                     "PACKET_FIELD_METADATA_MISMATCH")
    raw._require(all(value.requires_grad == flag for value, flag in zip(efm_fields, prepared.efm_gradient_flags)),
                 "PHYSICAL_GRADIENT_FLAG_MISMATCH")
    _control(control, prepared.spec)
    result = _OrderedProjectedPacket.apply(*base_fields, *efm_fields, control, prepared)
    split = len(base_fields)
    return ProjectedPacket(tuple(result[:split]), tuple(result[split:-1]), result[-1])


class _EnergyAnchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, energy, control, label, trace):
        ctx.dtype, ctx.device, ctx.label, ctx.trace = control.dtype, control.device, label, trace
        _trace(trace, "forward", label, "ENERGY_IDENTITY")
        return energy.clone()

    @staticmethod
    def backward(ctx, energy_gradient):
        raw._first_derivative_only()
        _trace(ctx.trace, "backward", ctx.label, "ENERGY_IDENTITY")
        return energy_gradient, torch.zeros((), dtype=ctx.dtype, device=ctx.device), None, None


def ordered_energy_anchor(local_energy, control, *, label="energy", trace=None):
    """Identity physical energy; its zero control VJP closes the ordered chain."""
    raw._require(isinstance(local_energy, torch.Tensor) and local_energy.numel() == 1 and
                 isinstance(control, torch.Tensor) and control.shape == torch.Size([]) and
                 local_energy.device == control.device and
                 local_energy.dtype in (torch.float32, torch.float64) and
                 control.dtype in (torch.float32, torch.float64) and
                 control.requires_grad, "ENERGY_CONTROL_METADATA_MISMATCH")
    _label(label)
    _trace_guard(trace, control.device)
    return _EnergyAnchor.apply(local_energy, control, label, trace)


__all__ = ["CollectiveTrace", "OrderedExchange", "ProjectedPacketSpec", "ProjectedPacket",
           "prepare_ordered_exchange", "prepare_projected_packet", "seed_control",
           "ordered_partial_gather", "ordered_global_softmax", "ordered_global_charge",
           "ordered_projected_packet", "ordered_energy_anchor"]
