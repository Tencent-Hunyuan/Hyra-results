"""Prepared fixed-epoch owned EFM operator, with eager input validation kept separate.

Prepare outside capture; evaluation uses metadata checks and tensor operations,
plus host scalar reads while compiled first-use parity probes are pending.
Each prepare allocates private fixed-shape metadata buffers and advances
the generation. Replacing an epoch invalidates the prior operator. The caller
must check assert_replay_generation before each graph replay, since replay
bypasses Python guards. CUDA graph capture of this operator is not validated.

The seven-input aggregate forward/backward follows projected_efm (with
fixed-shape padded tiles when compiled and optional retained tile graphs); its
edge, tile-input and mask functions are reused. This module installs nothing
into the model and performs no projection, collective, device selection or launch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

try:
    from hyff.multigpu.projected_efm import (
        MAX_TILE_EDGES, OwnedProjectedEFMError, _require, _input_shapes, _metadata, _tile_inputs, _masked,
        _edge_outputs, frame_matrices, smooth_switch,
    )
except ImportError:
    from hyff.multigpu.projected_efm import (
        MAX_TILE_EDGES, OwnedProjectedEFMError, _require, _input_shapes, _metadata, _tile_inputs, _masked,
        _edge_outputs, frame_matrices, smooth_switch,
    )


def _outside_capture(device):
    if device.type == "cuda":
        _require(device.index == torch.cuda.current_device(), "EFM_CURRENT_DEVICE_REQUIRED")
        _require(not torch.cuda.is_current_stream_capturing(), "EFM_PREPARE_OUTSIDE_CAPTURE_REQUIRED")


def _tensor_stamp(value):
    # Storage/shape/version metadata only: no tensor value reaches the host.
    return (id(value), value.data_ptr(), value._version, tuple(value.shape),
            tuple(value.stride()), value.dtype, value.device, value.requires_grad)


def _parameter_stamp(efm):
    return tuple((name, _tensor_stamp(parameter)) for name, parameter in efm.named_parameters())


def _structure_stamp(efm):
    return (efm.channels, efm.input_is_scalar, tuple(efm.m_blocks), efm.message_scale,
            efm.aggregation, efm.frame, efm.complex_structure,
            tuple((name, id(module), module.training) for name, module in efm.named_modules()))


@dataclass(frozen=True)
class PreparedOwnedEFMMetadata:
    n_global: int
    n_owned: int
    edge_capacity: int
    valid_edges: int
    tile_edges: int
    generation: int
    epoch: int
    dtype: torch.dtype
    device: torch.device
    source: torch.Tensor
    destination: torch.Tensor
    valid: torch.Tensor
    owned_global_indices: torch.Tensor
    unit_shifts: torch.Tensor
    safe_source: torch.Tensor
    global_destination: torch.Tensor

    def buffers(self):
        return (self.source, self.destination, self.valid, self.owned_global_indices,
                self.unit_shifts, self.safe_source, self.global_destination)


@dataclass(frozen=True)
class OwnedEFMFactoryReuse:
    """Only epoch-invariant code/model identity; no topology or autograd tape is stored."""
    efm: object
    tile_edges: int
    compile_requested: bool
    aggregate: object
    parameters: tuple
    structure: tuple
    retained: bool = False
    retention_max_rows: int | None = None
    capacity_retained_aggregate: object | None = None


COMPILED_TILE_PROBE_REL_TOL = 1e-4
COMPILED_TILE_PROBES = []  # per-process record of compiled-aggregate first-use parity


def _make_aggregate(efm, tile_edges, compiled, *, retained=False, retention_max_rows=None,
                    edge_factory=None):
    has_m = bool(efm.m_blocks)

    def native_edges(*values):
        return _edge_outputs(efm, *values)

    if edge_factory is not None:
        # Reuse the exact compiled callable across policy/epoch transitions.
        native_edges, edge_function = edge_factory.native_edges, edge_factory.edge_function
    else:
        edge_function = (torch.compile(native_edges, fullgraph=True, dynamic=False,
                                       options={"triton.cudagraphs": False})
                         if compiled else native_edges)

    def tiles(source, valid):
        # Compiled EFM tiles keep one static shape. The final partial
        # tile is padded with invalid rows (clamped index, mask False), so a new
        # epoch edge count reuses the graphs instead of hitting the dynamo
        # recompile limit, which fullgraph=True makes a hard error. Eager is unchanged.
        edges = source.numel()
        for begin in range(0, edges, tile_edges):
            stop = min(begin + tile_edges, edges)
            if not compiled or stop - begin == tile_edges:
                part = slice(begin, stop)
                yield begin, stop, part, valid[part]
            else:
                rows = torch.arange(begin, begin + tile_edges, device=source.device)
                part = rows.clamp_max(edges - 1)
                yield begin, stop, part, valid[part] & (rows < edges)

    # Compiled-tile parity probe (fails closed): CPU inductor produced a deterministic
    # wrong fused backward for this function (forward exact). The first tile with a
    # valid nonzero-envelope edge is compared with eager, forward and backward.
    # Skin-only tiles have zero input VJPs and cannot validate the compiled function;
    # a frame containing only such tiles leaves the probe pending. These first-use
    # checks synchronize; the steady path performs no tensor-value host reads.
    probe = dict(forward=None, backward=None)
    if compiled or retained:
        COMPILED_TILE_PROBES.append(dict(channels=efm.channels, m_blocks=tuple(efm.m_blocks),
                                   retained=retained, compiled=compiled, probe=probe))

    def pending(kind, mask, local):
        return (compiled or retained) and probe[kind] is None and bool((mask & (local[-1] != 0)).any())

    def check_parity(compiled_values, eager_values):
        _require(len(compiled_values) == len(eager_values), "EFM_COMPILED_TILE_PARITY_FAILED")
        rels = []
        for c, e in zip(compiled_values, eager_values):
            if c is None or e is None:
                other = e if c is None else c
                _require(other is None or not other.numel(), "EFM_COMPILED_TILE_PARITY_FAILED")
                continue
            _require(c.shape == e.shape, "EFM_COMPILED_TILE_PARITY_FAILED")
            if e.numel():
                rels.append(float((c - e).abs().max()) / max(float(e.abs().max()), 1e-30))
        _require(all(rel <= COMPILED_TILE_PROBE_REL_TOL for rel in rels), "EFM_COMPILED_TILE_PARITY_FAILED")
        return max(rels, default=0.0)

    def vjp(function, local, mask, cotangents, observation=None):
        leaves = tuple(value.detach().requires_grad_(True) for value in local)
        if observation is not None:
            return observation.vjp(lambda: tuple(_masked(edge, mask) for edge in function(*leaves)),
                                   leaves, cotangents, rows=mask.numel())
        outputs = tuple(_masked(edge, mask) for edge in function(*leaves))
        return torch.autograd.grad(outputs, leaves, cotangents,
                                   allow_unused=True, create_graph=False)

    def retained_vjp(pair, cotangents):
        leaves, outputs = pair
        return torch.autograd.grad(outputs, leaves, cotangents, allow_unused=True,
                                   create_graph=False, retain_graph=True)

    def independent_backward_probe(local, mask, pair=None):
        # The model's first actual cotangent can be zero (for example an inert
        # exit), and a zero cotangent would let a broken VJP pass. Use deterministic
        # nonzero channel weights as well; no global RNG state is consumed.
        channels = (torch.arange(efm.channels, device=local[0].device) % 17 + 1).to(local[0].dtype) / 17
        cotangents = tuple((channels * (index + 1)).expand(mask.numel(), *tail, efm.channels).contiguous()
                           for index, tail in enumerate(((), (3,), (5,))))
        # Each helper frees its autograd tape before the other starts. Only
        # compact gradient results overlap, never two full edge-chain tapes.
        compiled_gradients = (vjp(edge_function, local, mask, cotangents) if pair is None
                              else retained_vjp(pair, cotangents))
        eager_gradients = vjp(native_edges, local, mask, cotangents)
        return check_parity(compiled_gradients, eager_gradients)

    class OwnedAggregate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, S, V, T, rbf, R, D, envelope, source, destination, valid, n_owned,
                    retain_enabled=True, observation=None, retained_rows_limit=None):
            values = (S, V, T, rbf, R, D, envelope)
            ctx.storage_observation = observation
            ctx.retained = retained and retain_enabled and any(ctx.needs_input_grad[:7])
            if ctx.retained:
                limit = retention_max_rows if retained_rows_limit is None else retained_rows_limit
                _require(type(limit) is int and source.numel() <= limit, "EFM_RETENTION_ROWS_EXCEEDED")
            tile_saved = []
            ctx.n_owned = n_owned
            C = efm.channels
            # The final scratch row absorbs padding, including n_owned == 0.
            totals = (S.new_zeros((n_owned + 1, C)), S.new_zeros((n_owned + 1, 3, C)),
                      S.new_zeros((n_owned + 1, 5, C)))
            for _, _, part, mask in tiles(source, valid):
                dst = torch.where(mask, destination[part], torch.full_like(destination[part], n_owned))
                local = _tile_inputs(values, source[part], mask, part, has_m)
                if ctx.retained:
                    with torch.enable_grad():
                        leaves = tuple(value.detach().requires_grad_(True) for value in local)
                        edges = tuple(_masked(edge, mask) for edge in edge_function(*leaves))
                    tile_saved.extend((*leaves, *edges))
                else:
                    edges = tuple(_masked(edge, mask) for edge in edge_function(*local))
                if (not retained or ctx.retained) and pending('forward', mask, local):
                    probe['forward'] = check_parity(edges, tuple(_masked(edge, mask) for edge in native_edges(*local)))
                for total, edge in zip(totals, edges):
                    total.index_add_(0, dst, edge.detach())
            # Only the outer engine owns these private graphs. No ctx tensor
            # attributes, output references, or reusable factory tape cache.
            ctx.save_for_backward(*values, source, destination, valid, *tile_saved)
            return tuple(total[:n_owned] for total in totals)

        @staticmethod
        def backward(ctx, grad_s, grad_v, grad_t):
            if torch.is_grad_enabled():
                raise RuntimeError("EFM_FIRST_DERIVATIVE_ONLY")
            saved = ctx.saved_tensors
            values, source, destination, valid = saved[:7], saved[7], saved[8], saved[9]
            needs = ctx.needs_input_grad[:7]
            gradients = [torch.zeros_like(value) if needed else None
                         for value, needed in zip(values, needs)]
            # Empty output cotangents get the same harmless sentinel row.
            cotangent_rows = tuple(torch.cat((gradient, gradient.new_zeros((1, *gradient.shape[1:]))), 0)
                                   for gradient in (grad_s, grad_v, grad_t))
            for tile_index, (begin, stop, part, mask) in enumerate(tiles(source, valid)):
                src = torch.where(mask, source[part], torch.zeros_like(source[part]))
                dst = torch.where(mask, destination[part], torch.full_like(destination[part], ctx.n_owned))
                if ctx.retained:
                    offset = 10 + tile_index * 10
                    local, outputs = saved[offset:offset + 7], saved[offset + 7:offset + 10]
                    pair = (local, outputs)
                else:
                    local = _tile_inputs(values, source[part], mask, part, has_m)
                    pair = None
                with torch.enable_grad():
                    check_backward = pending('backward', mask, local)
                    independent_rel = independent_backward_probe(local, mask, pair) if check_backward else None
                    cotangents = tuple(gradient.index_select(0, dst) for gradient in cotangent_rows)
                    local_gradients = (vjp(edge_function, local, mask, cotangents,
                                           observation=ctx.storage_observation) if pair is None
                                       else retained_vjp(pair, cotangents))
                    if check_backward:
                        eager_gradients = vjp(native_edges, local, mask, cotangents)
                        probe['backward'] = max(independent_rel, check_parity(local_gradients, eager_gradients))
                        del eager_gradients
                for index, (total, gradient) in enumerate(zip(gradients, local_gradients)):
                    if total is None or gradient is None:
                        continue
                    if index < 3:
                        if total.shape[0]:
                            total.index_add_(0, src, gradient)
                    else:
                        total[begin:stop].copy_(gradient[:stop - begin])
            return (*gradients, *(None for _ in ctx.needs_input_grad[7:]))

    OwnedAggregate.edge_function = staticmethod(edge_function)
    OwnedAggregate.native_edges = staticmethod(native_edges)
    OwnedAggregate.tiles = staticmethod(tiles)
    OwnedAggregate.check_parity = staticmethod(check_parity)
    OwnedAggregate.probe = probe
    OwnedAggregate.retained = retained
    OwnedAggregate.compiled = compiled
    return OwnedAggregate


class PreparedOwnedEFM:
    """Fixed-buffer epoch. The caller owns ordered collective and capture use."""

    def __init__(self, efm, metadata, *, compile_requested, factory_reuse=None):
        self.efm, self.metadata = efm, metadata
        self._active_selector = None
        self.compile_requested = compile_requested
        self.compiled = compile_requested and not bool(efm.input_is_scalar)
        self._parameters = _parameter_stamp(efm)
        self._structure = _structure_stamp(efm)
        self._calls = 0
        self._retention_configured = False
        self._calibration_options = None
        self._calibration_started = False
        self.calibration_report = None
        # A fresh epoch always starts with replay and fresh row bounds. Reused
        # factories contain code/model identity only, never a previous plan.
        self.retained = False
        self.retention_max_rows = None
        self._capacity_token = None
        self._capacity_block_index = None
        self._capacity_epoch_generation = None
        self._capacity_failed = False
        self._replay_storage_observer = None
        self._capacity_retained_aggregate = None
        if factory_reuse is None:
            self._aggregate = _make_aggregate(efm, metadata.tile_edges, self.compiled)
        else:
            _require(isinstance(factory_reuse, OwnedEFMFactoryReuse) and factory_reuse.efm is efm and
                     factory_reuse.tile_edges == metadata.tile_edges and
                     factory_reuse.compile_requested == compile_requested and
                     factory_reuse.parameters == self._parameters and
                     factory_reuse.structure == self._structure,
                     "EFM_FACTORY_REUSE_IDENTITY_CHANGED")
            self._aggregate = factory_reuse.aggregate
            self._capacity_retained_aggregate = factory_reuse.capacity_retained_aggregate
        self._replay_aggregate = self._aggregate
        self._metadata_identity = id(metadata)
        self._buffers = tuple(_tensor_stamp(value) for value in metadata.buffers())
        self._shapes = ((metadata.n_global, efm.channels),
                        (metadata.n_global, 3, efm.channels) if efm.m_blocks else (0,),
                        (metadata.n_global, 5, efm.channels) if efm.m_blocks else (0,),
                        (metadata.edge_capacity, efm.radial[0].in_features),
                        (metadata.edge_capacity, 3, 3), (metadata.edge_capacity, 5, 5),
                        (metadata.edge_capacity,))
        self._superseded = False

    def observe_replay_storage(self, observation):
        """Attach scalar-only instrumentation to actual replay VJPs."""
        self._assert_current()
        _require(not self.retained, "EFM_CAPACITY_OBSERVATION_REPLAY_REQUIRED")
        self._replay_storage_observer = observation

    def apply_capacity_plan(self, enabled, *, max_rows, token, block_index):
        """Only the collective preflight controller calls this between forces."""
        from hyff.multigpu.efm_capacity_plan import POLICY_TOKEN
        self._assert_current()
        _require(type(enabled) is bool and type(max_rows) is int and max_rows >= 0 and
                 type(block_index) is int and 0 <= block_index < 4 and isinstance(token, dict) and
                 token.get("epoch") == self.metadata.epoch and
                 token.get("policy_token") == POLICY_TOKEN and
                 type(token.get("generation")) is int and token["generation"] >= 0 and
                 (self._capacity_epoch_generation is None or
                  token["generation"] == self._capacity_epoch_generation) and
                 len(token.get("capacities", [])) == len(token.get("capacity_generations", [])) == 4 and
                 token.get("capacities", [])[block_index] == max_rows,
                 "EFM_CAPACITY_PLAN_INVALID")
        if enabled and self._capacity_retained_aggregate is None:
            self._capacity_retained_aggregate = _make_aggregate(self.efm, self.metadata.tile_edges,
                self.compiled, retained=True, edge_factory=self._replay_aggregate)
        self._aggregate = self._capacity_retained_aggregate if enabled else self._replay_aggregate
        self.retained, self.retention_max_rows = enabled, max_rows if enabled else None
        self._capacity_token = dict(token, capacities=tuple(token["capacities"]),
                                    capacity_generations=tuple(token["capacity_generations"]))
        self._capacity_block_index, self._capacity_epoch_generation = block_index, token["generation"]

    def configure_retention(self, enabled, *, max_rows):
        """Setup-only fixed policy; never an OOM fallback or a per-call decision."""
        self._assert_current()
        _require(type(enabled) is bool and type(max_rows) is int and max_rows >= 0,
                 "EFM_RETENTION_PLAN_INVALID")
        _require((not self._calls or (self._calls == 1 and self.calibration_report is not None)) and
                 not self._retention_configured and
                 (not self._calibration_started or self.calibration_report is not None),
                 "EFM_RETENTION_SETUP_ONLY")
        self.retained, self.retention_max_rows = enabled, max_rows if enabled else None
        self._aggregate = _make_aggregate(self.efm, self.metadata.tile_edges, self.compiled,
            retained=enabled, retention_max_rows=self.retention_max_rows)
        self._retention_configured = True

    def enable_retention_calibration(self, *, calibration_headroom_bytes,
                                     allocator_ceiling_bytes=64 * 1024 ** 3,
                                     reserve_bytes=16 * 1024 ** 3):
        """Observe actual inputs once during the ordinary replay setup force."""
        _require(not self._calls and not self.retained and self._calibration_options is None and
                 not self._retention_configured, "EFM_RETENTION_CALIBRATION_SETUP_ONLY")
        self._calibration_options = dict(calibration_headroom_bytes=calibration_headroom_bytes,
            allocator_ceiling_bytes=allocator_ceiling_bytes, reserve_bytes=reserve_bytes)

    def calibrate_retained_tile(self, S, V, T, rbf, R, D, envelope, *, selection=None):
        """One real edge tile; returns storage counts only, never a live graph."""
        from hyff.multigpu.efm_retention import calibrate_retained_tile
        values = (S, V, T, rbf, R, D, envelope)
        self._check_inputs(values, selection)
        meta = self.metadata
        source, valid = ((meta.source, meta.valid) if selection is None else
                         (selection.source, selection.valid))
        _require(source.numel() >= meta.tile_edges, "EFM_RETENTION_CALIBRATION_FULL_TILE_REQUIRED")
        for _, _, part, mask in self._aggregate.tiles(source, valid):
            with torch.no_grad():
                local = _tile_inputs(values, source[part], mask, part, bool(self.efm.m_blocks))
            if bool((mask & (local[-1] != 0)).any()):
                _require(mask.numel() == meta.tile_edges, "EFM_RETENTION_CALIBRATION_FULL_TILE_REQUIRED")
                return calibrate_retained_tile(self.efm, self._aggregate.edge_function,
                    self._aggregate.native_edges, local, mask,
                    check_parity=self._aggregate.check_parity, compiled=self.compiled)
        raise OwnedProjectedEFMError("EFM_RETENTION_CALIBRATION_NONZERO_TILE_REQUIRED")

    def factory_reuse(self):
        self._assert_current()
        return OwnedEFMFactoryReuse(self.efm, self.metadata.tile_edges, self.compile_requested,
                                    self._replay_aggregate, self._parameters, self._structure,
                                    False, None, self._capacity_retained_aggregate)

    def _assert_current(self):
        _require(not self._capacity_failed, "EFM_CAPACITY_CONTROLLER_RETIRED")
        _require(not self._superseded, "EFM_PREPARED_EPOCH_SUPERSEDED")
        _require(id(self.metadata) == self._metadata_identity and _structure_stamp(self.efm) == self._structure,
                 "EFM_PREPARED_STRUCTURE_CHANGED")
        _require(_parameter_stamp(self.efm) == self._parameters, "EFM_PREPARED_PARAMETERS_CHANGED")
        _require(tuple(_tensor_stamp(value) for value in self.metadata.buffers()) == self._buffers,
                 "EFM_PREPARED_BUFFERS_CHANGED")

    def _check_inputs(self, values, selection=None):
        self._assert_current()
        shapes = self._shapes
        if selection is not None:
            from hyff.multigpu.active_efm import ActiveEFMSelection
            _require(isinstance(selection, ActiveEFMSelection), "EFM_ACTIVE_SELECTION_REQUIRED")
            selection.check(operator=self)
            rows = selection.source.numel()
            shapes = shapes[:3] + tuple((rows, *shape[1:]) for shape in shapes[3:])
        else:
            _require(self._active_selector is None, "EFM_ACTIVE_SELECTION_REQUIRED")
        _require(len(values) == 7 and all(isinstance(value, torch.Tensor) and
                 tuple(value.shape) == shape and value.dtype == self.metadata.dtype and
                 value.device == self.metadata.device for value, shape in zip(values, shapes)),
                 "EFM_PREPARED_INPUT_METADATA_MISMATCH")

    def aggregate(self, S, V, T, rbf, R, D, envelope, *, selection=None):
        values = (S, V, T, rbf, R, D, envelope)
        self._check_inputs(values, selection)
        meta = self.metadata
        rows = meta.source.numel() if selection is None else selection.source.numel()
        if self._capacity_token is not None:
            index, token = self._capacity_block_index, self._capacity_token
            capacity = meta.source.numel() if selection is None else selection.allocation_capacity
            generation = 0 if selection is None else selection.capacity_generation
            _require(token["epoch"] == meta.epoch and capacity == token["capacities"][index] and
                     generation == token["capacity_generations"][index] and rows <= capacity,
                     "EFM_CAPACITY_PLAN_STALE")
        _require(not self.retained or rows <= self.retention_max_rows, "EFM_RETENTION_ROWS_EXCEEDED")
        if self._calibration_options is not None and self.calibration_report is None:
            from hyff.multigpu.efm_retention import calibration_headroom
            _require(not self._calibration_started, "EFM_RETENTION_CALIBRATION_FAILED_NO_RETRY")
            self._calibration_started = True
            observed = calibration_headroom(meta.device, **self._calibration_options)
            report = self.calibrate_retained_tile(*values, selection=selection)
            report.update(observed)
            report["force_rows"] = rows
            report["tile_edges"] = meta.tile_edges
            report["calibration_budget"] = dict(self._calibration_options)
            if meta.device.type == "cuda":
                torch.cuda.synchronize(meta.device)
                report["allocator_peak_bytes"] = torch.cuda.max_memory_allocated(meta.device)
                report["reserved_peak_bytes"] = torch.cuda.max_memory_reserved(meta.device)
                report["peak_delta_upper_bytes"] = max(0, report["allocator_peak_bytes"] -
                                                       report["allocated_before_bytes"])
                _require(max(report["allocator_peak_bytes"], report["reserved_peak_bytes"]) <=
                         self._calibration_options["allocator_ceiling_bytes"],
                         "EFM_RETENTION_CALIBRATION_PEAK_EXCEEDED")
            self.calibration_report = report
        self._calls += 1
        retain_enabled = self.retained and torch.is_grad_enabled()
        if selection is not None:
            return self._aggregate.apply(*values, selection.source, selection.destination,
                                         selection.valid, meta.n_owned, retain_enabled,
                                         self._replay_storage_observer, self.retention_max_rows)
        return self._aggregate.apply(*values, meta.source, meta.destination, meta.valid, meta.n_owned,
                                     retain_enabled, self._replay_storage_observer, self.retention_max_rows)

    def evaluate(self, S, V, T, rbf, R, D, envelope, *, selection=None):
        """Aggregate plus exit modules; compiled first-use probes read host scalars."""
        scalar, vector, tensor = self.aggregate(S, V, T, rbf, R, D, envelope, selection=selection)
        return self.efm.exit_scalar(scalar), self.efm.exit_vector(vector), self.efm.exit_tensor(tensor)

    def validate_inputs(self, S, V, T, rbf, R, D, envelope):
        """Optional eager per-step finite guard, separate from graph evaluation."""
        _outside_capture(self.metadata.device)
        self._assert_current()
        meta = self.metadata
        with torch.no_grad():
            _input_shapes(self.efm, (S, V, T, rbf, R, D, envelope), meta.source,
                          meta.destination, meta.valid, meta.n_owned)

    def assert_replay_generation(self, generation):
        """Call before every replay, which cannot run Python generation guards."""
        _outside_capture(self.metadata.device)
        self._assert_current()
        _require(type(generation) is int and generation == self.metadata.generation,
                 "EFM_CAPTURE_GENERATION_STALE")

    def public_metadata(self):
        meta = self.metadata
        return {"schema": "LARGO_PREPARED_OWNED_EFM", "generation": meta.generation,
                "epoch": meta.epoch, "n_global": meta.n_global, "n_owned": meta.n_owned,
                "edge_capacity": meta.edge_capacity, "valid_edges": meta.valid_edges,
                "tile_edges": meta.tile_edges, "tile_edges_cap": MAX_TILE_EDGES,
                "block_policy": "ALL_ACTIVE_EFM_BLOCKS",
                "tile_policy": "FIXED_CONSTRUCTION_EPOCH_NO_OWNED_COUNT_THRESHOLD",
                "compile_tiles_requested": self.compile_requested, "compile_tiles": self.compiled,
                "backward_policy": "RETAINED_TILE_GRAPH" if self.retained else "EXPLICIT_TILE_REPLAY",
                "retention_max_rows": self.retention_max_rows,
                "retention_calibrated": self.calibration_report is not None,
                "compiler_internal_rematerialization_excluded": False,
                "compile_fallback": "KNOWN_SCALAR_SOURCE_BACKWARD_DEFECT" if
                    self.compile_requested and not self.compiled else None,
                "superseded": self._superseded, "fixed_metadata_buffers": True,
                "parameter_and_buffer_versions_guarded": True,
                "all_seven_cotangents": True, "native_edge_outputs_imported_unchanged": True,
                "evaluation_reads_tensor_values_on_host": self.compiled or self.retained,
                "first_use_parity_probe_reads_tensor_values_on_host": self.compiled or self.retained,
                "steady_state_after_probe_reads_tensor_values_on_host": False,
                "first_use_parity_probe_requires_nonzero_envelope": self.compiled or self.retained,
                "active_efm_compaction": self._active_selector is not None,
                "active_selection": (self._active_selector.public_metadata()
                                     if self._active_selector is not None else None),
                "capture_only": True, "cuda_capture_validated": False,
                "gpu_validated": False, "caller_must_check_generation_before_replay": True}


def prepare_owned_projected_efm(efm, values, source, destination, valid, *, owned_global_indices,
                              unit_shifts=None, tile_edges=MAX_TILE_EDGES, compile_tiles=False,
                              epoch=0, previous=None):
    """Eager control phase; input values are validated without an autograd tape.

    Buffers are cloned and fixed for the epoch. A successful replacement has a
    strictly newer epoch and invalidates the prior object, even if shapes match.
    Finish every prior forward/backward/replay before replacing its epoch.
    No collective agreement or CUDA capture is performed by this preparation.
    """
    _require(isinstance(values, (tuple, list)) and len(values) == 7 and
             isinstance(values[0], torch.Tensor), "EFM_PREPARED_SEVEN_INPUTS_REQUIRED")
    return _prepare_owned_projected_efm(efm, values, source, destination, valid,
        owned_global_indices=owned_global_indices, unit_shifts=unit_shifts,
        tile_edges=tile_edges, compile_tiles=compile_tiles, epoch=epoch, previous=previous,
        n_global=values[0].shape[0], dtype=values[0].dtype, device=values[0].device,
        factory_reuse=None)


def prepare_owned_projected_efm_metadata(efm, source, destination, valid, *, n_global,
                                       dtype, device, owned_global_indices, unit_shifts=None,
                                       tile_edges=MAX_TILE_EDGES, compile_tiles=False, epoch=0,
                                       factory_reuse=None):
    """Prepare topology before any physical node/geometry values exist.

    Runs the topology, ownership, parameter, device and variant checks of
    prepare_owned_projected_efm without allocating or scanning placeholder
    physical values. Actual inputs still pass the prepared evaluator's
    shape/version guards. No edge function, radial function or numerical
    policy changes here.
    """
    return _prepare_owned_projected_efm(efm, None, source, destination, valid,
        owned_global_indices=owned_global_indices, unit_shifts=unit_shifts,
        tile_edges=tile_edges, compile_tiles=compile_tiles, epoch=epoch, previous=None,
        n_global=n_global, dtype=dtype, device=torch.device(device), factory_reuse=factory_reuse)


def _prepare_owned_projected_efm(efm, values, source, destination, valid, *, owned_global_indices,
                               unit_shifts, tile_edges, compile_tiles, epoch, previous,
                               n_global, dtype, device, factory_reuse):
    _require(type(epoch) is int and epoch >= 0, "EFM_PREPARED_EPOCH_INVALID")
    _require(type(tile_edges) is int and 0 < tile_edges <= MAX_TILE_EDGES, "EFM_TILE_SIZE_INVALID")
    _require(type(compile_tiles) is bool, "EFM_COMPILE_FLAG_INVALID")
    _require(not any(parameter.requires_grad for parameter in efm.parameters()),
             "EFM_FROZEN_PARAMETERS_REQUIRED")
    _require(efm.aggregation == "sum" and efm.frame == "projector" and not efm.complex_structure and
             getattr(efm, "receiver_in", None) is None, "EFM_VARIANT_UNSUPPORTED")
    _require(type(efm.message_scale) in (int, float) and math.isfinite(efm.message_scale),
             "EFM_MESSAGE_SCALE_INVALID")
    _require(type(n_global) is int and n_global >= 0 and dtype in (torch.float32, torch.float64),
             "EFM_PREPARED_GLOBAL_METADATA_INVALID")
    _require(device.type in ("cpu", "cuda"), "EFM_CPU_OR_CUDA_REQUIRED")
    _outside_capture(device)
    _require(isinstance(owned_global_indices, torch.Tensor) and owned_global_indices.ndim == 1 and
             owned_global_indices.dtype == torch.int64 and owned_global_indices.device == device,
             "EFM_OWNER_MAP_INVALID")
    n_owned = owned_global_indices.numel()
    with torch.no_grad():
        if values is None:
            _metadata(source, destination, valid, n_global, n_owned, device)
            _require(all(parameter.device == device and parameter.dtype == dtype
                         for parameter in efm.parameters()), "EFM_PARAMETER_TYPE_INVALID")
        else:
            _input_shapes(efm, values, source, destination, valid, n_owned)
        _require(bool(((owned_global_indices >= 0) & (owned_global_indices < n_global)).all()) and
                 torch.unique(owned_global_indices).numel() == n_owned, "EFM_OWNER_MAP_INVALID")
        if unit_shifts is None:
            unit_shifts = torch.zeros((source.numel(), 3), dtype=torch.int64, device=device)
        _require(isinstance(unit_shifts, torch.Tensor) and unit_shifts.shape == (source.numel(), 3) and
                 unit_shifts.dtype == torch.int64 and unit_shifts.device == device,
                 "EFM_SHIFT_SHAPE_INVALID")
        source, destination, valid, owned, shifts = tuple(value.detach().clone().contiguous() for value in
            (source, destination, valid, owned_global_indices, unit_shifts))
        safe_source = torch.where(valid, source, torch.zeros_like(source))
        mapping = torch.cat((owned, owned.new_zeros(1)), 0)
        local = torch.where(valid, destination, torch.full_like(destination, n_owned))
        global_destination = mapping.index_select(0, local)
        valid_edges = int(valid.sum())
    generation = 1
    if previous is not None:
        _require(isinstance(previous, PreparedOwnedEFM) and previous.efm is efm and
                 not previous._superseded and epoch > previous.metadata.epoch,
                 "EFM_PREPARED_REPLACEMENT_INVALID")
        generation = previous.metadata.generation + 1
    metadata = PreparedOwnedEFMMetadata(n_global, n_owned, source.numel(), valid_edges, tile_edges,
                                       generation, epoch, dtype, device, source, destination, valid,
                                       owned, shifts, safe_source, global_destination)
    prepared = PreparedOwnedEFM(efm, metadata, compile_requested=compile_tiles,
                               factory_reuse=factory_reuse)
    if previous is not None:
        previous._superseded = True
    return prepared


@dataclass(frozen=True)
class PreparedOwnedEFMGeometry:
    operator: PreparedOwnedEFM
    centers: torch.Tensor
    gamma: torch.Tensor
    translation: torch.Tensor
    padding: torch.Tensor
    rbf_cutoff: float
    radius: float
    width: float
    buffer_stamps: tuple
    active_selector: object | None = None

    def select_active(self, positions, cache):
        """Caller owns one cache per force; all matching blocks share a view."""
        self.operator._assert_current()
        _require(self.active_selector is not None and isinstance(cache, dict),
                 "EFM_ACTIVE_SELECTOR_REQUIRED")
        key = id(self.active_selector)
        if key not in cache:
            cache[key] = self.active_selector.select(positions)
        selection = cache[key]
        selection.check(operator=self.operator, positions=positions, finalized=False)
        return selection

    def validate_force_inputs(self, positions, *, selection=None):
        """The same metadata/version guards for geometry cache hits and misses."""
        self.operator._assert_current()
        meta = self.operator.metadata
        _require(isinstance(positions, torch.Tensor) and positions.shape == (meta.n_global, 3) and
                 positions.device == meta.device and positions.dtype == meta.dtype,
                 "EFM_PREPARED_POSITION_METADATA_MISMATCH")
        _require(tuple(_tensor_stamp(value) for value in
                       (self.centers, self.gamma, self.translation, self.padding)) == self.buffer_stamps,
                 "EFM_PREPARED_GEOMETRY_CHANGED")
        if selection is not None:
            from hyff.multigpu.active_efm import ActiveEFMSelection
            _require(isinstance(selection, ActiveEFMSelection) and
                     selection.selector is self.active_selector, "EFM_ACTIVE_SELECTOR_MISMATCH")
            selection.check(operator=self.operator, positions=positions)
            source, destination, translation, valid = (selection.source, selection.global_destination,
                                                       selection.translation, selection.valid)
        else:
            _require(self.active_selector is None, "EFM_ACTIVE_SELECTION_REQUIRED")
            source, destination, translation, valid = (meta.safe_source, meta.global_destination,
                                                       self.translation, meta.valid)
        return source, destination, translation, valid

    def evaluate(self, positions, *, selection=None):
        """Global endpoint geometry; no value reads, unique, or boolean indexing."""
        source, destination, translation, valid = self.validate_force_inputs(positions, selection=selection)
        meta = self.operator.metadata
        readable = positions if meta.n_global else positions.new_zeros((1, 3))
        displacement = (readable.index_select(0, source) -
                        readable.index_select(0, destination) + translation)
        displacement = torch.where(valid[:, None], displacement, self.padding)
        distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-9)
        direction = displacement / distance[:, None]
        basis = torch.exp(-self.gamma * (distance[:, None] - self.centers[None, :]).square())
        radial_envelope = 0.5 * (torch.cos(math.pi * distance / self.rbf_cutoff) + 1.0)
        radial_envelope = radial_envelope * (distance < self.rbf_cutoff).to(distance.dtype)
        rbf = basis * radial_envelope[:, None]
        envelope = smooth_switch(distance, self.radius, self.width) * valid.to(distance.dtype)
        R, D = frame_matrices(direction)
        return rbf, R, D, envelope


def prepare_owned_efm_geometry(operator, cell, *, centers, gamma, rbf_cutoff, radius, width,
                               active_selector=None):
    """Eager fixed cell/radial constants; geometry does not differentiate cell."""
    _require(isinstance(operator, PreparedOwnedEFM), "EFM_PREPARED_OPERATOR_REQUIRED")
    meta = operator.metadata
    _outside_capture(meta.device)
    operator._assert_current()
    _require(all(type(value) in (int, float) and math.isfinite(value) and value > 0
                 for value in (rbf_cutoff, radius, width)) and width <= radius, "EFM_CUTOFF_INVALID")
    _require(isinstance(cell, torch.Tensor) and cell.shape == (3, 3) and
             cell.dtype == meta.dtype and cell.device == meta.device and not cell.requires_grad and
             isinstance(centers, torch.Tensor) and centers.shape == (operator.efm.radial[0].in_features,) and
             centers.dtype == meta.dtype and centers.device == meta.device and not centers.requires_grad,
             "EFM_GEOMETRY_TYPE_INVALID")
    _require(not isinstance(gamma, torch.Tensor) or (gamma.numel() == 1 and not gamma.requires_grad),
             "EFM_GAMMA_INVALID")
    with torch.no_grad():
        gamma = torch.as_tensor(gamma, dtype=meta.dtype, device=meta.device).reshape(()).detach().clone()
        _require(bool(torch.isfinite(cell).all() & torch.isfinite(centers).all() & torch.isfinite(gamma)) and
                 float(torch.linalg.det(cell).abs()) > 0 and float(gamma) > 0, "EFM_GEOMETRY_VALUE_INVALID")
        shifts = torch.where(meta.valid[:, None], meta.unit_shifts, torch.zeros_like(meta.unit_shifts))
        _require(bool((~meta.valid | (meta.safe_source != meta.global_destination) |
                       (shifts != 0).any(dim=-1)).all()), "EFM_ZERO_SHIFT_SELF_EDGE")
        translation = (shifts.to(meta.dtype) @ cell).detach().clone()
        centers = centers.detach().clone()
        padding = centers.new_tensor((1.0, 0.0, 0.0))
    buffers = (centers, gamma, translation, padding)
    if active_selector is not None:
        from hyff.multigpu.active_efm import PreparedActiveEFMSelector
        _require(isinstance(active_selector, PreparedActiveEFMSelector) and
                 active_selector.radius == float(radius), "EFM_ACTIVE_SELECTOR_MISMATCH")
        active_selector.bind(operator)
        _require(torch.equal(active_selector.translation, translation),
                 "EFM_ACTIVE_GEOMETRY_MISMATCH")
    return PreparedOwnedEFMGeometry(operator, *buffers, float(rbf_cutoff), float(radius), float(width),
                                    tuple(_tensor_stamp(value) for value in buffers), active_selector)


__all__ = ["PreparedOwnedEFMMetadata", "PreparedOwnedEFM", "PreparedOwnedEFMGeometry",
           "prepare_owned_projected_efm", "prepare_owned_projected_efm_metadata",
           "prepare_owned_efm_geometry"]
