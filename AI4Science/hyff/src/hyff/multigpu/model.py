"""One physical largo graph, composed from owned node/kernel/long-range components.

This is the tensor evaluator, not an MD driver. The caller collectively
prepares a fixed ownership/neighbor epoch, prepares the rank-zero LR plan
before every force call or graph replay, and rejects/grows/recaptures all
ranks together. This module never reports a globally summed differentiable
energy: every rank backpropagates its own energy with seed one.

"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math

import torch
from hyff.models.charge_bound import charge_bound_value, raw_charges

from hyff.multigpu import collectives as raw
from hyff.multigpu import ordered_collectives as ordered
from hyff.multigpu import ordered_kernels as boundary
from hyff.multigpu.base_messages import PreparedBaseOperator
from hyff.multigpu.pair_energy import OwnedPairOperator
from hyff.multigpu.electronic_state import OwnedPreparedState, _check_prepared
from hyff.multigpu.node_pipeline import (
    EFMChannelAggregates, OwnedNodeSequence, prepare_owned_node_pipeline, _execution_signature,
)
from hyff.multigpu.long_range import RankZeroLongRange
from hyff.multigpu import active_base_pair
from hyff.multigpu.shared_efm_geometry import prepare_shared_efm_geometry


class OwnedModelError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise OwnedModelError(code)


@dataclass(frozen=True)
class LayerInputs:
    base_operator: PreparedBaseOperator
    efm_operator: object | None = None
    efm_geometry: object | None = None


@dataclass(frozen=True)
class PreparedLayer:
    inputs: LayerInputs
    packet: ordered.ProjectedPacketSpec
    base: boundary.OrderedBaseSpec
    first_global_caches: tuple | None
    first_owned_caches: tuple | None


@dataclass(frozen=True)
class OwnedEnergyResult:
    local_energy: torch.Tensor
    local_short_range_energy: torch.Tensor
    root_long_range_energy: torch.Tensor
    local_raw_charge: torch.Tensor
    local_conserved_charge: torch.Tensor
    global_conserved_charge: torch.Tensor
    active_base_pair_numeric_status: torch.Tensor | None = None


@dataclass(frozen=True)
class EpochIndependentModelReuse:
    """Prepared model policy and constant buffers, without old edge objects."""
    model: object
    runtime: object
    nodes: object
    positions: object
    packets: tuple
    base_policies: tuple
    pair_policy: tuple
    charge: object
    charge_gather: object
    first_global_caches: tuple
    first_owned_caches: tuple
    initial_vector: object
    initial_tensor: object
    empty: object
    radial_stamp: tuple
    charge_head_stamp: tuple
    buffer_stamps: tuple

    def assert_integrity(self, model):
        current = (*self.first_global_caches, *self.first_owned_caches,
                   self.initial_vector, self.initial_tensor, self.empty,
                   self.charge.spec.fixed_total_charge)
        _require(len(current) == len(self.buffer_stamps) and
                 all(value is original for value, (original, _version) in zip(current, self.buffer_stamps)),
                 "MODEL_REUSE_BUFFER_REPLACED")
        _require(model is self.model and _execution_signature(model) == self.nodes.execution_signature and
                 _model_radial_stamp(model) == self.radial_stamp and
                 (id(model.latent_charge_energy.charge_head), charge_bound_value(model)) == self.charge_head_stamp and
                 all(value._version == version for value, version in self.buffer_stamps),
                 "MODEL_REUSE_INVARIANT_CHANGED")
        _check_prepared(model, self.nodes.state)


def _rebind_ordered(value, runtime):
    return replace(value, spec=replace(value.spec, runtime=runtime))


def _rebind_packet(value, runtime):
    return replace(value, spec=replace(value.spec, runtime=runtime),
                   reverse_spec=(replace(value.reverse_spec, runtime=runtime)
                                 if value.reverse_spec is not None else None))


def _same_epoch(runtime, graph, cell):
    _require(graph.metadata == runtime.metadata and graph.rank == runtime.rank and
             graph.cell.device == runtime.device and torch.equal(graph.cell, cell),
             "MODEL_COMPONENT_EPOCH_MISMATCH")


def _model_radial_stamp(model):
    return (tuple((id(table), float(table.cutoff), id(table.coefficients), table.coefficients._version)
                  for table in model.radial_tables),
            float(model.rbf.cutoff), id(model.rbf.centers), model.rbf.centers._version,
            id(model.rbf.gamma), model.rbf.gamma._version,
            tuple(float(value) for value in model.layer_cutoffs), float(model.config.switch_width))


def _prepare_active_efm_selections(runtime, geometries, global_positions, *, base_pair_refresh=None):
    """Finalize one force-local cache with bounded collective growth recovery.

    Both agreements precede every block collective. Only status is reduced;
    active counts and reserve growth remain rank-local. Recovery reuses the
    same gathered positions and retries selection once on every rank. The
    second agreement also covers local selection/finalization exceptions on
    the nonoverflow path. Fatal CUDA/communicator failures are not recoverable.
    """
    from hyff.multigpu.active_efm import _stamp, active_selection_control, finalize_active_selections
    raw._outside_capture(runtime.device)
    # Allocate the small failure carrier before any selection allocation, so
    # a recoverable selection OOM does not need another tensor allocation.
    status = torch.zeros(2, dtype=torch.int64, device=runtime.device)
    control = status
    selections = {}
    position_stamp = None
    base_pair_force = None
    try:
        position_stamp = _stamp(global_positions)
        if base_pair_refresh is not None:
            base_pair_force = base_pair_refresh(global_positions)
        for geometry in geometries:
            geometry.select_active(global_positions, selections)
        initial_status = base_pair_force.status if base_pair_force is not None else status[:1]
        control = torch.cat((initial_status, active_selection_control(selections)))
    except Exception:
        status[0] = 1
        control = status
    # [initial failure, overflow, local counts...]; counts are never reduced.
    raw._reduce(control[:2], runtime, raw.dist.ReduceOp.MAX)
    failure = 0
    counts = None
    try:
        observed = control.tolist()
        if observed[0]:
            failure = 1
        else:
            _require(len(observed) == len(selections) + 2, "EFM_ACTIVE_CONTROL_COUNTS_INVALID")
            counts = observed[2:]
    except Exception:
        failure = 1
    if not failure and observed[1]:
        try:
            for selector, count in zip((value.selector for value in selections.values()), counts):
                if count > selector.capacity:
                    selector.reserve_at_least(count)
        except Exception:
            failure = 2
        # Every successful rank discards even a nonoverflowing old cache.
        # No old short allocation may be finalized after reserve growth.
        selections = {}
        if not failure:
            try:
                _require(_stamp(global_positions) == position_stamp, "EFM_ACTIVE_FRAME_CHANGED")
                for geometry in geometries:
                    geometry.select_active(global_positions, selections)
                retry = active_selection_control(selections).tolist()
                _require(len(retry) == len(selections) + 1, "EFM_ACTIVE_CONTROL_COUNTS_INVALID")
                if retry[0]:
                    failure = 4
                else:
                    counts = retry[1:]
            except Exception:
                failure = 3
    if not failure:
        try:
            _require(_stamp(global_positions) == position_stamp, "EFM_ACTIVE_FRAME_CHANGED")
            finalize_active_selections(selections, counts)
            if base_pair_force is not None:
                base_pair_force.check(global_positions)
            _require(_stamp(global_positions) == position_stamp, "EFM_ACTIVE_FRAME_CHANGED")
        except Exception:
            failure = 5
    # No rank can enter a block while a peer failed growth, reselection, or
    # finalization. MAX gives all ranks the same bounded failure category.
    status[0] = failure
    raw._reduce(status[:1], runtime, raw.dist.ReduceOp.MAX)
    failure = int(status[0].item())
    codes = {1: "EFM_ACTIVE_INITIAL_SELECTION_FAILED",
             2: "EFM_ACTIVE_GROWTH_FAILED",
             3: "EFM_ACTIVE_RESELECTION_FAILED",
             4: "EFM_ACTIVE_OVERFLOW_AFTER_GROWTH",
             5: "EFM_ACTIVE_FINALIZATION_FAILED"}
    _require(failure == 0, codes.get(failure, "EFM_ACTIVE_CONTROL_FAILED"))
    return (selections, base_pair_force) if base_pair_refresh is not None else selections


class PreparedOwnedModel:
    """Fixed model/ownership epoch; evaluate with owned FP32 position leaves.

    Native base preparation (precision/modules/radial tables) precedes this
    constructor. No single-GPU EFM or local latent-charge forward is installed here.
    The real RankZeroLongRange is required only on rank zero. Its route
    (resolved for the global atom count) and backend are fixed at construction;
    other ranks have no LR evaluator.
    """
    def __init__(self, model, state, layers, pair_operator, root_long_range, *, total_charge, trace=None,
                 reuse=None, active_base_pair_compaction=False, share_efm_geometry=False):
        _require(isinstance(state, OwnedPreparedState), "MODEL_PREPARED_STATE_REQUIRED")
        _check_prepared(model, state)
        runtime = state.runtime
        if reuse is not None:
            _require(isinstance(reuse, EpochIndependentModelReuse), "MODEL_REUSE_CAPSULE_REQUIRED")
            reuse.assert_integrity(model)
            _require(reuse.runtime.device == runtime.device and reuse.runtime.rank == runtime.rank and
                     reuse.runtime.group is runtime.group and reuse.runtime.backend == runtime.backend and
                     reuse.runtime.metadata.global_ids == runtime.metadata.global_ids and
                     reuse.runtime.world_size == runtime.world_size and reuse.positions.trace is trace,
                     "MODEL_REUSE_RUNTIME_IDENTITY_CHANGED")
        reuse_shape = reuse is not None and reuse.runtime.metadata.counts == runtime.metadata.counts
        raw._outside_capture(runtime.device)
        _require(state.initial_scalar.dtype == torch.float32 and runtime.n_global > 0,
                 "MODEL_FP32_GLOBAL_GRAPH_REQUIRED")
        _require(all(not value.requires_grad for value in model.parameters()),
                 "MODEL_FROZEN_WEIGHTS_REQUIRED")
        _require(hasattr(model, "latent_charge_energy"), "MODEL_LATENT_CHARGE_TERM_REQUIRED")
        _require(len(layers) == len(model.blocks) and
                 all(isinstance(value, LayerInputs) for value in layers), "MODEL_LAYER_COUNT_MISMATCH")
        _require(type(total_charge) in (int, float) and math.isfinite(total_charge) and
                 torch.equal(state.totals[0, 0], state.totals.new_tensor(total_charge)), "MODEL_TOTAL_CHARGE_MISMATCH")
        if runtime.rank == 0:
            _require(isinstance(root_long_range, RankZeroLongRange) and
                     root_long_range.term is model.latent_charge_energy and
                     root_long_range.n_global == runtime.n_global,
                     "MODEL_ROOT_LR_REQUIRED")
        else:
            _require(root_long_range is None, "MODEL_NONROOT_LR_FORBIDDEN")
        _require(isinstance(pair_operator, OwnedPairOperator), "MODEL_PAIR_OPERATOR_REQUIRED")
        _same_epoch(runtime, pair_operator.graph, pair_operator.graph.cell)
        _require(pair_operator.n_elements == model.n_elements and
                 torch.equal(pair_operator.elements.index_select(0, runtime.local_indices),
                             model.element_lut[state.owned_atomic_numbers]),
                 "MODEL_PAIR_SPECIES_MISMATCH")
        raw._agreement(runtime, {"component": "model_pair_species", "sha256": hashlib.sha256(
            pair_operator.elements.detach().cpu().numpy().tobytes()).hexdigest()})
        _require(pair_operator.cutoff == float(model.cutoff) and
                 pair_operator.switch_width == float(model.config.switch_width), "MODEL_PAIR_CUTOFF_MISMATCH")
        tables = getattr(model, "radial_tables", None)
        _require(tables is not None and len(tables) == len(model.blocks) + 1,
                 "MODEL_NATIVE_RADIAL_TABLES_REQUIRED")
        _require(float(tables[-1].cutoff) == pair_operator.cutoff and
                 torch.equal(tables[-1].coefficients, pair_operator.coefficients),
                 "MODEL_PAIR_TABLE_MISMATCH")
        prepared_buffers = [pair_operator.elements, pair_operator.coefficients]
        epoch_graphs = [pair_operator.graph]


        self.model, self.state, self.runtime = model, state, runtime
        self.root_long_range, self.pair_operator = root_long_range, pair_operator
        self.trace = trace
        if reuse_shape:
            _require(reuse.positions.physical_gradient is True and
                     reuse.positions.spec.contract == raw.PARTIAL_SOURCE_SUM and
                     reuse.positions.spec.tail_shape == (3,) and reuse.positions.spec.dtype == torch.float32 and
                     reuse.positions.label == "model_positions", "MODEL_REUSE_POSITION_POLICY_CHANGED")
            self.positions = _rebind_ordered(reuse.positions, runtime)
        else:
            position_spec = raw.prepare_exchange(runtime, (3,), torch.float32, raw.PARTIAL_SOURCE_SUM)
            self.positions = ordered.prepare_ordered_exchange(position_spec, physical_gradient=True,
                                                              label="model_positions", trace=trace)
        self.nodes = prepare_owned_node_pipeline(model, state,
                     aee_physical_gradients=(True,) * len(model.blocks), trace=trace,
                     reuse=reuse.nodes if reuse_shape else None)
        prepared_layers = []
        for index, (block, inputs) in enumerate(zip(model.blocks, layers)):
            op = inputs.base_operator
            _require(isinstance(op, PreparedBaseOperator), "MODEL_BASE_OPERATOR_REQUIRED")
            _same_epoch(runtime, op.graph, pair_operator.graph.cell)
            epoch_graphs.append(op.graph)
            prepared_buffers.append(op.coefficients)
            _require(float(tables[index].cutoff) == op.cutoff and
                     torch.equal(tables[index].coefficients, op.coefficients), "MODEL_BASE_TABLE_MISMATCH")
            _require(block.message_hidden == 84 and op.first == (index == 0) and
                     op.cutoff == float(model.layer_cutoffs[index]) and
                     op.switch_width == float(model.config.switch_width), "MODEL_BASE_LAYOUT_MISMATCH")
            first = index == 0
            base_shapes = ((252,),) if first else ((588,), (3, 84), (5, 84))
            efm = getattr(block, "efm", None)
            if efm is None:
                _require(inputs.efm_operator is inputs.efm_geometry is None,
                         "MODEL_UNEXPECTED_EFM_COMPONENT")
                efm_shapes = ()
            else:
                prepared_efm, geometry = inputs.efm_operator, inputs.efm_geometry
                _require(prepared_efm is not None and prepared_efm.efm is efm and
                         geometry is not None and geometry.operator is prepared_efm,
                         "MODEL_EFM_COMPONENT_REQUIRED")
                meta = prepared_efm.metadata
                _require(meta.n_global == runtime.n_global and meta.n_owned == runtime.n_local and
                         meta.device == runtime.device and meta.dtype == torch.float32 and
                         torch.equal(meta.owned_global_indices, runtime.local_indices),
                         "MODEL_EFM_EPOCH_MISMATCH")
                _require(geometry.radius == float(efm.__dict__.get("edge_cutoff", model.layer_cutoffs[index])) and
                         geometry.width == float(model.config.switch_width) and
                         geometry.rbf_cutoff == float(model.rbf.cutoff) and
                         torch.equal(geometry.centers, model.rbf.centers) and
                         torch.equal(geometry.gamma, model.rbf.gamma.reshape(())),
                         "MODEL_EFM_RADIAL_MISMATCH")
                shifts = torch.where(meta.valid[:, None], meta.unit_shifts, torch.zeros_like(meta.unit_shifts))
                _require(torch.equal(geometry.translation, shifts.to(meta.dtype) @ pair_operator.graph.cell),
                         "MODEL_EFM_CELL_MISMATCH")
                prepared_buffers.extend((geometry.centers, geometry.gamma, geometry.translation, geometry.padding))
                prepared_buffers.extend(meta.buffers())
                efm_shapes = (((efm.channels,), (3, efm.channels), (5, efm.channels))
                              if efm.m_blocks else ((efm.channels,),))
            if reuse_shape:
                packet = _rebind_packet(reuse.packets[index], runtime)
                _require(packet.base_shapes == base_shapes and packet.efm_shapes == efm_shapes and
                         packet.efm_gradient_flags == (not first,) * len(efm_shapes) and
                         packet.spec.contract == raw.DETACHED_SOURCE and packet.spec.dtype == torch.float32 and
                         packet.spec.tail_shape == (sum(math.prod(shape) for shape in base_shapes + efm_shapes),) and
                         packet.label == "model_sources_" + str(index) and packet.trace is trace,
                         "MODEL_REUSE_PACKET_POLICY_CHANGED")
                reverse_width = sum(math.prod(shape) for shape, flag in zip(efm_shapes, packet.efm_gradient_flags) if flag)
                _require((packet.reverse_spec is None if reverse_width == 0 else
                         packet.reverse_spec is not None and
                         packet.reverse_spec.contract == raw.PARTIAL_SOURCE_SUM and
                         packet.reverse_spec.tail_shape == (reverse_width,) and
                         packet.reverse_spec.dtype == torch.float32), "MODEL_REUSE_PACKET_POLICY_CHANGED")
                flags, label, base_trace = reuse.base_policies[index]
                _require(flags == (True, not first, not first, not first) and
                         label == "model_base_" + str(index) and base_trace is trace,
                         "MODEL_REUSE_BASE_POLICY_CHANGED")
                base_spec = boundary.OrderedBaseSpec(op, flags, label, base_trace)
            else:
                packet = ordered.prepare_projected_packet(runtime, base_shapes, efm_shapes, torch.float32,
                    efm_gradient_flags=(not first,) * len(efm_shapes), label="model_sources_" + str(index), trace=trace)
                base_spec = boundary.prepare_ordered_base(op,
                    physical_gradient_flags=(True, not first, not first, not first),
                    label="model_base_" + str(index), trace=trace)
            global_caches = ((reuse.first_global_caches if reuse is not None else
                              boundary.prepare_first_base_caches(op)) if first else None)
            local_caches = ((reuse.first_owned_caches if reuse_shape else
                             tuple(state.initial_scalar.new_zeros((runtime.n_local, components, 84))
                                   for components in (3, 5))) if first else None)
            prepared_layers.append(PreparedLayer(inputs, packet, base_spec, global_caches, local_caches))
        for graph in epoch_graphs:
            prepared_buffers.extend((graph.local_rows, graph.neighbors, graph.shifts, graph.counts, graph.cell))
        self._model_radial_stamp = _model_radial_stamp(model)
        self._charge_head_stamp = (id(model.latent_charge_energy.charge_head), charge_bound_value(model))
        self.layers = tuple(prepared_layers)
        self._active_efm_geometries = tuple(value.inputs.efm_geometry for value in self.layers
            if getattr(value.inputs.efm_geometry, 'active_selector', None) is not None)
        _require(type(share_efm_geometry) is bool and
                 (not share_efm_geometry or bool(self._active_efm_geometries)),
                 "MODEL_SHARED_EFM_GEOMETRY_REQUIRES_ACTIVE_EFM")
        self._share_efm_geometry = share_efm_geometry
        self._shared_efm_geometry = (prepare_shared_efm_geometry(self._active_efm_geometries)
                                     if share_efm_geometry else None)
        self._last_shared_efm_geometry_metadata = None
        _require(type(active_base_pair_compaction) is bool and
                 (not active_base_pair_compaction or bool(self._active_efm_geometries)),
                 "MODEL_ACTIVE_BASE_PAIR_REQUIRES_ACTIVE_EFM")
        self._active_base_pair_prepared = active_base_pair_compaction
        self._active_base_pair_enabled = active_base_pair_compaction
        self._active_base_pair_generation = 0
        if reuse_shape:
            _require(reuse.pair_policy == ("model_pair", trace) and
                     reuse.charge.physical_gradient is True and
                     reuse.charge.spec.contract == raw.GLOBAL_CHARGE and
                     reuse.charge.spec.tail_shape == () and reuse.charge.spec.dtype == torch.float32 and
                     reuse.charge.label == "model_charge" and reuse.charge.trace is trace and
                     torch.equal(reuse.charge.spec.fixed_total_charge,
                                 reuse.charge.spec.fixed_total_charge.new_tensor(total_charge)) and
                     reuse.charge_gather.physical_gradient is True and
                     reuse.charge_gather.spec.contract == raw.PARTIAL_SOURCE_SUM and
                     reuse.charge_gather.spec.tail_shape == () and reuse.charge_gather.spec.dtype == torch.float32 and
                     reuse.charge_gather.label == "model_q_gather" and reuse.charge_gather.trace is trace,
                     "MODEL_REUSE_CHARGE_PAIR_POLICY_CHANGED")
            self.pair = boundary.OrderedPairSpec(pair_operator, *reuse.pair_policy)
            self.charge = _rebind_ordered(reuse.charge, runtime)
            self.charge_gather = _rebind_ordered(reuse.charge_gather, runtime)
            self._empty = reuse.empty
            self._initial_vector, self._initial_tensor = reuse.initial_vector, reuse.initial_tensor
        else:
            self.pair = boundary.prepare_ordered_pair(pair_operator, label="model_pair", trace=trace)
            charge_spec = raw.prepare_exchange(runtime, (), torch.float32, raw.GLOBAL_CHARGE,
                                               total_charge=total_charge)
            self.charge = ordered.prepare_ordered_exchange(charge_spec, physical_gradient=True,
                                                           label="model_charge", trace=trace)
            charge_gather = raw.prepare_exchange(runtime, (), torch.float32, raw.PARTIAL_SOURCE_SUM)
            self.charge_gather = ordered.prepare_ordered_exchange(charge_gather, physical_gradient=True,
                                                                 label="model_q_gather", trace=trace)
            self._empty = state.initial_scalar.new_empty((0,))
            self._initial_vector = state.initial_scalar.new_zeros((runtime.n_local, 3, model.tensor_hidden))
            self._initial_tensor = state.initial_scalar.new_zeros((runtime.n_local, 5, model.tensor_hidden))
        constants = (*self.layers[0].first_global_caches, *self.layers[0].first_owned_caches,
                     self._initial_vector, self._initial_tensor, self._empty,
                     self.charge.spec.fixed_total_charge)
        self._constant_buffer_stamps = tuple((value, value._version) for value in constants)
        self._prepared_buffer_stamps = tuple((value, value._version) for value in prepared_buffers) + self._constant_buffer_stamps

    def epoch_independent_reuse(self):
        """Extract before retirement; no old topology or autograd graph leaks."""
        _require(all(value._version == version for value, version in self._prepared_buffer_stamps),
                 "MODEL_COMPONENT_MUTATED_REPREPARE")
        result = EpochIndependentModelReuse(self.model, self.runtime, self.nodes, self.positions,
            tuple(value.packet for value in self.layers),
            tuple((value.base.physical_gradient_flags, value.base.label, value.base.trace)
                  for value in self.layers),
            (self.pair.label, self.pair.trace), self.charge, self.charge_gather,
            self.layers[0].first_global_caches, self.layers[0].first_owned_caches,
            self._initial_vector, self._initial_tensor, self._empty,
            self._model_radial_stamp, self._charge_head_stamp, self._constant_buffer_stamps)
        result.assert_integrity(self.model)
        return result

    def set_active_base_pair_enabled(self, enabled):
        """Single-rank toggle for comparison with the full skin path.

        With several ranks the policy is fixed when the model is prepared and
        agreed by all ranks. Disabling uses the full skin graph. Existing
        autograd tapes keep their already selected operators after a toggle.
        """
        _require(self.runtime.world_size == 1 and self._active_base_pair_prepared and
                 type(enabled) is bool, "MODEL_ACTIVE_BASE_PAIR_SINGLE_RANK_TOGGLE_REQUIRED")
        self._active_base_pair_enabled = enabled

    def evaluate(self, owned_positions):
        """Build one ordered energy graph; caller backpropagates seed one.

        Prepare/validate LR and neighbors collectively before entering. Active-edge compaction
        performs two status agreements before block collectives and can
        grow/reselect once using the same gathered positions;
        the compaction-off path has no tensor-value host reads here.
        Do not differentiate a replicated all-reduced copy of local_energy.
        """
        _require(_model_radial_stamp(self.model) == self._model_radial_stamp and
                 (id(self.model.latent_charge_energy.charge_head), charge_bound_value(self.model)) == self._charge_head_stamp and
                 all(value._version == version for value, version in self._prepared_buffer_stamps),
                 "MODEL_COMPONENT_MUTATED_REPREPARE")
        _require(isinstance(owned_positions, torch.Tensor) and owned_positions.requires_grad and owned_positions.is_leaf and
                 owned_positions.shape == (self.runtime.n_local, 3) and
                 owned_positions.dtype == torch.float32 and owned_positions.device == self.runtime.device,
                 "MODEL_OWNED_POSITION_LEAF_REQUIRED")
        control = ordered.seed_control(owned_positions, trace=self.trace)
        positions, control = boundary.exchange_positions(owned_positions, control, self.positions)
        layers, pair_operator, pair = self.layers, self.pair_operator, self.pair
        numeric_status = None
        if self._active_base_pair_enabled:
            self._active_base_pair_generation += 1
            def refresh(global_positions):
                return active_base_pair.refresh_active_base_pair(self.layers, self.pair_operator,
                    self.pair, global_positions, generation=self._active_base_pair_generation)
            selections, active_force = _prepare_active_efm_selections(self.runtime,
                self._active_efm_geometries, positions.global_positions, base_pair_refresh=refresh)
            layers, pair_operator, pair = active_force.layers, active_force.pair_operator, active_force.pair
            numeric_status = active_force.numeric_status
        else:
            selections = (_prepare_active_efm_selections(self.runtime, self._active_efm_geometries,
                          positions.global_positions) if self._active_efm_geometries else None)
        capacity_retention = getattr(self, "capacity_retention", None)
        if capacity_retention is not None:
            # Includes logical growth inside an existing selector reserve.
            # Every rank agrees before any block projection/packet begins.
            capacity_retention.preflight(selections)
        geometry_cache = (self._shared_efm_geometry.new_force(positions.global_positions)
                          if self._share_efm_geometry else None)
        values = (self.state.initial_scalar, self._initial_vector, self._initial_tensor)
        with OwnedNodeSequence(self.nodes) as nodes:
            for index, layer in enumerate(layers):
                projection = nodes.project_owned_sources(index, *values)
                if numeric_status is not None:
                    active_base_pair.guard_active_base_pair_numeric(numeric_status, projection.base_fields)
                packet = ordered.ordered_projected_packet(projection.base_fields, projection.efm_fields,
                                                           control, layer.packet)
                base_fields = ((projection.base_message, *layer.first_owned_caches)
                               if index == 0 else projection.base_fields)
                snapshot = boundary.base_snapshot(positions, base_fields, packet, layer.packet,
                    layer.inputs.base_operator, first_caches=layer.first_global_caches)
                base_aggregates, control = boundary.ordered_base_messages(snapshot, packet.control, layer.base)
                if numeric_status is not None:
                    active_base_pair.guard_active_base_pair_numeric(numeric_status, cotangents=base_aggregates)
                efm_aggregates = None
                if layer.inputs.efm_operator is not None:
                    fields = packet.efm
                    S, V, T = fields if len(fields) == 3 else (fields[0], self._empty, self._empty)
                    prepared_geometry = layer.inputs.efm_geometry
                    if getattr(prepared_geometry, 'active_selector', None) is None:
                        geometry = prepared_geometry.evaluate(positions.global_positions)
                        efm_aggregates = EFMChannelAggregates(*layer.inputs.efm_operator.aggregate(S, V, T, *geometry))
                    else:
                        selection = prepared_geometry.select_active(positions.global_positions, selections)
                        if geometry_cache is None:
                            geometry = prepared_geometry.evaluate(positions.global_positions, selection=selection)
                        else:
                            geometry = geometry_cache.evaluate(prepared_geometry, positions.global_positions,
                                                               selection=selection)
                        efm_aggregates = EFMChannelAggregates(*layer.inputs.efm_operator.aggregate(
                            S, V, T, *geometry, selection=selection))
                output = nodes.finish_block(index, values, base_aggregates=base_aggregates,
                                            efm_aggregates=efm_aggregates, control=control)
                values, control = (output.scalar, output.vector, output.tensor), output.control
            node_result = nodes.finish()
        pair_snapshot = boundary.pair_snapshot(positions, pair_operator)
        atomic_pair, control = boundary.ordered_pair_energy(pair_snapshot, node_result.control, pair)
        if numeric_status is not None:
            active_base_pair.guard_active_base_pair_numeric(numeric_status, cotangents=(atomic_pair,))
        # Native stable_sum promotes node energy BEFORE adding atomic pair.
        short_range = (node_result.atomic_energy.double() + atomic_pair).sum()
        with torch.autocast(device_type=owned_positions.device.type, enabled=False):
            raw_charge = raw_charges(self.model.latent_charge_energy, node_result.final_scalar)
        charge, control = ordered.ordered_global_charge(raw_charge, control, self.charge)
        global_charge, control = ordered.ordered_partial_gather(charge, control, self.charge_gather)
        if self.runtime.rank == 0:
            lr = self.root_long_range.evaluate(positions.global_positions, global_charge)
        else:
            # The position-connected control chain keeps both gather transposes
            # scheduled with zero local LR cotangents; no arithmetic 0*x anchor.
            lr = owned_positions.new_zeros((1,))
        energy = ordered.ordered_energy_anchor(short_range + lr.double().sum(), control,
                                               label="model_energy", trace=self.trace)
        self._last_shared_efm_geometry_metadata = geometry_cache.metadata() if geometry_cache is not None else None
        return OwnedEnergyResult(energy, short_range, lr, raw_charge, charge, global_charge, numeric_status)

    def energy_and_force(self, owned_positions):
        capacity_retention = getattr(self, "capacity_retention", None)
        try:
            if capacity_retention is not None:
                capacity_retention.begin_force()
            result = self.evaluate(owned_positions)
            force = -torch.autograd.grad(result.local_energy, owned_positions, create_graph=False)[0]
            if result.active_base_pair_numeric_status is not None:
                force = active_base_pair.guard_active_base_pair_numeric(
                    result.active_base_pair_numeric_status, force=force)
            if capacity_retention is not None:
                capacity_retention.finish_force()
            return result, force
        except Exception:
            # An in-flight failure is terminal to the supervised rank group.
            # No recovery collective, new force, or graph truncation is tried.
            if capacity_retention is not None:
                capacity_retention.retire("FORCE_FAILED")
            raise

    def metadata(self):
        raw._outside_capture(self.runtime.device)
        return {"schema": "LARGO_OWNED_MODEL_COMPOSITION", "global_atoms": self.runtime.n_global,
                "capacity_retention": (self.capacity_retention.public_metadata()
                    if getattr(self, "capacity_retention", None) is not None else None),
                "owned_atoms": self.runtime.n_local, "owned_counts": list(self.runtime.metadata.counts),
                "owned_count_sum_equals_global": sum(self.runtime.metadata.counts) == self.runtime.n_global,
                "source_packet_widths": [value.packet.spec.tail_shape[0] for value in self.layers],
                "base_adjoint_rule": "DISJOINT_ALL_GATHER_COMPLETE_OWNED_RETURN",
                "efm_and_lr_source_adjoint_rule": "SUM_TO_OWNER",
                "local_energy_seed_per_rank": 1, "reporting_reduction_in_graph": False,
                "native_stable_sum_promotion_order": True, "root_lr_only": True,
                "charge_bound": charge_bound_value(self.model),
                "topology": "CALLER_PREPARED_FIXED_EPOCH",
                "active_efm_compaction": bool(self._active_efm_geometries),
                "share_efm_geometry": self._share_efm_geometry,
                "shared_efm_geometry": self._shared_efm_geometry.metadata() if self._shared_efm_geometry is not None else None,
                "last_shared_efm_geometry": self._last_shared_efm_geometry_metadata,
                "active_base_pair_prepared": self._active_base_pair_prepared,
                "active_base_pair_compaction": self._active_base_pair_enabled,
                "active_base_pair_lifetime": "FRESH_FORCE_LOCAL_BUFFERS_RETAINED_BY_BACKWARD",
                "active_base_pair_control": "ACTIVE_EFM_INITIAL_STATUS_NO_ADDED_AGREEMENT",
                "active_base_pair_numeric_guard": "DEVICE_SOURCE_AND_COTANGENT_FINITE_FLAGS_THEN_FINAL_FORCE_POISON",
                "active_base_pair_numeric_rejection": "STEP_OUTPUT_FINITE_CHECK_AND_MAX_AGREEMENT",
                "active_efm_selection_lifetime": "ONE_FORCE_CALL",
                "active_efm_overflow_policy": "COLLECTIVE_LOCAL_GROW_RESELECT_ONCE_BEFORE_BLOCKS",
                "active_efm_control_agreements_per_force": 2 if self._active_efm_geometries else 0,
                "neighbor_and_lr_consensus_performed_here": False,
                "capture_validated": False, "full_model_validated": False,
                "gpu_validated": False, "performance_measured": False}


__all__ = ["OwnedModelError", "LayerInputs", "PreparedLayer", "OwnedEnergyResult", "PreparedOwnedModel"]
