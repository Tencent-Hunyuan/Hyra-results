"""Eager assembly of a fixed owned largo model/neighbor epoch.

The input is the frozen FP32 C84 native model after bind_base. This builder
does not bind or mutate it, select a device, create a process group, evaluate
energy, or prepare an LR plan. All members of the supplied existing group
must call this preparation in the same order. The caller is responsible for
collective adoption/retirement, each per-step neighbor/LR decision, and
capture/replay.

EFM preparation validates topology directly, without shaped zero values. It never
evaluates a full-edge radial geometry or a native/compiled EFM edge function.
Each epoch keeps a fixed skin-superset graph. GPU cutoff rounding, CUDA graph
capture and performance are not validated here.
Module import is stdlib-only; numerical imports occur inside preparation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent


class OwnedModelPreparationError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise OwnedModelPreparationError(code)



def required_owned_operator_radii(model):
    """One work weight per base block, active EFM block, and pair operator.

    The returned tuple intentionally keeps repeated cutoffs. Callers may use
    it when preparing the CPU neighbor epoch before device/runtime assembly.
    """
    _require(len(model.blocks) > 0 and len(model.layer_cutoffs) == len(model.blocks),
             "MODEL_BUILD_LAYER_LAYOUT_INVALID")
    base = tuple(float(value) for value in model.layer_cutoffs)
    efm = tuple(float(block.efm.__dict__.get("edge_cutoff", cutoff))
                for block, cutoff in zip(model.blocks, base) if getattr(block, "efm", None) is not None)
    radii = (*base, *efm, float(model.cutoff))
    _require(all(math.isfinite(value) and value > 0 for value in radii),
             "MODEL_BUILD_OPERATOR_RADIUS_INVALID")
    return radii


@dataclass(frozen=True)
class OwnedModelBundle:
    evaluator: object
    neighbor_epoch: object
    runtime: object
    state: object
    layers: tuple
    pair_operator: object
    root_long_range: object | None
    reference_global_positions: object
    canonical_global_ids: object
    global_atomic_numbers: object
    cell: object
    operator_radii: tuple[float, ...]
    root_lr_route: str
    tile_edges: int
    compile_efm_tiles: bool
    active_efm_compaction: bool
    active_base_pair_compaction: bool
    share_efm_geometry: bool
    capacity_retention: bool
    _versions: tuple
    _fixed_input_stamps: tuple
    _preparation_policy: tuple
    preparation_ms: dict

    def assert_integrity(self, *, generation=None):
        """Caller eager guard before use/replay; this does no consensus."""
        current = (self.reference_global_positions, self.cell, self.global_atomic_numbers,
                   self.canonical_global_ids, self.runtime.local_indices,
                   self.runtime.global_to_local, self.runtime.canonical_slots)
        _require(len(current) == len(self._versions) and
                 all(value is original for value, (original, _version) in zip(current, self._versions)),
                 "MODEL_BUILD_BUNDLE_BUFFER_REPLACED")
        self.neighbor_epoch.assert_generation(self.neighbor_epoch.generation if generation is None else generation)
        _require(all(value._version == version for value, version in self._versions),
                 "MODEL_BUILD_BUNDLE_BUFFER_CHANGED")

    def new_owned_position_leaf(self):
        """Independent initial-reference leaf; later MD positions are caller-owned."""
        self.assert_integrity()
        return self.reference_global_positions.index_select(0, self.runtime.local_indices).detach().clone().requires_grad_(True)

    def metadata(self):
        self.assert_integrity()
        selectors = {}
        for layer in self.layers:
            selector = getattr(layer.efm_geometry, "active_selector", None)
            if selector is not None:
                selectors[id(selector)] = selector
        return {"schema": "LARGO_OWNED_MODEL_BUILDER", "epoch": self.neighbor_epoch.epoch,
                "generation": self.neighbor_epoch.generation, "global_atoms": self.runtime.n_global,
                "owned_atoms": self.runtime.n_local, "owned_counts": list(self.runtime.metadata.counts),
                "operator_radii_A": list(self.operator_radii),
                "ownership_policy": "GREEDY_SUM_OF_REPEATED_OPERATOR_SKIN_DESTINATION_EDGE_COUNTS",
                "base_kernel_settings": list(self.layers[0].base_operator.settings),
                "pair_fma": self.pair_operator.fma, "efm_tile_edges": self.tile_edges,
                "efm_compile_requested": self.compile_efm_tiles,
                "active_efm_compaction": self.active_efm_compaction,
                "active_base_pair_compaction": self.active_base_pair_compaction,
                "share_efm_geometry": self.share_efm_geometry,
                "capacity_retention": self.capacity_retention,
                "active_efm_selectors": [value.public_metadata() for value in selectors.values()],
                "efm_compile_active": [value.efm_operator.compiled if value.efm_operator else False
                                       for value in self.layers],
                "efm_preparation_values": ("TOPOLOGY_AND_REFERENCE_ACTIVE_CAPACITY_COUNT"
                    if self.active_efm_compaction else "TOPOLOGY_ONLY_NO_PHYSICAL_VALUES"),
                "root_lr_requested": self.root_lr_route, "root_lr_present_here": self.root_long_range is not None,
                "root_lr_plan_prepared_here": False, "runtime_uses_existing_process_group": True,
                "model_bound_here": False, "model_mutated_here": False,
                "neighbor_policy": "FIXED_SKIN_SUPERSET_PER_EPOCH",
                "caller_owns_step_consensus_and_epoch_retirement": True,
                "gpu_cutoff_boundary_rounding_validated": False, "gpu_validated": False,
                "capture_validated": False, "full_model_validated": False,
                "native_compaction_performance_equivalence": False}

    def epoch_independent_reuse(self):
        """Detach only invariant products before retiring all old edge buffers."""
        self.assert_integrity()
        execution = self.evaluator.epoch_independent_reuse()
        efm_factories = tuple(layer.efm_operator.factory_reuse()
                              if layer.efm_operator is not None else None for layer in self.layers)
        buffers = (self.cell, self.global_atomic_numbers, self.canonical_global_ids,
                   self.pair_operator.elements, self.runtime.local_indices,
                   self.runtime.global_to_local, self.runtime.canonical_slots)
        return OwnedModelEpochReuse(self.runtime, self.state, self.root_long_range,
            self.cell, self.global_atomic_numbers, self.canonical_global_ids,
            self.pair_operator.elements, tuple(layer.base_operator.adjoint_specs for layer in self.layers),
            self.pair_operator.adjoint_spec, efm_factories, execution,
            self._fixed_input_stamps, self._preparation_policy,
            tuple((value, value._version) for value in buffers))


@dataclass(frozen=True)
class OwnedModelEpochReuse:
    runtime: object
    state: object
    root_long_range: object | None
    cell: object
    numbers: object
    ids: object
    elements: object
    base_adjoint_specs: tuple
    pair_adjoint_spec: object
    efm_factories: tuple
    execution: object
    fixed_input_stamps: tuple
    preparation_policy: tuple
    versions: tuple

    def assert_integrity(self, model, fixed_inputs, policy):
        current = (self.cell, self.numbers, self.ids, self.elements, self.runtime.local_indices,
                   self.runtime.global_to_local, self.runtime.canonical_slots)
        _require(len(current) == len(self.versions) and
                 all(value is original for value, (original, _version) in zip(current, self.versions)),
                 "MODEL_BUILD_REUSE_BUFFER_REPLACED")
        _require(self.fixed_input_stamps == tuple(_input_stamp(value) for value in fixed_inputs) and
                 self.preparation_policy == policy and
                 all(value._version == version for value, version in self.versions),
                 "MODEL_BUILD_REUSE_INVARIANT_CHANGED")
        self.execution.assert_integrity(model)


def _input_stamp(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape),
            tuple(value.stride()), value.dtype, value.device, value.requires_grad)


def _collective_reuse_check(torch, reuse, model, fixed_inputs, policy, *, device, group):
    """All ranks finish local reuse validation before runtime agreement."""
    error = None
    try:
        if reuse is not None:
            _require(isinstance(reuse, OwnedModelEpochReuse), "MODEL_BUILD_REUSE_CAPSULE_REQUIRED")
            reuse.assert_integrity(model, fixed_inputs, policy)
    except Exception as caught:
        error = caught
    code = torch.tensor([int(error is not None)], dtype=torch.int64, device=device)
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(code, op=dist.ReduceOp.MAX, group=group)
    if int(code.item()):
        raise OwnedModelPreparationError("MODEL_BUILD_COLLECTIVE_REUSE_INVARIANT_REJECTED") from error


def _collective_reuse_flags(torch, reuse, metadata, *, device, group):
    """Agree every branch that can skip a collective before entering it."""
    flags = (reuse is not None,
             reuse is not None and reuse.runtime.metadata.owners == metadata.owners,
             reuse is not None and reuse.runtime.metadata.counts == metadata.counts)
    packet = torch.tensor(flags, dtype=torch.int64, device=device)
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        copies = [torch.empty_like(packet) for _ in range(dist.get_world_size(group))]
        dist.all_gather(copies, packet, group=group)
        if any(not torch.equal(value, packet) for value in copies):
            raise OwnedModelPreparationError("MODEL_BUILD_COLLECTIVE_REUSE_POLICY_MISMATCH")
    return flags


def _bound_model(torch, model, device):
    _require(getattr(model, "_xl_capture_base_bound", False) is True and not model.training,
             "MODEL_BUILD_NATIVE_BOUND_EVAL_MODEL_REQUIRED")
    deployment = getattr(model, "deployment_metadata", {})
    from hyff.multigpu.launcher import multigpu_options
    if not multigpu_options()["native_nodes"]:
        _require(deployment.get("precision") == "fp32" and
                 not any(deployment.get(name, False) for name in
                         ("joined_projections", "folded_first_projections", "frozen_layernorm",
                          "fused_local_features", "grouped_experts")),
                 "MODEL_BUILD_NATIVE_FP32_NODE_POLICY_REQUIRED")
        _require(not any(getattr(model, name, False) for name in
                         ("compiled_nodes", "fast_nodes", "joined_nodes", "frozen_local_features")),
                 "MODEL_BUILD_ALTERNATE_NODE_PATH_UNSUPPORTED")
    else:
        # Native-nodes option: exactly the single-GPU largo node tuple; EFM stays FP32.
        _require(deployment.get("precision") == "tf32x3" and deployment.get("grouped_experts") is True and
                 not any(deployment.get(name, False) for name in
                         ("joined_projections", "folded_first_projections", "frozen_layernorm",
                          "fused_local_features")),
                 "MODEL_BUILD_NATIVE_NODES_NODE_POLICY_REQUIRED")
        _require(getattr(model, "fast_nodes", False) is True and
                 getattr(model, "sparse_expert_nodes", False) is True and
                 not any(getattr(model, name, False) for name in
                         ("compiled_nodes", "joined_nodes", "frozen_local_features")),
                 "MODEL_BUILD_NATIVE_NODES_NODE_PATH_REQUIRED")
        from hyff.multigpu.launcher import _precision_report
        try:
            _precision_report(model)
        except ValueError as error:
            raise OwnedModelPreparationError("MODEL_BUILD_NATIVE_NODES_PRECISION_POLICY_REQUIRED") from error
    parameters = tuple(model.parameters())
    _require(bool(parameters) and all(not value.requires_grad and value.dtype == torch.float32 and
                                     value.device == device for value in parameters),
             "MODEL_BUILD_FROZEN_FP32_CURRENT_DEVICE_REQUIRED")
    _require(model.n_elements == 83 and len(model.atomic_numbers) == 83 and
             all(block.message_hidden == 84 for block in model.blocks),
             "MODEL_BUILD_MAIN_C84_NE83_REQUIRED")
    _require(hasattr(model, "latent_charge_energy"), "MODEL_BUILD_LATENT_CHARGE_TERM_REQUIRED")
    _require(getattr(model, "radial_tables", None) is not None and
             len(model.radial_tables) == len(model.blocks) + 1, "MODEL_BUILD_NATIVE_RADIAL_TABLES_REQUIRED")
    _require(isinstance(model.element_lut, torch.Tensor) and model.element_lut.dtype == torch.int64 and
             model.element_lut.device == device and model.element_lut.ndim == 1,
             "MODEL_BUILD_ELEMENT_LUT_INVALID")


def _cpu_tensor(torch, value, *, shape, dtype, code):
    _require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
             value.dtype == dtype and tuple(value.shape) == tuple(shape) and not value.requires_grad, code)


def _digest(value):
    """Private eager rank-agreement digest; never returned as scientific output."""
    array = value.detach().contiguous().numpy()
    # Avoid a second complete byte copy for large edge/shift arrays. Empty
    # multidimensional views cannot be cast on every supported Python version.
    payload = memoryview(array).cast("B") if array.size else b""
    return hashlib.sha256(payload).hexdigest()


def _epoch_graph_signature(epoch):
    """Graph agreement payload; a device epoch has no CPU edge arrays to hash."""
    from hyff.multigpu.device_neighbor_epoch import DeviceOwnedNeighborEpoch
    if isinstance(epoch, DeviceOwnedNeighborEpoch):
        epoch.assert_integrity()
        return {"graph_agreement_kind": "ROOT_VALIDATION_TRUSTED_DEVICE_BROADCAST_LOCAL_COUNTS",
                "graph_certificate": dict(epoch.graph_certificate),
                "global_cpu_graph_sha256": None}
    return {"graph_agreement_kind": "FULL_CPU_EDGE_AND_SHIFT_SHA256",
            "edge_index": _digest(epoch.edge_index), "integer_shifts": _digest(epoch.integer_shifts)}


def _build_device_base_graph(epoch, radius, rank, *, device):
    """Use the local immutable matrices directly, without a host edge bridge."""
    import torch
    from hyff.multigpu.device_neighbor_epoch import DeviceOwnedNeighborEpoch
    from hyff.multigpu.base_messages import OwnedImageGraph, _eager
    _require(isinstance(epoch, DeviceOwnedNeighborEpoch), "MODEL_BUILD_DEVICE_EPOCH_REQUIRED")
    device = torch.device(device)
    _eager(device)
    epoch.assert_integrity()
    view = epoch.radius_view(radius, rank)
    _require(view.rank == rank and all(value.device == device for value in view.buffers()),
             "MODEL_BUILD_DEVICE_VIEW_ORIGIN_MISMATCH")
    _require(view.neighbors.dtype == torch.int64 and view.base_integer_shifts.dtype == torch.int32 and
             view.counts.dtype == torch.int32 and view.local_to_global.dtype == torch.int64 and
             tuple(view.local_to_global.cpu().tolist()) == epoch.metadata.indices_by_rank[rank],
             "MODEL_BUILD_DEVICE_VIEW_METADATA_INVALID")
    return OwnedImageGraph(epoch.metadata, rank, view.local_to_global, view.neighbors,
        view.base_integer_shifts, view.counts,
        epoch.cell.to(device=device, dtype=torch.float32, copy=True).contiguous(), view.base_capacity)


def _efm_validation_values(torch, efm, n_global, edge_capacity, device):
    # These values are consumed only by eager shape/finite validation. The
    # prepared operator keeps its cloned metadata, not these input buffers.
    zeros = lambda shape: torch.zeros(shape, dtype=torch.float32, device=device)
    scalar = zeros((n_global, efm.channels))
    vector, tensor = ((zeros((n_global, 3, efm.channels)), zeros((n_global, 5, efm.channels)))
                      if efm.m_blocks else (zeros((0,)), zeros((0,))))
    return (scalar, vector, tensor, zeros((edge_capacity, efm.radial[0].in_features)),
            zeros((edge_capacity, 3, 3)), zeros((edge_capacity, 5, 5)), zeros((edge_capacity,)))


def prepare_owned_model(model, global_atomic_numbers, global_positions, neighbor_epoch, *,
                        global_ids, cell, device, total_charge, multiplicity, group=None,
                        root_lr_route="auto", tile_edges=32768, compile_efm_tiles=False, trace=None,
                        reuse=None, active_efm_compaction=False, active_base_pair_compaction=False,
                        share_efm_geometry=False, capacity_retention=False):
    """Collective eager construction on an already selected CUDA device.

    Species/IDs, positions, and cell are CPU canonical-reference inputs. The
    FP32 coordinates/cell must exactly equal this epoch's private references.
    Every rank supplies identical global data and configuration. The global
    rank zero must be this group's rank zero, as RankZeroLongRange requires.
    The long-range route is resolved for the global atom count by the
    single-GPU largo resolver.

    No LR ``prepare`` is called: the returned evaluator is not yet ready for a
    force call until the caller performs neighbor/LR decisions for that step.
    Rebuilding creates independent topology/EFM metadata and does not retire
    old epochs. The ``reuse`` object keeps only guarded invariant products,
    never an old edge graph, position snapshot or autograd tape.
    """
    phase_start = time.perf_counter()
    phases = {}
    def phase(name):
        nonlocal phase_start
        now = time.perf_counter()
        phases[name] = 1000.0 * (now - phase_start)
        phase_start = now
    phase("source_verification_ms")
    import torch
    from hyff.models.charge_bound import charge_bound_value
    from hyff.multigpu import collectives as raw
    from hyff.multigpu.neighbor_epoch import OwnedNeighborEpoch, build_owned_base_graph
    from hyff.multigpu.device_neighbor_epoch import DeviceOwnedNeighborEpoch
    from hyff.multigpu.electronic_state import prepare_owned_state, rebind_owned_state
    from hyff.multigpu.base_messages import prepare_base_operator
    from hyff.multigpu.pair_energy import prepare_pair_operator
    from hyff.multigpu.projected_efm_prepared import (
        prepare_owned_projected_efm_metadata, prepare_owned_efm_geometry,
    )
    from hyff.multigpu.active_efm import prepare_owned_active_efm_selector
    from hyff.multigpu.long_range import RankZeroLongRange
    from hyff.multigpu.model import LayerInputs, PreparedOwnedModel

    device = torch.device(device)
    _require(device.type == "cuda" and device.index is not None and device.index == torch.cuda.current_device(),
             "MODEL_BUILD_EXPLICIT_CURRENT_CUDA_DEVICE_REQUIRED")
    raw._outside_capture(device)
    _require(isinstance(neighbor_epoch, (OwnedNeighborEpoch, DeviceOwnedNeighborEpoch)),
             "MODEL_BUILD_NEIGHBOR_EPOCH_REQUIRED")
    device_epoch = isinstance(neighbor_epoch, DeviceOwnedNeighborEpoch)
    neighbor_epoch.assert_integrity()
    n_global = neighbor_epoch.metadata.n_global
    _require(n_global > 0, "MODEL_BUILD_NONEMPTY_GLOBAL_GRAPH_REQUIRED")
    for value in (global_positions, neighbor_epoch.global_positions):
        _cpu_tensor(torch, value, shape=(n_global, 3), dtype=torch.float32,
                    code="MODEL_BUILD_FP32_CPU_REFERENCE_POSITIONS_REQUIRED")
    for value in (cell, neighbor_epoch.cell):
        _cpu_tensor(torch, value, shape=(3, 3), dtype=torch.float32,
                    code="MODEL_BUILD_FP32_CPU_REFERENCE_CELL_REQUIRED")
    for value in (global_ids, global_atomic_numbers):
        _cpu_tensor(torch, value, shape=(n_global,), dtype=torch.int64,
                    code="MODEL_BUILD_CANONICAL_CPU_INTEGER_ROWS_REQUIRED")
    _require(tuple(global_ids.tolist()) == neighbor_epoch.metadata.global_ids,
             "MODEL_BUILD_CANONICAL_IDS_MISMATCH")
    _require(torch.equal(global_positions, neighbor_epoch.global_positions) and torch.equal(cell, neighbor_epoch.cell),
             "MODEL_BUILD_REFERENCE_GEOMETRY_MISMATCH")
    _require(type(total_charge) in (int, float) and math.isfinite(total_charge), "MODEL_BUILD_TOTAL_CHARGE_INVALID")
    _require(type(multiplicity) in (int, float) and math.isfinite(multiplicity) and
             multiplicity >= 1 and int(multiplicity) == multiplicity, "MODEL_BUILD_MULTIPLICITY_INVALID")
    _require(root_lr_route in ("auto", "dense", "tree", "spme"), "MODEL_BUILD_ROOT_LR_ROUTE_INVALID")
    _require(type(tile_edges) is int and 0 < tile_edges <= 32768 and type(compile_efm_tiles) is bool,
             "MODEL_BUILD_EFM_SETTINGS_INVALID")
    _require(type(active_efm_compaction) is bool and (not active_efm_compaction or
             (compile_efm_tiles and tile_edges in (8192, 16384, 32768))),
             "MODEL_BUILD_ACTIVE_EFM_SETTINGS_INVALID")
    _require(type(active_base_pair_compaction) is bool and
             (not active_base_pair_compaction or active_efm_compaction),
             "MODEL_BUILD_ACTIVE_BASE_PAIR_REQUIRES_ACTIVE_EFM")
    _require(type(share_efm_geometry) is bool and
             (not share_efm_geometry or active_efm_compaction),
             "MODEL_BUILD_SHARED_EFM_GEOMETRY_REQUIRES_ACTIVE_EFM")
    _require(type(capacity_retention) is bool and (not capacity_retention or
             (active_efm_compaction and share_efm_geometry and compile_efm_tiles)),
             "MODEL_BUILD_CAPACITY_RETENTION_REQUIRES_SHARED_ACTIVE_COMPILED_EFM")
    _bound_model(torch, model, device)
    radii = required_owned_operator_radii(model)
    _require(Counter(radii) == Counter(neighbor_epoch.layer_radii), "MODEL_BUILD_NEIGHBOR_WORK_RADII_MISMATCH")
    species = set(model.atomic_numbers)
    _require(all(value in species and 0 <= value < model.element_lut.numel()
                 for value in global_atomic_numbers.tolist()), "MODEL_BUILD_GLOBAL_SPECIES_INVALID")
    settings = tuple(model.fused_message_settings)
    _require(len(settings) == 4, "MODEL_BUILD_BASE_KERNEL_SETTINGS_INVALID")
    fixed_inputs = (global_atomic_numbers, cell, global_ids)
    policy = (id(model), device, id(group), float(total_charge), int(multiplicity), root_lr_route,
              tile_edges, compile_efm_tiles, active_efm_compaction, active_base_pair_compaction, share_efm_geometry,
              capacity_retention,
              radii, settings, n_global, neighbor_epoch.skin,
              charge_bound_value(model), tuple(model.atomic_numbers), model.n_elements,
              float(model.cutoff), float(model.config.switch_width), id(trace),
              _input_stamp(model.element_lut), bool(model.condition_spin),
              int(model.config.hidden), int(model.tensor_hidden))
    _collective_reuse_check(torch, reuse, model, fixed_inputs, policy, device=device, group=group)
    phase("input_model_invariant_validation_ms")
    _reuse_present, same_owners, reuse_shape = _collective_reuse_flags(
        torch, reuse, neighbor_epoch.metadata, device=device, group=group)
    runtime = (replace(reuse.runtime, metadata=neighbor_epoch.metadata) if same_owners else
               raw.prepare_runtime(neighbor_epoch.metadata, device=device, group=group))
    raw._agreement(runtime, {"component": "prepare_owned_model", "epoch": neighbor_epoch.epoch,
        "generation": neighbor_epoch.generation, "operator_radii": radii, "settings": settings,
        "skin": neighbor_epoch.skin, "tile_edges": tile_edges, "compile_efm_tiles": compile_efm_tiles,
        "active_efm_compaction": active_efm_compaction,
        "active_base_pair_compaction": active_base_pair_compaction,
        "share_efm_geometry": share_efm_geometry,
        "capacity_retention": capacity_retention,
        "charge_bound": charge_bound_value(model),
        "root_lr_route": root_lr_route, "positions": _digest(global_positions), "cell": _digest(cell),
        "species": _digest(global_atomic_numbers), **_epoch_graph_signature(neighbor_epoch),
        "epoch_invariants_reused": reuse is not None, "exchange_shapes_reused": reuse_shape,
        "owned_rows_reused": same_owners})
    phase("runtime_and_signature_ms")
    with torch.no_grad():
        positions_device = global_positions.detach().to(device).clone().contiguous()
        cell_device = reuse.cell if reuse is not None else cell.detach().to(device).clone().contiguous()
        numbers_device = (reuse.numbers if reuse is not None else
                          global_atomic_numbers.detach().to(device).clone().contiguous())
        ids_cpu = reuse.ids if reuse is not None else global_ids.detach().clone().contiguous()
        owned_species = numbers_device.index_select(0, runtime.local_indices)
        elements = (reuse.elements if reuse is not None else
                    model.element_lut.index_select(0, numbers_device).contiguous())
    state = (rebind_owned_state(model, reuse.state, owned_species, runtime) if reuse is not None else
             prepare_owned_state(model, owned_species, runtime, total_charge=total_charge, multiplicity=multiplicity))
    phase("fixed_buffers_and_aee_state_ms")
    # All same-radius base/pair operators share this immutable owned graph.
    graph_cache = {}

    def graph_for(radius):
        radius = float(radius)
        if radius not in graph_cache:
            builder = _build_device_base_graph if device_epoch else build_owned_base_graph
            graph_cache[radius] = builder(neighbor_epoch, radius, runtime.rank, device=device)
        return graph_cache[radius]

    layers = []
    active_selectors = {}
    for index, (block, cutoff) in enumerate(zip(model.blocks, model.layer_cutoffs)):
        adjoints = (tuple(replace(spec, runtime=runtime) for spec in reuse.base_adjoint_specs[index])
                    if reuse_shape else
                    tuple(raw.prepare_exchange(runtime, shape, torch.float32, raw.DISJOINT_OUTPUT)
                          for shape in ((84,), (3, 84), (5, 84))))
        base = prepare_base_operator(graph_for(cutoff), model.radial_tables[index].coefficients,
                    cutoff=float(cutoff), switch_width=float(model.config.switch_width), first=index == 0,
                    settings=settings, adjoint_specs=adjoints)
        efm = getattr(block, "efm", None)
        prepared_efm = geometry = None
        if efm is not None:
            radius = float(efm.__dict__.get("edge_cutoff", cutoff))
            view = neighbor_epoch.radius_view(radius, runtime.rank)
            prepared_efm = prepare_owned_projected_efm_metadata(efm,
                view.source.to(device), view.destination.to(device), view.valid.to(device),
                n_global=n_global, dtype=torch.float32, device=device,
                owned_global_indices=runtime.local_indices, unit_shifts=view.integer_shifts.to(device),
                tile_edges=tile_edges, compile_tiles=compile_efm_tiles, epoch=neighbor_epoch.epoch,
                factory_reuse=reuse.efm_factories[index] if reuse is not None else None)
            active_selector = None
            if active_efm_compaction:
                if radius not in active_selectors:
                    active_selectors[radius] = prepare_owned_active_efm_selector(prepared_efm,
                        positions_device, cell_device, radius=radius, reserve_factor=1.05)
                active_selector = active_selectors[radius]
            geometry = prepare_owned_efm_geometry(prepared_efm, cell_device,
                centers=model.rbf.centers, gamma=model.rbf.gamma, rbf_cutoff=float(model.rbf.cutoff),
                radius=radius, width=float(model.config.switch_width), active_selector=active_selector)
        layers.append(LayerInputs(base, prepared_efm, geometry))
        phase("layer_" + str(index) + "_topology_and_operators_ms")
    pair_adjoint = (replace(reuse.pair_adjoint_spec, runtime=runtime) if reuse_shape else
                    raw.prepare_exchange(runtime, (), torch.float32, raw.DISJOINT_OUTPUT))
    pair = prepare_pair_operator(graph_for(model.cutoff), elements, model.radial_tables[-1].coefficients,
                cutoff=float(model.cutoff), switch_width=float(model.config.switch_width),
                n_elements=model.n_elements, fma=settings[3], adjoint_spec=pair_adjoint)
    phase("pair_operator_ms")
    root_lr = reuse.root_long_range if reuse is not None else None
    lr_construction_ok = True
    if runtime.rank == 0 and reuse is None:
        try:
            root_lr = RankZeroLongRange(model.latent_charge_energy, n_global=n_global,
                                       requested=root_lr_route, rank=0, cell=cell_device)
        except Exception:
            lr_construction_ok = False
    # All ranks enter this eager signature even if root backend construction
    # fails, so peers cannot proceed into evaluator preparation on their own.
    raw._agreement(runtime, {"component": "model_root_lr_constructed", "ok": lr_construction_ok})
    _require(lr_construction_ok, "MODEL_BUILD_ROOT_LR_CONSTRUCTION_FAILED")
    phase("root_long_range_ms")
    evaluator = PreparedOwnedModel(model, state, tuple(layers), pair, root_lr,
                                   total_charge=total_charge, trace=trace,
                                   reuse=reuse.execution if reuse is not None else None,
                                   active_base_pair_compaction=active_base_pair_compaction,
                                   share_efm_geometry=share_efm_geometry)
    if reuse is not None and root_lr is not None:
        root_lr.invalidate_neighbor_epoch()
    phase("model_composition_ms")
    buffers = (positions_device, cell_device, numbers_device, ids_cpu,
               runtime.local_indices, runtime.global_to_local, runtime.canonical_slots)
    bundle = OwnedModelBundle(evaluator, neighbor_epoch, runtime, state, tuple(layers), pair, root_lr,
        positions_device, ids_cpu, numbers_device, cell_device, radii, root_lr_route, tile_edges,
        compile_efm_tiles, active_efm_compaction, active_base_pair_compaction, share_efm_geometry, capacity_retention,
        tuple((value, value._version) for value in buffers),
        tuple(_input_stamp(value) for value in fixed_inputs), policy, phases)
    if capacity_retention:
        from hyff.multigpu.efm_capacity_runtime import install_capacity_retention
        install_capacity_retention(bundle)
    return bundle


__all__ = ["OwnedModelPreparationError", "OwnedModelBundle", "required_owned_operator_radii",
           "prepare_owned_model"]
