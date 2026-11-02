"""Collective eager FP32 largo force session, with bounded same-position rebuild.

This module creates no process/group, makes no persistent device selection and
integrates no MD.
The launcher starts every rank and sets the finite process-group timeout; the
caller supervises the whole job. The root runs one global skin query and
broadcasts it exactly. This module does not change the numerical operators,
charge bounds, long-range choice or OwnedEagerStep.
Imports are stdlib-only until the caller constructs a torch control/backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time


SUPPORTED_RANKS = (1, 2, 4, 8)


class MDSessionError(RuntimeError):
    def __init__(self, code, *, collective_safe=False):
        self.code = code
        self.collective_safe = collective_safe
        self.state_advance_allowed = False
        super().__init__(code)


def _require(condition, code, *, collective_safe=False):
    if not condition:
        raise MDSessionError(code, collective_safe=collective_safe)


@dataclass
class NeighborCacheView:
    """Counts are actual constructions; the initial graph counts as one build."""
    rebuilds: int = 0
    growths: int = 0
    epoch_replacements: int = 0
    graph_capacity: int = 0
    epoch: int = -1
    generation: int = 0
    owners_sha256: str | None = None
    changed_owner_count: int | None = None
    owner_change_events: int = 0


@dataclass(frozen=True)
class _RetiredEpochHeader:
    """O(N) ownership identity only; never holds retired topology buffers."""
    epoch: int
    generation: int
    metadata: object


@dataclass(frozen=True)
class ForceEvaluation:
    status: str
    reason: str
    energy: object | None
    forces: object | None
    diagnostics: dict

    @property
    def accepted(self):
        return self.status == "ACCEPTED"


class TorchCollectiveControl:
    """Control collectives over an existing NCCL/CUDA group (or Gloo/CPU for tests).

    No group is set up here. prepare_md_session below accepts CUDA only. Direct
    Gloo/CPU construction allows CPU tests of the collective protocol; the GPU
    numerical evaluator itself does not run on CPU.
    """
    def __init__(self, *, device, ranks, group=None):
        import torch
        import torch.distributed as dist
        _require(type(ranks) is int and ranks in SUPPORTED_RANKS, "MD_SUPPORTED_RANKS_REQUIRED")
        self.torch, self.dist = torch, dist
        self.device, self.group = torch.device(device), group
        if dist.is_available() and dist.is_initialized():
            self.rank, self.world_size = dist.get_rank(group), dist.get_world_size(group)
            backend = str(dist.get_backend(group))
            _require((backend == "nccl" and self.device.type == "cuda") or
                     (backend == "gloo" and self.device.type == "cpu"), "MD_GROUP_DEVICE_MISMATCH")
            self.root = dist.get_global_rank(group, 0) if group is not None else 0
            _require(self.root == 0, "MD_GLOBAL_RANK_ZERO_REQUIRED")
        else:
            _require(group is None and ranks == 1, "MD_CALLER_INITIALIZED_GROUP_REQUIRED")
            self.rank, self.world_size, self.root = 0, 1, 0
        _require(self.world_size == ranks, "MD_GROUP_SIZE_MISMATCH")
        if self.device.type == "cuda":
            _require(self.device.index is not None and self.device.index == torch.cuda.current_device(),
                     "MD_CALLER_SELECTED_CUDA_DEVICE_REQUIRED")
            _require(not torch.cuda.is_current_stream_capturing(), "MD_EAGER_ONLY")
        else:
            _require(self.device.type == "cpu", "MD_UNSUPPORTED_DEVICE")
            self.device = torch.device("cpu")

    def max_code(self, code):
        torch, dist = self.torch, self.dist
        _require(type(code) is int and 0 <= code < 2**31, "MD_CONTROL_CODE_INVALID")
        value = torch.tensor([code], dtype=torch.int64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self.group)
        return int(value.item())

    def broadcast_ints(self, values, count):
        torch = self.torch
        if self.rank == 0:
            _require(type(values) in (tuple, list) and len(values) == count and
                all(type(v) is int and -(2**63) <= v < 2**63 for v in values), "MD_CONTROL_PACKET_INVALID")
        packet = torch.tensor(values if self.rank == 0 else [0] * count,
                              dtype=torch.int64, device=self.device)
        if self.world_size > 1:
            self.dist.broadcast(packet, src=self.root, group=self.group)
        return packet.cpu().tolist()

    def agree(self, payload):
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        digest = hashlib.sha256(data).digest()
        words = [int.from_bytes(digest[i:i + 8], "little", signed=True) for i in range(0, 32, 8)]
        tensor = self.torch.tensor(words, dtype=self.torch.int64, device=self.device)
        if self.world_size > 1:
            copies = [self.torch.empty_like(tensor) for _ in range(self.world_size)]
            self.dist.all_gather(copies, tensor, group=self.group)
            _require(all(value.cpu().tolist() == words for value in copies),
                     "MD_CONTROL_AGREEMENT_FAILED", collective_safe=True)

    def broadcast_positions(self, root_positions, atoms):
        torch = self.torch
        code = 0
        if self.rank == 0:
            if not (isinstance(root_positions, torch.Tensor) and root_positions.dtype == torch.float32 and
                    root_positions.device == self.device and tuple(root_positions.shape) == (atoms, 3) and
                    root_positions.is_contiguous()):
                code = 1
        elif root_positions is not None:
            code = 1
        _require(self.max_code(code) == 0, "MD_FP32_ROOT_POSITIONS_REQUIRED", collective_safe=True)
        # Every evaluation owns a private complete snapshot, before graph
        # decisions or retry. The caller's FP64 integrator state is never aliased.
        snapshot = (root_positions.detach().clone().contiguous() if self.rank == 0 else
                    torch.empty((atoms, 3), dtype=torch.float32, device=self.device))
        if self.world_size > 1:
            self.dist.broadcast(snapshot, src=self.root, group=self.group)
        _require(self.max_code(int(not bool(torch.isfinite(snapshot).all()))) == 0,
                 "MD_NONFINITE_POSITION", collective_safe=True)
        return snapshot

    def stamp(self, snapshot):
        return snapshot._version

    def unchanged(self, snapshot, version):
        return self.max_code(int(snapshot._version != version)) == 0

    def output_ok(self, energy, forces, atoms):
        torch = self.torch
        valid = (isinstance(energy, torch.Tensor) and energy.device == self.device and
                 energy.dtype == torch.float64 and energy.numel() == 1 and
                 isinstance(forces, torch.Tensor) and forces.device == self.device and
                 forces.dtype == torch.float32 and tuple(forces.shape) == (atoms, 3) and
                 forces.is_contiguous())
        if valid:
            valid = bool(torch.isfinite(energy).all() & torch.isfinite(forces).all())
        return self.max_code(int(not valid)) == 0

    def copy_output(self, energy, forces):
        return energy.detach().reshape(()).clone(), forces.detach().clone().contiguous()

    def broadcast_graph(self, graph, *, atoms, max_matrix_entries, max_attempts):
        """Root's completed query only; no pickle/object collective or discovery."""
        torch = self.torch
        if self.rank == 0:
            header = [int(graph.edge_index.shape[1]), int(graph.capacity), int(graph.maximum_degree),
                      len(graph.attempts), int(graph.spatial_bins)]
        else:
            header = None
        edges, capacity, degree, attempts, bins = self.broadcast_ints(header, 5)
        _require(0 <= edges <= max_matrix_entries and 0 < capacity and 0 <= degree <= capacity and
            atoms * capacity <= max_matrix_entries and 1 <= attempts <= max_attempts and bins > 0,
            "MD_GRAPH_HEADER_INVALID", collective_safe=True)
        if self.rank == 0:
            history = [int(v) for pair in graph.attempts for v in pair]
        else:
            history = None
        history = self.broadcast_ints(history, 2 * attempts)
        _require(all(history[i] > 0 and history[i + 1] >= 0 for i in range(0, len(history), 2)) and
            history[-2:] == [capacity, degree], "MD_GRAPH_HISTORY_INVALID", collective_safe=True)
        output = []
        for name, shape in (("edge_index", (2, edges)), ("integer_shifts", (edges, 3))):
            value = (getattr(graph, name).to(device=self.device, dtype=torch.int64).contiguous() if self.rank == 0
                     else torch.empty(shape, dtype=torch.int64, device=self.device))
            if self.world_size > 1:
                self.dist.broadcast(value, src=self.root, group=self.group)
            output.append(value.cpu())
        return (*output, {"capacity": capacity, "maximum_degree": degree, "attempts": attempts,
            "growths": attempts - 1, "spatial_bins": bins,
            "capacity_history": [history[i:i + 2] for i in range(0, len(history), 2)]})


class OwnedMDSession:
    """All-rank force transaction; only ACCEPTED returns energy and forces.

    control and backend can be substituted in CPU tests; prepare_md_session
    below is the only public constructor of the native evaluator.
    The backend evaluates an already frozen private global-coordinate snapshot.
    """
    def __init__(self, model, backend, control, *, max_rebuilds=1):
        _require(type(max_rebuilds) is int and 0 <= max_rebuilds <= 2, "MD_REBUILD_RETRY_LIMIT_INVALID")
        self.model, self.backend, self.control = model, backend, control
        self.rank, self.world_size = control.rank, control.world_size
        self.device, self.cache = control.device, backend.cache
        self.atoms = backend.atoms
        self.max_rebuilds = max_rebuilds
        self.closed = self.fatal = self.in_flight = False
        self.evaluations = self.accepted_evaluations = 0
        self.last_diagnostics = {}

    def _rejected(self, reason, attempts, decisions, step=None):
        diagnostics = {"schema": "LARGO_OWNED_MD_EVALUATION", "status": "REJECTED",
            "reason": reason, "forces_returned": False, "state_advance_allowed": False,
            "attempts": attempts, "same_snapshot_retries": max(0, attempts - 1), "decisions": decisions,
            "successful_attempt_index": None, "adopted_epoch": None, "adopted_generation": None,
            "step": step}
        self.last_diagnostics = diagnostics
        return ForceEvaluation("REJECTED", reason, None, None, diagnostics)

    def evaluate_global(self, root_positions=None):
        _require(not self.closed and not self.fatal and not self.in_flight, "MD_SESSION_NOT_READY")
        self.in_flight = True
        self.evaluations += 1
        attempts, decisions, step_diagnostics = 0, [], None
        try:
            try:
                snapshot = self.control.broadcast_positions(root_positions, self.atoms)
            except MDSessionError as error:
                if error.collective_safe:
                    return self._rejected(error.code, 0, decisions)
                raise
            version = self.control.stamp(snapshot)
            while True:
                attempts += 1
                result = self.backend.evaluate(snapshot)
                step_diagnostics = dict(result.diagnostics)
                decision = result.decision
                _require(decision.status in ("READY", "RECAPTURE_REQUIRED", "REBUILD_REQUIRED", "REJECT") and
                    decision.accepted_eager is (decision.status in ("READY", "RECAPTURE_REQUIRED")),
                    "MD_STEP_DECISION_INVALID")
                decisions.append(decision.public())
                _require(self.control.unchanged(snapshot, version), "MD_SNAPSHOT_MUTATED", collective_safe=True)
                if decision.status == "REBUILD_REQUIRED":
                    if attempts > self.max_rebuilds:
                        return self._rejected("MD_REBUILD_RETRY_LIMIT", attempts, decisions, step_diagnostics)
                    self.backend.rebuild(snapshot)
                    _require(self.control.unchanged(snapshot, version), "MD_SNAPSHOT_MUTATED", collective_safe=True)
                    continue
                if not decision.accepted_eager:
                    return self._rejected(decision.reason, attempts, decisions, step_diagnostics)
                # Record the epoch that produced these forces before assembly/copy.
                # The cache can change later, so its final value does not
                # identify the epoch of this attempt.
                adopted_epoch, adopted_generation = self.cache.epoch, self.cache.generation
                energy, forces = self.backend.assemble(result)
                _require(self.control.unchanged(snapshot, version), "MD_SNAPSHOT_MUTATED", collective_safe=True)
                _require(self.control.output_ok(energy, forces, self.atoms),
                         "MD_INVALID_ENERGY_FORCE_OUTPUT", collective_safe=True)
                energy, forces = self.control.copy_output(energy, forces)
                # Every rank reaches this scalar collective after its output
                # copies on the same device stream. max_code's host read waits
                # for completion, so rank zero cannot end the step timer while
                # another rank still has an accepted output copy outstanding.
                # A copy failure is fatal and must not enqueue a recovery call.
                _require(self.control.max_code(0) == 0, "MD_COMPLETION_ACK_FAILED")
                complete_evaluation = getattr(self.backend, "complete_evaluation", None)
                if complete_evaluation is not None:
                    complete_evaluation()
                diagnostics = {"schema": "LARGO_OWNED_MD_EVALUATION", "status": "ACCEPTED",
                    "reason": "OK", "forces_returned": True, "state_advance_allowed": True,
                    "attempts": attempts, "same_snapshot_retries": attempts - 1, "decisions": decisions,
                    "successful_attempt_index": attempts, "adopted_epoch": adopted_epoch,
                    "adopted_generation": adopted_generation,
                    "coordinate_policy": "PRIVATE_ROOT_FP32_SNAPSHOT_UNCHANGED_ACROSS_RETRY",
                    "energy_dtype": "float64", "force_dtype": "float32",
                    "accepted_call_completion": "ALL_RANK_ACK_AFTER_OUTPUT_COPY",
                    "capture_used": False, "integration_performed": False,
                    "neighbor_rebuilds": self.cache.rebuilds, "neighbor_growths": self.cache.growths,
                    "step": dict(result.diagnostics)}
                self.accepted_evaluations += 1
                self.last_diagnostics = diagnostics
                return ForceEvaluation("ACCEPTED", "OK", energy, forces, diagnostics)
        except MDSessionError as error:
            if error.collective_safe:
                # A synchronized failure is safe to report without another
                # collective, but retired topology cannot serve another call.
                self.fatal = bool(getattr(self.backend, "retired", False))
                return self._rejected(error.code, attempts, decisions, step_diagnostics)
            self.fatal = True
            raise
        except BaseException:
            # No status collective after an unexpected model/collective error:
            # a peer may already be in a different backward collective.
            self.fatal = True
            raise
        finally:
            self.in_flight = False

    def metadata(self):
        return {"schema": "LARGO_OWNED_MD_SESSION", "experimental": True,
            "rank": self.rank, "world_size": self.world_size, "global_atoms": self.atoms,
            "closed": self.closed, "fatal": self.fatal, "evaluations": self.evaluations,
            "accepted_evaluations": self.accepted_evaluations, "max_rebuilds_per_evaluation": self.max_rebuilds,
            "neighbor_rebuilds": self.cache.rebuilds, "neighbor_growths": self.cache.growths,
            "neighbor_epoch": self.cache.epoch, "neighbor_generation": self.cache.generation,
            "graph_capacity": self.cache.graph_capacity, "root_model_reference_preserved": self.model is not None,
            "accepted_call_completion": "ALL_RANK_ACK_AFTER_OUTPUT_COPY",
            "caller_owns_group_and_timeout": True, "integration_performed": False,
            "capture_supported": False, "md_validated": False, "speed_validated": False,
            "numerical_backend": self.backend.metadata()}

    def close(self):
        """Local retirement only; service coordinates STOP before invoking it."""
        _require(not self.in_flight, "MD_SESSION_IN_FLIGHT")
        if not self.closed:
            self.backend.close()
            self.model = None
            self.closed = True


class _NativeOwnedBackend:
    def __init__(self, model, numbers, positions, cell, ids, control, *, dependency_root,
                 cache_directory, total_charge, multiplicity, skin, root_lr_route,
                 tile_edges, compile_efm_tiles, active_efm_compaction, active_base_pair_compaction,
                 share_efm_geometry, capacity_retention,
                 charge_absolute_bar, charge_guard, measure,
                 initial_capacity, max_capacity, max_attempts, max_matrix_entries):
        self.model, self.control = model, control
        self.numbers, self.cell, self.ids = (value.detach().clone().contiguous() for value in (numbers, cell, ids))
        self._physical_input_stamps = tuple((id(value), value._version)
            for value in (self.numbers, self.cell, self.ids))
        self.atoms = len(ids)
        self.dependency_root, self.cache_directory = dependency_root, cache_directory
        self.total_charge, self.multiplicity, self.skin = total_charge, multiplicity, skin
        self.root_lr_route, self.tile_edges = root_lr_route, tile_edges
        self.compile_efm_tiles, self.charge_absolute_bar, self.measure = compile_efm_tiles, charge_absolute_bar, measure
        self.active_efm_compaction = active_efm_compaction
        self.active_base_pair_compaction = active_base_pair_compaction
        self.share_efm_geometry, self.capacity_retention = share_efm_geometry, capacity_retention
        self.charge_guard = charge_guard
        self.initial_capacity, self.max_capacity = initial_capacity, max_capacity
        self.max_attempts, self.max_matrix_entries = max_attempts, max_matrix_entries
        self.cache = NeighborCacheView()
        self.bundle = self.step = self.force_spec = None
        self.retired = False
        self.graph_metadata = {}
        self.ownership_history = []
        self.epoch_preparation_history = []
        self.invariant_reuses = self.exchange_shape_reuses = self.owned_row_reuses = 0
        from hyff.models.charge_bound import charge_bound_value
        self.bound_value = charge_bound_value(model)
        from hyff.multigpu.prepare_model import required_owned_operator_radii
        self.radii = required_owned_operator_radii(model)
        self._replace(positions)

    def _replace(self, positions_cpu, *, started=None):
        started = time.perf_counter() if started is None else started
        return self._replace_epoch(positions_cpu, started=started)

    def _replace_epoch(self, positions_cpu, *, started=None):
        start = phase_start = time.perf_counter() if started is None else started
        phases = {}
        def phase(name):
            nonlocal phase_start
            now = time.perf_counter()
            phases[name] = 1000.0 * (now - phase_start)
            phase_start = now
        control = self.control
        physical_inputs_valid = all((id(value), getattr(value, "_version", None)) == stamp
            for value, stamp in zip((self.numbers, self.cell, self.ids), self._physical_input_stamps))
        _require(control.max_code(int(not physical_inputs_valid)) == 0,
            "MD_FIXED_PHYSICAL_INPUT_MUTATED", collective_safe=True)
        from hyff.multigpu.cache_scope import cache_configuration
        from hyff.multigpu.global_skin_graph import (
            prepare_global_skin_graph_device, preflight_global_skin_graph,
        )
        from hyff.multigpu.graph_transport import broadcast_device_neighbor_epoch
        from hyff.multigpu.prepare_model import prepare_owned_model
        from hyff.multigpu.step import OwnedEagerStep
        from hyff.multigpu import collectives
        previous = None
        initial = self.cache.graph_capacity or self.initial_capacity
        limits = dict(initial_capacity=initial, max_capacity=self.max_capacity,
            max_attempts=self.max_attempts, max_matrix_entries=self.max_matrix_entries)
        preflight_ok = True
        try:
            preflight_global_skin_graph(positions_cpu, self.cell, global_ids=self.ids,
                radii=self.radii, skin=self.skin, **limits)
        except Exception:
            preflight_ok = False
        _require(control.max_code(int(not preflight_ok)) == 0, "MD_GRAPH_PREFLIGHT_REJECTED", collective_safe=True)
        phase("physical_guard_and_graph_preflight_ms")
        reuse, previous_force_spec, retirement_ok = None, None, True
        try:
            if self.bundle is not None:
                self.bundle.assert_integrity()
                old = self.bundle.neighbor_epoch
                previous = _RetiredEpochHeader(old.epoch, old.generation, old.metadata)
                del old
                # The reuse object contains fixed model/runtime/spec templates only.
                # It must not keep any old edge graph, EFM metadata or tape.
                reuse = self.bundle.epoch_independent_reuse()
                previous_force_spec = self.force_spec
                controller = getattr(self.bundle.evaluator, "capacity_retention", None)
                if controller is not None:
                    controller.retire("EPOCH_RETIRED")
        except Exception:
            retirement_ok = False
        _require(control.max_code(int(not retirement_ok)) == 0,
                 "MD_EPOCH_RETIREMENT_REJECTED", collective_safe=True)
        # Retire topology BEFORE the root's next global query and sort. Root
        # integration still owns its last committed state. Any later failure is
        # terminal and cannot fall back to forces from the retired epoch.
        self.retired = True
        self.bundle = self.step = self.force_spec = None
        controller = None
        phase("invariant_extraction_and_retirement_ms")
        if self.capacity_retention and previous is not None:
            # A new replay baseline caps retention by physical free memory.
            # Release unused allocator cache from the retired epoch before
            # preparing its replacement; live invariant tensors stay referenced.
            cache_release_ok = True
            try:
                with control.torch.cuda.device(control.device):
                    control.torch.cuda.synchronize(control.device)
                    control.torch.cuda.empty_cache()
            except Exception:
                cache_release_ok = False
            _require(control.max_code(int(not cache_release_ok)) == 0,
                     "MD_EPOCH_CACHE_RELEASE_REJECTED", collective_safe=True)
            phase("retired_allocator_cache_release_ms")
        next_epoch = 0 if previous is None else previous.epoch + 1
        next_generation = 1 if previous is None else previous.generation + 1
        graph, graph_ok = None, True
        if control.rank == 0:
            try:
                graph = prepare_global_skin_graph_device(positions_cpu, self.cell, global_ids=self.ids,
                    radii=self.radii, skin=self.skin, device=control.device,
                    dependency_root=self.dependency_root, world_size=control.world_size,
                    epoch=next_epoch, generation=next_generation, **limits)
            except Exception:
                graph_ok = False
        [failure] = control.broadcast_ints([int(not graph_ok)] if control.rank == 0 else None, 1)
        _require(failure == 0, "MD_ROOT_GRAPH_BUILD_REJECTED", collective_safe=True)
        phase("root_global_graph_and_device_epoch_ms")
        epoch, graph_meta = broadcast_device_neighbor_epoch(control, graph,
            global_positions=positions_cpu, cell=self.cell, atoms=self.atoms,
            max_matrix_entries=self.max_matrix_entries, max_attempts=self.max_attempts,
            expected_epoch=next_epoch, expected_generation=next_generation)
        # Only the local epoch survives. Non-root ranks never held full
        # topology; root releases its E-sized stream after successful finish.
        del graph
        phase("device_transport_and_local_views_ms")
        ownership, ownership_ok = None, True
        try:
            ownership = _ownership_transition(epoch, previous)
        except Exception:
            ownership_ok = False
        _require(control.max_code(int(not ownership_ok)) == 0,
                 "MD_OWNER_TRANSITION_REJECTED", collective_safe=True)
        control.agree({"component": "owned_md_epoch_adoption", "epoch": epoch.epoch,
            "generation": epoch.generation, "owners": epoch.metadata.owners,
            "counts": epoch.metadata.counts, "ownership_transition": ownership, "graph": graph_meta})
        phase("ownership_transition_and_agreement_ms")
        with cache_configuration(self.cache_directory):
            bundle = prepare_owned_model(self.model, self.numbers, positions_cpu, epoch,
                global_ids=self.ids, cell=self.cell, device=control.device,
                total_charge=self.total_charge, multiplicity=self.multiplicity,
                group=control.group, root_lr_route=self.root_lr_route,
                tile_edges=self.tile_edges, compile_efm_tiles=self.compile_efm_tiles, reuse=reuse,
                active_base_pair_compaction=self.active_base_pair_compaction,
                share_efm_geometry=self.share_efm_geometry, capacity_retention=self.capacity_retention,
                **({'active_efm_compaction': True} if self.active_efm_compaction else {}))
            phase("owned_model_prepare_ms")
            step = OwnedEagerStep(bundle.evaluator, epoch, total_charge=self.total_charge,
                charge_absolute_bar=self.charge_absolute_bar, charge_guard=self.charge_guard, measure=self.measure)
            phase("eager_step_prepare_ms")
            if reuse is not None and reuse.runtime.metadata.counts == bundle.runtime.metadata.counts:
                from dataclasses import replace
                force_spec = replace(previous_force_spec, runtime=bundle.runtime)
            else:
                force_spec = collectives.prepare_exchange(bundle.runtime, (3,), control.torch.float32,
                    collectives.DETACHED_SOURCE)
            phase("force_exchange_prepare_ms")
        self.bundle, self.step, self.force_spec = bundle, step, force_spec
        self.retired = False
        self.cache.rebuilds += 1
        self.cache.growths += graph_meta["growths"]
        self.cache.epoch_replacements += int(previous is not None)
        self.cache.graph_capacity = graph_meta["capacity"]
        self.cache.epoch, self.cache.generation = epoch.epoch, epoch.generation
        self.cache.owners_sha256 = ownership["owners_sha256"]
        self.cache.changed_owner_count = ownership["changed_owner_count"]
        self.cache.owner_change_events += int((ownership["changed_owner_count"] or 0) > 0)
        self.invariant_reuses += int(reuse is not None)
        self.exchange_shape_reuses += int(reuse is not None and
            reuse.runtime.metadata.counts == bundle.runtime.metadata.counts)
        self.owned_row_reuses += int(reuse is not None and
            reuse.runtime.metadata.owners == bundle.runtime.metadata.owners)
        self.graph_metadata = graph_meta
        # Append only after the fresh bundle/step has actually been adopted.
        # Failed preparation never creates a successful ownership transition.
        self.ownership_history.append(ownership)
        self.epoch_preparation_history.append({"epoch": epoch.epoch, "generation": epoch.generation,
            "total_ms": 1000.0 * (time.perf_counter() - start), "phases_ms": phases,
            "model_phases_ms": dict(bundle.preparation_ms),
            "timing_scope": ("HOST_WALL_INCLUDES_EPOCH_CACHE_RELEASE_SYNCHRONIZATION"
                if "retired_allocator_cache_release_ms" in phases
                else "HOST_WALL_NO_ADDED_SYNCHRONIZATION"),
            "invariants_reused": reuse is not None,
            "owned_rows_reused": reuse is not None and
                reuse.runtime.metadata.owners == bundle.runtime.metadata.owners})
        self._pending_replacement_epoch = epoch.epoch

    def evaluate(self, snapshot):
        _require(not self.retired and self.bundle is not None, "MD_BACKEND_RETIRED")
        self._evaluation_started_at = time.perf_counter()
        self.bundle.assert_integrity()
        positions = snapshot.index_select(0, self.bundle.runtime.local_indices).detach().clone().requires_grad_(True)
        return self.step(positions)

    def complete_evaluation(self):
        """Called only after session validation, copies and all-rank ACK.

        Sum active replacement and accepted evaluation intervals rather than
        including arbitrary caller idle time between construction and use.
        Rejected attempts are excluded and cannot create accepted timing.
        """
        if getattr(self, "_pending_replacement_epoch", None) == self.cache.epoch:
            ended = time.perf_counter()
            row = self.epoch_preparation_history[-1]
            row["first_accepted_evaluation_ms"] = 1000.0 * (ended - self._evaluation_started_at)
            row["replacement_through_first_accepted_evaluation_ms"] = (
                row["total_ms"] + row["first_accepted_evaluation_ms"])
            row["accepted_evaluation_timing_scope"] = "BACKEND_ENTRY_THROUGH_OUTPUT_COPY_ALL_RANK_ACK"
            row["replacement_plus_evaluation_timing_scope"] = "ACTIVE_INTERVAL_SUM_EXCLUDES_CALLER_IDLE_AND_REJECTED_ATTEMPTS"
            row["replacement_includes_snapshot_host_copy_and_source_authentication"] = True
            row["includes_first_evaluation_compute"] = True
            row["initial_construction"] = self.cache.epoch == 0
            self._pending_replacement_epoch = None

    def rebuild(self, snapshot):
        started = time.perf_counter()
        self._replace(snapshot.detach().cpu(), started=started)

    def assemble(self, result):
        from hyff.multigpu.collectives import gather_detached_sources
        force = gather_detached_sources(result.owned_force, self.force_spec).contiguous()
        return result.global_energy, force

    def metadata(self):
        return {"root_lr_requested": self.root_lr_route,
            "root_lr_selected": self.bundle.root_long_range.route if self.bundle is not None and
                self.bundle.root_long_range is not None else None,
            "charge_bound": self.bound_value, "Q": self.total_charge,
            "M": self.multiplicity, "skin_A": self.skin, "efm_tile_edges": self.tile_edges,
            "charge_guard": self.charge_guard, "charge_absolute_bar": self.charge_absolute_bar,
            "charge_bar_applied": self.charge_guard == "ABSOLUTE",
            "compile_efm_tiles": self.compile_efm_tiles, "graph": dict(self.graph_metadata),
            "active_efm_compaction": self.active_efm_compaction,
            "active_base_pair_compaction": self.active_base_pair_compaction,
            "share_efm_geometry": self.share_efm_geometry,
            "capacity_retention": (self.bundle.evaluator.capacity_retention.public_metadata()
                if self.capacity_retention and self.bundle is not None else None),
            "owners_sha256": self.cache.owners_sha256,
            "changed_owner_count": self.cache.changed_owner_count,
            "owner_change_events": self.cache.owner_change_events,
            "epoch_invariant_reuses": self.invariant_reuses,
            "epoch_exchange_shape_reuses": self.exchange_shape_reuses,
            "epoch_owned_row_reuses": self.owned_row_reuses,
            "epoch_replacement_policy": "GUARDED_INVARIANTS_FRESH_TOPOLOGY",
            "epoch_topology_residency": "ROOT_DEVICE_STREAM_LOCAL_DEVICE_VIEWS",
            "source_verification_scope": "PACKAGE_SESSION_START_NATIVE_DEPENDENCIES_EACH_ROOT_QUERY",
            "package_source_verification_scope": "SESSION_START_AUTHENTICATION",
            "native_dependency_verification_scope": "EACH_ROOT_GLOBAL_GRAPH_QUERY",
            "epoch_preparation_history": [dict(row, phases_ms=dict(row["phases_ms"]),
                model_phases_ms=dict(row["model_phases_ms"])) for row in self.epoch_preparation_history],
            "ownership_history": [dict(row, counts=list(row["counts"])) for row in self.ownership_history],
            "compute_policy": "NATIVE_BOUND_FP32_C84_NE83_OWNED_EAGER",
            "neighbor_policy": "ROOT_COMPLETE_PINNED_QUERY_FIXED_SKIN_SUPERSET"}

    def close(self):
        self.bundle = self.step = self.force_spec = None
        self.model = None


def _ownership_transition(epoch, previous):
    """Identity of actual canonical (global ID, owner) pairs; no payload data."""
    def digest(metadata):
        body = {"schema": "LARGO_MD_CANONICAL_OWNERS",
                "world_size": metadata.world_size,
                "global_id_owner_pairs": list(zip(metadata.global_ids, metadata.owners))}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
    if previous is not None:
        _require(previous.metadata.global_ids == epoch.metadata.global_ids and
                 previous.metadata.world_size == epoch.metadata.world_size,
                 "MD_OWNER_TRANSITION_IDENTITY_CHANGED", collective_safe=True)
    changed = (sum(a != b for a, b in zip(previous.metadata.owners, epoch.metadata.owners))
               if previous is not None else None)
    return {"schema": "LARGO_MD_OWNERSHIP_TRANSITION",
            "epoch": epoch.epoch, "generation": epoch.generation,
            "global_atoms": epoch.metadata.n_global, "world_size": epoch.metadata.world_size,
            "owners_sha256": digest(epoch.metadata),
            "previous_owners_sha256": digest(previous.metadata) if previous is not None else None,
            "changed_owner_count": changed,
            "counts": tuple(epoch.metadata.counts),
            "source": "ACTUAL_CANONICAL_GLOBAL_ID_OWNER_PAIRS"}


def prepare_md_session(model, global_atomic_numbers, global_positions, cell, *, global_ids,
                       device, ranks, dependency_root, cache_directory, root_lr_route,
                       total_charge, multiplicity, experimental=False, group=None,
                       skin=1.5, tile_edges=32768, compile_efm_tiles=False, active_efm_compaction=False,
                       active_base_pair_compaction=False, share_efm_geometry=False, capacity_retention=False,
                       charge_absolute_bar=1e-5, charge_guard="OBSERVE_ONLY", measure=False, max_rebuilds=1,
                       initial_capacity=None, max_capacity=16384, max_attempts=8,
                       max_matrix_entries=2**28):
    """Collectively build a session from identical CPU canonical physical data.

    The caller loads/freezes and binds the largo FP32 model.
    ``root_lr_route`` is deliberately explicit: no default is derived from the workload.
    Total charge and multiplicity are required physical identities: no neutral
    singlet default is inferred.
    The launcher broadcasts initial physical data before calling on every rank.
    Root constraint projection may then change initial positions; the first
    evaluate uses the current supplied snapshot and rebuilds when needed.
    Total charge is observed by default; explicitly select charge_guard="ABSOLUTE"
    to enforce charge_absolute_bar. FP32 projection/VJP and finite checks stay unchanged.
    """
    _require(experimental is True, "EXPERIMENTAL_TRUE_REQUIRED")
    from hyff.multigpu.cache_scope import validate_cache_directory
    cache = validate_cache_directory(cache_directory)
    control = TorchCollectiveControl(device=device, ranks=ranks, group=group)
    torch = control.torch
    _require(control.device.type == "cuda", "MD_NATIVE_CUDA_REQUIRED")
    valid = (root_lr_route in ("auto", "dense", "tree", "spme") and
        type(max_rebuilds) is int and 0 <= max_rebuilds <= 2 and type(measure) is bool and
        type(compile_efm_tiles) is bool and type(tile_edges) is int and 0 < tile_edges <= 32768 and
        type(active_efm_compaction) is bool and (not active_efm_compaction or compile_efm_tiles) and
        type(active_base_pair_compaction) is bool and
        (not active_base_pair_compaction or active_efm_compaction) and
        type(share_efm_geometry) is bool and (not share_efm_geometry or active_efm_compaction) and
        type(capacity_retention) is bool and (not capacity_retention or share_efm_geometry) and
        type(total_charge) in (int, float) and math.isfinite(total_charge) and
        type(multiplicity) in (int, float) and math.isfinite(multiplicity) and
        multiplicity >= 1 and int(multiplicity) == multiplicity and
        type(skin) in (int, float) and math.isfinite(skin) and skin >= 0 and
        all(type(value) is int and value > 0 for value in (max_capacity, max_attempts, max_matrix_entries)) and
        max_capacity < 2**29 and max_attempts <= 32 and max_matrix_entries < 2**29 and
        (initial_capacity is None or (type(initial_capacity) is int and 0 < initial_capacity <= max_capacity)) and
        type(charge_absolute_bar) in (int, float) and math.isfinite(charge_absolute_bar) and charge_absolute_bar > 0 and
        type(charge_guard) is str and charge_guard in ("ABSOLUTE", "OBSERVE_ONLY"))
    atoms = len(global_ids) if isinstance(global_ids, torch.Tensor) and global_ids.ndim == 1 else -1
    for value, dtype, shape in ((global_atomic_numbers, torch.int64, (atoms,)),
                               (global_ids, torch.int64, (atoms,)),
                               (global_positions, torch.float32, (atoms, 3)), (cell, torch.float32, (3, 3))):
        valid = (valid and isinstance(value, torch.Tensor) and value.device.type == "cpu" and
            value.dtype == dtype and tuple(value.shape) == shape and not value.requires_grad)
    _require(control.max_code(int(not valid or atoms <= 0)) == 0, "MD_INITIAL_INPUT_REJECTED", collective_safe=True)
    fingerprint = lambda value: hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
    control.agree({"component": "owned_md_session", "atoms": atoms, "ranks": ranks,
        "positions": fingerprint(global_positions), "cell": fingerprint(cell), "ids": fingerprint(global_ids),
        "numbers": fingerprint(global_atomic_numbers), "Q": total_charge, "M": multiplicity,
        "skin": skin, "root_lr_route": root_lr_route, "tile_edges": tile_edges,
        "compile_efm_tiles": compile_efm_tiles, "charge_absolute_bar": charge_absolute_bar, "charge_guard": charge_guard,
        "max_rebuilds": max_rebuilds, "initial_capacity": initial_capacity, "max_capacity": max_capacity,
        "max_attempts": max_attempts, "max_matrix_entries": max_matrix_entries, "measure": measure,
        "active_base_pair_compaction": active_base_pair_compaction,
        "share_efm_geometry": share_efm_geometry, "capacity_retention": capacity_retention,
        **({"active_efm_compaction": True} if active_efm_compaction else {})})
    backend = _NativeOwnedBackend(model, global_atomic_numbers, global_positions, cell, global_ids, control,
        dependency_root=dependency_root, cache_directory=cache, total_charge=total_charge, multiplicity=multiplicity,
        skin=skin, root_lr_route=root_lr_route, tile_edges=tile_edges, compile_efm_tiles=compile_efm_tiles,
        active_efm_compaction=active_efm_compaction,
        active_base_pair_compaction=active_base_pair_compaction,
        share_efm_geometry=share_efm_geometry, capacity_retention=capacity_retention,
        charge_absolute_bar=charge_absolute_bar, charge_guard=charge_guard, measure=measure, initial_capacity=initial_capacity,
        max_capacity=max_capacity, max_attempts=max_attempts, max_matrix_entries=max_matrix_entries)
    return OwnedMDSession(model, backend, control, max_rebuilds=max_rebuilds)


__all__ = ["prepare_md_session", "OwnedMDSession", "MDSessionError", "ForceEvaluation", "NeighborCacheView"]
