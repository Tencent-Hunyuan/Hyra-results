"""Collective force-boundary capacity policy for four-block largo EFM.

One actual replay baseline per epoch, then an agreed subset before every
block sequence. No extra calibration force, OOM retry, mid-force policy
switch, device selection, or process-group creation occurs here.
"""
from __future__ import annotations

import copy

from hyff.multigpu.efm_capacity_plan import (
    POLICY_TOKEN, memory_budget, choose_capacity_plan, validate_plan_token,
)


def install_retention_process_policy():
    """Call at process startup before configure/model/compile; idempotent."""
    import torch._functorch.config as config
    config.donated_buffer = False
    return {"donated_buffer": False, "policy_token": POLICY_TOKEN,
            "lifetime": "PROCESS_STARTUP_THROUGH_REPLAY_FALLBACK"}


def assert_retention_process_policy():
    import torch._functorch.config as config
    if config.donated_buffer is not False:
        raise RuntimeError("EFM_CAPACITY_DONATION_POLICY_CHANGED")


class CapacityRetentionController:
    def __init__(self, evaluator, *, epoch, generation):
        import torch
        from hyff.multigpu.efm_replay_storage import ReplayStorageObservation
        self.runtime = evaluator.runtime
        self.epoch, self.generation = epoch, generation
        self.operators = tuple(layer.inputs.efm_operator for layer in evaluator.layers
                               if layer.inputs.efm_operator is not None)
        if len(self.operators) != 4 or any(operator is None for operator in self.operators):
            raise ValueError("EFM_CAPACITY_FOUR_BLOCKS_REQUIRED")
        assert_retention_process_policy()
        self.observers = tuple(ReplayStorageObservation(operator.efm) for operator in self.operators)
        for operator, observer in zip(self.operators, self.observers):
            operator.observe_replay_storage(observer)
        self.baseline = None
        self.baseline_rows = None
        self.plan = None
        self.selection_generation = 0
        self.preflight_count = 0
        self.transitions = []
        self._in_force = False
        self.retired_reason = None
        self._shape = None
        # Reserve control buffers before baseline memory pressure. Every rank
        # always participates, even when local input validation fails.
        self._packet = torch.zeros(43, dtype=torch.int64, device=self.runtime.device)
        self._packets = [torch.empty_like(self._packet) for _ in range(self.runtime.world_size)]
        self._status = torch.zeros(1, dtype=torch.int64, device=self.runtime.device)

    def retire(self, reason):
        """Terminal metadata retirement; never modify/truncate an existing tape."""
        if self.retired_reason is None:
            self.retired_reason = str(reason)
        for operator in self.operators:
            operator._replay_storage_observer = None
            operator._capacity_failed = True
        self.operators = ()

    def _fail(self, code, error=None):
        self.retire(code)
        raise RuntimeError(code) from error

    def _snapshot(self):
        import torch
        device = self.runtime.device
        if device.type != "cuda":
            raise ValueError("EFM_CAPACITY_CUDA_MEMORY_OBSERVATION_REQUIRED")
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        free, total = torch.cuda.mem_get_info(device)
        return dict(allocated_bytes=allocated, reserved_bytes=reserved,
                    physical_free_bytes=int(free), device_total_bytes=int(total))

    def begin_force(self):
        import torch
        import torch.distributed as dist
        from hyff.multigpu import collectives as raw
        error = None
        try:
            if self._in_force or self.retired_reason is not None:
                raise RuntimeError("EFM_CAPACITY_FORCE_ALREADY_ACTIVE")
            if self.baseline is None:
                torch.cuda.synchronize(self.runtime.device)
                torch.cuda.reset_peak_memory_stats(self.runtime.device)
        except Exception as caught:
            error = caught
        self._status.fill_(int(error is not None))
        raw._reduce(self._status, self.runtime, dist.ReduceOp.MAX)
        if int(self._status.item()):
            self._fail("EFM_CAPACITY_FORCE_BEGIN_FAILED", error)
        self._in_force = True

    def _shape_snapshot(self, selections):
        capacities, rows, generations = [], [], []
        for operator in self.operators:
            selector = operator._active_selector
            if selector is None:
                raise ValueError("EFM_CAPACITY_ACTIVE_SELECTION_REQUIRED")
            selection = selections[id(selector)]
            selection.check(operator=operator)
            capacities.append(selection.allocation_capacity)
            rows.append(selection.source.numel())
            generations.append(selection.capacity_generation)
        return capacities, rows, generations

    def _local_values(self):
        assert_retention_process_policy()
        capacities, rows, generations = self._shape
        memory_valid = True
        try:
            snapshot = self._snapshot()
        except Exception:
            memory_valid = False
            snapshot = dict(allocated_bytes=0, reserved_bytes=0, physical_free_bytes=0, device_total_bytes=1)
        reports = self._observation_reports()
        saved = [report["unique_retained_bytes"] if report["valid"] and
                 report["rows"] == operator.metadata.tile_edges else None
                 for report, operator in zip(reports, self.operators)]
        if self.baseline is not None and not self.baseline.get("memory_observation_valid", True):
            memory_valid = False
        peak = self.baseline.get("peak_allocated_bytes", 0) if self.baseline else snapshot["allocated_bytes"]
        baseline_free = self.baseline.get("physical_free_bytes", 0) if self.baseline else 0
        # Relative reserves scale to the physical device and baseline size;
        # there is no fixed allocator or retention ceiling.
        growth = (snapshot["device_total_bytes"] + 19) // 20
        if self.baseline_rows is not None:
            envelope_growth = max((peak * max(0, cap - original) + max(original, 1) - 1) //
                                  max(original, 1) for cap, original in zip(capacities, self.baseline_rows))
            growth += envelope_growth
        workspace = max((snapshot["device_total_bytes"] * 3 + 99) // 100,
                        2 * max((value or 0 for value in saved), default=0))
        # 1 status + 2 epoch + 8 memory + 4*8 per-block fields = 43 words.
        values = [0 if memory_valid else 2, self.epoch, self.generation, peak, baseline_free,
                  snapshot["allocated_bytes"], snapshot["reserved_bytes"],
                  snapshot["physical_free_bytes"], snapshot["device_total_bytes"], growth, workspace]
        for index, operator in enumerate(self.operators):
            work = operator.efm.channels ** 2 * (9 if operator.efm.m_blocks else 1)
            values.extend((capacities[index], rows[index], operator.metadata.tile_edges,
                           generations[index], saved[index] if saved[index] is not None else -1,
                           work, reports[index]["samples"], int(reports[index]["valid"])))
        return values

    def _observation_reports(self):
        reports = []
        for observer in self.observers:
            try:
                report = observer.public_metadata()
                if (not isinstance(report, dict) or type(report.get("valid")) is not bool or
                        type(report.get("samples")) is not int or report["samples"] < 0 or
                        (report["valid"] and (type(report.get("rows")) is not int or report["rows"] <= 0 or
                         type(report.get("unique_retained_bytes")) is not int or
                         report["unique_retained_bytes"] <= 0 or report["samples"] == 0))):
                    raise ValueError("EFM_CAPACITY_STORAGE_OBSERVATION_INVALID")
                report = dict(report)
                report.setdefault("schema", "EFM_ACTUAL_REPLAY_STORAGE")
                report.setdefault("rows", None)
                report.setdefault("unique_retained_bytes", None)
            except Exception:
                report = dict(schema="EFM_ACTUAL_REPLAY_STORAGE", valid=False, rows=None,
                              unique_retained_bytes=None, samples=0,
                              invalid_reason="OBSERVATION_UNAVAILABLE_REPLAY_ONLY")
            reports.append(report)
        return reports

    def _collect_plan(self, *, shape_error=False):
        import torch
        import torch.distributed as dist
        from hyff.multigpu import collectives as raw
        error = None
        try:
            if shape_error:
                raise ValueError("EFM_CAPACITY_SELECTION_INVALID")
            values = self._local_values()
        except Exception as caught:
            error = caught
            values = [1] + [0] * 42
        self._packet.copy_(torch.tensor(values, dtype=torch.int64))
        if self.runtime.world_size > 1:
            dist.all_gather(self._packets, self._packet, group=self.runtime.group)
        else:
            self._packets[0].copy_(self._packet)
        packets = [packet.cpu().tolist() for packet in self._packets]
        if any(packet[0] == 1 for packet in packets):
            self._fail("EFM_CAPACITY_COLLECTIVE_PREFLIGHT_FAILED", error)
        ranks = []
        for packet in packets:
            try:
                if packet[0] == 2:
                    raise ValueError("MEMORY_OBSERVATION_UNAVAILABLE")
                memory = memory_budget(baseline_peak_allocated_bytes=packet[3], baseline_free_bytes=packet[4],
                    allocated_bytes=packet[5], reserved_bytes=packet[6], physical_free_bytes=packet[7],
                    device_total_bytes=packet[8], growth_reserve_bytes=packet[9], workspace_reserve_bytes=packet[10])
            except (ValueError, TypeError):
                packet[0] = 2
                # Explicit unavailable observation, never a fabricated fit.
                # The placeholder budget is zero; rank metadata says invalid.
                memory = memory_budget(baseline_peak_allocated_bytes=0, baseline_free_bytes=0,
                    allocated_bytes=0, reserved_bytes=0, physical_free_bytes=0,
                    device_total_bytes=1, growth_reserve_bytes=0, workspace_reserve_bytes=0)
            blocks = [packet[11 + index * 8:19 + index * 8] for index in range(4)]
            ranks.append(dict(epoch=packet[1], generation=packet[2], memory=memory,
                capacities=[block[0] for block in blocks], logical_rows=[block[1] for block in blocks],
                tile_edges=[block[2] for block in blocks], capacity_generations=[block[3] for block in blocks],
                retained_bytes_per_tile=[None if block[4] < 0 else block[4] for block in blocks],
                replay_work_units_per_tile=[block[5] for block in blocks]))
        plan = choose_capacity_plan(ranks)
        for rank, packet in zip(plan["ranks"], packets):
            rank.update(memory_observation_valid=packet[0] != 2,
                        memory_observation_reason="OBSERVED" if packet[0] != 2 else "UNAVAILABLE_REPLAY_ONLY")
        raw._agreement(self.runtime, {"component": "capacity_retention_preblock", "plan": plan})
        plan.update(all_ranks_agreed=True, planning_only=False)
        # All ranks stage local wrappers, then agree on success BEFORE any
        # ordered block packet. Wrappers reuse the same edge callable.
        token = plan["tokens"][self.runtime.rank]
        error = None
        try:
            capacities, rows, generations = self._shape
            validate_plan_token(token, epoch=self.epoch, generation=self.generation,
                capacities=capacities, logical_rows=rows, capacity_generations=generations)
            for index, operator in enumerate(self.operators):
                operator.apply_capacity_plan(index in plan["blocks"], max_rows=capacities[index],
                                             token=token, block_index=index)
        except Exception as caught:
            error = caught
        self._status.fill_(int(error is not None))
        raw._reduce(self._status, self.runtime, dist.ReduceOp.MAX)
        if int(self._status.item()):
            # A retention wrapper can be unavailable without invalidating the
            # established replay factory. All ranks choose replay BEFORE the
            # first block and agree again; a broken replay contract is fatal.
            plan.update(blocks=[], mask=0, replay_work_score=0,
                        selected_reason="REPLAY_WRAPPER_UNAVAILABLE",
                        selection_override_reason="COLLECTIVE_PREBLOCK_WRAPPER_UNAVAILABLE")
            raw._agreement(self.runtime, {"component": "capacity_retention_replay_staging", "plan": plan})
            error = None
            try:
                for index, operator in enumerate(self.operators):
                    operator.apply_capacity_plan(False, max_rows=capacities[index], token=token, block_index=index)
            except Exception as caught:
                error = caught
            self._status.fill_(int(error is not None))
            raw._reduce(self._status, self.runtime, dist.ReduceOp.MAX)
            if int(self._status.item()):
                self._fail("EFM_CAPACITY_REPLAY_PLAN_APPLICATION_FAILED", error)
        previous = None if self.plan is None else (self.plan["blocks"], self.plan["tokens"])
        current = (plan["blocks"], plan["tokens"])
        changed = previous != current
        if changed:
            self.selection_generation += 1
            self.transitions.append(dict(selection_generation=self.selection_generation,
                before_blocks=[] if previous is None else list(previous[0]),
                blocks=list(plan["blocks"]), preflight=self.preflight_count,
                reason="FIRST_PLAN" if previous is None else "CAPACITY_OR_SUBSET_CHANGED"))
        self.plan = plan

    def preflight(self, selections):
        if not self._in_force:
            raise RuntimeError("EFM_CAPACITY_ENERGY_AND_FORCE_REQUIRED")
        error = False
        try:
            self._shape = self._shape_snapshot(selections)
        except Exception:
            error = True
        self.preflight_count += 1
        self._collect_plan(shape_error=error)

    def finish_force(self):
        import torch
        if self.baseline is None:
            try:
                self.baseline = self._snapshot()
                self.baseline["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(self.runtime.device))
                self.baseline["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(self.runtime.device))
                self.baseline["memory_observation_valid"] = True
            except Exception:
                self.baseline = {"memory_observation_valid": False}
            self.baseline_rows = list(self._shape[1])
            for operator in self.operators:
                operator.observe_replay_storage(None)
            self._collect_plan()
        self._in_force = False

    def public_metadata(self):
        plan = copy.deepcopy(self.plan)
        return dict(schema="CAPACITY_RETENTION_STATE", selected_blocks=[] if plan is None else plan["blocks"],
            selection_generation=self.selection_generation, preflight_count=self.preflight_count,
            retired_reason=self.retired_reason, preflight_control_inside_force_timing=True,
            baseline_observed=self.baseline is not None, baseline=copy.deepcopy(self.baseline),
            baseline_logical_rows=copy.deepcopy(self.baseline_rows),
            baseline_free_scope="PHYSICAL_DEVICE_FREE_AFTER_COMPLETED_REPLAY_BASELINE_PEAK",
            calibration={str(i): report for i, report in enumerate(self._observation_reports())},
            ranks=[] if plan is None else plan["ranks"], plan=plan, transitions=copy.deepcopy(self.transitions),
            separate_calibration_force=False, donated_buffer=False, policy_token=POLICY_TOKEN,
            every_force_preblock_agreement=True, no_midforce_fallback=True,
            reserve_policy="5_PERCENT_DEVICE_PLUS_BASELINE_ENVELOPE_GROWTH",
            workspace_policy="MAX_3_PERCENT_DEVICE_AND_TWO_MAX_OBSERVED_TILE_STORAGES",
            fixed_retention_cap_bytes=None, fixed_allocator_ceiling_bytes=None,
            replay_fallback_preserves_baseline_policy=True,
            maximum_system_preservation_proven=False, fit_guaranteed=False)


def install_capacity_retention(bundle):
    """Builder/helper entry; call before its first force on every rank."""
    evaluator = bundle.evaluator
    if getattr(evaluator, "capacity_retention", None) is not None:
        raise ValueError("EFM_CAPACITY_ALREADY_INSTALLED")
    if not evaluator._share_efm_geometry:
        raise ValueError("EFM_CAPACITY_SHARED_GEOMETRY_REQUIRED")
    evaluator.capacity_retention = CapacityRetentionController(evaluator,
        epoch=bundle.neighbor_epoch.epoch, generation=bundle.neighbor_epoch.generation)
    return evaluator.capacity_retention


__all__ = ["install_retention_process_policy", "assert_retention_process_policy",
           "CapacityRetentionController", "install_capacity_retention"]
