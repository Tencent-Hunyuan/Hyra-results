"""Eager collective reject-before-return protocol for one owned force step.

This coordinator does not rebuild, migrate or capture. It returns a typed
rebuild/reject outcome and no energy/forces when an epoch is unusable. A
caller installs a newly prepared complete epoch on all ranks and retries.
Only a fully accepted evaluation may advance an MD state.

Unexpected exceptions during the model's collective graph are fatal to the
whole supervised rank group; they are not recoverable per-rank rejections.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
import torch.distributed as dist

from hyff.multigpu import collectives
from hyff.multigpu.long_range import LRStatus


class OwnedStepError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise OwnedStepError(code)


STATUS = ("READY", "RECAPTURE_REQUIRED", "REBUILD_REQUIRED", "REJECT")
REASON = ("OK", "POSITION_METADATA", "NONFINITE_POSITION", "EPOCH_INVALID",
          "CELL_CHANGED", "SKIN_EXHAUSTED", "LR_PREPARE_REJECTED", "LR_PREPARE_FAILED",
          "NONFINITE_OUTPUT", "CHARGE_NOT_CONSERVED", "INPUT_MUTATED")


@dataclass(frozen=True)
class StepDecision:
    status: str
    reason: str
    neighbor_generation: int
    lr_generation: int
    lr_rebuilt: bool = False
    lr_reallocated: bool = False

    @property
    def accepted_eager(self):
        return self.status in ("READY", "RECAPTURE_REQUIRED")

    def public(self):
        return {"status": self.status, "reason": self.reason,
                "accepted_eager": self.accepted_eager,
                "neighbor_generation": self.neighbor_generation,
                "lr_generation": self.lr_generation,
                "lr_rebuilt": self.lr_rebuilt, "lr_reallocated": self.lr_reallocated,
                "all_ranks_decided": True, "capture_supported": False}


@dataclass(frozen=True)
class StepResult:
    decision: StepDecision
    local_energy: torch.Tensor | None
    owned_force: torch.Tensor | None
    global_energy: torch.Tensor | None
    diagnostics: dict


class OwnedEagerStep:
    """Collective prepare/evaluate/validate; eager execution only.

    All members call with leaves in canonical owned order. The detached
    pre-step gather is separate from the differentiable gather inside the
    model. This pays one extra coordinate exchange; timing includes it.
    """
    def __init__(self, evaluator, neighbor_epoch, *, total_charge,
                 charge_absolute_bar=1e-5, measure=False, charge_guard="ABSOLUTE"):
        self.evaluator = evaluator
        self.runtime = evaluator.runtime
        self.epoch = neighbor_epoch
        runtime = self.runtime
        collectives._outside_capture(runtime.device)
        neighbor_epoch.assert_integrity()
        _require(neighbor_epoch.metadata == runtime.metadata,
                 "STEP_EPOCH_OWNERSHIP_MISMATCH")
        _require(type(total_charge) in (int, float) and math.isfinite(total_charge) and
                 type(charge_absolute_bar) in (int, float) and
                 math.isfinite(charge_absolute_bar) and charge_absolute_bar > 0,
                 "STEP_CHARGE_GUARD_INVALID")
        _require(type(measure) is bool, "STEP_MEASURE_FLAG_INVALID")
        _require(type(charge_guard) is str and charge_guard in ("ABSOLUTE", "OBSERVE_ONLY"),
                 "STEP_CHARGE_GUARD_POLICY_INVALID")
        self.total_charge, self.charge_absolute_bar = float(total_charge), float(charge_absolute_bar)
        self.charge_guard = charge_guard
        self.measure = measure
        self.position_spec = collectives.prepare_exchange(runtime, (3,), torch.float32,
                                                          collectives.DETACHED_SOURCE)
        self.global_ids = torch.tensor(runtime.metadata.global_ids, dtype=torch.int64)
        self._global_ids_stamp = (id(self.global_ids), self.global_ids._version)
        self.cell = neighbor_epoch.cell.detach().clone()
        self.device_cell = self.cell.to(runtime.device)
        self._cell_stamp = self.cell._version
        self._device_cell_stamp = self.device_cell._version
        self._root = (dist.get_global_rank(runtime.group, 0)
                      if runtime.world_size > 1 and runtime.group is not None else 0)
        collectives._agreement(runtime, {"component": "owned_eager_step",
            "epoch": neighbor_epoch.epoch, "generation": neighbor_epoch.generation,
            "Q": self.total_charge, "charge_bar": self.charge_absolute_bar, "charge_guard": self.charge_guard,
            "cell": self.cell.tolist(), "measure": measure})

    def _sync(self):
        if self.measure and self.runtime.device.type == "cuda":
            torch.cuda.synchronize(self.runtime.device)

    def _all_max(self, code):
        value = torch.tensor([code], dtype=torch.int64, device=self.runtime.device)
        if self.runtime.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self.runtime.group)
        return int(value.item())

    def _broadcast(self, values):
        packet = torch.tensor(values if self.runtime.rank == 0 else [0] * 6,
                              dtype=torch.int64, device=self.runtime.device)
        if self.runtime.world_size > 1:
            dist.broadcast(packet, src=self._root, group=self.runtime.group)
        status, reason, ng, lg, rebuilt, reallocated = packet.cpu().tolist()
        _require(0 <= status < len(STATUS) and 0 <= reason < len(REASON),
                 "STEP_DECISION_PACKET_INVALID")
        return StepDecision(STATUS[status], REASON[reason], ng, lg, bool(rebuilt), bool(reallocated))

    def _empty(self, decision, diagnostics):
        diagnostics.update(forces_returned=False, state_advance_allowed=False)
        return StepResult(decision, None, None, None, diagnostics)

    def __call__(self, owned_positions):
        runtime = self.runtime
        collectives._outside_capture(runtime.device)
        self._sync()
        begin = time.perf_counter()
        code = 0
        if not (isinstance(owned_positions, torch.Tensor) and
                owned_positions.shape == (runtime.n_local, 3) and
                owned_positions.dtype == torch.float32 and owned_positions.device == runtime.device and
                owned_positions.requires_grad and owned_positions.is_leaf):
            code = REASON.index("POSITION_METADATA")
        elif not bool(torch.isfinite(owned_positions).all()):
            code = REASON.index("NONFINITE_POSITION")
        try:
            self.epoch.assert_integrity()
            if ((id(self.global_ids), self.global_ids._version) != self._global_ids_stamp or
                    self.cell._version != self._cell_stamp or self.device_cell._version != self._device_cell_stamp):
                code = max(code, REASON.index("EPOCH_INVALID"))
        except Exception:
            code = max(code, REASON.index("EPOCH_INVALID"))
        code = self._all_max(code)
        diagnostics = {"schema": "LARGO_OWNED_EAGER_STEP", "gpu_used": runtime.device.type == "cuda",
                       "timing_enabled": self.measure, "capture_supported": False,
                       "rattle_performed": False, "rattle_ms": 0.0,
                       "rank": runtime.rank, "global_atoms": runtime.n_global,
                       "owned_atoms": runtime.n_local,
                       "owned_count_sum_equals_global": sum(runtime.metadata.counts) == runtime.n_global,
                       "charge_guard": self.charge_guard, "charge_absolute_bar": self.charge_absolute_bar,
                       "charge_bar_applied": self.charge_guard == "ABSOLUTE",
                       "charge_observation": {"status": "UNAVAILABLE_BEFORE_EVALUATION", "target_e": self.total_charge,
                           "sum_dtype": "float64", "charge_dtype": None, "sum_e": None,
                           "residual_signed_e": None, "residual_abs_e": None,
                           "scope": "FP64_SUM_OF_REPLICATED_GLOBAL_FP32_CHARGES"}}
        if code:
            return self._empty(StepDecision("REJECT", REASON[code], self.epoch.generation, 0), diagnostics)
        input_version = owned_positions._version
        self._sync()
        gather_start = time.perf_counter()
        global_positions = collectives.gather_detached_sources(owned_positions.detach(), self.position_spec)
        self._sync()
        gather_end = time.perf_counter()
        values = [0, 0, self.epoch.generation, 0, 0, 0]
        neighbor_time = lr_prepare_time = 0.0
        if runtime.rank == 0:
            phase = "neighbor"
            try:
                t = time.perf_counter()
                skin = self.epoch.assess_skin(global_positions.detach().cpu(), self.cell)
                neighbor_time = time.perf_counter() - t
                if not skin.ready:
                    reason = "CELL_CHANGED" if "CELL_CHANGED" in skin.reasons else "SKIN_EXHAUSTED"
                    values[:2] = [STATUS.index("REBUILD_REQUIRED"), REASON.index(reason)]
                else:
                    phase = "lr"
                    self._sync()
                    t = time.perf_counter()
                    lr = self.evaluator.root_long_range.prepare(global_positions,
                            global_ids=self.global_ids, cell=self.device_cell)
                    self._sync()
                    lr_prepare_time = time.perf_counter() - t
                    values[3:] = [lr.generation, int(lr.rebuilt), int(lr.buffers_reallocated)]
                    if lr.status is LRStatus.REJECT:
                        values[:2] = [STATUS.index("REJECT"), REASON.index("LR_PREPARE_REJECTED")]
                    elif lr.status is LRStatus.RECAPTURE:
                        values[0] = STATUS.index("RECAPTURE_REQUIRED")
            except Exception:
                values[:2] = [STATUS.index("REJECT"), REASON.index(
                    "EPOCH_INVALID" if phase == "neighbor" else "LR_PREPARE_FAILED")]
        self._sync()
        decision_start = time.perf_counter()
        decision = self._broadcast(values)
        self._sync()
        decision_end = time.perf_counter()
        if self.measure:
            diagnostics.update(position_collect_ms=1000 * (gather_end - gather_start),
                               root_neighbor_check_ms=1000 * neighbor_time,
                               root_lr_prepare_ms=1000 * lr_prepare_time,
                               decision_broadcast_ms=1000 * (decision_end - decision_start))
        if not decision.accepted_eager:
            return self._empty(decision, diagnostics)
        # From here a rank-local unexpected exception is fatal to the group.
        # Never catch it and pretend remaining rank collectives can recover.
        model_start = time.perf_counter()
        result, force = self.evaluator.energy_and_force(owned_positions)
        self._sync()
        model_end = time.perf_counter()
        # Observe the already-computed charge tensor without changing any
        # projection/VJP arithmetic. Finite checks still apply in either mode.
        charge_finite = bool(torch.isfinite(result.global_conserved_charge).all())
        observation = diagnostics["charge_observation"]
        observation["charge_dtype"] = str(result.global_conserved_charge.dtype).removeprefix("torch.")
        if charge_finite:
            charge_sum = float(result.global_conserved_charge.detach().double().sum())
            charge_residual = charge_sum - self.total_charge
            charge_finite = math.isfinite(charge_sum) and math.isfinite(charge_residual)
            if charge_finite:
                observation.update(status="OBSERVED", sum_e=charge_sum,
                    residual_signed_e=charge_residual, residual_abs_e=abs(charge_residual))
        if not charge_finite:
            observation["status"] = "NONFINITE"
        code = 0
        if owned_positions._version != input_version:
            code = REASON.index("INPUT_MUTATED")
        elif not charge_finite or not bool(torch.isfinite(result.local_energy).all() & torch.isfinite(force).all()):
            code = REASON.index("NONFINITE_OUTPUT")
        else:
            if self.charge_guard == "ABSOLUTE" and observation["residual_abs_e"] > self.charge_absolute_bar:
                code = REASON.index("CHARGE_NOT_CONSERVED")
        code = self._all_max(code)
        if self.measure:
            diagnostics["evaluator_forward_backward_ms"] = 1000 * (model_end - model_start)
        if code:
            return self._empty(StepDecision("REJECT", REASON[code], decision.neighbor_generation,
                                           decision.lr_generation, decision.lr_rebuilt,
                                           decision.lr_reallocated), diagnostics)
        # Reporting SUM has no gradient. There is still exactly one loss seed
        # per rank, which was consumed by energy_and_force above.
        global_energy = result.local_energy.detach().double().clone()
        if runtime.world_size > 1:
            dist.all_reduce(global_energy, op=dist.ReduceOp.SUM, group=runtime.group)
        self._sync()
        if self.measure:
            diagnostics.update(eager_total_ms=1000 * (time.perf_counter() - begin),
                postcheck_energy_collect_ms=1000 * (time.perf_counter() - model_end))
        diagnostics.update(forces_returned=True, state_advance_allowed=True)
        return StepResult(decision, result.local_energy.detach(), force.detach(), global_energy, diagnostics)


__all__ = ["OwnedStepError", "StepDecision", "StepResult", "OwnedEagerStep"]
