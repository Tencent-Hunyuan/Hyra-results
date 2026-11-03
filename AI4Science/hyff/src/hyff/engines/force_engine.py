from __future__ import annotations

import time
import json
import math
import os

import torch

from hyff.data import physical
from hyff.graph import NeighborCache, PeriodicEdges, energy_forces


class CaptureEquivalenceError(AssertionError):
    """A rejected force evaluation, keeping the exact inputs for diagnosis."""

    def __init__(self, details, snapshot):
        self.details, self.snapshot = details, snapshot
        super().__init__(
            f"Captured energy/forces differ from the reference: {details}"
        )


class CaptureGrossDeploymentError(CaptureEquivalenceError):
    """Strict replay passed, but the gross bound against the plain-model branch failed."""

    def __init__(self, details, snapshot):
        super().__init__(details, snapshot)
        self.args = ("Prepared evaluator exceeds gross branch reference bound: " + str(details),)


_PREPARED_PREFIX = "prepared_replay_vs_prepared_uncaptured_"
_BRANCH_PREFIX = "branch_reference_vs_prepared_"


class ForceEngine:
    """Checked Verlet neighbors plus an optional exact CUDA graph execution.

    Every call copies current coordinates and recomputes the SAME energy and
    its gradient. Only connectivity within a valid skin and executable kernels
    are reused. Edge buffers grow when necessary; recapture cost is included
    whenever it happens inside a timed trajectory. No neighbors are truncated.
    """

    def __init__(self, model, batch, backend="eager", skin=0.6, *,
                 capture_force_atol=None, capture_diagnostics=None,
                 capture_branch_policy="first"):
        if backend not in ("eager", "cuda_graph", "compiled"):
            raise ValueError(backend)
        if capture_branch_policy not in ("off", "first", "every"):
            raise ValueError("capture_branch_policy must be off, first or every")
        self.model, self.backend = model, backend
        self.capture_branch_policy = capture_branch_policy
        self.capture_branch_status = None
        self.capture_branch_last_check = None
        self.capture_guard_diagnostics = None
        self._capture_eligible_successes = 0
        # Optional settings: absolute force tolerance and a diagnostics file.
        # Relative force and both energy tolerances are fixed. A relaxed absolute
        # force tolerance must be validated for the model and precision in use.
        self.capture_force_atol = float(os.environ.get("MLFF_CAPTURE_FORCE_ATOL", "1e-4")
                                        if capture_force_atol is None else capture_force_atol)
        if not math.isfinite(self.capture_force_atol) or self.capture_force_atol < 0:
            raise ValueError("capture_force_atol must be finite and nonnegative")
        self.capture_diagnostics = (os.environ.get("MLFF_CAPTURE_DIAGNOSTICS")
                                    if capture_diagnostics is None else capture_diagnostics)
        if backend == "compiled" and (getattr(model,"fast_nodes",False) or getattr(model,"fused",False) or
                (hasattr(model,"blocks") and any(getattr(b,"fused",False) for b in model.blocks))):
            raise ValueError("Use cuda_graph with Triton messages; compiled mode currently uses the reference PyTorch messages")
        self.cache = NeighborCache(model.cutoff, skin)
        self.batch = physical(batch)
        self.current_edges = None
        self.static_edges = None
        self.current_field_edges = None
        self.static_field_edges = None
        self.graph = None
        self.captures = 0
        self.capture_seconds = 0.0
        self.solver_recaptures = 0
        self.symbolic_recaptures = 0
        self.symbolic_history = None
        if backend in ("cuda_graph", "compiled") and self.batch["positions"].device.type != "cuda":
            raise ValueError("CUDA graph backend requires a CUDA device")

    @staticmethod
    def _edge_buffer(batch,edges):
        capacity = max(256, ((int(edges.shape[1] * 1.2) + 255) // 256) * 256)
        raw_edges = edges.edge_index if isinstance(edges, PeriodicEdges) else edges
        buffer = torch.zeros((2, capacity), dtype=raw_edges.dtype, device=raw_edges.device)
        # Distribute inactive self edges across atoms. Sending every padding
        # edge to atom zero causes extreme atomic-scatter contention in MD.
        padding_nodes = torch.arange(capacity, device=buffer.device) % len(batch["positions"])
        static_edges = PeriodicEdges(buffer, torch.zeros((capacity, 3), dtype=torch.int32, device=buffer.device)) if isinstance(edges, PeriodicEdges) else buffer
        return static_edges,padding_nodes

    def _capture(self, batch, edges,minimum_solver_budget=0):
        started = time.perf_counter()
        # Release the previous graph before allocating a replacement.
        self.graph=None
        previous=getattr(self,'_prepared_energy',None)
        if hasattr(previous,'close'):previous.close()
        self._prepared_energy=None
        self.static_edges,self.padding_nodes=self._edge_buffer(batch,edges)
        if self.current_field_edges is not None:
            self.static_field_edges,self.field_padding_nodes=self._edge_buffer(batch,self.current_field_edges)
        self._copy_edges(edges)
        self.static_batch = {key: value.detach().clone() for key, value in physical(batch).items()}
        self.static_batch["positions"].requires_grad_(True)
        # Paths that are not prepared always compare with the model-forward
        # reference evaluated before preparation. Eligible prepared paths can
        # omit this costly branch; the strict replay check still runs against
        # their third uncaptured warmup.
        prepared_path = (
            self.backend == "cuda_graph" and minimum_solver_budget == 0
            and hasattr(self.model, "prepare_energy")
            and not hasattr(self.model, "prepare_field_edges")
            and self.current_field_edges is None and self.symbolic_history is None
        )
        branch_requested = prepared_path and (
            self.capture_branch_policy == "every"
            or self.capture_branch_policy == "first" and self._capture_eligible_successes == 0
        )
        branch_e = branch_f = None
        if not prepared_path or branch_requested:
            reference_e, reference_f = energy_forces(self.model, batch, edges)
            branch_e, branch_f = reference_e.detach().clone(), reference_f.detach().clone()
            del reference_e, reference_f
        # Sparse-library data keeps the stream chosen before its analysis.
        # Preparation, warmup and capture therefore share one private stream.
        stream = torch.cuda.Stream(device=batch['positions'].device)
        stream.wait_stream(torch.cuda.current_stream(batch['positions'].device))
        with torch.cuda.stream(stream):
            if hasattr(self.model, "prepare_energy"):
                if hasattr(self.model,"prepare_field_edges"):
                    extra={'symbolic_history':self.symbolic_history} if self.symbolic_history is not None else {}
                    evaluate=self.model.prepare_energy(self.static_batch,self.static_edges,
                        field_edges=self.static_field_edges,minimum_solver_budget=minimum_solver_budget,**extra)
                else:
                    evaluate = self.model.prepare_energy(self.static_batch, self.static_edges)
            else:
                evaluate = lambda positions: self.model.energy({**self.static_batch, "positions": positions}, self.static_edges)
        # CUDA graphs hold addresses, not Python references to input tensors.
        # Prepared evaluator attributes/offsets must outlive every graph replay.
        self._prepared_energy = evaluate
        position = self.static_batch["positions"]
        use_prepared_reference = prepared_path and not any(
            hasattr(evaluate, name) for name in ("validate_result", "symbolic_valid", "solver_plan")
        )
        if not use_prepared_reference and branch_e is None:
            # Solver markers on the evaluator are known only after preparation.
            # When the branch was skipped (policy off, or first already done),
            # this model-forward reference runs after prepare but still BEFORE
            # warmup and graph allocation. No extra preparation is introduced.
            reference_e, reference_f = energy_forces(self.model, batch, edges)
            branch_e, branch_f = reference_e.detach().clone(), reference_f.detach().clone()
            del reference_e, reference_f
        self.capture_branch_status = dict(
            policy=self.capture_branch_policy, eligible=use_prepared_reference,
            checked=bool(use_prepared_reference and branch_requested),
            reason=("uncaptured_reference_required" if not use_prepared_reference
                    else "policy_off" if self.capture_branch_policy == "off"
                    else "first_already_succeeded" if not branch_requested
                    else "every_eligible_capture" if self.capture_branch_policy == "every"
                    else "first_eligible_capture"),
            prior_successful_eligible_captures=self._capture_eligible_successes,
            prior_successful_captures=self.captures,
        )

        if self.backend == "compiled":
            # Compile the energy AND its exact coordinate derivative. There is
            # no independent force predictor or changed interaction graph.
            self._compiled_gradient = torch.compile(
                torch.func.grad_and_value(lambda p: evaluate(p).sum()),
                fullgraph=True, dynamic=False, options={"triton.cudagraphs": False},
            )
            def compute():
                gradient, total = self._compiled_gradient(position)
                return total.reshape(1), -gradient
        else:
            def compute():
                energy = evaluate(position)
                force = -torch.autograd.grad(energy.sum(), position)[0]
                return energy, force

        with torch.cuda.stream(stream), torch.enable_grad():
            for _ in range(3):
                energy, force = compute()
            if use_prepared_reference:
                expected_e, expected_f = energy.detach().clone(), force.detach().clone()
        torch.cuda.current_stream(position.device).wait_stream(stream)
        torch.cuda.synchronize(position.device)
        del energy, force
        if not use_prepared_reference:
            expected_e, expected_f = branch_e, branch_f
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), torch.enable_grad():
            self.static_energy, self.static_force = compute()
        torch.cuda.synchronize(position.device)
        self.graph = graph
        graph.replay()
        if hasattr(evaluate,"validate_result") and not evaluate.validate_result():
            raise RuntimeError(f"Calibrated field capture failed its true-residual check: {evaluate.solver_plan.description()}")
        if use_prepared_reference:
            self._check_prepared_capture(batch, edges, expected_e, expected_f,
                                         branch_e, branch_f, branch_requested)
            self._capture_eligible_successes += 1
            self.captures += 1
            self.capture_seconds += time.perf_counter() - started
            return
        try:
            torch.testing.assert_close(self.static_energy, expected_e, rtol=5e-6,
                                       atol=1e-5 * len(position))
            torch.testing.assert_close(self.static_force, expected_f, rtol=1e-4,
                                       atol=self.capture_force_atol)
        except AssertionError as error:
            # A failed evaluation must not
            # return a force; preserve enough state to reproduce it instead.
            actual = self.static_force.detach()
            delta = actual - expected_f.detach()
            tolerance = self.capture_force_atol + 1e-4 * expected_f.detach().abs()
            ratio = delta.abs() / tolerance
            worst = torch.topk(ratio.flatten(), min(12, ratio.numel())).indices
            details = dict(
                n_atoms=len(position), prior_successful_captures=self.captures,
                neighbor_rebuilds=self.cache.rebuilds,
                symbolic_recaptures=self.symbolic_recaptures,
                force_max_error_eV_A=float(delta.abs().max()),
                force_rms_error_eV_A=float(delta.double().square().mean().sqrt()),
                force_max_reference_eV_A=float(expected_f.detach().abs().max()),
                force_max_captured_eV_A=float(actual.abs().max()),
                energy_error_eV=float((self.static_energy.detach()-expected_e.detach()).abs().max()),
                force_components_outside_tolerance=int((~torch.isclose(actual,expected_f,rtol=1e-4,atol=self.capture_force_atol)).sum()),
                force_max_tolerance_ratio=float(ratio.max()),
                worst_components=[dict(atom=int(i)//3,axis=int(i)%3,
                    captured=float(actual.flatten()[i]),reference=float(expected_f.flatten()[i]),
                    error=float(delta.flatten()[i])) for i in worst],
                solver=evaluate.solver_plan.description() if hasattr(evaluate,'solver_plan') else None,
            )
            def cpu_edges(graph):
                if graph is None:return None
                if isinstance(graph,PeriodicEdges):
                    return dict(edge_index=graph.edge_index.detach().cpu(),
                                unit_shifts=graph.unit_shifts.detach().cpu())
                return dict(edge_index=graph.detach().cpu())
            snapshot=dict(batch={k:v.detach().cpu() for k,v in physical(batch).items()},
                edges=cpu_edges(edges),field_edges=cpu_edges(self.current_field_edges),
                captured_energy=self.static_energy.detach().cpu(),captured_force=actual.cpu(),
                reference_energy=expected_e.detach().cpu(),reference_force=expected_f.detach().cpu(),
                details=details)
            self._record_capture_check(expected_e, expected_f, accepted=False)
            raise CaptureEquivalenceError(details,snapshot) from error
        self._record_capture_check(expected_e, expected_f, accepted=True)
        self.captures += 1
        self.capture_seconds += time.perf_counter() - started

    def _prepared_comparison_metrics(self, actual_e, actual_f, reference_e, reference_f,
                                     prefix, *, branch=False):
        actual, reference = actual_f.detach(), reference_f.detach()
        delta = actual - reference
        tolerance = (1e-4 if branch else self.capture_force_atol) + 1e-4 * reference.abs()
        energy_tolerance = 1e-5 * len(reference) + 5e-6 * reference_e.detach().abs()
        return {
            prefix + "force_max_error_eV_A": float(delta.abs().max()),
            prefix + "force_rms_error_eV_A": float(delta.double().square().mean().sqrt()),
            prefix + "force_max_reference_eV_A": float(reference.abs().max()),
            prefix + ("force_max_prepared_eV_A" if branch else "force_max_captured_eV_A"): float(actual.abs().max()),
            prefix + "energy_error_eV": float((actual_e.detach() - reference_e.detach()).abs().max()),
            prefix + "force_components_outside_default_tolerance": int(
                (~torch.isclose(actual, reference, rtol=1e-4, atol=1e-4)).sum()),
            prefix + "force_components_outside_tolerance": int(
                (~torch.isclose(actual, reference, rtol=1e-4, atol=self.capture_force_atol)).sum()),
            prefix + "force_max_tolerance_ratio": float((delta.abs() / tolerance).max()),
            prefix + "energy_tolerance_ratio": float(
                ((actual_e.detach() - reference_e.detach()).abs() / energy_tolerance).max()),
        }

    def _check_prepared_capture(self, batch, edges, expected_e, expected_f,
                                branch_e, branch_f, branch_requested):
        """Strict replay is mandatory; the optional branch has a separate bound."""
        branch_metrics = (self._prepared_comparison_metrics(
            expected_e, expected_f, branch_e, branch_f, _BRANCH_PREFIX, branch=True)
            if branch_requested else {})
        strict_metrics = self._prepared_comparison_metrics(
            self.static_energy, self.static_force, expected_e, expected_f, _PREPARED_PREFIX)
        if branch_requested:
            self.capture_branch_last_check = dict(self.capture_branch_status, **branch_metrics)
        record = dict(time_unix=time.time(), pid=os.getpid(), accepted=False,
            n_atoms=len(expected_f), prior_successful_captures=self.captures,
            max_abs_force_coordinate_A=float(self.static_batch["positions"].detach().abs().max()),
            force_atol_eV_A=self.capture_force_atol, force_rtol=1e-4,
            energy_atol_eV=1e-5 * len(expected_f), energy_rtol=5e-6,
            guard_semantics="prepared_replay_vs_prepared_uncaptured",
            compared_arms="strict: captured prepared replay vs detached third uncaptured warmup; "
                          "optional bound: modified-model branch reference vs prepared evaluator",
            capture_branch_policy=self.capture_branch_policy,
            branch_checked=self.capture_branch_status["checked"],
            branch_check_reason=self.capture_branch_status["reason"],
            **strict_metrics, **branch_metrics)

        def record_check(accepted):
            record["accepted"] = accepted
            self.capture_guard_diagnostics = dict(record)
            if self.capture_diagnostics:
                with open(self.capture_diagnostics, "a") as log:
                    log.write(json.dumps(record) + "\n")

        try:
            torch.testing.assert_close(self.static_energy, expected_e, rtol=5e-6,
                                       atol=1e-5 * len(expected_f))
            torch.testing.assert_close(self.static_force, expected_f, rtol=1e-4,
                                       atol=self.capture_force_atol)
        except AssertionError as error:
            # Drop the graph before diagnostic allocation or file I/O can
            # fail, so neither path can leave a rejected graph replayable.
            self.graph = None
            delta = self.static_force.detach() - expected_f.detach()
            tolerance = self.capture_force_atol + 1e-4 * expected_f.detach().abs()
            ratio = delta.abs() / tolerance
            worst = torch.topk(ratio.flatten(), min(12, ratio.numel())).indices
            details = dict(record, neighbor_rebuilds=self.cache.rebuilds,
                           symbolic_recaptures=self.symbolic_recaptures, solver=None)
            details[_PREPARED_PREFIX + "worst_components"] = [dict(
                atom=int(i) // 3, axis=int(i) % 3,
                captured=float(self.static_force.detach().flatten()[i]),
                reference=float(expected_f.detach().flatten()[i]),
                error=float(delta.flatten()[i])) for i in worst]

            def cpu_edges(graph_edges):
                if graph_edges is None:
                    return None
                if isinstance(graph_edges, PeriodicEdges):
                    return dict(edge_index=graph_edges.edge_index.detach().cpu(),
                                unit_shifts=graph_edges.unit_shifts.detach().cpu())
                return dict(edge_index=graph_edges.detach().cpu())

            snapshot = dict(batch={key: value.detach().cpu() for key, value in physical(batch).items()},
                edges=cpu_edges(edges), field_edges=cpu_edges(self.current_field_edges),
                captured_energy=self.static_energy.detach().cpu(),
                captured_force=self.static_force.detach().cpu(),
                reference_energy=expected_e.detach().cpu(), reference_force=expected_f.detach().cpu(),
                details=details)
            record_check(False)
            raise CaptureEquivalenceError(details, snapshot) from error
        # Reject when the prepared result is more than 10x the tolerance away from the
        # plain-model branch reference.
        if branch_requested and (
            branch_metrics[_BRANCH_PREFIX + "force_max_tolerance_ratio"] > 10.
            or branch_metrics[_BRANCH_PREFIX + "energy_tolerance_ratio"] > 10.
        ):
            self.graph = None
            record.update(rejection_reason="gross_branch_bound",
                          gross_branch_tolerance_ratio_limit=10.)
            record_check(False)
            details = dict(record)
            raise CaptureGrossDeploymentError(details, dict(details=details))
        record_check(True)

    def _record_capture_check(self, expected_e, expected_f, *, accepted):
        """Record the capture check; with a diagnostics file, append one JSONL line per capture, never per MD step."""
        # Without a diagnostics file, skip the tensor reductions below.
        self.capture_guard_diagnostics = dict(
            accepted=bool(accepted), n_atoms=len(expected_f),
            prior_successful_captures=self.captures,
            guard_semantics="reference_branch_guard",
            compared_arms="captured evaluator replay vs uncaptured model-forward reference evaluated before capture")
        if not self.capture_diagnostics:
            return
        actual = self.static_force.detach()
        reference = expected_f.detach()
        delta = actual - reference
        record = dict(time_unix=time.time(), pid=os.getpid(), accepted=accepted,
            n_atoms=len(reference), prior_successful_captures=self.captures,
            max_abs_force_coordinate_A=float(self.static_batch['positions'].detach().abs().max()),
            force_atol_eV_A=self.capture_force_atol, force_rtol=1e-4,
            energy_atol_eV=1e-5*len(reference), energy_rtol=5e-6,
            force_max_error_eV_A=float(delta.abs().max()),
            force_rms_error_eV_A=float(delta.double().square().mean().sqrt()),
            force_max_reference_eV_A=float(reference.abs().max()),
            energy_error_eV=float((self.static_energy.detach()-expected_e.detach()).abs().max()),
            force_components_outside_default_tolerance=int((~torch.isclose(actual,reference,rtol=1e-4,atol=1e-4)).sum()),
            force_components_outside_tolerance=int((~torch.isclose(actual,reference,rtol=1e-4,atol=self.capture_force_atol)).sum()))
        self.capture_guard_diagnostics = dict(record,
            guard_semantics="reference_branch_guard",
            compared_arms="captured evaluator replay vs uncaptured model-forward reference evaluated before capture")
        if self.capture_diagnostics:
            with open(self.capture_diagnostics, 'a') as stream:
                stream.write(json.dumps(record) + '\n')

    @staticmethod
    def _copy_edge_buffer(edges,static_edges,padding_nodes):
        if isinstance(edges, PeriodicEdges):
            static_edges.edge_index.copy_(padding_nodes[None].expand(2, -1))
            static_edges.unit_shifts.zero_()
            static_edges.edge_index[:, :edges.shape[1]].copy_(edges.edge_index)
            static_edges.unit_shifts[:edges.shape[1]].copy_(edges.unit_shifts)
        else:
            static_edges.copy_(padding_nodes[None].expand(2, -1))
            static_edges[:, :edges.shape[1]].copy_(edges)

    def _copy_edges(self,edges):
        self._copy_edge_buffer(edges,self.static_edges,self.padding_nodes)
        if self.current_field_edges is not None:
            self._copy_edge_buffer(self.current_field_edges,self.static_field_edges,self.field_padding_nodes)

    def __call__(self, batch):
        edges = self.cache.get(batch)
        if self.backend == "eager":
            return energy_forces(self.model, batch, edges)
        if edges is not self.current_edges and hasattr(self.model,"prepare_field_edges"):
            self.current_field_edges=self.model.prepare_field_edges(batch,edges,self.cache.skin)
        field_grew=self.current_field_edges is not None and (self.static_field_edges is None or self.current_field_edges.shape[1]>self.static_field_edges.shape[1])
        cell_changed=self.graph is not None and "cell" in batch and not torch.equal(self.static_batch["cell"],batch["cell"])
        moved_grid=self.graph is not None and hasattr(self._prepared_energy,"needs_recapture") and self._prepared_energy.needs_recapture(batch)
        if self.graph is None or edges.shape[1] > self.static_edges.shape[1] or field_grew or cell_changed or moved_grid:
            self._capture(batch, edges)
        elif edges is not self.current_edges:
            self._copy_edges(edges)
            if hasattr(self._prepared_energy,"update_edges") and not self._prepared_energy.update_edges():
                self._capture(batch,edges)
        self.current_edges = edges
        with torch.no_grad():
            self.static_batch["positions"].copy_(batch["positions"])
            if "cell" in batch:
                self.static_batch["cell"].copy_(batch["cell"])
        self.graph.replay()
        if hasattr(self._prepared_energy,'symbolic_valid') and not self._prepared_energy.symbolic_valid():
            # No result from an incomplete matrix may escape the engine.
            # Analyze the current graph and replay before checking residuals.
            self.symbolic_recaptures+=1
            self._capture(batch,edges)
            self.graph.replay()
            if not self._prepared_energy.symbolic_valid():
                raise RuntimeError('Fresh sparse analysis failed to cover every nonzero edge')
        while hasattr(self._prepared_energy,"validate_result") and not self._prepared_energy.validate_result():
            plan=self._prepared_energy.solver_plan
            if getattr(plan.components,'backend',None)=='cudss':
                if getattr(plan.components,'can_refresh_preconditioner',lambda:False)():
                    # An inexact preconditioner may become too weak as new
                    # couplings grow. Refresh its full current pattern, then
                    # retry; no force from the failed residual is returned.
                    self.solver_recaptures+=1
                    self._capture(batch,edges)
                    self.graph.replay()
                    continue
                raise RuntimeError(f'Direct field solve failed its original-operator residual check; force was not returned: {plan.description()}')
            if plan.budget>=plan.maximum:
                raise RuntimeError(f"Field solve did not converge; force was not returned: {plan.description()}")
            budget=min(plan.maximum,plan.budget*2)
            self.solver_recaptures+=1
            self._capture(batch,edges,minimum_solver_budget=budget)
            self.graph.replay()
        # Results own their storage, so callers can retain consecutive frames.
        return self.static_energy.detach().clone(), self.static_force.detach().clone()

    def close(self):
        self.graph=None
        prepared=getattr(self,'_prepared_energy',None)
        if hasattr(prepared,'close'):prepared.close()
        self._prepared_energy=None
        # Captured outputs also own private-pool allocations. Returned E/F
        # are independent clones, so releasing these does not invalidate them.
        for name in ('static_energy','static_force','static_batch','static_edges',
                     'static_field_edges','_compiled_gradient'):
            if hasattr(self,name):setattr(self,name,None)
