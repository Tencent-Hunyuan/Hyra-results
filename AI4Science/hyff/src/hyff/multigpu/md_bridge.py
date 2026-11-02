"""Rank-zero MD bridge to a distributed force service created by the launcher.

The launcher manages the worker processes and collectives. This module keeps
MDRun output, restart loading, mass/HMR rules, thermostat RNG and the FP64
RATTLE arithmetic of the single-GPU run. It never creates a process group or
another model.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

import torch

from hyff.md.simulation import MDRun


class DistributedMDStepError(RuntimeError):
    """A distributed chunk was rejected without committing its state or time."""


def checked_service_physical_metadata(service, total_charge, multiplicity):
    """Read back the actual prepared service Q/M before its first force call."""
    read = getattr(service, 'metadata', None)
    if not callable(read):
        raise TypeError('distributed MD service must expose physical metadata')
    report = read()
    session = report.get('session') if isinstance(report, dict) else None
    backend = session.get('numerical_backend') if isinstance(session, dict) else None
    if not isinstance(backend, dict):
        raise ValueError('distributed MD service physical metadata is missing')
    charge, spin = backend.get('Q'), backend.get('M')
    if (type(charge) not in (int, float) or not math.isfinite(charge) or charge != total_charge or
            type(spin) is not int or spin != multiplicity):
        raise ValueError('distributed MD service Q/M differ from the run')
    return dict(total_charge=charge, multiplicity=spin, source='service.metadata.session.numerical_backend')


def checked_force_result(energy, force, positions):
    """Check the service's actual E64/F32 contract without casting its output."""
    if (not isinstance(energy, torch.Tensor) or energy.dtype != torch.float64
            or energy.device != positions.device or energy.shape not in ((), (1,))):
        raise ValueError('distributed MD energy must be one FP64 scalar on the force device')
    if (not isinstance(force, torch.Tensor) or force.dtype != torch.float32
            or force.device != positions.device or force.shape != positions.shape
            or not force.is_contiguous()):
        raise ValueError('distributed MD forces must be contiguous FP32 (N,3) on the force device')
    if not bool(torch.isfinite(energy).all()) or not bool(torch.isfinite(force).all()):
        raise FloatingPointError('distributed MD requires finite energy and forces')
    return energy.detach(), force.detach()


class _DistributedForceEngine:
    backend = 'eager'

    def __init__(self, service, positions):
        for method in ('evaluate', 'close', 'abort'):
            if not callable(getattr(service, method, None)):
                raise TypeError('distributed MD service requires evaluate, close and abort methods')
        if (not isinstance(positions, torch.Tensor) or not positions.is_cuda
                or positions.dtype != torch.float32 or positions.ndim != 2 or positions.shape[1] != 3
                or not positions.is_contiguous()):
            raise ValueError('distributed MD force coordinates must be contiguous CUDA FP32 (N,3)')
        if not bool(torch.isfinite(positions).all()):
            raise FloatingPointError('distributed MD initial force coordinates must be finite')
        if not hasattr(getattr(service, 'cache', None), 'rebuilds'):
            raise TypeError('distributed MD service must expose cache.rebuilds')
        self.service, self.cache = service, service.cache
        self.shape, self.device, self.closed = positions.shape, positions.device, False

    def __call__(self, batch):
        if self.closed:
            raise RuntimeError('distributed MD force service is closed')
        positions = batch['positions']
        if (not isinstance(positions, torch.Tensor) or positions.dtype != torch.float32
                or positions.device != self.device or positions.shape != self.shape
                or not positions.is_contiguous()):
            raise ValueError('distributed MD force coordinate contract changed')
        if not bool(torch.isfinite(positions).all()):
            raise FloatingPointError('distributed MD force coordinates must be finite')
        energy, force = self.service.evaluate(positions)
        return checked_force_result(energy, force, positions)

    def abort(self, code):
        return self.service.abort(code)

    def close(self):
        if not self.closed:
            self.closed = True
            self.service.close()


class DistributedMDRun(MDRun):
    """Run only on rank zero after the launcher has prepared all rank services.

    ``engine_factory(run, positions32, numbers, cell, Q, M)`` returns the normal
    model/batch/engine/metadata dictionary, with ``engine`` being the root force
    service. positions32 is the actual constraint-projected setup frame, not
    necessarily the bootstrap frame. The service owns latent-charge
    model loading and all neighbor/collective work; its evaluation must use the
    positions passed by this bridge on every call.
    """
    _STATE_TENSORS = ('position', 'velocity', 'force', 'previous', 'position_error', 'velocity_error')

    def __init__(self, cfg, *, engine_factory, system=None, initial_state=None, service_policy=None,
                 requested_service_policy=None):
        from hyff.multigpu.launcher import (
            service_precision, multigpu_options, _service_policy, _agree_service_policy,
        )
        self.service_policy = _agree_service_policy(_service_policy() if service_policy is None else service_policy,
            requested_service_policy=requested_service_policy)
        self.requested_service_policy = (_service_policy() if requested_service_policy is None else
                                         requested_service_policy)
        if requested_service_policy is not None and cfg.route != 'auto':
            raise ValueError('capacity policy override requires an auto route')
        if cfg.model != 'largo' or cfg.route not in ('multigpu', 'auto'):
            raise ValueError('multi-GPU MD requires the largo model and a multigpu/auto route')
        if type(cfg.ranks) is not int or cfg.ranks not in (1, 2, 4, 8):
            raise ValueError('distributed MD requires an already selected rank count')
        if cfg.engine not in (None, 'largo'):
            raise ValueError('multi-GPU MD accepts only the largo engine')
        if cfg.precision not in (None, 'fp32'):
            raise ValueError('multi-GPU MD runs in fp32; tf32x3 is not supported')
        if cfg.capture_retries != 0:
            raise ValueError('distributed MD does not retry capture or collective failures')
        if cfg.recenter_interval_ps is not None or cfg.long_range == 'spme':
            raise ValueError('multi-GPU MD uses the isolated long-range sum; recentering and SPME are not supported')
        if not callable(engine_factory):
            raise TypeError('distributed MD requires a callable engine factory')
        self._engine_factory = engine_factory
        self._root_system, self._root_initial_state = system, initial_state
        # The single-GPU largo engine allows only tf32x3; resolve its options
        # without presenting the multi-GPU fp32 service as one of them.
        super().__init__(replace(cfg, precision=None))
        self.cfg = cfg
        resolved_engine = self.kind
        # The launcher replaces the force engine, while the base setup chooses
        # its integration driver from this name. Do not inherit a checkpoint's
        # native driver or mislabel the distributed eager force service.
        # The cfg precision stays fp32/None; the node-precision switch is a process setting.
        self.service_precision = service_precision()
        self.kind = 'largo_multigpu'
        self.execution_options = dict(self.execution_options, precision=self.service_precision)
        self.result['options'] = dict(self.execution_options)
        self.result.update(gpus=cfg.ranks, requested_route=cfg.route, service_options=multigpu_options(), service_policy=self.service_policy,
            requested_service_policy=self.requested_service_policy,
            distributed_md=dict(energy_dtype='float64', force_dtype='float32',
                force_coordinate_dtype='float32', integration_state_dtype='float64',
                active_efm_compaction=self.service_policy.get('active_efm_compaction', False),
                active_base_pair_compaction=self.service_policy.get('active_base_pair_compaction', False),
                capacity_retention=self.service_policy.get('capacity_retention', False),
                share_efm_geometry=self.service_policy.get('share_efm_geometry', False),
                efm_tile_edges=self.service_policy.get('efm_tile_edges', 32768),
                resolved_native_engine=resolved_engine,
                checkpoint_model_owned_by_service=True, collective_retry=False,
                chunk_commit='only_after_all_steps_validate', recovery='terminal'))

    def _load_system(self, name):
        if self._root_system is not None:
            return self._root_system
        return super()._load_system(name)

    def _load_initial_positions(self, name):
        if self._root_initial_state is not None:
            return self._root_initial_state
        return super()._load_initial_positions(name)

    def _create_force_engine(self, positions32, numbers, cell, total_charge, multiplicity):
        made = self._engine_factory(self, positions32, numbers, cell, total_charge, multiplicity)
        if not isinstance(made, dict) or not {'model', 'batch', 'engine', 'metadata'} <= set(made):
            raise TypeError('distributed MD factory must return model, batch, engine and metadata')
        if not isinstance(made['batch'], dict) or not isinstance(made['metadata'], dict):
            raise TypeError('distributed MD batch and metadata must be dictionaries')
        if total_charge != float(self.sys['total_charge']) or multiplicity != self.multiplicity:
            raise ValueError('distributed MD factory Q/M differ from the run')
        physical_readback = checked_service_physical_metadata(made['engine'], total_charge, multiplicity)
        positions = torch.as_tensor(positions32, dtype=torch.float32,
            device=torch.device('cuda', torch.cuda.current_device())).contiguous()
        made = dict(made, batch=dict(made['batch'], positions=positions))
        made['engine'] = _DistributedForceEngine(made['engine'], positions)
        made['metadata'] = dict(made['metadata'], route='multigpu', ranks=self.cfg.ranks,
            precision=self.service_precision, service_policy=self.service_policy,
            active_base_pair_compaction=self.service_policy.get('active_base_pair_compaction', False),
            energy_dtype='float64', force_dtype='float32',
            multiplicity=multiplicity, total_charge=total_charge, service_physical_readback=physical_readback)
        return made

    def _capture_check(self, error=None):
        record = dict(guard_scope='distributed_force_service', force_backend='eager',
            force_capture_supported=False, integrator_phases='single_gpu_captured_FP64_RATTLE',
            force_coordinate_dtype='float32', energy_dtype='float64', force_dtype='float32',
            native_capture_guard_used=False)
        if error is not None:
            self.capture_failure = dict(error_type=type(error).__name__)
        if getattr(self, 'capture_failure', None) is not None:
            record.update(deepcopy(self.capture_failure))
        return record

    def _abort_service(self, error):
        attempt = dict(code=type(error).__name__)
        try:
            if self.engine is not None:
                attempt['result'] = self.engine.abort(type(error).__name__)
            else:
                attempt['result'] = dict(attempted=False, reason='ENGINE_NOT_ADOPTED',
                                        caller_must_terminate_peers=True)
        except BaseException as abort_error:
            self.result['distributed_abort_error_type'] = type(abort_error).__name__
            attempt['error_type'] = type(abort_error).__name__
        self.result.setdefault('distributed_abort_attempts', []).append(attempt)

    def _persist_restored_failure(self, error, frames=None, frame_times=None, wall_md=None):
        """Only a typed rolled-back chunk may save its last committed restart."""
        failure = self.result.get('distributed_failure', {})
        if (not isinstance(error, DistributedMDStepError) or
                failure.get('chunk_state_restored') is not True or failure.get('safe_restart_written')):
            return
        try:
            if frames is not None:
                self.flush(frames, frame_times, wall_md, final=True)
            else:
                # A custom root runner records its own timing. Do not write a zero
                # integration time or overwrite timing fields it already set.
                self._write_final_state()
                self.result.update(steps_done=self.steps_done, simulated_ps=self.sim_fs / 1000)
            failure['safe_restart_written'] = True
            failure['safe_restart_scope'] = 'LAST_COMMITTED_STATE_AFTER_CHUNK_RESTORE'
            self.save()
        except BaseException as report_error:
            self.result['failure_report_error_type'] = type(report_error).__name__

    def _failed_advance(self, error, frames, frame_times, wall_md):
        self._persist_restored_failure(error, frames, frame_times, wall_md)

    def advance(self, count):
        if type(count) is not int or count < 1:
            raise ValueError('distributed MD advance count must be a positive integer')
        if self.driver is not None or self.periodic_molecules is not None:
            raise RuntimeError('distributed MD requires the ordinary RATTLE callback path')
        state = {name: getattr(self.stepper, name).detach().clone() for name in self._STATE_TENSORS}
        original_steps = self.stepper.steps
        original_energy = self.energy.detach().clone()
        completed = 0

        def evaluate(current):
            self.batch['positions'] = current.float()
            return self.engine(self.batch)

        try:
            energy = None
            for _ in range(count):
                _, _, energy, _force, _ = self.stepper.step(evaluate)
                if not all(bool(torch.isfinite(getattr(self.stepper, name)).all())
                           for name in ('position', 'velocity', 'force')):
                    raise FloatingPointError('distributed MD produced nonfinite integration state')
                completed += 1
            self.energy = energy
            return count, energy
        except BaseException as error:
            restored = False
            try:
                with torch.no_grad():
                    for name, value in state.items():
                        getattr(self.stepper, name).copy_(value)
                self.stepper.steps = original_steps
                self.energy = original_energy
                self.batch['positions'] = self.stepper.position.float()
                restored = True
            except BaseException as restore_error:
                self.result['distributed_restore_error_type'] = type(restore_error).__name__
            self.result.update(status='failed', nve_interrupted=self.cfg.tau_ps == 0,
                steps_done=self.steps_done, simulated_ps=self.sim_fs / 1000,
                distributed_failure=dict(error_type=type(error).__name__, chunk_state_restored=restored,
                    discarded_completed_steps=completed, attempted_chunk_steps=count,
                    incomplete_chunk_committed=False,
                    diagnostic_hint="Inspect the failing rank's stderr for its original typed error; "
                                    "a transport failure can prevent that code from reaching rank 0."))
            self._abort_service(error)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            # Preserve the original error as the cause, but do not let a native
            # PhaseCheckError's counters trigger base-run recovery or commits.
            message = ('distributed MD chunk failed; prior committed state restored' if restored else
                       'distributed MD chunk failed; rollback failed and no chunk was committed')
            message += "; inspect the failing rank's stderr for its original typed error"
            raise DistributedMDStepError(message) from error

    def recover(self, error):
        self.result.update(status='failed', recovery_disabled=True, nve_interrupted=self.cfg.tau_ps == 0)
        self._abort_service(error)
        raise DistributedMDStepError("distributed MD errors are terminal; no collective retry; "
                                     "inspect the failing rank's stderr for its original typed error") from error
