"""Two checked GPU phases for fixed-cell HyFF/RATTLE MD.

The first phase drifts and checks both constraints and the Verlet skin before
any force evaluation. A skin miss rebuilds at the actual drifted coordinates
and evaluates the force once, so no speculative force evaluation or
rollback/recomputation is needed.
"""
from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

from hyff.md.units import ACCELERATION


@triton.jit
def _pre_check(P, V, X, Ref, Error, Flags, N: tl.constexpr, G: tl.constexpr,
               LIMIT2: tl.constexpr, TOL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    atom = i < N
    displacement2 = tl.full((BLOCK,), 0., tl.float64)
    nonfinite = tl.full((BLOCK,), False, tl.int1)
    for component in tl.static_range(3):
        x = tl.load(X + 3*i + component, atom, 0.)
        ref = tl.load(Ref + 3*i + component, atom, 0.)
        p = tl.load(P + 3*i + component, atom, 0.)
        v = tl.load(V + 3*i + component, atom, 0.)
        delta = x.to(tl.float64) - ref.to(tl.float64)
        displacement2 += delta * delta
        nonfinite |= atom & ((x != x) | (tl.abs(x) == float('inf')) |
            (p != p) | (tl.abs(p) == float('inf')) |
            (v != v) | (tl.abs(v) == float('inf')))
    outside = atom & (displacement2 >= LIMIT2)
    error = tl.load(Error + i, i < G, 0.)
    constraint = (i < G) & ((error > TOL) | (error != error) |
                            (tl.abs(error) == float('inf')))
    tl.atomic_min(Flags, 1, mask=tl.max(outside.to(tl.int32), 0) > 0)
    tl.atomic_min(Flags + 1, 1, mask=tl.max(constraint.to(tl.int32), 0) > 0)
    tl.atomic_min(Flags + 3, 1, mask=tl.max(nonfinite.to(tl.int32), 0) > 0)
    # X is never sanitized: a skin miss must rebuild at these actual positions.


@triton.jit
def _post_check(P, V, F, E, Error, Flags, N: tl.constexpr, G: tl.constexpr,
                TOL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    atom = i < N
    nonfinite = tl.full((BLOCK,), False, tl.int1)
    bad_force = tl.full((BLOCK,), False, tl.int1)
    for component in tl.static_range(3):
        p = tl.load(P + 3*i + component, atom, 0.)
        v = tl.load(V + 3*i + component, atom, 0.)
        f = tl.load(F + 3*i + component, atom, 0.)
        nonfinite |= atom & ((p != p) | (tl.abs(p) == float('inf')) |
                             (v != v) | (tl.abs(v) == float('inf')))
        bad_force |= atom & ((f != f) | (tl.abs(f) == float('inf')))
    error = tl.load(Error + i, i < G, 0.)
    constraint = (i < G) & ((error > TOL) | (error != error) |
                            (tl.abs(error) == float('inf')))
    energy = tl.load(E)
    bad_energy = (energy != energy) | (tl.abs(energy) == float('inf'))
    tl.atomic_min(Flags + 2, 1, mask=tl.max(constraint.to(tl.int32), 0) > 0)
    tl.atomic_min(Flags + 4, 1, mask=tl.max(nonfinite.to(tl.int32), 0) > 0)
    tl.atomic_min(Flags + 5, 1,
                  mask=(tl.max(bad_force.to(tl.int32), 0) > 0) | bad_energy)


class PhaseCheckError(FloatingPointError):
    def __init__(self, flags, phase, committed_steps):
        self.details = dict(flags=flags, phase=phase,
            committed_steps=committed_steps, failing_step_committed=False,
            failing_step_restored=True)
        super().__init__(f'Checked MD phase failed: {self.details}')


class CheckedPhases:
    """Own a previously checked ForceEngine and CapturedRattleStep."""
    def __init__(self, engine, stepper, batch):
        if engine.graph is None or engine.current_edges is None:
            raise ValueError('Initialize and check the force engine first')
        if len(batch['ptr']) != 2 or 'cell' not in batch:
            raise ValueError('One fixed periodic system is required')
        if engine.static_energy.numel() != 1:
            raise ValueError('Expected exactly one system energy')
        c = stepper.constraints
        if (c.solver, c.check, c.tolerance) != ('coupled', True, 1e-10):
            raise ValueError('Preserve the checked coupled RATTLE policy')
        if any(hasattr(engine._prepared_energy, attr) for attr in
               ('validate_result', 'needs_recapture', 'symbolic_valid')):
            raise ValueError('This driver handles local HyFF without field solvers')
        self.engine, self.stepper, self.batch = engine, stepper, batch
        self.cell_guard = (batch['cell'], batch['cell']._version, batch['cell'].data_ptr())
        self.reference = engine.cache.reference.detach().clone()
        self.flags = torch.empty(6, dtype=torch.int32, device=self.reference.device)
        self.previous_velocity = stepper.velocity.detach().clone()
        self.previous_force = stepper.force.detach().clone()
        self.signature = None
        self.pre_graph = self.force_post_graph = None
        self.energy = None
        self.last_energy = engine.static_energy.detach().clone()
        self.committed_steps = self.attempted_steps = self.rebuild_steps = 0
        self.phase_captures = self.capture_force_evaluations = 0
        self.capture_seconds = 0.
        self.failed = False
        self.limit2 = (engine.cache.skin * .5 * (1. - 2.**-20))**2

    def _guard(self):
        if self.failed:
            raise RuntimeError('Close a failed driver; it cannot resume a trajectory')
        cell, version, pointer = self.cell_guard
        if (self.batch['cell'] is not cell or cell._version != version or
                cell.data_ptr() != pointer):
            raise ValueError('A changed cell requires a new checked MD driver')
        s, c = self.stepper, self.stepper.constraints
        if (c.solver, c.tolerance, c.max_iterations, c.check) != s.constraint_policy:
            raise ValueError('The captured constraint policy changed')
        if any(getattr(c, key)._version != version
               for key, version in s.constraint_versions.items()):
            raise ValueError('The integrator masses or constraints changed')

    @torch.no_grad()
    def _pre(self):
        s, c = self.stepper, self.stepper.constraints
        self.flags.fill_(2)
        s.previous.copy_(s.position)
        self.previous_velocity.copy_(s.velocity)
        self.previous_force.copy_(s.force)
        s.velocity.add_(.5*s.dt*ACCELERATION*s.force/s.mass[:, None])
        s.position.add_(s.dt*s.velocity)
        s._project(s.previous, True)
        p = self.engine.static_batch['positions']
        p.copy_(s.position)
        _pre_check[(triton.cdiv(max(len(p), len(c.error)), 256),)](
            s.position, s.velocity, p, self.reference, s.position_error, self.flags,
            len(p), len(c.error), self.limit2, c.tolerance, 256,
            num_warps=4, enable_fp_fusion=False)

    @torch.no_grad()
    def _post(self, energy):
        s, c = self.stepper, self.stepper.constraints
        s.velocity.add_(.5*s.dt*ACCELERATION*s.force/s.mass[:, None])
        s._project(s.position, False)
        _post_check[(triton.cdiv(max(len(s.position), len(c.error)), 256),)](
            s.position, s.velocity, s.force, energy, s.velocity_error, self.flags,
            len(s.position), len(c.error), c.tolerance, 256,
            num_warps=4, enable_fp_fusion=False)

    def _force_post(self):
        p = self.engine.static_batch['positions']
        with torch.enable_grad():
            energy = self.engine._prepared_energy(p)
            force = -torch.autograd.grad(energy.sum(), p)[0]
        with torch.no_grad():
            self.stepper.force.copy_(force)
        self._post(energy)
        return energy

    def _capture(self):
        started = time.perf_counter()
        self.pre_graph = self.force_post_graph = None
        s = self.stepper
        targets = (s.position, s.velocity, s.force,
                   self.engine.static_batch['positions'])
        saved = tuple(t.detach().clone() for t in targets)

        @torch.no_grad()
        def restore():
            for target, source in zip(targets, saved):
                target.copy_(source)
            s.previous.copy_(s.position)

        stream = torch.cuda.Stream(device=s.position.device)
        stream.wait_stream(torch.cuda.current_stream(s.position.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._pre()
                restore()
                # Warm force/post at the valid saved geometry, never at a
                # speculative next geometry outside its checked skin.
                self._force_post()
                restore()
        torch.cuda.current_stream(s.position.device).wait_stream(stream)
        torch.cuda.synchronize(s.position.device)
        pre, force_post = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(pre, stream=stream):
            self._pre()
        torch.cuda.synchronize(s.position.device)
        restore()
        torch.cuda.synchronize(s.position.device)
        with torch.cuda.graph(force_post, stream=stream):
            energy = self._force_post()
        torch.cuda.synchronize(s.position.device)
        restore()
        self.pre_graph, self.force_post_graph, self.energy = pre, force_post, energy
        self.phase_captures += 1
        self.capture_force_evaluations += 3
        self.capture_seconds += time.perf_counter() - started

    def prepare(self):
        self._guard()
        signature = (id(self.engine.graph), id(self.engine._prepared_energy),
                     self.engine.static_batch['positions'].data_ptr())
        if signature != self.signature:
            self._capture()
            self.signature = signature

    @torch.no_grad()
    def _reject(self, flags, phase):
        s = self.stepper
        s.position.copy_(s.previous)
        s.velocity.copy_(self.previous_velocity)
        s.force.copy_(self.previous_force)
        self.engine.static_batch['positions'].copy_(s.position)
        self.failed = True
        raise PhaseCheckError(flags, phase, self.committed_steps)

    def advance(self, count=1):
        if type(count) is not int or count != 1:
            raise ValueError('Checked phases advance exactly one step')
        self.prepare()
        self.pre_graph.replay()
        flags = self.flags.cpu().tolist()
        if flags[1] == 1 or flags[3] == 1:
            self._reject(flags, 'before_force')
        if flags[0] == 1:
            # The pre phase has already drifted/projected. Rebuild and evaluate
            # once here, rather than rolling back and repeating the force.
            with torch.no_grad():
                self.batch['positions'] = self.stepper.position.float()
            self.engine.cache.reference = None
            energy, force = self.engine(self.batch)
            with torch.no_grad():
                self.reference.copy_(self.engine.cache.reference)
                self.stepper.force.copy_(force)
            self._post(energy)
            self.rebuild_steps += 1
        else:
            self.force_post_graph.replay()
            energy = self.energy
        self.attempted_steps += 1
        flags = self.flags.cpu().tolist()
        if min(flags[2], flags[4], flags[5]) == 1:
            self._reject(flags, 'after_force')
        self.committed_steps += 1
        self.stepper.steps += 1
        self.last_energy = energy.detach().clone()
        return 1, self.last_energy, self.stepper.force.detach().clone()

    def description(self):
        return dict(backend='checked_gpu_md_phases',
            committed_steps=self.committed_steps,
            trajectory_force_evaluations=self.attempted_steps,
            rebuild_steps=self.rebuild_steps, phase_captures=self.phase_captures,
            phase_capture_force_evaluations=self.capture_force_evaluations,
            phase_capture_seconds=self.capture_seconds,
            constraint_tolerance=self.stepper.constraints.tolerance,
            constraint_checks='Every projection checked before committing the step',
            skin_check='Before every force; misses rebuild at actual drifted geometry',
            timestep_fs=self.stepper.dt, all_forces_recomputed=True,
            speculation=False, fixed_cell=True)

    def close(self):
        self.pre_graph = self.force_post_graph = self.energy = None
        self.engine.close()
