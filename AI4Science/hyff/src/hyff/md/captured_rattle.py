"""Capture the two ordinary RATTLE phases around the checked force evaluator.

This uses the coupled constraint kernel and the same PyTorch kick and drift
arithmetic as the uncaptured integrator. Every position and velocity projection
is checked on the host at the constraint tolerance before proceeding past that
phase. The force callback stays outside both graphs, so all neighbor checks,
updates, recaptures and numerical checks still run. The time step and physical
degrees of freedom are those of the uncaptured integrator.
"""
import math
import time

import torch
import triton

from hyff.md.constraints import RattleConstraints, _project_coupled
from hyff.md.units import ACCELERATION


class CapturedRattleStep:
    def __init__(self, position, velocity, force, mass, timestep_fs, constraints):
        started = time.perf_counter()
        if not isinstance(constraints, RattleConstraints) or constraints.solver != 'coupled' or not constraints.check:
            raise ValueError('Captured phases require the checked coupled RATTLE constraints')
        if not math.isfinite(timestep_fs) or timestep_fs <= 0:
            raise ValueError('A positive finite timestep is required')
        n, device = constraints.n_atoms, mass.device
        for value in (position, velocity):
            if (not value.is_cuda or value.device != device or value.dtype != torch.float64
                    or value.shape != (n, 3) or not value.is_contiguous()):
                raise ValueError('Integrator positions and velocities must be contiguous CUDA FP64')
        if mass.shape != (n,) or mass.dtype != torch.float64 or not mass.is_contiguous():
            raise ValueError('Masses must be a contiguous FP64 vector on the same CUDA device')
        if force.shape != (n, 3) or force.dtype != torch.float32 or force.device != device:
            raise ValueError('The ML force coordinates must be CUDA FP32')
        if not torch.equal(mass.reciprocal(), constraints.inverse_mass):
            raise ValueError('Integrator and constraint masses differ')
        if not all(bool(torch.isfinite(v).all()) for v in (position, velocity, force)):
            raise ValueError('Initial integration state must be finite')
        self.position = position.detach().clone()
        self.velocity = velocity.detach().clone()
        self.force = force.detach().clone()
        self.previous = position.detach().clone()
        self.mass = mass.detach().clone()
        self.dt = float(timestep_fs)
        self.constraints = constraints
        self.constraint_policy = (constraints.solver, constraints.tolerance, constraints.max_iterations, constraints.check)
        self.constraint_versions = {name: getattr(constraints, name)._version for name in
                                   ('inverse_mass', 'atoms', 'left', 'right', 'distance2')}
        self.position_error = torch.zeros_like(constraints.error)
        self.velocity_error = torch.zeros_like(constraints.error)
        self.steps = 0

        def pre():
            with torch.no_grad():
                self.previous.copy_(self.position)
                self.velocity.add_(.5*self.dt*ACCELERATION*self.force/self.mass[:, None])
                self.position.add_(self.dt*self.velocity)
                self._project(self.previous, True)

        def post():
            with torch.no_grad():
                self.velocity.add_(.5*self.dt*ACCELERATION*self.force/self.mass[:, None])
                self._project(self.position, False)

        def reset_setup():
            with torch.no_grad():
                self.position.copy_(position)
                self.previous.copy_(position)
                self.velocity.zero_()
                self.force.zero_()

        # Only zero-velocity/zero-force preparation is executed. No force
        # callback or simulated trajectory step is consumed during capture.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                reset_setup()
                pre()
                post()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self.pre_graph, self.post_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        reset_setup()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(self.pre_graph, stream=stream):
            pre()
        torch.cuda.synchronize(device)
        reset_setup()
        torch.cuda.synchronize(device)
        with torch.cuda.graph(self.post_graph, stream=stream):
            post()
        torch.cuda.synchronize(device)
        with torch.no_grad():
            self.position.copy_(position)
            self.previous.copy_(position)
            self.velocity.copy_(velocity)
            self.force.copy_(force)
            self.position_error.zero_()
            self.velocity_error.zero_()
        torch.cuda.synchronize(device)
        for actual, expected in ((self.position, position), (self.velocity, velocity), (self.force, force)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.capture_seconds = time.perf_counter() - started

    def _project(self, previous, positions):
        c = self.constraints
        if c.empty:
            return   # flexible MD: nothing to project (the kernel would have zero programs)
        _project_coupled[(triton.cdiv(len(c.atoms), 32),)](
            previous, self.position, self.velocity, c.inverse_mass, c.atoms, c.left,
            c.right, c.distance2, self.position_error if positions else self.velocity_error,
            len(c.atoms), self.dt if positions else 1., positions,
            c.tolerance, c.max_iterations, 32, num_warps=4, enable_fp_fusion=False)

    def _check(self, error, phase):
        if not error.numel():
            return
        maximum = float(error.max())
        if not math.isfinite(maximum) or maximum > self.constraints.tolerance:
            raise FloatingPointError(f'Constraint {phase} projection failed: residual={maximum}')

    def step(self, full_evaluate):
        c = self.constraints
        if (c.solver, c.tolerance, c.max_iterations, c.check) != self.constraint_policy:
            raise ValueError('The captured constraint policy changed')
        if any(getattr(c, key)._version != version for key, version in self.constraint_versions.items()):
            raise ValueError('Captured masses or constraint topology changed')
        self.pre_graph.replay()
        self._check(self.position_error, 'position')
        energy, force = full_evaluate(self.position)
        if force.shape != self.force.shape or force.device != self.force.device or force.dtype != self.force.dtype:
            raise ValueError('The force callback changed the declared shape, device or precision')
        with torch.no_grad():
            self.force.copy_(force.detach())
        self.post_graph.replay()
        self._check(self.velocity_error, 'velocity')
        self.steps += 1
        return self.position, self.velocity, energy, self.force, None

    def description(self):
        return dict(backend='captured_ordinary_rattle_phases', cuda_graphs=2,
            capture_seconds=self.capture_seconds, timestep_fs=self.dt,
            force_coordinate_precision='FP32', integration_state_precision='FP64',
            checked_every_projection=True, tolerance=self.constraints.tolerance,
            setup_force_evaluations=0, setup_trajectory_steps=0,
            force_callback='checked force evaluator outside the graphs; all neighbor checks/updates/recaptures kept',
            arithmetic='uncaptured kick/drift PyTorch expressions and the coupled constraint kernel')
