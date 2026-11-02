"""Checked GPU MD with device-side Verlet rebuilding into fixed matrices.

Every step evaluates the complete force. Neighbor overflow rejects
the step before force evaluation. COO metadata is deliberately not consumed
after a native rebuild; ordinary ForceEngine calls must rebuild their cache.
"""
from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

from hyff.md.device.neighbor_matrix import NativeMatrixUpdater
from hyff.md.checked_phases import PhaseCheckError
from hyff.md.device.conditional import CapturedChild, ConditionalLoop
from hyff.md.device.phases import ConditionalPhases


@triton.jit
def _accept_reference(Position, Reference, Status, N: tl.constexpr, B: tl.constexpr):
    index = tl.program_id(0) * B + tl.arange(0, B)
    valid = tl.load(Status) == 0
    value = tl.load(Position + index, (index < N) & valid, 0.)
    tl.store(Reference + index, value, (index < N) & valid)


class NeighborCapacityError(RuntimeError):
    """A rebuilt native neighbour list did not fit its matrices; no force was evaluated."""

    def __init__(self, status):
        self.status = status
        super().__init__(f'Native neighbour list does not fit: {status}')


class NativeMatrixPhases(ConditionalPhases):
    def __init__(self, engine, stepper, batch, cell_headroom=1.0):
        super().__init__(engine, stepper, batch)
        self.cell_headroom = float(cell_headroom)
        self.cell_neighbor_rebuilds = 0
        pbc = batch['pbc']
        self.pbc_guard = (pbc, pbc._version, pbc.data_ptr())
        self.rebuild_child = self.updater = None
        self.setup_neighbor_rebuilds = 0
        self.last_neighbor_status = None

    def _guard(self):
        super()._guard()
        pbc, version, pointer = self.pbc_guard
        if (self.batch['pbc'] is not pbc or pbc._version != version or
                pbc.data_ptr() != pointer):
            raise ValueError('Changed PBC requires a new native MD driver')

    def _release_loop(self):
        # Parent loops hold child graph references. Destroy them before the
        # owning CUDA graphs or the updater storage can be released.
        super()._release_loop()
        if self.rebuild_child is not None:
            self.rebuild_child.close()
            self.rebuild_child = None
        self.updater = None

    @torch.no_grad()
    def _rebuild(self):
        position = self.engine.static_batch['positions']
        status = self.updater.rebuild_gpu(position)
        _accept_reference[(triton.cdiv(position.numel(), 256),)](
            position, self.reference, status, position.numel(), 256, num_warps=4)

    def _capture(self):
        started = time.perf_counter()
        s = self.stepper
        torch.cuda.synchronize(s.position.device)
        self._release_loop()
        with torch.no_grad():
            self.engine.static_batch['positions'].copy_(s.position)
        self.updater = NativeMatrixUpdater(self.engine._prepared_energy,
                                           skin=self.engine.cache.skin, cell_headroom=self.cell_headroom)
        # The force child must be captured at a valid geometry with a matrix
        # built at that geometry, even after an external engine recapture.
        self._rebuild()
        self.last_neighbor_status = self.updater.checked_status()
        self.setup_neighbor_rebuilds += 1
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
                self._force_post()
                restore()
            self._rebuild()
        torch.cuda.current_stream(s.position.device).wait_stream(stream)
        torch.cuda.synchronize(s.position.device)
        self.pre_child = CapturedChild(self._pre, stream)
        torch.cuda.synchronize(s.position.device)
        restore()
        torch.cuda.synchronize(s.position.device)
        self.rebuild_child = CapturedChild(self._rebuild, stream)
        torch.cuda.synchronize(s.position.device)
        self.post_child = CapturedChild(self._force_post, stream)
        torch.cuda.synchronize(s.position.device)
        restore()
        self.last_neighbor_status = self.updater.checked_status()
        self.energy = self.post_child.output
        self.ledger = torch.empty(16, dtype=torch.int32, device=s.position.device)
        self.loop = ConditionalLoop(self.pre_child, self.post_child, self.flags,
            self.ledger, rebuild=self.rebuild_child, neighbor_status=self.updater.status)
        self.phase_captures += 1
        self.capture_force_evaluations += 3
        self.capture_seconds += time.perf_counter() - started
        self.engine.cache.reference = None

    def prepare(self):
        self._guard()
        if self.engine.graph is None:
            raise RuntimeError('The underlying force engine has been closed')
        # An external ordinary engine call can replace matrices without
        # changing their addresses. Its rebuild epoch also forces rebinding.
        signature = (id(self.engine.graph), id(self.engine._prepared_energy),
            self.engine.static_batch['positions'].data_ptr(), self.engine.cache.rebuilds)
        if signature != self.signature:
            self._capture()
            self.signature = signature

    @torch.no_grad()
    def set_cell(self, cell):
        """Adopt a new periodic cell (a host (3, 3) array) between chunks (constant-pressure MD).

        Updates the updater's bin grid and the cell read by the force kernels in place, so the captured
        children stay valid. The neighbour matrices are stale until ``evaluate_here`` rebuilds them.
        """
        self.prepare()
        self.updater.set_cell(cell)
        cell = self.updater.fixed_cell
        for target in (self.updater.current_cell, self.engine.static_batch['cell'], self.batch['cell']):
            target.copy_(cell.reshape(target.shape))
        batch_cell = self.batch['cell']
        self.cell_guard = (batch_cell, batch_cell._version, batch_cell.data_ptr())
        self.engine.cache.reference = None

    @torch.no_grad()
    def rebuild_here(self):
        """Neighbour rebuild at the integrator's positions between chunks (no force).

        Raises ``NeighborCapacityError`` if the rebuilt list does not fit its matrices.
        """
        self.prepare()
        self.engine.static_batch['positions'].copy_(self.stepper.position)
        self.rebuild_child.replay()
        status = dict(zip(('error_bits', 'true_max_degree', 'overflow_atoms', 'invalid_atoms'),
                          self.updater.status.cpu().tolist()))
        self.last_neighbor_status = status
        if status['error_bits']:
            raise NeighborCapacityError(status)
        self.cell_neighbor_rebuilds += 1
        self.engine.cache.reference = None

    @torch.no_grad()
    def evaluate_here(self):
        """Neighbour rebuild, energy and force at the integrator's positions, between chunks.

        Replays the captured rebuild and force children. The velocities are restored, so only the
        neighbour matrices, ``stepper.force`` and the returned energy change.
        """
        self.rebuild_here()
        velocity = self.stepper.velocity.clone()
        self.post_child.replay()
        self.stepper.velocity.copy_(velocity)
        return self.energy.detach().clone()

    def advance(self, count=1):
        if type(count) is not int or not 1 <= count <= 1024:
            raise ValueError('A conditional chunk must contain 1 to 1024 steps')
        try:
            self.prepare()
            self.loop.launch(count)
            record = self.ledger.cpu().tolist()
            requested, committed, attempted, iterations, reason = record[:5]
            flags, rebuilds, status = record[5:11], record[11], record[12:16]
            if (requested != count or not 0 <= committed <= attempted <= count
                    or not attempted <= iterations <= count or not 0 <= rebuilds <= iterations):
                raise RuntimeError('The device returned inconsistent step accounting')
            self.graph_launches += 1
            self.committed_steps += committed
            self.gpu_committed_steps += committed
            self.attempted_steps += attempted
            self.stepper.steps += committed
            self.rebuild_steps += rebuilds
            self.largest_committed_chunk = max(self.largest_committed_chunk, committed)
            if rebuilds:
                self.last_neighbor_status = dict(zip(
                    ('error_bits', 'true_max_degree', 'overflow_atoms', 'invalid_atoms'), status))
            if reason == 2:
                self._reject(flags, 'before_force')
            if reason == 3:
                self._reject(flags, 'after_force')
            if reason == 4:
                try:
                    self._reject(flags, 'neighbor_rebuild')
                except PhaseCheckError as error:
                    error.details['native_neighbor_status'] = self.last_neighbor_status
                    error.args = (f'Checked MD phase failed: {error.details}',)
                    raise
            if reason != 0 or committed != count or attempted != count or iterations != count:
                raise RuntimeError('A valid native loop ended before its requested budget')
            self.last_energy = self.energy.detach().clone()
            return committed, self.last_energy, self.stepper.force.detach().clone()
        finally:
            # Native matrices and the COO cache no longer describe the same
            # reference. Force a fresh ordinary rebuild before any later
            # ForceEngine evaluation, including later end-state checks.
            self.engine.cache.reference = None

    def description(self):
        result = super().description()
        result.update(backend='cuda_conditional_native_matrix_md',
            neighbor_update='GPU conditional cell list directly into fixed matrices',
            neighbor_setup_rebuilds=self.setup_neighbor_rebuilds,
            neighbor_rebuilds_between_chunks=self.cell_neighbor_rebuilds,
            last_native_neighbor_status=self.last_neighbor_status,
            neighbor_overflow_policy='reject and restore before force; never truncate',
            ordinary_engine_cache='invalidated on return; external rebuild epoch forces rebinding',
            control='CUDA WHILE with pre-check, optional neighbor rebuild, checked full force/post',
            native_neighbors=self.updater.description() if self.updater is not None else None)
        return result
