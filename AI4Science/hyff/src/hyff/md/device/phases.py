"""Batch exact checked RATTLE steps using CUDA conditional execution."""
from __future__ import annotations

import time

import torch

from hyff.md.checked_phases import CheckedPhases
from hyff.md.device.conditional import CapturedChild, ConditionalLoop, library


class ConditionalPhases(CheckedPhases):
    def __init__(self, engine, stepper, batch):
        super().__init__(engine, stepper, batch)
        library()
        self.ledger = torch.empty(11, dtype=torch.int32, device=stepper.position.device)
        self.loop = self.pre_child = self.post_child = None
        self.graph_launches = 0
        self.gpu_committed_steps = 0
        self.largest_committed_chunk = 0

    def _release_loop(self):
        if self.loop is not None:
            self.loop.close()
            self.loop = None
        for name in ('pre_child', 'post_child'):
            child = getattr(self, name)
            if child is not None:
                child.close()
                setattr(self, name, None)

    def _capture(self):
        started = time.perf_counter()
        torch.cuda.synchronize(self.stepper.position.device)
        self._release_loop()
        s = self.stepper
        targets = (s.position, s.velocity, s.force, self.engine.static_batch['positions'])
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
        torch.cuda.current_stream(s.position.device).wait_stream(stream)
        torch.cuda.synchronize(s.position.device)
        self.pre_child = CapturedChild(self._pre, stream)
        torch.cuda.synchronize(s.position.device)
        restore()
        torch.cuda.synchronize(s.position.device)
        self.post_child = CapturedChild(self._force_post, stream)
        torch.cuda.synchronize(s.position.device)
        restore()
        self.energy = self.post_child.output
        self.loop = ConditionalLoop(self.pre_child, self.post_child, self.flags, self.ledger)
        self.phase_captures += 1
        self.capture_force_evaluations += 3
        self.capture_seconds += time.perf_counter() - started

    def advance(self, count=1):
        if type(count) is not int or not 1 <= count <= 1024:
            raise ValueError('A conditional chunk must contain 1 to 1024 steps')
        self.prepare()
        self.loop.launch(count)
        record = self.ledger.cpu().tolist()
        requested, committed, attempted, iterations, reason = record[:5]
        flags = record[5:]
        if requested != count or not 0 <= committed <= attempted <= count:
            raise RuntimeError('The device returned inconsistent step accounting')
        self.graph_launches += 1
        self.committed_steps += committed
        self.gpu_committed_steps += committed
        self.attempted_steps += attempted
        self.stepper.steps += committed
        if reason == 2:
            self._reject(flags, 'before_force')
        if reason == 3:
            self._reject(flags, 'after_force')
        if reason not in (0, 1):
            raise RuntimeError(f'Unknown conditional stop reason: {reason}')
        energy = self.energy
        if reason == 1:
            # There have been `committed` valid steps and one extra drift.
            # Complete that drifted step after rebuilding; never replay a
            # trajectory force and never evaluate outside the checked skin.
            if iterations != committed + 1 or attempted != committed or committed >= count:
                raise RuntimeError('Unexpected skin-miss accounting')
            with torch.no_grad():
                self.batch['positions'] = self.stepper.position.float()
            self.engine.cache.reference = None
            energy, force = self.engine(self.batch)
            with torch.no_grad():
                self.reference.copy_(self.engine.cache.reference)
                self.stepper.force.copy_(force)
            self._post(energy)
            self.attempted_steps += 1
            self.rebuild_steps += 1
            flags = self.flags.cpu().tolist()
            if min(flags[2], flags[4], flags[5]) != 2:
                self._reject(flags, 'after_force')
            self.committed_steps += 1
            self.stepper.steps += 1
            committed += 1
        elif committed != count or attempted != count or iterations != count:
            raise RuntimeError('A valid conditional loop ended before its requested budget')
        self.largest_committed_chunk = max(self.largest_committed_chunk, committed)
        self.last_energy = energy.detach().clone()
        return committed, self.last_energy, self.stepper.force.detach().clone()

    def description(self):
        result = super().description()
        result.update(backend='cuda_conditional_checked_md',
            conditional_graph_launches=self.graph_launches,
            gpu_committed_steps=self.gpu_committed_steps,
            largest_committed_chunk=self.largest_committed_chunk,
            constraint_checks='Every projection checked on device before force or commit; host receives exact stop reason',
            skin_check='Before every force; the GPU stops before an invalid-neighbor force',
            control='CUDA WHILE containing checked pre and conditional full-force/post graphs')
        return result

    def close(self):
        torch.cuda.synchronize(self.stepper.position.device)
        self._release_loop()
        super().close()
