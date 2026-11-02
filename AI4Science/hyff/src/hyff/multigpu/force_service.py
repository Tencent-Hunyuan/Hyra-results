"""Rank-zero eager force adapter and worker service; no process/group ownership.

All ranks first construct the same session collectively. Rank zero calls
evaluate(FP32 coordinates), close() or abort(code); other ranks call serve().
Each request has a monotonic sequence and a common checked command. Accepted
outputs are a detached FP64 scalar energy and contiguous FP32 canonical force.
Unexpected exceptions propagate to the launcher; no recovery collectives are
attempted after an in-flight failure. The launcher must terminate every peer.
"""
from __future__ import annotations

from hyff.multigpu.md_session import MDSessionError


_PROTOCOL = 1
_EVALUATE, _STOP, _ABORT = 1, 2, 3
_ABORT_CODES = ("OK", "CALLER_ABORT", "INTEGRATOR_ERROR", "USER_STOP", "SUPERVISOR_STOP")


class OwnedForceService:
    backend = "eager"

    def __init__(self, session):
        self.session = session
        self.control = session.control
        self.rank, self.world_size = session.rank, session.world_size
        self.device, self.model, self.cache = session.device, session.model, session.cache
        self.sequence = self.evaluations = 0
        self.closed = self.fatal = self.in_flight = False
        self.terminal_reason = None

    def _root_ready(self):
        if self.rank != 0:
            raise MDSessionError("MD_ROOT_SERVICE_ONLY")
        if self.closed or self.fatal or self.in_flight or self.session.fatal:
            raise MDSessionError("MD_SERVICE_NOT_READY")

    def _exchange(self, command=None, code=0):
        expected = self.sequence + 1
        packet = [_PROTOCOL, expected, command, code] if self.rank == 0 else None
        protocol, sequence, opcode, reason = self.control.broadcast_ints(packet, 4)
        valid = (protocol == _PROTOCOL and sequence == expected and opcode in (_EVALUATE, _STOP, _ABORT) and
            type(reason) is int and 0 <= reason < len(_ABORT_CODES) and
            ((opcode == _ABORT and reason != 0) or (opcode != _ABORT and reason == 0)))
        if self.control.max_code(int(not valid)):
            raise MDSessionError("MD_SERVICE_PROTOCOL_MISMATCH")
        self.sequence = sequence
        return opcode, reason

    def _retire(self, reason):
        self.terminal_reason = reason
        self.closed = True
        self.session.close()
        self.model = None

    def _mark_fatal(self):
        self.fatal = True
        self.session.fatal = True
        self.terminal_reason = "FATAL_COLLECTIVE_OR_EVALUATOR_ERROR"

    def evaluate(self, positions_fp32):
        """Root-only call. The session copies inputs before any retry/rebuild."""
        self._root_ready()
        self.in_flight = True
        try:
            opcode, _ = self._exchange(_EVALUATE)
            if opcode != _EVALUATE:
                raise MDSessionError("MD_SERVICE_COMMAND_MISMATCH")
            result = self.session.evaluate_global(positions_fp32)
            if not result.accepted:
                self._retire(result.reason)
                raise MDSessionError(result.reason, collective_safe=True)
            self.evaluations += 1
            return result.energy, result.forces
        except MDSessionError as error:
            if not error.collective_safe:
                self._mark_fatal()
            raise
        except BaseException:
            self._mark_fatal()
            raise
        finally:
            self.in_flight = False

    def serve(self):
        """Nonroot loop; expected rejection ends all workers without more calls."""
        if self.rank == 0 or self.closed or self.fatal or self.in_flight:
            raise MDSessionError("MD_WORKER_SERVICE_NOT_READY")
        try:
            while not self.closed:
                self.in_flight = True
                command, code = self._exchange()
                if command == _STOP:
                    self._retire("STOP")
                elif command == _ABORT:
                    self._retire(_ABORT_CODES[code])
                else:
                    result = self.session.evaluate_global(None)
                    if not result.accepted:
                        self._retire(result.reason)
                    else:
                        self.evaluations += 1
                self.in_flight = False
            return self.metadata()
        except BaseException:
            self._mark_fatal()
            raise
        finally:
            self.in_flight = False

    def close(self):
        """Root STOP only at a safe request boundary; local after fatal/reject.

        This neither destroys the group nor proves peer processes have exited.
        The launcher's process-group timeout bounds transport operations.
        """
        if self.closed:
            return self.metadata()
        if self.in_flight or self.session.in_flight:
            self._mark_fatal()
            # Do not deallocate a bundle still used by an outstanding request.
            return self.metadata()
        if self.fatal or self.session.fatal:
            # Never enqueue STOP behind a possibly mismatched force collective.
            self._mark_fatal()
            self.session.close()
            self.model = None
            self.closed = True
            return self.metadata()
        self._root_ready()
        self.in_flight = True
        try:
            self._exchange(_STOP)
            self._retire("STOP")
            return self.metadata()
        except BaseException:
            self._mark_fatal()
            raise
        finally:
            self.in_flight = False

    def abort(self, error_code="CALLER_ABORT"):
        """Best effort at a safe boundary only; fatal teardown is left to the launcher.

        Unknown caller strings become CALLER_ABORT; arbitrary exception text
        is never broadcast. A transport exception propagates, without retry.
        """
        if self.closed or self.fatal or self.session.fatal or self.in_flight or self.session.in_flight:
            return {"attempted": False, "reason": "NO_SAFE_COMMAND_BOUNDARY",
                    "caller_must_terminate_peers": bool(self.fatal or self.session.fatal or
                        self.in_flight or self.session.in_flight)}
        self._root_ready()
        code = _ABORT_CODES.index(error_code) if error_code in _ABORT_CODES[1:] else 1
        self.in_flight = True
        try:
            self._exchange(_ABORT, code)
            self._retire(_ABORT_CODES[code])
            return {"attempted": True, "reason": _ABORT_CODES[code], "caller_must_terminate_peers": False}
        except BaseException:
            self._mark_fatal()
            raise
        finally:
            self.in_flight = False

    def metadata(self):
        return {"schema": "LARGO_OWNED_FORCE_SERVICE", "rank": self.rank,
            "world_size": self.world_size, "sequence": self.sequence, "evaluations": self.evaluations,
            "closed": self.closed, "fatal": self.fatal, "terminal_reason": self.terminal_reason,
            "energy_dtype": "float64", "force_dtype": "float32", "backend": "eager",
            "process_group_destroyed": False, "peer_exit_observed": False,
            "transport_timeout_owned_by_launcher": True, "session": self.session.metadata()}


__all__ = ["OwnedForceService"]
