"""Per-process limit on compiled-engine constructions; stdlib only.

Compiled engines keep compiler state for the life of the process, so at most
two constructions are allowed per process. The factory chooses which routes
reserve. This module counts attempts, not compiler specializations, live engines,
or guaranteed successful constructions.
"""
from __future__ import annotations

import os
import re
import threading


_ATTEMPT_LIMIT = 2
_PUBLIC_ROUTE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z", re.ASCII)


class CompiledEngineProcessLimitError(RuntimeError):
    """Factory/setup refusal before another compiled construction starts."""

    def __init__(self, *, reason: str, route: str, used: int, limit: int):
        self.reason = reason
        self.route = route
        self.used = used
        self.limit = limit
        super().__init__(
            "compiled engine construction refused: "
            f"reason={reason}, reserved_attempts={used}, limit={limit}; "
            "use a fresh Python process, preferably one engine per process"
        )


class _ProcessAttemptLedger:
    """Private test-isolatable ledger; no reset, release, or refund operation."""

    def __init__(self):
        self._owner_pid = os.getpid()
        self._used = 0
        self._lock = threading.Lock()

    def _require_owner(self, route: str) -> None:
        if os.getpid() != self._owner_pid:
            raise CompiledEngineProcessLimitError(
                reason="forked_process", route=route, used=self._used,
                limit=_ATTEMPT_LIMIT,
            )

    def reserve(self, route: str) -> int:
        # A fork child may inherit this mutex while another parent thread owns
        # it. Refuse before touching it; resetting it would discard the count.
        self._require_owner(route)
        if type(route) is not str or _PUBLIC_ROUTE_NAME.fullmatch(route) is None:
            raise ValueError("route must be a stable public name of 1 to 64 lowercase ASCII letters, digits, or underscores, starting with a letter")
        with self._lock:
            self._require_owner(route)
            if self._used >= _ATTEMPT_LIMIT:
                raise CompiledEngineProcessLimitError(
                    reason="attempt_limit", route=route, used=self._used,
                    limit=_ATTEMPT_LIMIT,
                )
            self._used += 1
            return self._used


_PROCESS_ATTEMPTS = _ProcessAttemptLedger()


def reserve_compiled_engine_attempt(route: str) -> int:
    """Permanently reserve one attempt and return its one-based ordinal.

    ``route`` is a fixed public route name chosen by the caller, never a path,
    checkpoint value, or user configuration. All routes share this module's
    singleton. A failed constructor still consumes its reservation. Close and
    garbage collection cannot refund it. A fresh interpreter (including spawn)
    has its own budget; a child inheriting this module through fork is refused.
    """
    return _PROCESS_ATTEMPTS.reserve(route)
