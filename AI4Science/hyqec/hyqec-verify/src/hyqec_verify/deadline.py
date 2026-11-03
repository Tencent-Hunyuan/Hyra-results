"""A process-wide wall-clock deadline for engine subprocesses.

`distance` sets the deadline; every engine starts its subprocesses through `run` here, which
limits each child to the remaining time and registers it so that `kill_children` (used by the
CLI watchdog) can stop all of them at once. A child killed for the deadline is reported as a
timeout, never as a result.
"""
from __future__ import annotations

import subprocess
import threading
import time

_deadline: float | None = None
_children: set = set()
_lock = threading.Lock()
expired = threading.Event()

__all__ = ["set_deadline", "clear_deadline", "remaining", "run", "kill_children", "expired"]


def set_deadline(t: float | None) -> None:
    global _deadline
    _deadline = t
    expired.clear()


def clear_deadline() -> None:
    set_deadline(None)


def remaining() -> float | None:
    if _deadline is None:
        return None
    return _deadline - time.monotonic()


def _limit(timeout):
    r = remaining()
    if r is None:
        return timeout
    r = max(0.0, r)
    return r if timeout is None else min(timeout, r)


def run(cmd, *, timeout=None, **popen_kw):
    """Run a child process; returns (returncode or None on timeout/deadline, elapsed seconds).

    Output is handled through popen_kw (stdout/stderr file objects). The effective timeout is the
    smaller of `timeout` and the remaining deadline."""
    start = time.monotonic()
    if expired.is_set() or (remaining() is not None and remaining() <= 0):
        return None, 0.0
    p = subprocess.Popen(cmd, **popen_kw)
    with _lock:
        _children.add(p)
    try:
        rc = p.wait(timeout=_limit(timeout))
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        rc = None
    finally:
        with _lock:
            _children.discard(p)
    if expired.is_set() and rc is not None and rc < 0:
        rc = None               # killed by the watchdog
    return rc, time.monotonic() - start


def kill_children() -> int:
    """Kill every registered child (watchdog). Returns how many were signalled."""
    expired.set()
    with _lock:
        kids = list(_children)
    for p in kids:
        try:
            p.kill()
        except OSError:
            pass
    return len(kids)
