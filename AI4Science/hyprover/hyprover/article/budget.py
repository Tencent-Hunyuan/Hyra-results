"""Monotonic deadlines shared by article network and subprocess operations."""
from __future__ import annotations

import http.client
import os
import queue
import signal
import socket
import subprocess
import threading
import time
from contextlib import contextmanager

from .models import ArticleError, Cancelled


class ArticleDeadlineExceeded(ArticleError):
    """The entire article job has used its configured wall-clock budget."""


class OperationTimeout(ArticleError):
    """The external operation exceeded its own timeout."""


class Budget:
    def __init__(self, timeout, error: str, *, deadline=None, cancelled=None):
        operation_end = time.monotonic() + timeout
        self.job_end = deadline
        self.end = operation_end if deadline is None else min(operation_end, deadline)
        self.error = error
        self.cancelled = cancelled

    def remaining(self) -> float:
        if self.cancelled is not None and self.cancelled():
            raise Cancelled("article_cancelled")
        now = time.monotonic()
        if self.job_end is not None and now >= self.job_end:
            raise ArticleDeadlineExceeded("article_total_time_budget_exhausted")
        remaining = self.end - now
        if remaining <= 0:
            raise OperationTimeout(self.error)
        return remaining

    def wait(self, seconds: float) -> None:
        target = time.monotonic() + seconds
        while True:
            available = self.remaining()
            delay = target - time.monotonic()
            if delay <= 0:
                return
            time.sleep(min(available, delay, 0.1))

    def resolve(self, resolver, *args, **kwargs):
        """Bound DNS waiting even when the system resolver ignores socket timeouts."""
        self.remaining()
        results = queue.Queue(maxsize=1)

        def resolve():
            try:
                results.put((resolver(*args, **kwargs), None))
            except Exception as exc:
                results.put((None, exc))

        # A system resolver call cannot be cancelled portably. The abandoned worker
        # only resolves a name, and never starts network I/O or updates job state.
        threading.Thread(target=resolve, daemon=True).start()
        while True:
            try:
                value, error = results.get(timeout=min(0.1, self.remaining()))
                break
            except queue.Empty:
                continue
        self.remaining()
        if error is not None:
            raise error
        return value

    @contextmanager
    def watch_http(self, connection):
        """Interrupt a blocking socket read when the operation must stop."""
        stopped = threading.Event()
        observed = None

        def update_socket():
            nonlocal observed
            remaining = self.remaining()
            sock = getattr(connection, "sock", None)
            if sock is not None:
                observed = sock
            if observed is not None:
                observed.settimeout(remaining)

        def connect(address, timeout=None, source_address=None):
            # http.client connects and handshakes synchronously. Bound name
            # resolution separately, then keep the socket visible during connect.
            nonlocal observed
            host, port = address
            addresses = self.resolve(socket.getaddrinfo, host, port, 0, socket.SOCK_STREAM)
            failure = None
            for family, socktype, proto, _, target in addresses:
                self.remaining()
                sock = socket.socket(family, socktype, proto)
                try:
                    observed = sock
                    sock.settimeout(self.remaining())
                    if source_address:
                        sock.bind(source_address)
                    sock.connect(target)
                    sock.settimeout(self.remaining())
                    return sock
                except OSError as exc:
                    failure = exc
                    sock.close()
                except BaseException:
                    sock.close()
                    raise
            raise failure or OSError("no_socket_addresses")

        def watch():
            while not stopped.is_set():
                try:
                    remaining = self.remaining()
                except (ArticleDeadlineExceeded, Cancelled, OperationTimeout):
                    sock = getattr(connection, "sock", None) or observed
                    if sock is not None:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except (OSError, AttributeError):
                            pass
                    return
                stopped.wait(min(0.1, remaining) if self.cancelled is not None else remaining)

        create_connection = getattr(connection, "_create_connection", None)
        if create_connection is not None:
            connection._create_connection = connect
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            yield update_socket
            self.remaining()
        except (OSError, ValueError, http.client.HTTPException):
            self.remaining()
            raise
        finally:
            stopped.set()
            watcher.join()
            if create_connection is not None:
                connection._create_connection = create_connection


def run_command(command, budget: Budget, *, stdout, stderr) -> int:
    """Wait in short intervals and kill/reap the isolated subprocess on abort."""
    budget.remaining()
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                          start_new_session=True) as child:
        try:
            while True:
                try:
                    # Check even after a child reports success: it may have
                    # completed immediately after cancellation or expiry.
                    status = child.wait(timeout=min(0.1, budget.remaining()))
                except subprocess.TimeoutExpired:
                    continue
                budget.remaining()
                return status
        except BaseException:
            try:
                if child.returncode is None:
                    os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise
