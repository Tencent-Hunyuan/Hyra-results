"""Offline CLI tests with temporary projects and controlled process fixtures.

Worker startup is replaced by a recording fake. Some process-liveness and stop
cases launch disposable local children; none starts Codex or makes model requests.
Checks cover JSON/PID reads, command dispatch, task-source priority, output formatting,
locks and the ordering of process-control effects. Pytest runs all cases; the
standalone runner executes the explicitly listed fixture-free subset.
"""

from __future__ import annotations

import io
import json
import os
import runpy
import tempfile
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

from hyprover.execution import layout as L
from hyprover.orchestration import cli


# --------------------------------------------------------------------------- #
# env / project helpers (mirrors test_orchestration.py so styles match)        #
# --------------------------------------------------------------------------- #

@contextmanager
def _env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def _project_env(tmp: Path, **extra):
    contract = tmp / "worker.md"
    contract.write_text("# worker contract (stub)\n", encoding="utf-8")
    env = {"HYPROVER_AGENTS_ROOT": str(tmp / "agents"),
           "HYPROVER_WORKER_CONTRACT": str(contract)}
    env.update(extra)
    with _env(**env):
        yield


class _FakeSpawn:
    """Records (wdir) calls; returns our *own* pid so ``_alive`` reports True and
    ``_start_one`` treats the worker as live — without launching anything."""

    def __init__(self):
        self.calls = []

    def __call__(self, wdir):
        self.calls.append(Path(wdir))
        return os.getpid()


@contextmanager
def _patch_spawn():
    fake = _FakeSpawn()
    orig = cli.spawn_loop
    cli.spawn_loop = fake
    try:
        yield fake
    finally:
        cli.spawn_loop = orig


def _wl(project: str, worker: str) -> L.WorkerLayout:
    return L.WorkerLayout(L.worker_dir(project, worker))


def _expect_exit(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except SystemExit as e:
        return e
    raise AssertionError(f"expected SystemExit from {getattr(fn, '__name__', fn)}")


# --------------------------------------------------------------------------- #
# read helpers: _read_pid / _alive / _read_status                              #
# --------------------------------------------------------------------------- #

def test_alive_short_circuit_and_proc_fallbacks(monkeypatch):
    from unittest.mock import Mock

    for pid, kill_error, text, expected, read_expected in [
        (None, None, "", False, False), (0, None, "", False, False),
        (42, ProcessLookupError(), "", False, False),
        (42, PermissionError(), "", True, False),
        (42, None, "42 (name with ) parentheses) Z 1", False, True),
        (-42, None, "42 (name) S 1", True, True),
        (42, None, "malformed", True, True),
        (42, None, "42 (name)", True, True),
        (42, None, OSError("unreadable"), True, True),
    ]:
        kill = Mock(side_effect=kill_error)
        read = Mock(side_effect=text if isinstance(text, Exception) else None, return_value=text)
        with monkeypatch.context() as patch:
            patch.setattr(cli.os, "kill", kill)
            patch.setattr(Path, "read_text", read)
            assert cli._alive(pid) is expected
        assert kill.call_args_list == ([] if not pid else [((pid, 0), {})])
        assert read.call_count == int(read_expected)


def test_start_order_and_partial_failure_are_preserved(monkeypatch, tmp_path):
    import pytest

    paths = [tmp_path / "first", tmp_path / "second", tmp_path / "third"]
    for fail in (False, True):
        events = []
        def start(wl):
            events.append(("start", wl.name))
            if fail and wl.name == "second":
                raise RuntimeError("start failed")
            return "started"
        monkeypatch.setattr(cli.L, "target_worker_dirs", lambda target: paths)
        monkeypatch.setattr(cli, "_start_one", start)
        monkeypatch.setattr(cli.time, "sleep", lambda delay: events.append(("sleep", delay)))
        if fail:
            with pytest.raises(RuntimeError, match="start failed"):
                cli.do_start("P")
            assert events == [("start", "first"), ("sleep", 0.2), ("start", "second")]
        else:
            assert cli.do_start("P") == [{"worker": p.name, "result": "started"} for p in paths]
            assert events == [("start", "first"), ("sleep", 0.2), ("start", "second"),
                              ("sleep", 0.2), ("start", "third")]


def test_force_stop_probe_signal_and_cleanup_order(monkeypatch, tmp_path):
    from unittest.mock import Mock

    wl = L.WorkerLayout(tmp_path / "P" / "workers" / "worker")
    wl.dir.mkdir(parents=True)
    for states, sleeps in [(None, 50), ([True, False, True], 0)]:
        wl.pid.write_text("42")
        events = []
        outcomes = iter(states) if states is not None else None
        def alive(pid):
            events.append(("alive", pid))
            return next(outcomes) if outcomes is not None else True
        groups = iter([100, 200])
        def getpgid(pid):
            events.append(("getpgid", pid))
            return next(groups)
        unlink = Mock(side_effect=lambda **kwargs: events.append(("unlink", kwargs)))
        with monkeypatch.context() as patch:
            patch.setattr(cli, "_alive", alive)
            patch.setattr(cli.os, "getpgid", getpgid)
            patch.setattr(cli.os, "killpg", lambda pgid, sig: events.append(("killpg", pgid, sig)))
            patch.setattr(cli.time, "sleep", lambda delay: events.append(("sleep", delay)))
            patch.setattr(Path, "unlink", unlink)
            assert cli._stop_one(wl, True) == "killed"
        middle = [("alive", 42), ("sleep", 0.1)] * sleeps if sleeps else [("alive", 42)]
        assert events == [
            ("alive", 42), ("getpgid", 42), ("killpg", 100, cli.signal.SIGTERM),
            *middle, ("alive", 42), ("getpgid", 42), ("killpg", 200, cli.signal.SIGKILL),
            ("unlink", {"missing_ok": True}),
        ]
        assert not wl.stop.exists()


def test_list_metadata_and_liveness_order(monkeypatch, tmp_path):
    import pytest

    (tmp_path / "P").mkdir()
    meta_path = tmp_path / "P" / "project.json"
    original_read = Path.read_text
    for metadata in ('{"model": null}', '[]'):
        events = []
        meta_path.write_text(metadata)
        def read(path, **kwargs):
            events.append("metadata")
            return original_read(path, **kwargs)
        monkeypatch.setattr(Path, "read_text", read)
        monkeypatch.setattr(cli.L, "list_projects", lambda: ["P"])
        monkeypatch.setattr(cli.L, "project_dir", lambda project: tmp_path / project)
        monkeypatch.setattr(cli.L, "list_workers", lambda project: events.append("workers") or ["one", "two"])
        monkeypatch.setattr(cli, "_read_pid", lambda wl: events.append(wl.name) or {"one": 41, "two": 42}[wl.name])
        monkeypatch.setattr(cli, "_alive", lambda pid: events.append(pid) or pid == 41)
        if metadata == '[]':
            with pytest.raises(AttributeError):
                cli.do_list()
        else:
            assert cli.do_list() == [{"project": "P", "workers": 2, "live": 1, "model": None}]
        assert events == ["metadata", "workers", "one", 41, "two", 42]


def test_json_readers_preserve_payloads_and_read_boundaries(monkeypatch, tmp_path):
    import pytest
    from unittest.mock import Mock

    project = tmp_path / "P"
    wl = L.WorkerLayout(project / "workers" / "worker")
    wl.dir.mkdir(parents=True)
    monkeypatch.setattr(cli.L, "project_dir", lambda name: project)
    for path, read in [(wl.status, lambda: cli._read_status(wl)),
                       (project / "project.json", lambda: cli._read_project_metadata("P"))]:
        assert read() == {}
        for text, expected in [('{"label": "中文"}', {"label": "中文"}),
                               ('[]', []), ('null', None), ('false', False), ('2', 2),
                               ('"text"', "text"), ('{broken', {})]:
            path.write_text(text, encoding="utf-8")
            assert read() == expected
        with monkeypatch.context() as patch:
            failure = OSError("read failed")
            reader = Mock(side_effect=failure)
            patch.setattr(Path, "read_text", reader)
            assert read() == {}
            reader.assert_called_once_with(encoding="utf-8")
        with monkeypatch.context() as patch:
            failure = PermissionError("cannot inspect path")
            reader = Mock()
            patch.setattr(Path, "exists", Mock(side_effect=failure))
            patch.setattr(Path, "read_text", reader)
            with pytest.raises(PermissionError) as caught:
                read()
            assert caught.value is failure
            reader.assert_not_called()
        path.write_bytes(b"\xff")
        with pytest.raises(UnicodeDecodeError):
            read()


def test_task_source_priority_preserves_empty_task_and_original_text(monkeypatch, tmp_path):
    import sys

    source = tmp_path / "task.txt"
    source.write_bytes("  文件\r\n".encode("utf-8"))
    stream = io.StringIO("  stdin\r\n")
    monkeypatch.setattr(sys, "stdin", stream)
    assert cli._task_from_args(_Args(task="", file=str(source), stdin=True)) == ""
    assert stream.tell() == 0
    assert cli._task_from_args(_Args(file=str(source), stdin=True)) == "  文件\n"
    assert stream.tell() == 0
    assert cli._task_from_args(_Args(stdin=True)) == "  stdin\r\n"


def test_read_pid_missing_and_garbage(tmp: Path):
    with _project_env(tmp):
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        assert cli._read_pid(wl) is None            # no .pid yet
        wl.pid.write_text("not-an-int\n")
        assert cli._read_pid(wl) is None            # ValueError -> None
        wl.pid.write_text("4321\n")
        assert cli._read_pid(wl) == 4321


def test_alive_variants(tmp: Path):
    # falsy pid / None -> dead
    assert cli._alive(None) is False
    assert cli._alive(0) is False
    # our own process is alive (and not a zombie)
    assert cli._alive(os.getpid()) is True
    # a pid that (almost certainly) does not exist -> ProcessLookupError -> dead
    assert cli._alive(2_000_000_000) is False


def test_stop_one_force_sigkill_fallback(tmp: Path):
    """A child that ignores SIGTERM must be finished by the SIGKILL fallback
    (cli.py lines 245-248). We launch a real python child that traps SIGTERM into
    a no-op, signals readiness via a file, then sleeps; force-stop should SIGTERM
    (ignored), wait ~5s, then SIGKILL. Real process we own — never codex.

    Waiting for the readiness file is essential: if we stop before the SIGTERM
    handler is installed, the default handler kills the child immediately and the
    SIGKILL branch is never reached."""
    import subprocess
    import sys
    import time
    ready = tmp / "handler_ready"
    prog = (
        "import signal, time, sys\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(ready)!r}, 'w').close()\n"
        "time.sleep(120)\n"
    )
    with _project_env(tmp):
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        proc = subprocess.Popen([sys.executable, "-c", prog], start_new_session=True)
        wl.pid.write_text(str(proc.pid))
        try:
            # wait until the child has installed its SIGTERM-ignoring handler
            end = time.time() + 10
            while time.time() < end and not ready.exists():
                time.sleep(0.02)
            assert ready.exists(), "child never signalled readiness"
            assert cli._alive(proc.pid) is True
            t0 = time.time()
            res = cli._stop_one(wl, force=True)       # SIGTERM ignored -> SIGKILL fallback
            assert res == "killed"
            assert time.time() - t0 >= 4.5, "should have waited the full SIGTERM grace"
            # confirm it's really gone
            end = time.time() + 5
            while time.time() < end and cli._alive(proc.pid):
                time.sleep(0.05)
            assert cli._alive(proc.pid) is False
            assert not wl.pid.exists()
        finally:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass


def test_stop_one_force_sigkill_killpg_raises(tmp: Path):
    """Defensive branch (cli.py 247-248): the final SIGKILL ``os.killpg`` itself
    raises ProcessLookupError (the process died in the tiny window between the last
    ``_alive`` check and the kill). It must be swallowed and still return 'killed'.
    Fully stubbed — ``_alive`` is forced True through the wait loop, ``os.getpgid``
    is a no-op, and ``os.killpg`` raises; no real process is touched."""
    with _project_env(tmp):
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        wl.pid.write_text("2000000000")

        real_alive = cli._alive
        real_getpgid = cli.os.getpgid
        real_killpg = cli.os.killpg
        real_sleep = cli.time.sleep

        cli._alive = lambda pid: True                 # always "alive" -> reaches SIGKILL
        cli.os.getpgid = lambda pid: 12345
        def boom_killpg(pgid, sig):
            raise ProcessLookupError("gone before SIGKILL")
        cli.os.killpg = boom_killpg
        cli.time.sleep = lambda s: None               # don't actually wait ~5s
        try:
            assert cli._stop_one(wl, force=True) == "killed"
            assert not wl.pid.exists()
        finally:
            cli._alive = real_alive
            cli.os.getpgid = real_getpgid
            cli.os.killpg = real_killpg
            cli.time.sleep = real_sleep


def test_alive_proc_read_failure_defaults_alive(tmp: Path):
    """Race branch (cli.py 70-71): ``os.kill(pid,0)`` succeeds (pid exists) but the
    ``/proc/<pid>/stat`` read fails (the process vanished between the two calls, or
    /proc is unavailable). ``_alive`` conservatively returns True. We simulate the
    race by patching ``cli.Path`` so the /proc read raises OSError, using our own
    (definitely-live, non-zombie) pid so os.kill succeeds."""
    real_Path = cli.Path

    class _BoomPath:
        def __init__(self, *a, **k):
            pass
        def read_text(self, *a, **k):
            raise OSError("simulated /proc read failure")

    cli.Path = _BoomPath
    try:
        assert cli._alive(os.getpid()) is True        # kill ok, /proc read boom -> True
    finally:
        cli.Path = real_Path


def test_stop_one_force_getpgid_raises(tmp: Path):
    """Force path where ``os.getpgid`` raises (cli.py 238-239): the process is seen
    alive by ``_alive`` but disappears before ``getpgid`` — the ProcessLookupError
    is swallowed, then the wait loop finds it dead and returns 'killed'. We stub
    ``cli.os.getpgid`` to raise and use a dead pid for the subsequent _alive checks
    so the loop exits at once."""
    import types
    with _project_env(tmp):
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")

        real_getpgid = cli.os.getpgid
        real_alive = cli._alive
        # first _alive (guard) True, then getpgid raises, then loop sees dead
        state = {"n": 0}

        def fake_alive(pid):
            state["n"] += 1
            return state["n"] == 1                     # alive once (the guard), dead after

        def boom_getpgid(pid):
            raise ProcessLookupError("gone between alive and getpgid")

        wl.pid.write_text("2000000000")
        cli.os.getpgid = boom_getpgid
        cli._alive = fake_alive
        try:
            assert cli._stop_one(wl, force=True) == "killed"
            assert not wl.pid.exists()
        finally:
            cli.os.getpgid = real_getpgid
            cli._alive = real_alive


def test_read_status_missing_and_bad_json(tmp: Path):
    with _project_env(tmp):
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        # do_new wrote a valid status; corrupt it -> {}
        wl.status.write_text("{ not json", encoding="utf-8")
        assert cli._read_status(wl) == {}
        # remove it -> {}
        wl.status.unlink()
        assert cli._read_status(wl) == {}


def test_alive_permission_error_means_alive():
    """A pid we can't signal (owned by another user, e.g. init pid 1) raises
    PermissionError from ``os.kill(pid, 0)`` -> treated as alive ('exists but not
    ours'). pid 1 always exists and is root-owned when we're not root."""
    if os.geteuid() == 0:
        return  # as root os.kill(1,0) succeeds; the PermissionError branch is unreachable
    assert cli._alive(1) is True


def test_alive_zombie_is_dead():
    """A child that exited but hasn't been reaped is a zombie; /proc reports state
    'Z' and ``_alive`` must call it dead. We fork a child that exits immediately
    and do NOT wait() it, so it lingers as a zombie we own."""
    import subprocess
    import time
    # 'true' exits at once; without wait() it becomes a zombie child of us.
    proc = subprocess.Popen(["true"])
    try:
        # wait for the kernel to mark it Z (exited, unreaped)
        pid = proc.pid
        def is_zombie():
            try:
                stat = Path(f"/proc/{pid}/stat").read_text()
                return stat.rsplit(")", 1)[1].split()[0] == "Z"
            except (OSError, IndexError):
                return False
        end = time.time() + 5
        while time.time() < end and not is_zombie():
            time.sleep(0.02)
        assert is_zombie(), "child did not become a zombie"
        assert cli._alive(pid) is False              # /proc state 'Z' => dead
    finally:
        proc.wait()                                   # reap it


# --------------------------------------------------------------------------- #
# worker_status labels                                                          #
# --------------------------------------------------------------------------- #

def test_worker_status_stuck_label(tmp: Path):
    """alive + state=running + round_started_at far in the past -> 'stuck?'."""
    with _project_env(tmp, HYPROVER_ROUND_HARD_TIMEOUT="10"):
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        wl.pid.write_text(str(os.getpid()))          # our pid => alive
        old = 1.0                                     # epoch ~1970 => hugely stale
        wl.status.write_text(json.dumps(
            {"state": "running", "round": 5, "round_started_at": old,
             "last_round_at": old, "last_fact_id": "F9"}))
        s = cli.worker_status(wl)
        assert s["alive"] is True and s["label"] == "stuck?"
        assert s["age_s"] is not None and s["last_fact_id"] == "F9"


def test_waiting_worker_is_alive_not_stuck_and_not_restarted(tmp: Path):
    from hyprover.execution.wakeup import InputWatcher

    with _project_env(tmp, HYPROVER_ROUND_HARD_TIMEOUT="1"), _patch_spawn() as fake:
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        cli.do_assign("P/high", "Assigned task")
        watcher = InputWatcher(wl.project_dir, wl.task)
        baseline = watcher.read()
        wl.pid.write_text(str(os.getpid()))
        wl.status.write_text(json.dumps({"state": "waiting", "round": 5, "round_started_at": 1,
                                         "waiting_since": 10, "wake_reason": ["task"],
                                         "input_fingerprint": baseline.digest, "input_error": None}))
        state = cli.worker_status(wl)
        assert state["alive"] and state["label"] == "waiting"
        assert state["waiting_since"] == 10 and state["wake_reason"] == ["task"]
        assert state["input_fingerprint"] == baseline.digest
        assert cli.do_start("P/high") == [{"worker": "high", "result": "already-running"}]
        assert fake.calls == []
        cli.do_assign("P/high", "Assigned task")
        assert watcher.read() == baseline
        cli.do_assign("P/high", "New follow-up")
        assert watcher.read().changed_from(baseline) == ("task",)
        wl.pid.write_text("2000000000")
        assert cli.worker_status(wl)["label"] == "dead"


def test_worker_status_working_and_dead_labels(tmp: Path):
    with _project_env(tmp):
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        # alive + running but fresh => 'working'
        import time
        wl.pid.write_text(str(os.getpid()))
        wl.status.write_text(json.dumps(
            {"state": "running", "round": 2, "round_started_at": time.time()}))
        assert cli.worker_status(wl)["label"] == "working"
        # not alive + unknown terminal state => 'dead'
        wl.pid.write_text("2000000000")
        wl.status.write_text(json.dumps({"state": "weird", "round": 3}))
        d = cli.worker_status(wl)
        assert d["alive"] is False and d["label"] == "dead" and d["age_s"] is None
        # not alive + recognized terminal state => that state as label
        wl.status.write_text(json.dumps({"state": "deadline", "round": 3}))
        assert cli.worker_status(wl)["label"] == "deadline"


# --------------------------------------------------------------------------- #
# do_start: mocked spawn, locked path, no-workers, project-wide + stagger       #
# --------------------------------------------------------------------------- #

def test_do_start_calls_spawn_with_worker_dir(tmp: Path):
    with _project_env(tmp), _patch_spawn() as fake:
        cli.do_new("P", roles="high:1")
        res = cli.do_start("P/high")
        assert res == [{"worker": "high", "result": "started"}]
        assert fake.calls == [_wl("P", "high").dir]
        wl = _wl("P", "high")
        assert cli._read_pid(wl) == os.getpid()      # pid file written from fake pid
        # second start sees our-own-pid as alive => idempotent already-running
        res2 = cli.do_start("P/high")
        assert res2 == [{"worker": "high", "result": "already-running"}]
        assert len(fake.calls) == 1                   # spawn NOT called again


def test_do_start_locked_returns_locked(tmp: Path):
    import fcntl
    with _project_env(tmp), _patch_spawn():
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        wl.dir.mkdir(parents=True, exist_ok=True)
        held = open(wl.lock, "w")
        fcntl.flock(held, fcntl.LOCK_EX)
        try:
            assert cli._start_one(wl) == "locked"
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)
            held.close()


def test_do_start_clears_stale_stop(tmp: Path):
    with _project_env(tmp), _patch_spawn():
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        wl.dir.mkdir(parents=True, exist_ok=True)
        wl.stop.touch()
        assert cli._start_one(wl) == "started"
        assert not wl.stop.exists()                   # stale stop cleared


def test_do_start_no_workers_raises(tmp: Path):
    with _project_env(tmp), _patch_spawn():
        e = _expect_exit(cli.do_start, "ghost")
        assert "no workers for target" in str(e)


def test_do_start_project_wide_stagger(tmp: Path):
    with _project_env(tmp), _patch_spawn() as fake:
        cli.do_new("P", roles="high:2")
        res = cli.do_start("P", stagger=0)            # stagger 0 => no sleep
        assert {r["worker"] for r in res} == {"high", "high2"}
        assert {r["result"] for r in res} == {"started"}
        assert {c.name for c in fake.calls} == {"high", "high2"}


# --------------------------------------------------------------------------- #
# do_status / do_stop no-workers                                                #
# --------------------------------------------------------------------------- #

def test_do_status_no_workers_raises(tmp: Path):
    with _project_env(tmp):
        assert "no workers" in str(_expect_exit(cli.do_status, "ghost"))


def test_do_stop_no_workers_raises(tmp: Path):
    with _project_env(tmp):
        assert "no workers" in str(_expect_exit(cli.do_stop, "ghost"))


# --------------------------------------------------------------------------- #
# _stop_one: not-running (graceful + force), graceful touch, force kill path     #
# --------------------------------------------------------------------------- #

def test_stop_one_not_running_graceful(tmp: Path):
    with _project_env(tmp):
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        assert cli._stop_one(wl, force=False) == "not-running"


def test_stop_one_not_running_force_cleans_pid(tmp: Path):
    with _project_env(tmp):
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        wl.pid.write_text("2000000000")               # dead pid
        assert cli._stop_one(wl, force=True) == "not-running"
        assert not wl.pid.exists()                     # stale pid removed


def test_stop_one_graceful_touches_stop(tmp: Path):
    with _project_env(tmp):
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        wl.pid.write_text(str(os.getpid()))            # alive (our pid)
        assert cli._stop_one(wl, force=False) == "stopping (graceful)"
        assert wl.stop.exists()
        wl.stop.unlink()                               # don't leave a stop flag on us


def test_stop_one_force_kills_a_real_child(tmp: Path):
    """Spawn a genuine harmless child (sleep), record its pid, force-stop it, and
    assert it's reaped as 'killed'. This is a real process we own — safe to kill,
    never a codex/worker (which the RULES forbid launching)."""
    import subprocess
    with _project_env(tmp):
        cli.do_new("P", roles="high:1")
        wl = _wl("P", "high")
        proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
        wl.pid.write_text(str(proc.pid))
        try:
            assert cli._stop_one(wl, force=True) == "killed"
            assert not wl.pid.exists()
        finally:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# existing-project validation and legacy compatibility                         #
# --------------------------------------------------------------------------- #

def test_existing_project_validation_precedes_every_write(tmp: Path):
    from hyprover.execution import loop, scaffold

    for broken in ("missing", "legacy", "malformed", "metadata"):
        case = tmp / broken
        case.mkdir()
        with _project_env(case), _patch_spawn() as fake:
            cli.do_new("P", roles="high:1")
            wl = _wl("P", "high")
            notebook = wl.project_dir / "KNOWLEDGE.md"
            if broken in ("missing", "legacy"):
                notebook.unlink()
                if broken == "legacy":
                    (wl.project_dir / "fact_graph").mkdir()
                    (wl.project_dir / "global_memory").mkdir()
            elif broken == "malformed":
                notebook.write_text("# Not a valid knowledge notebook\n")
            else:
                (wl.project_dir / "project.json").write_text('{"knowledge_format":"old"}')
            wl.stop.write_text("preserve stop")
            target = wl.project_dir / "TARGET.md"
            target.write_text("preserve target")

            def tree():
                return {str(p.relative_to(wl.project_dir)): (p.read_bytes(), p.stat().st_mtime_ns)
                        for p in wl.project_dir.rglob("*") if p.is_file() and not p.is_symlink()}

            before = tree()
            operations = (
                lambda: cli.do_new("P", roles="high:1"),
                lambda: cli.do_assign("P/high", "replacement"),
                lambda: cli.do_start("P/high"),
                lambda: scaffold.write_codex_config(wl),
                lambda: scaffold.spawn_loop(wl.dir),
                lambda: loop.write_status(wl, state="running"),
                lambda: loop.run_round(wl, {"MODEL": "fake", "REASONING_EFFORT": "high"},
                                       "prompt", wl.logs / "round.log", 1),
            )
            for operation in operations:
                _expect_exit(operation)
                assert tree() == before
            assert loop.main(str(wl.dir)) == 2
            assert tree() == before and fake.calls == []
            assert not (wl.project_dir / "papers").exists()
            assert not wl.lock.exists()


def test_stop_legacy_project_remains_available(tmp: Path):
    with _project_env(tmp):
        wl = _wl("old", "high")
        wl.dir.mkdir(parents=True)
        (wl.project_dir / "fact_graph").mkdir()
        wl.pid.write_text(str(os.getpid()))
        assert cli.do_stop("old/high") == [{"worker": "high", "result": "stopping (graceful)"}]
        assert wl.stop.exists()
        assert not (wl.project_dir / "KNOWLEDGE.md").exists()
        assert not (wl.project_dir / "knowledge").exists()


# --------------------------------------------------------------------------- #
# do_list: bad project.json, formatting                                         #
# --------------------------------------------------------------------------- #

def test_do_list_bad_project_json(tmp: Path):
    with _project_env(tmp):
        cli.do_new("P", roles="high:1", model="gpt-5.5")
        (L.project_dir("P") / "project.json").write_text("{ broken", encoding="utf-8")
        rows = {r["project"]: r for r in cli.do_list()}
        assert rows["P"]["model"] == "—"               # unparseable meta => dash


def test_do_list_missing_project_json(tmp: Path):
    with _project_env(tmp):
        cli.do_new("P", roles="high:1")
        (L.project_dir("P") / "project.json").unlink()
        rows = {r["project"]: r for r in cli.do_list()}
        assert rows["P"]["model"] == "—"


# --------------------------------------------------------------------------- #
# text formatters _fmt_list / _fmt_status                                       #
# --------------------------------------------------------------------------- #

def test_fmt_list_empty_and_rows():
    assert cli._fmt_list([]) == "(no projects under the agents root)"
    rows = [{"project": "Proj", "workers": 3, "live": 1, "model": "gpt-5.5"}]
    out = cli._fmt_list(rows)
    assert "PROJECT" in out and "Proj" in out and "gpt-5.5" in out


def test_fmt_status_rows():
    rows = [
        {"worker": "high", "label": "working", "state": "running", "round": 4,
         "age_s": 12.4, "last_fact_id": "F7"},
        {"worker": "xhigh", "label": "dead", "state": "created", "round": 0,
         "age_s": None, "last_fact_id": None},
    ]
    out = cli._fmt_status(rows)
    assert "WORKER" in out and "high" in out and "xhigh" in out
    assert "12s" in out                                # age rendered from float
    assert "—" in out                                  # None age / fact => dash


# --------------------------------------------------------------------------- #
# _task_from_args: --task / --file / --stdin / none                             #
# --------------------------------------------------------------------------- #

class _Args:
    def __init__(self, task=None, file=None, stdin=False):
        self.task = task
        self.file = file
        self.stdin = stdin


def test_task_from_args_task():
    assert cli._task_from_args(_Args(task="direct task")) == "direct task"


def test_task_from_args_file(tmp: Path):
    p = tmp / "task.txt"
    p.write_text("from a file\n", encoding="utf-8")
    assert cli._task_from_args(_Args(file=str(p))) == "from a file\n"


def test_task_from_args_stdin(monkeypatch=None):
    import sys
    old = sys.stdin
    sys.stdin = io.StringIO("piped task\n")
    try:
        assert cli._task_from_args(_Args(stdin=True)) == "piped task\n"
    finally:
        sys.stdin = old


def test_task_from_args_none_raises():
    assert "one of --task" in str(_expect_exit(cli._task_from_args, _Args()))


# --------------------------------------------------------------------------- #
# build_parser                                                                  #
# --------------------------------------------------------------------------- #

def test_build_parser_all_verbs():
    p = cli.build_parser()
    assert p.parse_args(["list", "--json"]).cmd == "list"
    a = p.parse_args(["new", "P", "--roles", "high:2", "--model", "m"])
    assert a.cmd == "new" and a.project == "P" and a.roles == "high:2" and a.model == "m"
    a = p.parse_args(["assign", "P/high", "--task", "t"])
    assert a.cmd == "assign" and a.target == "P/high" and a.task == "t"
    assert _expect_exit(p.parse_args, ["finalize", "P"]).code == 2
    assert p.parse_args(["start", "P"]).cmd == "start"
    assert p.parse_args(["status", "P", "--json"]).json is True
    assert p.parse_args(["stop", "P", "--force"]).force is True
    article = p.parse_args(["article", "resume", "P", "--run-id", "run", "--refresh-references"])
    assert article.cmd == "article" and article.command == "resume" and article.refresh_references
    # subcommand is required
    try:
        p.parse_args([])
        raise AssertionError("expected argparse to require a subcommand")
    except SystemExit:
        pass


# --------------------------------------------------------------------------- #
# main dispatch — every verb, text + json branches                             #
# --------------------------------------------------------------------------- #

def _run_main(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, buf.getvalue()


def test_main_article_status_dispatch(tmp):
    rc, out = _run_main(["article", "status", str(tmp.resolve())])
    assert rc == 0 and json.loads(out) == {"jobs": []}


def test_main_new_then_list_text_and_json(tmp: Path):
    with _project_env(tmp), _patch_spawn():
        rc, out = _run_main(["new", "P", "--roles", "high:2", "--model", "gpt-5.5"])
        assert rc == 0 and "created P with 2 workers" in out and "high" in out
        rc, out = _run_main(["list"])
        assert rc == 0 and "PROJECT" in out and "P" in out
        rc, out = _run_main(["list", "--json"])
        rows = json.loads(out)
        assert rc == 0 and rows[0]["project"] == "P" and rows[0]["workers"] == 2


def test_main_assign(tmp: Path):
    with _project_env(tmp), _patch_spawn():
        _run_main(["new", "P", "--roles", "high:1"])
        rc, out = _run_main(["assign", "P/high", "--task", "prove lemma 4"])
        assert rc == 0 and "assigned P/high" in out
        assert _wl("P", "high").task.read_text() == "prove lemma 4\n"


def test_main_start_status_stop(tmp: Path):
    with _project_env(tmp), _patch_spawn() as fake:
        _run_main(["new", "P", "--roles", "high:1"])
        rc, out = _run_main(["start", "P/high"])
        assert rc == 0 and "high: started" in out and len(fake.calls) == 1
        # status text branch (worker is "alive" = our pid)
        rc, out = _run_main(["status", "P/high"])
        assert rc == 0 and "WORKER" in out
        # status json branch
        rc, out = _run_main(["status", "P/high", "--json"])
        assert rc == 0 and json.loads(out)[0]["worker"] == "high"
        # stop graceful (worker "alive" via our pid => touches .stop)
        rc, out = _run_main(["stop", "P/high"])
        assert rc == 0 and "graceful" in out
        _wl("P", "high").stop.unlink(missing_ok=True)


def test_main_stop_force_not_running(tmp: Path):
    with _project_env(tmp), _patch_spawn():
        _run_main(["new", "P", "--roles", "high:1"])
        rc, out = _run_main(["stop", "P/high", "--force"])
        assert rc == 0 and "not-running" in out


# --------------------------------------------------------------------------- #
# python -m hyprover.orchestration  (the __main__ entry point)                     #
# --------------------------------------------------------------------------- #

def test_dunder_main_entrypoint(tmp: Path):
    """Exercise ``__main__.py`` via runpy with a mocked ``main`` so no real verb
    runs. Asserts it calls ``sys.exit`` with main()'s return code."""
    import hyprover.orchestration.cli as climod

    orig = climod.main
    calls = {}

    def fake_main(argv=None):
        calls["argv"] = argv
        return 0

    climod.main = fake_main
    old_argv = None
    try:
        import sys
        old_argv = sys.argv[:]
        sys.argv = ["hyprover", "list"]
        try:
            runpy.run_module("hyprover.orchestration", run_name="__main__")
            raise AssertionError("expected SystemExit from __main__")
        except SystemExit as e:
            assert e.code == 0
    finally:
        climod.main = orig
        if old_argv is not None:
            import sys
            sys.argv = old_argv


# --------------------------------------------------------------------------- #
# runner (standalone parity with test_orchestration.py)                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    no_arg = [
        test_fmt_list_empty_and_rows, test_fmt_status_rows,
        test_task_from_args_task, test_task_from_args_stdin,
        test_task_from_args_none_raises, test_build_parser_all_verbs,
        test_alive_permission_error_means_alive, test_alive_zombie_is_dead,
    ]
    tmp_tests = [
        test_alive_variants,
        test_stop_one_force_sigkill_fallback,
        test_stop_one_force_sigkill_killpg_raises,
        test_alive_proc_read_failure_defaults_alive,
        test_stop_one_force_getpgid_raises,
        test_read_pid_missing_and_garbage, test_read_status_missing_and_bad_json,
        test_worker_status_stuck_label, test_worker_status_working_and_dead_labels,
        test_do_start_calls_spawn_with_worker_dir, test_do_start_locked_returns_locked,
        test_do_start_clears_stale_stop, test_do_start_no_workers_raises,
        test_do_start_project_wide_stagger, test_do_status_no_workers_raises,
        test_do_stop_no_workers_raises, test_stop_one_not_running_graceful,
        test_stop_one_not_running_force_cleans_pid, test_stop_one_graceful_touches_stop,
        test_stop_one_force_kills_a_real_child, test_do_list_bad_project_json,
        test_do_list_missing_project_json, test_task_from_args_file,
        test_existing_project_validation_precedes_every_write,
        test_stop_legacy_project_remains_available,
        test_main_new_then_list_text_and_json, test_main_assign,
        test_main_start_status_stop, test_main_stop_force_not_running,
        test_dunder_main_entrypoint,
    ]
    for t in no_arg:
        t()
        print(f"  [ok] {t.__name__}")
    for t in tmp_tests:
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
        print(f"  [ok] {t.__name__}")
    print("ALL CLI VERB TESTS PASSED")


if __name__ == "__main__":
    main()
