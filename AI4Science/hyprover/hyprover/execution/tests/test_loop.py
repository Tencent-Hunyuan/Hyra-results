"""Offline tests for hyprover.execution.loop + __main__ (no real codex, no network).

Covers the round driver end-to-end without ever launching a real codex:

  - ``run_round`` against a FIXED fake-codex stub script: a chosen exit code, a
    hard-timeout (terminate → 124), and a missing binary (→ 127). These drive the
    real ``subprocess.Popen`` path in loop.py.
  - the ``main`` outer loop: stop-flag / deadline / max-rounds / consecutive-
    failure caps, the codex-missing (127) short-circuit, hash-gated ``waiting``,
    retries and terminal statuses. ``run_round`` is monkeypatched so no subprocess spawns.
  - the SIGTERM handler (_on_term): terminates the in-flight child, writes
    ``terminated`` status, and exits 0.
  - __main__: ``runpy.run_module("hyprover.execution", run_name="__main__")`` with the
    loop entry patched, covering the argv guard + dispatch without spawning.
  - the remaining small error/edge branches in loop / layout / scaffold helpers.

Runs standalone (``python -m hyprover.execution.tests.test_loop``) and pytest.
"""

from __future__ import annotations

import json
import os
import runpy
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from hyprover.execution import layout as L
from hyprover.execution import loop, scaffold


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
def _restore_sigterm():
    """main() installs a SIGTERM handler; save/restore so tests don't leak it."""
    old = signal.getsignal(signal.SIGTERM)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old)


def _mk_worker(tmp: Path, name: str = "high") -> L.WorkerLayout:
    """A minimal worker home under tmp: <tmp>/proj/workers/<name>."""
    wl = L.WorkerLayout(tmp / "proj" / "workers" / name)
    L.KnowledgeStore.initialize(wl.project_dir, problem_id="proj")
    wl.dir.mkdir(parents=True)
    wl.task.write_text("Initial task.\n")
    return wl


def _write_fake_codex(tmp: Path, body: str) -> Path:
    """Write an executable python fake-codex stub and return its path. The stub
    ignores all the exec args and just does what ``body`` says."""
    p = tmp / "fake_codex"
    p.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    p.chmod(0o755)
    return p


# --- run_round: chosen exit code ------------------------------------------- #

def test_run_round_returns_codex_rc(tmp: Path):
    wl = _mk_worker(tmp)
    fake = _write_fake_codex(tmp, "import sys\nsys.stdout.write('hello from codex\\n')\nsys.exit(3)\n")
    log = wl.dir / "round.log"
    with _env(HYPROVER_CODEX_BIN=str(fake)):
        rc = loop.run_round(wl, {"MODEL": "m", "REASONING_EFFORT": "high"},
                            "prompt", log, hard_timeout=30)
    assert rc == 3
    assert "hello from codex" in log.read_text()
    assert loop._Child.proc is None            # cleared in finally


def test_run_round_success_rc0(tmp: Path):
    wl = _mk_worker(tmp)
    fake = _write_fake_codex(tmp, "import sys\nsys.exit(0)\n")
    log = wl.dir / "round.log"
    with _env(HYPROVER_CODEX_BIN=str(fake)):
        rc = loop.run_round(wl, {"MODEL": "m", "REASONING_EFFORT": "high"},
                            "prompt", log, hard_timeout=0)   # 0 => no timeout (wait forever)
    assert rc == 0


# --- run_round: hard timeout → terminate → 124 ----------------------------- #

def test_run_round_hard_timeout_terminates(tmp: Path):
    wl = _mk_worker(tmp)
    # sleeps far past the tiny hard_timeout; a plain terminate() ends it.
    fake = _write_fake_codex(tmp, "import time\ntime.sleep(60)\n")
    log = wl.dir / "round.log"
    with _env(HYPROVER_CODEX_BIN=str(fake)):
        rc = loop.run_round(wl, {"MODEL": "m", "REASONING_EFFORT": "high"},
                            "prompt", log, hard_timeout=1)
    assert rc == 124
    assert "hard-timeout after 1s" in log.read_text()
    assert loop._Child.proc is None


# --- run_round: missing binary → 127 --------------------------------------- #

def test_run_round_missing_binary_returns_127(tmp: Path):
    wl = _mk_worker(tmp)
    missing = tmp / "does_not_exist_codex"
    log = wl.dir / "round.log"
    with _env(HYPROVER_CODEX_BIN=str(missing)):
        rc = loop.run_round(wl, {"MODEL": "m", "REASONING_EFFORT": "high"},
                            "prompt", log, hard_timeout=30)
    assert rc == 127
    assert "codex binary not found" in log.read_text()


# --- run_round: unresponsive child → terminate times out → kill → 124 ------ #

def test_run_round_timeout_then_kill(tmp: Path):
    """A child that ignores terminate() (wait(10) times out) is force-killed. We
    fake Popen so the 10s terminate-grace does not slow the test."""
    wl = _mk_worker(tmp)
    log = wl.dir / "round.log"

    class _StubProc:
        def __init__(self):
            self.terminated = False
            self.killed = False
            self._waits = 0

        def wait(self, timeout=None):
            self._waits += 1
            # 1st wait = the hard-timeout expiry; 2nd wait = the 10s grace expiry.
            raise subprocess.TimeoutExpired(cmd="codex", timeout=timeout)

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    stub = _StubProc()
    orig_popen = subprocess.Popen
    subprocess.Popen = lambda *a, **k: stub
    try:
        with _env(HYPROVER_CODEX_BIN=str(tmp / "anything")):
            rc = loop.run_round(wl, {"MODEL": "m", "REASONING_EFFORT": "high"},
                                "prompt", log, hard_timeout=1)
    finally:
        subprocess.Popen = orig_popen
    assert rc == 124
    assert stub.terminated and stub.killed          # terminate → (grace expired) → kill
    assert loop._Child.proc is None


# --- main loop: stop flag → graceful stop ---------------------------------- #

def test_main_stops_on_stop_flag(tmp: Path):
    wl = _mk_worker(tmp)
    wl.stop.touch()          # stop before the first round
    with _restore_sigterm(), _env(HYPROVER_ROUND_BEAT="0"):
        _patch_run_round(lambda *a, **k: 0)
        try:
            rc = loop.main(str(wl.dir))
        finally:
            _unpatch_run_round()
    assert rc == 0
    assert not wl.stop.exists()                       # consumed
    assert json.loads(wl.status.read_text())["state"] == "stopped"


# --- main loop: deadline → stop -------------------------------------------- #

def test_main_stops_on_deadline(tmp: Path):
    wl = _mk_worker(tmp)
    (wl.project_dir / L.DEADLINE_FILE).write_text("1")   # epoch 1 = long past
    with _restore_sigterm(), _env(HYPROVER_ROUND_BEAT="0"):
        _patch_run_round(lambda *a, **k: 0)
        try:
            rc = loop.main(str(wl.dir))
        finally:
            _unpatch_run_round()
    assert rc == 0
    assert json.loads(wl.status.read_text())["state"] == "deadline"


# --- main loop: max-rounds cap --------------------------------------------- #

def test_main_max_rounds_cap(tmp: Path):
    wl = _mk_worker(tmp)
    calls = []
    with _restore_sigterm(), _env(HYPROVER_ROUND_BEAT="0", HYPROVER_MAX_ROUNDS="2",
                                  HYPROVER_MAX_CONSEC_FAILURES="0"):
        def changed_round(*args):
            calls.append(1)
            wl.task.write_text(f"Task update {len(calls)}")
            return 0
        _patch_run_round(changed_round)
        try:
            rc = loop.main(str(wl.dir))
        finally:
            _unpatch_run_round()
    assert rc == 0
    assert len(calls) == 2                              # exactly max_rounds rounds ran
    st = json.loads(wl.status.read_text())
    assert st["state"] == "max_rounds"
    assert st["round"] == 2 and st["last_rc"] == 0


def test_restart_preserves_round_logs_and_injects_proof_only(tmp: Path, monkeypatch):
    wl = _mk_worker(tmp)
    wl.logs.mkdir()
    previous = wl.logs / "round_12.log"
    previous.write_text("previous research log")
    (wl.project_dir / "project.json").write_text(json.dumps({
        "research_mode": "proof_only", "knowledge_format": L.KNOWLEDGE_FORMAT,
    }))
    calls = []

    def resumed_round(w, role, prompt, log_path, hard_timeout):
        calls.append(log_path.name)
        assert prompt.startswith("PROOF-ONLY PROJECT:")
        assert "Still audit proofs" in prompt
        assert not log_path.exists()
        log_path.write_text("new round")
        wl.task.write_text("Follow up " + log_path.name)
        return 0

    monkeypatch.setattr(loop, "run_round", resumed_round)
    with _restore_sigterm(), _env(HYPROVER_ROUND_BEAT="0", HYPROVER_MAX_ROUNDS="2"):
        assert loop.main(str(wl.dir)) == 0
    assert calls == ["round_13.log", "round_14.log"]
    assert previous.read_text() == "previous research log"
    assert json.loads(wl.status.read_text())["round"] == 14


# --- main loop: consecutive-failure cap → error / rc 1 --------------------- #

def test_main_consecutive_failure_cap(tmp: Path):
    wl = _mk_worker(tmp)
    log = wl.logs / "round_1.log"   # a fact id in a round log flows into status
    with _restore_sigterm(), _env(HYPROVER_ROUND_BEAT="0", HYPROVER_MAX_CONSEC_FAILURES="2",
                                  HYPROVER_MAX_ROUNDS="0"):
        def _fail(w, role, prompt, log_path, ht):
            log_path.write_text('"fact_id": "0123456789abcdef"\n')
            return 5                                    # a failing rc (not 0/124)
        _patch_run_round(_fail)
        try:
            rc = loop.main(str(wl.dir))
        finally:
            _unpatch_run_round()
    assert rc == 1
    st = json.loads(wl.status.read_text())
    assert st["state"] == "error" and "consecutive failed rounds" in st["error"]
    assert st.get("last_fact_id") is None
    assert st["last_rc"] == 5


def test_main_timeout_rc124_does_not_count_as_failure(tmp: Path):
    """rc 124 (hard-timeout) resets the consecutive-failure counter, so a run of
    124s never trips the failure cap — it must stop via max_rounds instead."""
    wl = _mk_worker(tmp)
    with _restore_sigterm(), _env(HYPROVER_ROUND_BEAT="0", HYPROVER_MAX_CONSEC_FAILURES="2",
                                  HYPROVER_MAX_ROUNDS="3"):
        _patch_run_round(lambda *a, **k: 124)
        try:
            rc = loop.main(str(wl.dir))
        finally:
            _unpatch_run_round()
    assert rc == 0
    assert json.loads(wl.status.read_text())["state"] == "max_rounds"


# --- main loop: codex missing (127) short-circuits ------------------------- #

def test_main_codex_missing_127(tmp: Path):
    wl = _mk_worker(tmp)
    with _restore_sigterm(), _env(HYPROVER_ROUND_BEAT="0"):
        _patch_run_round(lambda *a, **k: 127)
        try:
            rc = loop.main(str(wl.dir))
        finally:
            _unpatch_run_round()
    assert rc == 127
    st = json.loads(wl.status.read_text())
    assert st["state"] == "error" and st["error"] == "codex binary not found"


# --- main: bad worker dir → rc 2 ------------------------------------------- #

def test_main_missing_worker_dir(tmp: Path):
    rc = loop.main(str(tmp / "nope"))
    assert rc == 2


# --- SIGTERM handler: terminate child, write terminated, exit 0 ------------ #

def test_main_sigterm_handler(tmp: Path):
    wl = _mk_worker(tmp)

    class _FakeProc:
        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

    fake_proc = _FakeProc()

    # run_round: install a live child then deliver SIGTERM to ourselves so the
    # loop's own handler fires (covers _on_term end to end).
    def _round(w, role, prompt, log_path, ht):
        loop._Child.proc = fake_proc
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(2)                     # give the signal time to be delivered
        return 0

    with _restore_sigterm(), _env(HYPROVER_ROUND_BEAT="0"):
        _patch_run_round(_round)
        try:
            try:
                loop.main(str(wl.dir))
                assert False, "handler should sys.exit(0)"
            except SystemExit as e:
                assert e.code == 0
        finally:
            _unpatch_run_round()
            loop._Child.proc = None
    assert fake_proc.terminated
    assert json.loads(wl.status.read_text())["state"] == "terminated"


# --- write_status: recovers from a corrupt existing status ----------------- #

def test_write_status_corrupt_existing_recovers(tmp: Path):
    wl = _mk_worker(tmp)
    wl.status.write_text("{not json")            # corrupt → JSONDecodeError branch
    loop.write_status(wl, state="running")
    st = json.loads(wl.status.read_text())
    assert st["state"] == "running" and st["worker"] == "high"


# --- _parse_last_fact_id: unreadable path → None --------------------------- #

def test_parse_last_fact_id_missing_file(tmp: Path):
    assert loop._parse_last_fact_id(tmp / "no_such.log") is None   # OSError branch


# --- _cleanup_pid: only removes a .pid that points at us ------------------- #

def test_cleanup_pid_removes_own(tmp: Path):
    wl = _mk_worker(tmp)
    wl.pid.write_text(str(os.getpid()))
    loop._cleanup_pid(wl)
    assert not wl.pid.exists()


def test_cleanup_pid_keeps_foreign(tmp: Path):
    wl = _mk_worker(tmp)
    wl.pid.write_text("999999999")            # some other pid
    loop._cleanup_pid(wl)
    assert wl.pid.exists()                     # left intact


def test_cleanup_pid_swallows_oserror(tmp: Path):
    """A .pid that cannot be read (here: it is a directory) → OSError swallowed."""
    wl = _mk_worker(tmp)
    wl.pid.mkdir()                             # read_text on a dir raises OSError
    loop._cleanup_pid(wl)                      # must not raise
    assert wl.pid.exists()


# --- main loop: successful rounds honor stop before any sleep -------------- #

def test_main_success_honors_stop_before_sleep(tmp: Path):
    """A stop at round completion takes priority over the old beat interval."""
    wl = _mk_worker(tmp)
    slept = []
    orig_sleep = time.sleep

    def _one_then_stop(*a, **k):
        wl.stop.touch()          # stop after the first round completes
        return 0

    time.sleep = lambda s: slept.append(s)
    try:
        with _restore_sigterm(), _env(HYPROVER_ROUND_BEAT="7", HYPROVER_MAX_ROUNDS="0",
                                      HYPROVER_MAX_CONSEC_FAILURES="0"):
            _patch_run_round(_one_then_stop)
            try:
                rc = loop.main(str(wl.dir))
            finally:
                _unpatch_run_round()
    finally:
        time.sleep = orig_sleep
    assert rc == 0
    assert slept == []


def test_unlimited_worker_retries_then_recovers(tmp: Path, monkeypatch):
    wl = _mk_worker(tmp)
    outcomes = iter([1] * 7 + [0])
    sleeps = []
    rounds = []

    def run_round(*args):
        rc = next(outcomes)
        rounds.append(rc)
        if rc == 0:
            wl.stop.touch()
        return rc

    monkeypatch.setattr(loop, "run_round", run_round)
    monkeypatch.setattr(loop.time, "sleep", lambda seconds: sleeps.append(seconds))
    with _restore_sigterm(), _env(HYPROVER_MAX_CONSEC_FAILURES="0", HYPROVER_MAX_ROUNDS="0",
                                  HYPROVER_RETRY_INTERVAL="30", HYPROVER_ROUND_BEAT="5"):
        assert loop.main(str(wl.dir)) == 0
    assert rounds == [1] * 7 + [0]
    assert sleeps == [30.0] * 7
    state = json.loads(wl.status.read_text())
    assert state["state"] == "stopped" and state["consecutive_failures"] == 0
    assert state["error"] is None


def test_worker_stop_during_retry_wait(tmp: Path, monkeypatch):
    wl = _mk_worker(tmp)
    calls = []
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        wl.stop.touch()

    monkeypatch.setattr(loop, "run_round", lambda *a: calls.append(1) or 1)
    monkeypatch.setattr(loop.time, "sleep", sleep)
    with _restore_sigterm(), _env(HYPROVER_MAX_CONSEC_FAILURES="0", HYPROVER_MAX_ROUNDS="0",
                                  HYPROVER_RETRY_INTERVAL="30"):
        assert loop.main(str(wl.dir)) == 0
    assert len(calls) == 1 and sleeps == [30.0]
    assert json.loads(wl.status.read_text())["state"] == "stopped"


def test_unlimited_worker_retries_missing_binary(tmp: Path, monkeypatch):
    wl = _mk_worker(tmp)
    monkeypatch.setattr(loop, "run_round", lambda *a: 127)
    monkeypatch.setattr(loop.time, "sleep", lambda seconds: wl.stop.touch())
    with _restore_sigterm(), _env(HYPROVER_MAX_CONSEC_FAILURES="0", HYPROVER_MAX_ROUNDS="0",
                                  HYPROVER_RETRY_INTERVAL="30"):
        assert loop.main(str(wl.dir)) == 0
    state = json.loads(wl.status.read_text())
    assert state["state"] == "stopped" and state["last_rc"] == 127


def test_success_waits_without_model_calls_or_status_churn(tmp: Path, monkeypatch):
    wl = _mk_worker(tmp)
    calls, sleeps, statuses = [], [], []

    def round_once(*args):
        calls.append(1)
        (wl.logs / "round_1.log").write_text("one completed round")
        return 0

    def poll(seconds):
        sleeps.append(seconds)
        assert calls == [1]
        statuses.append(wl.status.read_bytes())
        state = json.loads(statuses[-1])
        assert state["state"] == "waiting" and state["waiting_since"]
        assert state["next_round_at"] is None and state["round"] == 1
        assert state["wake_reason"] == ["initial"]
        if len(sleeps) == 4:
            wl.stop.touch()

    monkeypatch.setattr(loop, "run_round", round_once)
    monkeypatch.setattr(loop.time, "sleep", poll)
    with _restore_sigterm(), _env(HYPROVER_ROUND_BEAT="0", HYPROVER_MAX_ROUNDS="0"):
        assert loop.main(str(wl.dir)) == 0
    assert sleeps == [3.0] * 4
    assert len(set(statuses)) == 1
    assert len(list(wl.logs.glob("round_*.log"))) == 1


def test_task_change_wakes_once_and_records_reason(tmp: Path, monkeypatch):
    wl = _mk_worker(tmp)
    calls, sleeps = [], []

    def run(w, role, prompt, log_path, timeout):
        calls.append(prompt)
        log_path.write_text("finished")
        if len(calls) == 2:
            assert "Host wake reason: task." in prompt
        return 0

    def poll(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            wl.task.write_text("A new assignment")
        elif len(sleeps) == 2:
            wl.task.write_text("A new assignment")
        else:
            wl.stop.touch()

    monkeypatch.setattr(loop, "run_round", run)
    monkeypatch.setattr(loop.time, "sleep", poll)
    with _restore_sigterm(), _env(HYPROVER_MAX_ROUNDS="0"):
        assert loop.main(str(wl.dir)) == 0
    assert len(calls) == 2 and sleeps == [3.0] * 3
    assert json.loads(wl.status.read_text())["wake_reason"] == ["task"]


def test_main_agent_routes_wake_waiting_worker_once_per_content_change(tmp: Path, monkeypatch):
    wl = _mk_worker(tmp)
    store = L.KnowledgeStore(wl.project_dir)
    calls, sleeps = [], []
    route = None

    def run(w, role, prompt, log_path, timeout):
        calls.append(json.loads(wl.status.read_text())["wake_reason"])
        assert len(calls) <= 3
        log_path.write_text("completed")
        if len(calls) > 1:
            assert "Host wake reason: routes." in prompt
        return 0

    def poll(seconds):
        nonlocal route
        sleeps.append(seconds)
        assert json.loads(wl.status.read_text())["state"] == "waiting"
        if len(sleeps) == 1:
            route = store.record("route", "Main route", "Initial direction", "main_agent", "main")
            assert route.status == "proposed"
        elif len(sleeps) == 2:
            assert len(calls) == 2
            route = store.record("route", "Main route", "Initial direction", "main_agent", "main",
                                 entry_id=route.entry_id, expected_revision=route.revision)
        elif len(sleeps) == 3:
            assert len(calls) == 2
            route = store.record("route", "Main route", "New direction", "main_agent", "main",
                                 entry_id=route.entry_id, expected_revision=route.revision)
        elif len(sleeps) == 4:
            assert len(calls) == 3
            store.record("report", "Progress", "No new direction", "main_agent", "main")
        else:
            assert len(calls) == 3
            wl.stop.touch()

    monkeypatch.setattr(loop, "run_round", run)
    monkeypatch.setattr(loop.time, "sleep", poll)
    with _restore_sigterm(), _env(HYPROVER_MAX_ROUNDS="0"):
        assert loop.main(str(wl.dir)) == 0
    assert calls == [["initial"], ["routes"], ["routes"]]
    assert sleeps == [3.0] * 5
    assert len(list(wl.logs.glob("round_*.log"))) == 3
    assert json.loads(wl.status.read_text())["state"] == "stopped"


def test_in_round_updates_coalesce_without_report_self_wakeup(tmp: Path, monkeypatch):
    wl = _mk_worker(tmp)
    store = L.KnowledgeStore(wl.project_dir)
    calls = []

    def run(w, role, prompt, log_path, timeout):
        calls.append(prompt)
        store.record("report", "Progress", "Same progress", wl.name, "worker")
        if len(calls) == 1:
            wl.task.write_text("Assignment published during round")
            store.record("overview", "Plan", "New main overview", "main", "main")
        else:
            assert len(calls) == 2
            assert "Host wake reason: task, overview." in prompt
        return 0

    monkeypatch.setattr(loop, "run_round", run)
    monkeypatch.setattr(loop.time, "sleep", lambda seconds: wl.stop.touch())
    with _restore_sigterm(), _env(HYPROVER_MAX_ROUNDS="0"):
        assert loop.main(str(wl.dir)) == 0
    assert len(calls) == 2


def test_unreadable_inputs_wait_then_recover_without_losing_baseline(tmp: Path, monkeypatch, capsys):
    wl = _mk_worker(tmp)
    store = L.KnowledgeStore(wl.project_dir)
    contents = store.path.read_text()
    calls, sleeps = [], []

    def run(*args):
        calls.append(1)
        if len(calls) == 1:
            store.path.write_text("broken")
        return 0

    def poll(seconds):
        sleeps.append(seconds)
        if len(sleeps) < 3:
            assert json.loads(wl.status.read_text())["input_error"]
            assert len(calls) == 1
        if len(sleeps) == 2:
            store.path.write_text(contents)
            wl.task.write_text("Task arrived while inputs were unavailable")
        elif len(sleeps) == 3:
            wl.stop.touch()

    monkeypatch.setattr(loop, "run_round", run)
    monkeypatch.setattr(loop.time, "sleep", poll)
    with _restore_sigterm(), _env(HYPROVER_MAX_ROUNDS="0"):
        assert loop.main(str(wl.dir)) == 0
    assert len(calls) == 2
    stderr = capsys.readouterr().err
    assert stderr.count("waiting for readable inputs") == 1
    assert stderr.count("readable again") == 1
    assert json.loads(wl.status.read_text())["input_error"] is None


def test_stop_has_priority_even_with_corrupt_waiting_inputs(tmp: Path, monkeypatch):
    wl = _mk_worker(tmp)
    calls = []

    def run(*args):
        calls.append(1)
        (wl.project_dir / "KNOWLEDGE.md").write_text("broken")
        return 0

    monkeypatch.setattr(loop, "run_round", run)
    monkeypatch.setattr(loop.time, "sleep", lambda seconds: wl.stop.touch())
    with _restore_sigterm(), _env(HYPROVER_MAX_ROUNDS="0"):
        assert loop.main(str(wl.dir)) == 0
    assert calls == [1] and json.loads(wl.status.read_text())["state"] == "stopped"


def test_deadline_while_waiting_does_not_launch_again(tmp: Path, monkeypatch):
    wl = _mk_worker(tmp)
    calls = []
    monkeypatch.setattr(loop, "run_round", lambda *args: calls.append(1) or 0)
    monkeypatch.setattr(loop.time, "sleep", lambda seconds: (wl.project_dir / L.DEADLINE_FILE).write_text("1"))
    with _restore_sigterm(), _env(HYPROVER_MAX_ROUNDS="0"):
        assert loop.main(str(wl.dir)) == 0
    assert calls == [1] and json.loads(wl.status.read_text())["state"] == "deadline"


def test_stop_arriving_during_fingerprint_read_prevents_launch(tmp: Path, monkeypatch):
    wl = _mk_worker(tmp)
    original = loop.InputWatcher.read
    calls = []

    def read(watcher):
        value = original(watcher)
        (wl.project_dir / "stop-request.json").write_text("{}")
        return value

    monkeypatch.setattr(loop.InputWatcher, "read", read)
    monkeypatch.setattr(loop, "run_round", lambda *args: calls.append(1) or 0)
    with _restore_sigterm():
        assert loop.main(str(wl.dir)) == 0
    assert not calls and json.loads(wl.status.read_text())["state"] == "stopped"


# --- kickoff prompt -------------------------------------------------------- #

def test_kickoff_mentions_worker_and_project():
    p = loop.kickoff("ProjX", "wkrY")
    assert "wkrY" in p and "ProjX" in p and "TASK.md" in p
    assert (
        "If progress stalls, try to consider a useful intermediate claim or reformulation, "
        "and try to formulate a precise, testable conjecture that could help advance the proof."
    ) in p
    assert "Formed ideas may be shared as conjectures when useful" in p
    assert "Complete proofs may go directly to fact_submit" in p


# --- __main__ entry -------------------------------------------------------- #

def test_dunder_main_dispatches(tmp: Path):
    """runpy the package as __main__ with the loop entry patched: the guard runs
    and dispatches to main() without spawning anything."""
    seen = {}

    def _fake_main(arg):
        seen["arg"] = arg
        return 0

    orig = loop.main
    loop.main = _fake_main
    argv = sys.argv
    sys.argv = ["prog", "/some/worker/dir"]
    try:
        try:
            runpy.run_module("hyprover.execution", run_name="__main__")
            assert False, "should sys.exit"
        except SystemExit as e:
            assert e.code == 0
    finally:
        loop.main = orig
        sys.argv = argv
    assert seen["arg"] == "/some/worker/dir"


def test_dunder_main_usage_guard():
    """Wrong argc → usage message + exit 2 (no dispatch)."""
    argv = sys.argv
    sys.argv = ["prog"]                        # missing worker_dir
    try:
        try:
            runpy.run_module("hyprover.execution", run_name="__main__")
            assert False, "should sys.exit(2)"
        except SystemExit as e:
            assert e.code == 2
    finally:
        sys.argv = argv


# --- layout defaults (no env overrides) ------------------------------------ #

def test_layout_defaults_and_empties(tmp: Path):
    with _env(HYPROVER_WORKER_CONTRACT=None, HYPROVER_AGENTS_ROOT=None):
        # repo_root / worker_md defaults
        rr = L.repo_root()
        assert L.worker_md() == rr / "agents" / "contracts" / "worker.md"
        # agents_root default = <cwd>/runtime/projects
        assert L.agents_root() == (Path.cwd() / "runtime" / "projects").resolve()
    # list_workers / list_projects on a nonexistent root → []
    with _env(HYPROVER_AGENTS_ROOT=str(tmp / "no_such_root")):
        assert L.list_workers("ghost") == []
        assert L.list_projects() == []


# --- scaffold.symlink branches --------------------------------------------- #

def test_symlink_skips_existing(tmp: Path):
    target = tmp / "target"
    target.write_text("x")
    link = tmp / "link"
    link.write_text("already here")            # link path exists → early return
    scaffold.symlink(target, link)
    assert link.read_text() == "already here"  # untouched


def test_symlink_swallows_oserror(tmp: Path):
    target = tmp / "target"
    target.write_text("x")
    # a link path whose parent does not exist → os.symlink raises OSError, swallowed
    link = tmp / "no_parent_dir" / "link"
    scaffold.symlink(target, link)             # must not raise
    assert not link.exists()


# --- runner ---------------------------------------------------------------- #

# run_round monkeypatch helpers (so the standalone runner works without pytest's
# monkeypatch fixture): swap loop.run_round for the duration of a test.
_ORIG_RUN_ROUND = loop.run_round


def _patch_run_round(fn):
    loop.run_round = fn


def _unpatch_run_round():
    loop.run_round = _ORIG_RUN_ROUND


_NO_TMP = {test_kickoff_mentions_worker_and_project, test_dunder_main_usage_guard}


def main() -> None:
    tests = [
        test_run_round_returns_codex_rc,
        test_run_round_success_rc0,
        test_run_round_hard_timeout_terminates,
        test_run_round_missing_binary_returns_127,
        test_run_round_timeout_then_kill,
        test_main_stops_on_stop_flag,
        test_main_stops_on_deadline,
        test_main_max_rounds_cap,
        test_main_consecutive_failure_cap,
        test_main_timeout_rc124_does_not_count_as_failure,
        test_main_codex_missing_127,
        test_main_missing_worker_dir,
        test_main_sigterm_handler,
        test_write_status_corrupt_existing_recovers,
        test_parse_last_fact_id_missing_file,
        test_cleanup_pid_removes_own,
        test_cleanup_pid_keeps_foreign,
        test_cleanup_pid_swallows_oserror,
        test_main_success_honors_stop_before_sleep,
        test_kickoff_mentions_worker_and_project,
        test_dunder_main_dispatches,
        test_dunder_main_usage_guard,
        test_layout_defaults_and_empties,
        test_symlink_skips_existing,
        test_symlink_swallows_oserror,
    ]
    for t in tests:
        if t in _NO_TMP:
            t()
        else:
            with tempfile.TemporaryDirectory() as d:
                t(Path(d))
        print(f"  [ok] {t.__name__}")
    print("ALL LOOP TESTS PASSED")


if __name__ == "__main__":
    main()
