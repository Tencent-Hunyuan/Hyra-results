"""Persistent serial research queue with per-project deadlines and isolated reviewers."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from shutil import disk_usage

from hyprover.execution import layout as L
from . import controller as c

TAG_ENV = "HYPROVER_SEQUENCE_STAGE"
ENV_KEYS = {
    "HYPROVER_AGENTS_ROOT", "HYPROVER_MAIN_MODEL", "HYPROVER_MAIN_EFFORT",
    "HYPROVER_WORKER_MODEL", "HYPROVER_WORKER_EFFORT", "HYPROVER_VERIFY_MODEL",
    "HYPROVER_VERIFY_EFFORT",
    "HYPROVER_LITERATURE_SEARCH", "HYPROVER_MAIN_INTERVAL", "HYPROVER_MAX_ROUNDS",
    "HYPROVER_ROUND_HARD_TIMEOUT", "HYPROVER_MAX_CONSEC_FAILURES", "HYPROVER_ROUND_BEAT",
    "HYPROVER_RETRY_INTERVAL", "HYPROVER_MAIN_MAX_CONSEC_FAILURES",
    "HYPROVER_VERIFY_URL", "HYPROVER_VERIFY_TIMEOUT", "HYPROVER_MCP_TOOL_TIMEOUT",
    "VERIFY_HOST", "VERIFY_PORT", "VERIFY_AGENT_HOME", "VERIFIER_RESULTS_DIR",
    "CODEX_TIMEOUT_SECONDS", "HYPROVER_CODEX_BIN",
}
TERMINAL = {"completed", "stopped", "blocked", "error"}


def members(tag: str) -> dict[int, dict]:
    """Find tagged processes and descendants, including detached/orphaned sessions."""
    marker = f"{TAG_ENV}={tag}".encode()
    rows, found = {}, {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        pid = int(proc.name)
        try:
            owner = c.identity(pid)
            if owner is None:
                continue
            parent = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[1])
            rows[pid] = (parent, owner)
            if marker in (proc / "environ").read_bytes().split(b"\0"):
                found[pid] = owner
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    while True:
        more = {pid: owner for pid, (parent, owner) in rows.items()
                if parent in found and pid not in found}
        if not more:
            return found
        found.update(more)


def send(owner: dict, sig: int) -> None:
    pid = owner["pid"]
    if pid == os.getpid():
        raise ValueError("refusing_to_signal_supervisor")
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        if c.identity(pid) == owner:
            signal.pidfd_send_signal(fd, sig)
    except ProcessLookupError:
        pass
    finally:
        os.close(fd)


class Sequence:
    def __init__(self, config: Path):
        self.config = config.resolve(strict=True)
        self.home = self.config.parent
        self.spec = c._read(self.config)
        self.root = Path(self.spec["agents_root"])
        if not self.root.is_absolute() or self.root.resolve() != self.root or not self.root.is_dir():
            raise ValueError("invalid_agents_root")
        self.stages = self.spec.get("stages")
        if not isinstance(self.stages, list) or not self.stages:
            raise ValueError("nonempty_stages_required")
        minimum = self.spec.get("min_free_bytes", 5 * 1024**3)
        if type(minimum) is not int or minimum < 0:
            raise ValueError("invalid_min_free_bytes")
        self.minimum = minimum
        self.projects, self.environments, names, ports = [], [], set(), set()
        self.completion_permissions = []
        digest = hashlib.sha256(self.config.read_bytes())
        for stage in self.stages:
            name = stage["project"]
            duration, count = stage["time_limit_seconds"], stage["worker_count"]
            if not isinstance(name, str) or not c._NAME.fullmatch(name) or name in names:
                raise ValueError("invalid_or_duplicate_project")
            if (type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0
                    or type(count) is not int or not 1 <= count <= 10):
                raise ValueError("invalid_stage_limits")
            names.add(name)
            p = self.root / name
            if not p.is_dir() or p.is_symlink():
                raise ValueError("invalid_project_directory")
            L.require_knowledge(p)
            metadata = c._read(p / "project.json")
            self.completion_permissions.append(c.early_completion_allowed(p))
            roster = metadata.get("workers", [])
            actual = sorted(w.name for w in (p / "workers").iterdir() if w.is_dir())
            if len(roster) != count or len(set(roster)) != count or sorted(roster) != actual:
                raise ValueError("worker_roster_mismatch")
            env = c._read(p / "run-settings.json")
            if not set(env) <= ENV_KEYS or any(not isinstance(v, str) for v in env.values()):
                raise ValueError("invalid_stage_environment")
            port = int(env["VERIFY_PORT"])
            if not 1024 <= port <= 65535 or port in ports:
                raise ValueError("invalid_or_duplicate_port")
            ports.add(port)
            expected = {"HYPROVER_AGENTS_ROOT": str(self.root), "VERIFY_HOST": "127.0.0.1",
                        "HYPROVER_VERIFY_URL": f"http://127.0.0.1:{port}/verify",
                        "VERIFY_AGENT_HOME": str(p / "verifier"),
                        "VERIFIER_RESULTS_DIR": str(p / "verify-runs")}
            if any(env.get(k) != v for k, v in expected.items()):
                raise ValueError("verifier_isolation_required")
            for filename in ("PROBLEM.md", "RUN_POLICY.md", "project.json", "run-settings.json"):
                digest.update(str(p / filename).encode())
                digest.update((p / filename).read_bytes())
            self.projects.append(p)
            self.environments.append(env)
        self.digest = digest.hexdigest()
        self.state = c._read(self.home / "status.json") or {
            "state": "pending", "config_sha256": self.digest, "index": 0,
            "stages": [{"project": p.name, "state": "pending", "deadline": None,
                        "tag": str(self.config) + ":" + p.name} for p in self.projects]}
        if self.state.get("config_sha256") != self.digest:
            raise ValueError("sequence_inputs_changed")
        if ([r.get("project") for r in self.state.get("stages", [])] != [p.name for p in self.projects]
                or type(self.state.get("index")) is not int
                or not 0 <= self.state["index"] <= len(self.projects)):
            raise ValueError("invalid_sequence_state")
        self.interrupted = threading.Event()
        self.children: list[subprocess.Popen] = []
        self.monotonic_deadlines: dict[int, float] = {}

    def save(self, **fields) -> None:
        self.state.update(fields, updated_at=time.time())
        c._write(self.home / "status.json", self.state)

    def stopped(self) -> bool:
        return self.interrupted.is_set() or (self.home / ".stop").exists()

    def expired(self, index: int) -> bool:
        deadline = self.state["stages"][index].get("deadline")
        return deadline is not None and (time.time() >= deadline or
               time.monotonic() >= self.monotonic_deadlines.get(index, float("inf")))

    def reason(self, index: int) -> str | None:
        if self.stopped():
            return "stopped"
        p = self.projects[index]
        if (p / "stop-request.json").exists() or (p / "main/.stop").exists():
            return "stopped"
        if disk_usage(self.root).free < self.minimum:
            return "disk_low"
        if self.expired(index):
            return "deadline"
        return None

    def gate(self, index: int, state: str) -> None:
        record = self.state["stages"][index]
        c._write(self.projects[index] / ".sequence.json",
                 {"config": str(self.config), "tag": record["tag"], "state": state})

    def claim(self) -> list:
        locks, originals, written = [], {}, []
        try:
            for index, p in enumerate(self.projects):
                lock = (p / ".sequence.lock").open("a")
                locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                gate_path = p / ".sequence.json"
                originals[index] = gate_path.read_text() if gate_path.exists() else None
                gate = c._read(gate_path)
                if gate and gate.get("config") != str(self.config):
                    raise ValueError("project_owned_by_another_sequence")
                if self.state["stages"][index]["state"] == "pending":
                    for rel in ("main/status.json", "main/result.json", "main/.stop",
                                ".run_deadline", "stop-request.json"):
                        if (p / rel).exists():
                            raise ValueError("new_stage_requires_unused_project")
                    if any((p / "workers").glob("*/.pid")):
                        raise ValueError("new_stage_has_worker_processes")
            for index in range(len(self.projects)):
                written.append(index)
                self.gate(index, self.state["stages"][index]["state"])
            return locks
        except BaseException:
            try:
                for index in written:
                    path = self.projects[index] / ".sequence.json"
                    if originals[index] is None:
                        path.unlink(missing_ok=True)
                    else:
                        c.atomic_write_text(path, originals[index])
            finally:
                for lock in locks:
                    lock.close()
            raise

    def check_resume(self) -> int:
        index = self.state["index"]
        if self.state["state"] != "blocked" or index >= len(self.projects):
            raise ValueError("only_blocked_stage_can_resume")
        p, record = self.projects[index], self.state["stages"][index]
        if record["state"] != "stopped" or record.get("outcome") not in ("blocked", "error", "disk_low"):
            raise ValueError("stage_not_resumable")
        if self.stopped():
            raise ValueError("sequence_stop_requested")
        deadline, started = record.get("deadline"), record.get("started_at")
        if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                or type(started) not in (int, float) or not math.isfinite(started)
                or deadline != started + self.stages[index]["time_limit_seconds"]
                or float((p / ".run_deadline").read_text()) != deadline):
            raise ValueError("resume_deadline_mismatch")
        if time.time() >= deadline:
            raise ValueError("project_deadline_passed")
        if disk_usage(self.root).free < self.minimum:
            raise ValueError("disk_low")
        gate = c._read(p / ".sequence.json")
        if gate != {"config": str(self.config), "tag": record["tag"], "state": "stopped"}:
            raise ValueError("resume_gate_mismatch")
        if c._read(p / "stop-request.json").get("reason") != record["outcome"]:
            raise ValueError("resume_stop_reason_mismatch")
        main = c._read(p / "main/status.json")
        if main.get("state") == "completed" or (p / "main/result.json").exists():
            raise ValueError("completed_project_cannot_resume")
        owners = [record.get("main_identity"), record.get("verifier_identity"),
                  main.get("identity"), main.get("child_identity")]
        if members(record["tag"]) or any(owner and c.identity(owner["pid"]) == owner for owner in owners):
            raise ValueError("resume_requires_stopped_processes")
        if any((p / "workers").glob("*/.pid")):
            raise ValueError("resume_requires_clean_worker_pids")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", int(self.environments[index]["VERIFY_PORT"])))
        return index

    def prepare_resume(self) -> None:
        index = self.check_resume()
        p, record = self.projects[index], self.state["stages"][index]
        archive = self.home / "recoveries" / str(time.time_ns())
        archive.mkdir(parents=True)
        c._write(archive / "status.json", self.state)
        c._write(archive / "main-status.json", c._read(p / "main/status.json"))
        previous, moved = json.loads(json.dumps(self.state)), []
        try:
            for relative in ("stop-request.json", "main/.stop", "sequence-result.json"):
                source = p / relative
                if source.exists():
                    target = archive / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source.replace(target)
                    moved.append((source, target))
            for key in ("stop_intent", "outcome", "stopped_at", "main_identity", "verifier_identity"):
                record.pop(key, None)
            record.update(state="resuming", resumed_at=time.time())
            self.state.setdefault("recoveries", []).append({"index": index, "archive": str(archive),
                                                           "deadline": record["deadline"]})
            self.save(state="resuming", identity=None, error=None)
        except BaseException:
            self.state = previous
            for source, target in reversed(moved):
                target.replace(source)
            if c._read(self.home / "status.json") != previous:
                c._write(self.home / "status.json", previous)
            raise

    def spawn(self, command: list[str], log: Path, env: dict) -> subprocess.Popen:
        with log.open("a") as output:
            proc = subprocess.Popen(command, env=env, cwd=self.root,
                                    stdin=subprocess.DEVNULL, stdout=output,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(proc)
        return proc

    def activate(self, index: int) -> None:
        if Sequence(self.config).digest != self.digest:
            raise ValueError("sequence_inputs_changed")
        p, record = self.projects[index], self.state["stages"][index]
        if self.reason(index):
            raise InterruptedError(self.reason(index))
        c.prepare(p)
        env = os.environ.copy()
        env.update(self.environments[index])
        env[TAG_ENV] = record["tag"]
        port = int(env["VERIFY_PORT"])
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        start = time.time()
        if record["state"] == "resuming":
            duration = record["deadline"] - start
            if duration <= 0:
                raise InterruptedError("deadline")
            record.update(state="starting", resumed_at=start)
        else:
            duration = self.stages[index]["time_limit_seconds"]
            record.update(state="starting", started_at=start, deadline=start + duration)
        self.monotonic_deadlines[index] = time.monotonic() + duration
        env["HYPROVER_RUN_DEADLINE"] = str(record["deadline"])
        self.save()
        c.atomic_write_text(p / ".run_deadline", str(record["deadline"]) + "\n")
        self.gate(index, "active")
        verifier = self.spawn([sys.executable, "-m", "hyprover.verify"],
                              self.home / f"stage-{index}-verifier.log", env)
        record["verifier_identity"] = c.identity(verifier.pid)
        self.save()
        until = time.monotonic() + 30
        while True:
            reason = self.reason(index)
            if reason:
                raise InterruptedError(reason)
            if verifier.poll() is not None or time.monotonic() >= until:
                raise RuntimeError("verifier_startup_failed")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    healthy = json.load(response)
                if healthy.get("pid") == verifier.pid and healthy.get("status") == "ok":
                    break
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        c.prepare(p)
        main = self.spawn([sys.executable, "-m", "hyprover.orchestration.controller", "run", p.name],
                          self.home / f"stage-{index}-main.log", env)
        record.update(state="running", main_identity=c.identity(main.pid), main_launched_at=time.time())
        self.save()

    def completion(self, index: int) -> bool:
        p = self.projects[index]
        if not self.completion_permissions[index] or not c.early_completion_allowed(p):
            return False
        result = c._read(p / "main/result.json")
        state = c._read(p / "main/status.json")
        if result.get("state") != "completed" or state.get("state") != "completed":
            return False
        c.confirm_completion(p, result, results_root=p / "verify-runs")
        return True

    def finish(self, index: int, outcome: str) -> None:
        p, record = self.projects[index], self.state["stages"][index]
        record.update(state="stopping", stop_intent=outcome)
        write_error = None
        try:
            self.save()
            self.gate(index, "stopping")
            c._write(p / "stop-request.json", {"reason": outcome, "requested_at": time.time()})
            (p / "main").mkdir(exist_ok=True)
            (p / "main/.stop").touch()
        except OSError as exc:
            write_error = exc
        until, force = time.monotonic() + 20, time.monotonic() + 5
        sent = set()
        known = {owner["pid"]: owner for key in ("main_identity", "verifier_identity")
                 if (owner := record.get(key))}
        empty_polls = 0
        while True:
            known.update(members(record["tag"]))
            alive = {pid: owner for pid, owner in known.items() if c.identity(pid) == owner}
            if not alive:
                empty_polls += 1
                if empty_polls >= 2:
                    break
            else:
                empty_polls = 0
            sig = signal.SIGKILL if time.monotonic() >= force else signal.SIGTERM
            for owner in alive.values():
                key = (owner["pid"], owner["start_ticks"], sig)
                if key not in sent:
                    send(owner, sig)
                    sent.add(key)
            if time.monotonic() >= until:
                raise RuntimeError("stage_cleanup_incomplete")
            for child in self.children:
                child.poll()
            time.sleep(0.1)
        if write_error is not None:
            raise write_error
        for wdir in (p / "workers").iterdir():
            if wdir.is_dir():
                wl = L.WorkerLayout(wdir)
                state = c._read(wl.status)
                state.update(state="deadline" if outcome == "deadline" else "stopped",
                             pid=None, updated_at=time.time())
                c._write(wl.status, state)
                try:
                    pid_stat = wl.pid.lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(pid_stat.st_mode):
                    raise ValueError("unexpected_worker_pid_file")
                try:
                    pid_text = wl.pid.read_text().strip()
                    if not pid_text.isdecimal() or c.identity(int(pid_text)) is not None:
                        raise ValueError("worker_pid_not_confirmed_stopped")
                    wl.pid.unlink()
                except FileNotFoundError:
                    pass
        self.gate(index, "stopped")
        record.update(state="stopped", outcome=outcome, stopped_at=time.time())
        c._write(p / "sequence-result.json", record)
        self.save()

    def run(self, lock_fd: int | None = None, *, resume: bool = False) -> int:
        lock = os.fdopen(lock_fd, "a") if lock_fd is not None else (self.home / ".lock").open("a")
        locks, handlers = [], {}
        active = None
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0
            if resume:
                try:
                    self.check_resume()
                except (ValueError, OSError):
                    return 2
            elif self.state["state"] in TERMINAL:
                return 0
            locks = self.claim()
            if resume:
                self.prepare_resume()
                active = self.state["index"]
            for sig in (signal.SIGTERM, signal.SIGINT):
                handlers[sig] = signal.signal(sig, lambda *_: self.interrupted.set())
            self.save(state="running", identity=c.identity(os.getpid()), error=None)
            while self.state["index"] < len(self.stages):
                index = self.state["index"]
                active = index
                record = self.state["stages"][index]
                if record["state"] == "stopped":
                    if record.get("outcome") not in ("deadline", "completed"):
                        self.save(state="blocked", identity=None, error="previous_stage_blocked")
                        return 2
                    self.save(index=index + 1)
                    active = None
                    continue
                if record["state"] != "pending" and not (resume and record["state"] == "resuming"):
                    self.finish(index, "interrupted")
                    self.save(state="blocked", identity=None, error="interrupted_stage_requires_operator")
                    return 2
                try:
                    self.activate(index)
                except InterruptedError as exc:
                    reason = str(exc)
                    if reason != "deadline":
                        raise
                else:
                    while True:
                        reason = self.reason(index)
                        if reason:
                            break
                        if self.completion(index):
                            reason = "completed"
                            break
                        if any(not record.get(key) or c.identity(record[key]["pid"]) != record[key]
                               for key in ("main_identity", "verifier_identity")):
                            reason = "blocked"
                            break
                        for child in self.children:
                            child.poll()
                        time.sleep(0.5)
                self.finish(index, reason)
                active = None
                if reason not in ("completed", "deadline"):
                    self.save(state="stopped" if reason == "stopped" else "blocked",
                              identity=None, error=reason)
                    return 2
                self.save(index=index + 1)
            self.save(state="completed", identity=None, completed_at=time.time())
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            outcome = str(exc) if isinstance(exc, InterruptedError) else "error"
            if active is not None:
                try:
                    self.finish(active, outcome)
                except Exception as cleanup:
                    self.save(state="error", error=f"{type(exc).__name__}; cleanup: {cleanup}", identity=None)
                    return 2
            self.save(state="stopped" if outcome == "stopped" else "blocked",
                      identity=None, error=f"{type(exc).__name__}: {exc}")
            return 2
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            for child in self.children:
                child.poll()
            for handle in locks:
                handle.close()
            lock.close()


def status(config: Path) -> dict:
    state = c._read(config.resolve().parent / "status.json")
    owner = state.get("identity")
    alive = bool(owner and c.identity(owner["pid"]) == owner)
    return dict(state, alive=alive, stop_requested=(config.resolve().parent / ".stop").exists())


def start(config: Path, *, resume: bool = False) -> dict:
    sequence = Sequence(config)
    with (sequence.home / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status(config), launch="already-running")
        if resume:
            sequence.check_resume()
        elif sequence.stopped() or sequence.state["state"] in TERMINAL:
            return dict(sequence.state, launch="not-started")
        else:
            sequence.save(state="starting", identity=None)
        with (sequence.home / "supervisor.log").open("a") as log:
            proc = subprocess.Popen([sys.executable, "-m", __spec__.name, "run", str(sequence.config),
                                     "--lock-fd", str(lock.fileno())] + (["--resume"] if resume else []),
                                    cwd=sequence.root, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    pass_fds=(lock.fileno(),))
        return {"launch": "started", "pid": proc.pid, "config": str(sequence.config)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resume", "run", "status", "stop"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--lock-fd", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.action != "run":
        parser.error("--resume is only valid with run; use the resume action")
    if args.action == "run":
        return Sequence(args.config).run(args.lock_fd, resume=args.resume)
    if args.action == "stop":
        (args.config.resolve(strict=True).parent / ".stop").touch()
    result = (start(args.config, resume=args.action == "resume")
              if args.action in ("start", "resume") else status(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
