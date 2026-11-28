"""Project lifecycle commands and worker process-control helpers.

``list``, ``new``, ``assign``, ``start``, ``status`` and ``stop`` operate through
execution.layout and execution.scaffold. ``main`` delegates persistent coordination
to the controller. ``article`` provides local manuscript commands. These commands
manage assignments, processes and manuscripts, not fact publication.

PID checks are best-effort observations, not authentication or durable ownership.
Graceful worker stops set a flag; force stops signal a process group. Waiting is a
live worker state and must not be confused with a crash or a completed proof.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from hyprover.execution import layout as L
from hyprover.execution.scaffold import atomic_write, do_new, spawn_loop

__all__ = [
    "do_new", "do_assign", "do_start", "do_status", "worker_status",
    "do_list", "do_stop", "build_parser", "main",
]


# --------------------------------------------------------------------------- #
# read helpers                                                                 #
# --------------------------------------------------------------------------- #

def _read_pid(wl: L.WorkerLayout) -> Optional[int]:
    pf = wl.pid
    if not pf.exists():
        return None
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def _process_state(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat.rsplit(")", 1)[1].split()[0]


def _alive(pid: Optional[int]) -> bool:
    """Treat missing processes and zombies as dead; uncertain access stays live.

    Permission denial and unreadable/malformed proc state are not proof of exit.
    This check does not establish process ownership or protect against PID reuse.
    """
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return _process_state(pid) != "Z"
    except (OSError, IndexError):
        return True


def _read_json_or_empty(path: Path):
    """Read UTF-8 JSON, falling back to {} only for absence, read errors or bad JSON.

    Existence-check errors and UTF-8 decoding errors propagate. Valid non-object
    JSON is returned unchanged; callers retain responsibility for its shape.
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_status(wl: L.WorkerLayout) -> Dict:
    return _read_json_or_empty(wl.status)


# --------------------------------------------------------------------------- #
# assign                                                                       #
# --------------------------------------------------------------------------- #

def do_assign(target: str, task: str) -> Dict:
    """Overwrite (replace, NOT append) a worker's TASK.md, ensuring a trailing
    newline. Rejects a bare project, a nonexistent worker, and an empty task."""
    project, worker = L.resolve_target(target)
    if not worker:
        raise SystemExit("assign needs a specific worker: <project>/<worker>")
    wl = L.WorkerLayout(L.worker_dir(project, worker))
    if not wl.dir.is_dir():
        raise SystemExit(f"no such worker: {project}/{worker}")
    if not task.strip():
        raise SystemExit("refusing to assign an empty task")
    L.require_knowledge(wl.project_dir)
    atomic_write(wl.task, task if task.endswith("\n") else task + "\n")
    return {"worker": f"{project}/{worker}", "task_file": str(wl.task)}


# --------------------------------------------------------------------------- #
# start                                                                        #
# --------------------------------------------------------------------------- #

def _start_one(wl: L.WorkerLayout, *, before_start: Callable[[], bool] | None = None) -> str:
    """Start under the worker lock after project and launch-policy checks.

    Returns started, already-running, locked, or cancelled by before_start.
    A previous regular stop file is archived before spawn; unexpected stop-file
    types are rejected. Successful spawn records the returned PID atomically.
    """
    L.require_knowledge(wl.project_dir)
    L.require_launch_allowed(wl.project_dir)
    if not wl.dir.is_dir():
        raise SystemExit(f"no such worker: {wl.project}/{wl.name}")
    wl.logs.mkdir(exist_ok=True)
    lock = open(wl.lock, "w")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "locked"
        if _alive(_read_pid(wl)):
            return "already-running"
        if before_start is not None and not before_start():
            return "cancelled"
        if wl.stop.exists() or wl.stop.is_symlink():
            if wl.stop.is_symlink() or not wl.stop.is_file():
                raise ValueError("unexpected_worker_stop_file")
            wl.stop.replace(wl.logs / f"stop-before-start-{time.time_ns()}.flag")
        pid = spawn_loop(wl.dir)
        atomic_write(wl.pid, str(pid))
        return "started"
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _require_worker_dirs(target: str) -> List[Path]:
    directories = L.target_worker_dirs(target)
    if not directories:
        raise SystemExit(f"no workers for target {target!r}")
    return directories


def do_start(target: str, stagger: float = 0.2) -> List[Dict]:
    dirs = _require_worker_dirs(target)
    out = []
    for i, wdir in enumerate(dirs):
        if i and stagger:
            time.sleep(stagger)
        out.append({"worker": wdir.name, "result": _start_one(L.WorkerLayout(wdir))})
    return out


# --------------------------------------------------------------------------- #
# status                                                                       #
# --------------------------------------------------------------------------- #

def worker_status(wl: L.WorkerLayout) -> Dict:
    pid = _read_pid(wl)
    alive = _alive(pid)
    st = _read_status(wl)
    state = st.get("state", "—")
    now = time.time()
    last = st.get("last_round_at") or st.get("round_started_at") or st.get("updated_at")
    age = (now - last) if isinstance(last, (int, float)) else None

    if alive:
        # a round legitimately runs for hours; only flag truly stale running rounds
        rs = st.get("round_started_at")
        hard = int(os.environ.get("HYPROVER_ROUND_HARD_TIMEOUT", "14400"))
        if state == "waiting":
            label = "waiting"
        elif hard > 0 and state == "running" and isinstance(rs, (int, float)) and (now - rs) > hard * 1.5:
            label = "stuck?"
        else:
            label = "working"
    else:
        label = state if state in ("stopped", "deadline", "max_rounds", "error",
                                   "terminated", "created") else "dead"
    return {
        "worker": wl.name, "pid": pid, "alive": alive, "state": state,
        "round": st.get("round", 0), "age_s": round(age, 1) if age is not None else None,
        "last_fact_id": st.get("last_fact_id"), "label": label,
        "waiting_since": st.get("waiting_since"), "wake_reason": st.get("wake_reason"),
        "input_fingerprint": st.get("input_fingerprint"), "input_error": st.get("input_error"),
    }


def do_status(target: str) -> List[Dict]:
    dirs = _require_worker_dirs(target)
    return [worker_status(L.WorkerLayout(d)) for d in dirs]


# --------------------------------------------------------------------------- #
# list                                                                         #
# --------------------------------------------------------------------------- #

def _read_project_metadata(project: str):
    return _read_json_or_empty(L.project_dir(project) / "project.json")


def do_list() -> List[Dict]:
    """One row per project: roster + how many workers are live + model."""
    out: List[Dict] = []
    for project in L.list_projects():
        meta = _read_project_metadata(project)
        workers = L.list_workers(project)
        live = sum(1 for w in workers
                   if _alive(_read_pid(L.WorkerLayout(L.worker_dir(project, w)))))
        out.append({"project": project, "workers": len(workers), "live": live,
                    "model": meta.get("model", "—")})
    return out


def _fmt_list(rows: List[Dict]) -> str:
    head = f"{'PROJECT':<24}{'WORKERS':>8}{'LIVE':>6}  {'MODEL':<12}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['project']:<24}{r['workers']:>8}{r['live']:>6}  {str(r['model']):<12}")
    return "\n".join(lines) if rows else "(no projects under the agents root)"


def _fmt_status(rows: List[Dict]) -> str:
    head = f"{'WORKER':<14}{'LABEL':<12}{'STATE':<13}{'ROUND':>6}  {'AGE':>7}  {'LAST_FACT':<16}"
    lines = [head, "-" * len(head)]
    for r in rows:
        age = f"{r['age_s']:.0f}s" if r["age_s"] is not None else "—"
        lines.append(f"{r['worker']:<14}{r['label']:<12}{r['state']:<13}"
                     f"{r['round']:>6}  {age:>7}  {str(r['last_fact_id'] or '—'):<16}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# stop                                                                         #
# --------------------------------------------------------------------------- #

def _signal_process_group(pid: int, signum: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signum)
    except (ProcessLookupError, PermissionError):
        pass


def _stop_one(wl: L.WorkerLayout, force: bool) -> str:
    """Request a round-boundary stop, or send TERM then KILL with bounded polling.

    The force path removes the PID file after its attempts; the legacy 'killed'
    result is not a post-KILL exit confirmation. Other filesystem errors propagate.
    """
    pid = _read_pid(wl)
    if not force:
        if not _alive(pid):
            return "not-running"
        wl.stop.touch()      # graceful: loop exits at round boundary
        return "stopping (graceful)"
    # force: kill the loop's process group (loop + its codex child), then SIGKILL
    if not _alive(pid):
        wl.pid.unlink(missing_ok=True)
        return "not-running"
    _signal_process_group(pid, signal.SIGTERM)
    for _ in range(50):                          # up to ~5s for a clean exit
        if not _alive(pid):
            break
        time.sleep(0.1)
    if _alive(pid):
        _signal_process_group(pid, signal.SIGKILL)
    wl.pid.unlink(missing_ok=True)
    return "killed"


def do_stop(target: str, force: bool = False) -> List[Dict]:
    dirs = _require_worker_dirs(target)
    return [{"worker": d.name, "result": _stop_one(L.WorkerLayout(d), force)} for d in dirs]


# --------------------------------------------------------------------------- #
# argparse                                                                      #
# --------------------------------------------------------------------------- #

def _task_from_args(args) -> str:
    """Choose task (even empty), then file, then stdin; do not trim the selected text."""
    import sys
    if args.task is not None:
        return args.task
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.stdin:
        return sys.stdin.read()
    raise SystemExit("assign needs one of --task, --file, or --stdin")


def build_parser() -> argparse.ArgumentParser:
    from hyprover.article.__main__ import parser as article_parser

    p = argparse.ArgumentParser(prog="hyprover", description="Control codex workers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    li = sub.add_parser("list", help="list all projects + live worker counts")
    li.add_argument("--json", action="store_true")

    n = sub.add_parser("new", help="scaffold a project + worker dirs")
    n.add_argument("project")
    n.add_argument("--roles", default=L.DEFAULT_ROLES, help=f"role:count list (default: {L.DEFAULT_ROLES})")
    n.add_argument("--model", default=None)

    a = sub.add_parser("assign", help="write a worker's per-round TASK.md")
    a.add_argument("target", help="<project>/<worker>")
    a.add_argument("--task", default=None)
    a.add_argument("--file", default=None)
    a.add_argument("--stdin", action="store_true")

    s = sub.add_parser("start", help="launch worker loop(s)")
    s.add_argument("target", help="<project> or <project>/<worker>")

    st = sub.add_parser("status", help="liveness + progress")
    st.add_argument("target", help="<project> or <project>/<worker>")
    st.add_argument("--json", action="store_true")

    sp = sub.add_parser("stop", help="stop worker loop(s)")
    sp.add_argument("target", help="<project> or <project>/<worker>")
    sp.add_argument("--force", action="store_true", help="kill now (else finish current round)")
    coordinator = sub.add_parser("main", help="persistent autonomous project coordinator")
    coordinator.add_argument("action", choices=("start", "stop", "status"))
    coordinator.add_argument("project")
    article_parser(sub.add_parser("article", help="generate, inspect and compile local manuscripts"))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "main":
        from . import controller
        return controller.main([args.action, args.project])
    if args.cmd == "article":
        from hyprover.article.__main__ import run
        return run(args)
    if args.cmd == "list":
        rows = do_list()
        print(json.dumps(rows, ensure_ascii=False, indent=2) if args.json else _fmt_list(rows))
    elif args.cmd == "new":
        r = do_new(args.project, roles=args.roles, model=args.model)
        print(f"created {args.project} with {len(r['workers'])} workers: "
              f"{', '.join(r['workers'])}\n  {r['project_dir']}")
    elif args.cmd == "assign":
        r = do_assign(args.target, _task_from_args(args))
        print(f"assigned {r['worker']} -> {r['task_file']}")
    elif args.cmd == "start":
        for r in do_start(args.target):
            print(f"{r['worker']}: {r['result']}")
    elif args.cmd == "status":
        rows = do_status(args.target)
        print(json.dumps(rows, ensure_ascii=False, indent=2) if args.json else _fmt_status(rows))
    elif args.cmd == "stop":
        for r in do_stop(args.target, force=args.force):
            print(f"{r['worker']}: {r['result']}")
    return 0
