"""Persistent project coordinator; research decisions belong to the main model."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from hyprover import codex
from hyprover.core import KnowledgeStore
from hyprover.core._util import atomic_write_text
from hyprover.execution import layout as L
from hyprover.verify import launcher, completion, completion_evidence

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def project_path(project: str) -> Path:
    if not _NAME.fullmatch(project):
        raise ValueError("invalid_project_name")
    path = L.agents_root() / project
    if path.is_symlink() or not path.is_dir():
        raise ValueError("invalid_project_directory")
    KnowledgeStore(path).snapshot()
    if not (path / "PROBLEM.md").is_file():
        raise ValueError("PROBLEM.md_required")
    return path


def identity(pid: int) -> dict | None:
    """Linux process identity; never trust a reusable PID alone."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19]}
    except (OSError, IndexError, ValueError):
        return None


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("controller_state_must_be_object")
    return value


def _write(path: Path, data: dict) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def status(project: str) -> dict:
    p = project_path(project)
    state = _read(p / "main/status.json")
    owner = state.get("identity")
    state["alive"] = bool(owner and identity(owner["pid"]) == owner)
    state["project"] = project
    if not state["alive"] and state.get("state") in ("running", "waiting", "starting", "reviewing", "retrying"):
        state["state"] = "interrupted"
    return state


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _terminate_group(pgid: int) -> None:
    if pgid == os.getpgrp():
        raise ValueError("refusing_to_signal_own_process_group")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    until = time.monotonic() + 5
    while _group_alive(pgid) and time.monotonic() < until:
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stop_workers(p: Path, *, completion_token: str | None = None, recovery: list | None = None) -> list[dict]:
    results = []
    for wdir in sorted((p / "workers").iterdir()):
        if not wdir.is_dir():
            continue
        wl = L.WorkerLayout(wdir)
        state = _read(wl.status)
        try:
            pid = int(wl.pid.read_text())
        except FileNotFoundError:
            pid = state.get("pid", 0)
        owner = identity(pid) if pid else None
        if owner:
            try:
                args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                if (b"hyprover.execution" not in args or str(wdir).encode() not in args
                        or os.getpgid(pid) != pid or identity(pid) != owner):
                    results.append({"worker": wdir.name, "stopped": False, "error": "process_identity_mismatch"})
                    continue
            except (ProcessLookupError, FileNotFoundError):
                owner = None
        if completion_token is None:
            wl.stop.touch()
        else:
            try:
                with wl.stop.open("x") as handle:
                    handle.write(completion_token)
                    handle.flush()
                    stat = os.fstat(handle.fileno())
                    if owner and recovery is not None:
                        recovery.append({"worker": wdir.name, "token": completion_token,
                                         "marker": [stat.st_ino, stat.st_mtime_ns, stat.st_size],
                                         "original_identity": owner})
                        _write(p / "main/completion-shutdown.json", {"entries": recovery})
            except FileExistsError:
                pass
        if owner and identity(pid) == owner:
            try:
                _terminate_group(pid)
            except ProcessLookupError:
                pass
        results.append({"worker": wdir.name, "stopped": identity(pid) is None if pid else True})
    return results


def resume_completion_workers(p: Path, entries: list, stopped) -> list[dict]:
    from .cli import _start_one
    results, pending = [], []
    for entry in entries:
        if stopped() or (p / "main/.stop").exists() or (p / "stop-request.json").exists():
            break
        deadline = L.run_deadline(p)
        if deadline is not None and time.time() >= deadline:
            break
        name = entry.get("worker", "")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            continue
        wl = L.WorkerLayout(p / "workers" / name)
        try:
            def may_restart():
                if (stopped() or (p / "main/.stop").exists() or (p / "stop-request.json").exists()
                        or not wl.stop.is_file() or wl.stop.is_symlink()):
                    return False
                current_deadline = L.run_deadline(p)
                if current_deadline is not None and time.time() >= current_deadline:
                    return False
                stat = wl.stop.stat()
                return ([stat.st_ino, stat.st_mtime_ns, stat.st_size] == entry.get("marker")
                        and wl.stop.read_text() == entry.get("token"))
            if not may_restart():
                continue
            L.require_launch_allowed(p)
            result = _start_one(wl, before_start=may_restart)
            results.append({"worker": name, "launch": result})
            if result in ("locked", "already-running"):
                pending.append(entry)
        except (Exception, SystemExit) as exc:
            results.append({"worker": name, "error": str(exc)})
            pending.append(entry)
    if entries:
        _write(p / "main/completion-shutdown.json", {"entries": pending})
    return results


def stop(project: str) -> dict:
    p = project_path(project)
    home = p / "main"
    home.mkdir(exist_ok=True)
    (home / ".stop").touch()
    result = status(project)
    if not result["alive"]:
        result["workers"] = stop_workers(p)
    result["stop_requested"] = True
    return result


def prepare(p: Path) -> Path:
    home = p / "main"
    home.mkdir(exist_ok=True)
    (home / "logs").mkdir(exist_ok=True)
    contract = L.repo_root() / "agents/contracts/main_agent.md"
    link = home / "AGENTS.md"
    if link.exists() and link.resolve() != contract.resolve():
        raise ValueError("unexpected_main_contract")
    if not link.exists():
        link.symlink_to(contract)
    return home


def early_completion_allowed(p: Path) -> bool:
    value = _read(p / "project.json").get("allow_early_completion", True)
    if type(value) is not bool:
        raise ValueError("allow_early_completion_must_be_boolean")
    return value


def build_command(p: Path, output: Path, feedback: str = "") -> list[str]:
    metadata = json.loads((p / "project.json").read_text())
    roster = metadata["workers"]
    proof_only = metadata.get("research_mode") == "proof_only"
    direction = (
        "PROOF-ONLY PROJECT: seek a positive proof of the original assertion. "
        "Do not assign counterexample searches for the target or standalone counterexample hunts for auxiliary claims. "
        "Redirect those tasks to constructive proofs and justified reductions. This supersedes historical route notes. "
        "Repeat the restriction in every worker and subagent assignment. "
        "Still check proofs rigorously, report gaps honestly, and never assume the target as a premise. "
        "Only a complete positive proof is an acceptable completion.\n"
        if proof_only else ""
    )
    cli = f"{sys.executable} -m hyprover.orchestration"
    prompt = (
        f"You are the autonomous MAIN research agent for project {p.name}.\n"
        f"Project directory: {p}. Worker slots: {', '.join(roster)}.\n"
        f"{direction}"
        f"{L.math_computation_policy(L.math_computation_allowed(metadata=metadata))}\n"
        f"Read {p / 'PROBLEM.md'}, any {p / 'RUN_POLICY.md'}, and {p / 'KNOWLEDGE.md'}. "
        "Do not read other projects or private worker histories. The original problem is fixed; "
        "no mathematical route or worker specialization is supplied by this controller. "
        "YOU choose independent investigations, record overview/routes via knowledge_record, assign and "
        "start workers, inspect their public progress and feedback, reallocate when useful, and judge closure. "
        "When a key link is unclear, try to consider useful intermediate claims or reformulations, "
        "and try to formulate a precise, testable conjecture that could help advance the proof. "
        "Compare alternatives if they arise naturally, while preserving a clear proof path. "
        "Every pass explicitly read shared reports with knowledge_search(query='', "
        "kinds=['report'], sort_by='updated', ...) and service feedback with kinds=['feedback']. "
        "Browse the needed knowledge categories with empty query and judge relevance yourself; "
        "there are no relevance scores. Use IDs/labels to revisit records and follow pagination. "
        "Nonempty text queries are literal filters, not semantic retrieval or an exhaustive survey. "
        "Default search excludes reports and archived entries, not feedback. "
        "Use knowledge_record(kind='report', ...) for substantive progress, handoffs, blockers, next steps "
        "and no-issue manual audits: Pxxx labels, recorded/archived status, no subtype. "
        "Authors may revise their own report bodies; main may only archive others' reports, never rewrite "
        "or impersonate the author. Decisions remain route entries (workers only proposed), synthesis "
        "overview, and sources lead. Formed ideas may be shared as conjectures when useful; established "
        "claims cite existing facts, and service results stay service-owned feedback. New/content-edited experience requires "
        "attributes.subtype='method' (usage/improvement lessons from success or failure, "
        "not merely success announcements), "
        "'boundary' (unmet conditions/capability or extrapolation limits), or 'review' "
        "(concrete object-specific doubts/errors/corrections, not no-issue audits). One failed attempt "
        "is not automatically experience, method impossibility or refutation. Unclassified legacy "
        "experience stays readable; archive/status-only maintenance may omit subtype. Content edits, "
        "changed links or added semantic attributes require explicit subtype; never auto-classify. "
        "Audit reports/reviews require attributes.audits containing actual internal target_id, positive "
        "integer revision, checked scope, not_covered (explicit 'none' if empty), optional historical boolean, "
        "and each target in links. Use current revisions or historical=true for older versions. "
        "Never overwrite audit baselines; rechecks of new versions create new linked entries. "
        "Staleness/revocation reminders are not certification; ordinary progress reports may omit audits. "
        "Use a stable creation-only request_id for identical retries, not a fresh key; changed payload "
        "conflicts. Updates use entry_id/expected_revision. Never forge research_policy_version or "
        "record_request metadata. Keep a searchable body summary; read attachments only on demand. "
        "Page with offset/limit, has_more and expected_snapshot_revision, restarting on conflict. "
        "On a new project worker TASK.md files may be unassigned. Read existing assignments before changing them; "
        "do not start workers without meaningful assignments. "
        "Keep the configured roster productively occupied, not with duplicate cosmetic tasks. "
        "An alive worker in waiting state has finished a round and is waiting for changed inputs, "
        "not stuck, failed, or proof of project completion. Do not restart it to keep it busy. "
        "The host checks content hashes every three seconds: changed TASK.md, main overview/formal routes, "
        "facts or verification feedback wake it; ordinary reports, timestamps and identical rewrites do not. "
        "Assign a genuinely new follow-up when useful, not a cosmetic rewrite to force wakeup. "
        f"Use shell CLI {cli} assign {p.name}/WORKER --stdin (or --task TEXT), "
        f"{cli} start {p.name}[/WORKER], {cli} status {p.name} --json, and "
        f"{cli} stop {p.name}[/WORKER]. Do NOT use wrappers that reload other configuration. "
        "Assignment updates take effect when a worker next reads TASK.md; do not claim an immediate interruption. "
        "Do not create extra worker slots, modify software/configuration, delete records, or touch other projects. "
        "The main role has no fact_submit. Worker candidates and all lifecycle changes require independent "
        "verification; do not import KnowledgeStore to bypass the write gate. Follow the supplied computation policy. "
        "Retain scope, quantifiers and hypotheses. Routine lemmas, asymptotic results, or extra assumptions "
        "do not settle the original target. "
        f"{'Only a positive proof may complete this proof-only project. ' if proof_only else 'A published counterexample may settle a yes/no question negatively. '}"
        "Literature access follows RUN_POLICY.md; never fabricate access or bibliographic evidence.\n\n"
        "Perform ONE substantial coordination/reasoning pass, persist useful decisions in the notebook, "
        "then RETURN. A persistent controller will invoke you again; do not sleep for 30 minutes, create "
        "a Goal, depend on clock tools, or spawn another main loop. A returned pass does not stop the project. "
        + (f"Hard research deadline: Unix time {L.run_deadline(p)}. Save progress before it; never extend it. "
           if L.run_deadline(p) is not None else "There is no research deadline or round cap. ")
        + "At least every 30 minutes reassess all routes; "
        "record a timestamped macro audit as kind='report' with attributes.audits at least every four hours; "
        "link separate route decisions. Do not add timers or scheduled tasks. "
        "Recover from previous decisions rather than replanning from zero. Restart crashed workers only "
        "after addressing the failure, not in a tight retry loop.\n"
        "Return ONLY a JSON object: {\"state\":\"continue\"|\"complete\"|\"blocked\","
        "\"summary\":\"what changed, why, and next actions\",\"target_fact_ids\":[]}. "
        "Use complete only with active verified facts that fully answer the original problem and a "
        "precise final bridge argument. The controller will freshly revalidate EVERY dependency proof in full "
        "and then the original-problem bridge before stopping workers. A failed completion review means continue "
        "research: read shared feedback and detailed findings, repair assignments/routes and stop using revoked "
        "facts. Do not stop workers on a completion claim or repeat unchanged failed evidence. "
        "Use blocked only for an operational "
        "problem requiring operator action, never merely for difficult mathematics. "
        "No paper generation, publication, or source deletion is authorized.\n"
        + ("AUTOMATIC COMPLETION DISABLED for this project. Any complete response is only an unaccepted "
           "candidate for operator review, not a reason to stop. Return continue, preserve the candidate "
           "and its full dependency proof, and keep the eight or otherwise configured worker slots researching "
           "and independently auditing until the existing deadline or operator stop. Do not change this policy, "
           "stop workers because of claimed success, or mark the project blocked merely to end research. "
           "Read and enforce the completion restrictions in PROBLEM.md and RUN_POLICY.md.\n"
           if not early_completion_allowed(p) else "")
        + f"Previous controller outcome: {feedback or 'First pass; no research assignments have been seeded.'}"
    )
    env = {"HYPROVER_ROLE": "main", "HYPROVER_AUTHOR": "main_agent",
           "HYPROVER_PROJECT_DIR": str(p), "HYPROVER_AGENTS_ROOT": str(p.parent),
           "HYPROVER_VERIFY_URL": os.environ.get("HYPROVER_VERIFY_URL", ""),
           "HYPROVER_VERIFY_TIMEOUT": os.environ.get("HYPROVER_VERIFY_TIMEOUT", "3600"),
           "HYPROVER_LITERATURE_SEARCH": os.environ.get("HYPROVER_LITERATURE_SEARCH", "1")}
    env_text = ",".join(f"{k}={json.dumps(v)}" for k, v in env.items())
    timeout = int(os.environ.get("HYPROVER_MCP_TOOL_TIMEOUT", "3600"))
    mcp = (f'mcp_servers.hyprover={{command={json.dumps(sys.executable)},'
           f'args=["-m","hyprover.gateway"],tool_timeout_sec={timeout},env={{' + env_text + '}}')
    return codex.exec_cmd(
        codex.resolve_bin(), codex.model("HYPROVER_MAIN_MODEL"), codex.effort("HYPROVER_MAIN_EFFORT"),
        "-C", str(p / "main"), "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
        "-c", "features.current_time_reminder=false", "-c", "features.goals=false",
        "-c", mcp, "--output-last-message", str(output), prompt,
    )


def parse_decision(path: Path) -> dict:
    data = _read(path)
    if (data.get("state") not in ("continue", "complete", "blocked")
            or not isinstance(data.get("summary"), str) or not data["summary"].strip()
            or not isinstance(data.get("target_fact_ids"), list)
            or any(not isinstance(i, str) for i in data["target_fact_ids"])):
        raise ValueError("invalid_main_decision")
    return data


def confirm_completion(p: Path, result: dict, *, results_root: Path | None = None) -> dict:
    if not early_completion_allowed(p):
        raise completion_evidence.EvidenceChanged("automatic_completion_disabled")
    run_id = result.get("full_revalidation_run", "")
    if not isinstance(run_id, str) or not re.fullmatch(r"completion-[0-9a-f]{64}", run_id):
        raise ValueError("completion_full_revalidation_required")
    root = completion_evidence.safe_path(results_root or launcher._results_root(), run_id)
    manifest = completion_evidence.validate_completion(root, p, result=result)
    if run_id != "completion-" + manifest["digest"]:
        raise ValueError("completion_run_digest_mismatch")
    return manifest


def run_completion(p: Path, decision: dict, stopped, update) -> dict:
    config = launcher.completion_config()
    def cancelled():
        if stopped() or not early_completion_allowed(p):
            return True
        if config != launcher.completion_config():
            raise completion_evidence.EvidenceChanged("completion_verifier_configuration_changed")
        return False

    def verify(*, run_id, statement, proof, context):
        if cancelled():
            raise InterruptedError("completion_stop_requested")
        home = launcher.ensure_agent_home()
        command = launcher.build_file_verification_command(run_id, statement, proof, context)
        update(completion_run_id=run_id)
        timeout = config["timeout"]
        deadline = time.monotonic() + timeout if timeout else None
        expired = lambda: deadline is not None and time.monotonic() >= deadline
        try:
            rc = _run_child(command, launcher._results_dir(run_id) / "log.md", home,
                            lambda: cancelled() or expired(), update)
        except InterruptedError:
            if expired() and not cancelled():
                raise TimeoutError("completion_verifier_timeout")
            raise
        path = launcher._verification_path(run_id)
        if rc or path is None:
            raise RuntimeError("completion_verifier_failed")
        return _read(path)

    try:
        return completion.run(p, decision, results_root=launcher._results_root(), config=config,
                              verifier=verify, allocator=launcher._allocate_run_id,
                              check_cancel=cancelled, update=update)
    except InterruptedError:
        if stopped():
            raise
        return {"accepted": False, "outcome": "automatic_completion_disabled", "report_path": "", "revoked": []}


def finish_completion(p: Path, decision: dict, checked: dict, stopped, update) -> dict:
    result = {"state": "completed", "target_fact_ids": decision["target_fact_ids"],
              "summary": decision["summary"], "coverage_verification_run": checked["coverage_verification_run"],
              "full_revalidation_run": checked["full_revalidation_run"],
              "completion_evidence_digest": checked["evidence_digest"],
              "completion_report": checked["report_path"], "formal_certificate": False}
    manifest = confirm_completion(p, result)
    if manifest["config"] != launcher.completion_config():
        raise completion_evidence.EvidenceChanged("completion_verifier_configuration_changed")
    recovery = []
    try:
        if stopped():
            raise InterruptedError("completion_stop_requested")
        workers = stop_workers(p, completion_token=checked["full_revalidation_run"], recovery=recovery)
        if not all(w["stopped"] for w in workers):
            raise RuntimeError("completion_worker_shutdown_incomplete")
        with KnowledgeStore(p)._locked():
            if stopped():
                raise InterruptedError("completion_stop_requested")
            confirm_completion(p, result)
            if stopped():
                raise InterruptedError("completion_stop_requested")
            if not early_completion_allowed(p):
                raise completion_evidence.EvidenceChanged("automatic_completion_disabled")
            if manifest["config"] != launcher.completion_config():
                raise completion_evidence.EvidenceChanged("completion_verifier_configuration_changed")
            result.update(completed_at=time.time(), workers=workers)
            _write(p / "main/result.json", result)
        update(**result)
        return result
    except BaseException:
        if not (p / "main/result.json").exists():
            recovered = resume_completion_workers(p, recovery, stopped)
            update(completion_recovery=recovered)
        raise


def _run_child(cmd: list[str], log: Path, home: Path, stopped, update) -> int:
    with log.open("x") as handle:
        env = codex.subprocess_env(cmd[0])
        proc = subprocess.Popen(cmd, cwd=home, env=env,
                                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)
        update(child_identity=identity(proc.pid))
        try:
            while proc.poll() is None:
                if stopped():
                    raise InterruptedError("main_stop_requested")
                time.sleep(0.5)
            return proc.returncode
        finally:
            _terminate_group(proc.pid)
            proc.wait()
            update(child_identity=None)


def run(project: str, lock_fd: int | None = None) -> int:
    max_failures = int(os.environ.get("HYPROVER_MAIN_MAX_CONSEC_FAILURES", "5"))
    retry_interval = float(os.environ.get("HYPROVER_RETRY_INTERVAL", "30"))
    if max_failures < 0 or not math.isfinite(retry_interval) or retry_interval <= 0:
        raise ValueError("invalid_main_retry_configuration")
    p = project_path(project)
    allow_early_completion = early_completion_allowed(p)
    if (p / ".sequence.json").exists():
        L.require_launch_allowed(p)
    home = prepare(p)
    handle = os.fdopen(lock_fd, "a") if lock_fd is not None else (home / ".lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return 0
    interrupted = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: interrupted.set())
    signal.signal(signal.SIGINT, lambda *_: interrupted.set())
    deadline = L.run_deadline(p)
    stopped = lambda: (interrupted.is_set() or (home / ".stop").exists()
                       or (p / "stop-request.json").exists()
                       or (deadline is not None and time.time() >= deadline))
    state = _read(home / "status.json")
    state.update(identity=identity(os.getpid()), state="running", child_identity=None,
                 consecutive_failures=0, retry_interval_seconds=retry_interval,
                 max_consecutive_failures=max_failures,
                 started_at=time.time(), interval_seconds=float(os.environ.get("HYPROVER_MAIN_INTERVAL", "300")))
    def update(**fields):
        state.update(fields, updated_at=time.time())
        _write(home / "status.json", state)
    update()
    cycle = int(state.get("cycle", 0))
    failures = 0
    feedback = json.dumps(state.get("last_decision", {}), ensure_ascii=False)
    try:
        saved_result = _read(home / "result.json")
        if saved_result.get("state") == "completed":
            try:
                confirm_completion(p, saved_result)
            except (ValueError, OSError) as exc:
                update(state="blocked", completion_outcome="stale", error=f"Saved completion evidence invalid: {exc}")
                return 2
            update(**saved_result)
            return 0
        while not stopped():
            recovery = _read(home / "completion-shutdown.json").get("entries", [])
            if recovery:
                update(completion_recovery=resume_completion_workers(p, recovery, stopped))
            cycle += 1
            stem = f"cycle-{cycle:06d}-{time.time_ns()}"
            output = home / "logs" / (stem + ".json")
            update(state="running", cycle=cycle, output_file=str(output), error=None)
            try:
                rc = _run_child(build_command(p, output, feedback), output.with_suffix(".log"), home, stopped, update)
                if rc:
                    raise RuntimeError(f"main_session_exit_{rc}; see {output.with_suffix('.log')}")
                decision = parse_decision(output)
                update(last_decision=decision)
                if decision["state"] == "complete" and (
                        not allow_early_completion or not early_completion_allowed(p)):
                    candidate = {"state": "awaiting_operator_review", "accepted": False,
                                 "decision": decision, "recorded_at": time.time(), "cycle": cycle}
                    candidate_path = home / "logs" / (stem + "-candidate.json")
                    _write(candidate_path, candidate)
                    decision = {"state": "continue", "target_fact_ids": [], "summary":
                                "Completion claim recorded but NOT accepted. Automatic completion is disabled. "
                                "Continue research and independent proof auditing; do not stop workers or "
                                "change the deadline. Candidate summary: " + decision["summary"]}
                    update(last_decision=decision, completion_candidate_file=str(candidate_path))
                if decision["state"] == "complete":
                    checked = run_completion(p, decision, stopped, update)
                    if checked.get("accepted"):
                        finish_completion(p, decision, checked, stopped, update)
                        return 0
                    decision = {"state": "continue", "target_fact_ids": [],
                                "summary": completion.research_feedback(checked)}
                    update(last_decision=decision, completion_outcome=checked["outcome"],
                           completion_report=checked.get("report_path"),
                           completion_feedback_id=checked.get("feedback_id"))
                if decision["state"] == "blocked":
                    if max_failures == 0:
                        raise RuntimeError("main_reported_blockage: " + decision["summary"])
                    update(state="blocked", workers=stop_workers(p))
                    return 2
                failures = 0
                feedback = json.dumps(decision, ensure_ascii=False)
                delay = max(1, state["interval_seconds"])
                update(state="waiting", consecutive_failures=0, error=None,
                       next_pass_at=time.time() + delay)
            except InterruptedError:
                raise
            except completion_evidence.EvidenceChanged as exc:
                failures = 0
                decision = {"state": "continue", "target_fact_ids": [], "summary":
                            f"Completion evidence changed; no completion accepted. {exc}. Reassess and continue research."}
                feedback = json.dumps(decision, ensure_ascii=False)
                delay = max(1, state["interval_seconds"])
                update(state="waiting", last_decision=decision, completion_outcome="stale",
                       consecutive_failures=0, error=None, next_pass_at=time.time() + delay)
            except Exception as exc:
                failures += 1
                feedback = f"Previous pass failed: {type(exc).__name__}: {exc}"
                delay = retry_interval
                update(state="retrying", error=feedback, consecutive_failures=failures,
                       next_pass_at=time.time() + delay)
                if max_failures > 0 and failures >= max_failures:
                    update(state="blocked", workers=stop_workers(p))
                    return 2
            until = time.monotonic() + delay
            while not stopped() and time.monotonic() < until:
                time.sleep(min(1, max(0, until - time.monotonic())))
    except InterruptedError:
        pass
    finally:
        if stopped():
            update(state="stopped", workers=stop_workers(p))
        handle.close()
    return 0


def start(project: str) -> dict:
    p = project_path(project)
    home = prepare(p)
    with (home / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status(project), launch="already-running")
        state = _read(home / "status.json")
        if state.get("state") == "completed":
            confirm_completion(p, _read(home / "result.json"))
            return dict(status(project), launch="already-completed")
        L.require_launch_allowed(p)
        (home / ".stop").unlink(missing_ok=True)
        _write(home / "status.json", dict(state, state="starting", identity=None))
        with (home / "supervisor.log").open("a") as log:
            proc = subprocess.Popen([sys.executable, "-m", "hyprover.orchestration.controller", "run", project, "--lock-fd", str(lock.fileno())],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, pass_fds=(lock.fileno(),))
        return {"project": project, "pid": proc.pid, "launch": "started"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status", "run"))
    parser.add_argument("project")
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args(argv)
    if args.action == "run":
        return run(args.project, args.lock_fd)
    print(json.dumps(globals()[args.action](args.project), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
