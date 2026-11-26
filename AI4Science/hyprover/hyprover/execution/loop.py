"""Host-side worker lifecycle: run a session, wait for inputs, retry failures.

``hyprover start`` launches this process detached. Each round is one Codex session
resuming the worker's persisted research. A zero exit code enters hash-gated waiting;
it is not evidence that the project or any mathematical claim is complete. Waiting
checks run every three seconds without model calls. Failed or timed-out sessions
use the retry delay. Graceful stops, deadlines and round limits apply at boundaries.

Startup settings:
  HYPROVER_CODEX_BIN            override resolved by hyprover.codex; otherwise wrapper/PATH
  HYPROVER_ROUND_BEAT           fallback failure-retry delay (5 seconds)
  HYPROVER_ROUND_HARD_TIMEOUT   session timeout (14400 seconds; 0 disables)
  HYPROVER_MAX_ROUNDS           round limit for this process (0 disables)
  HYPROVER_MAX_CONSEC_FAILURES  failure limit (5; 0 disables; timeouts reset the counter)
  HYPROVER_RETRY_INTERVAL       failure/timeout delay (defaults to ROUND_BEAT)
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from . import layout as L
from . import scaffold
from .wakeup import InputWatcher, POLL_SECONDS, WakeupReadError
from hyprover import codex

# --- the per-round prompt (continuation semantics; see worker.md) ----------- #

def kickoff(project: str, worker: str) -> str:
    return (
        f"You are worker '{worker}' on project '{project}'. Solve the assigned problem. "
        f"Resume this project's persisted progress if it exists; otherwise start from its problem and task.\n"
        f"1. Read TASK.md — your current assignment (which direction/subgoal is yours).\n"
        f"2. Follow AGENTS.md (worker.md) exactly — your standing contract (the adaptive "
        f"control loop, memory discipline, the fact_submit gate). Drive toward a full "
        f"verified result.\n"
        "3. Every round explicitly recall relevant shared reports with "
        "knowledge_search(query='', kinds=['report'], sort_by='updated', ...) and "
        "service feedback with kinds=['feedback']; also browse facts, conjectures, experiences and leads as needed. "
        "Judge relevance yourself: there are no relevance scores. Browse categories with empty query and "
        "snapshot-checked pagination; use IDs/labels to revisit records. Nonempty text queries are literal "
        "filters only, and no match does not prove no useful knowledge exists. "
        "Default search excludes reports and archived entries, not feedback. "
        "Read ../../KNOWLEDGE.md overview, routes and verified facts relative to this worker directory "
        "(the notebook is at the project root). Resolve notebook links against that root. "
        "Open proof/note attachments only when needed; never read peers' private logs. "
        "DO NOT restart from zero; build on persisted progress.\n"
        "4. Keep going: assess -> reason -> act -> persist, repeatedly. "
        "If progress stalls, try to consider a useful intermediate claim or reformulation, "
        "and try to formulate a precise, testable conjecture that could help advance the proof. "
        "Weigh alternatives if they arise naturally, while following a clear proof path directly. "
        "An open problem is not a reason to stop. Do NOT declare completion prematurely or add timers.\n"
        "5. Keep rough drafts private. Use knowledge_record(kind='report', ...) for substantive progress, "
        "handoffs, blockers, next steps and no-issue manual audits: reports have Pxxx labels, "
        "recorded/archived status and no subtype. Main/workers share reports; revise only your own body. "
        "Main may only archive others' reports, never impersonate their authors. "
        "Keep decisions in proposed routes, synthesis in main's overview, and sources in leads. "
        "Formed ideas may be shared as conjectures when useful; established claims cite existing facts. "
        "Service results remain service-owned feedback. New/content-edited experience requires "
        "attributes.subtype='method' (usage/improvement lessons from success or failure, "
        "not merely success announcements), "
        "'boundary' (unmet conditions/capability or extrapolation limits), or 'review' "
        "(concrete object-specific doubts/errors/corrections, not no-issue audits). One failed attempt "
        "is not automatically experience, method impossibility or refutation. Unclassified legacy "
        "experience stays readable; archive/status-only maintenance may omit subtype. Content edits, "
        "changed links or added semantic attributes require explicit subtype; never auto-classify. "
        "Audit reports/reviews require attributes.audits with actual internal target_id, positive integer "
        "revision, checked scope, not_covered (explicit 'none' if empty), optional historical boolean, "
        "and each target in links. Use current revisions or historical=true for older versions. "
        "Never overwrite audit baselines; rechecks of new versions create new linked entries. "
        "Staleness/revocation reminders are not certification; ordinary progress reports may omit audits. "
        "Use a stable creation-only request_id for identical retries, not a fresh key; changed payload "
        "conflicts. Updates use entry_id/expected_revision. Never forge research_policy_version or "
        "record_request metadata. Keep a searchable body summary; page with offset/limit, has_more "
        "and expected_snapshot_revision, restarting on conflict. Remain within your bound project/identity. "
        "Complete proofs may go directly to fact_submit; count a result only when accepted=true and published=true."
    )


# --- config (read at call time) -------------------------------------------- #

# codex binary + model/effort defaults are resolved via the shared hyprover.codex
# launcher (HYPROVER_CODEX_BIN / HYPROVER_MAIN_MODEL / HYPROVER_MAIN_EFFORT); the worker's
# own model comes from HYPROVER_WORKER_MODEL (falls back to the neutral default).


# --- small helpers --------------------------------------------------------- #

def _read_role(wl: L.WorkerLayout) -> dict:
    out = {"MODEL": codex.model("HYPROVER_WORKER_MODEL"),
           "REASONING_EFFORT": codex.effort(), "ROLE": "max", "HYPROVER_AUTHOR": wl.name}
    rp = wl.role
    if rp.exists():
        for line in rp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    effort_override = os.environ.get("HYPROVER_WORKER_EFFORT", "").strip()
    if effort_override:
        out["REASONING_EFFORT"] = effort_override
    return out


def write_status(wl: L.WorkerLayout, *, validate_knowledge: bool = True, **fields) -> None:
    """Atomic status write; an initialized loop can report unreadable inputs too."""
    if validate_knowledge:
        L.require_knowledge(wl.project_dir)
    path = wl.status
    cur = {}
    if path.exists():
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cur = {}
    cur.update(fields)
    cur["worker"] = wl.name
    cur["pid"] = os.getpid()
    cur["updated_at"] = time.time()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _deadline_passed(project_dir: Path) -> bool:
    f = project_dir / L.DEADLINE_FILE
    if not f.exists():
        return False
    try:
        return time.time() >= float(f.read_text().strip())
    except (ValueError, OSError):
        return False


def _parse_last_fact_id(log_path: Path) -> Optional[str]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    decoder = json.JSONDecoder()
    latest = None

    def visit(value):
        nonlocal latest
        if isinstance(value, dict):
            fact_id = value.get("fact_id")
            if (value.get("accepted") is True and value.get("published") is True
                    and isinstance(fact_id, str)
                    and re.fullmatch(r"[0-9a-f]{16}", fact_id)):
                latest = fact_id
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            try:
                visit(json.loads(value))
            except (ValueError, RecursionError):
                pass

    offset = 0
    while offset < len(text):
        start = text.find("{", offset)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            offset = start + 1
            continue
        visit(value)
        offset = end
    return latest


# --- one round ------------------------------------------------------------- #

class _Child:
    """Holds the running codex subprocess so the SIGTERM handler can kill it."""
    proc: "subprocess.Popen | None" = None


def run_round(wl: L.WorkerLayout, role: dict, prompt: str, log_path: Path,
              hard_timeout: int) -> int:
    """Exec one ``codex exec`` continuation session. Returns codex's rc, 124 on
    hard-timeout (terminate → wait 10s → kill), or 127 if the codex binary is
    missing."""
    L.require_knowledge(wl.project_dir)
    prompt = L.math_computation_policy(L.math_computation_allowed(wl.project_dir)) + "\n" + prompt
    wdir = wl.dir
    codex_bin = codex.resolve_bin()
    cmd = codex.exec_cmd(
        codex_bin, role["MODEL"], role["REASONING_EFFORT"],
        # optional per-project worker-only model provider override (opt-in;
        # empty unless HYPROVER_PROJECT_<PROJECT>_WORKER_API_* is configured)
        *codex.project_worker_config_args(wl.project),
        "-C", str(wdir),
        # on an install without .git (tarball download), codex's
        # trusted-directory check refuses to run the worker round
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        prompt,
    )
    with open(log_path, "x", encoding="utf-8") as logf:
        try:
            _Child.proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, cwd=str(wdir),
                env=codex.subprocess_env(codex_bin, worker_project=wl.project),
            )
        except FileNotFoundError:
            logf.write(f"[worker_loop] codex binary not found: {cmd[0]}\n")
            return 127
        try:
            timeout = hard_timeout if hard_timeout > 0 else None
            deadline = L.run_deadline(wl.project_dir)
            if deadline is not None:
                remaining = max(0, deadline - time.time())
                timeout = remaining if timeout is None else min(timeout, remaining)
            return _Child.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _Child.proc.terminate()
            try:
                _Child.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _Child.proc.kill()
            logf.write(f"\n[worker_loop] round hard-timeout after {hard_timeout}s\n")
            return 124
        finally:
            _Child.proc = None


# --- the loop -------------------------------------------------------------- #

def _cleanup_pid(wl: L.WorkerLayout) -> None:
    """Best-effort exit cleanup: unlink only a PID file whose text still names us."""
    pf = wl.pid
    try:
        if pf.exists() and pf.read_text().strip() == str(os.getpid()):
            pf.unlink(missing_ok=True)
    except OSError:
        pass


def main(worker_dir: str) -> int:
    wdir = Path(worker_dir).resolve()
    if not wdir.is_dir():
        print(f"worker dir not found: {wdir}", file=sys.stderr)
        return 2
    wl = L.WorkerLayout(wdir)
    project_dir = wl.project_dir
    try:
        L.require_knowledge(project_dir)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if (project_dir / ".sequence.json").exists():
        L.require_launch_allowed(project_dir)
    project = wl.project
    worker = wl.name
    role = _read_role(wl)

    # Pin the worker's gateway to THIS interpreter (sys.executable = the venv
    # python hyprover runs on), rewritten every start: a moved/rebuilt venv is
    # picked up, and a bare `python3` on codex's PATH can never resolve the
    # gateway to a different install.
    scaffold.write_codex_config(wl)

    beat = float(os.environ.get("HYPROVER_ROUND_BEAT", "5"))
    hard_timeout = int(os.environ.get("HYPROVER_ROUND_HARD_TIMEOUT", "14400"))
    max_rounds = int(os.environ.get("HYPROVER_MAX_ROUNDS", "0"))
    max_fail = int(os.environ.get("HYPROVER_MAX_CONSEC_FAILURES", "5"))
    retry_interval = float(os.environ.get("HYPROVER_RETRY_INTERVAL", str(beat)))
    if max_fail < 0 or not math.isfinite(retry_interval) or retry_interval < 0:
        raise ValueError("invalid_worker_retry_configuration")
    wl.logs.mkdir(parents=True, exist_ok=True)
    prompt = kickoff(project, worker)
    metadata_path = project_dir / "project.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get("research_mode") == "proof_only":
        prompt = (
            "PROOF-ONLY PROJECT: read RUN_POLICY.md at " + str(project_dir / "RUN_POLICY.md") + ". "
            "Seek a positive proof; do not search for counterexamples to the target or pursue standalone "
            "counterexample hunts for auxiliary claims. This overrides older tasks, notes and generic "
            "counterexample suggestions. Still audit proofs, report and repair gaps honestly, and never "
            "assume the target or an unproved lemma as a premise. Resume only proof-oriented work.\n"
        ) + prompt

    def set_status(**fields):
        write_status(wl, validate_knowledge=False, **fields)

    def stopped() -> bool:
        state = None
        if wl.stop.exists() or (project_dir / "stop-request.json").exists():
            try:
                completion_stop = re.fullmatch(r"completion-[0-9a-f]{64}", wl.stop.read_text())
            except FileNotFoundError:
                completion_stop = False
            if not completion_stop:
                wl.stop.unlink(missing_ok=True)
            state = "stopped"
        elif _deadline_passed(project_dir):
            state = "deadline"
        elif max_rounds and rounds_this_run >= max_rounds:
            state = "max_rounds"
        if state:
            set_status(state=state, waiting_since=None, next_round_at=None)
        return state is not None

    def _on_term(signum, _frame):
        if _Child.proc is not None:
            _Child.proc.terminate()
        set_status(state="terminated", waiting_since=None, next_round_at=None)
        _cleanup_pid(wl)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_term)

    rnd = 0
    for path in wl.logs.glob("round_*.log"):
        match = re.fullmatch(r"round_(\d+)\.log", path.name)
        if match:
            rnd = max(rnd, int(match.group(1)))
    set_status(state="running", round=rnd, started_at=time.time(),
               waiting_since=None, next_round_at=None, input_error=None)
    rounds_this_run = 0
    consec_fail = 0
    watcher = InputWatcher(project_dir, wl.task)
    baseline = None
    wait_for_change = False
    waiting_since = None
    input_error = None
    try:
        while True:
            if stopped():
                break
            try:
                current = watcher.read()
            except WakeupReadError as exc:
                error = str(exc)
                if error != input_error:
                    input_error = error
                    waiting_since = waiting_since or time.time()
                    set_status(state="waiting", waiting_since=waiting_since,
                               input_error=error, next_round_at=None)
                    print(f"[worker_loop] waiting for readable inputs: {error}", file=sys.stderr, flush=True)
                time.sleep(POLL_SECONDS)
                continue
            if input_error is not None:
                input_error = None
                set_status(input_error=None)
                print("[worker_loop] research inputs readable again", file=sys.stderr, flush=True)
            changes = current.changed_from(baseline) if baseline is not None else ()
            if wait_for_change and not changes:
                time.sleep(POLL_SECONDS)
                continue
            if stopped():
                break

            reason = list(changes) if wait_for_change else ["initial" if baseline is None else "retry"]
            baseline = current
            waiting_since = None
            rnd += 1
            rounds_this_run += 1
            log_path = wl.logs / f"round_{rnd}.log"
            set_status(state="running", round=rnd, round_started_at=time.time(),
                       waiting_since=None, next_round_at=None, input_fingerprint=current.digest,
                       wake_reason=reason)
            round_prompt = prompt + "\nHost wake reason: " + ", ".join(reason) + ". " + (
                "Read changed inputs and resume persisted work. Do not poll for assignments inside the model "
                "or repeat an unchanged report. If no useful work remains until new inputs arrive, return; "
                "the host checks for changes every three seconds without model calls.\n")
            rc = run_round(wl, role, round_prompt, log_path, hard_timeout)
            wait_for_change = rc == 0
            waiting_since = time.time() if wait_for_change else None
            consec_fail = consec_fail + 1 if rc not in (0, 124) else 0
            set_status(state="waiting" if wait_for_change else "retrying", round=rnd,
                       last_round_at=time.time(), last_rc=rc, last_fact_id=_parse_last_fact_id(log_path),
                       waiting_since=waiting_since, consecutive_failures=consec_fail,
                       error=f"codex exit {rc}" if rc != 0 else None,
                       next_round_at=None if wait_for_change else time.time() + retry_interval)
            if rc == 127 and max_fail > 0:
                set_status(state="error", error="codex binary not found", next_round_at=None)
                return 127
            if max_fail and consec_fail >= max_fail:
                set_status(state="error", error=f"{consec_fail} consecutive failed rounds", next_round_at=None)
                return 1
            if rc != 0:
                if stopped():
                    break
                if retry_interval > 0:
                    time.sleep(retry_interval)
    finally:
        _cleanup_pid(wl)
    return 0
