"""Offline controller tests; no research models or network calls."""
import fcntl
import json
from copy import deepcopy
import os
import signal
from pathlib import Path
from unittest.mock import Mock

import pytest

from hyprover.core import Fact, KnowledgeStore
from hyprover.execution import scaffold, layout as L
from hyprover.orchestration import controller as c
from hyprover.verify import completion, completion_evidence as e

GOOD = {"verdict": "correct", "verification_report": {
    "summary": "Synthetic coverage", "critical_errors": [], "gaps": []}, "repair_hints": ""}


def verifier_pass(**kwargs):
    """Offline completion report binding the requested job and coverage fields."""
    ctx = kwargs["context"]["completion_review"]
    manifest = e.read_json(Path(ctx["manifest_path"]))
    report = deepcopy(GOOD)
    report["coverage"] = dict(e.expected_coverage(manifest, ctx["job"]),
                              uncovered=[], materials_complete=True, truncated=False)
    return report


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPROVER_AGENTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("VERIFIER_RESULTS_DIR", str(tmp_path / "verify-runs"))
    monkeypatch.setenv("VERIFY_AGENT_HOME", str(tmp_path / "verify-home"))
    scaffold.do_new("P", roles="ultra:10", model="test-model")
    p = L.project_dir("P")
    (p / "PROBLEM.md").write_text("Synthetic exact question")
    return p


def test_prompt_explains_waiting_without_restart_or_cosmetic_tasks(project):
    prompt = c.build_command(project, project / "out.json")[-1]
    assert "An alive worker in waiting state" in prompt
    assert "Do not restart it to keep it busy" in prompt
    assert "content hashes every three seconds" in prompt
    assert "ordinary reports, timestamps and identical rewrites do not" in prompt
    assert "genuinely new follow-up" in prompt
    assert "knowledge_search(query=''" in prompt
    assert "judge relevance yourself" in prompt
    assert "there are no relevance scores" in prompt


def test_child_environment_signature_matches_published_codex(project, monkeypatch):
    c.prepare(project)
    process = Mock(pid=7654321, returncode=0, poll=Mock(return_value=0))
    spawn = Mock(return_value=process)
    monkeypatch.setattr(c.subprocess, "Popen", spawn)
    monkeypatch.setattr(c, "identity", lambda pid: {"pid": pid})
    monkeypatch.setattr(c, "_terminate_group", lambda pid: None)
    assert c._run_child(["codex", "exec"], project / "main/logs/env.log", project / "main",
                        lambda: False, lambda **kw: None) == 0
    assert isinstance(spawn.call_args.kwargs["env"], dict)
    process.wait.assert_called_once()


def test_roster_has_no_preassigned_math(project):
    tasks = list(project.glob("workers/*/TASK.md"))
    assert len(tasks) == 10
    assert all("unassigned" in p.read_text() for p in tasks)
    assert not KnowledgeStore(project).snapshot().facts


def test_main_command_binds_role_project_and_model(project, monkeypatch):
    monkeypatch.setenv("HYPROVER_MAIN_MODEL", "test-main")
    monkeypatch.setenv("HYPROVER_MAIN_EFFORT", "ultra")
    cmd = c.build_command(project, project / "out.json")
    assert cmd[cmd.index("--model") + 1] == "test-main"
    assert 'model_reasoning_effort="ultra"' in cmd
    assert cmd[cmd.index("-C") + 1] == str(project / "main")
    assert any('HYPROVER_ROLE="main"' in x and str(project) in x for x in cmd)
    assert "YOU choose independent investigations" in cmd[-1]
    assert "ultra10" in cmd[-1] and "no research deadline or round cap" in cmd[-1]
    assert "hyprover.orchestration" in cmd[-1]
    assert "supplied by this controller" in cmd[-1]
    for required in ("kinds=['report']", "sort_by='updated'", "kinds=['feedback']",
                     "kind='report'", "Pxxx", "recorded/archived", "no subtype",
                     "attributes.subtype", "'method'", "'boundary'", "'review'",
                     "attributes.audits", "target_id", "not_covered", "historical=true",
                     "request_id", "entry_id/expected_revision", "record_request",
                     "research_policy_version", "expected_snapshot_revision",
                     "main may only archive others' reports", "not certification",
                     "Do not add timers or scheduled tasks",
                     "When a key link is unclear, try to consider useful intermediate claims or reformulations, "
                     "and try to formulate a precise, testable conjecture that could help advance the proof.",
                     "Compare alternatives if they arise naturally", "preserving a clear proof path",
                     "Formed ideas may be shared as conjectures when useful"):
        assert required in cmd[-1]
    assert "macro audit as kind='report'" in cmd[-1]
    assert "route/experience" not in cmd[-1]
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    mcp = next(x for x in cmd if x.startswith("mcp_servers.hyprover="))
    config = tomllib.loads(mcp)["mcp_servers"]["hyprover"]
    assert config["env"]["HYPROVER_PROJECT_DIR"] == str(project)
    assert config["env"]["HYPROVER_ROLE"] == "main"
    assert config["command"] == c.sys.executable


def test_controller_injects_only_knowledge_gateway(project):
    cmd = c.build_command(project, project / "out.json")
    overrides = [arg for arg in cmd if arg.startswith("mcp_servers.")]
    assert len(overrides) == 1
    assert overrides[0].startswith("mcp_servers.hyprover=")


def test_proof_only_direction_and_completion_coverage(project):
    metadata_path = project / "project.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["research_mode"] = "proof_only"
    metadata_path.write_text(json.dumps(metadata))
    prompt = c.build_command(project, project / "out.json")[-1]
    assert "PROOF-ONLY PROJECT" in prompt
    assert "never assume the target as a premise" in prompt
    assert "Do not assign counterexample searches for the target" in prompt
    assert "consider useful intermediate claims or reformulations" in prompt
    assert "A published counterexample may settle" not in prompt
    _, fid = publish(project)
    statements = []
    def verify(**kw):
        statements.append(kw["statement"])
        return verifier_pass(**kw)
    result = full_result(project, fid, verifier=verify)
    assert "complete positive proof" in statements[-1]
    assert "Do not assume the original assertion" in statements[-1]
    assert "positively or negatively" not in statements[-1]
    c.confirm_completion(project, result)


@pytest.mark.parametrize("name", ["../P", "", ".", "/P", "P/x"])
def test_bad_project_rejected(project, name):
    with pytest.raises(ValueError):
        c.project_path(name)


def test_contract_provision_and_duplicate_start_lock(project, monkeypatch):
    home = c.prepare(project)
    assert (home / "AGENTS.md").resolve() == L.repo_root() / "agents/contracts/main_agent.md"
    with (home / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(c.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate main"))
        assert c.start("P")["launch"] == "already-running"


def test_launch_passes_lock_to_correct_module(project, monkeypatch):
    calls = []
    def spawn(cmd, **kwargs):
        calls.append((cmd, kwargs))
        assert kwargs["pass_fds"]
        return Mock(pid=1234)
    monkeypatch.setattr(c.subprocess, "Popen", spawn)
    assert c.start("P")["launch"] == "started"
    assert calls[0][0][2] == "hyprover.orchestration.controller"
    assert calls[0][1]["start_new_session"] is True
    assert not list(project.glob("workers/*/.pid"))


def test_stale_pid_does_not_claim_active_main(project):
    home = c.prepare(project)
    c._write(home / "status.json", {"state": "running", "identity": {
        "pid": os.getpid(), "start_ticks": "not-the-current-process"}})
    assert c.status("P")["alive"] is False
    assert c.status("P")["state"] == "interrupted"


def test_stop_never_signals_mismatched_worker_process(project, monkeypatch):
    worker = project / "workers/ultra"
    (worker / ".pid").write_text(str(os.getpid()))
    monkeypatch.setattr(c, "_terminate_group", lambda pid: pytest.fail("wrong process signalled"))
    result = c.stop("P")
    assert result["stop_requested"]
    assert (project / "main/.stop").exists()
    assert any(r.get("error") == "process_identity_mismatch" for r in result["workers"])


@pytest.mark.parametrize("value", [{}, {"state": "complete", "summary": "", "target_fact_ids": []},
    {"state": "yes", "summary": "Done", "target_fact_ids": []},
    {"state": "complete", "summary": "Done", "target_fact_ids": [1]}])
def test_invalid_main_output(project, value):
    path = project / "decision.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        c.parse_decision(path)


def publish(project):
    store = KnowledgeStore(project)
    fid = store.publish(Fact("", "P", "worker", [], "Synthetic theorem", "Synthetic complete proof"), {"verdict": "correct"})
    return store, fid


def full_result(project, fid, verifier=verifier_pass):
    decision = {"target_fact_ids": [fid], "summary": "Exact coverage"}
    checked = completion.run(project, decision, results_root=c.launcher._results_root(),
                             config=c.launcher.completion_config(), verifier=verifier,
                             allocator=c.launcher._allocate_run_id)
    assert checked["accepted"]
    return dict(decision, state="completed", full_revalidation_run=checked["full_revalidation_run"],
                coverage_verification_run=checked["coverage_verification_run"],
                completion_evidence_digest=checked["evidence_digest"], completion_report=checked["report_path"])


@pytest.mark.parametrize("allowed", [False, True])
def test_completion_math_permission_binds_all_jobs_and_invalidates_cache(project, allowed):
    _, fid = publish(project)
    metadata_path = project / "project.json"
    metadata = c._read(metadata_path)
    c._write(metadata_path, dict(metadata, allow_math_computation=allowed))
    decision = {"target_fact_ids": [fid], "summary": "Exact coverage"}
    manifest, _ = e.capture(project, decision, {"chunk_chars": 8})
    assert manifest["allow_math_computation"] is allowed
    seen = []
    def verify(**kwargs):
        assert kwargs["context"]["allow_math_computation"] is allowed
        seen.append(kwargs["context"]["completion_review"]["job"]["kind"])
        return verifier_pass(**kwargs)
    checked = completion.run(project, decision, results_root=c.launcher._results_root(),
                             config={"chunk_chars": 8}, verifier=verify,
                             allocator=c.launcher._allocate_run_id)
    assert checked["accepted"] and {"chunk", "whole", "bridge"} <= set(seen)
    root = c.launcher._results_root() / checked["full_revalidation_run"]
    c._write(metadata_path, dict(metadata, allow_math_computation=not allowed))
    with pytest.raises(e.EvidenceChanged):
        e.validate_completion(root, project)
    new, _ = e.capture(project, decision, {"chunk_chars": 8})
    assert new["digest"] != manifest["digest"]
    assert KnowledgeStore(project).get(fid).status == "verified"


def test_legacy_missing_permission_normalizes_to_false(project):
    _, fid = publish(project)
    decision = {"target_fact_ids": [fid], "summary": "Exact coverage"}
    manifest, _ = e.capture(project, decision, {})
    path = project / "project.json"
    metadata = c._read(path)
    del metadata["allow_math_computation"]
    c._write(path, metadata)
    e.assert_current(project, manifest)
    assert "allow_math_computation" not in c._read(path)


def test_main_refreshes_math_permission_between_passes(project):
    path = project / "project.json"
    metadata = c._read(path)
    for allowed in (False, True, False):
        c._write(path, dict(metadata, allow_math_computation=allowed))
        prompt = c.build_command(project, project / "out.json")[-1]
        assert f"Project computation policy: allow_math_computation={str(allowed).lower()}." in prompt
        assert "All mathematics is text-only." not in prompt
    c._write(path, dict(metadata, allow_math_computation="true"))
    with pytest.raises(ValueError, match="must_be_boolean"):
        c.build_command(project, project / "out.json")


def test_completion_requires_real_verified_facts(project):
    with pytest.raises(ValueError):
        e.capture(project, {"target_fact_ids": [], "summary": "Done"}, {})
    with pytest.raises(ValueError):
        e.capture(project, {"target_fact_ids": ["a" * 16], "summary": "Done"}, {})
    store, fid = publish(project)
    result = full_result(project, fid)
    manifest = c.confirm_completion(project, result)
    assert (project / "PROBLEM.md").read_text() == manifest["problem"]
    assert manifest["facts"][0]["statement"] == "Synthetic theorem"
    store.revoke(fid, "Synthetic defect")
    with pytest.raises(ValueError, match="evidence_changed"):
        c.confirm_completion(project, result)


def test_completion_cannot_accept_problem_drift_or_bare_verdict(project):
    _, fid = publish(project)
    result = full_result(project, fid)
    with pytest.raises(ValueError, match="full_revalidation_required"):
        c.confirm_completion(project, {"verdict": "correct"})
    (project / "PROBLEM.md").write_text("Different question")
    with pytest.raises(ValueError, match="evidence_changed"):
        c.confirm_completion(project, result)


def test_continue_pass_is_not_project_completion(project, monkeypatch):
    def child(cmd, log, home, stopped, update):
        output = Path(cmd[cmd.index("--output-last-message") + 1])
        output.write_text(json.dumps({"state": "continue", "summary": "Workers started", "target_fact_ids": []}))
        (home / ".stop").touch()
        return 0
    monkeypatch.setattr(c, "_run_child", child)
    monkeypatch.setattr(c.signal, "signal", lambda *a: None)
    assert c.run("P") == 0
    assert c.status("P")["state"] == "stopped"
    assert not (project / "main/result.json").exists()


def test_completed_model_claim_gets_fresh_coverage_review(project, monkeypatch):
    _, fid = publish(project)
    verify_home = project / "verify-home"
    verify_home.mkdir()
    calls, jobs = [], {}
    def build(run_id, statement, proof, context):
        jobs[run_id] = dict(run_id=run_id, statement=statement, proof=proof, context=context)
        return ["full-review", run_id]
    def child(cmd, log, home, stopped, update):
        calls.append(cmd)
        if cmd[0] == "full-review":
            out = c.launcher._results_dir(cmd[1]) / "verification.json"
            out.write_text(json.dumps(verifier_pass(**jobs[cmd[1]])))
        else:
            target = Path(cmd[cmd.index("--output-last-message") + 1])
            target.write_text(json.dumps({"state": "complete", "summary": "Exact original coverage", "target_fact_ids": [fid]}))
        return 0
    monkeypatch.setattr(c, "_run_child", child)
    monkeypatch.setattr(c.signal, "signal", lambda *a: None)
    monkeypatch.setattr(c.launcher, "ensure_agent_home", lambda: verify_home)
    monkeypatch.setattr(c.launcher, "build_file_verification_command", build)
    assert c.run("P") == 0
    result = json.loads((project / "main/result.json").read_text())
    assert result["state"] == "completed" and len(calls) == 3
    assert result["coverage_verification_run"] == calls[-1][1]
    c.confirm_completion(project, result)
    assert result["formal_certificate"] is False
    assert c.start("P")["launch"] == "already-completed"


def test_failure_backoff_stops_hot_loop(project, monkeypatch):
    sleeps = []
    ticks = iter(range(0, 100000, 10000))
    monkeypatch.setattr(c.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(c.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(c.signal, "signal", lambda *a: None)
    monkeypatch.setattr(c, "_run_child", lambda *a: 1)
    assert c.run("P") == 2
    state = c.status("P")
    assert state["state"] == "blocked" and state["consecutive_failures"] == 5


@pytest.mark.parametrize("failure_kind", ["exit", "exception", "blocked"])
def test_unlimited_fixed_retry_recovers_and_remains_stoppable(project, monkeypatch, failure_kind):
    monkeypatch.setenv("HYPROVER_MAIN_MAX_CONSEC_FAILURES", "0")
    monkeypatch.setenv("HYPROVER_RETRY_INTERVAL", "30")
    clock = [0.0]
    starts = []
    states = []
    original_write = c._write

    def write(path, data):
        if path.name == "status.json":
            states.append(dict(data))
        original_write(path, data)

    def child(cmd, log, home, stopped, update):
        starts.append(clock[0])
        if len(starts) <= 7:
            if failure_kind == "exception":
                raise ConnectionError("offline test disconnection")
            if failure_kind == "exit":
                return 1
            decision = {"state": "blocked", "summary": "upstream unavailable", "target_fact_ids": []}
        else:
            decision = {"state": "continue", "summary": "connection recovered", "target_fact_ids": []}
            (home / ".stop").touch()
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text(json.dumps(decision))
        return 0

    monkeypatch.setattr(c, "_run_child", child)
    monkeypatch.setattr(c, "_write", write)
    monkeypatch.setattr(c.signal, "signal", lambda *a: None)
    monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(c.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    assert c.run("P") == 0
    assert starts == [30.0 * i for i in range(8)]
    assert [s["consecutive_failures"] for s in states if s["state"] == "retrying"] == list(range(1, 8))
    assert not any(s["state"] == "blocked" for s in states)
    assert c.status("P")["state"] == "stopped"
    assert c.status("P")["consecutive_failures"] == 0
    assert c.status("P")["error"] is None
    assert not (project / "main/result.json").exists()


def test_stop_during_retry_wait_does_not_launch_again(project, monkeypatch):
    monkeypatch.setenv("HYPROVER_MAIN_MAX_CONSEC_FAILURES", "0")
    monkeypatch.setenv("HYPROVER_RETRY_INTERVAL", "30")
    starts = []
    monkeypatch.setattr(c, "_run_child", lambda *a: starts.append(1) or 1)
    monkeypatch.setattr(c.signal, "signal", lambda *a: None)
    monkeypatch.setattr(c.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(c.time, "sleep", lambda seconds: (project / "main/.stop").touch())
    assert c.run("P") == 0
    assert len(starts) == 1
    assert c.status("P")["state"] == "stopped"


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_invalid_retry_interval_rejected(project, monkeypatch, interval):
    monkeypatch.setenv("HYPROVER_RETRY_INTERVAL", interval)
    with pytest.raises(ValueError, match="invalid_main_retry_configuration"):
        c.run("P")


def test_completion_policy_defaults_and_rejects_non_boolean(project):
    assert c.early_completion_allowed(project)
    path = project / "project.json"
    metadata = c._read(path)
    for invalid in ("false", 0, 1, None):
        c._write(path, dict(metadata, allow_early_completion=invalid))
        with pytest.raises(ValueError, match="must_be_boolean"):
            c.early_completion_allowed(project)
    c._write(path, dict(metadata, allow_early_completion=False))
    assert "AUTOMATIC COMPLETION DISABLED" in c.build_command(project, project / "out.json")[-1]
    with pytest.raises(ValueError, match="automatic_completion_disabled"):
        c.confirm_completion(project, {})


@pytest.mark.parametrize("stop_kind", ["operator", "deadline"])
def test_disabled_completion_preserves_candidates_without_failure_or_early_stop(project, monkeypatch, stop_kind):
    path = project / "project.json"
    metadata = c._read(path)
    c._write(path, dict(metadata, allow_early_completion=False))
    clock = [1000.0]
    (project / ".run_deadline").write_text("1010")
    monkeypatch.setenv("HYPROVER_MAIN_INTERVAL", "1")
    monkeypatch.setattr(c.time, "time", lambda: clock[0])
    monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(c.time, "sleep", lambda t: clock.__setitem__(0, clock[0] + t))
    monkeypatch.setattr(c.signal, "signal", lambda *a: None)
    calls, stops = [], []
    def child(cmd, log, home, stopped, update):
        assert not stops
        calls.append(1)
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text(json.dumps({
            "state": "complete", "summary": "Synthetic unsupported bridge", "target_fact_ids": []}))
        if len(calls) == 2:
            c._write(path, dict(metadata, allow_early_completion=True))
        if len(calls) == 7:
            if stop_kind == "operator":
                (home / ".stop").touch()
            else:
                clock[0] = 1010.0
        return 0
    def stop_workers(p):
        assert len(calls) == 7
        stops.append(1)
        return []
    monkeypatch.setattr(c, "_run_child", child)
    monkeypatch.setattr(c, "stop_workers", stop_workers)
    monkeypatch.setattr(c, "run_completion", lambda *a: pytest.fail("automatic review invoked"))
    assert c.run("P") == 0
    assert len(calls) == 7 and len(stops) == 1
    candidates = list((project / "main/logs").glob("*-candidate.json"))
    assert len(candidates) == 7
    assert all(c._read(p)["accepted"] is False for p in candidates)
    assert all(c._read(p)["state"] == "awaiting_operator_review" for p in candidates)
    assert not (project / "main/result.json").exists()
    assert c.status("P")["consecutive_failures"] == 0
    assert c.status("P")["state"] == "stopped"
    assert c.status("P")["last_decision"]["state"] == "continue"


@pytest.mark.parametrize("outcome", ["fact_defect", "bridge_gap", "disputed", "incomplete", "stale"])
def test_mathematical_failure_continues_without_failure_budget(project, monkeypatch, outcome):
    clock, calls, stops = [0.0], [], []
    monkeypatch.setenv("HYPROVER_MAIN_INTERVAL", "1")
    monkeypatch.setenv("HYPROVER_MAIN_MAX_CONSEC_FAILURES", "2")
    monkeypatch.setattr(c.signal, "signal", lambda *a: None)
    monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(c.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    def child(cmd, log, home, stopped, update):
        assert not stops
        calls.append(cmd)
        if len(calls) > 1:
            assert "Continue research" in cmd[-1]
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text(json.dumps({
            "state": "complete", "summary": "Same incomplete proof", "target_fact_ids": []}))
        if len(calls) == 6:
            (home / ".stop").touch()
        return 0
    monkeypatch.setattr(c, "_run_child", child)
    monkeypatch.setattr(c, "run_completion", lambda *a: {"accepted": False, "outcome": outcome,
                        "revoked": [], "report_path": "offline-report.json"})
    monkeypatch.setattr(c, "stop_workers", lambda p: stops.append(1) or [])
    assert c.run("P") == 0
    assert len(calls) == 6 and stops == [1]
    assert c.status("P")["consecutive_failures"] == 0
    assert not (project / "main/result.json").exists()


def owned_stop(project, name="ultra"):
    path = project / "workers" / name / ".stop"
    path.write_text("owned-completion")
    stat = path.stat()
    return {"worker": name, "token": "owned-completion", "marker": [stat.st_ino, stat.st_mtime_ns, stat.st_size]}


@pytest.mark.parametrize("block", ["none", "manual", "project", "deadline"])
def test_recovery_only_restarts_own_unmodified_stop(project, monkeypatch, block):
    from hyprover.orchestration import cli
    c.prepare(project)
    entry = owned_stop(project)
    started = []
    monkeypatch.setattr(cli, "_start_one", lambda wl, **kw: started.append(wl.name) or "started")
    if block == "manual":
        (project / "workers/ultra/.stop").write_text("operator stop")
    if block == "project":
        (project / "stop-request.json").write_text("{}")
    if block == "deadline":
        (project / ".run_deadline").write_text("1")
    c.resume_completion_workers(project, [entry], lambda: False)
    assert started == (["ultra"] if block == "none" else [])


def test_manual_stop_arriving_inside_start_lock_is_not_cleared(project, monkeypatch):
    from hyprover.orchestration import cli
    c.prepare(project)
    entry = owned_stop(project)
    original = cli.fcntl.flock
    def lock(fd, operation):
        original(fd, operation)
        if operation & fcntl.LOCK_EX:
            (project / "workers/ultra/.stop").write_text("manual stop after recovery inspection")
    monkeypatch.setattr(cli.fcntl, "flock", lock)
    monkeypatch.setattr(cli, "spawn_loop", lambda *a: pytest.fail("manual stop overridden"))
    result = c.resume_completion_workers(project, [entry], lambda: False)
    assert result == [{"worker": "ultra", "launch": "cancelled"}]
    assert (project / "workers/ultra/.stop").read_text().startswith("manual stop")


def test_post_shutdown_evidence_drift_restores_owned_workers(project, monkeypatch):
    from hyprover.orchestration import cli
    c.prepare(project)
    _, fid = publish(project)
    result = full_result(project, fid)
    checked = {"full_revalidation_run": result["full_revalidation_run"],
               "coverage_verification_run": result["coverage_verification_run"],
               "evidence_digest": result["completion_evidence_digest"], "report_path": result["completion_report"]}
    restored = []
    def shutdown(p, **kwargs):
        kwargs["recovery"].append(owned_stop(project))
        (p / "PROBLEM.md").write_text("Changed during final shutdown")
        return [{"worker": "ultra", "stopped": True}]
    monkeypatch.setattr(c, "stop_workers", shutdown)
    monkeypatch.setattr(cli, "_start_one", lambda wl, **kw: restored.append(wl.name) or "started")
    with pytest.raises(e.EvidenceChanged):
        c.finish_completion(project, result, checked, lambda: False, lambda **_: None)
    assert restored == ["ultra"]
    assert not (project / "main/result.json").exists()


@pytest.mark.parametrize("change", ["stop", "policy"])
def test_stop_or_policy_change_during_final_hash_check_blocks_commit(project, monkeypatch, change):
    c.prepare(project)
    _, fid = publish(project)
    result = full_result(project, fid)
    checked = {"full_revalidation_run": result["full_revalidation_run"],
               "coverage_verification_run": result["coverage_verification_run"],
               "evidence_digest": result["completion_evidence_digest"], "report_path": result["completion_report"]}
    original, count, stopped = c.confirm_completion, [0], [False]
    def confirm(*args, **kw):
        manifest = original(*args, **kw)
        count[0] += 1
        if count[0] == 2:
            if change == "stop":
                stopped[0] = True
            else:
                path = project / "project.json"
                c._write(path, dict(c._read(path), allow_early_completion=False))
        return manifest
    monkeypatch.setattr(c, "confirm_completion", confirm)
    monkeypatch.setattr(c, "stop_workers", lambda *a, **kw: [])
    with pytest.raises((InterruptedError, e.EvidenceChanged)):
        c.finish_completion(project, result, checked, lambda: stopped[0], lambda **_: None)
    assert not (project / "main/result.json").exists()


def test_fast_cooperative_worker_exit_still_records_stop_ownership(project, monkeypatch):
    c.prepare(project)
    wdir = project / "workers/ultra"
    (wdir / ".pid").write_text("12345")
    owner = {"pid": 12345, "start_ticks": "original"}
    monkeypatch.setattr(c, "identity", lambda pid: owner if pid == 12345 and not (wdir / ".stop").exists() else None)
    monkeypatch.setattr(c.os, "getpgid", lambda pid: pid)
    original_read = Path.read_bytes
    def read(path):
        if str(path) == "/proc/12345/cmdline":
            return b"python\0hyprover.execution\0" + str(wdir).encode()
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    recovery = []
    assert all(x["stopped"] for x in c.stop_workers(project, completion_token="owned", recovery=recovery))
    assert len(recovery) == 1 and recovery[0]["worker"] == "ultra"
    assert c._read(project / "main/completion-shutdown.json")["entries"] == recovery


def test_start_rejects_expired_completion_without_rerunning_history(project, monkeypatch):
    c.prepare(project)
    store, fid = publish(project)
    result = full_result(project, fid)
    c._write(project / "main/result.json", result)
    c._write(project / "main/status.json", result)
    store.set_external_refs(fid, [{"title": "Changed source"}])
    monkeypatch.setattr(c.subprocess, "Popen", lambda *a, **kw: pytest.fail("historical project restarted"))
    with pytest.raises(e.EvidenceChanged):
        c.start("P")
    monkeypatch.setattr(c.signal, "signal", lambda *a: None)
    assert c.run("P") == 2
    assert c.status("P")["completion_outcome"] == "stale"
    assert c._read(project / "main/result.json") == result


@pytest.mark.parametrize("completion_marker", [True, False])
def test_cooperative_loop_retains_only_owned_completion_marker(project, monkeypatch, completion_marker):
    from hyprover.execution import loop
    marker = project / "workers/ultra/.stop"
    marker.write_text("completion-" + "a" * 64 if completion_marker else "")
    original = marker.stat()
    monkeypatch.setattr(loop.signal, "signal", lambda *a: None)
    assert loop.main(str(marker.parent)) == 0
    assert marker.exists() is completion_marker
    if completion_marker:
        assert marker.stat().st_ino == original.st_ino
        assert marker.stat().st_mtime_ns == original.st_mtime_ns


def test_stop_after_revalidation_never_commits_completion(project, monkeypatch):
    c.prepare(project)
    _, fid = publish(project)
    result = full_result(project, fid)
    checked = {"full_revalidation_run": result["full_revalidation_run"],
               "coverage_verification_run": result["coverage_verification_run"],
               "evidence_digest": result["completion_evidence_digest"], "report_path": result["completion_report"]}
    monkeypatch.setattr(c, "stop_workers", lambda *a, **kw: pytest.fail("already stopped"))
    with pytest.raises(InterruptedError):
        c.finish_completion(project, result, checked, lambda: True, lambda **_: None)
    assert not (project / "main/result.json").exists()
