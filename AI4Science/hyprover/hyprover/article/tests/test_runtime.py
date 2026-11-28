import json
import os
import shutil
import subprocess
import sys
import time

import pytest

from hyprover.article import evidence, runtime
from hyprover.article.__main__ import main
from hyprover.article.models import ArticleConfig, ArticleError, EvidenceChanged, digest, read_json, write_json
from .conftest import complete, publish
from .test_audit import FakeAgent


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS process metadata")
def test_darwin_process_identity_is_stable_and_rejects_invalid_pid():
    pid = os.getpid()
    identity = runtime.process_identity(pid)
    assert identity is not None
    assert identity["pid"] == pid and identity["start_time"]
    assert runtime.process_identity(pid) == identity
    assert runtime.process_identity(-pid) is None


def test_wheel_contains_and_loads_versioned_template(tmp_path):
    import shutil
    import zipfile
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    build = tmp_path / "build-source"
    build.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copyfile(repo / name, build / name)
    for path in (repo / "hyprover").rglob("*.py"):
        relative = path.relative_to(repo)
        if any(part in ("tests", "runs", "__pycache__", "agent") for part in relative.parts):
            continue
        target = build / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copytree(repo / "hyprover/article/templates", build / "hyprover/article/templates")
    for path in (repo / "hyprover/core").glob("*.json"):
        shutil.copyfile(path, build / "hyprover/core" / path.name)
    output = tmp_path / "wheels"
    result = subprocess.run([sys.executable, "-m", "pip", "wheel", str(build), "--no-deps", "--no-index",
        "--no-build-isolation", "--wheel-dir", str(output)], cwd=build, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next(output.glob("hyprover-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "hyprover/article/templates/arxiv_2608_13536v2.tex" in names
        assert "hyprover/article/templates/arxiv_2608_13536v2.json" in names
    check = subprocess.run([sys.executable, "-I", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); from hyprover.article.render import template; print(template()[1]['author'])",
        str(wheel)], cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert check.returncode == 0, check.stderr
    assert check.stdout.strip() == "hyprover"


def test_duplicate_launch_never_waits_for_a_running_worker(project, monkeypatch):
    import fcntl
    identity = {"pid": os.getpid(), "start_ticks": "1"}
    monkeypatch.setattr(runtime, "process_identity", lambda pid: identity if pid == os.getpid() else None)
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    state = read_json(root / "status.json")
    state.update(state="running", identity=identity)
    write_json(root / "status.json", state)
    with (root / ".lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        started = time.monotonic()
        result = runtime.launch(root)
        assert time.monotonic() - started < 1
        assert result["state"] == "running" and result["busy"] and result["reused"]


def test_background_resume_preserves_new_cancel_request(project, monkeypatch):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    runtime.stop(project.root, root.name)
    class Process:
        pid = 98765
        def __init__(self, *args, **kwargs):
            runtime.stop(project.root, root.name)
        def wait(self):
            return 0
    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    monkeypatch.setattr(runtime, "process_identity", lambda _: None)
    runtime.launch(root, resume=True)
    agent = FakeAgent(evidence.load(root))
    result = runtime.work(project.root, root.name, resume=True, wait_for_launch=True, agent=agent)
    assert result["state"] == "cancelled" and not agent.calls
    assert (root / "cancel.json").exists()


def test_auto_disabled_has_no_evidence_io_or_spawn(project, monkeypatch):
    monkeypatch.setenv("HYPROVER_ARTICLE_AUTO", "0")
    def forbidden(*args, **kwargs):
        pytest.fail("disabled automatic article performed work")
    monkeypatch.setattr(runtime, "_metadata", forbidden)
    monkeypatch.setattr(runtime, "prepare", forbidden)
    monkeypatch.setattr(runtime, "launch", forbidden)
    assert runtime.maybe_start(project.root)["state"] == "disabled"
    assert not (project.root / "articles").exists()


def test_project_switch_default_and_sequence_defer(project, monkeypatch):
    monkeypatch.delenv("HYPROVER_ARTICLE_AUTO", raising=False)
    path = project.root / "project.json"
    metadata = read_json(path)
    metadata["article"] = {"auto_generate": False}
    write_json(path, metadata)
    assert runtime.maybe_start(project.root)["state"] == "disabled"
    monkeypatch.setenv("HYPROVER_SEQUENCE_STAGE", "fixture-stage")
    assert runtime.maybe_start(project.root)["state"] == "deferred_to_sequence"
    assert not (project.root / "articles").exists()


def test_prepare_is_idempotent_and_stale_evidence_gets_new_run(project):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    assert runtime.prepare(project.root, [fid]) == root
    before = (root / "evidence.json").read_bytes()
    project.set_external_refs(fid, [{"title": "Fixture corrected bibliography"}])
    assert runtime.status(project.root, root.name)["state"] == "stale"
    other = runtime.prepare(project.root, [fid])
    assert other != root and (root / "evidence.json").read_bytes() == before
    with pytest.raises(EvidenceChanged):
        runtime.launch(root)


def test_new_time_defaults_preserve_frozen_job_budget_on_resume(project, monkeypatch):
    fid = publish(project)
    old_config = ArticleConfig(timeout_seconds=600, total_seconds=7200, source_timeout=20)
    old_root = runtime.prepare(project.root, [fid], config=old_config)
    original = (old_root / "manifest.json").read_bytes()
    new_root = runtime.prepare(project.root, [fid])
    assert new_root != old_root
    current = runtime.load_manifest(new_root)["config"]
    assert (current["timeout_seconds"], current["total_seconds"], current["source_timeout"]) == (1800, 21600, 60)
    configs = []
    pipeline = runtime.Pipeline

    def record_config(root, frozen, config, **kwargs):
        configs.append(config)
        return pipeline(root, frozen, config, **kwargs)

    monkeypatch.setattr(runtime, "Pipeline", record_config)
    runtime.stop(project.root, old_root.name)
    state = runtime.work(project.root, old_root.name, resume=True, agent=FakeAgent(evidence.load(old_root)))
    assert state["state"] == "ready", state
    assert configs == [old_config]
    assert (old_root / "manifest.json").read_bytes() == original


def test_foreground_work_and_status_validate_outputs(project):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    frozen = evidence.load(root)
    agent = FakeAgent(frozen)
    state = runtime.work(project.root, root.name, agent=agent)
    assert state["state"] == "ready", state
    assert runtime.status(project.root, root.name)["state"] == "ready"
    count = len(agent.calls)
    assert runtime.work(project.root, root.name, agent=agent)["state"] == "ready"
    assert len(agent.calls) == count
    (root / "paper.tex").write_text((root / "paper.tex").read_text() + "Modified")
    assert runtime.status(project.root, root.name)["state"] == "stale"


def test_cancellation_and_explicit_resume(project):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    agent = FakeAgent(evidence.load(root))
    assert runtime.stop(project.root, root.name)["cancel_requested"]
    assert runtime.work(project.root, root.name, agent=agent)["state"] == "cancelled"
    assert agent.calls == []
    state = runtime.work(project.root, root.name, resume=True, agent=agent)
    assert state["state"] == "ready", state
    assert not (root / "cancel.json").exists()


def test_launch_detaches_removes_stage_tags_and_deduplicates(project, monkeypatch):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    spawned = []
    class Process:
        pid = 98765
        def __init__(self, command, **options):
            spawned.append((command, options))
        def wait(self):
            return 0
    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    monkeypatch.setattr(runtime, "process_identity", lambda pid: {"pid": pid, "start_ticks": "fixture"})
    monkeypatch.setenv("HYPROVER_SEQUENCE_STAGE", "stage-to-clean")
    monkeypatch.setenv("HYPROVER_RUN_DEADLINE", "1")
    assert runtime.launch(root)["state"] == "queued"
    assert runtime.launch(root)["reused"]
    assert len(spawned) == 1
    cmd, options = spawned[0]
    assert "hyprover.article" in cmd and options["start_new_session"]
    assert "HYPROVER_SEQUENCE_STAGE" not in options["env"]
    assert "HYPROVER_RUN_DEADLINE" not in options["env"]
    assert options["env"]["HYPROVER_LITERATURE_SEARCH"] == "0"


def test_sequence_settings_preserve_restrictions(project, monkeypatch):
    fid = publish(project)
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "1")
    (project.root / ".sequence.json").write_text("{}")
    (project.root / "run-settings.json").write_text(json.dumps({"HYPROVER_LITERATURE_SEARCH": "0", "HYPROVER_ARTICLE_MODEL": "fixture-model"}))
    root = runtime.prepare(project.root, [fid])
    frozen = evidence.load(root)
    assert frozen["permissions"]["literature_search"] is False
    assert runtime.load_manifest(root)["config"]["model"] == "fixture-model"
    state = runtime.work(project.root, root.name, agent=FakeAgent(frozen))
    assert state["state"] == "ready", state
    assert runtime.status(project.root, root.name)["state"] == "ready"


def test_auto_failure_never_changes_completion(project, monkeypatch):
    fid = publish(project)
    complete(project, [fid])
    before = (project.root / "main/result.json").read_bytes()
    def fail(*args, **kwargs):
        raise OSError("fixture spawn failure")
    monkeypatch.setattr(runtime, "launch", fail)
    assert runtime.maybe_start(project.root)["state"] == "failed"
    assert (project.root / "main/result.json").read_bytes() == before


def test_cli_status_and_unknown_tasks_do_not_create_directories(project, capsys):
    assert main(["status", str(project.root)]) == 0
    assert json.loads(capsys.readouterr().out) == {"jobs": []}
    assert not (project.root / "articles").exists()
    assert main(["resume", str(project.root), "--run-id", "../bad"]) == 2
    assert "invalid_article_run_id" in capsys.readouterr().err


def test_missing_api_background_task_fails_without_real_model(project, monkeypatch):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    monkeypatch.setenv("CODEX_BACKEND", "chatgpt")
    state = runtime.launch(root)
    assert state["state"] == "queued"
    until = time.monotonic() + 10
    while time.monotonic() < until:
        current = read_json(root / "status.json")
        if current["state"] == "failed":
            assert "tool_free_responses_api" in current["error"]
            break
        time.sleep(0.05)
    else:
        runtime.stop(project.root, root.name)
        pytest.fail("background worker did not report its unsupported backend")


def test_status_detects_tampered_source_and_traversal(project):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    manifest = read_json(root / "manifest.json")
    manifest["config"]["max_calls"] = 400
    write_json(root / "manifest.json", manifest)
    assert runtime.status(project.root, root.name)["state"] == "stale"


def test_per_run_restriction_survives_an_enabled_environment(project, monkeypatch):
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "1")
    fid = publish(project)
    override = {"HYPROVER_LITERATURE_SEARCH": "0"}
    root = runtime.prepare(project.root, [fid], overrides=override)
    assert runtime.status(project.root, root.name)["state"] == "pending"

    class Process:
        pid = 12345

        def __init__(self, command, *, env, **kwargs):
            assert env["HYPROVER_LITERATURE_SEARCH"] == "0"

        def wait(self):
            return 0

    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    monkeypatch.setattr(runtime, "process_identity", lambda _: None)
    assert runtime.launch(root, overrides=override)["state"] == "queued"
    frozen = evidence.load(root)
    assert frozen["permissions"]["literature_search"] is False
    assert runtime.work(project.root, root.name, agent=FakeAgent(frozen), wait_for_launch=True)["state"] == "ready"
    assert runtime.status(project.root, root.name)["state"] == "ready"


def test_global_retrieval_disable_blocks_execution_but_preserves_saved_policy(project, monkeypatch):
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "1")
    root = runtime.prepare(project.root, [publish(project)])
    assert evidence.load(root)["permissions"]["literature_search"]
    before = (root / "status.json").read_bytes()
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "0")
    with pytest.raises(ArticleError, match="article_literature_policy_disallows_frozen_run"):
        runtime.launch(root, overrides={"HYPROVER_LITERATURE_SEARCH": "1"})
    with pytest.raises(ArticleError, match="article_literature_policy_disallows_frozen_run"):
        runtime.work(project.root, root.name)
    assert runtime.status(project.root, root.name)["state"] == "pending"
    assert (root / "status.json").read_bytes() == before


def test_backend_endpoint_is_bound_without_storing_credentials_or_url(project, monkeypatch):
    monkeypatch.setenv("CODEX_BACKEND", "api")
    monkeypatch.setenv("CODEX_API_BASE_URL", "https://first.example.test/fixture-route/v1/")
    monkeypatch.setenv("HYPROVER_CODEX_API_KEY", "fixture-credential-not-real")
    config = ArticleConfig(api_backend="codex")
    root = runtime.prepare(project.root, [publish(project)], config=config)
    manifest_bytes = (root / "manifest.json").read_bytes()
    assert b"first.example.test" not in manifest_bytes and b"fixture-credential" not in manifest_bytes
    profile = runtime.load_manifest(root)["backend_profile"]
    assert profile["provider"] == "codex" and len(profile["endpoint_sha256"]) == 64

    monkeypatch.setenv("CODEX_API_BASE_URL", "https://second.example.test/v1")
    with pytest.raises(ArticleError, match="article_backend_route_changed"):
        runtime.launch(root)
    with pytest.raises(ArticleError, match="article_backend_route_changed"):
        runtime.work(project.root, root.name)
    assert runtime.status(project.root, root.name)["state"] == "pending"

    monkeypatch.setenv("CODEX_API_BASE_URL", "https://first.example.test/fixture-route/v1")
    monkeypatch.setenv("HYPROVER_CODEX_API_KEY", "rotated-fixture-credential-not-real")
    assert runtime.work(project.root, root.name, agent=FakeAgent(evidence.load(root)))["state"] == "ready"


def test_legacy_route_binding_preserves_manifest_and_reports_historical_gap(project, monkeypatch):
    monkeypatch.setenv("CODEX_BACKEND", "api")
    monkeypatch.setenv("CODEX_API_BASE_URL", "https://first.example.test/v1")
    config = ArticleConfig(api_backend="codex")
    root = runtime.prepare(project.root, [publish(project)], config=config)
    current = runtime.load_manifest(root)
    expected_profile = current["backend_profile"]
    historical = {key: value for key, value in current.items()
                  if key not in ("run_id", "fingerprint", "backend_profile")}
    fingerprint = digest(historical)
    old_id = "article-" + fingerprint[:24]
    historical.update(fingerprint=fingerprint, run_id=old_id)
    older = runtime.job_path(project.root, old_id)
    root.rename(older)
    write_json(older / "manifest.json", historical)
    state = read_json(older / "status.json")
    write_json(older / "status.json", dict(state, run_id=old_id))
    saved = (older / "manifest.json").read_bytes()

    assert runtime.status(project.root, old_id)["state"] == "pending"
    assert runtime._execution_profile(older, historical, config, runtime.environment(project.root),
                                      bind_legacy=True) == expected_profile
    assert read_json(older / "backend-binding.json") == {"manifest_fingerprint": fingerprint,
        "profile": expected_profile, "historical_routes": "unverified"}
    assert (older / "manifest.json").read_bytes() == saved

    monkeypatch.setenv("CODEX_API_BASE_URL", "https://second.example.test/v1")
    with pytest.raises(ArticleError, match="article_backend_route_changed"):
        runtime.launch(older)


@pytest.mark.parametrize("completed", [False, True])
def test_relocated_job_preserves_ids_and_revalidates_actual_evidence(project, tmp_path, completed):
    from hyprover.core import KnowledgeStore

    fid = publish(project)
    verification = complete(project, [fid]) if completed else None
    root = runtime.prepare(project.root, None if completed else [fid])
    frozen = evidence.load(root)
    writer = FakeAgent(frozen)
    assert runtime.work(project.root, root.name, agent=writer)["state"] == "ready"
    copies = {name: (root / name).read_bytes() for name in ("manifest.json", "evidence.json")}
    restored_project = tmp_path / "restored-project"
    shutil.move(project.root, restored_project)
    restored_root = runtime.job_path(restored_project, root.name)
    if verification is not None:
        (restored_project / "verify-runs").mkdir()
        shutil.move(verification, restored_project / "verify-runs" / verification.name)
    assert runtime.status(restored_project, root.name)["state"] == "ready"
    restored_writer = FakeAgent(frozen)
    assert runtime.work(restored_project, root.name, resume=True, agent=restored_writer)["state"] == "ready"
    assert not restored_writer.calls
    assert {name: (restored_root / name).read_bytes() for name in copies} == copies

    KnowledgeStore(restored_project).revoke(fid, "Synthetic invalidation fixture.")
    assert runtime.status(restored_project, root.name)["state"] == "stale"


def test_copy_cannot_cancel_an_owner_at_original_location(project, tmp_path, monkeypatch):
    root = runtime.prepare(project.root, [publish(project)])
    identity = {"pid": 12345, "start_ticks": "1"}
    saved = read_json(root / "status.json")
    write_json(root / "status.json", dict(saved, state="running", identity=identity))
    monkeypatch.setattr(runtime, "process_identity", lambda pid: identity)
    monkeypatch.setattr(runtime.os, "pidfd_open", lambda *args: pytest.fail("cannot signal the old location"), raising=False)
    duplicate = tmp_path / "copied-project"
    shutil.copytree(project.root, duplicate)
    assert runtime.status(duplicate, root.name)["state"] == "interrupted"
    assert runtime.stop(duplicate, root.name)["cancel_requested"]
    assert not (root / "cancel.json").exists()


def test_repeated_dispatch_cannot_start_or_resume_terminal_job(project, monkeypatch):
    root = runtime.prepare(project.root, [publish(project)])
    dispatch = "1" * 32
    writer = FakeAgent(evidence.load(root))
    state = runtime.work(project.root, root.name, resume=True, dispatch_id=dispatch, agent=writer)
    assert state["state"] == "ready" and state["dispatch_id"] == dispatch
    saved = (root / "status.json").read_bytes()
    call_count = len(writer.calls)

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("dispatch already finished"))
    assert runtime.launch(root, resume=True, dispatch_id=dispatch)["state"] == "ready"
    assert runtime.work(project.root, root.name, resume=True, dispatch_id=dispatch, agent=writer)["state"] == "ready"
    assert (root / "status.json").read_bytes() == saved and len(writer.calls) == call_count
    for run_id in ("../../outside", "article-not-a-hash", "/tmp/file"):
        with pytest.raises(ArticleError):
            runtime.job_path(project.root, run_id)
