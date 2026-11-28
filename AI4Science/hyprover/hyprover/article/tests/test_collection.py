import json
import sys
from pathlib import Path

import pytest

from hyprover.article import collection


@pytest.mark.parametrize("entrypoint", ["function", "cli"])
@pytest.mark.parametrize("timeout", [None, 2400])
def test_collection_skips_removed_items_and_preserves_completed_tasks(tmp_path, monkeypatch, entrypoint, timeout):
    from hyprover.article.models import ArticleConfig

    (tmp_path / "export-manifest.json").write_text(json.dumps({"projects": [
        {"benchmark": "fixture-beta", "directory": "fixture-beta/"},
        {"benchmark": "fixture-alpha", "directory": "fixture-alpha/"}]}))
    for folder, model_timeout, source_budget in (("fixture-alpha", 120, 5), ("fixture-beta", 960, 8)):
        project = tmp_path / folder
        project.mkdir()
        (project / "project.json").write_text(json.dumps({"article": {"timeout_seconds": model_timeout,
            "max_sources": source_budget, "max_input_chars": 65432, "total_seconds": 7200}}))
    monkeypatch.delenv("HYPROVER_ARTICLE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("HYPROVER_ARTICLE_MAX_INPUT_CHARS", raising=False)
    states, launches, configs = {}, [], []
    def prepare(project, **kwargs):
        configs.append(kwargs["config"].to_dict())
        states.setdefault(project.name, "pending")
        return project / "articles" / "fixture-run"
    def status(project, run_id):
        return {"state": states[project.name]}
    def launch(root):
        name = root.parents[1].name
        launches.append(name)
        states[name] = "ready"
        return {"state": "ready"}
    monkeypatch.setattr(collection.runtime, "prepare", prepare)
    monkeypatch.setattr(collection.runtime, "status", status)
    monkeypatch.setattr(collection.runtime, "job_path", lambda project, run_id: project / "articles" / run_id)
    monkeypatch.setattr(collection.runtime, "launch", launch)
    monkeypatch.setattr(collection.time, "sleep", lambda _: None)
    def generate():
        if entrypoint == "cli":
            args = ["article-collection", str(tmp_path), "--jobs", "1"]
            if timeout is not None:
                args += ["--timeout", str(timeout)]
            monkeypatch.setattr(sys, "argv", args)
            return collection.main()
        return collection.generate(tmp_path, concurrency=1, **({} if timeout is None else {"timeout": timeout}))
    assert generate() == 0
    assert launches == ["fixture-alpha", "fixture-beta"]
    saved = json.loads((tmp_path / "article-generation.json").read_text())
    assert saved["complete"] and saved["ready"] == 2
    assert "fixture-removed" not in saved["projects"]
    assert saved["timeout_override_seconds"] == timeout
    assert [cfg["timeout_seconds"] for cfg in configs] == ([120, 960] if timeout is None else [timeout, timeout])
    assert [cfg["max_sources"] for cfg in configs] == [5, 8]
    assert all(cfg["total_seconds"] == 7200 and cfg["max_input_chars"] == 65432
               and cfg["source_timeout"] == ArticleConfig().source_timeout for cfg in configs)
    assert generate() == 0
    assert len(launches) == 2


def test_resume_keeps_job_ids_and_does_not_automatically_retry_interrupted(tmp_path, monkeypatch):
    from hyprover.article.models import ArticleConfig, FORMAT
    config = ArticleConfig(timeout_seconds=1800, total_seconds=86400).to_dict()
    original = {"projects": [{"benchmark": "fixture-beta", "directory": "fixture-beta/"}]}
    (tmp_path / "export-manifest.json").write_text(json.dumps(original))
    state = {"format": FORMAT, "config": config, "concurrency": 1, "projects": {
        "fixture-beta": {"directory": "fixture-beta/", "run_id": "fixture-existing", "state": "running"}}}
    (tmp_path / "article-generation.json").write_text(json.dumps(state))
    monkeypatch.setattr(collection.runtime, "job_path", lambda p, run: p / "articles" / run)
    monkeypatch.setattr(collection.runtime, "load_manifest", lambda p: {})
    monkeypatch.setattr(collection.runtime, "status", lambda *a: {"state": "interrupted"})
    def forbidden(*args, **kwargs):
        pytest.fail("resume must not recreate jobs or blindly retry interrupted work")
    monkeypatch.setattr(collection.runtime, "prepare", forbidden)
    monkeypatch.setattr(collection.runtime, "launch", forbidden)
    assert collection.resume(tmp_path) == 2
    restored = json.loads((tmp_path / "article-generation.json").read_text())
    assert restored["config"] == config
    assert restored["projects"]["fixture-beta"]["run_id"] == "fixture-existing"
    assert restored["projects"]["fixture-beta"]["state"] == "interrupted"


def test_policy_upgrade_waits_for_old_worker_and_keeps_run_and_budget(tmp_path, monkeypatch):
    from hyprover.article.models import FORMAT, ArticleConfig, write_json
    from hyprover.article.citation_policy import POLICY_REVISION
    project = tmp_path / "fixture-beta"
    task = project / "articles/fixture-existing"
    task.mkdir(parents=True)
    write_json(tmp_path / "export-manifest.json", {"projects": [{"benchmark": "fixture-beta", "directory": "fixture-beta/"}]})
    write_json(tmp_path / "article-generation.json", {"format": FORMAT, "config": ArticleConfig().to_dict(),
        "concurrency": 1, "projects": {"fixture-beta": {"directory": "fixture-beta/", "run_id": task.name, "state": "running"}}})
    write_json(task / "status.json", {"state": "running", "identity": {"pid": 123, "start_ticks": "1"}})
    write_json(task / "checkpoint.json", {"calls": 9, "elapsed_seconds": 500})
    live = [True]
    calls = []
    monkeypatch.setattr(collection.runtime, "job_path", lambda *a: task)
    monkeypatch.setattr(collection.runtime, "load_manifest", lambda *a: {})
    monkeypatch.setattr(collection.runtime, "process_identity", lambda _: {"pid": 123, "start_ticks": "1"} if live[0] else None)
    monkeypatch.setattr(collection.runtime, "status", lambda *a: json.loads((task / "status.json").read_text()))
    def finish_old(_):
        assert not calls
        live[0] = False
        write_json(task / "status.json", {"state": "needs_review", "identity": None})
    monkeypatch.setattr(collection.time, "sleep", finish_old)
    def launch(path, **kwargs):
        assert path == task and not live[0] and kwargs["resume"] and len(kwargs["dispatch_id"]) == 32
        calls.append(path)
        checkpoint = json.loads((path / "checkpoint.json").read_text())
        assert checkpoint["calls"] == 9 and checkpoint["elapsed_seconds"] == 500
        write_json(path / "checkpoint.json", dict(checkpoint, policy_revision=POLICY_REVISION))
        write_json(path / "status.json", {"state": "ready", "identity": None})
        return {"state": "ready", "dispatch_id": kwargs["dispatch_id"]}
    monkeypatch.setattr(collection.runtime, "launch", launch)
    assert collection.resume(tmp_path, upgrade_policy=True) == 0
    assert calls == [task]


def test_collection_records_failures_without_changing_input(tmp_path, monkeypatch):
    original = {"projects": [{"benchmark": "fixture-beta", "directory": "fixture-beta/"}]}
    (tmp_path / "export-manifest.json").write_text(json.dumps(original))
    def prepare(*args, **kwargs):
        raise ValueError("invalid completion")
    monkeypatch.setattr(collection.runtime, "prepare", prepare)
    assert collection.generate(tmp_path) == 2
    state = json.loads((tmp_path / "article-generation.json").read_text())
    assert state["complete"] and state["projects"]["fixture-beta"]["state"] == "failed"
    assert json.loads((tmp_path / "export-manifest.json").read_text()) == original


@pytest.mark.parametrize("entrypoint", ["function", "cli"])
def test_nine_jobs_are_live_before_any_completion_and_all_export(tmp_path, monkeypatch, entrypoint):
    names = [f"fixture-{index:02d}" for index in range(9, 0, -1)]
    (tmp_path / "export-manifest.json").write_text(json.dumps({"projects": [
        {"benchmark": n, "directory": n + "/"} for n in names]}))
    states = {n: "pending" for n in names}
    launched, exports, cycles = [], [], []
    monkeypatch.setattr(collection.runtime, "prepare", lambda p, **kw: p / "articles/fresh-run")
    monkeypatch.setattr(collection.runtime, "job_path", lambda p, rid: p / "articles" / rid)
    monkeypatch.setattr(collection.runtime, "load_manifest", lambda p: {})
    monkeypatch.setattr(collection.runtime, "status", lambda p, rid: {"state": states[p.name]})

    def launch(path, **options):
        assert options == {"compile_output": True}
        name = path.parents[1].name
        launched.append(name)
        states[name] = "running"
        return {"state": "running"}

    def finish(_):
        assert len(launched) == 9 and set(launched) == set(names)
        assert all(s == "running" for s in states.values())
        cycles.append(1)
        states.update({n: "ready" for n in names})

    def deliver(project, record):
        exports.append(project.name)
        record["compilation"] = {"compiled": True}
        return False

    monkeypatch.setattr(collection.runtime, "launch", launch)
    monkeypatch.setattr(collection.time, "sleep", finish)
    monkeypatch.setattr(collection, "_deliver_pdf", deliver)
    if entrypoint == "cli":
        monkeypatch.setattr(sys, "argv", ["article-collection", str(tmp_path), "--jobs", "9", "--compile"])
        result = collection.main()
    else:
        result = collection.generate(tmp_path, concurrency=9, compile_output=True)
    assert result == 0 and cycles == [1]
    saved = json.loads((tmp_path / "article-generation.json").read_text())
    assert saved["concurrency"] == saved["ready"] == saved["pdf_ready"] == 9
    assert len(exports) == 9 and set(exports) == set(names)
    assert collection.resume(tmp_path, compile_output=True) == 0
    assert len(launched) == 9


@pytest.mark.parametrize("value", [0, 10, -1, True, 7.0, "7", None])
def test_invalid_concurrency_is_rejected_before_preparing_or_writing(tmp_path, monkeypatch, value):
    monkeypatch.setattr(collection.runtime, "prepare", lambda *a, **kw: pytest.fail("invalid input prepared tasks"))
    with pytest.raises(ValueError, match="concurrency_must_be_1_to_9"):
        collection.generate(tmp_path, concurrency=value)
    with pytest.raises(ValueError, match="concurrency_must_be_1_to_9"):
        collection._drive_locked(tmp_path, {"concurrency": value})
    assert not (tmp_path / "article-generation.json").exists()


@pytest.mark.parametrize("value", [0, 10, -1, True, 7.0, "7", None])
def test_resume_rejects_invalid_concurrency_before_manifests_or_mutations(tmp_path, monkeypatch, value):
    state_path = tmp_path / "article-generation.json"
    original = json.dumps({"concurrency": value})
    state_path.write_text(original)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid resume reached operation dispatch")

    monkeypatch.setattr(collection.runtime, "load_manifest", forbidden)
    monkeypatch.setattr(collection, "_drive_locked", forbidden)
    with pytest.raises(ValueError, match="concurrency_must_be_1_to_9"):
        collection.resume(tmp_path)
    assert state_path.read_text() == original


def test_concurrent_collection_cannot_overwrite_driver_state(tmp_path, monkeypatch):
    original = b'{"owner":"active driver checkpoint"}'
    state = tmp_path / "article-generation.json"
    state.write_bytes(original)
    monkeypatch.setattr(collection.runtime, "prepare", lambda *a, **kw: pytest.fail("busy collection prepared tasks"))
    with collection.runtime.locked(tmp_path / ".article-collection.lock"):
        for entrypoint in (collection.generate, collection.resume):
            with pytest.raises(ValueError, match="article_collection_busy"):
                entrypoint(tmp_path)
    assert state.read_bytes() == original


@pytest.mark.parametrize("worker_committed", [False, True])
def test_collection_recovers_interrupted_dispatch_without_duplicate_launch(tmp_path, monkeypatch, worker_committed):
    from hyprover.article.models import FORMAT, write_json

    class PowerCut(BaseException):
        pass

    task = tmp_path / "P/articles/run"
    task.mkdir(parents=True)
    write_json(tmp_path / "export-manifest.json", {"projects": [{"benchmark": "P", "directory": "P"}]})
    state = {"format": FORMAT, "concurrency": 1, "retry_failed": True,
             "projects": {"P": {"directory": "P", "run_id": "run", "state": "failed"}}}
    (task / "status.json").write_text('{"state":"failed"}')
    write_json(tmp_path / "article-generation.json", state)
    monkeypatch.setattr(collection.runtime, "load_manifest", lambda *args: {})
    monkeypatch.setattr(collection.runtime, "job_path", lambda *args: task)
    status = {"state": "failed"}
    monkeypatch.setattr(collection.runtime, "status", lambda *args: status)
    attempts = []

    def interrupt(path, *, resume, dispatch_id):
        attempts.append(dispatch_id)
        if worker_committed:
            status.update(state="ready", dispatch_id=dispatch_id)
        raise PowerCut()

    monkeypatch.setattr(collection.runtime, "launch", interrupt)
    with pytest.raises(PowerCut):
        collection.resume(tmp_path)
    persisted = json.loads((tmp_path / "article-generation.json").read_text())["projects"]["P"]
    assert persisted["dispatch_intent"]["id"] == attempts[0] and "recovery_attempted" not in persisted

    def launch(path, *, resume, dispatch_id):
        assert not worker_committed and dispatch_id == attempts[0]
        attempts.append(dispatch_id)
        status.update(state="ready", dispatch_id=dispatch_id)
        return dict(status)

    monkeypatch.setattr(collection.runtime, "launch", launch)
    assert collection.resume(tmp_path) == 0
    assert len(attempts) == (1 if worker_committed else 2)
    persisted = json.loads((tmp_path / "article-generation.json").read_text())["projects"]["P"]
    assert persisted["recovery_attempted"] == collection.RECOVERY_REVISION and "dispatch_intent" not in persisted
