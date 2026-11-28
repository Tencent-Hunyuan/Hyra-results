import copy
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from hyprover.article import collection, delivery, evidence, render, runtime
from hyprover.article.__main__ import main
from hyprover.article.models import ArticleConfig, FORMAT, read_json, write_json
from .conftest import prepare_legacy_review, publish
from .test_audit import FakeAgent


@pytest.mark.parametrize("compiled", [True, False])
def test_worker_stays_running_until_compilation_finishes(project, monkeypatch, compiled):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid], config=ArticleConfig(max_revisions=0))
    seen = []
    monkeypatch.setattr(runtime, "process_identity", lambda pid: {"pid": pid, "start_ticks": "1"})

    def compile_pdf(path, *, deadline, cancelled):
        assert not cancelled() and deadline > time.monotonic()
        state = read_json(path / "status.json")
        assert state["state"] == "running"
        assert state["phase"] == "compilation"
        assert state["identity"] is not None
        seen.append(state)
        return {"compiled": compiled}

    monkeypatch.setattr(render, "compile_pdf", compile_pdf)
    state = runtime.work(project.root, root.name, agent=FakeAgent(evidence.load(root)), compile_output=True)
    assert len(seen) == 1
    assert state["state"] == ("ready" if compiled else "needs_review") and state["phase"] == "finished"
    assert state["identity"] is None and state["compilation"]["compiled"] is compiled


def test_existing_escape_repair_archives_draft_and_invalidates_old_reviews(project, monkeypatch):
    fid = publish(project, proof=r"Use \(\frac{1}{2}\).")
    root = prepare_legacy_review(project.root, [fid])
    frozen = evidence.load(root)
    fake = FakeAgent(frozen)
    runtime.work(project.root, root.name, agent=fake)
    article = read_json(root / "article.json")
    part = next(b for b in article["sections"][3]["blocks"] if b["kind"] == "proof")["parts"][0]
    part["text"] = part["text"].replace(r"\frac", "\frac", 1)
    write_json(root / "article.json", article)
    (root / "paper.tex").write_text((root / "paper.tex").read_text().replace(r"\frac", "\frac", 1))
    state = read_json(root / "status.json")
    state["state"] = "needs_review"
    write_json(root / "status.json", state)
    monkeypatch.setattr(render, "compile_pdf", lambda root, **_: {"compiled": True, "path": str(root / "paper.pdf")})
    result = runtime.compile_article(project.root, root.name)
    assert result["state"] == "needs_review"
    assert result["compilation"]["compiled"]
    repaired = read_json(root / "article.json")
    assert next(b for b in repaired["sections"][3]["blocks"] if b["kind"] == "proof")["parts"][0]["text"] == r"Use \(\frac{1}{2}\)."
    assert "\f" not in (root / "paper.tex").read_text()
    assert read_json(root / "audit.json")["receipts"] == {}
    assert read_json(root / "typesetting-repair.json")["review_required"]
    assert list((root / "typesetting-revisions").glob("*/article.json"))
    assert evidence.load(root) == frozen


@pytest.mark.parametrize("compilation", [
    {"compiled": False, "error": "fixture_compile_failed"},
    {"compiled": True, "max_overfull_pt": 20},
])
@pytest.mark.parametrize("editorial_passed", [True, False])
def test_standalone_compilation_blocks_readiness_and_can_recover_layout_only(
        project, monkeypatch, compilation, editorial_passed):
    root = prepare_legacy_review(project.root, [publish(project)], config=ArticleConfig(max_revisions=0))
    agent = FakeAgent(evidence.load(root), truncated=not editorial_passed)
    expected = "ready" if editorial_passed else "needs_review"
    assert runtime.work(project.root, root.name, agent=agent)["state"] == expected
    original_report = read_json(root / "audit.json")
    results = iter((compilation, {"compiled": True, "max_overfull_pt": 8}))
    monkeypatch.setattr(render, "compile_pdf", lambda *_, **__: next(results))

    bad_result = runtime.compile_article(project.root, root.name)

    assert bad_result["state"] == "needs_review"
    assert bad_result["error"] == "typesetting_requires_revision"
    assert runtime.status(project.root, root.name)["state"] == "needs_review"
    assert read_json(root / "draft-status.json")["review_state"] == "needs_review"
    assert read_json(root / "audit.json")["failures"][-1] == "typesetting_requires_revision"
    assert read_json(root / "layout-review.json")["findings"][0]["severity"] == "blocking"
    assert delivery.export(project.root, root.name)["state"] == "needs_review"

    good_result = runtime.compile_article(project.root, root.name)

    assert good_result["state"] == expected
    assert runtime.status(project.root, root.name)["state"] == expected
    assert read_json(root / "draft-status.json")["review_state"] == expected
    assert read_json(root / "audit.json") == original_report
    assert read_json(root / "layout-review.json")["findings"][0]["severity"] == "advisory"


def test_pdf_file_must_not_be_arbitrary_compiler_output(tmp_path, monkeypatch):
    for name in ("paper.tex", "references.tex"):
        (tmp_path / name).write_text("Fixture input")
    monkeypatch.setattr(render.shutil, "which", lambda name: "/synthetic/" + name)

    def bad_compiler(command, budget, **kwargs):
        work = Path(command[command.index("--bind") + 1])
        (work / "paper.pdf").write_text("Not a PDF")
        (work / "paper.log").write_text("Output written on paper.pdf (1 page, 9 bytes).")
        return 0

    monkeypatch.setattr(render, "run_command", bad_compiler)
    assert render.compile_pdf(tmp_path)["error"] == "invalid_tex_output"
    assert not (tmp_path / "paper.pdf").exists()


def test_background_compile_is_passed_to_worker(project, monkeypatch):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    calls = []

    class Process:
        pid = 998877

        def __init__(self, command, **kwargs):
            calls.append(command)

        def wait(self):
            return 0

    monkeypatch.setattr(runtime.subprocess, "Popen", Process)
    monkeypatch.setattr(runtime, "process_identity", lambda _: None)
    result = runtime.launch(root, compile_output=True)
    assert result["state"] == "queued"
    assert "--compile" in calls[0]


def test_typesetting_repair_gets_one_review_attempt_per_article(tmp_path, monkeypatch):
    task = tmp_path / "P/articles/run"
    task.mkdir(parents=True)
    write_json(task / "typesetting-repair.json", {"article_digest": "repaired-article", "review_required": True})
    state = {"concurrency": 1, "compile_output": True, "projects": {
        "P": {"directory": "P", "run_id": "run", "state": "needs_review"}}}
    calls = []
    monkeypatch.setattr(runtime, "job_path", lambda *args: task)
    monkeypatch.setattr(runtime, "status", lambda *args: {
        "state": "needs_review", "error": "typesetting_repaired_requires_review"})
    def launch(root, *, dispatch_id, **options):
        calls.append(options)
        return {"state": "needs_review", "dispatch_id": dispatch_id}

    monkeypatch.setattr(runtime, "launch", launch)
    monkeypatch.setattr(collection, "_deliver_pdf", lambda *args: False)
    assert collection._drive_locked(tmp_path, state) == 2
    assert calls == [{"resume": True, "compile_output": True}]
    assert collection._drive_locked(tmp_path, state) == 2
    assert len(calls) == 1


def test_old_failed_jobs_resume_once_without_resetting_checkpoint(tmp_path, monkeypatch):
    project = tmp_path / "P"
    task = project / "articles" / "old-run"
    task.mkdir(parents=True)
    checkpoint = {"calls": 19, "elapsed_seconds": 820, "writer_sessions": ["old-session"]}
    write_json(task / "checkpoint.json", checkpoint)
    write_json(tmp_path / "export-manifest.json", {"projects": [{"benchmark": "P", "directory": "P/"}]})
    state = {"format": FORMAT, "config": ArticleConfig().to_dict(), "concurrency": 1, "projects": {
        "P": {"directory": "P/", "run_id": "old-run", "state": "failed"}}, "complete": True}
    write_json(tmp_path / "article-generation.json", state)
    calls = []
    monkeypatch.setattr(runtime, "load_manifest", lambda root: {})
    monkeypatch.setattr(runtime, "job_path", lambda root, run: root / "articles" / run)
    monkeypatch.setattr(runtime, "status", lambda *args: {"state": "failed", "error": "fixture_failure"})

    def launch(root, *, dispatch_id, **kwargs):
        calls.append(kwargs)
        return {"state": "failed", "error": "still_failed", "dispatch_id": dispatch_id}

    monkeypatch.setattr(runtime, "launch", launch)
    assert collection.resume(tmp_path, retry_failed=True, compile_output=True) == 2
    assert calls == [{"resume": True, "compile_output": True}]
    assert collection.resume(tmp_path, retry_failed=True, compile_output=True) == 2
    assert len(calls) == 1
    assert read_json(task / "checkpoint.json") == checkpoint


def test_cancelled_tasks_are_not_implicitly_retried(tmp_path, monkeypatch):
    task = tmp_path / "P/articles/run"
    task.mkdir(parents=True)
    write_json(task / "cancel.json", {"requested_at": 1})
    state = {"concurrency": 1, "retry_failed": True, "projects": {
        "P": {"directory": "P", "run_id": "run", "state": "failed"}}}
    monkeypatch.setattr(runtime, "job_path", lambda p, rid: task)
    monkeypatch.setattr(runtime, "status", lambda *args: {"state": "failed"})
    monkeypatch.setattr(runtime, "launch", lambda *args, **kwargs: pytest.fail("cancelled task restarted"))
    assert collection._drive_locked(tmp_path, state) == 2


def test_each_finished_draft_is_compiled_and_exported_without_waiting_for_batch(tmp_path, monkeypatch):
    record = {"run_id": "run", "state": "needs_review"}
    report = {"compiled": True, "pdf_sha256": "fixture-hash"}
    exports = []
    monkeypatch.setattr(runtime, "job_path", lambda *args: tmp_path)
    monkeypatch.setattr(render, "pdf_sources", lambda root: {"paper.tex": "fixture"})
    monkeypatch.setattr(render, "current_pdf", lambda root: dict(report))
    monkeypatch.setattr(delivery, "export", lambda *args: exports.append(args) or {"pdf_sha256": "fixture-hash"})
    assert not collection._deliver_pdf(tmp_path, record)
    assert not collection._deliver_pdf(tmp_path, record)
    assert len(exports) == 1 and record["state"] == "needs_review"
    assert record["compilation"]["compiled"]


@pytest.mark.skipif(not all(shutil.which(name) for name in ("bwrap", "pdflatex", "prlimit", "pdfinfo", "pdftotext")),
                    reason="Optional isolated TeX and Poppler runtime unavailable")
def test_real_isolated_pdf_and_delivery_preserve_review_state(project, monkeypatch, capsys):
    fid = publish(project)
    root = prepare_legacy_review(project.root, [fid])
    fake = FakeAgent(evidence.load(root), truncated=True)
    state = runtime.work(project.root, root.name, agent=fake)
    assert state["state"] == "needs_review"
    calls = list(fake.calls)
    original_status = copy.deepcopy(state)
    assert main(["compile", str(project.root), "--run-id", root.name]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["state"] == "needs_review" and output["compilation"]["compiled"]
    assert fake.calls == calls
    assert runtime.status(project.root, root.name)["state"] == original_status["state"]
    exported = delivery.export(project.root, root.name)
    assert exported["pdf_compiled"] and exported["state"] == "needs_review"
    pdf = project.root / "paper/paper.pdf"
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True)
    assert "Pages:" in info.stdout and "JavaScript:      no" in info.stdout
    text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    assert "HYPROVER" in text.upper()
    assert exported["author"] == "hyprover" and "affiliation" not in exported
    assert render.current_pdf(root)
    (root / "paper.pdf").write_bytes(b"%PDF-1.5\ncorrupted\n%%EOF\n")
    assert render.current_pdf(root) is None
    assert runtime.status(project.root, root.name)["compilation"]["compiled"] is False
