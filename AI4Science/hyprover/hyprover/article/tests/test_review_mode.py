import copy
import subprocess
import sys
from pathlib import Path

import pytest

from hyprover.article import audit, delivery, evidence, render, response_validation, runtime
from hyprover.article.models import ArticleError, read_json
from .conftest import prepare_legacy_review, publish
from .test_audit import FakeAgent
from .test_publication import fixture


def test_new_and_legacy_jobs_have_distinct_manifests_and_review_tasks(project):
    fact_id = publish(project)
    root = runtime.prepare(project.root, [fact_id])
    strict_root = prepare_legacy_review(project.root, [fact_id])
    assert root != strict_root
    agent = FakeAgent(evidence.load(root))
    strict_agent = FakeAgent(evidence.load(strict_root))

    result = runtime.work(project.root, root.name, agent=agent)
    assert result["state"] == "ready", result
    assert "compose_paper" in agent.calls
    assert not any(task.startswith(("review_", "revise_", "plan_review_")) for task in agent.calls)
    assert runtime.status(project.root, root.name)["review_mode"] == audit.FORMAT_COVERAGE
    result = runtime.work(project.root, strict_root.name, agent=strict_agent)
    assert result["state"] == "ready", result
    assert "review_whole" in strict_agent.calls
    assert runtime.status(project.root, strict_root.name)["review_mode"] == audit.STRICT


def test_relaxed_composition_does_not_require_editorial_roadmap(project):
    frozen, ledger, _, plan, article = fixture(project)
    article = copy.deepcopy(article)
    removed = article["sections"][0]["blocks"].pop()["id"]
    for row in article["source_map"]:
        row["blocks"] = [block for block in row["blocks"] if block != removed]
    material = {"evidence": frozen, "ledger": ledger, "publication_plan": plan}
    with pytest.raises(ArticleError, match="requires_concrete_roadmap"):
        response_validation.validate("compose_paper", {"article": article}, material)
    response_validation.validate("compose_paper", {"article": article},
                                 dict(material, review_mode=audit.FORMAT_COVERAGE))


def test_local_article_command_loads_without_optional_http_service(project):
    root = runtime.prepare(project.root, [publish(project)])
    repo = Path(__file__).resolve().parents[3]
    process = subprocess.run(
        [sys.executable, "-S", "-m", "hyprover.orchestration", "article", "status", str(project.root),
         "--run-id", root.name],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    assert '"review_mode": "format-coverage"' in process.stdout


def test_relaxed_layout_failure_blocks_without_automatic_revision(project, monkeypatch):
    root = runtime.prepare(project.root, [publish(project)])
    agent = FakeAgent(evidence.load(root))
    monkeypatch.setattr(render, "compile_pdf", lambda *_args, **_kwargs: {"compiled": True, "max_overfull_pt": 40})
    result = runtime.work(project.root, root.name, agent=agent, compile_output=True)
    assert result["state"] == "needs_review", result
    assert read_json(root / "audit.json")["typesetting_findings"]
    assert not any(task.startswith(("review_", "revise_")) for task in agent.calls)


def test_recheck_respects_lock_preserves_manuscript_and_persists_mode(project, monkeypatch):
    root = prepare_legacy_review(project.root, [publish(project)])
    agent = FakeAgent(evidence.load(root), truncated=True)
    original_status = runtime.work(project.root, root.name, agent=agent)
    assert original_status["state"] == "needs_review", original_status
    before = {name: (root / name).read_bytes() for name in (
        "paper.tex", "article.json", "references.json", "evidence.json", "checkpoint.json", "manifest.json")}
    original_audit = (root / "audit.json").read_bytes()

    def forbid(*_args, **_kwargs):
        pytest.fail("Recheck must not start a writing pipeline")

    monkeypatch.setattr(runtime, "Pipeline", forbid)
    with runtime.locked(root / ".lock"):
        assert runtime.recheck(project.root, root.name)["busy"]
    result = runtime.recheck(project.root, root.name)
    assert result["state"] == "ready", result
    assert runtime.status(project.root, root.name)["review_mode"] == audit.FORMAT_COVERAGE
    assert delivery.export(project.root, root.name)["review_mode"] == audit.FORMAT_COVERAGE
    assert runtime.work(project.root, root.name, resume=True)["state"] == "ready"
    assert original_audit in [path.read_bytes() for path in (root / "rechecks").glob("*/audit.json")]
    assert {name: (root / name).read_bytes() for name in before} == before
    (root / "paper.tex").write_bytes(before["paper.tex"] + b"\nmodified")
    assert runtime.status(project.root, root.name)["state"] == "stale"
