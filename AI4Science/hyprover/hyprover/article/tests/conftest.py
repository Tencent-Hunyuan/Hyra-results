"""Engineering fixtures only; no model calls or mathematical certification."""
import json
import shutil
from pathlib import Path

import pytest

from hyprover.core import Fact, KnowledgeStore
from hyprover.core.schema import KNOWLEDGE_FORMAT
from hyprover.verify import completion_evidence as ce


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "0")
    root = tmp_path / "project"
    store = KnowledgeStore.initialize(root, "fixture-problem")
    (root / "project.json").write_text(json.dumps({"knowledge_format": KNOWLEDGE_FORMAT,
        "research_mode": "standard", "allow_math_computation": False}))
    (root / "PROBLEM.md").write_text("Establish the fixture target.")
    return store


def publish(store, statement="Fixture target.", proof="Fixture proof.", predecessors=None, refs=None, definitions=None):
    return store.publish(Fact("", "fixture-problem", "fixture-worker", predecessors or [], statement,
                              proof, definitions or {}, external_refs=refs or []),
                         {"verdict": "correct", "run_id": "engineering-fixture-only"})


def prepare_legacy_review(project: Path, targets=None, **options) -> Path:
    """Use a manifest from before new jobs selected a local review mode."""
    from hyprover.article import runtime
    from hyprover.article.models import digest, read_json, write_json

    current = runtime.prepare(project, targets, **options)
    inputs = {key: value for key, value in read_json(current / "manifest.json").items()
              if key not in ("review_mode", "proof_repair_revision", "run_id", "fingerprint")}
    fingerprint = digest(inputs)
    run_id = "article-" + fingerprint[:24]
    root = runtime.job_path(project, run_id)
    shutil.copytree(current, root)
    write_json(root / "manifest.json", dict(inputs, run_id=run_id, fingerprint=fingerprint))
    write_json(root / "status.json", dict(read_json(root / "status.json"), run_id=run_id))
    return root


def complete(store, targets):
    manifest, bodies = ce.capture(store.root, {"target_fact_ids": targets, "summary": "Fixture bridge."},
                                  {"chunk_chars": 256})
    root = store.root.parent / "verify-runs" / ("completion-" + manifest["digest"])
    ce.persist(root, manifest, bodies)
    receipts = {}
    for job in ce.jobs(manifest):
        report = {"verdict": "correct", "repair_hints": "", "verification_report": {
            "summary": "Engineering fixture.", "critical_errors": [], "gaps": []},
            "coverage": dict(ce.expected_coverage(manifest, job), uncovered=[], materials_complete=True, truncated=False)}
        receipts[job["id"]] = {"run_id": "fixture-" + job["id"], "report": report}
    ce.write_json(root / "completion-progress.json", {"format": ce.FORMAT,
                  "evidence_digest": manifest["digest"], "receipts": receipts})
    ce.write_json(root / "completion-report.json", {"format": ce.FORMAT,
                  "evidence_digest": manifest["digest"], "outcome": "passed", "accepted": True,
                  "coverage_verification_run": "fixture-bridge"})
    (store.root / "main").mkdir(exist_ok=True)
    ce.write_json(store.root / "main/result.json", {"state": "completed", "target_fact_ids": targets,
        "summary": "Fixture bridge.", "formal_certificate": False, "full_revalidation_run": root.name,
        "completion_report": str(root / "completion-report.json"),
        "completion_evidence_digest": manifest["digest"], "coverage_verification_run": "fixture-bridge"})
    return root
