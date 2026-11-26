"""End-to-end notebook workflow with an in-process, fake independent verifier."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hyprover.core import KnowledgeStore
from hyprover.execution.scaffold import do_new
from hyprover.gateway import server as gateway
from hyprover.verify import service

STATEMENT = "For every integer n, n + 0 equals n."
PROOF = "Zero is the additive identity of the integers, so adding zero leaves every integer n unchanged."
OK = {"verdict": "correct", "verification_report": {"summary": "Synthetic verifier response only.",
      "critical_errors": [], "gaps": []}, "repair_hints": ""}


def prepare(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    monkeypatch.setenv("HYPROVER_AGENTS_ROOT", str(projects))
    made = do_new("workflow", roles="high:1")
    root = Path(made["project_dir"])
    (root / "PROBLEM.md").write_text("# Goal\nProve the additive identity and its iterated application.\n")
    monkeypatch.setenv("HYPROVER_PROJECT_DIR", str(root))
    monkeypatch.setenv("HYPROVER_AUTHOR", "high")
    monkeypatch.setenv("HYPROVER_ROLE", "worker")
    monkeypatch.delenv("HYPROVER_PROBLEM_ID", raising=False)
    monkeypatch.setenv("VERIFIER_RESULTS_DIR", str(tmp_path / "verification-runs"))
    calls = []
    def fake(**kwargs):
        calls.append(kwargs)
        return json.loads(json.dumps(OK))
    monkeypatch.setattr(service, "run_codex_verification", fake)
    client = TestClient(service.app)
    def transport(payload):
        assert "source" not in payload and "source_ref" not in payload
        response = client.post("/verify/fact", json=payload)
        response.raise_for_status()
        return response.json()
    monkeypatch.setattr(gateway, "_verify", transport)
    return root, calls, client


def test_create_conjecture_verify_publish_resume_and_revoke(tmp_path, monkeypatch):
    root, calls, client = prepare(tmp_path, monkeypatch)
    guess = gateway.knowledge_record("conjecture", "Additive identity", STATEMENT,
                                     details="Synthetic exploration context; not a premise.")
    store = KnowledgeStore(root)
    route = store.record("route", "Identity route", "Establish the identity then apply it twice.",
                         "main", "main", status="planned", links=[guess["entry_id"]])
    accepted = gateway.fact_submit(STATEMENT, PROOF, source_id=guess["entry_id"], source_revision=guess["revision"])
    assert accepted["accepted"] is True and accepted["published"] is True
    base = accepted["fact_id"]
    assert len(calls) == 1 and calls[0]["context"]["predecessors"] == []
    assert "Synthetic exploration context" not in json.dumps(calls[0])
    assert store.snapshot().entries[guess["entry_id"]].status == "proved"
    assert PROOF not in store.path.read_text()
    assert store.read_fact(base).proof == PROOF
    repeated = gateway.fact_submit(STATEMENT, PROOF, source_id=guess["entry_id"], source_revision=guess["revision"])
    assert repeated["fact_id"] == base and repeated["deduplicated"] is True
    assert len(calls) == 1
    child_statement = "For every integer n, n + 0 + 0 equals n."
    child_proof = "Apply the preceding additive-identity result twice, first to n + 0 and then to n; both values are integers."
    child = gateway.fact_submit(child_statement, child_proof, predecessors=[base])["fact_id"]
    assert calls[-1]["context"]["predecessors"][0]["statement"] == STATEMENT
    assert PROOF not in json.dumps(calls[-1]["context"])
    resumed = KnowledgeStore(root)
    assert set(resumed.list()) == {base, child}
    assert all(PROOF not in row["statement"] and "proof" not in row
               for row in resumed.search("", kinds=["fact"]))
    monkeypatch.setenv("HYPROVER_ROLE", "main")
    revoked = gateway.fact_revoke(base, "Synthetic audit: retract this dependency.")
    assert set(revoked["revoked"]) == {base, child}
    assert resumed.list() == []
    assert resumed.snapshot().entries[guess["entry_id"]].status == "reopened"
    assert route.entry_id in resumed.snapshot().entries
    assert (root / f"knowledge/proofs/{base}.md").exists()


def test_search_reference_to_submission_preserves_publication_and_retry(tmp_path, monkeypatch):
    root, calls, _ = prepare(tmp_path, monkeypatch)
    guess = gateway.knowledge_record("conjecture", "Identity", STATEMENT)
    gateway.knowledge_record("report", "Progress", "Synthetic report, not a premise.")
    page = gateway.knowledge_search(guess["display_id"], kinds=["conjecture"])
    reference = page["results"][0]["source_ref"]
    assert reference == {"id": guess["entry_id"], "revision": guess["revision"]}
    assert reference["revision"] != page["snapshot_revision"]
    accepted = gateway.fact_submit(STATEMENT, PROOF, source=reference)
    assert accepted["accepted"] and accepted["published"]
    store = KnowledgeStore(root)
    assert store.snapshot().entries[reference["id"]].status == "proved"
    assert store.read_fact(accepted["fact_id"]).proof == PROOF
    assert len(calls) == 1
    for fields in ({"source": reference},
                   {"source_id": reference["id"], "source_revision": reference["revision"]}):
        repeated = gateway.fact_submit(STATEMENT, PROOF, **fields)
        assert repeated["accepted"] and repeated["published"] and repeated["deduplicated"]
        assert repeated["fact_id"] == accepted["fact_id"]
    assert len(calls) == 1
    resolved_ref = gateway.knowledge_search(guess["entry_id"])["results"][0]["source_ref"]
    result = gateway.fact_submit(STATEMENT, PROOF + " This is a different candidate.", source=resolved_ref)
    assert not result["accepted"] and "source_revision_or_state_conflict" in result["error"]
    assert len(calls) == 1


@pytest.mark.parametrize("change,error", [
    ("revision", "source_revision_or_state_conflict"),
    ("statement", "source_statement_mismatch"),
    ("label", "invalid_conjecture_source"),
    ("report", "invalid_conjecture_source"),
    ("other-project", "invalid_conjecture_source"),
])
def test_source_reference_rechecks_before_verification(tmp_path, monkeypatch, change, error):
    root, calls, _ = prepare(tmp_path, monkeypatch)
    guess = gateway.knowledge_record("conjecture", "Identity", STATEMENT)
    reference = gateway.knowledge_search(guess["entry_id"])["results"][0]["source_ref"]
    statement = STATEMENT
    if change == "revision":
        gateway.knowledge_record("conjecture", "Retitled", STATEMENT,
                                 entry_id=guess["entry_id"], expected_revision=guess["revision"])
    elif change == "statement":
        statement += " Changed target."
    elif change == "label":
        reference["id"] = guess["display_id"]
    elif change == "report":
        report = gateway.knowledge_record("report", "Not a conjecture", STATEMENT)
        reference = {"id": report["entry_id"], "revision": report["revision"]}
    else:
        other = KnowledgeStore.initialize(root.parent / "other", "other")
        entry = other.record("conjecture", "Other project", STATEMENT, "high", "worker")
        reference = {"id": entry.entry_id, "revision": entry.revision}
    result = gateway.fact_submit(statement, PROOF, source=reference)
    assert result["accepted"] is False and result["published"] is False
    assert result["fact_id"] is None and error in result["error"]
    assert calls == [] and KnowledgeStore(root).list() == []
    source = KnowledgeStore(root).snapshot().entries[guess["entry_id"]]
    assert source.status == "open"
    if change == "revision":
        assert source.revision == guess["revision"] + 1
        assert reference["revision"] == guess["revision"]


def test_source_reference_rechecked_when_conjecture_changes_during_verification(tmp_path, monkeypatch):
    root, _, _ = prepare(tmp_path, monkeypatch)
    guess = gateway.knowledge_record("conjecture", "Identity", STATEMENT)
    reference = gateway.knowledge_search(guess["entry_id"])["results"][0]["source_ref"]
    store = KnowledgeStore(root)

    def verify_then_edit(**kwargs):
        store.record("conjecture", "Concurrent revision", STATEMENT, "high", "worker",
                     entry_id=reference["id"], expected_revision=reference["revision"])
        return json.loads(json.dumps(OK))

    monkeypatch.setattr(service, "run_codex_verification", verify_then_edit)
    result = gateway.fact_submit(STATEMENT, PROOF, source=reference)
    assert not result["accepted"] and not result["published"] and result["fact_id"] is None
    assert "source_revision_or_state_conflict" in result["write_error"]
    assert store.list() == []
    assert store.snapshot().entries[reference["id"]].status == "open"


def test_rejected_attempt_does_not_refute_the_guess(tmp_path, monkeypatch):
    root, _, _ = prepare(tmp_path, monkeypatch)
    guess = gateway.knowledge_record("conjecture", "Open claim", STATEMENT)
    monkeypatch.setattr(service, "run_codex_verification", lambda **kwargs: {
        "verdict": "wrong", "verification_report": {"summary": "Gap remains.", "critical_errors": [],
        "gaps": ["Missing argument."]}, "repair_hints": "Supply the missing argument."})
    result = gateway.fact_submit(STATEMENT, PROOF, source_id=guess["entry_id"], source_revision=guess["revision"])
    assert result["accepted"] is False and result["published"] is False
    snap = KnowledgeStore(root).snapshot()
    assert snap.entries[guess["entry_id"]].status == "open"
    assert not snap.facts
    assert any(e.kind == "feedback" for e in snap.entries.values())


def test_visible_commit_after_flush_failure_is_not_misreported(tmp_path, monkeypatch):
    root, _, _ = prepare(tmp_path, monkeypatch)
    original = KnowledgeStore.publish
    def publish_then_fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise OSError("simulated directory flush failure after publication")
    monkeypatch.setattr(KnowledgeStore, "publish", publish_then_fail)
    result = gateway.fact_submit(STATEMENT, PROOF)
    assert result["published"] is True and result["accepted"] is True
    assert "durability_warning" in result
    assert KnowledgeStore(root).read_fact(result["fact_id"]).proof == PROOF


def test_duplicate_fact_source_commit_after_flush_failure_is_reported(tmp_path, monkeypatch):
    root, calls, _ = prepare(tmp_path, monkeypatch)
    original_result = gateway.fact_submit(STATEMENT, PROOF)
    guess = gateway.knowledge_record("conjecture", "Existing identity", STATEMENT)
    original = KnowledgeStore.publish
    def publish_then_fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise OSError("directory flush failed after source association")
    monkeypatch.setattr(KnowledgeStore, "publish", publish_then_fail)
    result = gateway.fact_submit(STATEMENT, PROOF, source_id=guess["entry_id"], source_revision=guess["revision"])
    assert result["accepted"] is True and result["published"] is True
    assert result["fact_id"] == original_result["fact_id"]
    assert "durability_warning" in result and len(calls) == 1
    assert KnowledgeStore(root).snapshot().entries[guess["entry_id"]].status == "proved"


def test_synthetic_reports_and_lessons_never_enter_proof_context_or_facts(tmp_path, monkeypatch):
    root, calls, _ = prepare(tmp_path, monkeypatch)
    report = gateway.knowledge_record("report", "Synthetic failed attempt", "report-only sentinel",
                                      details="Synthetic transcript, not a verification result.", request_id="workflow-report")
    gateway.knowledge_record("experience", "Method", "method-only sentinel",
                             links=[report["entry_id"]], attributes={"subtype": "method"})
    gateway.knowledge_record("experience", "Boundary", "boundary-only sentinel",
                             links=[report["entry_id"]], attributes={"subtype": "boundary"})
    gateway.knowledge_record("experience", "Concrete review", "review-only sentinel",
                             links=[report["entry_id"]], attributes={"subtype": "review", "audits": [{
                                 "target_id": report["entry_id"], "revision": report["revision"],
                                 "scope": "The attempted inference.", "not_covered": "Other methods."}]})
    result = gateway.fact_submit(STATEMENT, PROOF)
    assert result["accepted"] and result["published"] and len(calls) == 1
    store = KnowledgeStore(root)
    assert store.snapshot().entries[report["entry_id"]].status == "recorded"
    assert store.list() == [result["fact_id"]]
    context = json.dumps(calls[0]["context"])
    facts = json.dumps([store.read_fact(fid).statement for fid in store.list()])
    for sentinel in ("report-only sentinel", "method-only sentinel", "boundary-only sentinel", "review-only sentinel"):
        assert sentinel not in context and sentinel not in facts
    retry = gateway.knowledge_record("report", "Synthetic failed attempt", "report-only sentinel",
                                     details="Synthetic transcript, not a verification result.", request_id="workflow-report")
    assert retry["entry_id"] == report["entry_id"]
    assert len([e for e in store.snapshot().entries.values() if e.kind == "report"]) == 1
