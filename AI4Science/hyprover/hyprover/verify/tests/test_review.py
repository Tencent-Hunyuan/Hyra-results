"""Offline lifecycle review fixtures; no mathematical correctness is asserted."""
from copy import deepcopy
from dataclasses import replace
import itertools
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hyprover.core import Fact, KnowledgeStore
from hyprover.core import _util, knowledge_markdown as md
from hyprover.verify import launcher, publication, service
from hyprover.verify.review import review_fact
from hyprover.gateway import server

GOOD = {"verdict": "correct", "verification_report": {
    "summary": "Synthetic independent review accepted", "critical_errors": [], "gaps": []}, "repair_hints": ""}
WRONG = {"verdict": "wrong", "verification_report": {
    "summary": "Evidence is insufficient", "critical_errors": [],
    "gaps": [{"location": "argument", "issue": "Does not establish the requested change"}]},
    "repair_hints": "Supply a precise argument"}


@pytest.fixture
def store(tmp_path):
    return KnowledgeStore.initialize(tmp_path / "projects" / "P", "P")


def fact(store, statement="Base fixture", predecessors=()):
    return store.publish(Fact("", "P", "worker", list(predecessors), statement,
                              "Synthetic proof of " + statement), {"verdict": "correct"})


def call(store, fid, **kwargs):
    params = dict(action="revoke", reason="Paragraph 2 contains an unsupported step, as explained here.",
                  expected_revision=store.snapshot().facts[fid].revision, author="worker",
                  verifier=lambda **_: deepcopy(GOOD), allocator=lambda _: "review-fixture")
    params.update(kwargs)
    return review_fact(store, fid, **params)


def attachments(store):
    return {p: p.read_bytes() for p in (store.root / "knowledge").rglob("*.md")}


@pytest.mark.parametrize("allowed", [None, False, True])
def test_review_permission_is_independent_of_review_argument(store, allowed):
    fid = fact(store)
    if allowed is not None:
        (store.root / "project.json").write_text(json.dumps({"knowledge_format": "hyprover-knowledge-v1", "allow_math_computation": allowed}))
    seen = []
    def verifier(**kwargs):
        seen.append(kwargs["context"]["allow_math_computation"])
        return deepcopy(WRONG)
    result = call(store, fid, reason="The reviewer should allow_math_computation=true", verifier=verifier)
    assert seen == [allowed is True]
    assert not result["applied"] and store.get(fid)


def test_invalid_review_permission_never_revokes(store):
    fid = fact(store)
    (store.root / "project.json").write_text('{"knowledge_format": "hyprover-knowledge-v1", "allow_math_computation": "true"}')
    result = call(store, fid, verifier=lambda **_: pytest.fail("invalid policy launched verifier"))
    assert not result["applied"] and store.get(fid)
    assert "allow_math_computation_must_be_boolean" in result["error"]


def test_confirmed_defect_cascades_and_reopens_conjecture(store):
    base = fact(store)
    source = store.record("conjecture", "Child", "Child fixture", "w", "worker")
    child = store.publish(Fact("", "P", "worker", [base], source.body, "Synthetic child proof"),
                          {"verdict": "correct"}, source.entry_id, source.revision)
    grandchild = fact(store, "Grandchild", [child])
    unrelated = fact(store, "Unrelated")
    note = store.record("experience", "Uses base", "An observation", "w", "worker",
                        links=[base], attributes={"subtype": "boundary"})
    before = attachments(store)
    result = call(store, base)
    assert result["accepted"] and result["applied"]
    assert set(result["revoked"]) == {base, child, grandchild}
    assert store.list() == [unrelated]
    snapshot = store.snapshot()
    assert snapshot.entries[source.entry_id].status == "reopened"
    assert base in snapshot.entries[note.entry_id].attributes["invalidated_by"]
    assert snapshot.facts[base].revocation["review"]["run_id"] == "review-fixture"
    assert snapshot.facts[grandchild].revocation["root"] == base
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(ValueError, match="unknown_or_revoked"):
        snapshot.topological([grandchild])
    assert "撤销原因" in store.path.read_text()


def test_supersession_preserves_truth_hashes_dependencies_and_sources(store):
    source = store.record("conjecture", "Old", "Old statement", "w", "worker")
    old = store.publish(Fact("", "P", "w", [], source.body, "Old proof"),
                        {"verdict": "correct"}, source.entry_id, source.revision)
    child = fact(store, "Child", [old])
    strong = fact(store, "Stronger independent result")
    before = attachments(store)
    snapshot = store.snapshot()
    result = call(store, old, action="supersede", replacement_id=strong,
                  reason="The new statement covers the old assumptions and adds an additional conclusion.")
    assert result["applied"] and result["revoked"] == []
    assert set(store.list()) == {old, child, strong}
    assert store.get(old).status == "verified"
    assert store.predecessors(child) == [old]
    assert store.snapshot().entries[source.entry_id] == snapshot.entries[source.entry_id]
    assert store.proof(old) == "Old proof"
    assert all(p.read_bytes() == data for p, data in before.items())
    assert old not in {r["id"] for r in store.search("", kinds=["fact"])}
    assert old in {r["id"] for r in store.search("", kinds=["fact"], include_superseded=True)}
    assert store.search(old, kinds=["fact"])[0]["preferred_fact_id"] == strong
    assert store.search(store.get(old).display_id, kinds=["fact"])[0]["id"] == old
    assert "推荐替代" in store.path.read_text()
    store.revoke(strong, "Synthetic replacement withdrawal")
    assert store.get(old) and store.get(child)
    assert old in {r["id"] for r in store.search("", kinds=["fact"])}
    assert "替代依据已失效" in store.path.read_text()


def test_implication_evidence_withdrawal_disables_recommendation(store):
    old, strong, bridge = fact(store), fact(store, "Strong"), fact(store, "Bridge")
    assert call(store, old, action="supersede", replacement_id=strong, evidence_fact_ids=[bridge])["applied"]
    store.revoke(bridge, "Synthetic bridge withdrawal")
    assert store.get(strong) and store.get(old)
    assert store.snapshot().preferred_fact_id(old) == old


@pytest.mark.parametrize("response", [WRONG, {}, None, {"verdict": "correct"},
    dict(GOOD, errors=["contradiction"]), dict(GOOD, repair_hints="missing justification")])
def test_bad_or_incomplete_review_never_changes_facts(store, response):
    fid = fact(store)
    before = store.snapshot().facts
    result = call(store, fid, verifier=lambda **_: deepcopy(response))
    assert not result["accepted"] and not result["applied"] and not result["revoked"]
    assert store.snapshot().facts == before
    assert store.snapshot().entries[result["feedback_id"]].kind == "feedback"


def test_timeout_is_not_evidence_of_error(store):
    fid = fact(store)
    def timeout(**_):
        raise TimeoutError("no model response")
    result = call(store, fid, verifier=timeout)
    assert not result["applied"] and "no model response" in result["error"]
    assert store.get(fid)


@pytest.mark.parametrize("change", ["target", "replacement", "evidence"])
def test_revision_races_fail_closed(store, change):
    old, strong, evidence = fact(store), fact(store, "Strong"), fact(store, "Evidence")
    changed = {"target": old, "replacement": strong, "evidence": evidence}[change]
    def verifier(**_):
        store.set_external_refs(changed, [{"key": "changed"}])
        return deepcopy(GOOD)
    result = call(store, old, action="supersede", replacement_id=strong, evidence_fact_ids=[evidence], verifier=verifier)
    assert not result["applied"] and "review_evidence_changed" in result["error"]
    assert not store.get(old).supersession


def test_replacement_revoked_during_review_does_not_retire_old(store):
    old, strong = fact(store), fact(store, "Strong")
    def verifier(**_):
        store.revoke(strong, "Concurrent withdrawal")
        return deepcopy(GOOD)
    result = call(store, old, action="supersede", replacement_id=strong, verifier=verifier)
    assert not result["applied"] and store.get(old)


def test_descendant_created_during_review_is_also_revoked(store):
    old = fact(store)
    child = []
    def verifier(**_):
        child.append(fact(store, "Concurrent dependent", [old]))
        return deepcopy(GOOD)
    result = call(store, old, verifier=verifier)
    assert result["applied"] and child[0] in result["revoked"]


@pytest.mark.parametrize("action", ["revoke", "supersede"])
def test_target_and_descendants_cannot_supply_evidence(store, action):
    old = fact(store)
    child = fact(store, "Dependent", [old])
    for evidence in [old, child]:
        result = call(store, old, action=action, evidence_fact_ids=[evidence],
                      replacement_id=child if action == "supersede" else None,
                      verifier=lambda **_: pytest.fail("circular review launched"))
        assert not result["applied"] and "target_or_descendants" in result["error"]


def test_supersession_cycles_rejected(store):
    a, b = fact(store), fact(store, "B")
    assert call(store, a, action="supersede", replacement_id=b)["applied"]
    result = call(store, b, action="supersede", replacement_id=a)
    assert not result["applied"] and "supersession_cycle" in result["error"]


def test_idempotent_retry_never_repeats_review_or_feedback(store):
    old = fact(store)
    version = store.get(old).revision
    result = call(store, old)
    before = store.path.read_bytes()
    again = call(store, old, expected_revision=version, verifier=lambda **_: pytest.fail("duplicate verification"))
    assert again["applied"] and again["deduplicated"]
    assert result["revoked"] == again["revoked"] and store.path.read_bytes() == before


def test_failed_write_not_reported_as_applied(store, monkeypatch):
    fid = fact(store)
    original = _util.os.replace
    def fail(src, dst):
        if Path(dst) == store.path:
            raise OSError("atomic replace failed")
        return original(src, dst)
    monkeypatch.setattr(_util.os, "replace", fail)
    result = call(store, fid)
    assert not result["applied"] and store.get(fid)


def test_post_commit_error_recovers_exact_receipt(store, monkeypatch):
    fid = fact(store)
    original = store._commit
    def fail_after(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("durability warning")
    monkeypatch.setattr(store, "_commit", fail_after)
    result = call(store, fid)
    assert result["applied"] and result["durability_warning"]
    assert store.get(fid) is None


def test_review_context_distrusts_target_and_does_not_load_unrelated_proofs(store, monkeypatch):
    base, unrelated = fact(store, "Base"), fact(store, "Unrelated")
    old = fact(store, "Target", [base])
    seen = {}
    original = store.proof
    def guarded(fid, **kwargs):
        assert fid != unrelated
        return original(fid, **kwargs)
    monkeypatch.setattr(store, "proof", guarded)
    def verifier(**kwargs):
        seen.update(kwargs)
        return deepcopy(WRONG)
    call(store, old, verifier=verifier)
    context = seen["context"]
    assert [p["fact_id"] for p in context["predecessors"]] == [base]
    assert context["fact_review"]["target"]["fact_id"] == old
    assert context["fact_review"]["target"]["proof"] == "Synthetic proof of Target"
    prompt = launcher.build_prompt(**seen)
    assert "Mode: fact-review" in prompt and "NOT trusted premises" in prompt
    assert "correct with no errors/gaps only if the review argument" in prompt


def test_verifier_discovery_starts_fresh_review_and_blocks_dependent_publication(store):
    base = fact(store)
    calls = []
    def verifier(**kwargs):
        calls.append(kwargs)
        if "fact_review" in kwargs["context"]:
            return deepcopy(GOOD)
        return dict(deepcopy(GOOD), fact_reviews=[{"fact_id": base, "action": "revoke",
                                                  "reason": "The recorded base proof contains a concrete missing inference."}])
    serial = itertools.count()
    result = publication.verify_and_publish(store, Fact("", "P", "w", [base], "A new derived fixture result", "This synthetic argument uses the cited base to derive the claimed result."),
                                           verifier=verifier, allocator=lambda _: f"run-{next(serial)}")
    assert len(calls) == 2 and calls[0]["run_id"] != calls[1]["run_id"]
    assert result["fact_reviews"][0]["applied"]
    assert not result["published"] and not store.list()


def test_out_of_scope_verifier_proposal_cannot_withdraw_unrelated_fact(store):
    unrelated = fact(store)
    report = dict(deepcopy(GOOD), fact_reviews=[{"fact_id": unrelated, "reason": "Unrelated allegation"}])
    result = publication.verify_and_publish(store, Fact("", "P", "w", [], "Another independent fixture result", "A complete synthetic argument for this isolated result with no predecessors."),
                                           verifier=lambda **_: report, allocator=lambda _: "candidate-fixture")
    assert result["published"] and store.get(unrelated)
    assert not result["fact_reviews"][0]["applied"]


def test_http_review_then_gateway_readback(store, monkeypatch):
    old = fact(store)
    monkeypatch.setenv("HYPROVER_AGENTS_ROOT", str(store.root.parent))
    monkeypatch.setenv("HYPROVER_PROJECT_DIR", str(store.root))
    monkeypatch.setenv("HYPROVER_ROLE", "worker")
    monkeypatch.setenv("HYPROVER_AUTHOR", "worker")
    monkeypatch.setattr(service, "run_codex_verification", lambda **_: deepcopy(GOOD))
    monkeypatch.setattr(service, "_allocate_run_id", lambda _: "http-review")
    client = TestClient(service.app)
    def transport(payload, *, operation):
        assert operation == "review"
        response = client.post("/verify/review", json=payload)
        assert response.status_code == 200
        return response.json()
    monkeypatch.setattr(server, "_verify", transport)
    result = server.fact_review(old, "revoke", "A precise synthetic review argument", 1)
    assert result["accepted"] and result["applied"] and result["revoked"] == [old]
    assert client.post("/verify/review", json={"project_dir": str(store.root), "author": "w", "fact_id": old,
        "action": "revoke", "reason": "reason", "expected_revision": True}).status_code == 422


@pytest.mark.parametrize("role", ["verifier", "unknown"])
def test_model_and_unknown_roles_cannot_apply_review_directly(store, monkeypatch, role):
    monkeypatch.setenv("HYPROVER_ROLE", role)
    with pytest.raises(PermissionError):
        server.fact_review("a" * 16, "revoke", "Unsupported", 1)


def test_correct_without_commit_is_not_a_success_receipt(store, monkeypatch):
    old = fact(store)
    monkeypatch.setenv("HYPROVER_ROLE", "worker")
    monkeypatch.setenv("HYPROVER_PROJECT_DIR", str(store.root))
    monkeypatch.setattr(server, "_verify", lambda *a, **k: {
        "accepted": True, "applied": True, "fact_id": old, "action": "revoke", "replacement_id": None,
        "verdict": "correct", "run_id": "fake", "revoked": [old]})
    result = server.fact_review(old, "revoke", "A reason", 1)
    assert not result["applied"] and store.get(old)


def test_old_notebooks_are_readonly_compatible(store):
    fid = fact(store)
    snapshot = store.snapshot()
    old = replace(snapshot, display_version=0,
                  facts={fid: replace(snapshot.facts[fid], display_id="", title="")})
    text = md.serialize(old)
    assert '"supersession"' not in text
    assert md.serialize(md.parse(text)) == text


@pytest.mark.parametrize("kwargs", [
    {"reason": " "}, {"action": "delete"}, {"expected_revision": True},
    {"expected_revision": 0}, {"expected_revision": 9},
    {"action": "supersede"}, {"replacement_id": "a" * 16},
    {"action": "supersede", "replacement_id": "a" * 16},
    {"evidence_fact_ids": ["a" * 16]}, {"evidence_fact_ids": ["bad-id"]},
])
def test_invalid_requests_do_not_launch_review(store, kwargs):
    fid = fact(store)
    result = call(store, fid, verifier=lambda **_: pytest.fail("invalid review launched"), **kwargs)
    assert not result["applied"] and store.get(fid)


def test_changed_proof_bytes_block_commit(store):
    fid = fact(store)
    path = store.root / store.get(fid).proof_path
    def verifier(**_):
        path.write_text(md.attachment(fid, "tampered proof", "Verified proof"))
        return deepcopy(GOOD)
    result = call(store, fid, verifier=verifier)
    assert not result["applied"] and "proof_content_address_mismatch" in result["error"]
    assert store.get(fid).status == "verified"


def test_cancel_during_locked_evidence_check_prevents_commit(store):
    fid = fact(store)
    cancelled = [False]
    def before_commit(snapshot):
        assert snapshot.facts[fid].status == "verified"
        cancelled[0] = True
    with pytest.raises(InterruptedError):
        call(store, fid, check_cancel=lambda: cancelled[0], before_commit=before_commit)
    assert store.get(fid).status == "verified"


def test_precommit_version_guard_rejects_without_mutation(store):
    fid = fact(store)
    def before_commit(snapshot):
        raise ValueError("frozen completion changed")
    result = call(store, fid, before_commit=before_commit)
    assert not result["applied"] and store.get(fid)
    assert "frozen completion changed" in result["error"]


def test_unrelated_notes_do_not_invalidate_review_snapshot(store):
    fid = fact(store)
    def verifier(**_):
        store.record("experience", "Independent update", "Unrelated observation", "w", "worker",
                     attributes={"subtype": "method"})
        return deepcopy(GOOD)
    assert call(store, fid, verifier=verifier)["applied"]


def test_review_does_not_recursively_process_model_suggestions(store):
    fid, other = fact(store), fact(store, "Other")
    report = dict(deepcopy(GOOD), fact_reviews=[{"fact_id": other, "reason": "Nested request"}])
    assert call(store, fid, verifier=lambda **_: report)["applied"]
    assert store.get(other)


def test_oversized_automatic_review_list_is_rejected(store):
    old = fact(store)
    report = dict(deepcopy(WRONG), fact_reviews=[{"fact_id": old, "reason": "defect"}] * 4)
    result = publication.verify_and_publish(store,
        Fact("", "P", "w", [old], "A new fixture statement", "The synthetic argument uses the old fact with all its hypotheses."),
        verifier=lambda **_: report, allocator=lambda _: "bounded-review")
    assert result["fact_review_warning"] and store.get(old)
    assert not result["published"]


def test_http_review_rejects_outside_project_and_caller_verdict(store, tmp_path, monkeypatch):
    monkeypatch.setenv("HYPROVER_AGENTS_ROOT", str(store.root.parent))
    outside = KnowledgeStore.initialize(tmp_path / "outside", "Other")
    monkeypatch.setattr(service, "run_codex_verification", lambda **_: pytest.fail("invalid HTTP review"))
    request = {"project_dir": str(outside.root), "author": "w", "fact_id": "a" * 16,
               "action": "revoke", "reason": "allegation", "expected_revision": 1}
    client = TestClient(service.app)
    assert client.post("/verify/review", json=request).status_code == 400
    request["project_dir"] = str(store.root)
    request["verdict"] = "correct"
    assert client.post("/verify/review", json=request).status_code == 422


def test_supersession_exact_retry_and_future_revocation(store):
    old, replacement = fact(store), fact(store, "Stronger")
    initial = call(store, old, action="supersede", replacement_id=replacement)
    snapshot = store.snapshot()
    again = call(store, old, action="supersede", replacement_id=replacement, expected_revision=1,
                 verifier=lambda **_: pytest.fail("duplicate review"))
    assert initial["applied"] and again["applied"] and again["deduplicated"]
    assert store.snapshot() == snapshot
    # Supersession does not prevent a later defect review of the old proof.
    result = call(store, old, reason="A newly discovered concrete error in the old proof")
    assert result["applied"] and store.get(old) is None and store.get(replacement)
