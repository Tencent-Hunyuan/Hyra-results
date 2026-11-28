import copy

import pytest

from hyprover.article import audit, citation_policy as policy, evidence, references as refs, render
from hyprover.article.models import ArticleError, digest
from hyprover.article.pipeline import Pipeline
from .conftest import publish
from .test_pipeline import article_fixture


def material(project, critical=False):
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    ledger = refs.seed(frozen)
    rid = refs.add_reference(ledger, {"title": "Real analysis", "authors": ["A. Writer"], "year": 2000})
    sid = refs.add_source(ledger, rid, "Real analysis. A. Writer. 2000. Integration and convergence.",
                          origin="https://publisher.example/book", kind="metadata")
    refs.corroborate(ledger, rid, {"title": "Real analysis", "authors": ["A. Writer"], "year": 2000}, sid)
    row = {"id": "convergence", "description": "Dominated convergence", "kind": "mathematical", "critical": critical,
           "source_units": [fid + ":proof:part-0001"], "reference_ids": [rid]}
    refs.add_obligations(ledger, [row], frozen)
    ledger["discovery_complete"] = True
    return frozen, ledger, rid, sid


def test_policy_upgrade_preserves_budget_and_draft_but_requests_new_reviews(project, tmp_path):
    from .test_audit import FakeAgent
    from hyprover.article.models import ArticleConfig, read_json, write_json
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    root = tmp_path / "article"
    evidence.persist(root, frozen)
    agent = FakeAgent(frozen)
    first = Pipeline(root, frozen, ArticleConfig(), agent=agent).run()
    assert first["state"] == "ready"
    article = read_json(root / "article.json")
    checkpoint = read_json(root / "checkpoint.json")
    spent = checkpoint["calls"]
    checkpoint.pop("policy_revision")
    ledger = read_json(root / "references.json")
    ledger["policy_revision"] = 1
    article["reference_digest"] = digest(ledger)
    write_json(root / "article.json", article)
    write_json(root / "references.json", ledger)
    write_json(root / "checkpoint.json", checkpoint)
    existing_calls = len(agent.calls)
    result = Pipeline(root, frozen, ArticleConfig(), agent=agent).run()
    assert result["state"] == "ready" and result["policy_revision"] == policy.POLICY_REVISION
    assert all(task.startswith("review_") for task in agent.calls[existing_calls:])
    saved = read_json(root / "checkpoint.json")
    assert saved["calls"] > spent and saved["elapsed_seconds"] >= checkpoint["elapsed_seconds"]
    assert (root / "migrations/citation-policy-2/article.json").exists()
    assert read_json(root / "article.json")["sections"] == article["sections"]
    assert read_json(root / "draft-status.json")["draft_complete"]


def test_advisory_does_not_block_and_exact_duplicates_count_once(project):
    frozen, ledger, rid, sid = material(project)
    duplicate = dict(ledger["obligations"]["convergence"], id="discovery-again")
    ledger["obligations"][duplicate["id"]] = duplicate
    result = refs.coverage(ledger)
    assert result["identified"] == 1 and result["raw_identified"] == 2
    assert result["duplicates_collapsed"] == 1
    assert result["complete"] and result["advisory_missing"] == ["convergence"]
    assert not result["all_identified_resolved"]
    ledger["obligations"]["discovery-again"]["critical"] = True
    assert not refs.coverage(ledger)["complete"]


def test_negative_standard_assessment_promotes_to_blocking_original_text_check(project):
    frozen, ledger, rid, _ = material(project)
    refs.apply_standard_checks(ledger, [{"obligation_id": "convergence", "reference_id": rid,
        "rationale": "This variant is nonstandard and needs its exact original theorem.",
        "standard_result": False, "supported": False}])
    assert ledger["obligations"]["convergence"]["critical"] is True
    assert ledger["standard_assessments"]
    assert refs.coverage(ledger)["blocking_missing"] == ["convergence"]
    assert not refs.coverage(ledger)["complete"]


def test_deduplication_never_extends_an_essential_check_to_other_proof_units(project):
    frozen, ledger, rid, _ = material(project, True)
    second = dict(ledger["obligations"]["convergence"], id="other-use", source_units=[frozen["targets"][0] + ":statement"])
    ledger["obligations"]["other-use"] = second
    sid = refs.add_source(ledger, rid, "Theorem 1. The identity holds.", origin="fixture", kind="fulltext")
    refs.apply_checks(ledger, [{"obligation_id": "convergence", "reference_id": rid, "source_id": sid,
        "quote": "The identity holds.", "locator": "Theorem 1", "premise_comparison": "Checked first use only", "supported": True}])
    result = refs.coverage(ledger)
    assert result["covered"] == 1 and result["identified"] == 2
    assert result["blocking_missing"] == ["other-use"] and not result["complete"]


def test_standard_reference_needs_real_metadata_but_not_fake_locator(project):
    frozen, ledger, rid, sid = material(project)
    check = {"obligation_id": "convergence", "reference_id": rid, "rationale": "Routine integration theorem covered by this standard text.",
             "standard_result": True, "supported": True}
    refs.apply_standard_checks(ledger, [check])
    cid = next(iter(ledger["checks"]))
    assert refs.usable_check(ledger, cid)
    assert render.render_text("See [[cite:" + cid + "]].", ledger, set()) == "See \\cite{" + rid + "}."
    assert refs.coverage(ledger, {cid})["covered"] == 1
    ledger["references"][rid]["metadata_verified"] = False
    assert not refs.usable_check(ledger, cid)


@pytest.mark.parametrize("kind,critical", [("mathematical", True), ("historical", False), ("comparison", False)])
def test_essential_results_cannot_be_downgraded_by_standard_check(project, kind, critical):
    frozen, ledger, rid, sid = material(project, critical)
    ledger["obligations"]["convergence"]["kind"] = kind
    with pytest.raises(ArticleError, match="cannot_discharge"):
        refs.apply_standard_checks(ledger, [{"obligation_id": "convergence", "reference_id": rid,
            "rationale": "Just call this routine", "standard_result": True, "supported": True}])
    ledger["exclusions"]["convergence"] = "Ignore it"
    assert not refs.coverage(ledger)["complete"]


def test_complete_draft_can_exist_while_critical_citation_blocks_ready(project):
    frozen, ledger, _, _ = material(project, True)
    article = article_fixture(frozen, ledger)
    result = audit.deterministic(article, frozen, ledger)
    assert result["draft_complete"] and not result["passed"]
    assert result["citation_coverage"]["blocking_missing"] == ["convergence"]
    assert render.render(article, ledger)["paper.tex"]


def test_internal_resolution_must_point_to_real_proof_not_statement(project):
    frozen, ledger, _, _ = material(project, True)
    ledger["obligations"]["convergence"]["reference_ids"] = []
    report = {"obligation_id": "convergence", "supported_internally": True,
              "proof_units": [frozen["targets"][0] + ":statement"], "reason": "Found in the full supplied proof"}
    with pytest.raises(ArticleError, match="actual_proof_units"):
        refs.apply_internal_resolution(ledger, frozen, report)
    report["proof_units"] = [frozen["targets"][0] + ":proof:part-0001"]
    refs.apply_internal_resolution(ledger, frozen, report)
    refs.validate(ledger, frozen)
    assert refs.coverage(ledger)["internally_supported"] == ["convergence"]


def test_new_review_policy_cannot_reuse_old_pass(project):
    from .test_audit import passed
    frozen, ledger, _, _ = material(project)
    article = article_fixture(frozen, ledger)
    outputs = render.render(article, ledger)
    expected = audit.expected_coverage(article, frozen, ledger, outputs, "whole", ["title"])
    receipt = passed(expected)
    receipt["coverage"].pop("policy_revision")
    with pytest.raises(ArticleError):
        audit.receipt_outcome(receipt, expected)


def test_inventory_groups_never_leave_a_statement_without_its_proof(project, tmp_path):
    from hyprover.article.models import ArticleConfig
    fid = publish(project, proof="Proof segment.\n\n" * 80)
    frozen = evidence.capture(project.root, [fid], config=ArticleConfig(chunk_chars=256))
    evidence.persist(tmp_path / "article", frozen)
    pipe = Pipeline(tmp_path / "article", frozen, ArticleConfig(chunk_chars=256))
    groups = list(pipe.groups())
    for group in groups:
        assert any(u["kind"] == "statement" for u in group)
        assert any(u["kind"] == "proof" for u in group)
    actual = [u["id"] for group in groups for u in group if u["kind"] == "proof"]
    assert len(actual) == len(set(actual)) == len(frozen["facts"][0]["ranges"])
