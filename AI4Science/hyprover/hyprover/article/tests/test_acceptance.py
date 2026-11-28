import copy
import shutil
import subprocess

import pytest

from hyprover.article import audit, evidence, improvement, publication, references, render, revision_patch, runtime
from hyprover.article.agent import Reply
from hyprover.article.models import ArticleError, digest, read_json, write_json
from .conftest import prepare_legacy_review, publish
from .test_audit import FakeAgent, passed
from .test_publication import fixture
from .test_reference_refresh import referenced


def receipt_fixture(project):
    frozen, ledger, _, _, article = fixture(project)
    job = audit.review_jobs(article)[-1]
    expected = audit.expected_coverage(article, frozen, ledger, render.render(article, ledger), job["id"], job["blocks"])
    return passed(expected), expected


@pytest.mark.parametrize("check", sorted(audit.ADVISORY_CHECKS))
def test_explicit_cosmetic_advice_can_pass_without_hiding_warning(project, check):
    report, expected = receipt_fixture(project)
    report[check] = False
    report["findings"] = [{"location": "A paragraph", "issue": "Optional polishing.", "severity": "advisory", "check": check}]
    assert audit.receipt_outcome(report, expected) == "passed"
    report["findings"] = []
    assert audit.receipt_outcome(report, expected) == "needs_review"


@pytest.mark.parametrize("check", ["faithful", "assumptions_preserved", "no_omissions", "citations_supported", "scope_respected",
                                   "preliminaries_complete", "notation_consistent", "dependency_ordered", "classical_credit"])
def test_substantive_errors_never_pass_as_editorial_advice(project, check):
    report, expected = receipt_fixture(project)
    report[check] = False
    report["findings"] = [{"location": "Theorem", "issue": "Unsupported premise.", "severity": "advisory", "check": check}]
    with pytest.raises(ArticleError, match="cannot_be_advisory"):
        audit.receipt_outcome(report, expected)
    report["findings"] = [{"location": "Theorem", "issue": "Unsupported premise.", "severity": "blocking", "check": check}]
    assert audit.receipt_outcome(report, expected) == "needs_review"


@pytest.mark.parametrize("fault", ["truncated", "incomplete", "uncovered", "unversioned", "legacy_finding", "needs_review"])
def test_advisory_acceptance_does_not_bypass_coverage_or_verdict(project, fault):
    report, expected = receipt_fixture(project)
    if fault == "truncated":
        report["coverage"]["truncated"] = True
    elif fault == "incomplete":
        report["coverage"]["materials_complete"] = False
    elif fault == "uncovered":
        report["coverage"]["uncovered"] = ["missing"]
    elif fault == "unversioned":
        report["coverage"].pop("review_revision")
    elif fault == "legacy_finding":
        report["findings"] = [{"location": "proof", "issue": "Old unclassified concern."}]
    else:
        report["verdict"] = "needs_review"
    if fault == "unversioned":
        with pytest.raises(ArticleError):
            audit.receipt_outcome(report, expected)
    else:
        assert audit.receipt_outcome(report, expected) == "needs_review"


def test_unused_explicitly_optional_background_is_advisory_without_changing_ledger(project):
    _, ledger, _, cid = referenced(project, background=True)
    oid = ledger["checks"][cid]["obligation_id"]
    ledger["obligations"][oid].update(critical=False, origin="discovery")
    rid = ledger["checks"][cid]["reference_id"]
    ledger["unresolved_candidates"] = [{"reference_id": rid, "reason": "focused_selection_uncertain", "critical": True}]
    before = copy.deepcopy(ledger)
    result = references.coverage(ledger, set())
    assert result["complete"] and not result["all_identified_resolved"]
    assert not result["blocking_candidates"] and result["advisory_candidates"]
    assert result["unresolved_candidates"] == before["unresolved_candidates"] and ledger == before
    assert not references.coverage(ledger)["blocking_candidates"]
    assert not references.coverage(ledger, {cid})["blocking_candidates"]
    ledger["references"][rid]["metadata_verified"] = False
    assert not references.coverage(ledger, set())["blocking_candidates"]
    assert not references.usable_check(ledger, cid)
    assert references.coverage(ledger, {cid})["blocking_candidates"]


@pytest.mark.parametrize("fault", ["critical", "historical", "comparison", "conflict", "unknown_scope", "verified_fact"])
def test_uncertain_critical_or_evidence_failed_candidate_cannot_be_downgraded(project, fault):
    _, ledger, _, cid = referenced(project, background=True)
    check = ledger["checks"][cid]
    rid, oid = check["reference_id"], check["obligation_id"]
    obligation = ledger["obligations"][oid]
    obligation.update(critical=False, origin="discovery")
    issue = {"reference_id": rid, "reason": "focused_selection_uncertain", "critical": True}
    if fault == "critical":
        obligation["critical"] = True
    elif fault in ("historical", "comparison"):
        obligation["kind"] = fault
    elif fault == "conflict":
        ledger["references"][rid]["conflicts"] = ["identity"]
    elif fault == "unknown_scope":
        issue = {"stage": "background_references", "material_digest": "unknown", "critical": True}
    else:
        obligation["origin"] = "verified_fact"
    ledger["unresolved_candidates"] = [issue]
    result = references.coverage(ledger, set())
    assert not result["complete"] and result["blocking_candidates"] == [issue]
    assert not result["advisory_candidates"]


@pytest.mark.parametrize("fault", ["math", "unbound"])
def test_nonessential_or_unbound_candidate_is_advisory_not_blocking(project, fault):
    _, ledger, _, cid = referenced(project, background=True)
    check = ledger["checks"][cid]
    rid, oid = check["reference_id"], check["obligation_id"]
    ledger["obligations"][oid].update(critical=False, origin="discovery")
    if fault == "math":
        ledger["obligations"][oid]["kind"] = "mathematical"
    else:
        ledger["obligations"][oid]["reference_ids"] = []
    issue = {"reference_id": rid, "reason": "focused_selection_uncertain", "critical": True}
    ledger["unresolved_candidates"] = [issue]
    result = references.coverage(ledger, set())
    assert result["complete"] and not result["blocking_candidates"]
    assert result["advisory_candidates"] and result["advisory_candidates"][0]["blocking"] is False


def test_optional_citation_finding_is_advisory_but_substantive_credit_is_not(project):
    report, expected = receipt_fixture(project)
    report["findings"] = [{"location": "Related work", "issue": "An optional lead remains unresolved.",
                           "severity": "advisory", "check": audit.OPTIONAL_CITATION_CHECK}]
    assert audit.receipt_outcome(report, expected) == "passed"
    report["findings"] = [{"location": "Preliminaries", "issue": "Missing credit for a substantive result.",
                           "severity": "blocking", "check": "classical_credit"}]
    assert audit.receipt_outcome(report, expected) == "needs_review"


def test_patch_is_atomic_and_cannot_change_evidence_or_drop_structure(project):
    frozen, ledger, _, plan, article = fixture(project)
    target = article["sections"][1]["blocks"][0]
    patch = {"block_id": target["id"], "before_digest": digest(target), "text": target["text"] + " All parameters retain their stated ranges.",
             "citations": [], "crossrefs": []}
    material = {"article": article, "evidence": frozen, "ledger": ledger, "publication_plan": plan}
    original = copy.deepcopy(article)
    fixed = revision_patch.apply({"patches": [patch], "unaddressed": []}, material)
    assert article == original and fixed["source_map"] == original["source_map"]
    assert fixed["sections"][1]["blocks"][0]["evidence"] == target["evidence"]
    assert fixed["sections"][0] == original["sections"][0]
    for mutation in ({"before_digest": "stale"}, {"evidence": []}, {"block_id": "invented"}, {"text": ""},
                     {"text": "Use [[cite:fake]].", "citations": ["fake"]}):
        with pytest.raises(ArticleError):
            revision_patch.apply({"patches": [dict(patch, **mutation)], "unaddressed": []}, material)
    assert article == original


def old_review_fixture(project):
    root = prepare_legacy_review(project.root, [publish(project)])
    frozen = evidence.load(root)
    assert runtime.work(project.root, root.name, agent=FakeAgent(frozen))["state"] == "ready"
    report = read_json(root / "audit.json")
    report.pop("review_revision")
    for receipt in report["receipts"].values():
        receipt["report"]["coverage"].pop("review_revision")
    report["state"] = "needs_review"
    write_json(root / "audit.json", report)
    status = read_json(root / "status.json")
    status["state"] = "needs_review"
    write_json(root / "status.json", status)
    return root, frozen


def test_policy_reassessment_preserves_math_sources_budget_and_old_stop_history(project):
    root, frozen = old_review_fixture(project)
    before = {name: (root / name).read_bytes() for name in ("article.json", "references.json", "evidence.json", "manifest.json")}
    cp = read_json(root / "checkpoint.json")
    calls = cp["calls"]

    class AdvisoryAgent(FakeAgent):
        def __call__(self, task, material, contract):
            assert task.startswith("review_")
            assert "editorial perfection" in contract
            reply = super().__call__(task, material, contract)
            reply.value["concise"] = False
            reply.value["findings"] = [{"location": "Introduction", "issue": "Could be shorter.", "severity": "advisory", "check": "concise"}]
            return Reply("new-review-" + str(len(self.calls)), reply.value)

    agent = AdvisoryAgent(frozen)
    state = runtime.work(project.root, root.name, resume=True, agent=agent)
    assert state["state"] == "ready", state
    assert all((root / name).read_bytes() == value for name, value in before.items())
    after = read_json(root / "checkpoint.json")
    assert after["calls"] == calls + len(agent.calls) and after["auto_improvement"] == cp["auto_improvement"]
    report = read_json(root / "audit.json")
    assert report["warnings"] and report["review_revision"] == audit.REVIEW_REVISION
    assert (root / ("reassessments/review-" + str(audit.REVIEW_REVISION) + "/original/audit.json")).exists()
    assert runtime.status(project.root, root.name)["state"] == "ready"
    count = len(agent.calls)
    runtime.work(project.root, root.name, resume=True, agent=agent)
    assert len(agent.calls) == count


@pytest.mark.parametrize("resolved", [True, False])
def test_reassessment_fixes_named_blocks_then_respects_revision_limit(project, resolved):
    root, frozen = old_review_fixture(project)
    before = read_json(root / "article.json")

    class RepairAgent(FakeAgent):
        def __call__(self, task, material, contract):
            if task == "revise_article_blocks":
                self.calls.append(task)
                paragraph = material["article"]["sections"][1]["blocks"][0]
                return Reply("new-patcher", {"patches": [{"block_id": paragraph["id"], "before_digest": material["block_versions"][paragraph["id"]],
                    "text": paragraph["text"] + " The hypotheses apply throughout.", "citations": [], "crossrefs": []}], "unaddressed": []})
            assert task.startswith("review_")
            reply = super().__call__(task, material, contract)
            if not resolved or "revise_article_blocks" not in self.calls:
                reply.value.update(verdict="needs_review", notation_consistent=False,
                    findings=[{"location": "notation", "issue": "Clarify the scope of the given hypotheses.", "severity": "blocking", "check": "notation_consistent"}])
            return Reply("new-review-" + str(len(self.calls)), reply.value)

    agent = RepairAgent(frozen)
    state = runtime.work(project.root, root.name, resume=True, agent=agent)
    assert state["state"] == ("ready" if resolved else "needs_review"), state
    assert agent.calls.count("revise_article_blocks") == 1 and agent.calls.count("review_whole") == 2
    after = read_json(root / "article.json")
    assert after["sections"][0] == before["sections"][0] and after["sections"][3] == before["sections"][3]
    assert after["source_map"] == before["source_map"]
    assert read_json(root / "checkpoint.json")["reassessment"]["repairs"] == 1
    status = read_json(root / "reassessment-status.json")
    assert status["max_repairs"] == 1 and status["stop_reason"] == ("ready" if resolved else "targeted_repair_limit")


def test_author_surnames_and_chapter_metadata_are_rendered_correctly(project):
    _, ledger, _, cid = referenced(project)
    ledger = copy.deepcopy(ledger)
    rid = ledger["checks"][cid]["reference_id"]
    row = ledger["references"][rid]
    row["metadata"].update(authors=["Sergey Bobkov"], type="book-chapter", venue="Collected Studies", series="Probability series", pages="45–60")
    other = copy.deepcopy(row)
    other["id"] = "ref-other"
    other["metadata"].update(authors=["Matthieu Fradelizi"], title="Other work")
    ledger["references"][other["id"]] = other
    ledger["checks"]["second"] = dict(ledger["checks"][cid], id="second", reference_id=other["id"])
    tex, bib = render.bibliography(ledger, {cid, "second"})
    assert tex.index("Bobkov") < tex.index("Fradelizi")
    assert "45--60" in tex and "Probability series" in tex and "Collected Studies" in tex
    assert "@incollection" in bib


@pytest.mark.parametrize("compiled,width,severity", [(True, 1, None), (True, 8, "advisory"), (True, 40, "blocking"), (False, 0, "blocking")])
def test_layout_distinguishes_minor_protrusion_from_compile_or_large_overflow(tmp_path, compiled, width, severity):
    findings = publication.layout_findings(tmp_path, {"compiled": compiled, "max_overfull_pt": width})
    assert (findings[0]["severity"] if findings else None) == severity


@pytest.mark.skipif(not all(shutil.which(n) for n in ("bwrap", "pdflatex", "prlimit", "pdfinfo", "pdftotext")), reason="TeX unavailable")
def test_advisory_only_pdf_can_be_ready_and_readable(project):
    root = runtime.prepare(project.root, [publish(project)])
    frozen = evidence.load(root)
    state = runtime.work(project.root, root.name, agent=FakeAgent(frozen), compile_output=True)
    assert state["state"] == "ready" and state["compilation"]["compiled"]
    text = subprocess.run(["pdftotext", str(root / "paper.pdf"), "-"], capture_output=True, text=True, check=True).stdout
    assert "Introduction" in text


@pytest.mark.parametrize("cancelled", [False, True])
def test_reassessment_scheduler_waits_for_running_jobs_then_launches_once(tmp_path, monkeypatch, cancelled):
    from hyprover.article import collection
    task = tmp_path / "P/articles/run"
    task.mkdir(parents=True)
    write_json(task / "audit.json", {"article_digest": "old", "state": "needs_review"})
    if cancelled:
        write_json(task / "cancel.json", {})
    state = {"concurrency": 7, "review_revision": audit.REVIEW_REVISION,
             "projects": {"P": {"directory": "P", "run_id": "run", "state": "running"}}}
    phase, launches = ["running"], []
    monkeypatch.setattr(runtime, "job_path", lambda *args: task)
    monkeypatch.setattr(runtime, "status", lambda *args: {"state": phase[0]})

    def sleep(_):
        assert not launches
        phase[0] = "needs_review"

    def launch(root, **options):
        dispatch_id = options.pop("dispatch_id")
        launches.append(options)
        phase[0] = "failed"
        return {"state": "failed", "dispatch_id": dispatch_id}

    monkeypatch.setattr(collection.time, "sleep", sleep)
    monkeypatch.setattr(runtime, "launch", launch)
    assert collection._drive_locked(tmp_path, state) == 2
    assert launches == ([] if cancelled else [{"resume": True}])
    assert collection._drive_locked(tmp_path, state) == 2
    assert len(launches) == (0 if cancelled else 1)


def test_old_ready_requires_new_review_but_changed_old_output_is_stale(project):
    root, _ = old_review_fixture(project)
    state = read_json(root / "status.json")
    state["state"] = "ready"
    write_json(root / "status.json", state)
    assert runtime.status(project.root, root.name)["state"] == "needs_review"
    (root / "paper.tex").write_text((root / "paper.tex").read_text() + "Tampered")
    assert runtime.status(project.root, root.name)["state"] == "stale"


def test_reassessment_without_supported_local_change_stops_and_keeps_critical_review(project):
    root, frozen = old_review_fixture(project)

    class NoFixAgent(FakeAgent):
        def __call__(self, task, material, contract):
            if task == "revise_article_blocks":
                self.calls.append(task)
                return Reply("no-fix", {"patches": [], "unaddressed": [{"location": "proof", "issue": "Need original source evidence."}]})
            reply = super().__call__(task, material, contract)
            reply.value.update(verdict="needs_review", citations_supported=False,
                findings=[{"location": "proof", "issue": "Need original source evidence.", "check": "citations_supported", "severity": "blocking"}])
            return Reply("new-check-" + str(len(self.calls)), reply.value)

    agent = NoFixAgent(frozen)
    result = runtime.work(project.root, root.name, resume=True, agent=agent)
    assert result["state"] == "needs_review"
    assert agent.calls.count("revise_article_blocks") == 1
    assert read_json(root / "reassessment-status.json")["stop_reason"] == "no_supported_local_change"


def test_lite_layout_warning_does_not_corrupt_ready_receipt_revalidation(project, monkeypatch):
    root = runtime.prepare(project.root, [publish(project)])
    monkeypatch.setattr(render, "compile_pdf", lambda *_, **__: {"compiled": True, "max_overfull_pt": 8})
    result = runtime.work(project.root, root.name, agent=FakeAgent(evidence.load(root)), compile_output=True)
    assert result["state"] == "ready"
    assert read_json(root / "layout-review.json")["findings"][0]["severity"] == "advisory"
    assert runtime.status(project.root, root.name)["state"] == "ready"
