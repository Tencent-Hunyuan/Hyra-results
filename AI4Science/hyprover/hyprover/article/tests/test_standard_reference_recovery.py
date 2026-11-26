import copy
import shutil
import subprocess

import pytest

from hyprover.article import citation_policy, collection, evidence, references, render, response_validation, runtime
from hyprover.article.agent import Reply
from hyprover.article.models import ArticleConfig, ArticleError, EvidenceChanged, read_json, write_json
from hyprover.article.pipeline import Pipeline
from .test_audit import FakeAgent
from .test_citation_policy import material


def standard_check(rid, **changes):
    return dict({"obligation_id": "convergence", "reference_id": rid, "rationale": "A routine integration result.",
                 "standard_result": True, "supported": True}, **changes)


def pipeline_fixture(project, tmp_path):
    frozen, ledger, rid, _ = material(project)
    root = tmp_path / "article"
    evidence.persist(root, frozen)
    pipe = Pipeline(root, frozen, ArticleConfig(), agent=FakeAgent(frozen))
    pipe.ledger = ledger
    pipe.checkpoint.update(discovery_done=True, selection_done=True)
    pipe.save_ledger()
    return pipe, rid


@pytest.mark.parametrize("value", ["false", "None. This is not a textbook result.", 0, 1, None, [], {}])
@pytest.mark.parametrize("flag", ["standard_result", "supported"])
def test_nonboolean_reference_flags_have_precise_error_without_mutation(project, value, flag):
    _, ledger, rid, _ = material(project)
    original = copy.deepcopy(ledger)
    with pytest.raises(ArticleError, match="flags_must_be_boolean"):
        references.apply_standard_checks(ledger, [standard_check(rid, **{flag: value})])
    assert ledger == original


@pytest.mark.parametrize("negative_first", [True, False])
def test_standard_batch_failure_is_atomic_even_after_critical_promotion(project, negative_first):
    _, ledger, rid, _ = material(project)
    ledger["obligations"]["other"] = dict(ledger["obligations"]["convergence"], id="other", description="Another result")
    original = copy.deepcopy(ledger)
    rows = [standard_check(rid, standard_result=not negative_first, supported=not negative_first),
            standard_check(rid, obligation_id="other", standard_result="Nonroutine")]
    with pytest.raises(ArticleError, match="flags_must_be_boolean"):
        references.apply_standard_checks(ledger, rows)
    assert ledger == original


def test_duplicate_group_cannot_override_negative_assessment(project):
    _, ledger, rid, _ = material(project)
    ledger["obligations"]["duplicate"] = dict(ledger["obligations"]["convergence"], id="duplicate")
    original = copy.deepcopy(ledger)
    with pytest.raises(ArticleError, match="cannot_discharge|duplicate_standard"):
        references.apply_standard_checks(ledger, [standard_check(rid, standard_result=False, supported=False),
                                                  standard_check(rid, obligation_id="duplicate")])
    assert ledger == original


def test_standard_validation_respects_current_ledger_and_request_scope(project):
    _, ledger, rid, _ = material(project)
    request = {"reference_id": rid, "obligations": list(citation_policy.groups(ledger).values()), "ledger": ledger}
    for row, error in [(standard_check(rid, obligation_id="invented"), "out_of_scope"),
                       (standard_check("invented"), "out_of_scope"),
                       (standard_check(rid, standard_result="No routine result"), "flags_must_be_boolean")]:
        with pytest.raises(ArticleError, match=error):
            response_validation.validate("standard_reference", {"checks": [row]}, request)
    ledger["obligations"]["convergence"]["critical"] = True
    original = copy.deepcopy(ledger)
    with pytest.raises(ArticleError, match="cannot_discharge"):
        response_validation.validate("standard_reference", {"checks": [standard_check(rid)]}, request)
    assert ledger == original and not ledger["checks"]


def test_repair_is_bounded_keeps_raw_cache_and_does_not_send_whole_ledger(project, tmp_path):
    pipe, rid = pipeline_fixture(project, tmp_path)
    requests = []

    def agent(task, request, contract):
        assert task == "standard_reference" and "ledger" not in request
        assert "MUST be JSON booleans" in contract
        requests.append(copy.deepcopy(request))
        correction = request.get("format_correction")
        if correction:
            assert "flags_must_be_boolean" in correction["error"]
        return Reply("standard-" + str(len(requests)), {"checks": [standard_check(rid, standard_result=True if correction else "Yes")]})

    pipe.agent = agent
    pipe.check_standard_references()
    assert len(requests) == 2
    assert any(references.usable_check(pipe.ledger, cid) for cid in pipe.ledger["checks"])
    calls = [read_json(p) for p in (pipe.root / "calls").glob("*.json")]
    assert any(c["value"]["checks"][0]["standard_result"] == "Yes" for c in calls)
    pipe.check_standard_references()
    assert len(requests) == 2 and not pipe.ledger["unresolved_candidates"]


@pytest.mark.parametrize("compile_output", [False, pytest.param(True, marks=pytest.mark.skipif(
    not all(shutil.which(name) for name in ("bwrap", "pdflatex", "prlimit", "pdfinfo", "pdftotext")),
    reason="Optional isolated TeX and Poppler runtime unavailable"))])
def test_exhausted_standard_format_does_not_interrupt_manuscript_or_hide_gap(project, tmp_path, compile_output):
    pipe, rid = pipeline_fixture(project, tmp_path)
    pipe.compile_output = compile_output
    fake, calls = pipe.agent, []

    def agent(task, request, contract):
        if task == "standard_reference":
            calls.append(task)
            return Reply("bad-standard-" + str(len(calls)), {"checks": [standard_check(rid, standard_result="No routine result")]})
        return fake(task, request, contract)

    pipe.agent = agent
    report = pipe.run()
    assert report["state"] == "needs_review", report
    assert report["draft_complete"] and (pipe.root / "paper.tex").is_file()
    if compile_output:
        pdf = render.current_pdf(pipe.root)
        assert pdf and pdf["compiled"] and pdf["pages"] > 0
        text = subprocess.run(["pdftotext", str(pipe.root / "paper.pdf"), "-"],
                              capture_output=True, text=True, check=True).stdout
        assert "Introduction" in text and "Proof" in text
    assert len(calls) == pipe.config.max_revisions + 1
    assert not pipe.ledger["checks"] and not pipe.ledger.get("standard_assessments")
    unresolved = references.coverage(pipe.ledger)["blocking_candidates"]
    assert any(row.get("stage") == "standard_reference" and row["failures"] for row in unresolved)
    pipe.check_standard_references()
    assert len(calls) == pipe.config.max_revisions + 1


@pytest.mark.parametrize("critical", [False, True])
def test_background_uses_relevance_path_never_standard_result_path(project, tmp_path, critical):
    pipe, rid = pipeline_fixture(project, tmp_path)
    pipe.ledger["obligations"]["convergence"].update(kind="background", critical=critical)
    original = copy.deepcopy(pipe.ledger)
    tasks = []

    def agent(task, request, contract):
        tasks.append(task)
        assert task == "background_references"
        return Reply("background", {"checks": [{"obligation_id": "convergence", "reference_id": rid,
                                                "rationale": "Directly related reading only.", "supported": True}]})

    pipe.agent = agent
    pipe.check_standard_references()
    assert pipe.ledger == original and not tasks
    pipe.check_background_references()
    assert tasks == ["background_references"]
    assert pipe.ledger["obligations"] == original["obligations"]
    assert all(c["evidence_level"] == "background_reference" for c in pipe.ledger["checks"].values())
    assert not pipe.ledger.get("standard_assessments")


def test_integrity_errors_are_not_swallowed_as_literature_uncertainty(project, tmp_path):
    pipe, _ = pipeline_fixture(project, tmp_path)

    def agent(*args):
        raise EvidenceChanged("cached_article_response_tampered")

    pipe.agent = agent
    with pytest.raises(EvidenceChanged):
        pipe.check_standard_references()
    assert not pipe.ledger["unresolved_candidates"]


def test_current_revision_is_not_retried_but_new_fix_recovers_once(tmp_path, monkeypatch):
    task = tmp_path / "P/articles/run"
    task.mkdir(parents=True)
    revision = response_validation.RECOVERY_REVISION
    checkpoint = {"response_validation_revision": revision, "calls": 7, "elapsed_seconds": 120}
    write_json(task / "checkpoint.json", checkpoint)
    state = {"concurrency": 7, "retry_failed": True, "projects": {"P": {"directory": "P", "run_id": "run", "state": "failed"}}}
    monkeypatch.setattr(runtime, "job_path", lambda *args: task)
    monkeypatch.setattr(runtime, "status", lambda *args: {"state": "failed"})
    launched = []
    def launch(root, *, dispatch_id, **options):
        launched.append(options)
        return {"state": "failed", "dispatch_id": dispatch_id}

    monkeypatch.setattr(runtime, "launch", launch)
    assert collection._drive_locked(tmp_path, state) == 2
    assert not launched
    checkpoint["response_validation_revision"] = revision - 1
    write_json(task / "checkpoint.json", checkpoint)
    assert collection._drive_locked(tmp_path, state) == 2
    assert launched == [{"resume": True}]
    assert collection._drive_locked(tmp_path, state) == 2 and len(launched) == 1
    assert read_json(task / "checkpoint.json") == checkpoint
