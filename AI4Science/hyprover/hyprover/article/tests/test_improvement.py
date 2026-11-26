import copy
import json

import pytest

from hyprover.article import collection, evidence, improvement, literature, references, render, runtime
from hyprover.article.agent import Reply
from hyprover.article.models import ArticleConfig, ArticleError, Cancelled, digest, read_json, write_json
from hyprover.article.pipeline import Pipeline
from .conftest import publish
from .test_audit import FakeAgent
from .test_reference_refresh import ITEM


class MissingReferenceAgent(FakeAgent):
    def __call__(self, task, material, contract):
        reply = super().__call__(task, material, contract)
        if task == "plan_references":
            reply.value["obligations"] = [{"id": "related", "description": "The directly related fixture source",
                "source_units": [self.frozen["targets"][0] + ":statement"], "reference_ids": [],
                "kind": "background", "critical": True}]
        return reply


def setup(project, tmp_path, agent_type=MissingReferenceAgent, **config):
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    root = tmp_path / "article"
    evidence.persist(root, frozen)
    agent = agent_type(frozen)
    pipe = Pipeline(root, frozen, ArticleConfig(**config), agent=agent)
    return pipe, agent


def test_missing_citations_trigger_revisions_then_explicit_no_progress_stop(project, tmp_path):
    pipe, agent = setup(project, tmp_path, max_revisions=3)
    frozen = copy.deepcopy(pipe.evidence)
    report = pipe.run()
    assert report["state"] == "needs_review"
    assert not report["deterministic"]["passed"] and report["draft_complete"]
    assert agent.calls.count("revise_article") == improvement.MAX_STAGNANT_ROUNDS
    status = read_json(pipe.root / "improvement-status.json")
    assert status["stop_reason"] == "no_progress"
    assert status["completed"] == 2 and status["calls_used"] < status["call_budget"]
    assert pipe.evidence == frozen
    assert list((pipe.root / "improvements").glob("*/article.json"))
    spent, calls = pipe.checkpoint["calls"], list(agent.calls)
    resumed = Pipeline(pipe.root, pipe.evidence, pipe.config, agent=agent).run()
    assert resumed["state"] == "needs_review" and agent.calls == calls
    assert read_json(pipe.root / "checkpoint.json")["calls"] == spent


@pytest.mark.parametrize("max_revisions", [0, 1])
def test_automatic_revision_count_respects_configuration(project, tmp_path, max_revisions):
    pipe, agent = setup(project, tmp_path, max_revisions=max_revisions)
    assert pipe.run()["state"] == "needs_review"
    status = read_json(pipe.root / "improvement-status.json")
    assert agent.calls.count("revise_article") == status["completed"] == status["max_rounds"] == max_revisions
    assert status["stop_reason"] == ("round_limit" if max_revisions else "automatic_revision_disabled")


def test_supplement_new_evidence_insert_citation_and_re_review_until_ready(project, tmp_path, monkeypatch):
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "1")

    class Agent(MissingReferenceAgent):
        def __call__(self, task, material, contract):
            if task == "plan_review_references":
                self.calls.append(task)
                assert material["obligations"] and material["query_limit"] > 0
                return Reply("plan-" + str(len(self.calls)), {"checked_obligations": list(material["obligations"]),
                    "queries": ["directly related fixture theorem"], "candidates": []})
            if task == "select_focused_references":
                self.calls.append(task)
                rid, oid = next(iter(material["references"])), next(iter(material["obligations"]))
                return Reply("selection-" + str(len(self.calls)), {"checked_results": list(material["references"]),
                    "assignments": [{"reference_id": rid, "obligation_id": oid, "reason": "Direct fixture background"}],
                    "discarded": [], "unresolved": []})
            if task == "background_references":
                self.calls.append(task)
                return Reply("background-" + str(len(self.calls)), {"checks": [{"obligation_id": o["id"],
                    "reference_id": o["reference_ids"][0], "rationale": "The identified fixture topic matches.",
                    "supported": True} for o in material["obligations"]]})
            reply = super().__call__(task, material, contract)
            if task == "revise_article":
                article = reply.value["article"]
                cid = next(iter(material["citations"]))
                block = article["sections"][1]["blocks"][0]
                block["text"] += " Related background [[cite:" + cid + "]]."
                block["citations"] = [cid]
            return reply

    pipe, agent = setup(project, tmp_path, Agent)
    urls = []

    def get(url):
        pipe.reader.requests += 1
        urls.append(url)
        return json.dumps({"message": {"items": [ITEM]}}).encode(), "application/json", url

    monkeypatch.setattr(pipe.reader, "get", get)
    report = pipe.run()
    assert report["state"] == "ready", report
    assert len(urls) == 1 and pipe.reader.requests == 1
    assert agent.calls.count("compose_paper") == 1 and agent.calls.count("revise_article") == 1
    assert agent.calls.count("review_whole") == 2
    assert "\\bibitem" in (pipe.root / "references.tex").read_text()
    assert read_json(pipe.root / "improvement-status.json")["stop_reason"] == "ready"
    assert report["deterministic"]["citation_coverage"]["complete"]


def test_editorial_findings_get_revised_even_with_unresolved_external_evidence(project, tmp_path):
    class Agent(MissingReferenceAgent):
        def __call__(self, task, material, contract):
            reply = super().__call__(task, material, contract)
            if task.startswith("review_") and "revise_article" not in self.calls:
                reply.value.update(verdict="needs_review", concise=False,
                    findings=[{"location": "notation", "issue": "Remove duplicated exposition."}])
            if task == "revise_article":
                assert material["findings"] or material["round"] > 1
                reply.value["article"]["sections"][1]["blocks"][0]["text"] = "We use the original hypotheses."
            return reply

    pipe, agent = setup(project, tmp_path, Agent)
    report = pipe.run()
    state = read_json(pipe.root / "improvement-status.json")
    assert report["state"] == "needs_review" and not report["deterministic"]["citation_coverage"]["complete"]
    assert agent.calls.count("revise_article") >= 1
    assert state["history"][0]["progress"]
    assert state["stop_reason"] in ("round_limit", "no_progress")
    assert state["completed"] <= improvement.MAX_ROUNDS


def test_budget_failure_restores_prior_draft_and_pdf_without_resetting_costs(project, tmp_path, monkeypatch):
    pipe, agent = setup(project, tmp_path, max_calls=6)
    saved = {}
    original_start = improvement.Improvement.start

    def start(loop, article, report):
        if not saved:
            (pipe.root / "paper.pdf").write_bytes(b"original-pdf-fixture")
            saved.update(article=(pipe.root / "article.json").read_bytes(), pdf=(pipe.root / "paper.pdf").read_bytes(),
                         ledger=(pipe.root / "references.json").read_bytes(), tex=(pipe.root / "paper.tex").read_bytes())
        return original_start(loop, article, report)

    monkeypatch.setattr(improvement.Improvement, "start", start)
    report = pipe.run()
    assert report["state"] == "needs_review" and "call_budget" in report["error"]
    assert (pipe.root / "article.json").read_bytes() == saved["article"]
    assert (pipe.root / "references.json").read_bytes() == saved["ledger"]
    assert (pipe.root / "paper.tex").read_bytes() == saved["tex"]
    assert (pipe.root / "paper.pdf").read_bytes() == saved["pdf"]
    assert pipe.checkpoint["calls"] == 6
    assert read_json(pipe.root / "improvement-status.json")["stop_reason"] == "model_call_budget"


def test_interruption_resumes_same_round_and_keeps_paid_revision_cache(project, tmp_path):
    pipe, agent = setup(project, tmp_path, max_revisions=3)
    base_agent = pipe.agent

    def stop_at_review(task, material, contract):
        if task.startswith("review_") and material.get("review_generation") and "revise_article" in agent.calls:
            raise Cancelled("fixture-interruption")
        return base_agent(task, material, contract)

    pipe.agent = stop_at_review
    with pytest.raises(Cancelled):
        pipe.run()
    cp = read_json(pipe.root / "checkpoint.json")
    assert cp["auto_improvement"]["phase"] == "review"
    assert cp["auto_improvement"]["completed"] == 0
    calls_before = cp["calls"]
    report = Pipeline(pipe.root, pipe.evidence, pipe.config, agent=agent).run()
    assert report["state"] == "needs_review"
    assert agent.calls.count("revise_article") == 2
    assert read_json(pipe.root / "checkpoint.json")["calls"] >= calls_before
    state = read_json(pipe.root / "improvement-status.json")
    assert state["completed"] == 2 and state["stop_reason"] == "no_progress"


@pytest.mark.parametrize("stopped,cancelled", [(False, False), (True, False), (False, True)])
def test_collection_resumes_review_once_and_respects_stop_and_cancel(tmp_path, monkeypatch, stopped, cancelled):
    task = tmp_path / "P/articles/run"
    task.mkdir(parents=True)
    write_json(task / "article.json", {"fixture": True})
    write_json(task / "audit.json", {"state": "needs_review"})
    cp = {"calls": 17, "elapsed_seconds": 70, "auto_improvement": {"stop_reason": "no_progress"} if stopped else {}}
    write_json(task / "checkpoint.json", cp)
    if cancelled:
        write_json(task / "cancel.json", {"requested_at": 1})
    state = {"concurrency": 7, "projects": {"P": {"directory": "P", "run_id": "run", "state": "needs_review"}}}
    launches = []
    monkeypatch.setattr(runtime, "status", lambda *args: {"state": "needs_review"})
    monkeypatch.setattr(runtime, "job_path", lambda *args: task)
    def launch(path, *, dispatch_id, **options):
        launches.append(options)
        return {"state": "needs_review", "dispatch_id": dispatch_id}

    monkeypatch.setattr(runtime, "launch", launch)
    assert collection._drive_locked(tmp_path, state) == 2
    assert collection._drive_locked(tmp_path, state) == 2
    assert len(launches) == (0 if stopped or cancelled else 1)
    if launches:
        assert launches == [{"resume": True}]
    assert read_json(task / "checkpoint.json") == cp


def test_review_search_extends_first_six_only_within_total_budget(project):
    frozen = evidence.capture(project.root, [publish(project)])
    ledger = references.seed(frozen)
    ledger["focus"] = {"revision": 1, "searches": {str(i): {"state": "empty", "attempts": 1} for i in range(6)}}
    reader = references.SourceReader(ArticleConfig(max_queries=7), True)
    urls = []

    def get(url):
        reader.requests += 1
        urls.append(url)
        return b'{"message":{"items":[]}}', "application/json", url

    reader.get = get
    literature.focused_search(ledger, reader, ["new-query"], review=True)
    literature.focused_search(ledger, reader, ["new-query", "over-budget"], review=True)
    assert len(urls) == 1 and len(ledger["focus"]["searches"]) == 7
    assert not ledger["focus"]["search_complete"]
    assert "over-budget" in ledger["focus"]["skipped_queries"]


@pytest.mark.parametrize("bad", ["outside", "repeat", "too_many", "missing"])
def test_followup_plan_cannot_erase_obligations_or_reset_query_budget(bad):
    material = {"obligations": {"need": {}}, "query_limit": 1, "candidate_limit": 1, "previous_queries": ["old"]}
    plan = {"checked_obligations": ["need"], "queries": [], "candidates": []}
    if bad == "outside":
        plan["candidates"] = [{"metadata": {"title": "Fixture"}, "obligation_ids": ["unknown"]}]
    elif bad == "repeat":
        plan["queries"] = ["old"]
    elif bad == "too_many":
        plan["queries"] = ["new1", "new2"]
    else:
        plan["checked_obligations"] = []
    with pytest.raises(ArticleError):
        improvement.validate_plan(plan, material)


def test_reference_refresh_scope_never_rewrites_existing_proofs(project, tmp_path):
    pipe, agent = setup(project, tmp_path)
    pipe.refresh_references = True
    report = pipe.run()
    assert report["state"] == "needs_review"
    assert "revise_article" not in agent.calls
    state = read_json(pipe.root / "checkpoint.json")["auto_improvement"]
    assert state["completed"] == 0 and "stop_reason" not in state


def test_existing_review_is_verified_and_reused_only_before_new_round(project, tmp_path, monkeypatch):
    pipe, agent = setup(project, tmp_path)
    monkeypatch.setattr(improvement.Improvement, "start", lambda *args: False)
    first = pipe.run()
    assert first["state"] == "needs_review"
    resumed = Pipeline(pipe.root, pipe.evidence, pipe.config, agent=agent)
    article = read_json(pipe.root / "article.json")
    outputs = render.render(article, resumed.ledger)
    calls = len(agent.calls)
    assert resumed.review(article, outputs) == first
    assert len(agent.calls) == calls
    resumed.checkpoint["auto_improvement"]["phase"] = "review"
    resumed.checkpoint["review_generation"] = "new-round"
    resumed.review(article, outputs)
    assert len(agent.calls) > calls
