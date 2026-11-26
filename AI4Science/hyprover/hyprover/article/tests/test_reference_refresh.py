import copy
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from hyprover.article import audit, collection, evidence, literature, references as refs, render, runtime
from hyprover.article.agent import Reply
from hyprover.article.models import ArticleConfig, ArticleError, Cancelled, digest, read_json, write_json
from hyprover.article.pipeline import Pipeline
from .conftest import prepare_legacy_review, publish
from .test_audit import FakeAgent
from .test_pipeline import fixture


META = {"title": "A related fixture study", "authors": ["Ada Example"], "year": 2020, "doi": "10.1234/fixture"}
ITEM = {"title": [META["title"]], "author": [{"given": "Ada", "family": "Example"}],
        "published": {"date-parts": [[2020]]}, "DOI": META["doi"], "type": "journal-article"}


def referenced(project, *, background=False):
    frozen, ledger, article = fixture(project)
    fid = frozen["targets"][0]
    rid = refs.add_reference(ledger, META)
    text = "Theorem 1. Every fixture has the property."
    sid = refs.add_source(ledger, rid, text, origin="https://api.crossref.org/works/fixture", kind="metadata" if background else "fulltext")
    refs.corroborate(ledger, rid, META, sid)
    oid = "important-reference"
    refs.add_obligations(ledger, [{"id": oid, "description": "Related research" if background else "Borrowed result",
        "source_units": [fid + (":statement" if background else ":proof:part-0001")], "reference_ids": [rid],
        "kind": "background" if background else "mathematical", "critical": True}], frozen)
    if background:
        refs.apply_background_checks(ledger, [{"obligation_id": oid, "reference_id": rid,
            "rationale": "Verified title identifies directly related material.", "supported": True}])
    else:
        refs.apply_checks(ledger, [{"obligation_id": oid, "reference_id": rid, "source_id": sid,
            "quote": "Every fixture has the property.", "locator": "Theorem 1", "premise_comparison": "Conditions match.", "supported": True}])
    return frozen, ledger, article, next(iter(ledger["checks"]))


def test_focused_search_uses_prioritized_bibliographic_queries_and_cache(project):
    frozen = evidence.capture(project.root, [publish(project)])
    ledger = refs.seed(frozen)
    ledger["focus"] = {"revision": 1}
    calls = []

    class Reader(refs.SourceReader):
        def get(self, url):
            self.requests += 1
            calls.append(url)
            return json.dumps({"message": {"items": [ITEM]}}).encode(), "application/json", url

    reader = Reader(ArticleConfig(), True)
    queries = ["origin of the fixture problem", "directly related fixture estimates"]
    literature.focused_search(ledger, reader, queries)
    literature.focused_search(ledger, reader, queries)
    assert [parse_qs(urlsplit(url).query)["query.bibliographic"][0] for url in calls] == queries
    assert len(ledger["references"]) == 1
    assert next(iter(ledger["references"].values()))["metadata_verified"]
    assert all(s["kind"] == "metadata" for s in ledger["sources"].values())
    assert not ledger["checks"] and ledger["focus"]["search_complete"]


def test_search_retries_are_bounded_and_failure_is_not_completeness(project):
    frozen = evidence.capture(project.root, [publish(project)])
    ledger = refs.seed(frozen)
    ledger["focus"] = {"revision": 1, "selection_done": True}
    ledger["discovery_complete"] = True
    calls = []

    class Reader(refs.SourceReader):
        def get(self, url):
            calls.append(url)
            self.requests += 1
            raise OSError("offline fixture")

    reader = Reader(ArticleConfig(), True)
    for _ in range(3):
        literature.focused_search(ledger, reader, ["main result origin"])
    assert len(calls) == 2
    assert not ledger["focus"]["search_complete"]
    assert not refs.coverage(ledger)["complete"]
    reader.allowed = False
    literature.focused_search(ledger, reader, ["another query"])
    assert len(calls) == 2


def test_arxiv_identity_reuses_existing_metadata_after_topic_budget_exhausted(project, monkeypatch):
    frozen = evidence.capture(project.root, [publish(project)])
    ledger = refs.seed(frozen)
    rid = refs.add_reference(ledger, dict(META, doi="", arxiv="2301.12345v1"))
    metadata = {"arxiv_id": "2301.12345v1", "title": META["title"], "authors": META["authors"],
                "published": "2020-01-01", "url": "https://arxiv.org/abs/2301.12345v1"}
    ledger["searches"] = [{"query": "topic", "result": {"results": [metadata]}}]
    reader = refs.SourceReader(ArticleConfig(max_queries=1), True)
    monkeypatch.setattr(reader, "get_text", lambda url: (_ for _ in ()).throw(OSError("offline")))
    refs.retrieve(ledger, reader, search=lambda *a, **k: pytest.fail("should reuse known identity"))
    assert ledger["references"][rid]["metadata_verified"]


def test_patch_only_inserts_citation_at_real_use_and_preserves_proof(project):
    frozen, ledger, article, cid = referenced(project)
    part = article["sections"][3]["blocks"][0]["parts"][0]
    original = copy.deepcopy(article)
    patch = {"insertions": [{"block_id": part["id"], "after": part["text"], "citations": [cid]}], "background": [], "unplaced": []}
    changed = literature.apply_citation_patch(article, ledger, patch)
    changed_part = changed["sections"][3]["blocks"][0]["parts"][0]
    assert changed_part["text"] == part["text"] + " [[cite:" + cid + "]]"
    assert article == original
    stripped = copy.deepcopy(changed)
    stripped["sections"][3]["blocks"][0]["parts"][0] = original["sections"][3]["blocks"][0]["parts"][0]
    stripped["reference_digest"] = original["reference_digest"]
    assert stripped == original
    assert audit.deterministic(changed, frozen, ledger)["passed"]
    assert "\\bibitem" in render.render(changed, ledger)["references.tex"]
    patch["insertions"][0]["after"] = "invented anchor"
    with pytest.raises(ArticleError, match="anchor"):
        literature.apply_citation_patch(article, ledger, patch)


@pytest.mark.parametrize("has_roadmap", [True, False])
def test_background_patch_preserves_closing_roadmap_and_all_existing_proofs(project, has_roadmap):
    from .test_pipeline import block
    frozen, ledger, article, cid = referenced(project, background=True)
    uid = frozen["targets"][0] + ":statement"
    intro = article["sections"][0]["blocks"]
    route_ids = [s["id"] for s in article["sections"][1:]]
    if has_roadmap:
        roadmap = block("route", "Then read " + ", ".join("[[ref:" + sid + "]]" for sid in route_ids) + ".", [uid])
        roadmap["crossrefs"] = route_ids
        intro.append(roadmap)
    original = copy.deepcopy(article)
    backgrounds = [block("background-" + str(i), "Related fixture context [[cite:" + cid + "]].", [uid]) for i in range(2)]
    for row in backgrounds:
        row["citations"] = [cid]
    updated = literature.apply_citation_patch(article, ledger, {"insertions": [], "background": backgrounds, "unplaced": []})
    changed_intro = updated["sections"][0]["blocks"]
    expected_ids = [b["id"] for b in intro[:-1]] + [b["id"] for b in backgrounds] + ["route"] if has_roadmap else [b["id"] for b in intro + backgrounds]
    assert [b["id"] for b in changed_intro] == expected_ids
    assert article == original
    assert [b for b in changed_intro if b["id"] not in {row["id"] for row in backgrounds}] == intro
    assert updated["sections"][1:] == original["sections"][1:]
    assert audit.deterministic(updated, frozen, ledger)["passed"]


def test_metadata_related_work_cannot_certify_a_proof_or_priority(project):
    frozen, ledger, article, cid = referenced(project, background=True)
    part = article["sections"][3]["blocks"][0]["parts"][0]
    patch = {"insertions": [{"block_id": part["id"], "after": part["text"], "citations": [cid]}], "background": [], "unplaced": []}
    with pytest.raises(ArticleError, match="cannot_support_proof"):
        literature.apply_citation_patch(article, ledger, patch)
    for kind in ("historical", "comparison", "mathematical"):
        ledger["obligations"]["important-reference"]["kind"] = kind
        assert not refs.usable_check(ledger, cid)


def test_plan_covers_paper_not_separate_elementary_queries(project):
    frozen = evidence.capture(project.root, [publish(project)])
    units = list(evidence.units(frozen).values())
    material = {"original_units": units, "existing_references": {}}
    value = {"checked_units": [u["id"] for u in units], "materials_complete": True, "truncated": False,
        "candidates": [{"key": "important", "metadata": META}], "obligations": [],
        "queries": ["main problem author title"],
        "scope_notes": {k: "No claim in this fixture." for k in ("problem_origin", "related_work", "borrowed_results")}}
    literature.validate_plan(value, material)
    value["queries"] = ["query " + str(i) for i in range(7)]
    with pytest.raises(ArticleError, match="prioritized"):
        literature.validate_plan(value, material)


def test_refresh_does_not_rewrite_or_reset_budget_and_keeps_legacy_issues(project):
    fid = publish(project)
    root = prepare_legacy_review(project.root, [fid])
    frozen = evidence.load(root)
    fake = FakeAgent(frozen)
    runtime.work(project.root, root.name, agent=fake)
    article = read_json(root / "article.json")
    ledger = read_json(root / "references.json")
    ledger.pop("focus")
    ledger["unresolved_candidates"].append({"reason": "existing important attribution gap", "critical": True})
    article["reference_digest"] = digest(ledger)
    write_json(root / "article.json", article)
    write_json(root / "references.json", ledger)
    render.write(root, render.render(article, ledger))
    state = read_json(root / "status.json")
    state["state"] = "needs_review"
    write_json(root / "status.json", state)
    before = read_json(root / "checkpoint.json")["calls"]
    manifest = (root / "manifest.json").read_bytes()
    calls = len(fake.calls)
    result = runtime.work(project.root, root.name, resume=True, refresh_references=True, agent=fake)
    assert result["state"] == "needs_review"
    assert fake.calls[calls] == "plan_references"
    assert "compose_paper" not in fake.calls[calls:]
    assert "rewrite_proof" not in fake.calls[calls:] and "revise_article" not in fake.calls[calls:]
    assert read_json(root / "article.json")["sections"] == article["sections"]
    assert read_json(root / "checkpoint.json")["calls"] >= before
    assert (root / "manifest.json").read_bytes() == manifest
    assert (root / "migrations/references-1/references.json").exists()
    assert read_json(root / "references.json")["unresolved_candidates"]
    count = len(fake.calls)
    runtime.work(project.root, root.name, resume=True, refresh_references=True, agent=fake)
    assert len(fake.calls) == count


def test_modern_refresh_repeats_queries_once_per_slot_and_recovers_before_rebinding(project, monkeypatch):
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "1")
    fid = publish(project)
    root = runtime.prepare(project.root, [fid], config=ArticleConfig(max_queries=2))
    query = "fixture problem origin"
    urls = []

    class Agent(FakeAgent):
        def __call__(self, task, material, contract):
            if task == "background_references":
                self.calls.append(task)
                return Reply("background-recheck", {"checks": []})
            reply = super().__call__(task, material, contract)
            if task == "plan_references":
                reply.value["queries"] = [query]
            return reply

    def get(reader, url):
        reader.requests += 1
        urls.append(url)
        return b'{"message":{"items":[]}}', "application/json", url

    monkeypatch.setattr(refs.SourceReader, "get", get)
    agent = Agent(evidence.load(root))
    assert runtime.work(project.root, root.name, agent=agent)["state"] == "ready"
    baseline = read_json(root / "article.json")
    original = read_json(root / "references.json")
    prior_searches = copy.deepcopy(original["focus"]["searches"])
    rid = refs.add_reference(original, META)
    sid = refs.add_source(original, rid, json.dumps(ITEM), origin="https://api.crossref.org/works/fixture", kind="metadata")
    refs.corroborate(original, rid, META, sid)
    refs.add_obligations(original, [{"id": "previous-context", "description": "Prior bibliography context",
        "source_units": [fid + ":statement"], "reference_ids": [rid], "kind": "background", "critical": False}], agent.frozen)
    refs.apply_background_checks(original, [{"obligation_id": "previous-context", "reference_id": rid,
        "rationale": "The book directly concerns the stated fixture background.", "supported": True}])
    cid = next(iter(original["checks"]))
    context = baseline["sections"][0]["blocks"][0]
    assert context["kind"] == "paragraph"
    context["text"] += " [[cite:" + cid + "]]"
    context["citations"].append(cid)
    baseline["reference_digest"] = digest(original)
    write_json(root / "references.json", original)
    write_json(root / "article.json", baseline)
    render.write(root, render.render(baseline, original))
    load = Pipeline.load_or_write_article
    with monkeypatch.context() as interrupted:
        def cancel_before_binding(pipe):
            raise Cancelled("fixture interruption after refreshing references")

        interrupted.setattr(Pipeline, "load_or_write_article", cancel_before_binding)
        state = runtime.work(project.root, root.name, resume=True, refresh_references=True, agent=agent)
        assert state["state"] == "cancelled"
    assert read_json(root / "article.json") == baseline
    state = runtime.work(project.root, root.name, resume=True, agent=agent)
    assert state["state"] == "ready", state
    ledger = read_json(root / "references.json")
    focus = ledger["focus"]
    assert focus["searches"] == prior_searches
    assert focus["refreshes"][0]["searches"][query]["state"] == "empty"
    assert literature.focused_queries_used(focus) == 2
    assert ledger["checks"][cid] == original["checks"][cid] and refs.usable_check(ledger, cid)
    assert read_json(root / "checkpoint.json")["network_requests"] == 2
    assert [parse_qs(urlsplit(url).query)["query.bibliographic"][0] for url in urls] == [query, query]
    refreshed = read_json(root / "article.json")
    assert refreshed["reference_digest"] == digest(ledger)
    assert refreshed["sections"] == baseline["sections"]
    assert agent.calls.count("compose_paper") == 1 and "revise_article" not in agent.calls
    state = runtime.work(project.root, root.name, resume=True, refresh_references=True, agent=agent)
    assert state["reference_refresh"] == "budget_exhausted" and len(urls) == 2


def test_explicit_reference_refresh_preserves_pending_proof_revision(project):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    write_json(root / "checkpoint.json", {"auto_improvement": {"phase": "revision"}})
    ns = root / "status.json"
    before = ns.read_bytes()
    with pytest.raises(ArticleError, match="resume_without_refresh_references"):
        runtime.work(project.root, root.name, resume=True, refresh_references=True)
    with pytest.raises(ArticleError, match="resume_without_refresh_references"):
        runtime.launch(root, resume=True, refresh_references=True)
    assert ns.read_bytes() == before


def test_collection_refresh_waits_and_is_once_without_proof_upgrade(project, monkeypatch):
    fid = publish(project)
    task = runtime.prepare(project.root, [fid])
    write_json(task / "article.json", {"writing_revision": 1})
    state = {"concurrency": 1, "references_revision": 1, "writing_revision": 1,
        "projects": {"P": {"directory": project.root.name, "run_id": task.name}}}
    phase, calls = ["running"], []
    monkeypatch.setattr(runtime, "status", lambda *a: {"state": phase[0]})
    monkeypatch.setattr(collection.time, "sleep", lambda _: phase.__setitem__(0, "needs_review"))
    def launch(root, *, dispatch_id, **options):
        calls.append(options)
        return {"state": "failed", "dispatch_id": dispatch_id}

    monkeypatch.setattr(runtime, "launch", launch)
    assert collection._drive_locked(project.root.parent, state) == 2
    assert calls == [{"resume": True, "refresh_references": True}]
    collection._drive_locked(project.root.parent, state)
    assert len(calls) == 1
