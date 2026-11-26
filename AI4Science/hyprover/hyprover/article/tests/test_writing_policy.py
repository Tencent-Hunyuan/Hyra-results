import copy
import json

import pytest

from hyprover.article import audit, collection, delivery, evidence, references, render, runtime, writing_policy as writing
from hyprover.article.agent import Reply
from hyprover.article.models import ArticleConfig, ArticleError, FORMAT, digest, read_json, write_json
from hyprover.article.pipeline import Pipeline
from .conftest import prepare_legacy_review, publish
from .test_audit import FakeAgent, passed
from .test_pipeline import article_fixture, block


def paper_fixture(project):
    first = publish(project, statement="Under H, A holds.", proof="Apply the fixture identity under H.")
    second = publish(project, statement="Under H, B holds.", proof="Combine A with the fixture estimate.", predecessors=[first])
    target = publish(project, statement="Under H, C holds.", proof="Use A and B to conclude C.", predecessors=[second])
    frozen = evidence.capture(project.root, [target])
    ledger = references.seed(frozen)
    ledger["discovery_complete"] = True
    units = evidence.units(frozen)
    statement = target + ":statement"
    deps = [uid for uid, u in units.items() if u["kind"] == "statement" and uid != statement]
    proofs = [uid for uid, u in units.items() if u["kind"] == "proof"]
    article = {"format": FORMAT, "writing_revision": writing.REVISION, "evidence_digest": frozen["digest"],
        "reference_digest": digest(ledger), "title": {"text": "A concise fixture result", "evidence": [statement]},
        "abstract": block("abstract", "Under H we establish C using the fixture estimate.", [statement]),
        "keywords": ["fixture"], "msc": [], "sections": [
            {"id": "intro", "role": "introduction", "title": "Introduction", "blocks": [
                block("main", "Under H, C holds.", [statement], "theorem")]},
            {"id": "prelim", "role": "preliminaries", "title": "Preliminaries", "blocks": [
                block("hypotheses", "Throughout we assume H.", deps)]},
            {"id": "core", "role": "core", "title": "The common estimate", "blocks": [
                block("auxiliary", "The same estimate establishes A and B under H.", deps)]},
            {"id": "proofs", "role": "main_proofs", "title": "Proofs of the main results", "blocks": [
                block("main-proof", "Proof of the main theorem.", [statement], "proof", proves="main", parts=[
                    block("combined-argument", "Apply the identity under H to obtain A, use the estimate for B, and conclude C.", proofs)])]}]}
    remap(article, frozen)
    return frozen, ledger, article


def remap(article, frozen):
    blocks = list(render.walk_blocks(article))
    article["source_map"] = [{"unit_id": uid, "disposition": "merged", "blocks": [b["id"] for b in blocks if uid in b["evidence"]],
                              "citations": [], "reason": "The common argument includes this premise or deduction."}
                             for uid in evidence.units(frozen)]


def test_dependencies_can_be_inlined_without_numbering_every_fact(project):
    frozen, ledger, article = paper_fixture(project)
    report = audit.deterministic(article, frozen, ledger)
    assert report["draft_complete"] and report["passed"]
    assert report["editorial_metrics"]["original_facts"] == 3
    assert report["editorial_metrics"]["numbered_results"] == 1
    tex = render.render(article, ledger)["paper.tex"]
    assert tex.count(r"\begin{theorem}") == 1 and r"\begin{lemma}" not in tex
    assert "source_map" not in tex and frozen["targets"][0] not in tex


def test_multiple_dependency_statements_can_form_one_useful_lemma(project):
    frozen, ledger, article = paper_fixture(project)
    core = article["sections"][2]
    core["blocks"][0].update(id="common-lemma", kind="lemma")
    dependent_proofs = [uid for uid, u in evidence.units(frozen).items()
                        if u["kind"] == "proof" and u["fact_id"] not in frozen["targets"]]
    core["blocks"].append(block("common-proof", "Prove both auxiliary claims together.", core["blocks"][0]["evidence"],
                               "proof", proves="common-lemma", parts=[block("aux-proof", "Use the identity and estimate under H.", dependent_proofs)]))
    main_part = article["sections"][3]["blocks"][0]["parts"][0]
    main_part["evidence"] = [uid for uid in main_part["evidence"] if uid not in dependent_proofs]
    remap(article, frozen)
    assert audit.deterministic(article, frozen, ledger)["draft_complete"]
    assert render.render(article, ledger)["paper.tex"].count(r"\begin{lemma}") == 1


@pytest.mark.parametrize("mistake", ["missing_unit", "abstract_only", "new_result", "unverified_citation", "wrong_block"])
def test_concision_never_waives_evidence_coverage(project, mistake):
    frozen, ledger, article = paper_fixture(project)
    uid = next(uid for uid, u in evidence.units(frozen).items() if u["kind"] == "proof")
    row = next(r for r in article["source_map"] if r["unit_id"] == uid)
    if mistake == "missing_unit":
        article["source_map"].remove(row)
    elif mistake == "abstract_only":
        for b in render.walk_blocks(article):
            b["evidence"] = [u for u in b["evidence"] if u != uid]
        article["abstract"]["evidence"].append(uid)
        row["blocks"] = ["abstract"]
    elif mistake == "new_result":
        article["sections"][2]["blocks"][0].update(kind="lemma", evidence=[uid])
    elif mistake == "unverified_citation":
        row["disposition"] = "cited"
    else:
        row["blocks"] = ["nonexistent"]
    try:
        checked = audit.deterministic(article, frozen, ledger)
    except ArticleError:
        return
    assert not checked["draft_complete"]


def test_review_requires_concision_and_structure_and_all_original_units(project):
    frozen, ledger, article = paper_fixture(project)
    outputs = render.render(article, ledger)
    job = audit.review_jobs(article)[-1]
    expected = audit.expected_coverage(article, frozen, ledger, outputs, job["id"], job["blocks"])
    assert expected["checked_units"] == sorted(evidence.units(frozen))
    assert expected["writing_revision"] == writing.REVISION
    result = passed(expected)
    assert audit.receipt_outcome(result, expected) == "passed"
    result["concise"] = False
    assert audit.receipt_outcome(result, expected) == "needs_review"
    result.pop("publication_structure")
    with pytest.raises(ArticleError):
        audit.receipt_outcome(result, expected)


def test_new_composition_reads_originals_once_and_keeps_map_outside_prose(project, tmp_path):
    frozen, ledger, article = paper_fixture(project)
    root = tmp_path / "paper-run"
    evidence.persist(root, frozen)
    received = []

    def agent(task, material, contract):
        if task == "plan_publication":
            return FakeAgent(frozen)(task, material, contract)
        assert task == "compose_paper"
        received.append(material)
        assert {u["id"]: u["text"] for u in material["original_units"]} == {
            uid: u["text"] for uid, u in evidence.units(frozen).items()}
        assert "not a transcript" in material["writing_policy"]
        result = copy.deepcopy(article)
        result["sections"][0]["blocks"][:0] = [
            block("context", "The setting concerns the fixture implications under H.", result["title"]["evidence"]),
            block("motivation", "We seek the conclusion C from the two intermediate facts.", result["title"]["evidence"])]
        result["sections"][3]["blocks"].insert(0, block("strategy", "Combine the established facts under H.", result["title"]["evidence"]))
        for role, index in (("preliminaries", 1), ("core", 2)):
            for topic in material["publication_plan"][role]:
                result["sections"][index]["blocks"].extend([
                    block(topic["id"], topic["title"], topic["evidence"], "subsection"),
                    block(topic["id"] + "-body", "We develop the fixture argument under H.", topic["evidence"])])
        route = block("roadmap", "See [[ref:prelim]], [[ref:core]], then [[ref:proofs]].", result["title"]["evidence"])
        route["crossrefs"] = ["prelim", "core", "proofs"]
        result["sections"][0]["blocks"].append(route)
        remap(result, frozen)
        return Reply("writer-paper", {"article": result})

    pipe = Pipeline(root, frozen, ArticleConfig(), agent=agent)
    pipe.ledger = ledger
    composed = pipe.load_or_write_article()
    assert composed["sections"][3]["blocks"][1:] == article["sections"][3]["blocks"] and len(received) == 1
    assert (root / "source-map.json").is_file() and (root / "publication-plan.json").is_file()
    assert pipe.load_or_write_article() == composed and len(received) == 1


def test_legacy_upgrade_preserves_inputs_budget_draft_and_requests_fresh_reviews(project):
    fid = publish(project)
    root = prepare_legacy_review(project.root, [fid])
    frozen = evidence.load(root)
    ledger = references.seed(frozen)
    ledger["discovery_complete"] = True
    old = article_fixture(frozen, ledger)
    render.write(root, render.render(old, ledger))
    write_json(root / "article.json", old)
    write_json(root / "references.json", ledger)
    write_json(root / "audit.json", {"policy_revision": 2, "state": "needs_review", "receipts": {}})
    state = read_json(root / "status.json")
    state.update(state="needs_review")
    write_json(root / "status.json", state)
    manifest_before = (root / "manifest.json").read_bytes()
    manifest = runtime.load_manifest(root)
    pipe = Pipeline(root, frozen, ArticleConfig.from_dict(manifest["config"]), supplied=manifest["supplied"])
    pipe.checkpoint.update(calls=11, elapsed_seconds=120, discovery_done=True, selection_done=True, references_done=True)
    write_json(root / "checkpoint.json", pipe.checkpoint)
    delivery.export(project.root, root.name)
    old_export = (project.root / "paper/paper.tex").read_bytes()
    fake = FakeAgent(frozen)
    result = runtime.work(project.root, root.name, resume=True, upgrade_writing=True, agent=fake)
    assert result["state"] == "ready", result
    assert fake.calls[:2] == ["plan_publication", "compose_paper"] and all(x.startswith("review_") for x in fake.calls[2:])
    assert read_json(root / "checkpoint.json")["calls"] == 11 + len(fake.calls)
    assert (root / "manifest.json").read_bytes() == manifest_before
    assert list((root / ("migrations/writing-" + str(writing.REVISION))).glob("*/article.json"))
    assert (project.root / "paper/paper.tex").read_bytes() == old_export
    provenance = delivery.export(project.root, root.name)
    assert provenance["writing_revision"] == writing.REVISION
    assert list((project.root / "paper/history").glob("*/provenance.json"))
    before = len(fake.calls)
    runtime.work(project.root, root.name, resume=True, upgrade_writing=True, agent=fake)
    assert len(fake.calls) == before


def test_collection_waits_for_old_worker_then_upgrades_once(project, monkeypatch):
    fid = publish(project)
    task = runtime.prepare(project.root, [fid])
    state = {"concurrency": 1, "writing_revision": writing.REVISION, "projects": {
        "P": {"directory": project.root.name, "run_id": task.name}}}
    seen, phase = [], ["running"]
    monkeypatch.setattr(runtime, "status", lambda *a: {"state": phase[0]})
    def sleep(_):
        assert not seen
        phase[0] = "needs_review"
    def launch(root, *, dispatch_id, **options):
        seen.append(options)
        return {"state": "failed", "dispatch_id": dispatch_id}
    monkeypatch.setattr(collection.time, "sleep", sleep)
    monkeypatch.setattr(runtime, "launch", launch)
    assert collection._drive_locked(project.root.parent, state) == 2
    assert seen == [{"resume": True, "upgrade_writing": True}]
    assert collection._drive_locked(project.root.parent, state) == 2
    assert len(seen) == 1


def test_updated_audit_is_delivered_even_when_pdf_bytes_are_unchanged(tmp_path, monkeypatch):
    write_json(tmp_path / "audit.json", {"state": "needs_review"})
    record = {"run_id": "run", "state": "needs_review"}
    monkeypatch.setattr(runtime, "job_path", lambda *a: tmp_path)
    monkeypatch.setattr(render, "pdf_sources", lambda root: {"paper.tex": "fixed"})
    monkeypatch.setattr(render, "current_pdf", lambda root: {"compiled": True, "pdf_sha256": "same-pdf"})
    seen = []
    monkeypatch.setattr(delivery, "export", lambda *a: seen.append(1) or {"pdf_sha256": "same-pdf"})
    collection._deliver_pdf(tmp_path, record)
    collection._deliver_pdf(tmp_path, record)
    assert len(seen) == 1
    write_json(tmp_path / "audit.json", {"state": "ready"})
    collection._deliver_pdf(tmp_path, record)
    assert len(seen) == 2
