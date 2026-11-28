import copy
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from hyprover.article import audit, credit, evidence, improvement, literature, publication, references, render, runtime, writing_policy
from hyprover.article.agent import Reply
from hyprover.article.models import ArticleConfig, ArticleError, digest, read_json, write_json
from hyprover.article.pipeline import Pipeline
from .conftest import publish
from .test_audit import FakeAgent
from .test_pipeline import block
from .test_publication import fixture
from .test_reference_refresh import referenced


@pytest.mark.parametrize("fault", ["definition_first", "no_motivation", "theorem_after_heading", "no_strategy", "stacked_results"])
def test_narrative_is_checked_not_only_prompted(project, fault):
    _, _, _, plan, article = fixture(project)
    if fault == "definition_first":
        article["sections"][0]["blocks"].pop(0)
    elif fault == "no_motivation":
        article["sections"][0]["blocks"][1]["kind"] = "definition"
    elif fault == "theorem_after_heading":
        article["sections"][1]["blocks"][-1]["kind"] = "definition"
    elif fault == "no_strategy":
        article["sections"][3]["blocks"].pop(0)
    else:
        article["sections"][0]["blocks"].insert(3, copy.deepcopy(article["sections"][0]["blocks"][2]))
    with pytest.raises(ArticleError):
        publication.validate_article(article, plan)
    article["writing_revision"] = 1
    publication.validate_article(article, plan)


def test_handles_roundtrip_all_reference_fields_without_touching_math_or_crossrefs():
    mapping = {"C001": "cite-verified-original-long-identifier"}
    article = {"text": "Apply $C001$ [[cite:C001]] and [[ref:C001]].", "citations": ["C001"],
               "external_result": "C001", "source_map": [{"citations": ["C001"]}],
               "unplaced": [{"citation_id": "C001", "reason": "not used"}]}
    original = copy.deepcopy(article)
    restored = credit.translate(article, mapping, strict=True)
    assert "$C001$" in restored["text"] and "[[ref:C001]]" in restored["text"]
    assert restored["external_result"] == mapping["C001"]
    assert credit.translate(restored, {v: k for k, v in mapping.items()}) == original
    assert article == original
    for value in ({"text": "[[cite:C999]]"}, {"citations": ["C999"]}, {"citation_id": "C999"}):
        with pytest.raises(ArticleError, match="unknown_citation_handle"):
            credit.translate(value, mapping, strict=True)


def credit_pipe(project, tmp_path, background=False):
    frozen, ledger, article, cid = referenced(project, background=background)
    root = tmp_path / "credit-run"
    evidence.persist(root, frozen)
    pipe = Pipeline(root, frozen, ArticleConfig(), agent=FakeAgent(frozen))
    pipe.ledger = ledger
    pipe.save_ledger()
    return pipe, article, cid


def test_accounting_inserts_short_handles_and_rechecks_rewritten_article(project, tmp_path):
    pipe, article, cid = credit_pipe(project, tmp_path)
    received = []

    def agent(task, material, contract):
        assert task == "insert_citations"
        handle = material["required_placement"][0]
        assert handle.startswith("C") and len(handle) == 4
        assert material["citations"][handle]["bibliography"]["title"]
        received.append(copy.deepcopy(material))
        part = material["article"]["sections"][3]["blocks"][0]["parts"][0]
        return Reply("placement-" + str(len(received)), {"insertions": [
            {"block_id": part["id"], "after": part["text"], "citations": [handle]}], "background": [], "unplaced": []})

    pipe.agent = agent
    old = copy.deepcopy(article)
    fixed = pipe.insert_citations(article)
    assert len(received) == 1 and article == old
    assert cid in fixed["sections"][3]["blocks"][0]["parts"][0]["citations"]
    assert pipe.insert_citations(fixed) == fixed and len(received) == 1
    rewritten = copy.deepcopy(old)
    rewritten["abstract"]["text"] += " The argument is unchanged."
    assert pipe.insert_citations(rewritten) != rewritten and len(received) == 2
    assert all("C001" in str(read_json(p)["value"]) for p in (pipe.root / "calls").glob("*.json"))
    assert list((pipe.root / "citation-bindings").glob("*.json"))


def test_unplaced_is_explicit_and_bad_placement_does_not_interrupt_draft(project, tmp_path):
    pipe, article, cid = credit_pipe(project, tmp_path)
    calls = []

    def bad(task, material, contract):
        calls.append(task)
        return Reply("bad-" + str(len(calls)), {"insertions": [], "background": [], "unplaced": []})

    pipe.agent = bad
    fixed = pipe.insert_citations(article)
    assert fixed["sections"] == article["sections"]
    assert len(calls) == pipe.config.max_revisions + 1
    assert read_json(pipe.root / "citation-patch.json")["unresolved_placement"] == [cid]
    assert references.coverage(pipe.ledger)["blocking_candidates"]
    pipe.insert_citations(fixed)
    assert len(calls) == pipe.config.max_revisions + 1


def test_same_doi_is_one_bibliography_entry_but_checks_remain_distinct(project):
    _, ledger, article, cid = referenced(project)
    original_check = ledger["checks"][cid]
    rid = original_check["reference_id"]
    second = "ref-duplicate-record"
    ledger["references"][second] = dict(copy.deepcopy(ledger["references"][rid]), id=second)
    ledger["obligations"][original_check["obligation_id"]]["reference_ids"].append(second)
    original = ledger["sources"][original_check["source_id"]]
    second_source = references.add_source(ledger, second, original["text"],
                                          origin=original["origin"], kind="fulltext", version=original["version"])
    other = copy.deepcopy(original_check)
    other.update(id="cite-second", reference_id=second, source_id=second_source)
    ledger["checks"][other["id"]] = other
    before = copy.deepcopy(ledger)
    mapping = credit.bibliography_aliases(ledger, {cid, other["id"]})
    assert len(set(mapping.values())) == 1
    tex, bib = render.bibliography(ledger, {cid, other["id"]}, mapping)
    assert tex.count(r"\bibitem") == 1 and bib.count("@article{") == 1
    assert render.render_text("[[cite:" + cid + "]] [[cite:" + other["id"] + "]]", ledger, set(), mapping).count(next(iter(mapping.values()))) == 2
    assert ledger == before and len(credit.missing(ledger, set())) == 2
    ledger["references"][second]["metadata"]["edition"] = "Second"
    assert len(set(credit.bibliography_aliases(ledger, {cid, other["id"]}).values())) == 2


def test_identical_background_checks_are_accounted_once_not_repeated_citations(project):
    _, ledger, article, cid = referenced(project, background=True)
    duplicate = dict(ledger["checks"][cid], id="cite-duplicate")
    ledger["checks"][duplicate["id"]] = duplicate
    assert len(credit.missing(ledger, set())) == 1
    patch = {"insertions": [], "background": [], "unplaced": [{"citation_id": cid, "reason": "Only background; no relevant discussion in this fixture."}]}
    assert literature.apply_citation_patch(article, ledger, patch)["sections"] == article["sections"]


@pytest.mark.parametrize("phase", ["stopped", "references", "revision", "tampered"])
def test_nonzero_old_writing_upgrade_preserves_budget_and_archives_pending_round(project, tmp_path, phase):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    frozen = evidence.load(root)
    fake = FakeAgent(frozen)
    assert runtime.work(project.root, root.name, agent=fake)["state"] == "ready"
    article = read_json(root / "article.json")
    article["writing_revision"] = 1
    article["sections"][0]["blocks"] = article["sections"][0]["blocks"][2:]
    article["sections"][3]["blocks"].pop(0)
    article["source_map"] = [dict(row, blocks=[bid for bid in row["blocks"] if bid not in ("context", "motivation", "strategy")]) for row in article["source_map"]]
    write_json(root / "article.json", article)
    ledger = read_json(root / "references.json")
    render.write(root, render.render(article, ledger))
    cp = read_json(root / "checkpoint.json")
    cp["auto_improvement"] = {"revision": 1, "completed": 3, "stagnant": 0, "phase": "stopped", "history": [], "stop_reason": "round_limit"}
    if phase != "stopped":
        baseline = root / "improvements/round-1"
        baseline.mkdir(parents=True)
        write_json(baseline / "references.json", ledger)
        write_json(baseline / "article.json", article)
        cp["auto_improvement"] = {"revision": 1, "completed": 0, "stagnant": 0,
            "phase": "references" if phase == "tampered" else phase, "history": [],
            "baseline_article_digest": digest(article), "baseline_reference_digest": digest(ledger)}
        ledger["errors"].append({"kind": "fixture_unavailable", "error": "No new original text"})
        write_json(root / "references.json", ledger)
        if phase == "tampered":
            corrupted = copy.deepcopy(article)
            corrupted["title"]["text"] = "Tampered baseline"
            write_json(baseline / "article.json", corrupted)
    write_json(root / "checkpoint.json", cp)
    state = read_json(root / "status.json")
    state["state"] = "needs_review"
    write_json(root / "status.json", state)
    calls = cp["calls"]
    writer = FakeAgent(frozen)
    result = runtime.work(project.root, root.name, resume=True, upgrade_writing=True, agent=writer)
    if phase == "tampered":
        assert result["state"] == "stale" and result["error"] == "editorial_upgrade_baseline_changed"
        assert not writer.calls
        return
    assert result["state"] == "ready", result
    assert writer.calls[0] == "compose_paper"
    assert read_json(root / "checkpoint.json")["calls"] == calls + len(writer.calls)
    assert read_json(root / "article.json")["writing_revision"] == writing_policy.REVISION
    history = list((root / ("migrations/writing-" + str(writing_policy.REVISION))).glob("*/article.json"))
    assert any(read_json(p) == article for p in history)


@pytest.mark.skipif(not all(shutil.which(n) for n in ("bwrap", "pdflatex", "prlimit", "pdfinfo", "pdftotext")), reason="TeX/Poppler unavailable")
@pytest.mark.parametrize("revision", [2, 3])
def test_real_pdf_heading_and_step_follow_versioned_line_spacing(project, tmp_path, revision):
    _, ledger, _, _, article = fixture(project)
    article["writing_revision"] = revision
    proof = next(b for b in article["sections"][3]["blocks"] if b["kind"] == "proof")
    proof["parts"][0].update(kind="step", heading="SeparationMarker", text="BodyMarker starts the substantive argument.")
    render.write(tmp_path, render.render(article, ledger))
    compiled = render.compile_pdf(tmp_path)
    assert compiled["compiled"], compiled
    xml = subprocess.run(["pdftotext", "-bbox", str(tmp_path / "paper.pdf"), "-"], check=True, text=True, capture_output=True).stdout
    words = list(ET.fromstring(xml).iter("{http://www.w3.org/1999/xhtml}word"))
    heading = next(w for w in words if (w.text or "").startswith("SeparationMarker"))
    body = next(w for w in words if w.text == "BodyMarker")
    step_gap = float(body.attrib["yMin"]) - float(heading.attrib["yMax"])
    subsection_end = next(w for w in words if w.text == "objects.")
    following = words[words.index(subsection_end) + 1]
    subsection_gap = float(following.attrib["yMin"]) - float(subsection_end.attrib["yMax"])
    if revision >= 3:
        assert 0 < step_gap < 10 and 0 < subsection_gap < 10
        assert r"\vspace{\baselineskip}" not in (tmp_path / "paper.tex").read_text()
    else:
        assert step_gap >= 10 and subsection_gap >= 10


@pytest.mark.parametrize("text,expected", [("𝐿²", r"\ensuremath{L}\textsuperscript{2}"),
    ("Cα", r"C\ensuremath{\alpha}"), ("ℝ", r"\ensuremath{\mathbb{R}}"), (r"\input{secret}", r"\textbackslash{}input\{secret\}")])
def test_bibliographic_unicode_math_is_safe_and_does_not_execute_commands(text, expected):
    assert render.bibliography_text(text) == expected


@pytest.mark.skipif(not all(shutil.which(n) for n in ("bwrap", "pdflatex", "prlimit", "pdfinfo")), reason="TeX unavailable")
def test_real_unicode_math_bibliography_compiles(project, tmp_path):
    _, ledger, article, cid = referenced(project)
    rid = ledger["checks"][cid]["reference_id"]
    ledger["references"][rid]["metadata"]["title"] = "Regularity with 𝐿² drifts and Cα estimates"
    part = article["sections"][3]["blocks"][0]["parts"][0]
    part["text"] += " [[cite:" + cid + "]]"
    part["citations"] = [cid]
    article["writing_revision"] = 2
    outputs = render.render(article, ledger)
    assert "𝐿" not in outputs["references.tex"] and "𝐿" not in outputs["references.bib"]
    render.write(tmp_path, outputs)
    assert render.compile_pdf(tmp_path)["compiled"]


def test_two_independent_publication_faults_receive_precise_bounded_corrections(project, tmp_path):
    frozen, ledger, _, plan, article = fixture(project)
    metadata = {"title": "Fixture background", "authors": ["A. Author"], "year": 2020}
    rid = references.add_reference(ledger, metadata)
    sid = references.add_source(ledger, rid, "Fixture background. A. Author. 2020.", origin="fixture", kind="metadata")
    references.corroborate(ledger, rid, metadata, sid)
    references.add_obligations(ledger, [{"id": "context", "description": "Context of the setting", "reference_ids": [rid],
        "source_units": [frozen["targets"][0] + ":statement"], "kind": "background", "critical": False}], frozen)
    references.apply_background_checks(ledger, [{"obligation_id": "context", "reference_id": rid,
        "rationale": "Related fixture setting only.", "supported": True}])
    root = tmp_path / "bounded-editorial"
    evidence.persist(root, frozen)
    requests = []

    def agent(task, material, contract):
        assert task == "compose_paper"
        requests.append(material)
        value = copy.deepcopy(article)
        if len(requests) == 1:
            value["source_map"][0]["blocks"] = ["missing"]
        elif len(requests) == 2:
            part = next(b for b in value["sections"][3]["blocks"] if b["kind"] == "proof")["parts"][0]
            part["text"] += " For background see [[cite:C001]]."
            part["citations"] = ["C001"]
        else:
            assert len(requests) == 3
            assert "keep the mathematical proof unchanged" in material["format_correction"]["error"]
            value["sections"][0]["blocks"][0]["text"] += " For background see [[cite:C001]]."
            value["sections"][0]["blocks"][0]["citations"] = ["C001"]
        return Reply("composition-" + str(len(requests)), {"article": value})

    pipe = Pipeline(root, frozen, ArticleConfig(max_revisions=1), agent=agent)
    pipe.ledger = ledger
    result = pipe.ask("compose_paper", {"publication_plan": plan, "citations": pipe.citation_material()}, "Compose",
        validation_material={"evidence": frozen, "ledger": ledger, "publication_plan": plan})
    assert len(requests) == 3 and pipe.checkpoint["calls"] == 3
    assert audit.deterministic(result.value["article"], frozen, ledger)["draft_complete"]
