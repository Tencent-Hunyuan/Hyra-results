import copy

import pytest

from hyprover.article import evidence, references, response_validation
from hyprover.article.agent import Reply
from hyprover.article.models import ArticleConfig, ArticleError
from hyprover.article.pipeline import Pipeline
from .conftest import publish
from .test_audit import FakeAgent


def make_pipeline(project, tmp_path, agent=None):
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    root = tmp_path / "article"
    evidence.persist(root, frozen)
    return Pipeline(root, frozen, ArticleConfig(), agent=agent or FakeAgent(frozen))


def selection(material, complete=True):
    return {"checked_results": list(material["results"]), "selected": [], "discarded": [],
            "unresolved": [{"arxiv_id": aid, "reason": "Original publication not yet checked."} for aid in material["results"]],
            "materials_complete": complete, "truncated": False}


def test_selection_repairs_flag_semantics_without_erasing_uncertainty(project, tmp_path):
    pipe = make_pipeline(project, tmp_path)
    calls = []

    def agent(task, material, contract):
        calls.append(material)
        result = selection(material, complete="format_correction" in material)
        return Reply("session-" + str(len(calls)), result)

    pipe.agent = agent
    pipe.ledger["searches"] = [{"query": "fixture", "result": {"results": [{"arxiv_id": "2301.12345v1"}]}}]
    pipe.select_search_results()
    assert len(calls) == 2
    assert "only the supplied search metadata" in calls[1]["format_correction"]["error"]
    assert pipe.checkpoint["selection_done"]
    assert pipe.checkpoint["literature_selection"]["materials_complete"] is True
    assert len(pipe.ledger["unresolved_candidates"]) == 1
    assert not references.coverage(pipe.ledger)["complete"]
    pipe.checkpoint["selection_done"] = False
    pipe.select_search_results()
    assert len(calls) == 2


def test_selection_exhaustion_still_writes_needs_review_draft(project, tmp_path):
    pipe = make_pipeline(project, tmp_path)
    original_agent = pipe.agent
    attempts = []

    def agent(task, material, contract):
        if task == "select_related_literature":
            attempts.append(task)
            return Reply("selection-" + str(len(attempts)), selection(material, complete=False))
        return original_agent(task, material, contract)

    pipe.agent = agent
    pipe.discover()
    pipe.ledger.pop("focus", None)
    pipe.ledger["searches"] = [{"query": "fixture", "result": {"results": [{"arxiv_id": "2301.12345v1"}]}}]
    pipe.save_ledger()
    report = pipe.run()
    assert report["state"] == "needs_review", report
    assert (pipe.root / "paper.tex").is_file()
    assert len(attempts) == 2
    assert pipe.checkpoint["literature_selection"]["materials_complete"] is False
    assert references.coverage(pipe.ledger)["blocking_candidates"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "invented", "truncated"])
def test_selection_cannot_hide_coverage_errors(change):
    material = {"results": {"a": {}, "b": {}}, "units": []}
    value = selection(material)
    if change == "missing":
        value["unresolved"].pop()
    elif change == "duplicate":
        value["unresolved"].append(dict(value["unresolved"][0]))
    elif change == "invented":
        value["unresolved"][0]["arxiv_id"] = "not-supplied"
    else:
        value["truncated"] = True
    with pytest.raises(ArticleError):
        response_validation.validate("select_related_literature", value, material)


def test_whitespace_matching_keeps_original_offsets_and_rejects_math_changes():
    text = "Heading\nLemma 1\n\nFor x < 1,\n\n\nthe result holds.\nEnding"
    quote = "Lemma 1\n\nFor x < 1,\n\nthe result holds."
    span = references.source_span(text, quote)
    assert span["match"] == "whitespace"
    assert text[span["start"]:span["end"]] == "Lemma 1\n\nFor x < 1,\n\n\nthe result holds."
    for bad in (quote.replace("<", ">"), quote.replace("1", "2"), "Forx < 1,", "invented"):
        with pytest.raises(ArticleError):
            references.source_span(text, bad)


def test_control_escapes_are_rejected_and_only_source_backed_commands_can_be_restored():
    from hyprover.article import render
    damaged = "Use \\(\frac{1}{2}\\)."
    with pytest.raises(ArticleError, match="unescaped_control"):
        render.safe_tex(damaged)
    fixed, changes = render.restore_json_escapes(damaged, r"The proof uses \frac1{2}.")
    assert fixed == r"Use \(\frac{1}{2}\)."
    assert changes == [{"offset": 6, "control_code": 12, "command": "frac"}]
    render.safe_tex(fixed)
    with pytest.raises(ArticleError, match="not_grounded"):
        render.restore_json_escapes(damaged, "No matching command in the source.")
    with pytest.raises(ArticleError, match="unrecoverable"):
        render.restore_json_escapes("bad\x01text", "source")


def test_source_classification_is_bounded_correctable_and_atomic(project, tmp_path):
    pipe = make_pipeline(project, tmp_path)
    rid = references.add_reference(pipe.ledger, {"title": "Fixture book"})
    text = "Fixture book\nLemma 1\n\nThe result holds.\n\n\nProof. Apply H."
    sid = references.add_source(pipe.ledger, rid, text, origin="project:literature/fixture.txt", kind="unconfirmed")
    original = copy.deepcopy(pipe.ledger["sources"][sid])
    calls = []

    def agent(task, material, contract):
        calls.append(material)
        value = {k: material[k] for k in ("source_id", "source_sha256", "reference_id")}
        value.update(document_type="fulltext", identity_supported=True, outside_abstract=True,
                     body_quote="The result holds.\n\nProof. Apply H.", locator="Lemma 1")
        if "format_correction" not in material:
            value["body_quote"] = "Invented quotation"
        return Reply("source-" + str(len(calls)), value)

    pipe.agent = agent
    pipe.classify_sources()
    assert len(calls) == 2
    source = pipe.ledger["sources"][sid]
    assert source["kind"] == "fulltext"
    assert source["sha256"] == original["sha256"]
    span = source["classification_spans"]["body_quote"]
    assert source["classification"]["body_quote"] == text[span["start"]:span["end"]]
    assert source["classification_original"]["body_quote"] != source["classification"]["body_quote"]
    assert pipe.checkpoint["calls"] == 2


def test_source_excerpt_does_not_accept_quote_from_unseen_tail():
    text = "Fixture book\nLemma 1. Supplied text."
    material = {"source_id": "s", "source_sha256": "hash", "reference_id": "r",
                "origin": "project:literature/fixture.txt", "text": text, "range_start": 0, "range_end": len(text)}
    value = {k: material[k] for k in ("source_id", "source_sha256", "reference_id")}
    value.update(document_type="fulltext", identity_supported=True, outside_abstract=True,
                 body_quote="Only available later in the publication.", locator="Lemma 1")
    with pytest.raises(ArticleError, match="source_quote_not_found"):
        response_validation.validate("classify_source", value, material)


def test_source_classification_searches_beyond_front_matter(project, tmp_path):
    pipe = make_pipeline(project, tmp_path)
    rid = references.add_reference(pipe.ledger, {"title": "Long fixture book"})
    front = "Long fixture book\n" + "Front matter.\n" * 1300
    text = front + "Theorem 3.\nThe substantive result appears here."
    sid = references.add_source(pipe.ledger, rid, text, origin="project:literature/long.txt", kind="unconfirmed")
    calls = []

    def agent(task, material, contract):
        calls.append(material["range_start"])
        found = "Theorem 3." in material["text"]
        if found:
            assert "Long fixture book" in material["identity_header"]
        value = {key: material[key] for key in ("source_id", "source_sha256", "reference_id")}
        value.update(document_type="fulltext" if found else "unknown", identity_supported=found,
                     outside_abstract=found, body_quote="The substantive result appears here." if found else "",
                     locator="Theorem 3." if found else "")
        return Reply("source-range-" + str(len(calls)), value)

    pipe.agent = agent
    pipe.classify_sources()
    assert calls[0] == 0 and calls[-1] > 0
    assert references.source_is_fulltext(pipe.ledger, sid, rid)
    report = references.source_classification(pipe.ledger["sources"][sid], rid)
    assert report["locator"] == "Theorem 3."
    checked = list(calls)
    pipe.classify_sources()
    assert calls == checked
