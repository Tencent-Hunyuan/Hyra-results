import pytest

from hyprover.article import audit, evidence, references, render, writing_policy
from hyprover.article.models import ArticleError, digest

from .conftest import complete, publish
from .test_pipeline import article_fixture, block


@pytest.fixture
def manuscript(project):
    first = publish(project, statement="The fixture premise.", proof="Here is its argument.",
                    definitions={"fixture term": "A definition used in this fixture."})
    target = publish(project, predecessors=[first], statement="The fixture conclusion.",
                     proof="Here is the final argument.")
    complete(project, [target])
    frozen = evidence.capture(project.root)
    ledger = references.seed(frozen)
    article = article_fixture(frozen, ledger)
    article["writing_revision"] = writing_policy.REVISION
    intro, prelim, core, proofs = [section["blocks"] for section in article["sections"]]
    intro[:] = [block("context", "This fixture studies the supplied argument.", [target + ":statement"])]
    prelim[:] = [block("notation", "Use the fixture term as defined.", [first + ":definition:0001"], "definition")]
    core[:] = [block("result", "The fixture premise and conclusion both hold.",
                     [first + ":statement", target + ":statement"], "theorem")]
    duplicated = "The original proof applies in this case. " * 8
    proofs[:] = [
        block("proof", "We establish both assertions.", [first + ":statement"], "proof", proves="result", parts=[
            block("first-part", duplicated, [first + ":proof:part-0001"]),
            block("final-part", duplicated, [target + ":proof:part-0001"]),
        ]),
        block("bridge", "Together, these give the fixture target.", ["completion:bridge"]),
    ]
    article["source_map"] = [
        {"unit_id": uid, "disposition": "developed", "citations": [], "reason": "Integrated in the manuscript.",
         "blocks": [piece["id"] for piece in render.walk_blocks(article) if uid in piece["evidence"]]}
        for uid in evidence.units(frozen)
    ]
    return frozen, ledger, article


def _coverage(frozen, ledger, article):
    return audit.deterministic(article, frozen, ledger, review_mode=audit.FORMAT_COVERAGE)


def test_format_coverage_accepts_merged_result_without_editorial_or_citation_gates(manuscript):
    frozen, ledger, article = manuscript
    checked = _coverage(frozen, ledger, article)
    assert checked["checks"] == {"format": True, "coverage": True}
    assert not checked["citation_coverage"]["complete"]
    assert checked["statement_to_proof"] == {"result": ["proof"]}
    assert checked["missing_evidence"] == []
    strict = audit.deterministic(article, frozen, ledger)
    assert "principal-result:" + frozen["targets"][0] in strict["missing_evidence"]
    assert strict["editorial_metrics"]["duplicate_substantive_passages"] == 1
    assert not strict["passed"]
    outputs = render.render(article, ledger)
    report = audit.finalize(article, frozen, ledger, outputs, {}, [], review_mode=audit.FORMAT_COVERAGE)
    assert report["state"] == "ready"
    assert report["review_mode"] == "format-coverage"
    assert report["checks"] == checked["checks"]
    assert report["rendered_digest"] == digest(outputs)
    assert report["receipts"] == {}
    assert report["formal_certificate"] is False
    changed = {**outputs, "paper.tex": "different"}
    with pytest.raises(ArticleError, match="article_rendered_output_mismatch"):
        audit.finalize(article, frozen, ledger, changed, {}, [], review_mode=audit.FORMAT_COVERAGE)
    with pytest.raises(ArticleError, match="unexpected_model_receipts"):
        audit.finalize(article, frozen, ledger, outputs, {"whole": {}}, [], review_mode=audit.FORMAT_COVERAGE)


def test_original_omissions_reported_by_kind(manuscript):
    frozen, ledger, article = manuscript
    first = frozen["facts"][0]["fact_id"]
    target = frozen["targets"][0]
    rows = {row["unit_id"]: row for row in article["source_map"]}
    # This source map places the prerequisite definition and closing bridge only in the abstract.
    for uid in (first + ":definition:0001", "completion:bridge"):
        article["abstract"]["evidence"].append(uid)
        rows[uid]["blocks"] = ["abstract"]
    # This claim is mapped only to an introductory paragraph. The proof target still carries the first statement.
    rows[target + ":statement"]["blocks"] = ["context"]
    # This child result's proof text is attached to a proof whose result now only carries the first statement.
    checked = _coverage(frozen, ledger, article)
    assert checked["checks"] == {"format": True, "coverage": False}
    assert checked["missing_by_kind"] == {
        "statement": [target + ":statement"], "definition": [first + ":definition:0001"],
        "proof": [target + ":proof:part-0001"], "bridge": ["completion:bridge"],
    }
    report = audit.finalize(article, frozen, ledger, render.render(article, ledger), {}, [],
                            review_mode=audit.FORMAT_COVERAGE)
    assert report["state"] == "needs_review"
    assert report["manuscript_available"] is True
    assert report["failures"] == ["incomplete_original_source_coverage"]


def test_source_map_and_reference_targets_have_to_resolve(manuscript):
    frozen, ledger, article = manuscript
    row = next(row for row in article["source_map"] if row["unit_id"] == "completion:bridge")
    row["blocks"] = ["result"]
    with pytest.raises(ArticleError, match="publication_source_map_not_bound_to_text"):
        _coverage(frozen, ledger, article)
    row["blocks"] = ["bridge"]
    article["sections"][3]["blocks"][0]["proves"] = "core"
    with pytest.raises(ArticleError, match="proof_has_no_result_statement"):
        _coverage(frozen, ledger, article)
    article["sections"][3]["blocks"][0]["proves"] = "result"
    article["abstract"]["text"] += " [[ref:unknown]]"
    article["abstract"]["crossrefs"] = ["unknown"]
    with pytest.raises(ArticleError, match="dangling_article_cross_reference"):
        _coverage(frozen, ledger, article)


def test_unresolved_citations_are_a_format_error(manuscript):
    frozen, ledger, article = manuscript
    article["abstract"]["text"] += " [[cite:unknown]]"
    article["abstract"]["citations"] = ["unknown"]
    with pytest.raises(ArticleError, match="unverified_article_citation: block=abstract"):
        _coverage(frozen, ledger, article)


def test_reference_marker_alone_does_not_cover_original_body(manuscript):
    frozen, ledger, article = manuscript
    bridge = article["sections"][3]["blocks"][1]
    bridge.update(text="[[ref:result]]", crossrefs=["result"])
    checked = _coverage(frozen, ledger, article)
    assert checked["checks"] == {"format": True, "coverage": False}
    assert checked["missing_evidence"] == ["completion:bridge"]


@pytest.mark.parametrize("invalid", [r"\[x=x\]", r"\begin{align*}x&=x\end{align*}"])
def test_implicit_equation_display_rejects_nested_display_but_allows_aligned(manuscript, invalid):
    frozen, ledger, article = manuscript
    part = article["sections"][3]["blocks"][0]["parts"][0]
    part.update(kind="equation", text=r"\begin{aligned}x&=x\\[1pt]y&=y\end{aligned}")
    assert _coverage(frozen, ledger, article)["passed"]
    part["text"] = invalid
    with pytest.raises(ArticleError, match="nested_tex_display_math"):
        _coverage(frozen, ledger, article)
