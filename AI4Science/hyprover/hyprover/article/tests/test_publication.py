import copy

import pytest

from hyprover.article import audit, evidence, publication, references, render, response_validation, runtime, writing_policy
from hyprover.article.agent import Reply
from hyprover.article.models import ArticleConfig, ArticleError, digest, read_json, write_json
from .conftest import prepare_legacy_review, publish
from .test_audit import FakeAgent, passed
from .test_pipeline import block
from .test_reference_refresh import referenced


def fixture(project):
    frozen = evidence.capture(project.root, [publish(project)])
    ledger = references.seed(frozen)
    material = {"original_units": list(evidence.units(frozen).values()), "targets": frozen["targets"],
                "obligations": ledger["obligations"]}
    fake = FakeAgent(frozen)
    plan = fake("plan_publication", material, "").value
    value = fake("compose_paper", {"reference_digest": digest(ledger), "publication_plan": plan}, "").value
    value, _ = response_validation.bind_article_metadata(value, {"evidence": frozen, "ledger": ledger, "publication_plan": plan})
    return frozen, ledger, material, plan, value["article"]


def test_plan_defines_used_preliminaries_and_ordered_core_before_composition(project):
    frozen, ledger, material, plan, article = fixture(project)
    publication.validate_plan(plan, material)
    publication.validate_article(article, plan)
    response_validation.validate("compose_paper", {"article": article},
        {"evidence": frozen, "ledger": ledger, "publication_plan": plan})
    tex = render.render(article, ledger)["paper.tex"]
    assert tex.index("Fixture objects") < tex.index("Fixture construction")
    assert "source_map" not in tex and "HYM" not in tex
    assert article["publication_plan_digest"] == digest(plan)


@pytest.mark.parametrize("fault", ["unknown_unit", "empty_prelim", "forward_dependency", "missing_target", "unknown_citation", "repeated_topic"])
def test_invalid_publication_plan_requires_correction(project, fault):
    _, _, material, plan, _ = fixture(project)
    if fault == "unknown_unit":
        plan["global_conventions"]["evidence"] = ["invented"]
    elif fault == "empty_prelim":
        plan["preliminaries"] = []
    elif fault == "forward_dependency":
        plan["core"][0]["requires"] = ["future-topic"]
    elif fault == "missing_target":
        plan["main_proofs"] = []
    elif fault == "unknown_citation":
        plan["preliminaries"][0]["citation_obligations"] = ["invented"]
    else:
        plan["core"][0]["id"] = plan["preliminaries"][0]["id"]
    with pytest.raises(ArticleError):
        publication.validate_plan(plan, material)


@pytest.mark.parametrize("fault", ["empty_topic", "no_conventions", "no_subsections", "no_roadmap"])
def test_four_headings_alone_are_not_a_publication_structure(project, fault):
    _, _, _, plan, article = fixture(project)
    if fault == "empty_topic":
        article["sections"][1]["blocks"].pop()
    elif fault == "no_conventions":
        article["sections"][1]["blocks"].pop(0)
    elif fault == "no_subsections":
        article["sections"][1]["blocks"] = article["sections"][1]["blocks"][:1]
    else:
        article["sections"][0]["blocks"].pop()
    with pytest.raises(ArticleError):
        publication.validate_article(article, plan)


def test_necessary_external_preliminary_is_not_forced_into_fake_internal_proof(project):
    frozen, ledger, _, cid = referenced(project)
    article = FakeAgent(frozen)("compose_paper", {"reference_digest": digest(ledger)}, "").value["article"]
    helper = block("borrowed-tool", "Every fixture has the property [[cite:" + cid + "]].",
                   [frozen["targets"][0] + ":proof:part-0001"], "lemma", external_result=cid)
    helper["citations"] = [cid]
    article["sections"][1]["blocks"].append(helper)
    assert audit.deterministic(article, frozen, ledger)["draft_complete"]
    ledger["checks"][cid]["evidence_level"] = "background_reference"
    with pytest.raises(ArticleError):
        audit.deterministic(article, frozen, ledger)


@pytest.mark.parametrize("check", writing_policy.REVIEW_CHECKS)
@pytest.mark.parametrize("fault", ["missing", "false"])
def test_added_quality_checks_are_required_in_independent_review(project, check, fault):
    frozen, ledger, _, _, article = fixture(project)
    expected = audit.expected_coverage(article, frozen, ledger, render.render(article, ledger), "whole",
        ["title", *[b["id"] for b in render.walk_blocks(article)]])
    receipt = passed(expected)
    assert all(k in receipt for k in writing_policy.REVIEW_CHECKS)
    if fault == "missing":
        receipt.pop(check)
        with pytest.raises(ArticleError):
            audit.receipt_outcome(receipt, expected)
    else:
        receipt[check] = False
        assert audit.receipt_outcome(receipt, expected) == "needs_review"


def test_internal_benchmark_names_never_become_public_context(project):
    frozen, _, _, _, article = fixture(project)
    article["abstract"]["text"] = "This paper solves HYM999."
    with pytest.raises(ArticleError, match="internal_evidence_id"):
        publication.check_internal_ids(article, frozen)


def test_actual_typesetting_failure_enters_revision_even_when_math_review_passes(project, monkeypatch):
    root = prepare_legacy_review(project.root, [publish(project)])
    fake = FakeAgent(evidence.load(root))
    compiled = []

    def compile_pdf(path, **_):
        compiled.append(path)
        return {"compiled": True, "max_overfull_pt": 40 if len(compiled) == 1 else 0}

    class Agent:
        def __call__(self, task, material, contract):
            if task == "revise_article":
                assert "typesetting" in material["findings"]
            return fake(task, material, contract)

    monkeypatch.setattr(render, "compile_pdf", compile_pdf)
    result = runtime.work(project.root, root.name, agent=Agent(), compile_output=True)
    assert result["state"] == "ready", result
    assert fake.calls.count("revise_article") == 1 and len(compiled) == 2
    assert fake.calls.count("review_whole") == 2


def test_plan_tampering_is_rejected_on_status_and_resume(project):
    root = runtime.prepare(project.root, [publish(project)])
    result = runtime.work(project.root, root.name, agent=FakeAgent(evidence.load(root)))
    assert result["state"] == "ready"
    plan = read_json(root / "publication-plan.json")
    plan["plan"]["global_conventions"]["text"] = "tampered"
    write_json(root / "publication-plan.json", plan)
    assert runtime.status(project.root, root.name)["state"] == "stale"
    with pytest.raises(ArticleError, match="publication_plan_changed"):
        runtime.current(root)


@pytest.mark.parametrize("title", [r"Spaces $E$", r"The \\R case", "Facts [[cite:source]]", "Tools [[ref:core]]", "Two\nlines"])
def test_topic_titles_cannot_print_unrendered_math_or_reference_tokens(project, title):
    _, _, material, plan, article = fixture(project)
    plan["preliminaries"][0]["title"] = title
    with pytest.raises(ArticleError, match="plain_text"):
        publication.validate_plan(plan, material)
    heading = next(b for b in article["sections"][1]["blocks"] if b["kind"] == "subsection")
    heading["text"] = title
    with pytest.raises(ArticleError, match="plain_text"):
        publication.validate_article(article, plan)


@pytest.mark.parametrize("field,value", [("evidence", []), ("citations", ["source"]), ("crossrefs", ["core"])])
def test_topic_headings_require_evidence_and_leave_citations_in_body(project, field, value):
    _, _, _, plan, article = fixture(project)
    heading = next(b for b in article["sections"][1]["blocks"] if b["kind"] == "subsection")
    heading[field] = value
    with pytest.raises(ArticleError, match="heading_requires_evidence"):
        publication.validate_article(article, plan)


@pytest.mark.parametrize("role,maximum", [("preliminaries", 2), ("core", 3)])
def test_compact_topic_limits_in_plan_and_manuscript_without_dropping_evidence(project, role, maximum):
    frozen, ledger, material, plan, article = fixture(project)
    material["writing_revision"] = 3
    section = next(s for s in article["sections"] if s["role"] == role)
    source_row = copy.deepcopy(plan[role][0])
    for index in range(maximum):
        row = dict(copy.deepcopy(source_row), id=role + "-extra-" + str(index), title="Additional topic " + str(index))
        plan[role].append(row)
        section["blocks"].extend([block(row["id"], row["title"], row["evidence"], "subsection"),
                                  block(row["id"] + "-body", "This passage develops the same original material.", row["evidence"])])
        if index == maximum - 2:
            publication.validate_plan(plan, material)
            publication.validate_article(article, plan)
            assert audit.deterministic(article, frozen, ledger)["draft_complete"]
    with pytest.raises(ArticleError, match="publication_plan_requires_topic_groups"):
        publication.validate_plan(plan, material)
    with pytest.raises(ArticleError, match="publication_subsection_limit_exceeded"):
        publication.validate_article(article, plan)
    article["writing_revision"] = 2
    publication.validate_plan(plan, dict(material, writing_revision=2))
    publication.validate_article(article, plan)


@pytest.mark.parametrize("role", ["introduction", "main_proofs"])
def test_compact_layout_cannot_hide_extra_subsections_elsewhere(project, role):
    _, _, _, plan, article = fixture(project)
    section = next(s for s in article["sections"] if s["role"] == role)
    section["blocks"].insert(0, block("extra-heading", "Unneeded subdivision", article["title"]["evidence"], "subsection"))
    with pytest.raises(ArticleError, match="publication_subsection_limit_exceeded"):
        publication.validate_article(article, plan)


def test_current_publication_plan_records_compact_policy_for_resume(project):
    root = runtime.prepare(project.root, [publish(project)])
    frozen = evidence.load(root)
    agent = FakeAgent(frozen)
    result = runtime.work(project.root, root.name, agent=agent)
    assert result["state"] == "ready", result
    saved = read_json(root / "publication-plan.json")
    assert saved["writing_revision"] == writing_policy.REVISION == 3
    assert len(saved["plan"]["preliminaries"]) <= 2 and len(saved["plan"]["core"]) <= 3
    calls = list(agent.calls)
    assert runtime.work(project.root, root.name, resume=True, agent=agent)["state"] == "ready"
    assert agent.calls == calls
