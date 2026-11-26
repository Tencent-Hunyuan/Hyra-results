import copy

import pytest

from hyprover.article import audit, evidence, publication, references, render, runtime, writing_policy
from hyprover.article.agent import Reply
from hyprover.article.models import ArticleConfig, ArticleError, EvidenceChanged, ResponseContractError, digest, read_json
from hyprover.article.pipeline import Pipeline
from .conftest import complete, prepare_legacy_review, publish
from .test_audit import FakeAgent
from .test_pipeline import block


def completed_fixture(project, tmp_path, *, multiple_targets=False):
    proof = "Apply the given identity under the stated assumptions. " * 30
    target = publish(project, proof=proof, definitions={"X": "The fixture object under its stated assumptions."})
    targets = [target]
    if multiple_targets:
        targets.append(publish(project, statement="Second fixture conclusion.", proof="Apply the second fixture identity."))
    complete(project, targets)
    config = ArticleConfig(chunk_chars=256)
    frozen = evidence.capture(project.root, config=config)
    ledger = references.seed(frozen)
    request = {"original_units": list(evidence.units(frozen).values()), "targets": frozen["targets"],
               "obligations": ledger["obligations"], "writing_revision": writing_policy.REVISION}
    plan = FakeAgent(frozen)("plan_publication", request, publication.PLAN_CONTRACT).value
    path = tmp_path / "publication-run"
    evidence.persist(path, frozen)
    pipe = Pipeline(path, frozen, config, agent=FakeAgent(frozen))
    return pipe, request, plan


def remove_assignment(plan, uid, request):
    replacement = next(u["id"] for u in request["original_units"] if u["id"] != uid)
    rows = [plan["global_conventions"], *plan["notation"], *plan["preliminaries"], *plan["core"], *plan["main_proofs"]]
    for row in rows:
        for field in ("evidence", "used_by"):
            if field in row:
                row[field] = [value for value in row[field] if value != uid] or [replacement]


@pytest.mark.parametrize("kind", ["statement", "definition", "proof", "bridge"])
@pytest.mark.parametrize("location", ["checked_units", "assignment"])
def test_diagnostics_identify_exact_missing_units_of_every_kind(project, tmp_path, kind, location):
    _, request, plan = completed_fixture(project, tmp_path)
    assert {u["kind"] for u in request["original_units"]} == {"statement", "definition", "proof", "bridge"}
    assert sum(u["kind"] == "proof" for u in request["original_units"]) > 1
    uid = next(u["id"] for u in request["original_units"] if u["kind"] == kind)
    if location == "checked_units":
        plan[location].remove(uid)
    else:
        remove_assignment(plan, uid, request)
    with pytest.raises(ArticleError) as error:
        publication.validate_plan(plan, request)
    assert uid in str(error.value)
    if location == "assignment":
        assert isinstance(error.value, publication.PlanCoverageError)
        assert error.value.missing_units == [uid] and not error.value.missing_targets


@pytest.mark.parametrize("multiple_targets", [False, True])
def test_only_unassigned_checked_bridge_is_routed_to_final_assembly(project, tmp_path, multiple_targets):
    pipe, request, plan = completed_fixture(project, tmp_path, multiple_targets=multiple_targets)
    remove_assignment(plan, "completion:bridge", request)
    original = copy.deepcopy(plan)
    fixed, changes = publication.bind_completion_bridge(plan, request)
    assert plan == original and fixed is not plan
    assert len(changes) == len(request["targets"])
    assert fixed["checked_units"] == original["checked_units"]
    for i, row in enumerate(fixed["main_proofs"]):
        assert row["evidence"] == original["main_proofs"][i]["evidence"] + ["completion:bridge"]
        assert row["strategy"] == original["main_proofs"][i]["strategy"]
        assert row["uses"] == original["main_proofs"][i]["uses"]
        assert changes[i]["source_sha256"] == evidence.units(pipe.evidence)["completion:bridge"]["sha256"]
    for key in ("global_conventions", "notation", "preliminaries", "core"):
        assert fixed[key] == original[key]
    publication.validate_plan(fixed, request)
    assert publication.bind_completion_bridge(fixed, request) == (fixed, [])


@pytest.mark.parametrize("fault", ["unchecked_bridge", "unknown_unit", "missing_proof", "missing_target", "unknown_target",
                                   "unknown_citation", "forward_dependency", "too_many_topics", "spoofed_bridge"])
def test_bridge_binding_does_not_mask_other_plan_errors(project, tmp_path, fault):
    _, request, plan = completed_fixture(project, tmp_path)
    remove_assignment(plan, "completion:bridge", request)
    if fault == "unchecked_bridge":
        plan["checked_units"].remove("completion:bridge")
    elif fault == "unknown_unit":
        plan["core"][0]["evidence"].append("invented")
    elif fault == "missing_proof":
        remove_assignment(plan, next(u["id"] for u in request["original_units"] if u["kind"] == "proof"), request)
    elif fault == "missing_target":
        plan["main_proofs"] = []
    elif fault == "unknown_target":
        plan["main_proofs"][0]["target_id"] = "invented"
    elif fault == "unknown_citation":
        plan["preliminaries"][0]["citation_obligations"] = ["invented"]
    elif fault == "forward_dependency":
        plan["core"][0]["requires"] = ["unknown-topic"]
    elif fault == "too_many_topics":
        plan["preliminaries"] *= 3
    else:
        next(u for u in request["original_units"] if u["id"] == "completion:bridge")["kind"] = "proof"
    before = copy.deepcopy(plan)
    with pytest.raises(ArticleError):
        publication.bind_completion_bridge(plan, request)
    assert plan == before


def test_missing_target_diagnostic_is_distinct_from_missing_unit(project, tmp_path):
    _, request, plan = completed_fixture(project, tmp_path)
    plan["main_proofs"] = []
    with pytest.raises(publication.PlanCoverageError) as error:
        publication.validate_plan(plan, request)
    assert error.value.missing_targets == request["targets"]
    assert not error.value.missing_units


def test_bound_plan_preserves_raw_reply_and_reuses_cache_without_extra_model_call(project, tmp_path):
    pipe, request, plan = completed_fixture(project, tmp_path)
    remove_assignment(plan, "completion:bridge", request)
    calls = []

    def agent(task, material, contract):
        calls.append(task)
        return Reply("planner", copy.deepcopy(plan))

    pipe.agent = agent
    fixed = pipe.ask("plan_publication", request, publication.PLAN_CONTRACT).value
    assert calls == ["plan_publication"] and pipe.checkpoint["calls"] == 1
    records = list((pipe.root / "calls").glob("*.json"))
    assert len(records) == 1 and read_json(records[0])["value"] == plan
    receipt_path = next((pipe.root / "plan-bindings").glob("*.json"))
    receipt = read_json(receipt_path)
    assert receipt["original_value_digest"] == digest(plan)
    assert receipt["bound_plan_digest"] == digest(fixed)
    assert receipt["evidence_digest"] == pipe.evidence["digest"]
    assert receipt["manuscript_coverage_still_required"] and receipt["independent_review_required"]
    assert receipt["mathematical_text_rewritten"] is False
    assert pipe.ask("plan_publication", request, publication.PLAN_CONTRACT).value == fixed
    assert len(calls) == 1 and read_json(records[0])["value"] == plan
    assert read_json(receipt_path) == receipt


def test_actual_proof_omission_is_corrected_with_precise_feedback_not_automatically_filled(project, tmp_path):
    pipe, request, plan = completed_fixture(project, tmp_path)
    uid = next(u["id"] for u in request["original_units"] if u["kind"] == "proof")
    damaged = copy.deepcopy(plan)
    remove_assignment(damaged, uid, request)
    calls = []

    def agent(task, material, contract):
        calls.append(material)
        if "format_correction" in material:
            assert uid in material["format_correction"]["error"]
            assert material["format_correction"]["previous_response"] == damaged
            return Reply("corrected-plan", copy.deepcopy(plan))
        return Reply("bad-plan", copy.deepcopy(damaged))

    pipe.agent = agent
    assert pipe.ask("plan_publication", request, publication.PLAN_CONTRACT).value == plan
    assert len(calls) == pipe.config.max_revisions + 1 == 2
    assert not (pipe.root / "plan-bindings").exists()


def test_plan_correction_exhaustion_remains_bounded_and_reports_missing_proof(project, tmp_path):
    pipe, request, plan = completed_fixture(project, tmp_path)
    uid = next(u["id"] for u in request["original_units"] if u["kind"] == "proof")
    remove_assignment(plan, uid, request)
    pipe.agent = lambda *args: Reply("bad-plan", copy.deepcopy(plan))
    with pytest.raises(ResponseContractError, match=uid):
        pipe.ask("plan_publication", request, publication.PLAN_CONTRACT)
    assert pipe.checkpoint["calls"] == pipe.config.max_revisions + 1


def test_bridge_binding_rejects_changed_source_context(project, tmp_path):
    pipe, request, plan = completed_fixture(project, tmp_path)
    remove_assignment(plan, "completion:bridge", request)
    request["original_units"][0]["text"] += " Changed."
    pipe.agent = lambda *args: Reply("planner", copy.deepcopy(plan))
    with pytest.raises(EvidenceChanged, match="publication_plan_validation_context_changed"):
        pipe.ask("plan_publication", request, publication.PLAN_CONTRACT)
    assert pipe.checkpoint["calls"] == 1 and not (pipe.root / "plan-bindings").exists()


@pytest.mark.parametrize("truncated_review", [False, True])
def test_completion_bridge_still_requires_manuscript_coverage_and_independent_review(project, truncated_review):
    fid = publish(project)
    complete(project, [fid])
    root = prepare_legacy_review(project.root)
    frozen = evidence.load(root)

    class CompletedAgent(FakeAgent):
        def __call__(self, task, material, contract):
            reply = super().__call__(task, material, contract)
            if task == "plan_publication":
                remove_assignment(reply.value, "completion:bridge", material)
            elif task == "compose_paper":
                article = reply.value["article"]
                article["sections"][3]["blocks"].append(block("assembly", "The established fixture implication gives the requested conclusion.", ["completion:bridge"]))
                next(row for row in article["source_map"] if row["unit_id"] == "completion:bridge")["blocks"] = ["assembly"]
            return reply

    agent = CompletedAgent(frozen, truncated=truncated_review)
    state = runtime.work(project.root, root.name, agent=agent)
    assert state["state"] == ("needs_review" if truncated_review else "ready"), state
    assert agent.calls.count("plan_publication") == 1 and "review_whole" in agent.calls
    assert (root / "paper.tex").exists()
    article = read_json(root / "article.json")
    ledger = read_json(root / "references.json")
    assert audit.deterministic(article, frozen, ledger)["draft_complete"]
    damaged = copy.deepcopy(article)
    damaged["sections"][3]["blocks"].pop()
    damaged["source_map"] = [r for r in damaged["source_map"] if r["unit_id"] != "completion:bridge"]
    result = audit.deterministic(damaged, frozen, ledger)
    assert not result["draft_complete"] and "completion:bridge" in result["missing_evidence"]
    before = list(agent.calls)
    runtime.work(project.root, root.name, resume=True, agent=agent)
    assert agent.calls == before
