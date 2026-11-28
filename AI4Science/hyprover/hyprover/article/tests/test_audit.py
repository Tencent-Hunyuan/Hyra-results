import copy
import json
from io import BytesIO
from uuid import uuid4

import pytest

from hyprover.article import audit, evidence, references, render
from hyprover.article.agent import Agent, Reply, response_text
from hyprover.article.models import ArticleConfig, ArticleError, Cancelled, digest
from hyprover.article.pipeline import Pipeline
from .test_pipeline import article_fixture, block, fixture
from .conftest import prepare_legacy_review, publish


def passed(expected):
    from hyprover.article import writing_policy
    checks = audit._CHECKS + (writing_policy.REVIEW_CHECKS if expected.get("writing_revision") else ())
    return {"coverage": dict(expected, materials_complete=True, truncated=False, uncovered=[]),
            "verdict": "pass", "findings": [], **{name: True for name in checks}}


class FakeAgent:
    def __init__(self, frozen, *, truncated=False, invalid_binding=False):
        self.frozen = frozen
        self.calls = []
        self.session_prefix = uuid4().hex
        self.truncated, self.invalid_binding = truncated, invalid_binding

    def __call__(self, task, material, contract):
        self.calls.append(task)
        if task == "plan_references":
            value = {"checked_units": [u["id"] for u in material["original_units"]], "materials_complete": True,
                "truncated": False, "candidates": [], "obligations": [], "queries": [],
                "scope_notes": {k: "Synthetic fixture has no external claims." for k in
                    ("problem_origin", "related_work", "borrowed_results")}}
        elif task == "plan_publication":
            ids = [u["id"] for u in material["original_units"]]
            value = {"checked_units": ids, "global_conventions": {"text": "Use the original fixture setting.", "evidence": ids[:1]},
                "notation": [], "preliminaries": [{"id": "fixture-definitions", "title": "Fixture objects",
                    "evidence": ids[:1], "concepts": ["The fixture object"], "used_by": ids, "citation_obligations": []}],
                "core": [{"id": "fixture-construction", "title": "Fixture construction", "evidence": ids,
                    "requires": ["fixture-definitions"], "goal": "Establish the intermediate argument."}],
                "main_proofs": [{"target_id": fid, "evidence": ids, "uses": ["fixture-construction"],
                    "strategy": "Apply the original argument to the target."} for fid in material["targets"]]}
        elif task == "plan_review_references":
            value = {"checked_obligations": list(material["obligations"]), "queries": [], "candidates": []}
        elif task == "revise_article":
            value = {"article": copy.deepcopy(material["article"])}
        elif task == "citation_inventory":
            value = {"checked_units": [u["id"] for u in material["units"]], "materials_complete": True,
                     "truncated": False, "candidates": [], "obligations": [], "queries": [], "exclusions": [],
                     "proof_summaries": [{"unit_id": u["id"], "summary": "Use the fixture argument."} for u in material["units"]]}
        elif task == "compose_paper":
            whole = article_fixture(self.frozen, references.seed(self.frozen))
            whole["reference_digest"] = material["reference_digest"]
            from hyprover.article import writing_policy
            whole["writing_revision"] = writing_policy.REVISION
            if material.get("publication_plan"):
                whole["sections"][0]["blocks"][:0] = [
                    block("context", "The fixture concerns the property under its fixed assumptions.", whole["title"]["evidence"]),
                    block("motivation", "The question is whether the supplied argument establishes the property under those assumptions.", whole["title"]["evidence"])]
                whole["sections"][3]["blocks"].insert(0, block("strategy", "We assemble the supplied argument to prove the main result.", whole["title"]["evidence"]))
                plan = material["publication_plan"]
                for role, index in (("preliminaries", 1), ("core", 2)):
                    for topic in plan[role]:
                        whole["sections"][index]["blocks"].extend([
                            block(topic["id"], topic["title"], topic["evidence"], "subsection"),
                            block(topic["id"] + "-body", "This topic supplies the fixture objects and argument used below.", topic["evidence"])])
                roadmap = block("roadmap", "See [[ref:prelim]] for definitions, [[ref:core]] for the construction, "
                                "and [[ref:proofs]] for the main proof.", whole["title"]["evidence"])
                roadmap["crossrefs"] = ["prelim", "core", "proofs"]
                whole["sections"][0]["blocks"].append(roadmap)
            all_blocks = list(render.walk_blocks(whole))
            whole["source_map"] = [{"unit_id": uid, "disposition": "developed",
                "blocks": [b["id"] for b in all_blocks if uid in b["evidence"]], "citations": [],
                "reason": "The argument is retained in the mapped passage."} for uid in evidence.units(self.frozen)]
            value = {"article": whole}
        elif task == "write_front":
            whole = article_fixture(self.frozen, references.seed(self.frozen))
            whole["sections"][0]["blocks"][0]["id"] = material["statement_ids"][self.frozen["targets"][0]]
            whole["sections"][3]["blocks"] = []
            value = {k: whole[k] for k in ("title", "abstract", "keywords", "msc", "sections")}
        elif task == "rewrite_proof":
            value = {"block": block(material["block_id"], material["unit"]["text"], [material["unit"]["id"]])}
        elif task == "detect_proof_gaps":
            value = {"binding": material["binding"], "materials_complete": True, "findings": []}
        elif task.startswith("review_"):
            value = passed(material["expected_coverage"])
            if self.truncated:
                value["coverage"]["truncated"] = True
            if self.invalid_binding:
                value["coverage"]["article_digest"] = "not-the-reviewed-article"
        else:
            pytest.fail("Unexpected model task: " + task)
        return Reply(f"fake-session-{self.session_prefix}-{len(self.calls)}", value)


def test_large_proof_is_written_and_independently_reviewed_in_parts(project):
    from hyprover.article import runtime
    from hyprover.article.models import read_json

    fid = publish(project, proof="Synthetic engineering fixture text. " * 6000)
    root = prepare_legacy_review(project.root, [fid])
    frozen = evidence.load(root)
    scanned, compared = set(), set()

    class TrackingAgent(FakeAgent):
        def __call__(self, task, material, contract):
            if task == "plan_references":
                scanned.update(u["id"] for u in material["original_units"])
            elif task.startswith("review_part-"):
                compared.update(u["id"] for u in material["original_units"])
            elif task == "review_whole":
                assert material["expected_coverage"]["material_scope"] == "indexed"
                assert "rendered" not in material and all("text" not in b for b in material["blocks"])
                assert material["earlier_receipts"]
            return super().__call__(task, material, contract)

    agent = TrackingAgent(frozen)
    result = runtime.work(project.root, root.name, agent=agent)
    assert result["state"] == "ready", result
    assert runtime.status(project.root, root.name)["state"] == "ready"
    assert scanned == compared == set(evidence.units(frozen))
    assert agent.calls.count("plan_references") > 1 and agent.calls.count("rewrite_proof") > 1
    assert read_json(root / "article.json")["writing_mode"] == "segmented"
    assert read_json(root / "audit.json")["receipts"]["whole"]["material_scope"] == "indexed"


def test_large_proof_with_unresolved_citation_still_produces_a_draft(project):
    from hyprover.article import runtime

    fid = publish(project, proof="Synthetic engineering fixture text. " * 7500)
    config = ArticleConfig(max_revisions=0)
    root = prepare_legacy_review(project.root, [fid], config=config)
    frozen = evidence.load(root)
    reviewed_units = set()

    class CitationAgent(FakeAgent):
        def __call__(self, task, material, contract):
            if task == "resolve_internal_obligation":
                assert len(json.dumps({"task": task, "material": material, "contract": contract})) <= config.max_input_chars
                reviewed_units.update(unit["id"] for unit in material["complete_evidence"])
                self.calls.append(task)
                return Reply(f"resolve-{len(self.calls)}", {"obligation_id": material["obligation"]["id"],
                    "supported_internally": False, "proof_units": [], "reason": "Unresolved in this fixture."})
            reply = super().__call__(task, material, contract)
            if task == "plan_references" and self.calls.count(task) == 1:
                reply.value["obligations"] = [{"id": "external-result", "description": "Source of an external result",
                    "source_units": reply.value["checked_units"][:1], "reference_ids": [],
                    "kind": "mathematical", "critical": True}]
            return reply

    agent = CitationAgent(frozen)
    status = runtime.work(project.root, root.name, agent=agent)
    assert status["state"] == "needs_review" and (root / "paper.tex").is_file(), status
    assert reviewed_units == set(evidence.units(frozen))
    assert agent.calls.count("resolve_internal_obligation") > 1


def test_review_parts_split_on_the_actual_request_size_and_bind_their_partition(project, tmp_path):
    fid = publish(project, proof="Engineering fixture evidence. " * 500)
    config = ArticleConfig(max_input_chars=36000)
    frozen = evidence.capture(project.root, [fid], config=config)
    root = tmp_path / "split-review"
    evidence.persist(root, frozen)
    agent = FakeAgent(frozen)
    pipe = Pipeline(root, frozen, config, agent=agent)
    pipe.ledger.update(discovery_complete=True, stop_reason="fixture completion")
    article = article_fixture(frozen, pipe.ledger)
    uid = fid + ":proof:part-0001"
    article["sections"][2]["blocks"].extend([
        block("scope-a", "A fixture passage. " * 500, [uid]),
        block("scope-b", "Another fixture passage. " * 360, [uid]),
    ])
    article["sections"][3]["blocks"][0]["parts"][0]["text"] = "A short fixture proof."
    outputs = render.render(article, pipe.ledger)
    report = pipe.review(article, outputs)
    middle = report["receipts"]["whole"]["partition"]
    assert len(middle) > len(audit.review_jobs(article)) - 1
    assert report["state"] == "ready"
    tampered = copy.deepcopy(report["receipts"])
    tampered["whole"]["partition"][0].pop()
    with pytest.raises(ArticleError, match="invalid_article_review_partition"):
        audit.finalize(article, frozen, pipe.ledger, outputs, tampered, pipe.writer_sessions)


def test_invalid_composition_gets_bounded_structural_repair_before_review(project, tmp_path):
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    root = tmp_path / "article"
    evidence.persist(root, frozen)
    class RepairingAgent(FakeAgent):
        def __call__(self, task, material, contract):
            reply = super().__call__(task, material, contract)
            if task == "compose_paper" and "format_correction" not in material:
                reply.value["article"]["sections"][0]["blocks"][0]["id"] = "wrong-result-id"
            return reply
    agent = RepairingAgent(frozen)
    report = Pipeline(root, frozen, ArticleConfig(), agent=agent).run()
    assert report["state"] == "ready", report
    assert agent.calls.count("compose_paper") == 2
    assert "write_front" not in agent.calls and "rewrite_proof" not in agent.calls
    assert agent.calls.index("review_part-1") > max(i for i, task in enumerate(agent.calls) if task == "compose_paper")
    before = len(agent.calls)
    assert Pipeline(root, frozen, ArticleConfig(), agent=agent).run()["state"] == "ready"
    assert len(agent.calls) == before


def test_deterministic_requires_proofs_not_just_mentions(project):
    frozen, ledger, article = fixture(project)
    assert audit.deterministic(article, frozen, ledger)["passed"]
    article["sections"][3]["blocks"][0]["parts"][0]["evidence"] = [frozen["targets"][0] + ":statement"]
    report = audit.deterministic(article, frozen, ledger)
    assert not report["passed"]
    assert frozen["targets"][0] + ":proof:part-0001" in report["missing_evidence"]


def test_unknown_evidence_duplicate_ids_and_new_results_are_rejected(project):
    frozen, ledger, original = fixture(project)
    article = copy.deepcopy(original)
    article["abstract"]["evidence"] = ["invented"]
    with pytest.raises(ArticleError, match="unknown_article_evidence"):
        audit.deterministic(article, frozen, ledger)
    article = copy.deepcopy(original)
    article["sections"][0]["blocks"][0]["id"] = "abstract"
    with pytest.raises(ArticleError, match="duplicate"):
        audit.deterministic(article, frozen, ledger)
    article = copy.deepcopy(original)
    article["sections"][2]["blocks"][0]["kind"] = "lemma"
    with pytest.raises(ArticleError, match="new_or_ambiguous"):
        audit.deterministic(article, frozen, ledger)


def test_receipts_require_every_block_hash_and_independent_session(project):
    frozen, ledger, article = fixture(project)
    outputs = render.render(article, ledger)
    receipts = {}
    for i, job in enumerate(audit.review_jobs(article)):
        expected = audit.expected_coverage(article, frozen, ledger, outputs, job["id"], job["blocks"])
        receipts[job["id"]] = {"session_id": f"reviewer-{i}", "report": passed(expected)}
    assert audit.finalize(article, frozen, ledger, outputs, receipts, ["writer"])["state"] == "ready"
    whole = receipts["whole"]
    whole["report"]["coverage"]["checked_blocks"].pop()
    assert audit.finalize(article, frozen, ledger, outputs, receipts, ["writer"])["state"] == "needs_review"
    whole["session_id"] = "writer"
    assert "audit_session_not_independent:whole" in audit.finalize(article, frozen, ledger, outputs, receipts, ["writer"])["failures"]
    with pytest.raises(ArticleError):
        audit.receipt_outcome({"verdict": "pass"}, {})


def test_end_to_end_and_resume_reuse_completed_model_steps(project, tmp_path):
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    root = tmp_path / "article"
    evidence.persist(root, frozen)
    agent = FakeAgent(frozen)
    report = Pipeline(root, frozen, ArticleConfig(), agent=agent).run()
    assert report["state"] == "ready", report
    assert report["formal_certificate"] is False
    assert (root / "paper.tex").is_file() and (root / "audit.json").is_file()
    first_calls = list(agent.calls)
    report = Pipeline(root, frozen, ArticleConfig(), agent=agent).run()
    assert report["state"] == "ready", report
    assert agent.calls == first_calls


@pytest.mark.parametrize("option", ["truncated", "invalid_binding"])
def test_incomplete_or_unbound_reviews_never_pass(project, tmp_path, option):
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    root = tmp_path / "article"
    evidence.persist(root, frozen)
    report = Pipeline(root, frozen, ArticleConfig(), agent=FakeAgent(frozen, **{option: True})).run()
    assert report["state"] == "needs_review"


def test_call_budget_and_cancel_are_explicit(project, tmp_path):
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    root = tmp_path / "article"
    evidence.persist(root, frozen)
    agent = FakeAgent(frozen)
    report = Pipeline(root, frozen, ArticleConfig(max_calls=1), agent=agent).run()
    assert "call_budget" in report["error"] and len(agent.calls) == 1
    with pytest.raises(Cancelled):
        Pipeline(root, frozen, ArticleConfig(max_calls=1), agent=agent, check_cancel=lambda: True).run()


@pytest.mark.parametrize("streamed", [False, True])
def test_tool_free_agent_uses_existing_api_without_cli_or_history(monkeypatch, streamed):
    from hyprover.article import agent as module
    monkeypatch.setenv("CODEX_BACKEND", "api")
    monkeypatch.setenv("CODEX_API_BASE_URL", "http://127.0.0.1:9876/v1")
    monkeypatch.setenv("HYPROVER_CODEX_API_KEY", "test-only-not-a-secret")
    requests = []
    completed = {
        "status": "completed", "id": "resp_fixture", "model": "fixture-model-2026",
        "reasoning": {"effort": "high"}, "max_output_tokens": 2048,
        "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4}},
        "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": '{"answer":"fixture"}'}]}],
    }
    if streamed:
        body = b"data: " + json.dumps({"type": "response.completed", "response": completed}).encode() + b"\n\n"
    else:
        body = json.dumps(completed).encode()

    class Response(BytesIO):
        status = 200

        def getheader(self, name, default=""):
            return "text/event-stream" if streamed else "application/json"

    class Connection:
        def __init__(self, host, port, timeout):
            assert (host, port) == ("127.0.0.1", 9876)

        def request(self, method, path, body, headers):
            requests.append((method, path, json.loads(body), headers))

        def getresponse(self):
            return Response(body)

        def close(self):
            pass

    monkeypatch.setattr(module.http.client, "HTTPConnection", Connection)
    config = ArticleConfig(model="fixture-model", effort="high", max_output_tokens=2048)
    profile = module.backend_profile(config)
    first = Agent(config, profile=profile)("task", {"data": "fixture"}, "Return JSON")
    second = Agent(config, profile=profile)("task", {"data": "fixture"}, "Return JSON")
    assert first.session_id != second.session_id
    assert first.value == {"answer": "fixture"}
    method, path, request, headers = requests[0]
    assert (method, path) == ("POST", "/v1/responses")
    assert headers == {"Authorization": "Bearer test-only-not-a-secret", "Content-Type": "application/json",
                       "Accept": "text/event-stream"}
    assert request["reasoning"] == {"effort": "high"} and request["max_output_tokens"] == 2048
    assert request["tools"] == [] and request["tool_choice"] == "none" and request["parallel_tool_calls"] is False
    assert request["store"] is False and request["stream"] is True and "previous_response_id" not in request
    assert first.backend_response == {
        "route": profile, "requested_model": "fixture-model", "requested_reasoning": {"effort": "high"},
        "model": "fixture-model-2026", "response_id": "resp_fixture", "reasoning": {"effort": "high"},
        "max_output_tokens": 2048, "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4}},
    }
    monkeypatch.setenv("CODEX_BACKEND", "chatgpt")
    with pytest.raises(ArticleError, match="tool_free"):
        Agent(config)("task", {}, "JSON")


@pytest.mark.parametrize("field,value", [("api_backend", "retired-provider"), ("reasoning_mode", "pro")])
def test_unsupported_article_settings_cannot_fall_back_to_responses(field, value):
    error = "unsupported_article_" + field
    with pytest.raises(ArticleError, match=error):
        ArticleConfig.from_dict({field: value})
    with pytest.raises(ArticleError, match=error):
        ArticleConfig.resolve({}, {"HYPROVER_ARTICLE_" + field.upper(): value})
    assert ArticleConfig.from_dict({"api_backend": "codex", "reasoning_mode": None}) == ArticleConfig()


def test_article_model_endpoint_requires_https_off_loopback():
    from hyprover.article.agent import backend_profile

    with pytest.raises(ArticleError, match="article_api_requires_https_except_loopback"):
        backend_profile(ArticleConfig(), {"CODEX_BACKEND": "api", "CODEX_API_BASE_URL": "http://fixture.invalid/v1"})


def test_backend_tool_calls_and_incomplete_response_are_not_executed():
    with pytest.raises(ArticleError, match="tool_or_unknown"):
        response_text({"status": "completed", "output": [{"type": "function_call", "name": "shell"}]})
    with pytest.raises(ArticleError, match="incomplete"):
        response_text({"status": "incomplete", "output": []})
