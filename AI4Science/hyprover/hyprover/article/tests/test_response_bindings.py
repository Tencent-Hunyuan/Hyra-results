import copy

import pytest

from hyprover.article import audit, field_recovery, response_validation
from hyprover.article.agent import Reply
from hyprover.article.models import ArticleError, EvidenceChanged, ResponseContractError, digest, read_json, write_json
from .test_audit import FakeAgent
from .test_recovery import make_pipeline


def composition(pipe):
    return FakeAgent(pipe.evidence)("compose_paper", {"reference_digest": digest(pipe.ledger)}, "").value


def validation(pipe):
    return {"evidence": pipe.evidence, "ledger": pipe.ledger}


@pytest.mark.parametrize("mode", ["missing", "wrong"])
def test_machine_fields_bound_without_requesting_model_correction(project, tmp_path, mode):
    pipe = make_pipeline(project, tmp_path)
    original = composition(pipe)
    for field in ("format", "evidence_digest", "reference_digest", "writing_revision"):
        if mode == "missing":
            original["article"].pop(field)
        else:
            original["article"][field] = "incorrect-machine-field"
    frozen = copy.deepcopy(original)
    calls = []

    def agent(task, material, contract):
        assert task == "compose_paper"
        calls.append(task)
        return Reply("writer", original)

    pipe.agent = agent
    request = {"original_units": list(pipe.source_units.values())}
    result = pipe.ask("compose_paper", request, "Fixture", validation_material=validation(pipe))
    response_validation.validate("compose_paper", result.value, validation(pipe))
    assert calls == ["compose_paper"] and original == frozen
    assert result.value["article"]["evidence_digest"] == pipe.evidence["digest"]
    for name in ("title", "abstract", "sections", "source_map"):
        assert result.value["article"][name] == original["article"][name]
    receipt = read_json(next((pipe.root / "response-bindings").glob("*.json")))
    assert len(receipt["machine_owned_fields"]) == 4
    assert receipt["mathematical_text_rewritten"] is False
    cached = next((pipe.root / "calls").glob("*.json"))
    before = cached.read_bytes()
    again = pipe.ask("compose_paper", request, "Fixture", validation_material=validation(pipe))
    assert again.value == result.value and len(calls) == 1
    assert cached.read_bytes() == before and read_json(cached)["value"] == original


def test_short_digest_and_null_character_are_repaired_independently(project, tmp_path):
    pipe = make_pipeline(project, tmp_path)
    original = composition(pipe)
    original["article"]["evidence_digest"] = pipe.evidence["digest"][:61]
    original["article"]["sections"][3]["blocks"][0]["parts"][0]["text"] += " Use \\(U(\x00eta)\\)."
    saved = copy.deepcopy(original)
    calls = []

    def agent(task, material, contract):
        calls.append(task)
        if task == "compose_paper":
            return Reply("writer", original)
        assert task == "repair_article_fields" and len(material["fields"]) == 1
        field = material["fields"][0]
        assert field["path"][-1] == "text"
        return Reply("patch-writer", {"patches": [{"path": field["path"], "before_digest": field["before_digest"],
                      "value": field["value"].replace("\x00", "\\")} ]})

    pipe.agent = agent
    result = pipe.ask("compose_paper", {}, "Fixture", validation_material=validation(pipe))
    response_validation.validate("compose_paper", result.value, validation(pipe))
    assert calls == ["compose_paper", "repair_article_fields"]
    assert original == saved and len(original["article"]["evidence_digest"]) == 61
    assert r"U(\eta)" in result.value["article"]["sections"][3]["blocks"][0]["parts"][0]["text"]
    assert read_json(next((pipe.root / "response-errors").glob("*.json")))["resolved"]


@pytest.mark.parametrize("recover", [False, True])
def test_new_global_error_uses_whole_response_scope_and_reports_latest_cause(project, tmp_path, recover):
    pipe = make_pipeline(project, tmp_path)
    original = composition(pipe)
    original["article"]["sections"][3]["blocks"][0]["parts"][0]["text"] += "\x00"
    original["article"]["source_map"][0]["blocks"] = ["missing-block"]
    calls = []

    def agent(task, material, contract):
        calls.append(task)
        if task == "compose_paper":
            if "format_correction" not in material:
                return Reply("writer", original)
            correction = material["format_correction"]
            assert correction["error"] == "publication_source_map_not_bound_to_text"
            assert "\x00" not in correction["previous_response"]["article"]["sections"][3]["blocks"][0]["parts"][0]["text"]
            return Reply("second-writer", composition(pipe) if recover else correction["previous_response"])
        assert task == "repair_article_fields"
        return Reply("patch-writer", {"patches": [dict(path=f["path"], before_digest=f["before_digest"],
                      value=f["value"].replace("\x00", "")) for f in material["fields"]]})

    pipe.agent = agent
    if recover:
        reply = pipe.ask("compose_paper", {}, "Fixture", validation_material=validation(pipe))
        response_validation.validate("compose_paper", reply.value, validation(pipe))
    else:
        with pytest.raises(ResponseContractError, match="publication_source_map_not_bound_to_text") as failure:
            pipe.ask("compose_paper", {}, "Fixture", validation_material=validation(pipe))
        errors = [row["error"] for row in failure.value.failures]
        assert "unescaped_control_character_in_tex" in errors[0]
        assert errors[-1] == "publication_source_map_not_bound_to_text"
        assert "unescaped_control" not in str(failure.value)
    assert calls == ["compose_paper", "repair_article_fields", "compose_paper"]


def test_nested_patch_failure_is_not_swallowed_and_has_finite_budget(project, tmp_path):
    pipe = make_pipeline(project, tmp_path)
    original = composition(pipe)
    original["article"]["sections"][3]["blocks"][0]["parts"][0]["text"] += "\x00"
    calls = []

    def agent(task, material, contract):
        calls.append(task)
        return Reply("session-" + str(len(calls)), original if task == "compose_paper" else {"patches": []})

    pipe.agent = agent
    with pytest.raises(ResponseContractError, match="incomplete_article_field_repairs") as failure:
        pipe.ask("compose_paper", {}, "Fixture", validation_material=validation(pipe))
    assert len(calls) <= 6
    assert any(row["stage"] == "field_repair" and row["causes"] for row in failure.value.failures)
    assert any(not read_json(p)["resolved"] for p in (pipe.root / "response-errors").glob("*.json"))


def test_tampered_cache_cannot_be_rebound_to_another_request(project, tmp_path):
    pipe = make_pipeline(project, tmp_path)
    original = composition(pipe)
    contract, request = "Fixture", {}
    key = digest([pipe.fingerprint, "compose_paper", request, contract, False])
    path = pipe.root / "calls" / (key + ".json")
    path.parent.mkdir(exist_ok=True)
    write_json(path, {"request_digest": "different-request", "session_id": "foreign-writer", "task": "compose_paper",
                      "reviewer": False, "value": original, "value_digest": digest(original)})
    pipe.agent = lambda *args: pytest.fail("Invalid cache must not call the model")
    with pytest.raises(ArticleError, match="invalid_article_call_checkpoint"):
        pipe.ask("compose_paper", request, contract, validation_material=validation(pipe))
    assert not (pipe.root / "response-bindings").exists()


def test_saved_article_and_review_receipts_still_reject_wrong_binding(project, tmp_path):
    pipe = make_pipeline(project, tmp_path)
    original = composition(pipe)
    original["article"]["evidence_digest"] = "wrong"
    with pytest.raises(ArticleError, match="article_input_binding_mismatch"):
        audit.deterministic(original["article"], pipe.evidence, pipe.ledger)
    material = validation(pipe)
    material["evidence"] = copy.deepcopy(pipe.evidence)
    material["evidence"]["digest"] = "wrong"
    with pytest.raises(EvidenceChanged, match="validation_context_changed"):
        pipe.bind_article_reply("compose_paper", Reply("writer", original), {}, material)


def test_successful_patch_replans_remaining_invalid_fields(project, tmp_path, monkeypatch):
    pipe = make_pipeline(project, tmp_path)
    original = composition(pipe)
    original["article"]["sections"][0]["blocks"][0]["text"] += "\x00"
    original["article"]["sections"][3]["blocks"][0]["parts"][0]["text"] += "\x00"
    real_plan = field_recovery.plan
    monkeypatch.setattr(field_recovery, "plan", lambda value, frozen: real_plan(value, frozen)[:1])
    targets = []

    def agent(task, material, contract):
        if task == "compose_paper":
            return Reply("writer", original)
        f = material["fields"][0]
        targets.append(f["path"])
        return Reply("patch-" + str(len(targets)), {"patches": [dict(path=f["path"], before_digest=f["before_digest"],
                     value=f["value"].replace("\x00", ""))]})

    pipe.agent = agent
    result = pipe.ask("compose_paper", {}, "Fixture", validation_material=validation(pipe))
    response_validation.validate("compose_paper", result.value, validation(pipe))
    assert len(targets) == 2 and targets[0] != targets[1]
