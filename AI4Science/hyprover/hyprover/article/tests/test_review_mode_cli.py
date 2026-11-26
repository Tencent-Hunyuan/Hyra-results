import json
import sys

import pytest

from hyprover.article import audit, collection, runtime
from hyprover.article.__main__ import main
from hyprover.article.models import FORMAT, read_json, write_json


def test_generate_uses_default_review_mode(tmp_path, monkeypatch, capsys):
    calls = []

    def prepare(project, targets, **options):
        assert "review_mode" not in options
        calls.append((project, targets))
        return project / "articles" / "fixture"

    monkeypatch.setattr(runtime, "prepare", prepare)
    monkeypatch.setattr(runtime, "work", lambda *_, **__: {"state": "ready", "review_mode": audit.FORMAT_COVERAGE})
    assert main(["generate", str(tmp_path), "--fact", "example"]) == 0
    assert calls == [(tmp_path, ["example"])]
    assert json.loads(capsys.readouterr().out)["review_mode"] == audit.FORMAT_COVERAGE


@pytest.mark.parametrize("state,code", [("ready", 0), ("needs_review", 2), ("busy", 2)])
def test_recheck_dispatches_without_generation_or_compile(tmp_path, monkeypatch, capsys, state, code):
    calls = []

    def forbidden(*_, **__):
        pytest.fail("recheck called generation or compilation")

    def recheck(project, run_id):
        calls.append((project, run_id))
        return {"state": state, "review_mode": audit.FORMAT_COVERAGE}

    for name in ("prepare", "work", "launch", "compile_article"):
        monkeypatch.setattr(runtime, name, forbidden)
    monkeypatch.setattr(runtime, "recheck", recheck)
    assert main(["recheck", str(tmp_path), "--run-id", "fixture"]) == code
    assert calls == [(tmp_path, "fixture")]
    assert json.loads(capsys.readouterr().out)["state"] == state


def test_collection_generation_persists_default_and_per_run_modes(tmp_path, monkeypatch):
    (tmp_path / "example").mkdir()
    (tmp_path / "export-manifest.json").write_text(
        json.dumps({"projects": [{"benchmark": "example", "directory": "example"}]}))
    calls = []

    def prepare(project, **options):
        assert "review_mode" not in options
        calls.append(project)
        return project / "articles" / "fixture"

    monkeypatch.setattr(runtime, "prepare", prepare)
    monkeypatch.setattr(runtime, "status", lambda *_: {"state": "ready", "review_mode": audit.FORMAT_COVERAGE})
    monkeypatch.setattr(sys, "argv", ["article-collection", str(tmp_path)])
    assert collection.main() == 0
    assert calls == [tmp_path / "example"]
    batch = read_json(tmp_path / "article-generation.json")
    assert batch["review_mode"] == batch["projects"]["example"]["review_mode"] == audit.FORMAT_COVERAGE


@pytest.mark.parametrize("entrypoint", ["single", "collection", "collection_resume"])
def test_cli_rejects_removed_review_mode(tmp_path, monkeypatch, capsys, entrypoint):
    args = [str(tmp_path), "--review-mode", audit.STRICT]
    with pytest.raises(SystemExit) as exc:
        if entrypoint == "single":
            main(["generate", *args])
        else:
            if entrypoint == "collection_resume":
                args.append("--resume")
            monkeypatch.setattr(sys, "argv", ["article-collection", *args])
            collection.main()
    assert exc.value.code == 2
    assert "unrecognized arguments: --review-mode strict" in capsys.readouterr().err


def test_collection_drops_pending_strict_review_after_local_recheck(tmp_path, monkeypatch):
    (tmp_path / "example").mkdir()
    write_json(tmp_path / "export-manifest.json", {"projects": [{"benchmark": "example", "directory": "example"}]})
    generation = tmp_path / "article-generation.json"
    write_json(generation, {"format": FORMAT, "concurrency": 1, "projects": {"example": {
        "directory": "example", "run_id": "fixture", "state": "reassessment_pending",
        "dispatch_intent": {"id": "earlier-intent", "action": "reassessment_pending", "revision": 3}}}})
    monkeypatch.setattr(runtime, "job_path", lambda project, run_id: project / "articles" / run_id)
    monkeypatch.setattr(runtime, "load_manifest", lambda _: {})
    monkeypatch.setattr(runtime, "status", lambda *_: {"state": "ready", "review_mode": audit.FORMAT_COVERAGE})
    monkeypatch.setattr(runtime, "launch", lambda *_, **__: pytest.fail("unexpected strict review"))
    assert collection.resume(tmp_path) == 0
    assert read_json(generation)["review_mode"] == audit.STRICT
    member = read_json(generation)["projects"]["example"]
    assert member["state"] == "ready" and member["review_mode"] == audit.FORMAT_COVERAGE
    assert "dispatch_intent" not in member
