import pytest

from hyprover.article import stage_state
from hyprover.article.models import digest, read_json, write_json


@pytest.mark.parametrize("interrupted_file", ["references.json", "checkpoint.json"])
def test_interrupted_reference_commit_uses_one_committed_generation(tmp_path, monkeypatch, interrupted_file):
    old_ledger = {"searches": []}
    new_ledger = {"searches": [], "focus": {"plan": {"queries": ["synthetic query"]}}}
    old_checkpoint = {"fingerprint": "test-project", "queries": [], "references_done": False}
    new_checkpoint = dict(old_checkpoint, queries=["synthetic query"])
    write_json(tmp_path / "references.json", old_ledger)
    write_json(tmp_path / "checkpoint.json", old_checkpoint)
    real_write = stage_state.atomic_write_text

    def interrupt(path, contents):
        if path.name == interrupted_file:
            raise OSError("simulated process loss")
        real_write(path, contents)

    with monkeypatch.context() as patch:
        patch.setattr(stage_state, "atomic_write_text", interrupt)
        with pytest.raises(OSError, match="simulated process loss"):
            stage_state.commit(tmp_path, digest(old_ledger), new_ledger, new_checkpoint)

    committed = interrupted_file == "checkpoint.json"
    assert read_json(tmp_path / "references.json") == (new_ledger if committed else old_ledger)
    assert stage_state.read_checkpoint(tmp_path) == (new_checkpoint if committed else old_checkpoint)
    stage_state.recover(tmp_path, "test-project")
    assert read_json(tmp_path / "checkpoint.json") == (new_checkpoint if committed else old_checkpoint)
    assert not (tmp_path / "reference-commit.json").exists()


def test_oversize_serialized_state_does_not_replace_previous_document(tmp_path):
    from hyprover.article.models import ArticleError

    path = tmp_path / "references.json"
    old = {"text": "old"}
    write_json(path, old)
    with pytest.raises(ArticleError, match="article_document_too_large_to_save"):
        write_json(path, {"text": "文" * 15}, max_bytes=40)
    assert read_json(path, 40) == old


def test_resume_runs_the_query_committed_immediately_before_interruption(project, monkeypatch):
    from hyprover.article import evidence, references, runtime
    from .conftest import publish
    from .test_audit import FakeAgent

    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "1")
    root = runtime.prepare(project.root, [publish(project)])
    agent_calls = []

    class SearchAgent(FakeAgent):
        def __call__(self, task, material, contract):
            reply = super().__call__(task, material, contract)
            if task == "plan_references":
                agent_calls.append(task)
                reply.value["queries"] = ["a synthetic search"]
            return reply

    calls = []

    def get(reader, url):
        calls.append(url)
        reader.requests += 1
        return b'{"message":{"items":[]}}', "application/json", url

    monkeypatch.setattr(references.SourceReader, "get", get)
    agent = SearchAgent(evidence.load(root))
    real_write = stage_state.atomic_write_text

    def interrupt(path, contents):
        if path.name == "checkpoint.json" and '"a synthetic search"' in contents:
            raise OSError("simulated process loss at cursor write")
        real_write(path, contents)

    with monkeypatch.context() as patch:
        patch.setattr(stage_state, "atomic_write_text", interrupt)
        assert runtime.work(project.root, root.name, agent=agent)["state"] == "failed"
    assert calls == [] and (root / "reference-commit.json").exists()
    outcome = runtime.work(project.root, root.name, agent=agent, resume=True)
    assert outcome["state"] == "ready", outcome
    assert len(calls) == len(agent_calls) == 1
    assert not (root / "reference-commit.json").exists()
