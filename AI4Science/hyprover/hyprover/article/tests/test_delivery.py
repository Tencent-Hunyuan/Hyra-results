import json

import pytest

from hyprover.article import delivery, evidence, runtime
from hyprover.article.models import ArticleError, write_json
from .conftest import publish
from .test_audit import FakeAgent


def test_collection_delivery_distinguishes_failures_and_never_copies_sessions(tmp_path):
    (tmp_path / "export-manifest.json").write_text(json.dumps({"projects": [{"benchmark": "fixture-beta", "directory": "fixture-beta/"}]}))
    (tmp_path / "article-generation.json").write_text(json.dumps({"complete": True, "projects": {
        "fixture-beta": {"state": "failed", "run_id": "fixture", "error": "source unavailable"}}}))
    result, files = delivery.collect(tmp_path)
    assert result["all_ready"] is False
    assert result["projects"]["fixture-beta"]["manuscript_available"] is False
    assert files == ["article-delivery.json"]


def test_public_delivery_rejects_actual_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-private-value-123456")
    (tmp_path / "paper.tex").write_text("fixture-private-value-123456")
    with pytest.raises(ArticleError, match="credential_detected"):
        delivery.check_public_files(tmp_path, ["paper.tex"])


def test_git_delivery_does_not_commit_existing_staged_changes(tmp_path, monkeypatch):
    import subprocess
    calls = []
    def command(argv, **kwargs):
        args = argv[3:]
        calls.append(args)
        if args == ["branch", "--show-current"]:
            return subprocess.CompletedProcess(argv, 0, "master\n", "")
        if args == ["diff", "--cached", "--quiet"]:
            return subprocess.CompletedProcess(argv, 1, "", "")
        pytest.fail("existing changes must stop delivery before staging or pushing")
    monkeypatch.setattr(delivery.subprocess, "run", command)
    with pytest.raises(ArticleError, match="existing_staged_changes"):
        delivery.commit_and_push(tmp_path, ["paper.tex"])
    assert len(calls) == 2


def test_git_delivery_does_not_push_existing_local_commits(tmp_path, monkeypatch):
    import subprocess
    (tmp_path / "paper.tex").write_text("Public fixture manuscript")
    expected = {
        ("branch", "--show-current"): "master\n",
        ("diff", "--cached", "--quiet"): "",
        ("rev-parse", "HEAD"): f"{'a' * 40}\n",
        ("ls-remote", "--exit-code", "origin", "refs/heads/master"): f"{'b' * 40}\trefs/heads/master\n",
    }

    def command(argv, **kwargs):
        args = tuple(argv[3:])
        assert args in expected, "branch divergence must block staging and pushing"
        return subprocess.CompletedProcess(argv, 0, expected[args], "")

    monkeypatch.setattr(delivery.subprocess, "run", command)
    with pytest.raises(ArticleError, match="master_at_remote_tip"):
        delivery.commit_and_push(tmp_path, ["paper.tex"])


@pytest.mark.parametrize("changed_at", ["worktree", "index"])
def test_git_delivery_rejects_unapproved_bytes_before_commit(tmp_path, monkeypatch, changed_at):
    import hashlib
    import subprocess

    approved = b"Approved manuscript"
    (tmp_path / "paper.tex").write_bytes(b"Changed manuscript" if changed_at == "worktree" else approved)
    tip = "a" * 40
    commands = {
        ("branch", "--show-current"): "master\n",
        ("diff", "--cached", "--quiet"): "",
        ("rev-parse", "HEAD"): tip + "\n",
        ("ls-remote", "--exit-code", "origin", "refs/heads/master"): tip + "\trefs/heads/master\n",
    }
    if changed_at == "index":
        commands.update({("add", "--", "paper.tex"): "", ("diff", "--cached", "--name-only", "-z"): "paper.tex\0",
                         ("ls-files", "--stage", "-z", "--", "paper.tex"): "100644 fixture 0\tpaper.tex\0",
                         ("cat-file", "blob", "fixture"): b"Changed in staging",
                         ("reset", "-q", tip, "--", "paper.tex"): ""})

    def command(argv, **kwargs):
        args = tuple(argv[3:])
        assert args in commands, "changed bytes must stop delivery before commit and push"
        return subprocess.CompletedProcess(argv, 0, commands[args], "")

    monkeypatch.setattr(delivery.subprocess, "run", command)
    with pytest.raises(ArticleError, match="changed_before_stage" if changed_at == "worktree" else "staged_bytes_changed"):
        delivery.commit_and_push(tmp_path, ["paper.tex"], file_hashes={"paper.tex": hashlib.sha256(approved).hexdigest()})


def test_delivery_exports_only_manuscript_and_public_audits(project):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    result = runtime.work(project.root, root.name, agent=FakeAgent(evidence.load(root)))
    assert result["state"] == "ready"
    (root / "private-session.json").write_text('{"private":"must not be exported"}')
    provenance = delivery.export(project.root, root.name)
    paper = project.root / "paper"
    assert provenance["state"] == "ready" and provenance["model_sessions_exported"] is False
    assert provenance["author"] == "hyprover" and "affiliation" not in provenance
    assert sorted(p.name for p in paper.iterdir() if not p.name.startswith(".")) == [
        "article.json", "audit.json", "citation-audit.json", "current.json", "paper.tex", "provenance.json",
        "references.bib", "references.tex", "source-map.json"]
    snapshot, hashes = delivery.current_artifacts(paper)
    assert snapshot.name == provenance["artifact_version"] and sorted(hashes) == provenance["files"]
    from hyprover.article import writing_policy
    assert provenance["writing_revision"] == writing_policy.REVISION
    assert "source_map" in json.loads((paper / "source-map.json").read_text())
    assert "sources" not in json.loads((paper / "citation-audit.json").read_text())
    assert (paper / "paper.tex").read_text() == (root / "paper.tex").read_text()


def test_interrupted_export_keeps_previous_verified_pointer_and_can_retry(project, monkeypatch):
    def make_article(statement):
        fid = publish(project, statement=statement)
        root = runtime.prepare(project.root, [fid])
        state = runtime.work(project.root, root.name, agent=FakeAgent(evidence.load(root)))
        assert state["state"] == "ready"
        return root

    original = make_article("The original fixture.")
    first = delivery.export(project.root, original.name)
    replacement = make_article("The replacement fixture.")
    paper = project.root / "paper"
    saved = (paper / "current.json").read_bytes()
    write = delivery.write_json

    def abort_pointer(path, value):
        if path == paper / "current.json":
            raise OSError("simulated interruption before activation")
        write(path, value)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(delivery, "write_json", abort_pointer)
        with pytest.raises(OSError, match="simulated interruption"):
            delivery.export(project.root, replacement.name)
    assert (paper / "current.json").read_bytes() == saved
    assert delivery.current_artifacts(paper)[0].name == first["artifact_version"]
    second = delivery.export(project.root, replacement.name)
    snapshot, _ = delivery.current_artifacts(paper)
    assert snapshot.name == second["artifact_version"] != first["artifact_version"]
    assert (paper / "paper.tex").read_bytes() == (snapshot / "paper.tex").read_bytes()
    (snapshot / "paper.tex").write_text("Changed artifact")
    with pytest.raises(ArticleError, match="snapshot_changed"):
        delivery.current_artifacts(paper)


def test_batch_delivery_lists_only_artifacts_from_current_export(project, tmp_path):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    state = runtime.work(project.root, root.name, agent=FakeAgent(evidence.load(root)))
    assert state["state"] == "ready"
    paper = project.root / "paper"
    paper.mkdir()
    unrelated = paper / "private-notes.txt"
    unrelated.write_text("untracked work")
    (paper / "compile.log").symlink_to(unrelated)
    (paper / "compilation.json").write_text("{}")
    write_json(tmp_path / "export-manifest.json", {"projects": [{"benchmark": "P", "directory": "project"}]})
    write_json(tmp_path / "article-generation.json", {"complete": True, "projects": {
        "P": {"state": "ready", "run_id": root.name}}})

    result, files = delivery.collect(tmp_path)

    assert result["all_ready"]
    names = {name.removeprefix("project/paper/") for name in files if name != "article-delivery.json"}
    flat = {"paper.tex", "references.tex", "references.bib", "article.json", "audit.json",
            "citation-audit.json", "source-map.json", "provenance.json"}
    version = result["projects"]["P"]["artifact_version"]
    assert names == flat | {"current.json"} | {f".article-versions/{version}/{name}" for name in flat | {"artifact-manifest.json"}}
    assert set(result["file_hashes"]) == set(files) - {"article-delivery.json"}
    assert unrelated.read_text() == "untracked work"


def test_delivery_refuses_pending_or_tampered_work(project):
    fid = publish(project)
    root = runtime.prepare(project.root, [fid])
    with pytest.raises(ArticleError, match="not_available"):
        delivery.export(project.root, root.name)
    assert not (project.root / "paper").exists()
    runtime.work(project.root, root.name, agent=FakeAgent(evidence.load(root)))
    (root / "paper.tex").write_text("not the actual manuscript")
    with pytest.raises(ArticleError, match="not_available"):
        delivery.export(project.root, root.name)
