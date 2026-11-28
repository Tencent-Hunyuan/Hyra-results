import json

import pytest

from hyprover.article import evidence
from hyprover.article.models import ArticleConfig, ArticleError, EvidenceChanged, digest
from hyprover.verify import completion_evidence as ce
from .conftest import complete, publish


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("detached_inventory", [False, True])
def test_relocated_archive_keeps_historical_receipts_unchanged(project, tmp_path, detached_inventory):
    import hashlib
    import shutil
    fid = publish(project)
    verification = complete(project, [fid])
    target = tmp_path / "export"
    shutil.copytree(project.root, target)
    shutil.copytree(verification, target / "verify-runs" / verification.name)
    if detached_inventory:
        result_path = target / "main/result.json"
        result = ce.read_json(result_path)
        result["completion_report"] = str(project.root / "verify-runs" / verification.name / "completion-report.json")
        ce.write_json(result_path, result)
    files = [{"path": str(p.relative_to(target)), "bytes": p.stat().st_size,
              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in target.rglob("*") if p.is_file()]
    manifest = {"format": "hyprover-original-project-export-v1", "source_directory": str(project.root),
        "layout": {"original_verification_root": str(verification.parent)},
        "completion": {"result": "main/result.json", "report": "verify-runs/" + verification.name + "/completion-report.json",
            "accepted": True, "outcome": "passed", "target_fact_ids": [fid]}, "files": files}
    if detached_inventory:
        manifest.pop("files")
        manifest.update(project=project.root.name, layout={"file_inventory_hashes_and_exclusions": "export-inventory.json"},
                        counts={"included_original_files": len(files)})
        (target / "export-inventory.json").write_text(json.dumps({"format": "hyprover-export-file-inventory-v1",
            "source_project": project.root.name, "included_source_files": files}))
    (target / "export-manifest.json").write_text(json.dumps(manifest))
    before = tree(target)
    frozen = evidence.capture(target)
    assert frozen["completion"]["archive"]["historical_receipts_modified"] is False
    assert tree(target) == before
    evidence.assert_current(frozen)
    with pytest.raises(ArticleError, match="root_mismatch"):
        evidence.capture(target, completion_root=str(verification))
    (target / "PROBLEM.md").write_text("Changed problem")
    with pytest.raises(ArticleError, match="archive_file"):
        evidence.capture(target)


def test_repository_export_validates_public_evidence_without_reading_private_files(project, tmp_path):
    import hashlib
    import shutil

    fid = publish(project)
    verification = complete(project, [fid])
    archive = tmp_path / "archive"
    shutil.copytree(project.root, archive)
    shutil.copytree(verification, archive / "verification-runs" / verification.name)
    report = f"verification-runs/{verification.name}/completion-report.json"
    exported = {"export_type": "complete_research_project_snapshot",
        "source_directory": str(project.root), "project": project.root.name,
        "layout": {"original_verification_root": str(verification.parent)},
        "completion": {"accepted": True, "formal_certificate": False,
                       "result": "main/result.json", "report": report, "target_fact_id": fid}}
    export_path = archive / "export-manifest.json"
    ce.write_json(export_path, exported)
    inventory = [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                 for name, data in tree(archive).items()]
    inventory.append({"path": "workers/private/local_memory.md", "bytes": 0, "sha256": "0" * 64})
    ce.write_json(archive / "migration-manifest.json", {"format": "hyprover-repository-migration-v1",
        "benchmark": "fixture-001", "source_project": project.root.name, "method": {"target_prefix": "fixture-001/"},
        "integrity": {"tracked_files": len(inventory)}, "files": inventory,
        "completion": {"accepted": True, "formal_certificate": False, "target_fact_id": fid,
                       "source_export_manifest": "fixture-001/export-manifest.json", "result": "fixture-001/main/result.json",
                       "report": f"fixture-001/{report}", "frozen_evidence_digest": ce.load_manifest(verification)["digest"]}})
    before = tree(archive)
    frozen = evidence.capture(archive)
    assert frozen["targets"] == [fid] and frozen["mode"] == "completed"
    evidence.assert_current(frozen, project=archive)
    assert tree(archive) == before
    export_path.write_text(export_path.read_text() + " ")
    with pytest.raises(ArticleError, match="archive_file_size_changed: export-manifest.json"):
        evidence.capture(archive)


def test_read_only_dependency_closure_and_every_character(project):
    parent = publish(project, "Fixture premise.", definitions={"x": "A fixture symbol."})
    proof = ("A fixture proof paragraph.\n\n" * 50)
    child = publish(project, predecessors=[parent], proof=proof)
    before = tree(project.root)
    frozen = evidence.capture(project.root, [child], config=ArticleConfig(chunk_chars=256))
    assert [f["fact_id"] for f in frozen["facts"]] == [parent, child]
    assert frozen["mode"] == "selected_facts" and not frozen["summary"]
    assert frozen["formal_certificate"] is False
    parts = [u["text"] for u in evidence.units(frozen).values()
             if u["fact_id"] == child and u["kind"] == "proof"]
    assert "".join(parts) == proof
    assert tree(project.root) == before
    evidence.assert_current(frozen)


def test_only_current_published_facts_are_usable(project):
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    project.revoke(fid, "Fixture withdrawal")
    with pytest.raises(EvidenceChanged):
        evidence.assert_current(frozen)
    with pytest.raises(ValueError, match="unknown_or_revoked"):
        evidence.capture(project.root, [fid])
    with pytest.raises(ValueError):
        evidence.capture(project.root, ["correct"])


def test_bibliography_changes_invalidate_article(project):
    fid = publish(project, refs=[{"key": "source", "title": "Fixture source"}])
    frozen = evidence.capture(project.root, [fid])
    project.set_external_refs(fid, [{"key": "source", "title": "Corrected source"}])
    with pytest.raises(EvidenceChanged):
        evidence.assert_current(frozen)


def test_corrupt_attachment_is_rejected(project):
    fid = publish(project)
    path = project.root / project.get(fid).proof_path
    path.write_text(path.read_text().replace("Fixture proof.", "Different proof."))
    with pytest.raises(ValueError, match="content_address"):
        evidence.capture(project.root, [fid])


def test_auto_requires_complete_bound_receipts(project):
    fid = publish(project)
    root = complete(project, [fid])
    frozen = evidence.capture(project.root)
    assert frozen["mode"] == "completed"
    assert "completion:bridge" in evidence.units(frozen)
    evidence.assert_current(frozen)
    progress = ce.read_json(root / "completion-progress.json")
    progress["receipts"].pop("bridge")
    ce.write_json(root / "completion-progress.json", progress)
    with pytest.raises(ValueError, match="checks_missing"):
        evidence.capture(project.root)
    with pytest.raises(EvidenceChanged):
        evidence.assert_current(frozen)


def test_fake_completion_verdict_cannot_enable_article(project):
    publish(project)
    (project.root / "main").mkdir()
    (project.root / "main/result.json").write_text(json.dumps({"verdict": "correct", "formal_certificate": False}))
    with pytest.raises(ArticleError, match="committed_completion"):
        evidence.capture(project.root)


def test_frozen_evidence_is_immutable_and_paths_are_guarded(project, tmp_path):
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    root = tmp_path / "article"
    evidence.persist(root, frozen)
    assert evidence.load(root) == frozen
    changed = dict(frozen, problem="different")
    changed["digest"] = digest({k: v for k, v in changed.items() if k != "digest"})
    with pytest.raises(ArticleError, match="immutable"):
        evidence.persist(root, changed)
    with pytest.raises(ArticleError, match="unsafe"):
        evidence.capture(project.root, result_path="../result.json")
    (tmp_path / "linked").symlink_to(project.root, target_is_directory=True)
    with pytest.raises(ArticleError, match="symlink"):
        evidence.capture(tmp_path / "linked", [fid])


def test_permission_and_problem_changes_invalidate(project, monkeypatch):
    fid = publish(project)
    frozen = evidence.capture(project.root, [fid])
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "1")
    with pytest.raises(EvidenceChanged):
        evidence.assert_current(frozen)
    monkeypatch.setenv("HYPROVER_LITERATURE_SEARCH", "0")
    (project.root / "PROBLEM.md").write_text("A changed problem")
    with pytest.raises(EvidenceChanged):
        evidence.assert_current(frozen)


def test_configuration_is_strict_and_user_identity_is_not_model_controlled():
    from hyprover.article.models import AUTHOR
    assert AUTHOR == "hyprover"
    assert ArticleConfig.resolve({"article": {"auto_generate": False}}, {}).auto_generate is False
    assert ArticleConfig.resolve({"article": {"auto_generate": False}}, {"HYPROVER_ARTICLE_AUTO": "1"}).auto_generate
    for value in ({"max_calls": True}, {"max_queries": -1}, {"author": "someone"}):
        with pytest.raises(ArticleError):
            ArticleConfig.from_dict(value)


@pytest.mark.parametrize("field,default,maximum", [
    ("timeout_seconds", 1800, 10800),
    ("total_seconds", 21600, 259200),
    ("source_timeout", 60, 360),
])
def test_article_time_defaults_and_bounds(field, default, maximum):
    assert getattr(ArticleConfig(), field) == default
    assert getattr(ArticleConfig.resolve({}, {}), field) == default
    for value in (1, default // 3, maximum):
        config = ArticleConfig.resolve({"article": {field: value}}, {})
        assert getattr(config, field) == value
        assert ArticleConfig.from_dict(config.to_dict()) == config
    for value in (0, -1, maximum + 1, True, 1.5, str(default)):
        with pytest.raises(ArticleError, match="invalid_article_config: " + field):
            ArticleConfig.from_dict({field: value})
