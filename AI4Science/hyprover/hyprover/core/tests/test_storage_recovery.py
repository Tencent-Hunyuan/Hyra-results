"""Real filesystem fault injection; all data is disposable engineering fixtures."""
from dataclasses import replace
import multiprocessing
import os
from pathlib import Path

import pytest

from hyprover.core import Fact, KnowledgeStore, knowledge_markdown as md
from hyprover.core import _util
from hyprover.core.schema import clean_external_refs


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def store(tmp_path):
    return KnowledgeStore.initialize(tmp_path / "project", "P")


def fact(**changes):
    return Fact(**{**dict(fact_id="", problem_id="P", author="fixture", predecessors=[],
                         statement="Synthetic statement", proof="Synthetic proof"), **changes})


@pytest.mark.parametrize("extra", [
    {2: "a", 10: "b"}, {"safe": [{1: "nested"}]}, {"safe": float("nan")},
    {"safe": float("inf")}, {"safe": {1, 2}}, {"safe": "\ud800"},
])
def test_invalid_reference_metadata_never_breaks_committed_notebook(store, extra):
    fid = store.publish(fact(), {"verdict": "correct"})
    before = files(store.root)
    with pytest.raises(ValueError):
        store.set_external_refs(fid, [{"key": "R", "extra": extra}])
    assert files(store.root) == before
    assert KnowledgeStore(store.root).get(fid).statement == "Synthetic statement"
    with pytest.raises(ValueError):
        store.publish(fact(statement="Another statement", external_refs=[{"extra": extra}]),
                      {"verdict": "correct"})
    assert files(store.root) == before


def test_reference_copy_keeps_valid_types_and_original_keys(store):
    fid = store.publish(fact(), {"verdict": "correct"})
    payload = [{"key": "R", "extra": {"2": "a", "10": [None, True, 1, 1.5]}}]
    returned = store.set_external_refs(fid, payload)
    payload[0]["extra"]["10"].append("caller mutation")
    assert "caller mutation" not in returned[0]["extra"]["10"]
    text = store.path.read_text()
    assert md.serialize(md.parse(text)) == text
    assert KnowledgeStore(store.root).external_refs(fid) == returned


def test_cyclic_and_invalid_container_references_are_rejected():
    circular = {}
    circular["loop"] = circular
    with pytest.raises(ValueError, match="too_deep"):
        clean_external_refs([circular])
    for value in (1, True, {"key": "R"}, "R"):
        with pytest.raises(ValueError, match="external_refs_must_be_list"):
            clean_external_refs(value)
    assert clean_external_refs(None) == []
    assert clean_external_refs([None, "ignored"]) == []


def test_all_fact_metadata_is_validated_before_attachments(store):
    before = files(store.root)
    with pytest.raises(ValueError, match="keys_must_be_strings"):
        store.publish(fact(), {"verdict": "correct", "provenance": {2: "x", 10: "y"}})
    assert files(store.root) == before
    fid = store.publish(fact(), {"verdict": "correct"})
    snap = store.snapshot()
    changed = replace(snap.facts[fid], revocation={"extra": {1: "invalid key"}})
    with pytest.raises(ValueError, match="keys_must_be_strings"):
        md.validate(replace(snap, facts={fid: changed}))


@pytest.mark.parametrize("field", ["title", "body", "details"])
def test_unencodable_entry_text_is_rejected_before_any_write(store, field):
    args = dict(kind="report", title="Title", body="Body", details="Detail", author="a", role="worker")
    args[field] = "\ud800"
    before = files(store.root)
    with pytest.raises(ValueError, match="unicode"):
        store.record(**args)
    assert files(store.root) == before


def test_failed_update_can_change_details_without_overwriting_or_skipping_revision(store, monkeypatch):
    old = store.record("report", "Title", "Version one", "a", "worker", details="First detail")
    before_notebook = store.path.read_bytes()
    original_replace = _util.os.replace

    def fail_notebook(src, dst):
        if Path(dst) == store.path:
            raise OSError("injected notebook rename failure")
        return original_replace(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(_util.os, "replace", fail_notebook)
        with pytest.raises(OSError, match="rename failure"):
            store.record("report", old.title, "Failed attempt", "a", "worker", entry_id=old.entry_id,
                         expected_revision=1, details="Failed attempt A")
    orphan = store.root / f"knowledge/notes/{old.entry_id}-2.md"
    orphan_bytes = orphan.read_bytes()
    assert store.path.read_bytes() == before_notebook
    restarted = KnowledgeStore(store.root)
    new = restarted.record("report", old.title, "Corrected attempt", "a", "worker", entry_id=old.entry_id,
                           expected_revision=1, details="Corrected attempt B")
    assert new.revision == 2 and new.display_id == old.display_id
    assert new.details_path != str(orphan.relative_to(store.root))
    assert new.details_path.startswith(f"knowledge/notes/{old.entry_id}-2-")
    assert orphan.read_bytes() == orphan_bytes
    assert restarted.entry_detail(new.entry_id)["details"] == "Corrected attempt B"
    assert restarted.snapshot().entries[new.entry_id] == new
    assert md.attachment_body((store.root / old.details_path).read_text()) == "First detail"


@pytest.mark.parametrize("directory", ["notes", "project"])
def test_same_request_retry_completes_failed_directory_sync(store, monkeypatch, directory):
    target = store.root / "knowledge/notes" if directory == "notes" else store.root
    original_sync = _util.fsync_directory
    calls = []

    def fail_directory(path):
        calls.append(Path(path))
        if Path(path) == target:
            raise OSError("injected directory sync failure")
        return original_sync(path)

    args = dict(kind="report", title="Retry", body="Same payload", author="a", role="worker",
                details="Same detail", request_id="durable-retry")
    with monkeypatch.context() as patch:
        patch.setattr(_util, "fsync_directory", fail_directory)
        with pytest.raises(OSError, match="sync failure"):
            store.record(**args)
        count = calls.count(target)
        with pytest.raises(OSError, match="sync failure"):
            KnowledgeStore(store.root).record(**args)
        assert calls.count(target) > count, "retry must not skip the failed fsync"
    recovered = KnowledgeStore(store.root).record(**args)
    assert len(store.snapshot().entries) == 1 and store.snapshot().revision == 1
    before = files(store.root)
    assert store.record(**args) == recovered
    assert files(store.root) == before
    assert store.entry_detail(recovered.entry_id)["details"] == "Same detail"
    assert len(list((store.root / "knowledge/notes").glob("*.md"))) == 1


def test_fact_retry_cannot_hide_failed_directory_sync(store, monkeypatch):
    candidate = fact()
    original_sync = _util.fsync_directory
    attempts = []

    def fail_root(path):
        if Path(path) == store.root:
            attempts.append(path)
            raise OSError("injected sync failure")
        return original_sync(path)

    with monkeypatch.context() as patch:
        patch.setattr(_util, "fsync_directory", fail_root)
        with pytest.raises(OSError, match="sync failure"):
            store.publish(candidate, {"verdict": "correct"})
        with pytest.raises(OSError, match="sync failure"):
            KnowledgeStore(store.root).publish(candidate, {"verdict": "correct"})
    assert len(attempts) == 2
    fid = store.publish(candidate, {"verdict": "correct"})
    assert len(store.list()) == 1 and store.proof(fid) == candidate.proof


def test_content_addressed_fallback_never_overwrites_collision(store, monkeypatch):
    old = store.record("report", "T", "B", "a", "worker")
    base = store.root / f"knowledge/notes/{old.entry_id}-2.md"
    base.write_text(md.attachment(old.entry_id, "Orphan A", "Research note — NOT A FACT"))
    relative = store._entry_attachment_path(old.entry_id, 2, "Attempt B")
    (store.root / relative).write_text("Conflicting file at the fallback name")
    before = files(store.root)
    with pytest.raises(ValueError, match="immutable_attachment_conflict"):
        store.record("report", "T", "B", "a", "worker", entry_id=old.entry_id,
                     expected_revision=1, details="Attempt B")
    assert files(store.root) == before


def test_missing_or_symlinked_retry_attachment_is_not_acknowledged(store, tmp_path):
    args = dict(kind="report", title="T", body="B", author="a", role="worker",
                details="D", request_id="same")
    entry = store.record(**args)
    path = store.root / entry.details_path
    path.unlink()
    with pytest.raises(FileNotFoundError):
        store.record(**args)
    private = tmp_path / "private"
    private.write_text("not shared")
    path.symlink_to(private)
    with pytest.raises(ValueError, match="symlink"):
        store.record(**args)


def _exit_after_rename(root, phase):
    root = Path(root)
    original = _util.fsync_directory
    target = root if phase == "notebook" else root / "knowledge/notes"

    def terminate_at_sync(path):
        if Path(path) == target:
            os._exit(77)
        return original(path)

    _util.fsync_directory = terminate_at_sync
    KnowledgeStore(root).record("report", "Crash fixture", "Unchanged payload", "a", "worker",
                                details="Crash fixture detail", request_id="process-crash")


@pytest.mark.parametrize("phase", ["attachment", "notebook"])
def test_process_exit_after_rename_releases_lock_and_retry_recovers(store, phase):
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_exit_after_rename, args=(str(store.root), phase))
    process.start()
    try:
        process.join(20)
        assert not process.is_alive(), "fault-injection child timed out"
        assert process.exitcode == 77
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
    fresh = KnowledgeStore(store.root)
    assert len(fresh.snapshot().entries) == (1 if phase == "notebook" else 0)
    recovered = fresh.record("report", "Crash fixture", "Unchanged payload", "a", "worker",
                             details="Crash fixture detail", request_id="process-crash")
    assert len(fresh.snapshot().entries) == 1 and fresh.snapshot().revision == 1
    assert fresh.entry_detail(recovered.entry_id)["details"] == "Crash fixture detail"
    assert len(list((store.root / "knowledge/notes").glob("*.md"))) == 1
