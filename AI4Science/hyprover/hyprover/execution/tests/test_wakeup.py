"""Semantic wakeup tests; all verdicts and facts are offline engineering fixtures."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from hyprover.core import Fact, KnowledgeStore
from hyprover.core import knowledge_markdown as md
from hyprover.execution import wakeup


@pytest.fixture
def inputs(tmp_path):
    store = KnowledgeStore.initialize(tmp_path / "project", "P")
    task = store.root / "TASK.md"
    task.write_text("Investigate the assigned subproblem.\n")
    return store, task, wakeup.InputWatcher(store.root, task)


def record(store, kind, body="content", **kwargs):
    role = "verifier" if kind == "feedback" else "main"
    return store.record(kind, "title", body, role, role, **kwargs)


def publication(store, fid, **changes):
    payload = dict(fact_id=fid, accepted=True, published=True, verdict="correct",
                   verification_report={"summary": "checked", "critical_errors": [], "gaps": []},
                   repair_hints="", run_id="run-1")
    payload.update(changes)
    return record(store, "feedback", details=json.dumps(payload),
                  attributes={"run_id": payload["run_id"], "published": payload["published"]})


def test_task_line_endings_and_same_content_rewrite(inputs):
    _, task, watcher = inputs
    before = watcher.read()
    task.write_bytes(task.read_bytes().replace(b"\n", b"\r\n"))
    assert watcher.read() == before
    task.write_text("A genuinely new task.\n")
    assert watcher.read().changed_from(before) == ("task",)


def test_ordinary_entries_logs_and_self_reports_do_not_wake(inputs):
    store, task, watcher = inputs
    before = watcher.read()
    for author in ("main", "high", "peer"):
        store.record("report", "progress", "unchanged result", author, "main" if author == "main" else "worker")
    for kind in ("conjecture", "lead", "route"):
        store.record(kind, "proposal", "idea", "high", "worker")
    store.record("experience", "method", "lesson", "high", "worker", attributes={"subtype": "method"})
    task.with_name("round_1.log").write_text("a new log")
    task.with_name(".status.json").write_text('{"updated_at":999}')
    assert watcher.read() == before


@pytest.mark.parametrize("kind,status", [("overview", "current"), ("route", "planned")])
def test_main_content_and_attributes_wake_but_revision_only_does_not(inputs, kind, status):
    store, _, watcher = inputs
    before = watcher.read()
    entry = record(store, kind, status=status, details="details")
    current = watcher.read()
    category = "routes" if kind == "route" else "overview"
    assert current.changed_from(before) == (category,)
    entry = record(store, kind, status=status, details="details", entry_id=entry.entry_id, expected_revision=1)
    assert watcher.read() == current
    record(store, kind, status=status, entry_id=entry.entry_id, expected_revision=entry.revision,
           attributes={"next_step": "new direction"})
    assert watcher.read().changed_from(current) == (category,)


@pytest.mark.parametrize("author", ["main", "main_agent"])
def test_main_proposed_route_creation_and_edits_wake(inputs, author):
    store, _, watcher = inputs
    before = watcher.read()
    entry = store.record("route", "Main route", "Initial direction", author, "main")
    assert entry.status == "proposed" and entry.author == author
    created = watcher.read()
    assert created.changed_from(before) == ("routes",)
    entry = store.record("route", "Main route", "Initial direction", author, "main",
                         entry_id=entry.entry_id, expected_revision=entry.revision)
    assert entry.revision == 2 and watcher.read() == created
    revised = store.record("route", "Main route", "A different direction", author, "main",
                           entry_id=entry.entry_id, expected_revision=entry.revision)
    assert revised.author == author and revised.status == "proposed"
    assert watcher.read().changed_from(created) == ("routes",)


def test_worker_proposed_route_edits_do_not_wake(inputs):
    store, _, watcher = inputs
    before = watcher.read()
    entry = store.record("route", "Worker proposal", "Initial idea", "high", "worker")
    assert watcher.read() == before
    store.record("route", "Worker proposal", "Changed idea", "high", "worker",
                 entry_id=entry.entry_id, expected_revision=entry.revision)
    assert watcher.read() == before


def test_main_adopted_worker_route_and_status_changes_wake(inputs):
    store, _, watcher = inputs
    entry = store.record("route", "title", "content", "high", "worker")
    before = watcher.read()
    entry = record(store, "route", status="active", entry_id=entry.entry_id, expected_revision=1)
    assert entry.author == "high"
    active = watcher.read()
    assert active.changed_from(before) == ("routes",)
    record(store, "route", status="parked", entry_id=entry.entry_id, expected_revision=entry.revision)
    assert watcher.read().changed_from(active) == ("routes",)


def test_own_new_fact_revocation_and_supersession_wake(inputs):
    store, _, watcher = inputs
    before = watcher.read()
    fid = store.publish(Fact(fact_id="", problem_id="P", author="high", predecessors=[],
                             statement="fixture statement", proof="fixture proof", glossary_introduces={}),
                        {"verdict": "correct", "run_id": "fixture"})
    published = watcher.read()
    assert published.changed_from(before) == ("facts",)
    store.refresh_display({fid: "A different display title"})
    assert watcher.read() == published
    store.revoke(fid, "fixture withdrawal")
    assert watcher.read().changed_from(published) == ("facts",)


def test_feedback_deduplicates_run_metadata_but_preserves_diagnostics(inputs):
    store, _, watcher = inputs
    before = watcher.read()
    publication(store, "0123456789abcdef")
    accepted = watcher.read()
    assert accepted.changed_from(before) == ("feedback",)
    publication(store, "0123456789abcdef", run_id="run-2", updated_at="tomorrow", retry_after=30,
                raw_verification={"run_id": "anything", "opaque_trace": "different"})
    assert watcher.read() == accepted
    publication(store, "0123456789abcdef", run_id="run-3", repair_hints="Check boundary conditions")
    assert watcher.read().changed_from(accepted) == ("feedback",)


def test_legacy_symbol_warnings_do_not_wake_and_remain_readable(inputs):
    store, _, watcher = inputs
    publication(store, "0123456789abcdef")
    before = watcher.read()
    entry = publication(store, "0123456789abcdef", run_id="run-2",
                        undefined_symbols=["S_M"], glossary_warning="Historical diagnostic")
    assert watcher.read() == before
    saved = json.loads(store.entry_detail(entry.entry_id)["details"])
    assert saved["undefined_symbols"] == ["S_M"] and saved["glossary_warning"]


def test_completion_feedback_ignores_embedded_run_paths(inputs):
    store, _, watcher = inputs
    def add(run):
        payload = {"evidence_digest": "evidence", "outcome": "incomplete", "accepted": False,
                   "full_revalidation_run": run, "report_path": f"/runs/{run}/report.json",
                   "retry_after": 123, "error": f"Read /runs/{run}/report.json"}
        return record(store, "feedback", f"Read /runs/{run}/report.json",
                      attributes={"operation": "completion_revalidation", "run_id": run},
                      details=json.dumps(payload))
    add("first-run")
    before = watcher.read()
    add("second-run")
    assert watcher.read() == before


def test_unidentified_failed_candidates_are_not_conflated(inputs):
    store, _, watcher = inputs
    publication(store, None, published=False, accepted=False, verdict="incorrect")
    before = watcher.read()
    publication(store, None, published=False, accepted=False, verdict="incorrect", run_id="run-2")
    assert watcher.read().changed_from(before) == ("feedback",)


def test_attachment_contents_not_revision_path_trigger(inputs):
    store, _, watcher = inputs
    entry = record(store, "overview", details="unchanged attachment")
    before = watcher.read()
    entry = record(store, "overview", details="unchanged attachment", entry_id=entry.entry_id, expected_revision=1)
    assert entry.details_path.endswith("-2.md")
    assert watcher.read() == before
    record(store, "overview", details="changed attachment", entry_id=entry.entry_id, expected_revision=2)
    assert watcher.read().changed_from(before) == ("overview",)


def test_snapshot_and_unchanged_entry_cache_avoid_attachment_reads(inputs, monkeypatch):
    store, _, watcher = inputs
    record(store, "overview", details="cached")
    before = watcher.read()
    def unexpected(*args):
        pytest.fail("unchanged attachments must not be read")
    monkeypatch.setattr(watcher, "_attachment", unexpected)
    assert watcher.read() == before
    store.record("report", "progress", "new report", "high", "worker")
    assert watcher.read() == before


def test_attachment_cache_is_bounded(inputs, monkeypatch):
    store, _, watcher = inputs
    monkeypatch.setattr(wakeup, "_ATTACHMENT_CACHE_SIZE", 2)
    for i in range(4):
        record(store, "feedback", str(i), details="attachment " + str(i))
    watcher.read()
    assert len(watcher._attachments) == 2


@pytest.mark.parametrize("failure", ["missing", "corrupt", "symlink"])
def test_bad_attachment_keeps_baseline_and_recovers(inputs, failure):
    store, _, watcher = inputs
    before = watcher.read()
    entry = record(store, "overview", details="new research")
    path = store.root / entry.details_path
    original = path.read_text()
    path.unlink()
    if failure == "corrupt":
        path.write_text("not an attachment")
    elif failure == "symlink":
        path.symlink_to(store.path)
    with pytest.raises(wakeup.WakeupReadError):
        watcher.read()
    assert watcher._snapshot.revision == 0
    path.unlink(missing_ok=True)
    path.write_text(original)
    assert watcher.read().changed_from(before) == ("overview",)


def test_task_or_notebook_missing_or_corrupt_does_not_advance(inputs):
    store, task, watcher = inputs
    before = watcher.read()
    original = task.read_text()
    task.unlink()
    with pytest.raises(wakeup.WakeupReadError):
        watcher.read()
    task.write_text(original)
    contents = store.path.read_text()
    store.path.write_text("corrupt")
    with pytest.raises(wakeup.WakeupReadError):
        watcher.read()
    store.path.write_text(contents)
    assert watcher.read() == before


def test_racing_task_update_is_not_cached_as_a_valid_read(inputs, monkeypatch):
    store, task, watcher = inputs
    before = watcher.read()
    original = watcher._knowledge_parts
    def race(snapshot):
        task.write_text("new task")
        return original(snapshot)
    with monkeypatch.context() as patch:
        patch.setattr(watcher, "_knowledge_parts", race)
        with pytest.raises(wakeup.WakeupReadError, match="changed_during_read"):
            watcher.read()
    assert watcher.read().changed_from(before) == ("task",)


def test_extra_raw_verification_diagnostics_wake_for_same_source(inputs):
    store, _, watcher = inputs
    source = store.record("conjecture", "source", "candidate statement", "high", "worker")

    def add(error, run):
        payload = {"fact_id": None, "published": False, "verdict": "correct",
                   "error": "inconsistent_or_incomplete_verification_report",
                   "raw_verification": {"errors": [error], "run_id": run}}
        record(store, "feedback", links=[source.entry_id], details=json.dumps(payload),
               attributes={"published": False, "run_id": run})

    add("Boundary gap", "run-one")
    before = watcher.read()
    add("Boundary gap", "run-two")
    assert watcher.read() == before
    add("Different hypothesis gap", "run-three")
    assert watcher.read().changed_from(before) == ("feedback",)


def test_swapping_identical_routes_preserves_identity(inputs):
    store, _, watcher = inputs
    first = record(store, "route", status="active")
    second = record(store, "route", status="parked")
    before = watcher.read()
    record(store, "route", status="parked", entry_id=first.entry_id, expected_revision=1)
    record(store, "route", status="active", entry_id=second.entry_id, expected_revision=1)
    assert watcher.read().changed_from(before) == ("routes",)


def test_failed_batch_reuses_successful_entries_beyond_attachment_cache(inputs, monkeypatch):
    store, _, watcher = inputs
    before = watcher.read()
    monkeypatch.setattr(wakeup, "_ATTACHMENT_CACHE_SIZE", 2)
    for index in range(5):
        record(store, "feedback", f"outcome {index}", details=f"details {index}")
    bad = record(store, "feedback", "last feedback", details="last detail")
    path = store.root / bad.details_path
    original = path.read_text()
    path.unlink()
    with pytest.raises(wakeup.WakeupReadError):
        watcher.read()
    assert watcher._snapshot.revision == 0
    read_paths = []
    read = Path.read_text

    def track(target, *args, **kwargs):
        read_paths.append(target)
        return read(target, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", track)
    with pytest.raises(wakeup.WakeupReadError):
        watcher.read()
    assert not any(p.parent.name == "notes" for p in read_paths)
    path.write_text(original)
    assert watcher.read().changed_from(before) == ("feedback",)


def test_same_semantic_content_survives_metadata_only_update(inputs):
    store, _, watcher = inputs
    entry = record(store, "overview")
    before = watcher.read()
    snap = store.snapshot()
    updated = replace(entry, revision=entry.revision + 1, updated_at="different", display_id=entry.display_id)
    store.path.write_text(md.serialize(replace(snap, revision=snap.revision + 1, entries={entry.entry_id: updated})))
    assert watcher.read() == before
