"""Knowledge persistence regressions; accepted verdicts are engineering fixtures."""
from dataclasses import FrozenInstanceError, replace
import json
import multiprocessing
from pathlib import Path

import pytest

from hyprover.core import Fact, KnowledgeStore, compute_fact_id
from hyprover.core import knowledge_markdown as md
from hyprover.core import _util
from hyprover.core.schema import KNOWLEDGE_FORMAT, KNOWLEDGE_MARKERS

ACCEPTED = {"verdict": "correct", "errors": [], "gaps": [],
            "run_id": "engineering-fixture-only"}


def candidate(statement="fixture statement", **changes):
    values = dict(fact_id="", problem_id="P", author="worker", predecessors=[],
                  statement=statement, proof="fixture proof", glossary_introduces={})
    values.update(changes)
    return Fact(**values)


def digest(fact):
    return compute_fact_id(problem_id=fact.problem_id, predecessors=fact.predecessors,
                           glossary_introduces=fact.glossary_introduces,
                           statement=fact.statement, proof=fact.proof)


def tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def store(tmp_path):
    return KnowledgeStore.initialize(tmp_path / "project", "P")


def _process_writer(root, index, ready):
    assert ready.wait(15), "writer start timed out"
    KnowledgeStore(Path(root)).record("experience", f"writer {index}",
                                      f"process record {index}", "worker", "worker",
                                      attributes={"subtype": "method"})


def test_markdown_preserves_nested_body_and_stable_glossary(store):
    body = "  ## nested\n### inner\n```python\nx = 1\n```\n\n$$x^2+y^2$$\n  "
    fid = store.publish(candidate(body, glossary_introduces={"Zeta": "z", "Alpha": "a"}), ACCEPTED)
    entry = store.record("experience", "nested note", body, "worker", "worker",
                         attributes={"subtype": "method"})
    text = store.path.read_text(encoding="utf-8")
    restored = md.parse(text)
    assert restored.facts[fid].statement == body
    assert restored.entries[entry.entry_id].body == body
    assert len(restored.facts) == len(restored.entries) == 1
    assert md.serialize(restored) == text
    assert text.index("- `Alpha`: a") < text.index("- `Zeta`: z")
    assert f"**Proof:** [完整证明 / Full proof — F001](knowledge/proofs/{fid}.md)" in text
    assert "### F001 · nested [verified]" in text


def test_crlf_statements_and_proofs_roundtrip_without_mutation(store):
    statement = "Every element has an identity.\r\nIncluding the boundary case."
    proof = "Apply the identity law.\r\nThen check the boundary case."
    fact = candidate(statement, proof=proof, intuition="First line.\r\nSecond line.")
    fid = store.publish(fact, ACCEPTED)
    entry = store.record("experience", "Line endings", statement, "worker", "worker",
                         details=proof, attributes={"subtype": "boundary"})
    restored = KnowledgeStore(store.root)
    assert restored.get(fid).statement == statement
    assert restored.proof(fid) == proof
    assert restored.read_fact(fid).intuition == fact.intuition
    assert restored.snapshot().entries[entry.entry_id].body == statement
    assert restored.publish(fact, ACCEPTED) == fid


def test_previous_v1_keeps_its_format_and_existing_proof_on_record(store):
    fact = candidate(proof="Apply the identity law.\r\nThen check the boundary case.")
    fid = store.publish(fact, ACCEPTED)
    report = store.record("report", "Progress", "Initial progress.", "worker", "worker")
    original = store.snapshot()
    proof_path = store.root / original.facts[fid].proof_path
    legacy_format, = KNOWLEDGE_MARKERS.keys() - {KNOWLEDGE_FORMAT}
    legacy_marker = KNOWLEDGE_MARKERS[legacy_format].encode()
    assert b'"format": "hyprover-knowledge-v1"' in store.path.read_bytes()
    assert b"<!-- hyprover:body -->" in proof_path.read_bytes()

    store.path.write_bytes(store.path.read_bytes()
                           .replace(b"<!-- hyprover:", legacy_marker)
                           .replace(KNOWLEDGE_FORMAT.encode(), legacy_format.encode()))
    proof_path.write_bytes(proof_path.read_bytes().replace(b"<!-- hyprover:", legacy_marker))
    metadata = store.root / "project.json"
    metadata.write_text(json.dumps({"knowledge_format": legacy_format}), encoding="utf-8")

    before = tree_bytes(store.root)
    restored = KnowledgeStore(store.root)
    previous = restored.snapshot()
    assert previous.storage_format == legacy_format
    assert previous.revision == original.revision
    assert previous.facts[fid] == original.facts[fid]
    assert previous.entries[report.entry_id] == report
    assert restored.proof(fid) == fact.proof
    assert tree_bytes(store.root) == before

    details = "A detailed progress update."
    updated = restored.record("report", report.title, "Updated progress.", "worker", "worker",
                              entry_id=report.entry_id, expected_revision=report.revision,
                              links=[fid], details=details)
    after = KnowledgeStore(store.root).snapshot()
    assert after.storage_format == legacy_format
    assert after.revision == previous.revision + 1
    assert after.facts[fid] == previous.facts[fid]
    assert updated.entry_id == report.entry_id and updated.revision == report.revision + 1
    assert after.entries[report.entry_id] == updated
    assert restored.entry_detail(updated.entry_id)["details"] == details
    assert legacy_marker + b"body -->" in (store.root / updated.details_path).read_bytes()
    assert legacy_marker + b"document " in store.path.read_bytes()
    assert b"<!-- hyprover:" not in store.path.read_bytes()
    assert metadata.read_bytes() == before["project.json"]
    assert proof_path.read_bytes() == before[original.facts[fid].proof_path]


def test_previous_v1_treats_current_markers_as_literal_content(tmp_path):
    legacy_format, = KNOWLEDGE_MARKERS.keys() - {KNOWLEDGE_FORMAT}
    root = tmp_path / "existing"
    root.mkdir()
    (root / "project.json").write_text(json.dumps({"knowledge_format": legacy_format}), encoding="utf-8")
    store = KnowledgeStore.initialize(root, "P")
    literal = "Example:\n<!-- hyprover:body -->\nText\n<!-- hyprover:end-body -->"
    fid = store.publish(candidate(proof=literal), ACCEPTED)
    report = store.record("report", "Progress", literal, "worker", "worker", details=literal)

    notebook = store.path.read_bytes()
    reopened = KnowledgeStore(root)
    assert reopened.proof(fid) == literal
    assert reopened.entry_detail(report.entry_id)["details"] == literal
    assert reopened.snapshot().entries[report.entry_id].body == literal
    assert md.serialize(reopened.snapshot()).encode() == notebook


@pytest.mark.parametrize("mismatch", ["metadata", "record marker"])
def test_mixed_v1_formats_are_rejected_without_writes(store, mismatch):
    legacy_format, = KNOWLEDGE_MARKERS.keys() - {KNOWLEDGE_FORMAT}
    store.publish(candidate(), ACCEPTED)
    if mismatch == "metadata":
        (store.root / "project.json").write_text(
            json.dumps({"knowledge_format": legacy_format}), encoding="utf-8")
    else:
        legacy_marker = KNOWLEDGE_MARKERS[legacy_format].encode()
        store.path.write_bytes(store.path.read_bytes().replace(
            b"<!-- hyprover:fact ", legacy_marker + b"fact ", 1))
    before = tree_bytes(store.root)
    with pytest.raises(ValueError):
        store.snapshot()
    assert tree_bytes(store.root) == before


def test_default_reads_never_read_or_index_attachments(store, monkeypatch):
    fid = store.publish(candidate("visiblefixture", proof="proofsecretxyz",
                                  intuition="intuitionsecretxyz"), ACCEPTED)
    store.record("experience", "note", "visible note", "worker", "worker",
                 details="detailsecretxyz", attributes={"subtype": "method"})
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.parent not in (store.root / "knowledge/proofs", store.root / "knowledge/notes")
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        assert store.snapshot().facts[fid].statement == "visiblefixture"
        assert store.list() == [fid] and store.get(fid).fact_id == fid
        assert store.search("visiblefixture")[0]["id"] == fid
        for token in ("proofsecretxyz", "intuitionsecretxyz", "detailsecretxyz"):
            assert store.search(token) == []
    assert store.proof(fid) == "proofsecretxyz"


@pytest.mark.parametrize("kind,role", [("overview", "worker"), ("feedback", "worker"),
                                      ("experience", "verifier"), ("feedback", "main")])
def test_role_cannot_record_other_kinds(store, kind, role):
    with pytest.raises(ValueError, match="role_cannot_record_kind"):
        store.record(kind, "title", "body", role, role)
    assert store.snapshot().revision == 0


def test_owner_and_main_route_controls(store):
    entry = store.record("route", "route", "body", "alice", "worker")
    with pytest.raises(ValueError, match="entry_ownership_conflict"):
        store.record("route", "route", "changed", "bob", "worker",
                     entry_id=entry.entry_id, expected_revision=1)
    with pytest.raises(ValueError, match="only_main_controls_routes"):
        store.record("route", "route", "changed", "alice", "worker",
                     entry_id=entry.entry_id, expected_revision=1, status="active")
    active = store.record("route", "route", "changed", "main", "main",
                          entry_id=entry.entry_id, expected_revision=1, status="active")
    assert active.author == "alice" and active.status == "active"
    with pytest.raises(ValueError, match="only_main_controls_routes"):
        store.record("route", "route", "changed", "alice", "worker",
                     entry_id=entry.entry_id, expected_revision=2, status="proposed")
    assert store.record("overview", "overview", "body", "main", "main").status == "current"
    assert store.record("feedback", "feedback", "body", "v", "verifier").status == "recorded"


@pytest.mark.parametrize("status", ["proved", "refuted"])
def test_conjecture_cannot_promote_itself(store, status):
    with pytest.raises(ValueError, match="illegal_state_promotion"):
        store.record("conjecture", "guess", "body", "worker", "worker", status=status)
    assert not store.snapshot().entries


def test_source_revision_conflict_is_atomic(store):
    source = store.record("conjecture", "guess", "fixture statement", "worker", "worker")
    store.record("conjecture", "guess revised", source.body, "worker", "worker",
                 entry_id=source.entry_id, expected_revision=1)
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="source_revision_or_state_conflict"):
        store.publish(candidate(), ACCEPTED, source.entry_id, source.revision)
    assert tree_bytes(store.root) == before
    assert store.snapshot().entries[source.entry_id].status == "open"


def test_different_source_statement_does_not_close_conjecture(store):
    source = store.record("conjecture", "guess", "original claim", "worker", "worker")
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="source_statement_mismatch"):
        store.publish(candidate("different claim"), ACCEPTED, source.entry_id, source.revision)
    assert tree_bytes(store.root) == before
    assert store.snapshot().entries[source.entry_id].status == "open"
    assert store.list() == []


def test_matching_source_closes_once_and_is_immutable(store):
    source = store.record("conjecture", "guess", "fixture statement", "worker", "worker")
    fact = candidate()
    fid = store.publish(fact, ACCEPTED, source.entry_id, source.revision)
    resolved = store.snapshot().entries[source.entry_id]
    assert resolved.status == "proved" and resolved.revision == source.revision + 1
    assert resolved.attributes["resolution_fact_id"] == fid and fid in resolved.links
    before = tree_bytes(store.root)
    assert store.publish(fact, ACCEPTED, source.entry_id, source.revision) == fid
    assert tree_bytes(store.root) == before
    with pytest.raises(ValueError, match="resolved_conjecture_is_immutable"):
        store.record("conjecture", "guess", "changed", "main", "main",
                     entry_id=source.entry_id, expected_revision=resolved.revision)


def test_concurrent_process_writes_are_not_lost(store):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    processes = [context.Process(target=_process_writer, args=(str(store.root), i, ready))
                 for i in range(4)]
    try:
        for process in processes:
            process.start()
        ready.set()
        for process in processes:
            process.join(20)
            assert not process.is_alive(), "writer did not finish"
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    snapshot = store.snapshot()
    assert snapshot.revision == 4
    assert {entry.display_id for entry in snapshot.entries.values()} == {"E001", "E002", "E003", "E004"}
    assert {entry.body for entry in snapshot.entries.values()} == {
        f"process record {i}" for i in range(4)}


def test_entry_revisions_are_local_not_global(store):
    first = store.record("experience", "first", "body", "worker", "worker",
                         attributes={"subtype": "method"})
    second = store.record("experience", "second", "body", "worker", "worker",
                          attributes={"subtype": "boundary"})
    other = KnowledgeStore(store.root)
    updated = store.record("experience", "first", "first changed", "worker", "worker",
                           entry_id=first.entry_id, expected_revision=first.revision)
    assert updated.revision == 2
    result = other.record("experience", "second", "second changed", "worker", "worker",
                          entry_id=second.entry_id, expected_revision=second.revision)
    assert result.revision == 2 and other.snapshot().revision == 4
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="entry_revision_conflict"):
        other.record("experience", "first", "stale", "worker", "worker",
                     entry_id=first.entry_id, expected_revision=first.revision)
    assert tree_bytes(store.root) == before


def test_exact_fact_retry_is_idempotent(store):
    fact = candidate()
    fid = store.publish(fact, ACCEPTED)
    before = tree_bytes(store.root)
    assert store.publish(fact, ACCEPTED) == fid
    assert store.snapshot().revision == 1
    assert tree_bytes(store.root) == before


@pytest.mark.parametrize("field", ["statement", "proof"])
def test_normalized_hash_does_not_allow_nonidentical_retry(store, field):
    fact = candidate()
    changed = replace(fact, **{field: getattr(fact, field) + "  \n"})
    assert digest(fact) == digest(changed)
    store.publish(fact, ACCEPTED)
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="content_address_collision_or_nonidentical_retry"):
        store.publish(changed, ACCEPTED)
    assert tree_bytes(store.root) == before


def test_unknown_revoked_and_duplicate_dependencies_are_rejected(store):
    with pytest.raises(ValueError, match="unknown_or_revoked_predecessor"):
        store.publish(candidate("unknown", predecessors=["deadbeefdeadbeef"]), ACCEPTED)
    base = store.publish(candidate("base"), ACCEPTED)
    with pytest.raises(ValueError, match="duplicate_predecessors"):
        store.publish(candidate("duplicate", predecessors=[base, base]), ACCEPTED)
    store.revoke(base, "fixture withdrawal")
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="unknown_or_revoked_predecessor"):
        store.publish(candidate("revoked", predecessors=[base]), ACCEPTED)
    with pytest.raises(ValueError, match="unknown_or_revoked_fact"):
        store.snapshot().topological([base])
    assert tree_bytes(store.root) == before


def test_glossary_conflict_is_atomic_and_same_definition_is_allowed(store):
    store.publish(candidate("first", glossary_introduces={"S_M": "fixture set"}), ACCEPTED)
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="glossary_conflict"):
        store.publish(candidate("second", glossary_introduces={"S_M": "different set"}), ACCEPTED)
    assert tree_bytes(store.root) == before
    store.publish(candidate("third", glossary_introduces={"S_M": "fixture set"}), ACCEPTED)
    assert store.glossary() == {"S_M": "fixture set"}


def test_reference_edits_preserve_mathematical_content(store):
    base = store.publish(candidate("base"), ACCEPTED)
    fact = candidate("## nested\n$$x=x$$\n", proof="fixture proof\n  with spacing",
                     predecessors=[base], glossary_introduces={"S_M": "fixture set"})
    fid = store.publish(fact, ACCEPTED)
    before = store.read_fact(fid)
    record = store.get(fid)
    proof_bytes = (store.root / record.proof_path).read_bytes()
    refs = [{"title": "Fixture bibliography", "key": "F", "authors": ["Fixture"]}]
    assert store.set_external_refs(fid, refs) == refs
    after = store.read_fact(fid)
    assert after.fact_id == digest(fact) == fid
    assert (after.statement, after.proof, after.predecessors, after.glossary_introduces) == (
        before.statement, before.proof, before.predecessors, before.glossary_introduces)
    assert (store.root / record.proof_path).read_bytes() == proof_bytes
    assert store.external_refs(fid) == refs
    assert store.get(fid).revision == record.revision + 1


def test_cascade_revocation_reopens_supported_conjecture(store):
    base = store.publish(candidate("root premise"), ACCEPTED)
    source = store.record("conjecture", "guess", "derived claim", "worker", "worker")
    child = store.publish(candidate(source.body, predecessors=[base]), ACCEPTED,
                          source.entry_id, source.revision)
    grandchild = store.publish(candidate("grandchild", predecessors=[child]), ACCEPTED)
    assert store.snapshot().topological([grandchild]) == [base, child, grandchild]
    assert set(store.descendants(base)) == {child, grandchild}
    assert set(store.revoke(base, "fixture withdrawal")) == {base, child, grandchild}
    assert store.list() == [] and store.search("", kinds=["fact"]) == []
    for fid in (base, child, grandchild):
        assert store.get(fid) is None
        assert store.get(fid, include_revoked=True).status == "revoked"
    reopened = store.snapshot().entries[source.entry_id]
    assert reopened.status == "reopened"
    assert child in reopened.attributes["invalidated_by"]
    assert child in reopened.links
    hit = store.search("derived claim", kinds=["conjecture"])[0]
    assert hit["is_fact"] is False and child in hit["invalidated_links"]


@pytest.mark.parametrize("relative", ["root", "KNOWLEDGE.md", "knowledge/proofs",
                                      "knowledge/notes", "knowledge/.write.lock", "proof"])
def test_symlink_paths_are_rejected(store, tmp_path, relative):
    if relative == "proof":
        fid = store.publish(candidate(), ACCEPTED)
        path = store.root / store.get(fid).proof_path
    else:
        path = store.root if relative == "root" else store.root / relative
    target = tmp_path / "symlink-target"
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())
    before = target.read_bytes() if target.is_file() else tree_bytes(target)
    with pytest.raises(ValueError, match="symlink"):
        if relative == "root":
            KnowledgeStore(path)
        elif relative == "KNOWLEDGE.md":
            store.snapshot()
        elif relative == "proof":
            store.proof(fid)
        else:
            store.publish(candidate(intuition="fixture intuition"), ACCEPTED)
    after = target.read_bytes() if target.is_file() else tree_bytes(target)
    assert after == before


def test_failed_main_replace_leaves_unpublished_attachments_and_can_retry(store, monkeypatch):
    source = store.record("conjecture", "guess", "fixture statement", "worker", "worker")
    fact = candidate(intuition="fixture intuition")
    fid = digest(fact)
    before = store.path.read_bytes()
    revision = store.snapshot().revision
    original = _util.os.replace
    def fail_main(src, dst):
        if Path(dst) == store.path:
            raise OSError("fixture main replace failure")
        return original(src, dst)
    with monkeypatch.context() as patch:
        patch.setattr(_util.os, "replace", fail_main)
        with pytest.raises(OSError, match="fixture main replace failure"):
            store.publish(fact, ACCEPTED, source.entry_id, source.revision)
    assert store.path.read_bytes() == before
    assert store.snapshot().revision == revision
    assert not store.exists(fid)
    assert store.snapshot().entries[source.entry_id].status == "open"
    proof_path = store.root / f"knowledge/proofs/{fid}.md"
    assert md.attachment_body(proof_path.read_text(encoding="utf-8")) == fact.proof
    assert (store.root / f"knowledge/notes/{fid}-intuition.md").is_file()
    assert not list(store.root.rglob(".pending-*"))
    assert store.publish(fact, ACCEPTED, source.entry_id, source.revision) == fid
    assert store.exists(fid) and store.snapshot().entries[source.entry_id].status == "proved"
    assert store.snapshot().revision == revision + 1


def test_conflicting_orphan_attachment_is_not_overwritten(store):
    fact = candidate()
    fid = digest(fact)
    path = store.root / f"knowledge/proofs/{fid}.md"
    path.write_text(md.attachment(fid, "different orphan proof", "Verified proof"), encoding="utf-8")
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="immutable_attachment_conflict"):
        store.publish(fact, ACCEPTED)
    assert tree_bytes(store.root) == before and not store.exists(fid)


def test_snapshot_cache_revision_and_deep_immutability(store):
    fact = candidate(external_refs=[{"key": "F", "authors": ["Fixture"]}])
    fid = store.publish(fact, dict(ACCEPTED, provenance={"checks": ["fixture"]}))
    entry = store.record("experience", "note", "body", "worker", "worker",
                         attributes={"subtype": "method", "nested": {"values": [1, 2]}})
    old = store.snapshot()
    other = KnowledgeStore(store.root)
    assert other.snapshot() is old
    with pytest.raises(TypeError):
        old.facts[fid] = old.facts[fid]
    with pytest.raises(FrozenInstanceError):
        old.facts[fid].statement = "mutated"
    with pytest.raises(TypeError):
        old.facts[fid].verification["provenance"]["checks"] = ()
    with pytest.raises(TypeError):
        old.entries[entry.entry_id].attributes["nested"]["values"] = ()
    assert old.entries[entry.entry_id].attributes["nested"]["values"] == (1, 2)
    copied = old.facts[fid].to_dict()
    copied["external_refs"][0]["authors"].append("Changed")
    assert old.facts[fid].external_refs[0]["authors"] == ("Fixture",)
    other.set_external_refs(fid, [{"key": "new"}])
    current = store.snapshot()
    assert current is not old and current.revision == old.revision + 1
    assert old.facts[fid].external_refs[0]["key"] == "F"
    assert current.facts[fid].external_refs[0]["key"] == "new"


@pytest.mark.parametrize("legacy", ["fact_graph", "global_memory"])
def test_legacy_directories_are_rejected_without_writes(tmp_path, legacy):
    root = tmp_path / "legacy"
    (root / legacy).mkdir(parents=True)
    (root / legacy / "original.txt").write_text("untouched fixture", encoding="utf-8")
    before = tree_bytes(root)
    with pytest.raises(ValueError, match="unsupported_knowledge_format"):
        KnowledgeStore.initialize(root, "P")
    with pytest.raises(ValueError, match="unsupported_knowledge_format"):
        KnowledgeStore(root).record("experience", "note", "body", "worker", "worker",
                                    attributes={"subtype": "method"})
    assert tree_bytes(root) == before
    assert not (root / "knowledge").exists() and not (root / "KNOWLEDGE.md").exists()


@pytest.mark.parametrize("metadata", ["{}", "[]", '{"knowledge_format":"legacy"}', "{broken"])
def test_legacy_or_malformed_metadata_is_rejected_before_initialization(tmp_path, metadata):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "project.json").write_text(metadata, encoding="utf-8")
    before = tree_bytes(root)
    with pytest.raises(ValueError):
        KnowledgeStore.initialize(root, "P")
    assert tree_bytes(root) == before
    assert sorted(p.name for p in root.iterdir()) == ["project.json"]


@pytest.mark.parametrize("verification", [{"verdict": "wrong"}, {"verdict": "correct", "errors": ["fixture"]},
                                           {"verdict": "correct", "gaps": ["fixture"]}])
def test_unaccepted_verification_never_publishes(store, verification):
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="publication_requires_accepted_verification"):
        store.publish(candidate(), verification)
    assert tree_bytes(store.root) == before


def test_explicit_proof_and_full_fact_reject_tampered_attachment(store):
    fid = store.publish(candidate(), ACCEPTED)
    path = store.root / store.get(fid).proof_path
    path.write_text(md.attachment(fid, "tampered fixture proof", "Verified proof"), encoding="utf-8")
    assert store.get(fid).statement == "fixture statement"
    for read in (store.proof, store.read_fact):
        with pytest.raises(ValueError, match="proof_content_address_mismatch"):
            read(fid)


def test_reserved_record_boundaries_rejected_and_json_details_reversible(store):
    report = {"report": "fixture <!-- hyprover:fact boundary -->", "nested": ["<x>"]}
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="reserved_record_boundary"):
        store.record("feedback", "Verification", "Fixed summary", "v", "verifier",
                     details=report["report"])
    assert tree_bytes(store.root) == before
    encoded = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
    entry = store.record("feedback", "Verification", "Fixed summary", "v", "verifier", details=encoded)
    stored = md.attachment_body((store.root / entry.details_path).read_text(encoding="utf-8"))
    assert json.loads(stored) == report


def test_display_numbers_titles_links_and_source_fallback(store):
    source = store.record("conjecture", "A descriptive lemma", "lemma statement", "w", "worker")
    first = store.publish(candidate(source.body), ACCEPTED, source.entry_id, source.revision)
    second = store.publish(candidate("derived statement", title="Derived theorem", predecessors=[first]), ACCEPTED)
    snap = store.snapshot()
    assert snap.display_version == 1
    assert snap.facts[first].display_id == "F001"
    assert snap.facts[first].title == source.title
    assert snap.facts[second].display_id == "F002"
    assert source.display_id == "C001"
    text = store.path.read_text()
    assert "### F001 · A descriptive lemma [verified]" in text
    assert f"[F001 · A descriptive lemma](#fact-{first})" in text
    assert "本区目录 / Contents" in text
    assert "fixture proof" not in text
    hit = store.search("F002", kinds=["fact"])[0]
    assert hit["id"] == second and hit["display_name"] == "F002 · Derived theorem"
    assert hit["predecessor_labels"][first] == "F001 · A descriptive lemma"
    assert store.search("descriptive", kinds=["fact"])[0]["id"] == first
    assert store.read_fact(second).title == "Derived theorem"


def test_entry_numbers_survive_edits_and_new_insertions(store):
    first = store.record("experience", "One", "body", "w", "worker",
                         attributes={"subtype": "method"})
    second = store.record("experience", "Two", "body", "w", "worker",
                          attributes={"subtype": "boundary"})
    updated = store.record("experience", "One renamed", "body revised", "w", "worker",
                           entry_id=first.entry_id, expected_revision=first.revision)
    assert updated.display_id == first.display_id == "E001"
    assert second.display_id == "E002"
    assert store.record("experience", "Three", "body", "w", "worker",
                        attributes={"subtype": "method"}).display_id == "E003"
    assert store.record("route", "Explore", "body", "w", "worker").display_id == "R001"


def _previous_layout(store):
    snap = store.snapshot()
    old = replace(snap, display_version=0,
                  facts={fid: replace(f, title="", display_id="") for fid, f in snap.facts.items()},
                  entries={eid: replace(e, display_id="") for eid, e in snap.entries.items()})
    text = md.serialize(old)
    store.path.write_text(text, encoding="utf-8")
    return text, old


def test_previous_layout_reads_without_writes_or_proof_reads(store, monkeypatch):
    fid = store.publish(candidate(), ACCEPTED)
    entry = store.record("experience", "Note", "body", "w", "worker", details="long note",
                         attributes={"subtype": "method"})
    original_text, old = _previous_layout(store)
    before = tree_bytes(store.root)
    assert "display_version" not in original_text and "display_id" not in original_text
    original = Path.open
    def guard(path, mode="r", *args, **kwargs):
        assert not any(m in mode for m in "wax+")
        assert path.parent not in (store.root / "knowledge/proofs", store.root / "knowledge/notes")
        return original(path, mode, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guard)
        assert store.snapshot() == old
        assert md.serialize(md.parse(original_text)) == original_text
        assert store.search("F001")[0]["id"] == fid
        assert store.search("E001")[0]["id"] == entry.entry_id
    assert tree_bytes(store.root) == before


def test_explicit_display_refresh_preserves_all_mathematics_and_attachments(store):
    source = store.record("conjecture", "Source title", "fixture statement", "w", "worker", details="Long note")
    fact = candidate(intuition="An intuition")
    fid = store.publish(fact, ACCEPTED, source.entry_id, source.revision)
    _, old = _previous_layout(store)
    before = tree_bytes(store.root)
    new = store.refresh_display({fid: "Readable title"}, expected_revision=old.revision)
    assert new.revision == old.revision + 1 and new.facts[fid].display_id == "F001"
    assert store.read_fact(fid).proof == fact.proof
    assert store.read_fact(fid).statement == fact.statement
    assert digest(fact) == fid
    for name, data in before.items():
        if name != "KNOWLEDGE.md":
            assert (store.root / name).read_bytes() == data
    before_retry = tree_bytes(store.root)
    assert store.refresh_display({fid: "Readable title"}) == new
    assert tree_bytes(store.root) == before_retry
    assert store.publish(fact, ACCEPTED, source.entry_id, source.revision) == fid
    assert tree_bytes(store.root) == before_retry
    with pytest.raises(ValueError, match="document_revision_conflict"):
        store.refresh_display({fid: "Stale title"}, expected_revision=old.revision)


def test_revocation_keeps_numbers_and_titles_do_not_change_identity(store):
    fact = candidate(title="First title")
    fid = store.publish(fact, ACCEPTED)
    assert digest(replace(fact, title="Different title")) == fid
    assert store.publish(replace(fact, title="Different title"), ACCEPTED) == fid
    assert store.get(fid).title == "First title"
    store.refresh_display({fid: "Edited title"})
    store.revoke(fid, "withdrawn")
    next_id = store.publish(candidate("next statement"), ACCEPTED)
    assert store.get(next_id).display_id == "F002"
    assert store.get(fid, include_revoked=True).display_id == "F001"
    assert "### F001 · Edited title [revoked]" in store.path.read_text()


@pytest.mark.parametrize("title", ["two\nlines", "two\rlines", " ", "x" * 161, "<!-- hyprover:fake -->"])
def test_invalid_titles_never_change_notebook(store, title):
    before = tree_bytes(store.root)
    with pytest.raises(ValueError):
        store.publish(candidate(title=title), ACCEPTED)
    assert tree_bytes(store.root) == before


def test_display_title_html_and_markdown_are_literal(store):
    title = "Lemma [x](javascript:bad) <img> `code` \\psi *term*"
    fid = store.publish(candidate(title=title), ACCEPTED)
    text = store.path.read_text()
    assert "&lt;img&gt;" in text and "&#91;x&#93;" in text
    assert "&#92;psi" in text and "&#96;code&#96;" in text
    assert store.get(fid).title == title
    assert md.parse(text).facts[fid].title == title


@pytest.mark.parametrize("change", ["number", "title", "version"])
def test_display_canonical_tampering_is_rejected(store, change):
    fid = store.publish(candidate(title="Readable title"), ACCEPTED)
    text = store.path.read_text()
    if change == "number":
        text = text.replace("### F001 ·", "### F002 ·")
    elif change == "title":
        text = text.replace("### F001 · Readable title", "### F001 · Invented title")
    else:
        text = text.replace('"display_version": 1', '"display_version": 99')
    with pytest.raises(ValueError):
        md.parse(text)
    assert store.get(fid).display_id == "F001"


def test_duplicate_or_wrong_kind_display_ids_are_rejected(store):
    first = store.publish(candidate("first"), ACCEPTED)
    second = store.publish(candidate("second"), ACCEPTED)
    snap = store.snapshot()
    for code in ("F001", "C001", "F000", "F1"):
        facts = dict(snap.facts)
        facts[second] = replace(facts[second], display_id=code)
        with pytest.raises(ValueError, match="display_id"):
            md.validate(replace(snap, facts=facts))
    assert snap.facts[first].display_id == "F001"


def test_report_is_nonfact_author_editable_and_main_status_only(store):
    report = store.record("report", "Synthetic attempt", "No general conclusion.", "alice", "worker",
                          details="Synthetic attempt transcript.")
    assert report.status == "recorded" and report.display_id
    assert store.list() == [] and store.get(report.entry_id) is None
    assert store.search("Synthetic attempt") == []
    hit = store.search("Synthetic attempt", kinds=["report"])[0]
    assert hit["is_fact"] is False and hit["id"] == report.entry_id
    for author, role in (("bob", "worker"), ("main", "main")):
        before = tree_bytes(store.root)
        with pytest.raises(ValueError):
            store.record("report", report.title, "Rewritten by someone else.", author, role,
                         entry_id=report.entry_id, expected_revision=report.revision)
        assert tree_bytes(store.root) == before
    edited = store.record("report", "Synthetic attempt revised", "Only this attempt failed.", "alice", "worker",
                          entry_id=report.entry_id, expected_revision=report.revision)
    assert edited.author == "alice" and edited.display_id == report.display_id
    archived = store.record("report", edited.title, edited.body, "main", "main", status="archived",
                            entry_id=edited.entry_id, expected_revision=edited.revision)
    assert archived.author == "alice" and archived.status == "archived"
    assert store.search("", kinds=["report"]) == []
    assert store.search("", kinds=["report"], include_archived=True)[0]["id"] == report.entry_id


@pytest.mark.parametrize("status", [None, "archived"])
def test_main_cannot_add_null_metadata_to_another_authors_report(store, status):
    report = store.record("report", "Owned report", "Original author content.", "alice", "worker")
    assert "new" not in report.attributes
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="report_author_required"):
        store.record("report", report.title, report.body, "main", "main", status=status,
                     entry_id=report.entry_id, expected_revision=report.revision,
                     attributes={"new": None})
    assert tree_bytes(store.root) == before
    assert store.snapshot().entries[report.entry_id].to_dict() == report.to_dict()


@pytest.mark.parametrize("status", [None, "archived"])
@pytest.mark.parametrize("replacement", [True, 1.0])
def test_main_cannot_change_json_type_in_another_authors_report(store, status, replacement):
    report = store.record("report", "Owned metadata", "Original author content.", "alice", "worker",
                          attributes={"x": 1})
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="report_author_required"):
        store.record("report", report.title, report.body, "main", "main", status=status,
                     entry_id=report.entry_id, expected_revision=report.revision,
                     attributes={"x": replacement})
    assert tree_bytes(store.root) == before
    assert type(store.snapshot().entries[report.entry_id].attributes["x"]) is int


@pytest.mark.parametrize("status", ["verified", "proved", "refuted", "open"])
def test_report_cannot_claim_fact_or_conjecture_status(store, status):
    before = tree_bytes(store.root)
    with pytest.raises(ValueError):
        store.record("report", "Invalid promotion", "Synthetic only.", "w", "worker", status=status,
                     details="Must not become an attachment.")
    assert tree_bytes(store.root) == before


def test_report_cannot_be_a_predecessor_or_conjecture_source(store):
    report = store.record("report", "Synthetic observation", "fixture statement", "w", "worker")
    before = tree_bytes(store.root)
    with pytest.raises(ValueError):
        store.publish(candidate(predecessors=[report.entry_id]), ACCEPTED)
    with pytest.raises(ValueError):
        store.publish(candidate(), ACCEPTED, report.entry_id, report.revision)
    assert tree_bytes(store.root) == before and not store.list()


@pytest.mark.parametrize("subtype", [None, "", "success", "report", "audit", 1])
def test_new_experience_never_infers_a_subtype(store, subtype):
    before = tree_bytes(store.root)
    attributes = {} if subtype is None else {"subtype": subtype}
    with pytest.raises(ValueError):
        store.record("experience", "Unclassified", "A single attempt is not a general lesson.", "w", "worker",
                     attributes=attributes, details="Do not persist rejected details.")
    assert tree_bytes(store.root) == before


@pytest.mark.parametrize("subtype", ["method", "boundary", "review"])
def test_only_three_explicit_experience_types_roundtrip(store, subtype):
    target = store.record("report", "Synthetic attempt", "A scoped observation.", "w", "worker")
    attrs = {"subtype": subtype}
    if subtype == "review":
        attrs["audits"] = [{"target_id": target.entry_id, "revision": target.revision,
                            "scope": "The reported inference only.", "not_covered": "Other arguments."}]
    note = store.record("experience", subtype.title(), "A concrete scoped lesson.", "w", "worker",
                        links=[target.entry_id], attributes=attrs)
    restored = KnowledgeStore(store.root).snapshot().entries[note.entry_id]
    assert restored.attributes["subtype"] == subtype
    assert restored.attributes["research_policy_version"] == 1
    assert store.search_page("", subtype=subtype)["results"][0]["id"] == note.entry_id


@pytest.mark.parametrize("case", ["missing", "empty", "future", "zero", "bool", "duplicate", "unlinked",
                                 "unknown", "missing_scope", "missing_not_covered", "blank_scope"])
def test_invalid_review_audits_fail_before_writing_attachments(store, case):
    target = store.record("report", "Target", "Original scope.", "w", "worker")
    audit = {"target_id": target.entry_id, "revision": target.revision,
             "scope": "One inference.", "not_covered": "Everything else."}
    links = [target.entry_id]
    attrs = {"subtype": "review", "audits": [audit]}
    if case == "missing":
        attrs.pop("audits")
    elif case == "empty":
        attrs["audits"] = []
    elif case in ("future", "zero", "bool"):
        audit["revision"] = {"future": target.revision + 1, "zero": 0, "bool": True}[case]
    elif case == "duplicate":
        attrs["audits"].append(dict(audit))
    elif case == "unlinked":
        links = []
    elif case == "unknown":
        audit["target_id"] = "entry-ffffffffffffffff"
    elif case == "blank_scope":
        audit["scope"] = " "
    else:
        audit.pop(case.removeprefix("missing_"))
    before = tree_bytes(store.root)
    with pytest.raises(ValueError):
        store.record("experience", "Review", "An object-specific issue.", "reviewer", "worker",
                     links=links, attributes=attrs, details="Rejected review detail.")
    assert tree_bytes(store.root) == before


def test_historical_audit_requires_opt_in_and_keeps_immutable_baseline(store):
    target = store.record("report", "Target", "First revision.", "alice", "worker")
    updated = store.record("report", target.title, "Second revision.", "alice", "worker",
                           entry_id=target.entry_id, expected_revision=target.revision)
    audit = {"target_id": target.entry_id, "revision": target.revision,
             "scope": "The original argument.", "not_covered": "Later revisions."}
    before = tree_bytes(store.root)
    with pytest.raises(ValueError, match="historical"):
        store.record("report", "Historical review", "No finding within this scope.", "bob", "worker",
                     attributes={"audits": [audit]}, links=[target.entry_id])
    assert tree_bytes(store.root) == before
    audit["historical"] = True
    report = store.record("report", "Historical review", "No finding within this scope.", "bob", "worker",
                          attributes={"audits": [audit]}, links=[target.entry_id])
    for field, value in (("revision", updated.revision), ("scope", "A different scope."),
                         ("not_covered", "Nothing omitted."), ("historical", False)):
        changed = dict(audit, **{field: value})
        before = tree_bytes(store.root)
        with pytest.raises(ValueError, match="immutable"):
            store.record("report", report.title, report.body, "bob", "worker",
                         entry_id=report.entry_id, expected_revision=report.revision,
                         attributes={"audits": [changed]})
        assert tree_bytes(store.root) == before
    assert store.snapshot().entries[report.entry_id].attributes["audits"][0]["revision"] == 1


@pytest.mark.parametrize("attributes", [
    {"research_policy_version": 1}, {"record_request": {"id": "forged", "fingerprint": "0" * 64}},
    {"value": float("nan")}, {"value": float("inf")}, {"value": {1: "nonstring key"}}, {"value": {1, 2}},
])
def test_reserved_or_nonfinite_metadata_cannot_be_submitted(store, attributes):
    before = tree_bytes(store.root)
    with pytest.raises(ValueError):
        store.record("report", "Invalid metadata", "Synthetic.", "w", "worker",
                     attributes=attributes, details="Rejected detail.")
    assert tree_bytes(store.root) == before


def test_report_request_retry_deduplicates_without_changing_bytes(store):
    arguments = dict(kind="report", title="Retry", body="Exact original payload.", author="alice", role="worker",
                     details="Full details.\r\nUnchanged bytes.", attributes={"scope": ["one", 2]}, request_id="attempt-1")
    first = store.record(**arguments)
    before = tree_bytes(store.root)
    assert KnowledgeStore(store.root).record(**arguments) == first
    assert tree_bytes(store.root) == before
    assert store.snapshot().revision == 1
    assert first.attributes["record_request"]["id"] == "attempt-1"
    with pytest.raises(ValueError):
        store.record(**dict(arguments, entry_id=first.entry_id, expected_revision=first.revision))
    assert tree_bytes(store.root) == before


@pytest.mark.parametrize("field", ["title", "body", "details", "attributes", "links", "status"])
def test_request_id_rejects_any_payload_change(store, field):
    linked = store.record("report", "Link target", "Synthetic.", "alice", "worker")
    args = dict(kind="report", title="Retry", body="Original.", author="alice", role="worker",
                details="Original details.", attributes={"scope": "original"}, links=[], request_id="same-request")
    store.record(**args)
    changes = {"title": "Different", "body": "Original. ", "details": "Different details.",
               "attributes": {"scope": "changed"}, "links": [linked.entry_id], "status": "archived"}
    before = tree_bytes(store.root)
    with pytest.raises(ValueError):
        store.record(**dict(args, **{field: changes[field]}))
    assert tree_bytes(store.root) == before


def test_record_request_key_is_scoped_by_project_author_and_kind(store, tmp_path):
    first = store.record("report", "One", "Body.", "alice", "worker", request_id="shared-key")
    other_author = store.record("report", "Two", "Different body.", "bob", "worker", request_id="shared-key")
    other_kind = store.record("experience", "Three", "Technique.", "alice", "worker",
                              attributes={"subtype": "method"}, request_id="shared-key")
    other = KnowledgeStore.initialize(tmp_path / "other", "P")
    separate = other.record("report", "Four", "Independent project.", "alice", "worker", request_id="shared-key")
    assert len({first.entry_id, other_author.entry_id, other_kind.entry_id}) == 3
    assert separate.body == "Independent project." and len(other.snapshot().entries) == 1


def _process_report_retry(root, ready):
    assert ready.wait(15), "retry start timed out"
    KnowledgeStore(Path(root)).record("report", "Concurrent retry", "One synthetic attempt.", "alice", "worker",
                                      details="Same full detail.", request_id="concurrent-report")


def test_concurrent_report_retries_create_one_entry_and_attachment(store):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    processes = [context.Process(target=_process_report_retry, args=(str(store.root), ready)) for _ in range(4)]
    try:
        for process in processes:
            process.start()
        ready.set()
        for process in processes:
            process.join(20)
            assert not process.is_alive() and process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    snapshot = store.snapshot()
    assert snapshot.revision == 1 and len(snapshot.entries) == 1
    assert len(list((store.root / "knowledge/notes").glob("*.md"))) == 1


def test_report_main_replace_failure_keeps_notebook_atomic_and_retryable(store, monkeypatch):
    before = store.path.read_bytes()
    original = _util.os.replace
    def fail_main(src, dst):
        if Path(dst) == store.path:
            raise OSError("synthetic report commit failure")
        return original(src, dst)
    args = dict(kind="report", title="Atomic report", body="Synthetic.", author="w", role="worker",
                details="Detached details.", request_id="atomic-report")
    with monkeypatch.context() as patch:
        patch.setattr(_util.os, "replace", fail_main)
        with pytest.raises(OSError, match="synthetic report commit failure"):
            store.record(**args)
    assert store.path.read_bytes() == before and store.snapshot().revision == 0
    committed = store.record(**args)
    assert store.record(**args).entry_id == committed.entry_id
    assert len(store.snapshot().entries) == 1 and store.snapshot().revision == 1
    assert not list(store.root.rglob(".pending-*"))


@pytest.mark.parametrize("filename", ["knowledge-v1-before-display.md", "knowledge-v1-before-reports.md"])
def test_real_old_notebooks_roundtrip_without_normalizing_history(store, filename):
    raw = (Path(__file__).parent / "fixtures" / filename).read_bytes()
    store.path.write_bytes(raw)
    before = tree_bytes(store.root)
    snapshot = store.snapshot()
    assert md.serialize(md.parse(raw.decode("utf-8"))).encode("utf-8") == raw
    legacy = next(iter(snapshot.entries.values()))
    assert legacy.attributes == {}
    assert store.search("Historical observation")[0]["id"] == legacy.entry_id
    assert store.entry_detail(legacy.entry_id)["body"] == legacy.body
    assert tree_bytes(store.root) == before
    with pytest.raises(ValueError, match="explicit_subtype"):
        store.record("experience", legacy.title, "Changed without classification.", "worker", "worker",
                     entry_id=legacy.entry_id, expected_revision=legacy.revision)
    assert tree_bytes(store.root) == before
    archived = store.record("experience", legacy.title, legacy.body, "main", "main", status="archived",
                            entry_id=legacy.entry_id, expected_revision=legacy.revision)
    assert archived.attributes == {} and archived.status == "archived"
    classified = store.record("experience", legacy.title, "A scoped method with conditions.", "worker", "worker",
                              entry_id=archived.entry_id, expected_revision=archived.revision,
                              attributes={"subtype": "method"})
    assert classified.attributes["subtype"] == "method"


def test_first_report_enables_section_without_renumbering_historical_entries(store):
    raw = (Path(__file__).parent / "fixtures/knowledge-v1-before-reports.md").read_bytes()
    store.path.write_bytes(raw)
    before = store.snapshot()
    legacy = next(iter(before.entries.values()))
    assert '<a id="report"></a>' not in raw.decode("utf-8")
    report = store.record("report", "First report", "Synthetic attempt.", "worker", "worker")
    snapshot = store.snapshot()
    assert '<a id="report"></a>' in store.path.read_text()
    assert snapshot.entries[legacy.entry_id].to_dict() == legacy.to_dict()
    assert report.display_id and legacy.display_id == "E001"
    note = store.record("experience", "A method", "Technique with prerequisites.", "worker", "worker",
                        attributes={"subtype": "method"})
    assert note.display_id == "E002"
    text = store.path.read_text()
    assert md.serialize(md.parse(text)) == text


def test_browse_page_filters_before_time_order_and_pagination(store):
    linked = store.record("conjecture", "Target", "Open target.", "alice", "worker")
    for index in range(3):
        store.record("report", f"Noise {index}", "needle needle needle", "bob", "worker")
    notes = [store.record("experience", f"Method {index}", "needle", "alice", "worker",
                          attributes={"subtype": "method"}, links=[linked.entry_id]) for index in range(3)]
    store.record("experience", "Other boundary", "needle needle", "alice", "worker",
                 attributes={"subtype": "boundary"})
    store.record("experience", "Other author", "needle needle", "bob", "worker",
                 attributes={"subtype": "method"}, links=[linked.entry_id])
    store.record("experience", "Unlinked method", "needle needle", "alice", "worker",
                 attributes={"subtype": "method"})
    first = store.search_page("needle", subtype="method", author="alice", linked_to=linked.display_id,
                              limit=2, sort_by="updated")
    second = store.search_page("needle", subtype="method", author="alice", linked_to=linked.entry_id,
                               limit=2, offset=2, expected_snapshot_revision=first["snapshot_revision"])
    assert first["total"] == second["total"] == 3
    assert first["offset"] == 0 and first["limit"] == 2 and first["has_more"] is True
    assert second["offset"] == 2 and second["has_more"] is False
    ids = [row["id"] for row in first["results"] + second["results"]]
    assert len(ids) == len(set(ids)) == 3 and set(ids) == {note.entry_id for note in notes}
    exhausted = store.search_page("needle", subtype="method", author="alice", linked_to=linked.entry_id, offset=10)
    assert exhausted["results"] == [] and exhausted["total"] == 3 and not exhausted["has_more"]
    with pytest.raises(ValueError):
        store.search_page("needle", subtype="method", kinds=["report"])
    with pytest.raises(ValueError):
        store.search_page("needle", linked_to="E999")


def test_browse_returns_all_candidates_without_scoring_and_is_read_only(store):
    older = store.record("conjecture", "Convexity", "convex convex convex", "alice", "worker")
    newer = store.record("conjecture", "Positive curvature", "A different mathematical formulation.", "bob", "worker")
    before = tree_bytes(store.root)
    page = store.search_page(limit=1, kinds=["conjecture"])
    assert page["sort_by"] == "updated" and page["total"] == 2 and page["has_more"]
    assert page["results"][0]["id"] == newer.entry_id
    more = store.search_page(limit=1, kinds=["conjecture"], offset=1,
                              expected_snapshot_revision=page["snapshot_revision"])
    assert more["results"][0]["id"] == older.entry_id and not more["has_more"]
    assert all("score" not in row for row in page["results"] + more["results"])
    assert tree_bytes(store.root) == before


def test_literal_lookup_and_exact_ids_do_not_rank_or_expand_keywords(store):
    target = store.record("conjecture", "Boundary condition", "边界条件 alpha beta", "alice", "worker")
    store.record("conjecture", "Related note", "Only alpha; also refers to " + target.display_id, "bob", "worker")
    assert store.search("ALPHA BETA")[0]["id"] == target.entry_id
    assert store.search("边界条件")[0]["id"] == target.entry_id
    assert store.search("alpha.*beta") == []
    assert store.search("beta alpha") == []
    for key in (target.entry_id.upper(), target.display_id.lower()):
        assert [row["id"] for row in store.search(key)] == [target.entry_id]


def test_time_sort_and_legacy_sort_alias_do_not_score(store):
    first = store.record("report", "First", "needle needle", "w", "worker")
    second = store.record("report", "Second", "needle", "w", "worker")
    store.record("report", first.title, "needle needle revised", "w", "worker",
                 entry_id=first.entry_id, expected_revision=first.revision)
    assert store.search_page(kinds=["report"])["results"][0]["id"] == first.entry_id
    assert store.search_page(kinds=["report"], sort_by="created")["results"][0]["id"] == second.entry_id
    legacy = store.search_page("needle", kinds=["report"], sort_by="relevance")
    assert legacy == store.search_page("needle", kinds=["report"], sort_by="updated")
    assert legacy["sort_by"] == "updated" and all("score" not in row for row in legacy["results"])


def test_search_defaults_preserve_feedback_and_hide_reports_and_archived(store):
    report = store.record("report", "Report", "needle", "w", "worker")
    feedback = store.record("feedback", "Verifier response", "needle", "v", "verifier")
    active = store.record("experience", "Active method", "needle", "w", "worker",
                          attributes={"subtype": "method"})
    store.record("experience", "Archived boundary", "needle", "w", "worker", status="archived",
                 attributes={"subtype": "boundary"})
    hits = store.search("needle")
    assert isinstance(hits, list)
    assert {row["id"] for row in hits} == {feedback.entry_id, active.entry_id}
    assert store.search_page("needle", kinds=["report"])["results"][0]["id"] == report.entry_id
    assert store.search_page("needle", kinds=["experience"], include_archived=True)["total"] == 2


def test_search_snapshot_and_updated_order_are_stable(store):
    first = store.record("report", "First", "needle", "w", "worker")
    second = store.record("report", "Second", "needle", "w", "worker")
    baseline = store.snapshot()
    page = store.search_page("", kinds=["report"], sort_by="updated", snapshot=baseline)
    rows = page["results"]
    assert [row["updated_at"] for row in rows] == sorted((row["updated_at"] for row in rows), reverse=True)
    store.record("report", first.title, "needle revised", "w", "worker",
                 entry_id=first.entry_id, expected_revision=first.revision)
    with pytest.raises(ValueError, match="search_snapshot_conflict"):
        store.search_page("", kinds=["report"], expected_snapshot_revision=page["snapshot_revision"])
    pinned = store.search_page("", kinds=["report"], snapshot=baseline,
                               expected_snapshot_revision=baseline.revision)
    assert pinned["snapshot_revision"] == baseline.revision
    assert {row["id"] for row in pinned["results"]} == {first.entry_id, second.entry_id}
    assert all(row["revision"] == 1 and row["body"] == "needle" for row in pinned["results"])
    with pytest.raises(ValueError):
        store.search_page("", sort_by="not-a-sort")


def test_entry_detail_is_explicit_and_missing_attachment_is_reported(store):
    body = "Visible summary.\n" + "Long original body. " * 100
    note = store.record("report", "Detail report", body, "alice", "worker", details="unindexedattachmenttokenxyz")
    row = store.entry_detail(note.entry_id)
    assert row["id"] == note.entry_id and row["body"] == body and row["is_fact"] is False
    assert row["details"] == "unindexedattachmenttokenxyz" and row["author"] == "alice"
    assert store.search("unindexedattachmenttokenxyz", kinds=["report"]) == []
    path = store.root / note.details_path
    path.unlink()
    before = tree_bytes(store.root)
    missing = store.entry_detail(note.entry_id)
    assert missing["body"] == body and missing["details_error"] == "missing_attachment"
    assert tree_bytes(store.root) == before


def test_entry_detail_rejects_symlink_instead_of_hiding_it_as_missing(store, tmp_path):
    note = store.record("report", "Unsafe detail", "Safe body.", "alice", "worker", details="Safe detail.")
    path = store.root / note.details_path
    target = tmp_path / "private-file"
    target.write_text("Must not be returned.")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        store.entry_detail(note.entry_id)


@pytest.mark.parametrize("status", [None, "recorded"])
def test_main_noop_on_another_authors_report_preserves_audit_baseline(store, status):
    report = store.record("report", "Owned report", "Original author content.", "alice", "worker",
                          details="Original long detail.")
    review = store.record("report", "Scoped audit", "No issue found within the checked scope.", "reviewer", "worker",
                          links=[report.entry_id], attributes={"audits": [{
                              "target_id": report.entry_id, "revision": report.revision,
                              "scope": "The stated inference.", "not_covered": "Other methods."}]})
    revision = store.snapshot().revision
    before = tree_bytes(store.root)
    unchanged = store.record("report", report.title, report.body, "main", "main", status=status,
                             entry_id=report.entry_id, expected_revision=report.revision)
    assert unchanged == report
    assert store.snapshot().revision == revision
    detail = store.entry_detail(review.entry_id)
    assert detail["warnings"] == [] and detail["audits"][0]["version_changed"] is False
    assert tree_bytes(store.root) == before


def test_random_new_entry_identity_collision_cannot_overwrite_existing_entry(store, monkeypatch):
    original = store.record("report", "Original report", "Original author content.", "alice", "worker")
    class Collision:
        hex = original.entry_id.removeprefix("entry-") + "0" * 16
    before = tree_bytes(store.root)
    monkeypatch.setattr("hyprover.core.knowledge.uuid.uuid4", lambda: Collision())
    with pytest.raises(ValueError):
        store.record("report", "Colliding report", "A different author's attempted creation.", "bob", "worker",
                     details="No attachment should be written for a rejected identity.")
    assert tree_bytes(store.root) == before
    assert store.snapshot().entries[original.entry_id] == original
