"""Entry policy uses synthetic records, not mathematical certification."""
from dataclasses import replace

import pytest

from hyprover.core.entry_policy import (
    annotations, json_object, prepare_attributes, validate_audits, validate_entry_policy,
)
from hyprover.core.schema import KnowledgeEntry

TARGET = "entry-0000000000000001"
ENTRY = "entry-0000000000000002"


@pytest.fixture
def records():
    return {TARGET: KnowledgeEntry(TARGET, "route", "Synthetic route", "An open step", "main",
                                    status="active", revision=2, display_id="R001")}


def audit(**changes):
    return dict(target_id=TARGET, revision=2, scope="Only the specified step",
                not_covered="All other steps", **changes)


def prepare(records, subtype="method", **kwargs):
    incoming = {"subtype": subtype, **kwargs}
    return prepare_attributes("experience", None, incoming, content_changed=True,
                              links=[TARGET], records=records)


@pytest.mark.parametrize("subtype", ["method", "boundary", "review"])
def test_three_subtypes(records, subtype):
    attrs = prepare(records, subtype, **({"audits": [audit()]} if subtype == "review" else {}))
    assert attrs["subtype"] == subtype
    assert attrs["research_policy_version"] == 1


@pytest.mark.parametrize("subtype", [None, "other", "failure", "", [], True])
def test_no_fourth_subtype(records, subtype):
    with pytest.raises(ValueError, match="experience_subtype"):
        prepare(records, subtype)


def test_review_requires_specific_baseline(records):
    with pytest.raises(ValueError, match="review_requires_audits"):
        prepare(records, "review")
    with pytest.raises(ValueError, match="subtype_only_for_experience"):
        prepare_attributes("report", None, {"subtype": "review"}, content_changed=True,
                           links=[TARGET], records=records)


@pytest.mark.parametrize("key", ["research_policy_version", "record_request", "invalidated_by",
                                 "resolved_at", "resolution_fact_id"])
def test_service_fields_cannot_be_supplied(records, key):
    with pytest.raises(ValueError, match="reserved_entry_attributes"):
        prepare(records, **{key: None})


@pytest.mark.parametrize("metadata", [[], "text", {1: "value"}, {"x": float("nan")},
                                     {"x": float("inf")}, {"x": {1}}, {"x": object()}])
def test_metadata_must_be_finite_json(metadata):
    with pytest.raises(ValueError):
        json_object(metadata)


def test_metadata_is_copied_and_cycles_rejected():
    metadata = {"x": [1, {"y": True}]}
    copied = json_object(metadata)
    copied["x"][1]["y"] = False
    assert metadata["x"][1]["y"] is True
    metadata["loop"] = metadata
    with pytest.raises(ValueError, match="too_deep"):
        json_object(metadata)


def test_legacy_read_and_status_maintenance_do_not_guess(records):
    old = KnowledgeEntry(ENTRY, "experience", "Old", "Original text", "alice",
                         attributes={"subtype": "free historical classification"})
    validate_entry_policy(old, records)
    assert annotations(old, records)["legacy_experience"]
    attrs = prepare_attributes("experience", old, {}, content_changed=False,
                               links=[], records=records)
    assert attrs == old.to_dict()["attributes"]
    with pytest.raises(ValueError, match="explicit_subtype"):
        prepare_attributes("experience", old, {}, content_changed=True, links=[], records=records)
    attrs = prepare_attributes("experience", old, {"subtype": "boundary"}, content_changed=True,
                               links=[], records=records)
    assert attrs == {"subtype": "boundary", "research_policy_version": 1}


@pytest.mark.parametrize("version", [0, 2, True, "1", None])
def test_illegal_policy_version_rejected_on_read(records, version):
    e = KnowledgeEntry(ENTRY, "experience", "Title", "Body", "a",
                       attributes={"subtype": "method", "research_policy_version": version})
    with pytest.raises(ValueError, match="unsupported_research_policy_version"):
        validate_entry_policy(e, records)


@pytest.mark.parametrize("changes,error", [
    ({"target_id": "missing"}, "unknown_audit_target"),
    ({"revision": 0}, "invalid_audit_revision"),
    ({"revision": True}, "invalid_audit_revision"),
    ({"revision": 3}, "invalid_audit_revision"),
    ({"revision": 1}, "audit_revision_conflict"),
    ({"scope": " "}, "invalid_audit_scope"),
    ({"not_covered": None}, "invalid_audit_not_covered"),
    ({"historical": "true"}, "invalid_audit_historical"),
    ({"extra": 1}, "invalid_audit_fields"),
])
def test_invalid_audit(records, changes, error):
    a = {**audit(), **changes}
    with pytest.raises(ValueError, match=error):
        validate_audits([a], [TARGET], records, submitting=True)


def test_links_duplicates_and_empty_audits(records):
    for audits, links, error in [([], [TARGET], "nonempty"), ([audit()], [], "requires_link"),
                                 ([audit(), audit()], [TARGET], "duplicate")]:
        with pytest.raises(ValueError, match=error):
            validate_audits(audits, links, records)


def test_historical_baseline_and_read_only_warnings(records):
    a = {**audit(), "revision": 1, "historical": True}
    attrs = prepare(records, "review", audits=[a])
    e = KnowledgeEntry(ENTRY, "experience", "Review", "Specific doubt", "a",
                       links=(TARGET,), attributes=attrs)
    view = annotations(e, records)
    assert not view["legacy_experience"]
    assert view["audits"][0]["current_revision"] == 2
    assert view["warnings"][0]["code"] == "audit_target_updated"
    assert e.attributes["audits"][0]["revision"] == 1
    updated_records = {TARGET: replace(records[TARGET], revision=3)}
    validate_entry_policy(e, updated_records)
    assert annotations(e, updated_records)["audits"][0]["version_changed"]


def test_baseline_cannot_be_rewritten_and_stale_entry_can_be_maintained(records):
    attrs = prepare(records, "review", audits=[audit()])
    old = KnowledgeEntry(ENTRY, "experience", "Review", "A doubt", "a", links=(TARGET,), attributes=attrs)
    newer = {TARGET: replace(records[TARGET], revision=3)}
    preserved = prepare_attributes("experience", old, {}, content_changed=True,
                                   links=[TARGET], records=newer)
    assert preserved["audits"][0]["revision"] == 2
    with pytest.raises(ValueError, match="audit_baseline_is_immutable"):
        prepare_attributes("experience", old, {"audits": [{**audit(), "revision": 3}]},
                           content_changed=True, links=[TARGET], records=newer)


@pytest.mark.parametrize("custom", ["legacy-ticket", None, [1, 2], {"id": "old", "fingerprint": "custom"}])
def test_unmarked_request_metadata_is_not_new_protocol(records, custom):
    old = KnowledgeEntry(ENTRY, "experience", "Old", "Original", "alice",
                         attributes={"record_request": custom})
    validate_entry_policy(old, records)
    preserved = prepare_attributes("experience", old, {}, content_changed=False, links=[], records=records)
    assert preserved == {"record_request": custom}
    with pytest.raises(ValueError, match="legacy_reserved_attribute_conflict"):
        prepare_attributes("experience", old, {"subtype": "method"}, content_changed=True,
                           links=[], records=records)


def test_legacy_audit_metadata_needs_explicit_reinterpretation(records):
    old = KnowledgeEntry(ENTRY, "experience", "Old", "Original", "alice",
                         links=(TARGET,), attributes={"audits": [{**audit(), "revision": 1}]})
    validate_entry_policy(old, records)
    with pytest.raises(ValueError, match="legacy_audits_require_explicit_baseline"):
        prepare_attributes("experience", old, {"subtype": "review"}, content_changed=True,
                           links=[TARGET], records=records)
    with pytest.raises(ValueError, match="audit_revision_conflict"):
        prepare_attributes("experience", old, {"subtype": "review", "audits": [{**audit(), "revision": 1}]},
                           content_changed=True, links=[TARGET], records=records)


@pytest.mark.parametrize("custom", ["legacy-key", {"id": "legacy-key", "fingerprint": "not-a-receipt"}])
def test_old_custom_request_field_survives_reads_and_archive(tmp_path, custom):
    from hyprover.core import KnowledgeStore, knowledge_markdown as md
    from hyprover.core.schema import KnowledgeSnapshot

    store = KnowledgeStore.initialize(tmp_path / "project", "P")
    old = KnowledgeEntry(ENTRY, "experience", "Legacy", "Original body", "alice",
                         attributes={"record_request": custom})
    text = md.serialize(md.prepare_display(KnowledgeSnapshot("P", 1, entries={ENTRY: old})))
    store.path.write_text(text, encoding="utf-8")
    assert store.entry_detail(ENTRY)["attributes"]["record_request"] == custom
    assert store.path.read_text(encoding="utf-8") == text
    created = store.record("experience", "New", "A scoped lesson", "alice", "worker",
                           attributes={"subtype": "method"}, request_id="legacy-key")
    assert created.entry_id != ENTRY
    archived = store.record("experience", old.title, old.body, "main", "main",
                            entry_id=ENTRY, expected_revision=1, status="archived")
    assert archived.to_dict()["attributes"] == {"record_request": custom}
    assert "research_policy_version" not in archived.attributes


@pytest.mark.parametrize("kind", ["overview", "route", "conjecture", "lead", "feedback"])
def test_other_entry_kinds_have_explicit_retry_policy(tmp_path, kind):
    from hyprover.core import KnowledgeStore

    store = KnowledgeStore.initialize(tmp_path / "project", "P")
    args = dict(kind=kind, title="Synthetic", body="Original", author="a",
                role="verifier" if kind == "feedback" else "main", request_id="one-request")
    first = store.record(**args)
    assert first.attributes["research_policy_version"] == 1
    assert store.record(**args) == first
    update = {k: v for k, v in args.items() if k != "request_id"}
    update.update(body="Updated", entry_id=first.entry_id, expected_revision=first.revision)
    revised = store.record(**update)
    assert revised.revision == 2
    assert store.record(**args) == revised
