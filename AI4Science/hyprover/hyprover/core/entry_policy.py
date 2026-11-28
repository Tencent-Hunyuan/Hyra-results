"""Research-entry policy; validation and read-only warnings never certify claims."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

from .schema import EXPERIENCE_SUBTYPES, KNOWLEDGE_MARKERS, RESEARCH_POLICY_VERSION, FactRecord, KnowledgeEntry

RESERVED_ATTRIBUTES = frozenset({
    "resolution_fact_id", "invalidated_by", "resolved_at", "research_policy_version", "record_request",
})
REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def json_object(value: object) -> dict:
    """Copy JSON metadata before locking; reject nonfinite, cyclic or non-JSON input."""
    def copy(item, depth=0):
        if depth > 64:
            raise ValueError("entry_metadata_too_deep")
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("entry_metadata_keys_must_be_strings")
            return {key: copy(val, depth + 1) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [copy(val, depth + 1) for val in item]
        raise ValueError("entry_metadata_must_be_finite_json")

    if not isinstance(value, Mapping):
        raise ValueError("entry_attributes_must_be_object")
    return copy(value)


def check_request_id(request_id: object) -> None:
    if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
        raise ValueError("invalid_record_request_id")


def _text(value: object, field: str, marker: str | None) -> None:
    reserved = (marker,) if marker is not None else KNOWLEDGE_MARKERS.values()
    if not isinstance(value, str) or not value.strip() or any(prefix in value for prefix in reserved):
        raise ValueError(f"invalid_audit_{field}")


def validate_audits(audits: object, links, records: Mapping, *, submitting: bool = False,
                    marker: str | None = None) -> None:
    if not isinstance(audits, (list, tuple)) or not audits:
        raise ValueError("audits_must_be_nonempty_list")
    seen = set()
    for audit in audits:
        if not isinstance(audit, Mapping) or set(audit) - {
            "target_id", "revision", "scope", "not_covered", "historical",
        }:
            raise ValueError("invalid_audit_fields")
        target_id = audit.get("target_id")
        if not isinstance(target_id, str) or target_id not in records:
            raise ValueError("unknown_audit_target")
        if target_id in seen:
            raise ValueError("duplicate_audit_target")
        seen.add(target_id)
        if target_id not in links:
            raise ValueError("audit_target_requires_link")
        revision = audit.get("revision")
        if type(revision) is not int or revision < 1 or revision > records[target_id].revision:
            raise ValueError("invalid_audit_revision")
        historical = audit.get("historical", False)
        if type(historical) is not bool:
            raise ValueError("invalid_audit_historical")
        if submitting and revision != records[target_id].revision and not historical:
            raise ValueError("audit_revision_conflict: historical=true required for an older baseline")
        _text(audit.get("scope"), "scope", marker)
        _text(audit.get("not_covered"), "not_covered", marker)


def validate_entry_policy(entry: KnowledgeEntry, records: Mapping, *, marker: str | None = None) -> None:
    """Validate marked records; preserve unmarked historical metadata verbatim."""
    attrs = entry.attributes
    version = attrs.get("research_policy_version")
    if "research_policy_version" not in attrs:
        if entry.kind == "report":
            raise ValueError("report_requires_research_policy")
        return
    if type(version) is not int or version != RESEARCH_POLICY_VERSION:
        raise ValueError("unsupported_research_policy_version")
    json_object(attrs)
    if "record_request" in attrs:
        request = attrs["record_request"]
        if not isinstance(request, Mapping) or set(request) != {"id", "fingerprint"}:
            raise ValueError("invalid_record_request_metadata")
        check_request_id(request["id"])
        if not isinstance(request["fingerprint"], str) or not re.fullmatch(r"[0-9a-f]{64}", request["fingerprint"]):
            raise ValueError("invalid_record_request_fingerprint")
    elif entry.kind not in ("experience", "report"):
        raise ValueError("research_policy_kind_mismatch")
    if "audits" in attrs and entry.kind not in ("experience", "report"):
        raise ValueError("audits_only_for_research_entries")
    if entry.kind == "experience":
        subtype = attrs.get("subtype")
        if not isinstance(subtype, str) or subtype not in EXPERIENCE_SUBTYPES:
            raise ValueError("invalid_experience_subtype: use method, boundary or review")
        if subtype == "review" and "audits" not in attrs:
            raise ValueError("review_requires_audits")
    elif "subtype" in attrs:
        raise ValueError("subtype_only_for_experience")
    if "audits" in attrs:
        validate_audits(attrs["audits"], entry.links, records, marker=marker)


def prepare_attributes(kind: str, previous: KnowledgeEntry | None, incoming: dict, *,
                       content_changed: bool, links, records: Mapping, marker: str | None = None) -> dict:
    if RESERVED_ATTRIBUTES.intersection(incoming):
        raise ValueError("reserved_entry_attributes")
    if "subtype" in incoming and kind != "experience":
        raise ValueError("subtype_only_for_experience")
    if "audits" in incoming and kind not in ("experience", "report"):
        raise ValueError("audits_only_for_research_entries")
    attrs = json_object(previous.attributes) if previous else {}
    attrs.update(incoming)
    normalize = kind == "report" or "research_policy_version" in attrs or (
        kind == "experience" and (previous is None or content_changed or bool(incoming))
    )
    if normalize:
        if (kind == "experience" and previous and "research_policy_version" not in previous.attributes
                and "subtype" not in incoming):
            raise ValueError("legacy_experience_requires_explicit_subtype")
        if previous and "research_policy_version" not in previous.attributes:
            if "record_request" in previous.attributes:
                raise ValueError("legacy_reserved_attribute_conflict: create a new linked entry")
            if "audits" in previous.attributes and "audits" not in incoming:
                raise ValueError("legacy_audits_require_explicit_baseline")
        attrs["research_policy_version"] = RESEARCH_POLICY_VERSION
        old_audits = (previous.attributes.get("audits")
                      if previous and "research_policy_version" in previous.attributes else None)
        if old_audits is not None and attrs.get("audits") != json_object({"audits": old_audits})["audits"]:
            raise ValueError("audit_baseline_is_immutable: create a new linked review")
        if "audits" in attrs:
            validate_audits(attrs["audits"], links, records, submitting=old_audits is None, marker=marker)
        probe = KnowledgeEntry("", kind, "", "", "", links=tuple(links), attributes=attrs)
        validate_entry_policy(probe, records, marker=marker)
    return attrs


def annotations(entry: KnowledgeEntry, records: Mapping) -> dict:
    """Describe stale baselines and historical records without mutating the notebook."""
    normalized = entry.attributes.get("research_policy_version") == RESEARCH_POLICY_VERSION
    result = {"legacy_experience": entry.kind == "experience" and not normalized,
              "warnings": [], "audits": []}
    if not normalized:
        return result
    for audit in entry.attributes.get("audits", ()):
        target_id = audit["target_id"]
        target = records[target_id]
        changed = target.revision != audit["revision"]
        revoked = isinstance(target, FactRecord) and target.status == "revoked"
        name = f"{target.display_id} · {target.title}" if target.display_id else target.title
        result["audits"].append(dict(audit, current_revision=target.revision,
                                     display_name=name, status=target.status,
                                     version_changed=changed, revoked=revoked))
        if changed:
            result["warnings"].append({"code": "audit_target_updated", "target_id": target_id,
                                       "audited_revision": audit["revision"], "current_revision": target.revision})
        if revoked:
            result["warnings"].append({"code": "audit_target_revoked", "target_id": target_id})
        elif isinstance(target, FactRecord) and target.supersession:
            result["warnings"].append({"code": "audit_target_superseded", "target_id": target_id})
    return result
