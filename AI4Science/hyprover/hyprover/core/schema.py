"""Knowledge records, immutable read snapshots, and content-addressed fact IDs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping

KNOWLEDGE_FORMAT = "hyprover-knowledge-v1"
# The previous name used the same v1 layout. Keep its framing for existing data.
KNOWLEDGE_MARKERS = {
    KNOWLEDGE_FORMAT: "<!-- hyprover:",
    "hyra-knowledge-v1": "<!-- hyra:",
}
ENTRY_STATUSES = {
    "overview": ("current",),
    "route": ("proposed", "planned", "active", "parked", "failed", "completed"),
    "conjecture": ("open", "proved", "refuted", "reopened"),
    "experience": ("open", "archived"),
    "lead": ("open", "archived"),
    "feedback": ("recorded",),
    "report": ("recorded", "archived"),
}
EXPERIENCE_SUBTYPES = ("method", "boundary", "review")
RESEARCH_POLICY_VERSION = 1


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def plain(value: Any) -> Any:
    """JSON-safe copy, including immutable records and their nested metadata."""
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


@dataclass(frozen=True)
class FactRecord:
    """A complete usable statement, deliberately without proof text."""

    fact_id: str
    problem_id: str
    author: str
    statement: str
    predecessors: tuple[str, ...] = ()
    glossary_introduces: Mapping[str, str] = field(default_factory=dict)
    external_refs: tuple[Mapping[str, Any], ...] = ()
    proof_path: str = ""
    intuition_path: str = ""
    verification: Mapping[str, Any] = field(default_factory=dict)
    status: str = "verified"
    revision: int = 1
    created_at: str = ""
    revocation: Mapping[str, Any] = field(default_factory=dict)
    display_id: str = ""
    title: str = ""
    supersession: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("predecessors", "glossary_introduces", "external_refs", "verification", "revocation", "supersession"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def to_dict(self) -> dict:
        return plain(self)


@dataclass(frozen=True)
class KnowledgeEntry:
    """Non-factual shared research material; links never confer truth."""

    entry_id: str
    kind: str
    title: str
    body: str
    author: str
    status: str = "open"
    revision: int = 1
    links: tuple[str, ...] = ()
    details_path: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    display_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "links", freeze(self.links))
        object.__setattr__(self, "attributes", freeze(self.attributes))

    def to_dict(self) -> dict:
        return plain(self)


@dataclass(frozen=True)
class KnowledgeSnapshot:
    problem_id: str
    revision: int = 0
    facts: Mapping[str, FactRecord] = field(default_factory=dict)
    entries: Mapping[str, KnowledgeEntry] = field(default_factory=dict)
    display_version: int = 0
    storage_format: str = KNOWLEDGE_FORMAT

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", freeze(self.facts))
        object.__setattr__(self, "entries", freeze(self.entries))

    def active_facts(self) -> dict[str, FactRecord]:
        return {fid: f for fid, f in self.facts.items() if f.status == "verified"}

    def preferred_fact_id(self, fact_id: str) -> str:
        """Follow usable replacement recommendations without changing proof premises."""
        current, seen = fact_id, set()
        while current not in seen:
            seen.add(current)
            record = self.facts[current]
            successor = record.supersession.get("replacement_id")
            evidence = record.supersession.get("review", {}).get("evidence_fact_ids", ())
            if (not successor or successor not in self.facts or self.facts[successor].status != "verified"
                    or any(i not in self.facts or self.facts[i].status != "verified" for i in evidence)):
                return current
            current = successor
        raise ValueError("supersession_cycle")

    def topological(self, targets=None) -> list[str]:
        """Dependency-first closure. Unknown or revoked targets fail closed."""
        active = self.active_facts()
        result: list[str] = []
        state: dict[str, int] = {}
        for root in sorted(active) if targets is None else targets:
            stack = [(root, False)]
            while stack:
                fid, finished = stack.pop()
                if fid not in active:
                    raise ValueError(f"unknown_or_revoked_fact: {fid}")
                if finished:
                    state[fid] = 2
                    result.append(fid)
                    continue
                if state.get(fid) == 2:
                    continue
                if state.get(fid) == 1:
                    raise ValueError(f"dependency_cycle: {fid}")
                state[fid] = 1
                stack.append((fid, True))
                stack.extend((pid, False) for pid in reversed(active[fid].predecessors))
        return result


# --------------------------------------------------------------------------- #
# full fact content and advisory bibliography                                 #
# --------------------------------------------------------------------------- #

# Canonical key order for references cited by a proof. Extra fields are preserved
# after these keys; JSON validity is checked, bibliographic accuracy is not.
EXTERNAL_REF_KEYS = ("key", "authors", "title", "arxiv", "year", "venue", "doi", "cited_for")


def validate_json_value(value: object, *, depth: int = 0) -> None:
    """Require lossless UTF-8 JSON: never coerce object keys or nonfinite numbers."""
    if depth > 64:
        raise ValueError("json_metadata_too_deep")
    if value is None or type(value) in (bool, int):
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("invalid_json_unicode") from exc
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("json_metadata_keys_must_be_strings")
            validate_json_value(key, depth=depth + 1)
            validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            validate_json_value(item, depth=depth + 1)
        return
    raise ValueError("metadata_must_be_finite_json")


def clean_external_refs(refs: object) -> List[Dict[str, object]]:
    """Order advisory reference fields, rejecting metadata that cannot round-trip.

    Non-object items remain ignored; malformed nested objects are never silently
    changed or discarded, since that could lose citation information.
    """
    if refs is None:
        return []
    if not isinstance(refs, (list, tuple)):
        raise ValueError("external_refs_must_be_list")
    out: List[Dict[str, object]] = []
    for r in refs:
        if not isinstance(r, Mapping):
            continue
        validate_json_value(r)
        copied = plain(r)
        ordered = {k: copied[k] for k in EXTERNAL_REF_KEYS if k in copied}
        for k in sorted(copied):
            if k not in ordered:
                ordered[k] = copied[k]
        out.append(ordered)
    return out


@dataclass
class Fact:
    """A full candidate or explicitly loaded proof, not the default read view.

    Constructing this object does not certify it; only independent verification
    followed by successful publication makes it an active FactRecord.
    """

    fact_id: str
    problem_id: str
    author: str
    predecessors: List[str]                    # bare-hex fact ids (the DAG)
    statement: str
    proof: str
    glossary_introduces: Dict[str, str] = field(default_factory=dict)  # symbol -> definition
    intuition: str = ""
    # Mutable bibliography is excluded from fact_id so metadata corrections do
    # not invalidate dependency links. Citation keys within `proof` remain hashed.
    external_refs: List[Dict[str, object]] = field(default_factory=list)
    title: str = ""  # Display only; never a mathematical premise or hash input.


def _normalize(text: str) -> str:
    """Collapse whitespace runs for hashing; preserve all other characters.

    This is not Unicode normalization or a mathematical equivalence check.
    """
    return re.sub(r"\s+", " ", text or "").strip()


def _fact_content_bytes(
    problem_id: str, predecessors: List[str], glossary_introduces: Dict[str, str],
    statement: str, proof: str,
) -> bytes:
    """Encode the persisted fact-ID format, retaining its historical JSON spacing.

    Dependencies are sorted, not deduplicated. Definitions are string-converted
    and sorted; only statement/proof whitespace is collapsed. UTF-8, unescaped
    non-ASCII characters and default JSON separators are compatibility requirements.
    """
    body = {
        "problem_id": problem_id,
        "predecessors": sorted(predecessors),
        "glossary_introduces": dict(
            sorted((str(k), str(v)) for k, v in glossary_introduces.items())
        ),
        "statement": _normalize(statement),
        "proof": _normalize(proof),
    }
    return json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")


def compute_fact_id(
    *,
    problem_id: str,
    predecessors: List[str],
    glossary_introduces: Dict[str, str],
    statement: str,
    proof: str,
) -> str:
    """Return the first 16 SHA-256 hex characters of the canonical fact payload.

    Titles and external bibliography are not inputs; citations written into the
    proof remain hashed. The ID identifies content, not its correctness. Publication
    separately compares original content and checks independent verification.
    """
    content = _fact_content_bytes(problem_id, predecessors, glossary_introduces, statement, proof)
    return hashlib.sha256(content).hexdigest()[:16]
