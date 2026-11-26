"""Read-only, semantic input fingerprints for the host-side worker wait loop."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hyprover.core import KnowledgeStore
from hyprover.core.knowledge_markdown import attachment_body
from hyprover.core.schema import KnowledgeEntry, plain

POLL_SECONDS = 3.0
_ATTACHMENT_CACHE_SIZE = 128
_VOLATILE = frozenset({
    "run_id", "full_revalidation_run", "coverage_verification_run", "report_path",
    "created_at", "updated_at", "revoked_at", "superseded_at", "checked_at",
    "started_at", "finished_at", "retry_after", "revision", "display_id",
    "request_id", "record_request", "research_policy_version", "feedback_id",
    "feedback_error", "deduplicated", "requested_by", "expected_revision",
    "expected_revisions", "raw_verification",
})
_FEEDBACK_FIELDS = frozenset({
    "operation", "action", "fact_id", "failed_fact_id", "replacement_id",
    "evidence_digest", "evidence_fact_ids", "source_id", "accepted", "published",
    "applied", "revoked", "verdict", "stage", "outcome", "error", "errors",
    "critical_errors", "gaps", "verification_report", "repair_hints", "reason",
    "fact_reviews", "failure", "review", "coverage", "findings", "summary",
    "fact_review_warning",
})


def _text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _semantic(value, noise: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        return {key: _semantic(item, noise) for key, item in value.items() if key not in _VOLATILE}
    if isinstance(value, (list, tuple)):
        return [_semantic(item, noise) for item in value]
    if isinstance(value, str):
        value = _text(value)
        for token in noise:
            value = value.replace(token, "<run>")
    return value


def _noise(value) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in ("run_id", "full_revalidation_run", "coverage_verification_run", "report_path"):
                if isinstance(item, str) and item:
                    result.add(item)
            else:
                result.update(_noise(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.update(_noise(item))
    return result


def _signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class WakeupReadError(Exception):
    """Inputs are unavailable or changed during collection; never use an empty baseline."""


@dataclass(frozen=True)
class InputFingerprint:
    parts: tuple[tuple[str, str], ...]

    @property
    def digest(self) -> str:
        return _digest(self.parts)

    def changed_from(self, previous: InputFingerprint) -> tuple[str, ...]:
        old = dict(previous.parts)
        return tuple(kind for kind, digest in self.parts if old.get(kind) != digest)


class InputWatcher:
    def __init__(self, project_dir: Path, task_path: Path):
        self.store = KnowledgeStore(project_dir)
        self.task_path = task_path
        self._snapshot = None
        self._parts: tuple[tuple[str, str], ...] = ()
        self._entries: dict[str, tuple[KnowledgeEntry, str]] = {}
        self._attachments: OrderedDict[str, tuple[tuple[int, ...], str]] = OrderedDict()

    def _attachment(self, relative: str) -> str:
        if not relative:
            return ""
        path = self.store._safe_path(relative)
        before = _signature(path)
        cached = self._attachments.get(relative)
        if cached is not None and cached[0] == before:
            self._attachments.move_to_end(relative)
            return cached[1]
        body = attachment_body(path.read_text(encoding="utf-8"))
        if before != _signature(path):
            raise WakeupReadError("attachment_changed_during_read")
        self._attachments[relative] = before, body
        self._attachments.move_to_end(relative)
        while len(self._attachments) > _ATTACHMENT_CACHE_SIZE:
            self._attachments.popitem(last=False)
        return body

    def _feedback(self, entry: KnowledgeEntry, details: str) -> dict:
        attrs = plain(entry.attributes)
        operation = attrs.get("operation", "publication" if "published" in attrs else "feedback")
        payload = {}
        if details and (operation != "feedback" or details.lstrip().startswith("{")):
            payload = json.loads(details)
            if not isinstance(payload, dict):
                raise ValueError("feedback_attachment_must_be_object")
        noise = tuple(sorted(_noise(attrs) | _noise(payload), key=len, reverse=True))
        selected = {k: v for k, v in payload.items() if k in _FEEDBACK_FIELDS}
        selected.update({k: v for k, v in attrs.items() if k in _FEEDBACK_FIELDS})
        raw = payload.get("raw_verification")
        if isinstance(raw, dict):
            diagnostics = {k: v for k, v in raw.items() if k in _FEEDBACK_FIELDS and selected.get(k) != v}
            if diagnostics:
                selected["verification_diagnostics"] = diagnostics
        targets = {k: selected[k] for k in ("fact_id", "failed_fact_id", "source_id", "evidence_digest")
                   if selected.get(k)}
        result = {"operation": operation, "links": sorted(entry.links),
                  "content": _semantic(selected, noise)}
        # A rejected publication may have neither a candidate ID nor a source link.
        # Do not conflate unrelated failed candidates using their fixed summary.
        if not targets and not entry.links:
            result["unidentified_entry"] = entry.entry_id
        if operation != "completion_revalidation":
            result["body"] = _semantic(entry.body, noise)
        if not payload:
            result["details"] = _text(details)
        return result

    def _entry_digest(self, entry: KnowledgeEntry) -> str:
        cached = self._entries.get(entry.entry_id)
        if cached is not None and cached[0] == entry:
            return cached[1]
        details = self._attachment(entry.details_path)
        if entry.kind == "feedback":
            content = self._feedback(entry, details)
        else:
            content = {"entry_id": entry.entry_id, "title": entry.title, "body": entry.body,
                       "status": entry.status, "links": sorted(entry.links),
                       "attributes": _semantic(entry.attributes), "details": details}
        digest = _digest(_semantic(content))
        self._entries[entry.entry_id] = entry, digest
        return digest

    def _knowledge_parts(self, snapshot):
        if snapshot is self._snapshot:
            return self._parts, self._entries
        groups: dict[str, set[str]] = {kind: set() for kind in ("overview", "routes", "facts", "feedback")}
        entries = {}
        for entry in snapshot.entries.values():
            if entry.kind == "route":
                if entry.status == "proposed" and entry.author not in ("main", "main_agent"):
                    continue
                category = "routes"
            elif entry.kind in ("overview", "feedback"):
                category = entry.kind
            else:
                continue
            digest = self._entry_digest(entry)
            entries[entry.entry_id] = entry, digest
            groups[category].add(digest)
        for fact in snapshot.facts.values():
            groups["facts"].add(_digest(_semantic({
                "fact_id": fact.fact_id, "statement": fact.statement, "status": fact.status,
                "predecessors": sorted(fact.predecessors), "definitions": fact.glossary_introduces,
                "external_refs": fact.external_refs, "revocation": fact.revocation,
                "supersession": fact.supersession,
            })))
        return tuple((kind, _digest(sorted(values))) for kind, values in groups.items()), entries

    def read(self) -> InputFingerprint:
        try:
            paths = (self.task_path, self.store._safe_path("KNOWLEDGE.md"))
            before = tuple(_signature(path) for path in paths)
            task = _text(self.task_path.read_text(encoding="utf-8"))
            snapshot = self.store.snapshot()
            parts, entries = self._knowledge_parts(snapshot)
            if before != tuple(_signature(path) for path in paths):
                raise WakeupReadError("inputs_changed_during_read")
            self._snapshot, self._parts, self._entries = snapshot, parts, entries
            return InputFingerprint((("task", _digest(task)), *parts))
        except (OSError, ValueError) as exc:
            raise WakeupReadError(type(exc).__name__) from exc
