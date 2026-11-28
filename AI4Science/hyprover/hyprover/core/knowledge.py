"""Single shared notebook; proof reads are explicit and writes serialized.

Only the verification publication service invokes publish. This is an application
contract, not an OS-level sandbox against processes sharing the same user.
"""
from __future__ import annotations
import fcntl
import hashlib
import heapq
import json
import os
import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from . import entry_policy, knowledge_markdown as md
from ._util import atomic_write_text, sync_existing_file, utc_now
from .schema import (ENTRY_STATUSES, EXPERIENCE_SUBTYPES, KNOWLEDGE_FORMAT, KNOWLEDGE_MARKERS, RESEARCH_POLICY_VERSION,
                     Fact, FactRecord, KnowledgeEntry, KnowledgeSnapshot, clean_external_refs, compute_fact_id, plain)

_CACHE: OrderedDict = OrderedDict()
_CACHE_LOCK = threading.RLock()


def _read_exact(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.read()


class KnowledgeStore:
    def __init__(self, root: Path):
        raw = Path(root).absolute()
        if raw.is_symlink():
            raise ValueError("symlink_project_root")
        self.root = raw.resolve()
        self.path = self.root / "KNOWLEDGE.md"

    def _safe_path(self, relative: str) -> Path:
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("unsafe_knowledge_path")
        path = self.root
        for component in rel.parts:
            path = path / component
            if path.is_symlink():
                raise ValueError("symlink_knowledge_path")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("unsafe_knowledge_path")
        return path

    @classmethod
    def initialize(cls, root: Path, problem_id: str | None = None) -> KnowledgeStore:
        store = cls(root)
        if any((store.root / old).exists() for old in ("fact_graph", "global_memory")):
            raise ValueError("unsupported_knowledge_format: legacy project left unchanged")
        config = store._safe_path("project.json")
        storage_format = KNOWLEDGE_FORMAT
        if config.exists():
            metadata = json.loads(config.read_text(encoding="utf-8"))
            storage_format = metadata.get("knowledge_format") if isinstance(metadata, dict) else None
            if not isinstance(storage_format, str) or storage_format not in KNOWLEDGE_MARKERS:
                raise ValueError("unsupported_knowledge_format: existing project metadata left unchanged")
        if store.path.exists() or store.path.is_symlink():
            raise FileExistsError("KNOWLEDGE.md already exists")
        snapshot = KnowledgeSnapshot(problem_id=problem_id or store.root.name, display_version=1,
                                     storage_format=storage_format)
        md.validate(snapshot)
        store.root.mkdir(parents=True, exist_ok=True)
        for rel in ("knowledge/proofs", "knowledge/notes"):
            store._safe_path(rel).mkdir(parents=True, exist_ok=True)
        fd = os.open(store._safe_path("knowledge/.write.lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if store.path.exists():
                raise FileExistsError("KNOWLEDGE.md already exists")
            atomic_write_text(store._safe_path("KNOWLEDGE.md"), md.serialize(snapshot))
        finally:
            os.close(fd)
        return store

    def snapshot(self) -> KnowledgeSnapshot:
        path = self._safe_path("KNOWLEDGE.md")
        if not path.is_file():
            raise ValueError("unsupported_knowledge_format: KNOWLEDGE.md is required; legacy data is not read or migrated")
        config = self._safe_path("project.json")
        storage_format = None
        if config.exists():
            try:
                metadata = json.loads(config.read_text(encoding="utf-8"))
                storage_format = metadata.get("knowledge_format")
                if not isinstance(storage_format, str) or storage_format not in KNOWLEDGE_MARKERS:
                    raise ValueError("unsupported_knowledge_format: project.json")
            except (json.JSONDecodeError, AttributeError) as exc:
                raise ValueError("invalid_project_metadata") from exc
        with path.open(encoding="utf-8", newline="") as handle:
            stat = os.fstat(handle.fileno())
            key = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, storage_format)
            with _CACHE_LOCK:
                if key in _CACHE:
                    _CACHE.move_to_end(key)
                    return _CACHE[key]
            snapshot = md.parse(handle.read())
        if storage_format is not None and snapshot.storage_format != storage_format:
            raise ValueError("unsupported_knowledge_format: project.json and KNOWLEDGE.md disagree")
        with _CACHE_LOCK:
            _CACHE[key] = snapshot
            while len(_CACHE) > 16:
                _CACHE.popitem(last=False)
        return snapshot

    @contextmanager
    def _locked(self):
        self.snapshot()
        fd = os.open(self._safe_path("knowledge/.write.lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield self.snapshot()
        finally:
            os.close(fd)

    def _commit(self, previous: KnowledgeSnapshot, *, facts=None, entries=None) -> KnowledgeSnapshot:
        snapshot = md.prepare_display(replace(
            previous, revision=previous.revision + 1,
            facts=previous.facts if facts is None else facts,
            entries=previous.entries if entries is None else entries))
        md.validate(snapshot)
        atomic_write_text(self._safe_path("KNOWLEDGE.md"), md.serialize(snapshot))
        return snapshot

    def refresh_display(self, titles: dict[str, str] | None = None, *,
                        expected_revision: int | None = None) -> KnowledgeSnapshot:
        """Explicitly refresh display metadata without editing mathematical content."""
        with self._locked() as snapshot:
            if expected_revision is not None and snapshot.revision != expected_revision:
                raise ValueError("document_revision_conflict")
            facts = dict(snapshot.facts)
            for fid, title in (titles or {}).items():
                if fid not in facts:
                    raise ValueError(f"unknown_fact: {fid}")
                md.check_title(title, marker=KNOWLEDGE_MARKERS[snapshot.storage_format])
                facts[fid] = replace(facts[fid], title=title)
            proposed = md.prepare_display(replace(snapshot, facts=facts))
            if proposed == snapshot:
                return snapshot
            return self._commit(snapshot, facts=proposed.facts, entries=proposed.entries)

    def _attachment(self, relative: str, identifier: str, body: str, title: str, *, storage_format: str) -> None:
        path = self._safe_path(relative)
        content = md.attachment(identifier, body, title, storage_format=storage_format)
        if path.exists():
            if _read_exact(path) != content:
                raise ValueError("immutable_attachment_conflict")
            sync_existing_file(path)
            return
        atomic_write_text(path, content)

    def _sync_retry(self, attachments=()) -> None:
        """A visible commit may come from an interrupted directory fsync."""
        for relative in attachments:
            if relative:
                sync_existing_file(self._safe_path(relative))
        sync_existing_file(self._safe_path("KNOWLEDGE.md"))

    def _entry_attachment_path(self, entry_id: str, revision: int, details: str, *,
                               storage_format: str = KNOWLEDGE_FORMAT) -> str:
        """Keep revisions sequential without overwriting a failed attempt's note."""
        relative = f"knowledge/notes/{entry_id}-{revision}.md"
        path = self._safe_path(relative)
        if path.exists():
            content = md.attachment(entry_id, details, "Research note — NOT A FACT", storage_format=storage_format)
            if _read_exact(path) != content:
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
                relative = f"knowledge/notes/{entry_id}-{revision}-{digest}.md"
                self._safe_path(relative)
        return relative

    def list(self) -> list[str]:
        return sorted(self.snapshot().active_facts())

    def get(self, fact_id: str, *, include_revoked: bool = False) -> FactRecord | None:
        fact = self.snapshot().facts.get(fact_id)
        return fact if fact and (include_revoked or fact.status == "verified") else None

    def exists(self, fact_id: str) -> bool:
        return self.get(fact_id) is not None

    def proof(self, fact_id: str, *, snapshot: KnowledgeSnapshot | None = None) -> str:
        snap = snapshot or self.snapshot()
        fact = snap.facts.get(fact_id)
        if not fact or fact.status != "verified":
            raise ValueError(f"unknown_or_revoked_fact: {fact_id}")
        proof = md.attachment_body(_read_exact(self._safe_path(fact.proof_path)))
        digest = compute_fact_id(problem_id=fact.problem_id, predecessors=list(fact.predecessors),
                                 glossary_introduces=dict(fact.glossary_introduces), statement=fact.statement, proof=proof)
        if digest != fact_id:
            raise ValueError("proof_content_address_mismatch")
        return proof

    def read_fact(self, fact_id: str, *, snapshot: KnowledgeSnapshot | None = None) -> Fact:
        snap = snapshot or self.snapshot()
        f = snap.facts.get(fact_id)
        if not f or f.status != "verified":
            raise ValueError(f"unknown_or_revoked_fact: {fact_id}")
        proof = self.proof(fact_id, snapshot=snap)
        intuition = md.attachment_body(_read_exact(self._safe_path(f.intuition_path))) if f.intuition_path else ""
        return Fact(f.fact_id, f.problem_id, f.author, list(f.predecessors), f.statement, proof,
                    dict(f.glossary_introduces), intuition, plain(f.external_refs), title=f.title)

    def predecessors(self, fact_id: str) -> list[str]:
        fact = self.get(fact_id)
        if not fact:
            raise ValueError(f"unknown_or_revoked_fact: {fact_id}")
        return list(fact.predecessors)

    def external_refs(self, fact_id: str) -> list[dict]:
        fact = self.get(fact_id)
        if not fact:
            raise ValueError(f"unknown_or_revoked_fact: {fact_id}")
        return plain(fact.external_refs)

    def set_external_refs(self, fact_id: str, external_refs: list[dict]) -> list[dict]:
        refs = clean_external_refs(external_refs)
        with self._locked() as snap:
            fact = snap.facts.get(fact_id)
            if not fact or fact.status != "verified":
                raise ValueError(f"unknown_or_revoked_fact: {fact_id}")
            facts = dict(snap.facts)
            facts[fact_id] = replace(fact, external_refs=refs, revision=fact.revision + 1)
            self._commit(snap, facts=facts)
        return refs

    def glossary(self) -> dict[str, str]:
        return {k: v for f in self.snapshot().active_facts().values() for k, v in f.glossary_introduces.items()}

    @staticmethod
    def _descendants(snap: KnowledgeSnapshot, fact_id: str) -> list[str]:
        children: dict[str, list[str]] = {}
        for fid, fact in snap.active_facts().items():
            for pid in fact.predecessors:
                children.setdefault(pid, []).append(fid)
        seen = set()
        frontier = list(children.get(fact_id, []))
        while frontier:
            fid = frontier.pop()
            if fid not in seen:
                seen.add(fid)
                frontier.extend(children.get(fid, []))
        return sorted(seen)

    def descendants(self, fact_id: str) -> list[str]:
        return self._descendants(self.snapshot(), fact_id)

    @staticmethod
    def _entry_row(entry: KnowledgeEntry, snap: KnowledgeSnapshot, records: dict) -> dict:
        row = entry.to_dict()
        row.update(id=entry.entry_id, is_fact=False, link=f"KNOWLEDGE.md#{entry.entry_id}",
                   display_name=md.display_name(entry),
                   related_labels={i: md.display_name(records[i]) for i in entry.links},
                   invalidated_links=[i for i in entry.links if i in snap.facts and snap.facts[i].status != "verified"])
        row.update(entry_policy.annotations(entry, records))
        return row

    def entry_detail(self, entry_id: str) -> dict:
        """Explicit attachment read; a missing note is not an empty successful read."""
        snap = md.prepare_display(self.snapshot())
        entry = snap.entries.get(entry_id)
        if entry is None:
            raise KeyError(entry_id)
        row = self._entry_row(entry, snap, {**snap.facts, **snap.entries})
        row["details"] = ""
        if entry.details_path:
            path = self._safe_path(entry.details_path)
            try:
                row["details"] = md.attachment_body(_read_exact(path))
            except FileNotFoundError:
                row["details_error"] = "missing_attachment"
        return row

    def search(self, query: str = "", limit: int = 10, kinds=None, *, include_superseded: bool = False,
               subtype: str | None = None, author: str | None = None, linked_to: str | None = None,
               include_archived: bool = False, sort_by: str = "updated", offset: int = 0,
               expected_snapshot_revision: int | None = None) -> list[dict]:
        """Compatibility list API; filtering and pagination share one snapshot."""
        return self.search_page(query, limit, kinds, include_superseded=include_superseded,
            subtype=subtype, author=author, linked_to=linked_to, include_archived=include_archived,
            sort_by=sort_by, offset=offset, expected_snapshot_revision=expected_snapshot_revision)["results"]

    def search_page(self, query: str = "", limit: int = 10, kinds=None, *, include_superseded: bool = False,
                    subtype: str | None = None, author: str | None = None, linked_to: str | None = None,
                    include_archived: bool = False, sort_by: str = "updated", offset: int = 0,
                    expected_snapshot_revision: int | None = None,
                    snapshot: KnowledgeSnapshot | None = None) -> dict:
        """Browse summaries; optional literal/ID lookup never assigns relevance scores."""
        if not isinstance(query, str):
            raise ValueError("invalid_search_query")
        if type(limit) is not int or limit < 1 or limit > 1000:
            raise ValueError("invalid_search_limit")
        if type(offset) is not int or offset < 0:
            raise ValueError("invalid_search_offset")
        if kinds is not None and (not isinstance(kinds, (list, tuple)) or any(not isinstance(k, str) for k in kinds)):
            raise ValueError("invalid_search_kind")
        selected = set(kinds or ["fact", *(k for k in ENTRY_STATUSES if k != "report")])
        if selected - {"fact", *ENTRY_STATUSES}:
            raise ValueError("invalid_search_kind")
        if subtype is not None:
            if not isinstance(subtype, str) or subtype not in EXPERIENCE_SUBTYPES:
                raise ValueError("invalid_experience_subtype")
            if kinds and selected != {"experience"}:
                raise ValueError("subtype_filter_requires_experience")
            selected = {"experience"}
        if type(include_superseded) is not bool or type(include_archived) is not bool:
            raise ValueError("invalid_search_boolean")
        if sort_by == "relevance":
            sort_by = "updated"
        if sort_by not in ("updated", "created"):
            raise ValueError("invalid_search_sort")
        for value in (author, linked_to):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("invalid_search_filter")
        if expected_snapshot_revision is not None and (type(expected_snapshot_revision) is not int or expected_snapshot_revision < 0):
            raise ValueError("invalid_search_revision")
        snap = md.prepare_display(snapshot if snapshot is not None else self.snapshot())
        if expected_snapshot_revision is not None and expected_snapshot_revision != snap.revision:
            raise ValueError("search_snapshot_conflict")
        records = {**snap.facts, **snap.entries}
        if linked_to is not None and linked_to not in records:
            linked_to = next((identifier for identifier, record in records.items()
                              if record.display_id.lower() == linked_to.lower()), None)
            if linked_to is None:
                raise ValueError("unknown_link_filter")
        needle = query.strip().casefold()
        exact_id = next((identifier for identifier, record in records.items()
                         if needle and needle in (identifier.casefold(), record.display_id.casefold())), None)

        def matches(record, text: str) -> bool:
            identifier = record.fact_id if isinstance(record, FactRecord) else record.entry_id
            if exact_id is not None:
                return identifier == exact_id
            return not needle or needle in text.casefold()

        results = []
        if "fact" in selected:
            for fact in snap.active_facts().values():
                if (author is not None and author != fact.author) or (linked_to is not None and linked_to not in fact.predecessors):
                    continue
                preferred = snap.preferred_fact_id(fact.fact_id)
                if preferred != fact.fact_id and not include_superseded and exact_id != fact.fact_id:
                    continue
                text = " ".join((fact.fact_id, fact.display_id, fact.title, fact.statement,
                                 json.dumps(dict(fact.glossary_introduces), ensure_ascii=False)))
                if not matches(fact, text):
                    continue
                row = fact.to_dict()
                row.update(kind="fact", id=fact.fact_id, is_fact=True, link=f"KNOWLEDGE.md#{md.anchor(fact.fact_id)}",
                           preferred_fact_id=preferred, superseded=preferred != fact.fact_id,
                           display_name=md.display_name(fact),
                           predecessor_labels={pid: md.display_name(snap.facts[pid]) for pid in fact.predecessors})
                results.append(row)
        for entry in snap.entries.values():
            if entry.kind not in selected or (entry.status == "archived" and not include_archived):
                continue
            if (author is not None and author != entry.author) or (linked_to is not None and linked_to not in entry.links):
                continue
            if subtype is not None and (entry.attributes.get("research_policy_version") != 1 or entry.attributes.get("subtype") != subtype):
                continue
            if matches(entry, " ".join((entry.entry_id, entry.display_id, entry.title, entry.body))):
                results.append(self._entry_row(entry, snap, records))
        if sort_by == "updated":
            key = lambda row: (row.get("updated_at", row.get("created_at", "")), row["id"])
        else:
            key = lambda row: (row.get("created_at", ""), row["id"])
        page = heapq.nlargest(offset + limit, results, key=key)[offset:]
        return {"results": page, "total": len(results), "sort_by": sort_by,
                "offset": offset, "limit": limit, "has_more": offset + len(page) < len(results),
                "snapshot_revision": snap.revision}

    def record(self, kind: str, title: str, body: str, author: str, role: str, *,
               entry_id: str | None = None, expected_revision: int | None = None,
               status: str | None = None, links=None, details: str = "", attributes=None,
               request_id: str | None = None) -> KnowledgeEntry:
        allowed = {"main": {"overview", "route", "conjecture", "experience", "lead", "report"},
                   "worker": {"route", "conjecture", "experience", "lead", "report"},
                   "verifier": {"feedback"}}
        if not isinstance(kind, str) or not isinstance(role, str) or kind not in allowed.get(role, set()):
            raise ValueError("role_cannot_record_kind")
        if entry_id is not None and (not isinstance(entry_id, str) or not md.ENTRY_ID.fullmatch(entry_id)):
            raise ValueError("invalid_entry_identity")
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 1):
            raise ValueError("invalid_expected_revision")
        incoming = entry_policy.json_object({} if attributes is None else attributes)
        if entry_policy.RESERVED_ATTRIBUTES.intersection(incoming):
            raise ValueError("reserved_entry_attributes")
        if links is not None and (not isinstance(links, (list, tuple)) or any(not isinstance(i, str) for i in links)):
            raise ValueError("invalid_entry_links")
        if request_id is not None:
            entry_policy.check_request_id(request_id)
            if entry_id is not None or expected_revision is not None:
                raise ValueError("request_id_only_for_creation")
        with self._locked() as snap:
            marker = KNOWLEDGE_MARKERS[snap.storage_format]
            for text in (title, body, author, details):
                md.check_text(text, marker=marker)
            if (not title.strip() or not body.strip() or not author.strip()
                    or any(c in title for c in "\r\n")):
                raise ValueError("invalid_entry_content")
            previous = snap.entries.get(entry_id) if entry_id else None
            if entry_id and previous is None:
                raise ValueError("unknown_entry")
            if previous:
                if expected_revision != previous.revision:
                    raise ValueError("entry_revision_conflict")
                if previous.kind != kind or (role == "worker" and previous.author != author):
                    raise ValueError("entry_ownership_conflict")
                if previous.kind == "conjecture" and previous.status in ("proved", "refuted"):
                    raise ValueError("resolved_conjecture_is_immutable")
            elif expected_revision is not None:
                raise ValueError("unexpected_entry_revision")
            state = status if status is not None else previous.status if previous else ENTRY_STATUSES[kind][0]
            if not isinstance(state, str) or state not in ENTRY_STATUSES[kind] or (kind == "conjecture" and state not in ("open", "reopened")):
                raise ValueError("illegal_state_promotion")
            if role == "worker" and kind == "route" and (state != "proposed" or (previous and previous.status != "proposed")):
                raise ValueError("only_main_controls_routes")
            refs = tuple(links if links is not None else previous.links if previous else ())
            if len(set(refs)) != len(refs):
                raise ValueError("duplicate_entry_links")
            fingerprint = None
            if request_id is not None:
                payload = {"kind": kind, "title": title, "body": body, "status": state,
                           "links": refs, "details": details, "attributes": incoming}
                fingerprint = hashlib.sha256(md._json(payload).encode("utf-8")).hexdigest()
                for existing in snap.entries.values():
                    if "research_policy_version" not in existing.attributes:
                        continue
                    saved = existing.attributes.get("record_request", {})
                    if existing.author == author and existing.kind == kind and saved.get("id") == request_id:
                        if saved.get("fingerprint") != fingerprint:
                            raise ValueError("record_request_conflict")
                        self._sync_retry((existing.details_path,))
                        return existing
            records = {**snap.facts, **snap.entries}
            for identifier in refs:
                if identifier not in records:
                    raise ValueError(f"unknown_link: {identifier}")
            changed = previous is None or title != previous.title or body != previous.body or bool(details)
            if previous and kind == "report" and previous.author != author:
                original_attrs = entry_policy.json_object(previous.attributes)
                metadata_changed = any(k not in original_attrs or md._json(original_attrs[k]) != md._json(v)
                                       for k, v in incoming.items())
                if changed or refs != previous.links or metadata_changed or state not in (previous.status, "archived"):
                    raise ValueError("report_author_required")
                if state == previous.status:
                    return previous
            attrs = entry_policy.prepare_attributes(kind, previous, incoming,
                content_changed=changed or bool(previous and refs != previous.links), links=refs, records=records,
                marker=marker)
            if request_id is not None:
                attrs["research_policy_version"] = RESEARCH_POLICY_VERSION
                attrs["record_request"] = {"id": request_id, "fingerprint": fingerprint}
                identity = md._json([snap.problem_id, author, kind, request_id])
                eid = "entry-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                if eid in snap.entries:
                    raise ValueError("record_request_identity_conflict")
            else:
                eid = entry_id or "entry-" + uuid.uuid4().hex[:16]
            if previous is None and eid in records:
                raise ValueError("entry_identity_conflict")
            revision = previous.revision + 1 if previous else 1
            detail_path = previous.details_path if previous else ""
            if details:
                detail_path = self._entry_attachment_path(eid, revision, details, storage_format=snap.storage_format)
            now = utc_now()
            entry = KnowledgeEntry(eid, kind, title, body, previous.author if previous else author, state, revision,
                                   refs, detail_path, attrs, previous.created_at if previous else now, now,
                                   display_id=previous.display_id if previous else "")
            entries = dict(snap.entries)
            entries[eid] = entry
            proposed = md.prepare_display(replace(snap, revision=snap.revision + 1, entries=entries))
            md.validate(proposed)
            md.serialize(proposed).encode("utf-8")
            if details:
                self._attachment(detail_path, eid, details, "Research note — NOT A FACT",
                                 storage_format=snap.storage_format)
            committed = self._commit(snap, entries=entries)
            return committed.entries[eid]

    def publish(self, candidate: Fact, verification: dict, source_id: str | None = None,
                source_revision: int | None = None) -> str:
        """Commit the exact independently checked candidate; service-internal API."""
        candidate = Fact(**plain(candidate))
        verification = plain(verification)
        if (verification.get("verdict") != "correct" or verification.get("errors")
                or verification.get("critical_errors") or verification.get("gaps")):
            raise ValueError("publication_requires_accepted_verification")
        with self._locked() as snap:
            marker = KNOWLEDGE_MARKERS[snap.storage_format]
            for text in (candidate.statement, candidate.proof, candidate.intuition, candidate.author):
                md.check_text(text, marker=marker)
            if not candidate.proof.strip() or not candidate.statement.strip() or not candidate.author.strip():
                raise ValueError("empty_fact_content")
            if candidate.title:
                md.check_title(candidate.title, marker=marker)
            fid = compute_fact_id(problem_id=candidate.problem_id, predecessors=candidate.predecessors,
                                  glossary_introduces=candidate.glossary_introduces, statement=candidate.statement,
                                  proof=candidate.proof)
            if candidate.fact_id and candidate.fact_id != fid:
                raise ValueError("candidate_content_address_mismatch")
            if candidate.problem_id != snap.problem_id:
                raise ValueError("candidate_project_mismatch")
            existing = snap.facts.get(fid)
            if existing:
                if existing.status != "verified":
                    raise ValueError("cannot_republish_revoked_fact")
                old = self.read_fact(fid, snapshot=snap)
                if old.statement != candidate.statement or old.proof != candidate.proof or old.glossary_introduces != candidate.glossary_introduces or sorted(old.predecessors) != sorted(candidate.predecessors):
                    raise ValueError("content_address_collision_or_nonidentical_retry")
            if len(set(candidate.predecessors)) != len(candidate.predecessors):
                raise ValueError("duplicate_predecessors")
            for pid in candidate.predecessors:
                if pid == fid or pid not in snap.facts or snap.facts[pid].status != "verified":
                    raise ValueError(f"unknown_or_revoked_predecessor: {pid}")
            source = snap.entries.get(source_id) if source_id else None
            if source_id:
                if not source or source.kind != "conjecture":
                    raise ValueError("invalid_conjecture_source")
                if existing and source.status == "proved" and source.attributes.get("resolution_fact_id") == fid:
                    self._sync_retry((existing.proof_path, existing.intuition_path))
                    return fid
                if source.revision != source_revision or source.status not in ("open", "reopened"):
                    raise ValueError("source_revision_or_state_conflict")
                if " ".join(source.body.split()) != " ".join(candidate.statement.split()):
                    raise ValueError("source_statement_mismatch")
            elif source_revision is not None:
                raise ValueError("source_revision_without_source")
            if existing and not source:
                self._sync_retry((existing.proof_path, existing.intuition_path))
                return fid
            now = utc_now()
            verification_info = {k: v for k, v in verification.items() if k in ("verdict", "run_id", "verified_at", "verifier", "provenance")}
            verification_info.setdefault("verified_at", now)
            record = FactRecord(fid, candidate.problem_id, candidate.author, candidate.statement,
                                tuple(candidate.predecessors), candidate.glossary_introduces,
                                clean_external_refs(candidate.external_refs), f"knowledge/proofs/{fid}.md",
                                f"knowledge/notes/{fid}-intuition.md" if candidate.intuition else "",
                                verification_info, created_at=now,
                                title=candidate.title or (md.short_title(source.title) if source else md.short_title(candidate.statement)))
            facts = dict(snap.facts)
            facts[fid] = existing or record
            entries = dict(snap.entries)
            if source:
                attrs = dict(source.attributes, resolution_fact_id=fid, resolved_at=now)
                entries[source.entry_id] = replace(source, status="proved", revision=source.revision + 1,
                    links=tuple(dict.fromkeys((*source.links, fid))), attributes=attrs, updated_at=now)
            md.validate(md.prepare_display(replace(snap, revision=snap.revision + 1, facts=facts, entries=entries)))
            if not existing:
                self._attachment(record.proof_path, fid, candidate.proof, "Verified proof", storage_format=snap.storage_format)
                if record.intuition_path:
                    self._attachment(record.intuition_path, fid, candidate.intuition,
                                     "Intuition — explanatory, not an additional fact", storage_format=snap.storage_format)
            self._commit(snap, facts=facts, entries=entries)
            return fid

    def _revoke_state(self, snap: KnowledgeSnapshot, fact_id: str, reason: str,
                      review: dict | None = None) -> tuple[dict, dict, list[str]]:
        revoked = [fact_id, *self._descendants(snap, fact_id)]
        facts = dict(snap.facts)
        now = utc_now()
        for fid in revoked:
            f = facts[fid]
            metadata = {"reason": reason, "root": fact_id, "revoked_at": now}
            if review:
                metadata["review_root"] = fact_id
                if fid == fact_id:
                    metadata["review"] = dict(review, revoked=revoked)
            facts[fid] = replace(f, status="revoked", revision=f.revision + 1, revocation=metadata)
        entries = dict(snap.entries)
        for eid, entry in snap.entries.items():
            affected = sorted(set(entry.links) & set(revoked))
            if affected:
                invalid = sorted(set(entry.attributes.get("invalidated_by", ())) | set(affected))
                attrs = dict(entry.attributes, invalidated_by=invalid)
                state = entry.status
                if entry.kind == "conjecture" and attrs.get("resolution_fact_id") in revoked:
                    state = "reopened"
                entries[eid] = replace(entry, status=state, revision=entry.revision + 1, attributes=attrs, updated_at=now)
        return facts, entries, revoked

    def revoke(self, fact_id: str, reason: str) -> list[str]:
        """Operator/main emergency withdrawal; independent review uses apply_review."""
        md.check_text(reason)
        if not reason.strip():
            raise ValueError("revocation_requires_reason")
        with self._locked() as snap:
            if fact_id not in snap.facts or snap.facts[fact_id].status != "verified":
                raise ValueError(f"unknown_or_revoked_fact: {fact_id}")
            facts, entries, revoked = self._revoke_state(snap, fact_id, reason)
            self._commit(snap, facts=facts, entries=entries)
            return revoked

    def review_receipt(self, fact_id: str, request_id: str, *, run_id: str | None = None,
                       snapshot: KnowledgeSnapshot | None = None) -> dict | None:
        snap = snapshot or self.snapshot()
        fact = snap.facts.get(fact_id)
        if fact is None:
            return None
        metadata = fact.revocation if fact.status == "revoked" else fact.supersession
        review = metadata.get("review", {})
        if not request_id or review.get("request_id") != request_id or (run_id and review.get("run_id") != run_id):
            return None
        return {"accepted": True, "applied": True, "stage": "applied", "verdict": "correct",
                "fact_id": fact_id, "action": review["action"], "replacement_id": review.get("replacement_id"),
                "request_id": request_id, "run_id": review["run_id"], "revoked": list(review.get("revoked", ())),
                "verification_report": plain(review["verification"]["verification_report"])}

    def apply_review(self, review: dict, *, before_commit=None) -> dict:
        """Service-internal gate: recheck the reviewed snapshot and commit atomically."""
        review = plain(review)
        verification = review.get("verification", {})
        report = verification.get("verification_report", {})
        if (verification.get("verdict") != "correct" or not isinstance(report, dict)
                or not isinstance(report.get("summary"), str) or not report["summary"].strip()
                or report.get("critical_errors") != [] or report.get("gaps") != []
                or verification.get("repair_hints") != ""
                or any(verification.get(k) for k in ("errors", "critical_errors", "gaps"))):
            raise ValueError("review_requires_accepted_verification")
        fact_id, action = review["fact_id"], review["action"]
        if action not in ("revoke", "supersede"):
            raise ValueError("invalid_review_action")
        for key in ("reason", "requested_by", "request_id", "run_id"):
            md.check_text(review[key])
            if not review[key].strip():
                raise ValueError("incomplete_review_metadata")
        with self._locked() as snap:
            previous = self.review_receipt(fact_id, review["request_id"], snapshot=snap)
            if previous:
                self._sync_retry()
                return previous
            versions = review["expected_revisions"]
            if fact_id not in versions:
                raise ValueError("missing_review_target_revision")
            for fid, revision in versions.items():
                f = snap.facts.get(fid)
                if type(revision) is not int or not f or f.status != "verified" or f.revision != revision:
                    raise ValueError(f"review_evidence_changed: {fid}")
            evidence = snap.topological(review["evidence_fact_ids"])
            if set(evidence) - set(versions) or fact_id in evidence:
                raise ValueError("invalid_review_evidence")
            for fid in review["proof_checks"]:
                if fid not in versions:
                    raise ValueError("untracked_review_proof")
                self.proof(fid, snapshot=snap)
            if fact_id not in review["proof_checks"]:
                raise ValueError("missing_review_target_proof")
            if action == "revoke":
                if review.get("replacement_id"):
                    raise ValueError("replacement_not_allowed_for_revocation")
                facts, entries, _ = self._revoke_state(snap, fact_id, review["reason"], review)
            else:
                replacement_id = review["replacement_id"]
                if replacement_id not in evidence or replacement_id == fact_id:
                    raise ValueError("invalid_replacement")
                if fact_id in snap.topological([replacement_id]):
                    raise ValueError("replacement_depends_on_target")
                cursor, seen = replacement_id, {fact_id}
                while cursor:
                    if cursor in seen:
                        raise ValueError("supersession_cycle")
                    seen.add(cursor)
                    cursor = snap.facts[cursor].supersession.get("replacement_id")
                facts, entries = dict(snap.facts), dict(snap.entries)
                f = facts[fact_id]
                facts[fact_id] = replace(f, revision=f.revision + 1, supersession={
                    "replacement_id": replacement_id, "reason": review["reason"],
                    "superseded_at": utc_now(), "review": review})
            if before_commit is not None:
                before_commit(snap)
            committed = self._commit(snap, facts=facts, entries=entries)
            return self.review_receipt(fact_id, review["request_id"], snapshot=committed)
