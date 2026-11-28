"""Deterministic, readable Markdown with explicit machine record boundaries."""
from __future__ import annotations

import html
import json
import re
from dataclasses import replace
from typing import Any

from .schema import (
    ENTRY_STATUSES, KNOWLEDGE_FORMAT, KNOWLEDGE_MARKERS,
    FactRecord, KnowledgeEntry, KnowledgeSnapshot, validate_json_value,
)
from .entry_policy import validate_entry_policy

MARKER = KNOWLEDGE_MARKERS[KNOWLEDGE_FORMAT]
_HEADERS = {marker: re.compile(rf"\A{re.escape(marker)}document (.+) -->\n")
            for marker in KNOWLEDGE_MARKERS.values()}
_RECORDS = {marker: re.compile(rf"^{re.escape(marker)}(fact|entry) ([^\n]+) -->\n(.*?)^{re.escape(marker)}end -->$",
                               re.M | re.S)
            for marker in KNOWLEDGE_MARKERS.values()}
FACT_ID = re.compile(r"[0-9a-f]{16}\Z")
ENTRY_ID = re.compile(r"entry-[0-9a-f]{16}\Z")
SECTIONS = (
    ("overview", "Project overview — strategy, not fact"),
    ("fact", "Verified facts — statements only"),
    ("route", "Exploration routes — not facts"),
    ("conjecture", "Conjectures — not facts"),
    ("experience", "Experience and leads — not facts"),
    ("history", "Revocations and history — not usable as premises"),
)


def check_text(text: str, *, marker: str | None = None) -> None:
    reserved = (marker,) if marker is not None else KNOWLEDGE_MARKERS.values()
    if not isinstance(text, str) or any(prefix in text for prefix in reserved):
        raise ValueError("invalid_text_or_reserved_record_boundary")
    validate_json_value(text)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e")


def anchor(identifier: str) -> str:
    return f"fact-{identifier}" if FACT_ID.fullmatch(identifier) else identifier


PREFIXES = {"fact": "F", "overview": "O", "route": "R", "conjecture": "C",
            "experience": "E", "lead": "L", "feedback": "V", "report": "P"}
DISPLAY_ID = re.compile(r"[FORCELVP](?:[0-9]{3}|[1-9][0-9]{3,})\Z")


def short_title(text: str) -> str:
    line = next((line.strip().lstrip("# ") for line in text.splitlines() if line.strip()), "Fact")
    line = re.sub(r"\s+", " ", line).strip() or "Fact"
    return line if len(line) <= 100 else line[:97].rstrip() + "..."


def check_title(title: str, *, marker: str | None = None) -> None:
    check_text(title, marker=marker)
    if not title.strip() or any(c in title for c in "\r\n") or len(title) > 160:
        raise ValueError("invalid_display_title: use a nonempty single line up to 160 characters")


def prepare_display(snapshot: KnowledgeSnapshot) -> KnowledgeSnapshot:
    """Allocate missing presentation metadata in memory, never by reading proofs."""
    counters = {prefix: 0 for prefix in PREFIXES.values()}
    for record in (*snapshot.facts.values(), *snapshot.entries.values()):
        if record.display_id:
            if not DISPLAY_ID.fullmatch(record.display_id):
                raise ValueError("invalid_display_id")
            prefix = record.display_id[0]
            counters[prefix] = max(counters[prefix], int(record.display_id[1:]))
    sources = {e.attributes.get("resolution_fact_id"): e.title
               for e in sorted(snapshot.entries.values(), key=lambda e: (e.created_at, e.entry_id))
               if e.kind == "conjecture" and e.status == "proved"}
    def number(record, kind):
        if record.display_id:
            return record.display_id
        prefix = PREFIXES[kind]
        counters[prefix] += 1
        return f"{prefix}{counters[prefix]:03d}"
    facts = {f.fact_id: replace(f, display_id=number(f, "fact"),
                               title=f.title or short_title(sources.get(f.fact_id, f.statement)))
             for f in sorted(snapshot.facts.values(), key=lambda f: (f.created_at, f.fact_id))}
    entries = {e.entry_id: replace(e, display_id=number(e, e.kind))
               for e in sorted(snapshot.entries.values(), key=lambda e: (e.created_at, e.entry_id))}
    return replace(snapshot, facts=facts, entries=entries, display_version=1)


def display_name(record: FactRecord | KnowledgeEntry) -> str:
    title = record.title or short_title(record.statement) if isinstance(record, FactRecord) else record.title
    return f"{record.display_id} · {title}" if record.display_id else title


def _label(text: str) -> str:
    text = html.escape(re.sub(r"\s+", " ", text), quote=False)
    return re.sub(r"([\\`*_\[\]{}])", lambda m: "&#" + str(ord(m[0])) + ";", text)


def link(identifier: str, records: dict | None = None) -> str:
    record = records.get(identifier) if records is not None else None
    label = _label(display_name(record)) if record else identifier
    if record and isinstance(record, FactRecord) and record.status == "revoked":
        label += " (revoked)"
    return f"[{label}](#{anchor(identifier)})"


def _render_body(body: str, marker: str) -> str:
    return f"{marker}body -->\n{body}\n{marker}end-body -->"


def _fact(fact: FactRecord, records: dict | None, marker: str) -> str:
    meta = fact.to_dict()
    if not fact.supersession:
        meta.pop("supersession")
    body = meta.pop("statement")
    if records is None:
        meta.pop("title")
        meta.pop("display_id")
    else:
        lines = [f"{marker}fact {_json(meta)} -->", f'<a id="{anchor(fact.fact_id)}"></a>',
                 f"### {_label(display_name(fact))} [{fact.status}]", "", _render_body(body, marker),
                 "", "**Depends on:** " +
                 (", ".join(link(pid, records) for pid in fact.predecessors) or "none"),
                 f"**Proof:** [完整证明 / Full proof — {fact.display_id}]({fact.proof_path})"]
        if fact.supersession:
            replacement_id = fact.supersession["replacement_id"]
            evidence = fact.supersession["review"].get("evidence_fact_ids", ())
            available = all(records[i].status == "verified" for i in [replacement_id, *evidence])
            state = "推荐替代 / Preferred replacement" if available else "替代依据已失效 / Replacement unavailable"
            lines += [f"**{state}:** {link(replacement_id, records)}",
                      "原结论未被判错；历史证明依赖保持不变。 / Original proof dependencies remain valid.",
                      "**Reason:** " + _label(fact.supersession["reason"])]
        if fact.revocation.get("review") or fact.revocation.get("review_root"):
            lines += ["**撤销原因 / Withdrawal reason:** " + _label(fact.revocation["reason"]),
                      "该认证不可作为前提；撤销不等于证明命题为假。 / Certification withdrawn, not necessarily disproved."]
        if fact.intuition_path:
            lines.append(f"**Intuition:** [直觉说明 / Intuition — {fact.display_id}]({fact.intuition_path})")
        if fact.glossary_introduces:
            lines += ["**Definitions:**"] + [f"- `{k}`: {v}" for k, v in sorted(fact.glossary_introduces.items())]
        lines += ["", "<details>", "<summary>记录信息 / Record metadata</summary>", "",
                  f"Internal ID: `{fact.fact_id}`", f"**Verification:** {_json(meta['verification'])}"]
        if fact.revocation:
            lines.append(f"**Revocation:** {_json(meta['revocation'])}")
        return "\n".join(lines + ["", "</details>", f"{marker}end -->"])
    lines = [f"{marker}fact {_json(meta)} -->", f'<a id="{anchor(fact.fact_id)}"></a>',
             f"### {fact.fact_id} [{fact.status}]", "", _render_body(body, marker),
             "", "**Depends on:** " + (", ".join(map(link, fact.predecessors)) or "none"),
             f"**Proof:** [open when needed]({fact.proof_path})"]
    if fact.intuition_path:
        lines.append(f"**Intuition:** [explanation]({fact.intuition_path})")
    if fact.glossary_introduces:
        lines += ["**Definitions:**"] + [f"- `{key}`: {value}" for key, value in sorted(fact.glossary_introduces.items())]
    lines += [f"**Verification:** {_json(meta['verification'])}"]
    if fact.revocation:
        lines.append(f"**Revocation:** {_json(meta['revocation'])}")
    return "\n".join(lines + [f"{marker}end -->"])


def _entry(entry: KnowledgeEntry, records: dict | None, marker: str) -> str:
    meta = entry.to_dict()
    body = meta.pop("body")
    if records is None:
        meta.pop("display_id")
    heading = entry.title if records is None else _label(display_name(entry))
    lines = [f"{marker}entry {_json(meta)} -->", f'<a id="{entry.entry_id}"></a>',
             f"### {heading} [{entry.kind}: {entry.status}; NOT A FACT]", "",
             _render_body(body, marker), "",
             "**Related:** " + (", ".join(link(i, records) for i in entry.links) or "none")]
    if entry.details_path:
        label = "read explanation" if records is None else f"详细说明 / Details — {entry.display_id}"
        lines.append(f"**Details:** [{label}]({entry.details_path})")
    if records is not None:
        lines += ["", "<details>", "<summary>记录信息 / Record metadata</summary>", "",
                  f"Internal ID: `{entry.entry_id}`"]
    if entry.attributes:
        lines.append(f"**Context:** {_json(meta['attributes'])}")
    if records is not None:
        lines += ["", "</details>"]
    return "\n".join(lines + [f"{marker}end -->"])


def serialize(snapshot: KnowledgeSnapshot) -> str:
    marker = KNOWLEDGE_MARKERS[snapshot.storage_format]
    meta = {"format": snapshot.storage_format, "problem_id": snapshot.problem_id, "revision": snapshot.revision}
    labels = None
    if snapshot.display_version:
        meta["display_version"] = snapshot.display_version
        labels = {**snapshot.facts, **snapshot.entries}
    has_reports = any(e.kind == "report" for e in snapshot.entries.values())
    navigation = ("[Problem](PROBLEM.md) · [Overview](#overview) · [Facts](#fact) · [Routes](#route) · "
                  "[Conjectures](#conjecture) · [Experience](#experience) · [History](#history)")
    if has_reports:
        navigation += " · [Public reports](#report)"
    lines = [f"{marker}document {_json(meta)} -->", "# Research knowledge", "", navigation, "",
             "Only active verified facts are usable premises. Proofs and long notes are linked, not embedded.",
             "Shared updates go through the knowledge service; private drafts and runtime logs stay separate."]
    if labels is not None:
        lines += ["", "**阅读编号 / Reading labels:** F = 事实 Facts · O = 概览 Overview · R = 路线 Routes · "
                  "C = 猜想 Conjectures · E = 经验 Experience · L = 线索 Leads · V = 验证反馈 Feedback.",
                  "Numbers are stable within this project; titles are navigation aids, not mathematical statements. "
                  "Use the internal IDs for tool arguments. Full statements below remain authoritative."]
        if has_reports:
            lines += ["P = 公开报告 Public reports — project-shared coordination, never verified premises."]
    sections = SECTIONS[:-1] + (("report", "Public reports — project-shared, not facts"),) + SECTIONS[-1:] if has_reports else SECTIONS
    for section, title in sections:
        lines += ["", f'<a id="{section}"></a>', f"## {title}"]
        if section in ("fact", "history"):
            selected = [f for f in snapshot.facts.values() if (f.status == "verified") == (section == "fact")]
            selected.sort(key=lambda f: int(f.display_id[1:]) if labels is not None else f.fact_id)
        else:
            kinds = {"experience", "lead", "feedback"} if section == "experience" else {section}
            selected = [e for e in snapshot.entries.values() if e.kind in kinds]
            selected.sort(key=lambda e: (e.display_id[0], int(e.display_id[1:])) if labels is not None else e.entry_id)
        if labels is not None and selected:
            lines += ["", "**本区目录 / Contents**"]
            for record in selected:
                identifier = record.fact_id if isinstance(record, FactRecord) else record.entry_id
                lines.append("- " + link(identifier, labels))
        for record in selected:
            lines.append("\n" + (_fact(record, labels, marker) if isinstance(record, FactRecord) else _entry(record, labels, marker)))
    return "\n".join(lines) + "\n"


def validate(snapshot: KnowledgeSnapshot) -> None:
    if not isinstance(snapshot.storage_format, str) or snapshot.storage_format not in KNOWLEDGE_MARKERS:
        raise ValueError("unsupported_knowledge_format")
    marker = KNOWLEDGE_MARKERS[snapshot.storage_format]
    check_text(snapshot.problem_id, marker=marker)
    if not snapshot.problem_id.strip() or type(snapshot.revision) is not int or snapshot.revision < 0:
        raise ValueError("invalid_document_metadata")
    if type(snapshot.display_version) is not int or snapshot.display_version not in (0, 1):
        raise ValueError("unsupported_display_version")
    labels = set()
    for record in (*snapshot.facts.values(), *snapshot.entries.values()):
        if isinstance(record, FactRecord):
            for field in (record.external_refs, record.verification, record.revocation, record.supersession):
                validate_json_value(field)
        else:
            validate_json_value(record.attributes)
        code = record.display_id
        if snapshot.display_version:
            prefix = "F" if isinstance(record, FactRecord) else PREFIXES.get(record.kind)
            if (not isinstance(code, str) or not DISPLAY_ID.fullmatch(code)
                    or code[0] != prefix or int(code[1:]) < 1 or code in labels):
                raise ValueError("invalid_or_duplicate_display_id")
            labels.add(code)
            if isinstance(record, FactRecord):
                check_title(record.title, marker=marker)
        elif code or (isinstance(record, FactRecord) and record.title):
            raise ValueError("display_metadata_requires_version")
    glossary: dict[str, str] = {}
    for fid, fact in snapshot.facts.items():
        if not FACT_ID.fullmatch(fid) or fid != fact.fact_id or fact.problem_id != snapshot.problem_id:
            raise ValueError("invalid_fact_identity")
        if fact.status not in ("verified", "revoked") or type(fact.revision) is not int or fact.revision < 1:
            raise ValueError("invalid_fact_state")
        for value in (fact.statement, fact.author, fact.created_at):
            check_text(value, marker=marker)
        if not fact.statement.strip() or not fact.author.strip():
            raise ValueError("empty_fact_content")
        if fact.proof_path != f"knowledge/proofs/{fid}.md":
            raise ValueError("invalid_proof_path")
        if fact.intuition_path not in ("", f"knowledge/notes/{fid}-intuition.md"):
            raise ValueError("invalid_intuition_path")
        if len(set(fact.predecessors)) != len(fact.predecessors) or fid in fact.predecessors:
            raise ValueError("invalid_predecessors")
        for pid in fact.predecessors:
            if pid not in snapshot.facts:
                raise ValueError(f"unknown_predecessor: {pid}")
        for key, value in fact.glossary_introduces.items():
            check_text(key, marker=marker)
            check_text(value, marker=marker)
            if fact.status == "verified":
                if key in glossary and glossary[key] != value:
                    raise ValueError(f"glossary_conflict: {key}")
                glossary[key] = value
        if fact.verification.get("verdict") != "correct":
            raise ValueError("missing_accepted_verification")
        if any(not isinstance(ref, dict) for ref in fact.to_dict()["external_refs"]):
            raise ValueError("invalid_external_refs")
    for fid, fact in snapshot.facts.items():
        if fact.supersession:
            replacement_id = fact.supersession.get("replacement_id")
            review = fact.supersession.get("review", {})
            verification = fact.to_dict()["supersession"].get("review", {}).get("verification", {})
            report = verification.get("verification_report", {})
            if (replacement_id not in snapshot.facts or replacement_id == fid
                    or review.get("action") != "supersede" or review.get("fact_id") != fid
                    or review.get("replacement_id") != replacement_id
                    or verification.get("verdict") != "correct"
                    or not isinstance(report, dict) or not isinstance(report.get("summary"), str)
                    or report.get("critical_errors") != [] or report.get("gaps") != []
                    or verification.get("repair_hints") != ""
                    or not review.get("run_id") or not review.get("request_id")
                    or any(i not in snapshot.facts for i in review.get("evidence_fact_ids", ()))):
                raise ValueError("invalid_supersession")
            check_text(fact.supersession.get("reason", ""), marker=marker)
            cursor, seen = fid, set()
            while cursor:
                if cursor in seen:
                    raise ValueError("supersession_cycle")
                seen.add(cursor)
                cursor = snapshot.facts[cursor].supersession.get("replacement_id")
                if cursor and cursor not in snapshot.facts:
                    raise ValueError("unknown_replacement")
    records = {**snapshot.facts, **snapshot.entries}
    requests = set()
    for eid, entry in snapshot.entries.items():
        validate_entry_policy(entry, records, marker=marker)
        if "research_policy_version" in entry.attributes and "record_request" in entry.attributes:
            key = (entry.author, entry.kind, entry.attributes["record_request"]["id"])
            if key in requests:
                raise ValueError("duplicate_record_request")
            requests.add(key)
        if not ENTRY_ID.fullmatch(eid) or eid != entry.entry_id:
            raise ValueError("invalid_entry_identity")
        if entry.kind not in ENTRY_STATUSES or entry.status not in ENTRY_STATUSES[entry.kind]:
            raise ValueError("invalid_entry_state")
        if type(entry.revision) is not int or entry.revision < 1:
            raise ValueError("invalid_entry_revision")
        for value in (entry.title, entry.body, entry.author, entry.created_at, entry.updated_at):
            check_text(value, marker=marker)
        if not entry.title.strip() or not entry.body.strip() or not entry.author.strip() or "\n" in entry.title:
            raise ValueError("invalid_entry_content")
        if entry.details_path and not re.fullmatch(rf"knowledge/notes/{eid}-[1-9][0-9]*(?:-[0-9a-f]{{16}})?\.md", entry.details_path):
            raise ValueError("invalid_details_path")
        for identifier in entry.links:
            if identifier not in snapshot.facts and identifier not in snapshot.entries:
                raise ValueError(f"unknown_link: {identifier}")
        if entry.kind == "conjecture" and entry.status in ("proved", "refuted"):
            support = entry.attributes.get("resolution_fact_id")
            if support not in snapshot.facts or snapshot.facts[support].status != "verified" or support not in entry.links:
                raise ValueError("conjecture_resolution_requires_active_fact")
    snapshot.topological()


def parse(text: str) -> KnowledgeSnapshot:
    marker = next((prefix for prefix in _HEADERS if text.startswith(prefix)), MARKER)
    match = _HEADERS[marker].match(text)
    if not match:
        raise ValueError(f"unsupported_knowledge_format: expected KNOWLEDGE.md ({KNOWLEDGE_FORMAT})")
    try:
        meta = json.loads(match.group(1))
        storage_format = meta.pop("format")
        if not isinstance(storage_format, str) or KNOWLEDGE_MARKERS.get(storage_format) != marker:
            raise ValueError("unsupported_knowledge_format")
        facts: dict[str, FactRecord] = {}
        entries: dict[str, KnowledgeEntry] = {}
        body_start = f"{marker}body -->\n"
        body_end = f"\n{marker}end-body -->"
        for record in _RECORDS[marker].finditer(text, match.end()):
            kind, payload, content = record.groups()
            data = json.loads(payload)
            if content.count(body_start) != 1 or content.count(body_end) != 1:
                raise ValueError("malformed_record_body")
            body = content.split(body_start, 1)[1].split(body_end, 1)[0]
            check_text(body, marker=marker)
            if kind == "fact":
                fact = FactRecord(statement=body, **data)
                if fact.fact_id in facts:
                    raise ValueError("duplicate_fact_id")
                facts[fact.fact_id] = fact
            else:
                entry = KnowledgeEntry(body=body, **data)
                if entry.entry_id in entries:
                    raise ValueError("duplicate_entry_id")
                entries[entry.entry_id] = entry
        snapshot = KnowledgeSnapshot(facts=facts, entries=entries, storage_format=storage_format, **meta)
        validate(snapshot)
        if serialize(snapshot) != text:
            raise ValueError("noncanonical_or_unparsed_knowledge_content")
        return snapshot
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed_knowledge_document: {exc}") from exc


def attachment(identifier: str, body: str, title: str, *, storage_format: str = KNOWLEDGE_FORMAT) -> str:
    marker = KNOWLEDGE_MARKERS[storage_format]
    check_text(body, marker=marker)
    return (f"# {title}\n\n[Back to knowledge](../../KNOWLEDGE.md#{anchor(identifier)})\n\n"
            f"{_render_body(body, marker)}\n")


def attachment_body(text: str) -> str:
    starts = [(index, marker) for marker in KNOWLEDGE_MARKERS.values()
              if (index := text.find(f"{marker}body -->\n")) != -1]
    if not starts:
        raise ValueError("malformed_knowledge_attachment")
    start, marker = min(starts)
    body_start = f"{marker}body -->\n"
    body_end = f"\n{marker}end-body -->"
    start += len(body_start)
    end = text.find(body_end, start)
    if end == -1 or text.count(body_start) != 1 or text.count(body_end) != 1:
        raise ValueError("malformed_knowledge_attachment")
    return text[start:end]
