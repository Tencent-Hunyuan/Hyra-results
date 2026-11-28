"""Request-bound citation handles and conservative bibliography identity grouping."""
from __future__ import annotations

import re

from . import references
from .models import ArticleError

REVISION = 1
HANDLE_CONTRACT = """Use the supplied short citation handles C001, C002, ... literally in
[[cite:C001]], citations arrays, source_map citations and external_result. The caller
restores verified internal IDs; never invent or transcribe long citation hashes. Fix a
bad citation handle locally, not by deleting other valid citations. Available background
checks support neutral context/related reading, not theorem applicability or first priority.
Standard-reference checks may credit classical tools even when we include a self-contained
proof. Keep every relevant attribution at its actual use; do not add irrelevant padding.
"""
_MARKER = re.compile(r"\[\[cite:([^\]]+)\]\]")
_HANDLE = re.compile(r"C\d{3,}\Z")


def aliases(ledger: dict) -> dict[str, str]:
    return {"C" + str(i).zfill(3): cid for i, cid in enumerate(sorted(
        cid for cid in ledger["checks"] if references.usable_check(ledger, cid)), 1)}


def translate(value, mapping: dict[str, str], *, strict: bool = False, field: str = ""):
    def identifier(text):
        if text in mapping:
            return mapping[text]
        if strict and _HANDLE.fullmatch(text):
            raise ArticleError("unknown_citation_handle: " + text)
        return text

    if isinstance(value, dict):
        return {identifier(k) if field == "citations" else k: translate(v, mapping, strict=strict, field=k)
                for k, v in value.items()}
    if isinstance(value, list):
        return [translate(v, mapping, strict=strict, field=field) for v in value]
    if isinstance(value, str):
        if field in ("citations", "citation_id", "external_result", "required_placement"):
            return identifier(value)
        return _MARKER.sub(lambda m: "[[cite:" + identifier(m.group(1)) + "]]", value)
    return value


def bibliography_aliases(ledger: dict, used: set[str]) -> dict[str, str]:
    result, identities = {}, {}
    for rid in sorted({ledger["checks"][cid]["reference_id"] for cid in used}):
        row = ledger["references"][rid]
        if not row["metadata_verified"] or row["conflicts"]:
            raise ArticleError("unverified_bibliography_entry")
        data = row["metadata"]
        doi, aid = references.identifiers(data)
        edition = references.normal(data.get("edition", ""))
        if doi:
            key = ("doi", doi, edition)
        elif aid:
            key = ("arxiv", aid, edition)
        else:
            authors = data.get("authors", [])
            authors = [authors] if isinstance(authors, str) else authors
            key = ("metadata", references.normal(data.get("title", "")),
                   tuple(references.normal(a) for a in authors), str(data.get("year", "")), edition)
            if not authors or not data.get("year"):
                key = ("record", rid)
        result[rid] = identities.setdefault(key, rid)
    return result


def groups(ledger: dict) -> dict[str, tuple]:
    usable = {cid for cid in ledger["checks"] if references.usable_check(ledger, cid)}
    rids = bibliography_aliases(ledger, usable)
    canonical = references.policy.canonical_ids(ledger)
    result = {}
    for cid in usable:
        row = ledger["checks"][cid]
        level = row.get("evidence_level", "original_text")
        result[cid] = (rids[row["reference_id"]], canonical[row["obligation_id"]], level,
                       cid if level == "original_text" else "")
    return result


def missing(ledger: dict, used: set[str], accounted: set[str] | None = None) -> list[str]:
    grouped = groups(ledger)
    done = {grouped[cid] for cid in used | (accounted or set()) if cid in grouped}
    selected = {}
    for cid in sorted(grouped):
        if grouped[cid] not in done:
            selected.setdefault(grouped[cid], cid)
    return list(selected.values())
