"""Bounded title/author discovery, distinct from arXiv topical searching."""
from __future__ import annotations

import copy
import http.client
import json
import tarfile
from collections.abc import Callable
from urllib.parse import urlencode, urlsplit

from .budget import ArticleDeadlineExceeded
from .models import ArticleError, Cancelled, EvidenceChanged, digest, nonempty_text, require_keys, string_list

FOCUS_REVISION = 1
PLAN_CONTRACT = """Return {checked_units,materials_complete,truncated,candidates,obligations,queries,scope_notes}.
Read the supplied problem, principal results, method and original units as ONE paper.
Prioritize (1) the source of the problem, (2) directly related research, (3) important
borrowed theorems or methods. Do NOT create a separate obligation/query for each elementary
fact, Fubini, compactness, or routine algebra. A small number of authoritative textbooks
may cover genuinely useful standard background. No fixed bibliography count or padding.
checked_units lists ALL supplied original unit IDs. materials_complete concerns these
inputs, not completion of literature verification; truncated must be false if fully read.
candidates is [{key,metadata}], metadata with an identifiable title, authors/year/DOI/arxiv
only when actually known. These are unverified leads, never evidence. Do not copy internal
IDs into titles, invent bibliographic fields or treat the sample style paper as related work.
obligations is [{id,description,source_units,reference_ids,kind,critical}]. source_units
are original unit IDs; reference_ids uses candidate keys or existing reference IDs and may
be [] when the source is unknown. kind is historical for origin/priority, comparison for
claimed advances, mathematical for borrowed results, background for neutral related work.
Substantive borrowed results and actual historical/priority/comparison claims are critical.
Group genuine routine prerequisite tools by an authoritative textbook; these may be
noncritical mathematical needs, assessed for general applicability without invented locators.
Do not invent a historical attribution requirement merely from an internal benchmark label:
if provenance is unknown and no historical claim is made, record that limit in scope_notes,
not a fictitious citation obligation. Never erase a real borrowed theorem or claimed priority.
queries is at most SIX short targeted searches, including known author/title clues and needed
preliminary topics, not one query per elementary calculation. scope_notes is
{problem_origin,related_work,borrowed_results}; explain actual claims and unknown provenance.
Do not copy the style paper's bibliography or count; unresolved sources for real claims remain gaps.
"""


def validate_plan(value: dict, material: dict) -> None:
    require_keys(value, {"checked_units", "materials_complete", "truncated", "candidates", "obligations", "queries", "scope_notes"})
    known = {u["id"] for u in material["original_units"]}
    if (set(string_list(value["checked_units"])) != known or value["materials_complete"] is not True
            or value["truncated"] is not False):
        raise ArticleError("focused_reference_plan_incomplete")
    if not isinstance(value["candidates"], list) or len(value["candidates"]) > 12:
        raise ArticleError("focused_reference_candidates_must_be_bounded")
    keys = set(material["existing_references"])
    for row in value["candidates"]:
        require_keys(row, {"key", "metadata"})
        key = nonempty_text(row["key"])
        if key in keys or not isinstance(row["metadata"], dict):
            raise ArticleError("invalid_focused_reference_candidate")
        nonempty_text(row["metadata"].get("title"))
        keys.add(key)
    if not isinstance(value["obligations"], list):
        raise ArticleError("focused_obligations_must_be_array")
    ids = set()
    for row in value["obligations"]:
        require_keys(row, {"id", "description", "source_units", "reference_ids", "kind", "critical"})
        oid = nonempty_text(row["id"])
        if (oid in ids or set(string_list(row["source_units"], nonempty=True)) - known
                or set(string_list(row["reference_ids"])) - keys or type(row["critical"]) is not bool
                or row["kind"] not in ("historical", "comparison", "mathematical", "background")):
            raise ArticleError("invalid_focused_reference_obligation")
        nonempty_text(row["description"])
        ids.add(oid)
    if len(string_list(value["queries"])) > 6 or any(len(q) > 350 for q in value["queries"]):
        raise ArticleError("focused_searches_must_be_short_and_prioritized")
    require_keys(value["scope_notes"], {"problem_origin", "related_work", "borrowed_results"})
    for text in value["scope_notes"].values():
        nonempty_text(text)


def lookup(ledger: dict, rid: str, reader, checkpoint=lambda: None) -> None:
    from . import references as refs
    row = ledger["references"][rid]
    if not reader.allowed or row["metadata_verified"]:
        return
    declared = row["declared"]
    authors = declared.get("authors", [])
    if isinstance(authors, str):
        authors = [authors]
    params = {"query.title": declared["title"], "rows": 5}
    if authors:
        params["query.author"] = " ".join(authors)
    url = "https://api.crossref.org/works?" + urlencode(params)
    requests = ledger.setdefault("retrieval_requests", {})
    key = digest([rid, url])
    record = requests.get(key)
    if record is None:
        record = {"reference_id": rid, "url": url, "state": "pending", "attempts": 0}
        requests[key] = record
    if record["state"] in ("complete", "not_found", "conflict") or record["attempts"] >= 2:
        return
    record["attempts"] += 1
    checkpoint()
    try:
        raw, _, final = reader.get(url)
        payload = json.loads(raw)
        items = payload["message"]["items"]
        if not isinstance(items, list):
            raise ArticleError("invalid_crossref_title_results")
        matches = [item for item in items if isinstance(item, dict)
                   and refs._metadata_match(declared, refs.crossref_metadata(item))]
        identities = {str(item.get("DOI", "")).lower() for item in matches}
        if len(identities) > 1:
            record.update(state="conflict", candidates=sorted(identities))
            return
        if not matches:
            record.update(state="not_found")
            return
        found = refs.crossref_metadata(matches[0])
        refs.identifiers(found)
        sid = refs.add_source(ledger, rid, json.dumps(matches[0], ensure_ascii=False), origin=final, kind="metadata")
        refs.corroborate(ledger, rid, found, sid)
        refs.merge_crossref_source_urls(row, matches[0])
        record.update(state="complete", source_id=sid, doi=found.get("doi"))
    except (ArticleDeadlineExceeded, Cancelled, EvidenceChanged):
        raise
    except (ValueError, OSError, KeyError, TypeError, http.client.HTTPException) as exc:
        record.update(state="unavailable", error_type=type(exc).__name__, error=refs.source_error(exc))
    finally:
        checkpoint()


def publisher_sources(ledger: dict, rid: str, reader, checkpoint=lambda: None, *, supplement: bool = False,
                      classify: Callable[[str, str], None] | None = None) -> None:
    from . import references as refs
    if not reader.allowed:
        return
    row = ledger["references"][rid]
    if not supplement and any(refs.source_is_fulltext(ledger, sid, rid) for sid in row["source_ids"]):
        return
    candidates = [*row.get("source_urls", []), *(row["declared"].get(field) for field in ("url", "source_url", "fulltext_url"))]
    urls = dict.fromkeys(url for candidate in candidates if (url := refs.source_url(candidate)))
    attempted = 0
    existing = {ledger["sources"][sid]["origin"] for sid in row["source_ids"]}
    for url in urls:
        if urlsplit(url).hostname not in reader.config.allowed_source_hosts or url in existing:
            continue
        key = digest([rid, "fulltext", url])
        requests = ledger.setdefault("retrieval_requests", {})
        previous = requests.get(key, {})
        if previous.get("state") == "complete" or previous.get("attempts", 0) >= 2:
            continue
        if attempted >= 3 or reader.requests >= reader.config.max_sources:
            break
        attempted += 1
        record = {"reference_id": rid, "url": url, "attempts": previous.get("attempts", 0) + 1, "state": "pending"}
        requests[key] = record
        checkpoint()
        try:
            text, final = reader.get_text(url)
            sid = refs.add_source(ledger, rid, text, origin=final, kind=refs.source_kind(text, final),
                                  version=str(row["metadata"].get("edition") or row["metadata"].get("year", "")))
            record.update(state="complete", source_id=sid)
        except (ArticleDeadlineExceeded, Cancelled, EvidenceChanged):
            raise
        except (ValueError, OSError, http.client.HTTPException, tarfile.TarError) as exc:
            record.update(state="unavailable", error_type=type(exc).__name__, error=refs.source_error(exc))
            continue
        finally:
            checkpoint()
        if classify and ledger["sources"][sid]["kind"] != "metadata" and not refs.source_is_fulltext(ledger, sid, rid):
            classify(rid, sid)
        if not supplement and refs.source_is_fulltext(ledger, sid, rid):
            break


def focused_queries_used(focus: dict) -> int:
    """Every original query and explicitly refreshed query spends a search slot."""
    return len(focus.get("searches", {})) + sum(len(batch["searches"]) for batch in focus.get("refreshes", []))


def focused_search(ledger: dict, reader, queries: list[str], checkpoint=lambda: None, *, review: bool = False,
                   refresh: dict | None = None) -> None:
    from . import references as refs
    if not reader.allowed:
        ledger["stop_reason"] = "literature_search_disabled_for_this_run"
        ledger["focus"]["search_complete"] = not queries
        checkpoint()
        return
    focus = ledger["focus"]
    batch = focus if refresh is None else refresh
    searches = batch.setdefault("searches", {})
    for rid in list(ledger["references"]):
        doi, aid = refs.identifiers(ledger["references"][rid]["declared"])
        if not doi and not aid:
            lookup(ledger, rid, reader, checkpoint)
    for query in queries:
        if query not in searches:
            if (focused_queries_used(focus) >= reader.config.max_queries
                    or (not review and refresh is None and len(searches) >= 6)):
                skipped = batch.setdefault("skipped_queries", [])
                if query not in skipped:
                    skipped.append(query)
                continue
        record = searches.setdefault(query, {"attempts": 0, "state": "pending", "reference_ids": []})
        if record["state"] in ("complete", "empty"):
            continue
        while record["attempts"] < 2:
            reader.tick()
            if reader.requests >= reader.config.max_sources:
                record.update(state="unavailable", error="article_source_budget_exhausted")
                break
            record["attempts"] += 1
            checkpoint()
            try:
                url = "https://api.crossref.org/works?" + urlencode({"query.bibliographic": query, "rows": 5})
                raw, _, final = reader.get(url)
                items = json.loads(raw)["message"]["items"]
                if not isinstance(items, list):
                    raise ArticleError("invalid_crossref_topic_results")
                for item in items[:5]:
                    found = refs.crossref_metadata(item)
                    if not all(found.get(k) for k in ("title", "authors", "year", "doi")):
                        continue
                    rid = refs.add_reference(ledger, found)
                    sid = refs.add_source(ledger, rid, json.dumps(item, ensure_ascii=False), origin=final, kind="metadata")
                    refs.corroborate(ledger, rid, found, sid)
                    row = ledger["references"][rid]
                    refs.merge_crossref_source_urls(row, item)
                    if rid not in record["reference_ids"]:
                        record["reference_ids"].append(rid)
                record.update(state="complete" if record["reference_ids"] else "empty")
                break
            except (ArticleDeadlineExceeded, Cancelled, EvidenceChanged):
                raise
            except (ValueError, OSError, KeyError, TypeError, http.client.HTTPException) as exc:
                record.update(state="unavailable", error_type=type(exc).__name__, error=refs.source_error(exc))
            finally:
                checkpoint()
    all_batches = [focus, *focus.get("refreshes", [])]
    completed = {q for batch in all_batches for q, row in batch.get("searches", {}).items()
                 if row["state"] in ("complete", "empty")}
    skipped = list(dict.fromkeys(focus.get("skipped_queries", [])))
    resolved = [q for q in skipped if q in completed]
    if resolved:
        focus.setdefault("resolved_skipped_queries", []).extend(q for q in resolved
            if q not in focus.get("resolved_skipped_queries", []))
        focus["skipped_queries"] = [q for q in skipped if q not in resolved]
    attempted = {q for batch in all_batches for q in batch.get("searches", {})}
    focus["search_complete"] = attempted <= completed and not focus.get("skipped_queries")
    checkpoint()


def validate_selection(value: dict, material: dict) -> None:
    require_keys(value, {"checked_results", "assignments", "discarded", "unresolved"})
    known = set(material["references"])
    if set(string_list(value["checked_results"])) != known:
        raise ArticleError("focused_selection_missing_results")
    handled = set()
    if not isinstance(value["assignments"], list):
        raise ArticleError("focused_assignments_must_be_array")
    pairs = set()
    for row in value["assignments"]:
        require_keys(row, {"reference_id", "obligation_id", "reason"})
        pair = row["reference_id"], row["obligation_id"]
        if pair in pairs or pair[0] not in known or pair[1] not in material["obligations"]:
            raise ArticleError("invalid_focused_selection_binding")
        nonempty_text(row["reason"])
        handled.add(pair[0])
        pairs.add(pair)
    for field in ("discarded", "unresolved"):
        for rid in string_list(value[field]):
            if rid not in known or rid in handled:
                raise ArticleError("focused_selection_duplicate_or_unknown")
            handled.add(rid)
    if handled != known:
        raise ArticleError("focused_selection_unaccounted_results")


PATCH_CONTRACT = """Return {insertions,background,unplaced}. DO NOT return or rewrite the article.
insertions is [{block_id,after,citations}]. after is a nonempty EXACT unique substring of
that block, ending OUTSIDE math at the real use of a result; citations lists usable check IDs
to insert immediately after it. Do not change any existing words, equations or hypotheses.
Do not attach background-only citations to proof steps, or force an unrelated citation.
background is at most TWO short paragraph blocks {id,kind,text,evidence,citations,crossrefs}
for the Introduction only, where necessary to describe directly related work or provenance.
Use safe LaTeX and [[cite:CHECK_ID]] markers, no new theorem, priority/novelty claims unless
supported by original-source checks. Metadata/background checks allow neutral related-reading
statements only, never asserted mathematical results or historical origin. Text at most
1200 characters per background block, no crossrefs. Existing title/abstract/sections/proofs
must remain unchanged except inserted citation markers. Do not append a textbook survey.
unplaced=[{citation_id,reason}] accounts for usable checks not already cited or added; keep
uncertain placements here, never invent an anchor. Account for every required_placement ID
(and all supplied usable checks if that list is absent). Explain specific reasons for an
unplaced source; a self-contained proof alone is NOT a reason to withhold classical credit.
Prefer an insertion in the actual preliminary-tool paragraph for standard references.
For related-work background, credit the supplied authors and relevant topic without claiming
first priority or an unread theorem. Do not duplicate the same publication under two IDs.
"""


def apply_citation_patch(article: dict, ledger: dict, patch: dict) -> dict:
    from . import audit, references as refs, render
    require_keys(patch, {"insertions", "background", "unplaced"})
    if any(not isinstance(patch[k], list) for k in patch) or len(patch["background"]) > 2:
        raise ArticleError("citation_patch_requires_bounded_arrays")
    result = copy.deepcopy(article)
    blocks = {b["id"]: b for b in render.walk_blocks(result)}
    labels = set(blocks) | {s["id"] for s in result["sections"]} | {"title"}
    context_ids = {b["id"] for s in result["sections"][:2] for b in s["blocks"] if b["kind"] == "paragraph"}
    valid = {cid for cid in ledger["checks"] if refs.usable_check(ledger, cid)}
    used = {cid for b in blocks.values() for cid in b["citations"]}
    changes = {}
    for row in patch["insertions"]:
        require_keys(row, {"block_id", "after", "citations"})
        if row["block_id"] not in blocks:
            raise ArticleError("unknown_citation_insertion_block")
        block = blocks[row["block_id"]]
        anchor = nonempty_text(row["after"])
        if block["text"].count(anchor) != 1:
            raise ArticleError("citation_anchor_must_be_exact_and_unique")
        cids = string_list(row["citations"], nonempty=True)
        if set(cids) - valid or set(cids) & set(block["citations"]):
            raise ArticleError("citation_insertion_unverified_or_redundant")
        for cid in cids:
            check = ledger["checks"][cid]
            if check.get("evidence_level") == "background_reference":
                if block["id"] not in context_ids:
                    raise ArticleError("background_citation_cannot_support_proof")
            elif not set(block["evidence"]) & set(ledger["obligations"][check["obligation_id"]]["source_units"]):
                raise ArticleError("citation_does_not_match_block_evidence")
        position = block["text"].index(anchor) + len(anchor)
        changes.setdefault(block["id"], []).append((position, cids))
        used.update(cids)
    for bid, inserts in changes.items():
        block = blocks[bid]
        for position, cids in sorted(inserts, reverse=True):
            block["text"] = block["text"][:position] + " " + " ".join("[[cite:" + cid + "]]" for cid in cids) + block["text"][position:]
            block["citations"] = list(dict.fromkeys([*block["citations"], *cids]))
        audit._block(block, nested=block["kind"] in ("step", "equation"))
    known_units = {uid for b in blocks.values() for uid in b["evidence"]} | set(article["title"]["evidence"])
    introduction = result["sections"][0]["blocks"]
    background_position = len(introduction)
    roadmap_targets = {s["id"] for s in result["sections"] if s["role"] in ("preliminaries", "core", "main_proofs")}
    if (introduction and introduction[-1]["kind"] == "paragraph"
            and roadmap_targets <= set(introduction[-1]["crossrefs"])):
        background_position -= 1
    if article.get("writing_revision", 0) >= 2:
        background_position = next((i for i, b in enumerate(introduction) if b["kind"] != "paragraph"), background_position)
    for block in patch["background"]:
        audit._block(block)
        if (block["kind"] != "paragraph" or block["id"] in labels or len(block["text"]) > 1200
                or block["crossrefs"] or not block["citations"] or set(block["citations"]) - valid
                or set(block["evidence"]) - known_units):
            raise ArticleError("invalid_citation_background_paragraph")
        introduction.insert(background_position, copy.deepcopy(block))
        background_position += 1
        labels.add(block["id"])
        used.update(block["citations"])
    unplaced = set()
    for row in patch["unplaced"]:
        require_keys(row, {"citation_id", "reason"})
        nonempty_text(row["reason"])
        cid = row["citation_id"]
        if cid not in valid or cid in used or cid in unplaced:
            raise ArticleError("invalid_unplaced_citation")
        unplaced.add(cid)
    from . import credit
    if credit.missing(ledger, used, unplaced):
        raise ArticleError("citation_patch_did_not_account_for_verified_sources")
    result["reference_digest"] = digest(ledger)
    return result


def topical_queries(queries: list[str], ledger: dict) -> list[str]:
    """Book identity is resolved by title/author, not all-word AND on arXiv."""
    from .references import normal
    book_titles = [normal(r["declared"]["title"]) for r in ledger["references"].values()
                   if not r["declared"].get("arxiv") and not r["declared"].get("doi")]
    result = []
    for query in queries:
        normalized = normal(query)
        if any(title and title in normalized for title in book_titles):
            continue
        query = query.strip()
        if query and query not in result:
            result.append(query)
    return result
