"""Publication-oriented composition, independently versioned from citation policy."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from . import evidence as ev, references, render
from .publication import STYLE
from .models import ArticleError, digest, nonempty_text, read_json, require_keys, safe_path, string_list, write_json

REVISION = 3
REVIEW_CHECKS = ("concise", "publication_structure", "preliminaries_complete", "notation_consistent",
                 "literature_positioned", "dependency_ordered", "typography_readable",
                 "history_and_motivation", "classical_credit", "narrative_flow")
_RESULT_KINDS = {"theorem", "lemma", "fact", "corollary"}

POLICY = """Write a research paper for a specialist reader, not a transcript of an internal
verification run. Preserve the mathematical logic, hypotheses, quantifiers, dependency
order and all nontrivial arguments, NOT the wording, chunk boundaries or fact-by-fact
organization of the source. Organize around the main contribution and proof strategy.
State notation and assumptions once, merge overlapping arguments, inline one-use auxiliary
facts and retain separate lemmas only when they genuinely clarify the proof or are reused.
A combined lemma may express precisely the conjunction or a justified specialization of
supplied facts, never a stronger unproved result. Do not turn every internal fact into a
numbered lemma. Develop the main proof as a coherent argument, not a list of repeated
'we prove the stated result' introductions. Concisely present routine calculations while
still naming the operation and necessary conditions; 'obvious' or 'standard' cannot hide
a nontrivial gap. Replace a standard proof by a citation only if an available verified
citation check supports exactly that use. Unavailable citations remain visible audit gaps.
Avoid repetition between abstract, introduction, preliminaries and proofs. Use inline
mathematics for short expressions and displayed equations for important or multi-line
relations; do not split every elementary manipulation into a new display. Never shrink
fonts, margins or line spacing, impose an arbitrary page quota, drop necessary arguments,
invent novelty/priority, or treat a shorter text as necessarily better. No target compression
ratio: concision means removing redundancy while preserving readability and checkability.
All original units must be accounted for in a separate source_map, not copied verbatim
into the manuscript. The source_map is audit metadata, NOT a section of the paper. Mapping
IDs alone do not establish faithfulness; an independent reviewer checks the original text.
""" + STYLE

NARRATIVE = """Open Introduction with at least two concise substantive paragraphs BEFORE any
formal definition or theorem: first explain the research setting and the relevant published
line of work with author credit and supplied citations; then explain the concrete question,
why it matters mathematically, the obstacle and the contribution. Verified title/author/year
may establish publication identity and neutral topic context, NOT historical firsts or
unread theorem statements. Explain the distinction between our problem and adjacent work.
Do not call elementary self-contained proofs a reason to erase classical attribution:
cite appropriate supplied standard references in Preliminaries where their tools enter.
Give a brief purpose/transition paragraph after every subsection heading and immediately
before each formal definition or numbered result; explain why it is needed next rather
than restating its formula. Open the final proofs section with the proof strategy. Introduce
notation in dependency order. Integrate one-use auxiliary steps into prose instead of
stacking theorem/definition environments. Keep transitions specific and usually 1-3 sentences.
Keep the four main sections, with 1-2 broad subsections in Preliminaries and 1-3 in the
core section (at most five overall), none in Introduction or the final proofs section.
These are upper bounds, not targets. Group related tools and construction stages; do not
make a subsection for each lemma or calculation. Use ordinary transition prose within a
topic and a few named Steps only for genuinely long arguments, not substitute mini-sections.
After each subsection or Step heading the renderer starts the body on the next normal line,
without an extra blank line. Do not add manual spacing, empty blocks or TeX spacing commands
to recreate a blank line. Keep all mathematical content and necessary classical citations.
Keep previously correct citations unless there is a concrete relevance or evidence problem.
Every supplied usable reference must be used at a justified point or explicitly accounted
for by the citation-placement pass. Never delete all citations to fix one identifier.
Unverified original results remain open review issues; do not invent provenance, pad the
bibliography, or make unsupported novelty claims merely to supply a historical narrative.
"""
POLICY += NARRATIVE

CONTRACT = """Return exactly {article}. article contains title, abstract, keywords, msc,
sections, source_map. Omit format, evidence_digest, reference_digest and writing_revision:
the caller binds these machine-owned fields from the verified request. Never transcribe hashes.
title is {text,evidence}; abstract
is a paragraph block with id 'abstract'. keywords is a nonempty string array, msc is real
2020 MSC codes or []. Exactly four sections {id,role,title,blocks}, with roles introduction,
preliminaries, core, main_proofs. Use titles Introduction, Preliminaries, a topic-specific
technical title, Proofs of the main results. All sections must have substantive blocks.
When publication_plan is supplied, start Preliminaries with a paragraph of actual global
conventions and start the core section with a paragraph explaining its technical objective.
Then use ALL planned topic IDs/titles as subsection blocks, in the planned order; develop
actual definitions, applicable tools and local proofs under them, not summary placeholders.
End Introduction with a paragraph roadmap containing [[ref:SECTION_ID]] for preliminaries,
core and main_proofs, and matching crossrefs. Omit publication_plan_digest: the caller binds it.
A proof block has the ordinary block fields plus proves (ONE result block ID) and nonempty
parts (paragraph/step/equation blocks). Parts are complete prose, not source-chunk replicas.
The proof container text must begin the actual reasoning, such as fixing hypotheses or
introducing the construction, not 'We prove the stated result' or 'Proof of ...' boilerplate:
the renderer supplies the proof heading. Use one coherent part for a short argument; use
multiple named steps only when they expose distinct nontrivial dependencies. Never remove
necessary content or turn a proof container into a paragraph to reduce spacing.
Subsection headings are plain text without TeX or citation/cross-reference markers; bind
their evidence to original units and place citations in the explanatory text underneath.
Keep main targets precisely stated as theorem blocks in Introduction and fully proved.
Dependency statements may be merged into useful lemmas, definitions or explanatory prose,
and their arguments integrated into other proofs. An internally established numbered result
must bind original statement units and have one proof block. A preliminary external theorem
may instead carry external_result=CHECK_ID only when that original_text check verifies its
precise statement and applicability; never use metadata-only background as theorem evidence.
Do not label our main result as external. Every original proof unit must
be represented in actual proof text, with several units allowed in one concise passage.
Do not put internal IDs, audits, coverage percentages or source_map explanations in prose.

source_map is [{unit_id,disposition,blocks,citations,reason}], EXACTLY one row per supplied
original unit. disposition is developed, merged, routine, or cited. blocks lists nonempty
IDs of manuscript blocks whose evidence includes that unit; use proof parts/proof blocks
for proof units, not a mention in the abstract. citations lists verified citation-check IDs
used at those blocks, or []. reason specifically explains how this original argument or
assumption is represented. merged must still account for all distinct original conditions;
routine must identify the calculation/method retained. cited requires an applicable verified
citation check bound to that unit, not merely a related title or a generic bibliography.
No omitted/discarded disposition. Introduce no new mathematical results. Only cross-reference
numbered result/definition/remark/equation/subsection blocks or section IDs, never plain
paragraphs, source IDs, proof blocks or proof parts. Include all citations actually used.
"""

FORMAT_COVERAGE_CONTRACT = """
For original-source coverage, map each original statement to a numbered theorem,
lemma, fact or corollary in the body that carries that statement ID. A numbered
result may represent several original statements. For each original proof unit,
map it to the proof block or a part of the proof whose proves field names that
same numbered result. If a long original proof contains subsidiary arguments,
develop those arguments as steps within the proof of its original result. Mapping
to proofs of subsidiary lemmas alone does not cover that original proof. These
requirements take precedence over the editorial guidance about inlining facts.
"""


def archive(root: Path) -> Path | None:
    path = safe_path(root, "article.json")
    if not path.exists():
        return None
    article = read_json(path)
    target = safe_path(root, "migrations/writing-" + str(REVISION) + "/" + digest(article)[:24])
    target.mkdir(parents=True, exist_ok=True)
    for name in ("article.json", "paper.tex", "paper.pdf", "references.tex", "references.bib", "audit.json",
                 "draft-status.json", "checkpoint.json", "compilation.json", "source-map.json", "status.json"):
        source = safe_path(root, name)
        if source.exists() and not (target / name).exists():
            shutil.copyfile(source, target / name)
    return target


def revision(article: dict) -> int:
    value = article.get("writing_revision", 0)
    if type(value) is not int or value not in (0, 1, 2, REVISION):
        raise ArticleError("unsupported_article_writing_revision")
    return value


def validate_map(article: dict) -> None:
    if not revision(article):
        if "source_map" in article:
            raise ArticleError("source_map_requires_writing_revision")
        return
    if not isinstance(article.get("source_map"), list) or not article["source_map"]:
        raise ArticleError("publication_article_requires_source_map")
    seen = set()
    for row in article["source_map"]:
        require_keys(row, {"unit_id", "disposition", "blocks", "citations", "reason"})
        uid = nonempty_text(row["unit_id"])
        if uid in seen or row["disposition"] not in ("developed", "merged", "routine", "cited"):
            raise ArticleError("invalid_publication_source_treatment")
        seen.add(uid)
        string_list(row["blocks"], nonempty=True)
        string_list(row["citations"])
        nonempty_text(row["reason"])
        if row["disposition"] == "cited" and not row["citations"]:
            raise ArticleError("citation_cannot_replace_proof_without_verified_source")


def structural_coverage(article: dict, evidence: dict, ledger: dict) -> dict:
    """Count only mapped manuscript passages that can carry their claimed unit."""
    units = ev.units(evidence)
    all_blocks = {b["id"]: b for b in render.walk_blocks(article)}
    if set(article["title"]["evidence"]) - units.keys():
        raise ArticleError("unknown_title_evidence")
    for block in all_blocks.values():
        if set(block["evidence"]) - units.keys():
            raise ArticleError("unknown_article_evidence_or_cross_reference")

    writing_revision = revision(article)
    if writing_revision:
        rows = {row["unit_id"]: row for row in article["source_map"]}
        if rows.keys() - units.keys():
            raise ArticleError("unknown_publication_source_map_unit")
        source_map = {"writing_revision": writing_revision, "source_map_digest": digest(article["source_map"])}
    else:
        # Segmented and older drafts encode the same links directly on each passage.
        source_map = {}
        rows = {uid: {"blocks": [], "citations": []} for uid in units}
        for block in all_blocks.values():
            for uid in block["evidence"]:
                rows[uid]["blocks"].append(block["id"])

    top_level = [block for section in article["sections"] for block in section["blocks"]]
    body = {block["id"] for block in all_blocks.values()
            if block["id"] != article["abstract"]["id"] and block["kind"] != "subsection"
            and render._MARKER.sub("", block["text"]).strip()}
    result_units = {}
    for block in top_level:
        if block["kind"] not in _RESULT_KINDS:
            continue
        bid = block["id"]
        result_units[bid] = set()
        if bid in body and not block.get("external_result"):
            result_units[bid] = {uid for uid in block["evidence"] if uid in rows
                                 and units[uid]["kind"] == "statement" and bid in rows[uid]["blocks"]}
    proof_facts, proofs = {}, {}
    for block in top_level:
        if block["kind"] != "proof":
            continue
        target = block["proves"]
        if target not in result_units:
            raise ArticleError("proof_has_no_result_statement")
        proofs.setdefault(target, []).append(block["id"])
        facts = {units[uid]["fact_id"] for uid in result_units[target]}
        for piece in [block, *block["parts"]]:
            proof_facts[piece["id"]] = facts

    mapped = {uid: [] for uid in units}
    for uid, row in rows.items():
        available_citations = set()
        unit = units[uid]
        for bid in row["blocks"]:
            block = all_blocks.get(bid)
            if block is None or uid not in block["evidence"]:
                raise ArticleError("publication_source_map_not_bound_to_text")
            available_citations.update(block["citations"])
            if bid not in body:
                continue
            if unit["kind"] == "statement" and uid not in result_units.get(bid, ()):
                continue
            if unit["kind"] == "proof" and unit["fact_id"] not in proof_facts.get(bid, ()):
                continue
            mapped[uid].append(bid)
        if set(row["citations"]) - available_citations:
            raise ArticleError("publication_source_map_has_unbound_citation")

    missing = sorted(uid for uid, passages in mapped.items() if not passages)
    missing_by_kind = {kind: [uid for uid in missing if units[uid]["kind"] == kind]
                       for kind in ("statement", "definition", "proof", "bridge")}
    used = {cid for block in all_blocks.values() for cid in block["citations"]}
    return {**source_map, "passed": not missing, "draft_complete": not missing, "missing_evidence": missing,
            "missing_by_kind": missing_by_kind, "citation_coverage": references.coverage(ledger, used),
            "evidence_to_blocks": mapped, "statement_to_proof": proofs, "formal_certificate": False}


def coverage(article: dict, evidence: dict, ledger: dict) -> dict:
    source = ev.units(evidence)
    blocks = list(render.walk_blocks(article))
    by_id = {b["id"]: b for b in blocks}
    labels = {s["id"] for s in article["sections"]} | {
        b["id"] for b in blocks if b["kind"] in render._NUMBERED or b["kind"] == "subsection"}
    if set(article["title"]["evidence"]) - source.keys():
        raise ArticleError("unknown_title_evidence")
    mapping, statements, used = {uid: [] for uid in source}, {}, set()
    for block in blocks:
        if set(block["evidence"]) - source.keys() or set(block["crossrefs"]) - labels:
            raise ArticleError("unknown_article_evidence_or_cross_reference")
        if any(not references.usable_check(ledger, cid) for cid in block["citations"]):
            raise ArticleError("unverified_article_citation")
        used.update(block["citations"])
        for uid in block["evidence"]:
            mapping[uid].append(block["id"])
        if block["kind"] in ("theorem", "lemma", "fact", "corollary"):
            bound = [uid for uid in block["evidence"] if source[uid]["kind"] == "statement"]
            if block.get("external_result"):
                check = ledger["checks"][block["external_result"]]
                obligation = ledger["obligations"][check["obligation_id"]]
                if (bound or check.get("evidence_level", "original_text") != "original_text"
                        or obligation["kind"] != "mathematical"
                        or not set(block["evidence"]) & set(obligation["source_units"])):
                    raise ArticleError("external_preliminary_result_requires_applicable_original_text")
            else:
                if not bound:
                    raise ArticleError("publication_result_has_no_original_statement")
                statements[block["id"]] = bound
    proofs, proof_blocks = {}, set()
    for block in blocks:
        if block["kind"] == "proof":
            if block["proves"] not in statements or block["proves"] in proofs:
                raise ArticleError("missing_or_duplicate_proof_target")
            proofs[block["proves"]] = block["id"]
            proof_blocks.update(b["id"] for b in [block, *block["parts"]])
    rows = {row["unit_id"]: row for row in article["source_map"]}
    if rows.keys() - source.keys():
        raise ArticleError("unknown_publication_source_map_unit")
    missing = [uid for uid in source if uid not in rows or not mapping[uid]]
    for uid, row in rows.items():
        for bid in row["blocks"]:
            if bid not in by_id or uid not in by_id[bid]["evidence"]:
                raise ArticleError("publication_source_map_not_bound_to_text")
        if source[uid]["kind"] == "proof" and not set(row["blocks"]) & proof_blocks:
            missing.append(uid)
        available = {cid for bid in row["blocks"] for cid in by_id[bid]["citations"]}
        if set(row["citations"]) - available:
            raise ArticleError("publication_source_map_has_unbound_citation")
        if row["disposition"] == "cited":
            if not any(uid in ledger["obligations"][ledger["checks"][cid]["obligation_id"]]["source_units"]
                       for cid in row["citations"]):
                raise ArticleError("publication_citation_does_not_support_original_unit")
    missing.extend("proof-of:" + sid for sid in statements if sid not in proofs)
    intro = {b["id"] for b in article["sections"][0]["blocks"] if b["kind"] == "theorem"}
    for fid in evidence["targets"]:
        if not any(fid + ":statement" in statements.get(sid, []) for sid in intro):
            missing.append("principal-result:" + fid)
    citations = references.coverage(ledger, used)
    text_chars = sum(len(b["text"]) for b in blocks)
    normalized = [re.sub(r"\s+", " ", b["text"]).strip() for b in blocks if len(b["text"]) > 240]
    duplicate_passages = sorted({t for t in normalized if normalized.count(t) > 1})
    return {"passed": not missing and citations["complete"] and not duplicate_passages,
            "draft_complete": not missing, "missing_evidence": sorted(set(missing)),
            "citation_coverage": citations, "evidence_to_blocks": mapping, "statement_to_proof": proofs,
            "formal_certificate": False, "writing_revision": REVISION,
            "source_map_digest": digest(article["source_map"]),
            "editorial_metrics": {"original_facts": len(evidence["facts"]), "numbered_results": len(statements),
                "source_units": len(source), "text_chars": text_chars,
                "duplicate_substantive_passages": len(duplicate_passages)}}
