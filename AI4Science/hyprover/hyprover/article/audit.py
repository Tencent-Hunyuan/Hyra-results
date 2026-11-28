"""Structural checks and, in strict mode, independent editorial receipts."""
from __future__ import annotations

import re

from . import evidence as ev, references, render, writing_policy as writing
from .models import (FORMAT, FORMAT_COVERAGE, IDENTIFIER, REVIEW_MODES, STRICT,
                     ArticleError, digest, nonempty_text, require_keys, string_list)

_BLOCK_KINDS = {"paragraph", "subsection", "step", "proof", "theorem", "lemma", "fact",
                "corollary", "problem", "conjecture", "definition", "example", "remark", "equation"}
_CHECKS = ("faithful", "assumptions_preserved", "no_omissions", "citations_supported", "scope_respected", "style_conformant")
REVIEW_REVISION = 3
ADVISORY_CHECKS = frozenset({"style_conformant", "concise", "typography_readable"})
OPTIONAL_CITATION_CHECK = "optional_context_citation"
_MATH_DELIMITER = re.compile(r"\\(?:(begin|end)\s*\{([^}]+)\}|.)|\$\$?")
_DISPLAY_ENVIRONMENTS = {"equation*", "align*", "gather*"}
REVIEW_POLICY = """Acceptance is substantive correctness and a usable research paper, not
editorial perfection. Findings may add severity=blocking|advisory and check=CHECK_FIELD.
A blocking finding identifies an actual mathematical error/omission, materially ambiguous
assumption, unsupported claimed attribution, necessary missing external result, unreadable
formula, or violation of the requested structure. Do not block for exhaustive literature
search, lack of optional background, preferred phrasing, minor repetition, cosmetic spacing,
or supplementary bibliographic detail when the source is unambiguously identifiable.
Advisory findings are permitted ONLY for style_conformant, concise, typography_readable;
other failed checks remain blocking. Name the check, location and actionable issue. Keep
hard checks true only when actually verified. A false advisory check must have a matching
advisory finding. If only advisory issues remain, return verdict=pass with the warnings.
Positive observations belong nowhere in findings; do not turn a successful proof check into
an issue. Judge historical claims at their stated specificity, not at an invented stronger
claim. Metadata can support neutral related reading; verified standard references can credit
routine tools without a fabricated quotation. Necessary external theorems still require
applicable evidence. Do not request fresh proof of every elementary textbook fact. Preserve
all genuine critical citation obligations and unresolved unknown-scope candidates. Advisory
unresolved candidates have no essential obligation, identity conflict, or evidence failure:
record their limited credit gap as an advisory optional_context_citation finding, but never treat
an unresolved nonessential lead as a blocking theorem, priority, or proof defect. Critical
external results, historical/priority claims, method comparisons, conflicting identities and
failed evidence tasks remain blocking.
"""


def _block(block: dict, *, nested: bool = False) -> None:
    require_keys(block, {"id", "kind", "text", "evidence", "citations", "crossrefs"}, {"proves", "parts", "heading", "external_result"})
    if not isinstance(block["id"], str) or not IDENTIFIER.fullmatch(block["id"]):
        raise ArticleError("invalid_article_block_id")
    if block["kind"] not in _BLOCK_KINDS or (nested and block["kind"] not in ("paragraph", "step", "equation")):
        raise ArticleError("invalid_article_block_kind")
    render.safe_tex(block["text"])
    for field in ("evidence", "citations", "crossrefs"):
        string_list(block[field])
    if render.markers(block["text"], "cite") != set(block["citations"]):
        raise ArticleError("article_citation_map_mismatch")
    if render.markers(block["text"], "ref") != set(block["crossrefs"]):
        raise ArticleError("article_cross_reference_map_mismatch")
    if not block["evidence"] and not block["citations"]:
        raise ArticleError("article_block_has_no_evidence")
    if "external_result" in block:
        if (block["kind"] not in ("theorem", "lemma", "fact", "corollary")
                or nonempty_text(block["external_result"]) not in block["citations"]):
            raise ArticleError("external_result_requires_explicit_citation")
    if block["kind"] == "proof":
        nonempty_text(block.get("proves"))
        if not isinstance(block.get("parts"), list) or not block["parts"]:
            raise ArticleError("article_proof_requires_complete_parts")
        for part in block["parts"]:
            _block(part, nested=True)
    elif "proves" in block or "parts" in block:
        raise ArticleError("only_proofs_have_parts")
    if block["kind"] == "step":
        nonempty_text(block.get("heading"))
        if render._FORBIDDEN_PROSE.search(block["heading"]):
            raise ArticleError("disallowed_article_heading")
    elif "heading" in block:
        raise ArticleError("only_steps_have_headings")


def validate_article(article: dict) -> None:
    require_keys(article, {"format", "evidence_digest", "reference_digest", "title", "abstract", "keywords", "msc", "sections"},
                 {"writing_revision", "source_map", "publication_plan_digest", "writing_mode"})
    if "writing_mode" in article and (article["writing_mode"] != "segmented" or writing.revision(article)):
        raise ArticleError("invalid_article_writing_mode")
    writing.validate_map(article)
    if article["format"] != FORMAT:
        raise ArticleError("unsupported_article_format")
    require_keys(article["title"], {"text", "evidence"})
    render.safe_tex(article["title"]["text"])
    if render.markers(article["title"]["text"], "cite") or render.markers(article["title"]["text"], "ref"):
        raise ArticleError("article_title_cannot_contain_references")
    string_list(article["title"]["evidence"], nonempty=True)
    string_list(article["keywords"], nonempty=True)
    string_list(article["msc"])
    _block(article["abstract"])
    if article["abstract"]["kind"] != "paragraph":
        raise ArticleError("article_abstract_must_be_paragraph")
    if not isinstance(article["sections"], list) or len(article["sections"]) != 4:
        raise ArticleError("article_requires_four_structural_sections")
    ids = {"title", article["abstract"]["id"]}
    if len(ids) != 2:
        raise ArticleError("duplicate_article_id")
    for section in article["sections"]:
        require_keys(section, {"id", "role", "title", "blocks"})
        nonempty_text(section["title"])
        if not isinstance(section["id"], str) or not IDENTIFIER.fullmatch(section["id"]) or section["id"] in ids:
            raise ArticleError("duplicate_or_invalid_article_section_id")
        ids.add(section["id"])
        if not isinstance(section["blocks"], list) or not section["blocks"]:
            raise ArticleError("article_section_cannot_be_empty")
        for block in section["blocks"]:
            _block(block)
            for item in [block, *block.get("parts", [])]:
                if item["id"] in ids:
                    raise ArticleError("duplicate_article_id")
                ids.add(item["id"])


def _check_math_delimiters(text: str, *, implicit_display: bool = False) -> None:
    """Catch explicit display nesting and mismatched literal delimiters before TeX runs."""
    stack = []
    for match in _MATH_DELIMITER.finditer(text):
        token = match.group()
        operation, environment = match.groups()
        if environment in _DISPLAY_ENVIRONMENTS:
            if operation == "begin":
                if stack or implicit_display:
                    raise ArticleError("nested_tex_display_math")
                stack.append(environment)
            elif not stack or stack.pop() != environment:
                raise ArticleError("unbalanced_tex_math_delimiter")
        elif token in (r"\[", "$$"):
            if token == "$$" and stack and stack[-1] == "$$":
                stack.pop()
            elif token == "$$" and stack and stack[-1] == "$":
                # TeX can close one inline formula and open the next with adjacent dollars.
                continue
            elif stack or implicit_display:
                raise ArticleError("nested_tex_display_math")
            else:
                stack.append(r"\]" if token == r"\[" else token)
        elif token in (r"\]", r"\)"):
            if not stack or stack.pop() != token:
                raise ArticleError("unbalanced_tex_math_delimiter")
        elif token == r"\(":
            stack.append(r"\)")
        elif token == "$":
            if stack and stack[-1] == "$":
                stack.pop()
            else:
                stack.append(token)
    if stack:
        raise ArticleError("unbalanced_tex_math_delimiter")


def _validate_inputs(article: dict, evidence: dict, ledger: dict) -> None:
    validate_article(article)
    ev.validate(evidence)
    references.validate(ledger, evidence)
    if article["evidence_digest"] != evidence["digest"] or article["reference_digest"] != digest(ledger):
        raise ArticleError("article_input_binding_mismatch")


def _format_coverage(article: dict, evidence: dict, ledger: dict, outputs: dict | None = None) -> dict:
    _validate_inputs(article, evidence, ledger)
    _check_math_delimiters(article["title"]["text"])
    for block in render.walk_blocks(article):
        if any(not references.usable_check(ledger, cid) for cid in block["citations"]):
            raise ArticleError("unverified_article_citation: block=" + block["id"])
        if block["kind"] == "subsection":
            continue
        try:
            _check_math_delimiters(block["text"], implicit_display=block["kind"] == "equation")
        except ArticleError as exc:
            raise ArticleError(f"{exc}: block={block['id']}") from exc
    checked = writing.structural_coverage(article, evidence, ledger)
    # The renderer is the authority for resolvable citations, labels, and output schema.
    rendered = render.render(article, ledger)
    if outputs is not None and rendered != outputs:
        raise ArticleError("article_rendered_output_mismatch")
    return dict(checked, checks={"format": True, "coverage": checked["draft_complete"]})


def deterministic(article: dict, evidence: dict, ledger: dict, *, review_mode: str = STRICT) -> dict:
    if not isinstance(review_mode, str) or review_mode not in REVIEW_MODES:
        raise ArticleError("invalid_article_review_mode")
    if review_mode == FORMAT_COVERAGE:
        return _format_coverage(article, evidence, ledger)
    _validate_inputs(article, evidence, ledger)
    context_ids = {b["id"] for section in article["sections"][:2] for b in section["blocks"] if b["kind"] == "paragraph"}
    for block in render.walk_blocks(article):
        if block["id"] not in context_ids and any(ledger["checks"].get(cid, {}).get("evidence_level") == "background_reference"
                                                  for cid in block["citations"]):
            raise ArticleError("background_citation_cannot_support_proof_or_result: block=" + block["id"]
                + "; keep the mathematical proof unchanged and move its neutral background/credit sentence and citations "
                "to a top-level paragraph in Introduction or Preliminaries, immediately before the relevant tool. "
                "Do not delete valid attribution or claim this metadata verifies the theorem.")
    if writing.revision(article):
        return writing.coverage(article, evidence, ledger)
    source = ev.units(evidence)
    blocks = list(render.walk_blocks(article))
    by_id = {b["id"]: b for b in blocks}
    labels = {s["id"] for s in article["sections"]} | {
        b["id"] for b in blocks if b["kind"] in render._NUMBERED or b["kind"] == "subsection"}
    if set(article["title"]["evidence"]) - source.keys():
        raise ArticleError("unknown_title_evidence")
    mapping = {uid: [] for uid in source}
    proof_coverage, statements, used = set(), {}, set()
    for b in blocks:
        if set(b["evidence"]) - source.keys() or set(b["crossrefs"]) - labels:
            raise ArticleError("unknown_article_evidence_or_cross_reference")
        if any(not references.usable_check(ledger, cid) for cid in b["citations"]):
            raise ArticleError("unverified_article_citation")
        used.update(b["citations"])
        for uid in b["evidence"]:
            mapping[uid].append(b["id"])
        if b["kind"] in ("theorem", "lemma", "fact", "corollary"):
            bound = [uid for uid in b["evidence"] if source[uid]["kind"] == "statement"]
            if len(bound) != 1:
                raise ArticleError("new_or_ambiguous_article_theorem")
            statements[b["id"]] = bound[0]
    proofs = {}
    for b in blocks:
        if b["kind"] != "proof":
            continue
        target = b["proves"]
        if target not in statements or target in proofs:
            raise ArticleError("missing_or_duplicate_proof_target")
        fid = source[statements[target]]["fact_id"]
        checked = {uid for p in [b, *b["parts"]] for uid in p["evidence"]
                   if source[uid]["kind"] == "proof" and source[uid]["fact_id"] == fid}
        proof_coverage.update(checked)
        proofs[target] = b["id"]
    missing = []
    for uid, unit in source.items():
        if unit["kind"] == "proof" and uid not in proof_coverage:
            missing.append(uid)
        elif unit["kind"] == "statement" and uid not in statements.values():
            missing.append(uid)
        elif unit["kind"] in ("definition", "bridge") and not mapping[uid]:
            missing.append(uid)
    for sid in statements:
        if sid not in proofs:
            missing.append("proof-of:" + sid)
    intro_ids = {b["id"] for b in article["sections"][0]["blocks"]}
    for fid in evidence["targets"]:
        if not any(sid in intro_ids and uid == fid + ":statement" for sid, uid in statements.items()):
            missing.append("principal-result:" + fid)
    coverage = references.coverage(ledger, used)
    return {"passed": not missing and coverage["complete"], "draft_complete": not missing,
            "missing_evidence": sorted(missing), "citation_coverage": coverage, "evidence_to_blocks": mapping,
            "statement_to_proof": proofs, "formal_certificate": False}


def expected_coverage(article: dict, evidence: dict, ledger: dict, outputs: dict,
                      job_id: str, block_ids: list[str], *, indexed: bool = False,
                      partition: list[list[str]] | None = None) -> dict:
    by_id = {b["id"]: b for b in render.walk_blocks(article)}
    by_id["title"] = {"evidence": article["title"]["evidence"], "citations": []}
    required_units = sorted({u for bid in block_ids for u in by_id[bid]["evidence"]})
    citations = sorted({c for bid in block_ids for c in by_id[bid]["citations"]})
    editorial = {}
    if ledger.get("focus"):
        editorial["references_revision"] = ledger["focus"]["revision"]
    if writing.revision(article):
        if job_id == "whole":
            required_units = sorted(ev.units(evidence))
        editorial.update(writing_revision=writing.REVISION, source_map_digest=digest(article["source_map"]))
    if indexed:
        if job_id != "whole":
            raise ArticleError("indexed_review_requires_whole_manuscript")
        editorial["material_scope"] = "indexed"
    if partition is not None:
        if job_id != "whole":
            raise ArticleError("partition_receipt_requires_whole_manuscript")
        editorial["partition_digest"] = digest(partition)
    return {**editorial, "policy_revision": references.policy.POLICY_REVISION, "review_revision": REVIEW_REVISION,
            "job_id": job_id, "article_digest": digest(article), "evidence_digest": evidence["digest"],
            "reference_digest": digest(ledger), "rendered_digest": digest(outputs),
            "checked_blocks": block_ids, "checked_units": required_units, "checked_citations": citations,
            "checked_obligations": sorted(ledger["obligations"]) if job_id == "whole" else sorted({
                ledger["checks"][c]["obligation_id"] for c in citations})}


def receipt_outcome(report: dict, expected: dict) -> str:
    checks = _CHECKS + (writing.REVIEW_CHECKS if expected.get("writing_revision") else ())
    require_keys(report, {"coverage", "verdict", "findings", *checks})
    coverage = require_keys(report["coverage"], set(expected) | {"materials_complete", "truncated", "uncovered"})
    if digest({k: coverage[k] for k in expected}) != digest(expected):
        raise ArticleError("article_audit_coverage_mismatch")
    if (type(coverage["materials_complete"]) is not bool or type(coverage["truncated"]) is not bool
            or not isinstance(coverage["uncovered"], list) or not isinstance(report["findings"], list)
            or report["verdict"] not in ("pass", "needs_review") or any(type(report[k]) is not bool for k in checks)):
        raise ArticleError("invalid_article_audit_receipt")
    advisory_checks, blockers = set(), []
    for finding in report["findings"]:
        require_keys(finding, {"location", "issue"}, {"severity", "check"})
        nonempty_text(finding["location"])
        nonempty_text(finding["issue"])
        severity = finding.get("severity", "blocking")
        if severity not in ("blocking", "advisory"):
            raise ArticleError("invalid_article_finding_severity")
        if severity == "advisory":
            if finding.get("check") == OPTIONAL_CITATION_CHECK:
                advisory_checks.add(finding["check"])
            elif finding.get("check") not in ADVISORY_CHECKS or finding["check"] not in checks:
                raise ArticleError("substantive_review_issue_cannot_be_advisory")
            else:
                advisory_checks.add(finding["check"])
        else:
            blockers.append(finding)
    failed_checks = [k for k in checks if not report[k] and k not in advisory_checks]
    if (coverage["materials_complete"] is not True or coverage["truncated"] is not False
            or coverage["uncovered"] or blockers or failed_checks
            or report["verdict"] != "pass"):
        return "needs_review"
    return "passed"


def review_jobs(article: dict, *, partition: list[list[str]] | None = None) -> list[dict]:
    blocks = list(render.walk_blocks(article))
    whole = ["title", *[b["id"] for b in blocks]]
    if partition is not None:
        if (not isinstance(partition, list) or not partition
                or any(not isinstance(part, list) or not part for part in partition)
                or [bid for part in partition for bid in part] != whole):
            raise ArticleError("invalid_article_review_partition")
        return [{"id": f"part-{i}", "blocks": part} for i, part in enumerate(partition, 1)] + [
            {"id": "whole", "blocks": whole}]
    jobs, group, size = [], ["title"], len(article["title"]["text"])
    for block in blocks:
        if group and size + len(block["text"]) > 24000:
            jobs.append({"id": "part-" + str(len(jobs) + 1), "blocks": group})
            group, size = [], 0
        group.append(block["id"])
        size += len(block["text"])
    if group:
        jobs.append({"id": "part-" + str(len(jobs) + 1), "blocks": group})
    jobs.append({"id": "whole", "blocks": whole})
    return jobs


def finalize(article: dict, evidence: dict, ledger: dict, outputs: dict, receipts: dict,
             writer_sessions: list[str], *, review_mode: str = STRICT) -> dict:
    if review_mode == FORMAT_COVERAGE:
        if receipts:
            raise ArticleError("unexpected_model_receipts_for_format_coverage")
        if not isinstance(outputs, dict):
            raise ArticleError("article_rendered_output_mismatch")
        checked = _format_coverage(article, evidence, ledger, outputs)
        editorial = {"writing_revision": checked["writing_revision"]} if "writing_revision" in checked else {}
        return {**editorial, "format": FORMAT, "policy_revision": references.policy.POLICY_REVISION,
                "review_revision": REVIEW_REVISION, "review_mode": FORMAT_COVERAGE,
                "article_digest": digest(article), "evidence_digest": evidence["digest"],
                "reference_digest": digest(ledger), "rendered_digest": digest(outputs),
                "deterministic": checked, "checks": checked["checks"], "manuscript_available": True,
                "draft_complete": checked["draft_complete"], "warnings": [],
                "advisory_candidates": checked["citation_coverage"].get("advisory_candidates", []),
                "citation_warnings": checked["citation_coverage"]["advisory_missing"], "receipts": {},
                "failures": [] if checked["passed"] else ["incomplete_original_source_coverage"],
                "formal_certificate": False, "state": "ready" if checked["passed"] else "needs_review",
                "meaning_of_ready": "Readiness is based on deterministic checks of manuscript format and structural source coverage."}
    checked = deterministic(article, evidence, ledger, review_mode=review_mode)
    partition = receipts.get("whole", {}).get("partition")
    expected_jobs = review_jobs(article, partition=partition)
    failures, seen = [], set(writer_sessions)
    if set(receipts) != {j["id"] for j in expected_jobs}:
        failures.append("missing_or_extra_audit_jobs")
    for job in expected_jobs:
        receipt = receipts.get(job["id"], {})
        session = receipt.get("session_id")
        if not isinstance(session, str) or not session or session in seen:
            failures.append("audit_session_not_independent:" + job["id"])
            continue
        seen.add(session)
        scope = receipt.get("material_scope")
        if scope not in (None, "indexed") or (scope and job["id"] != "whole"):
            failures.append("invalid_audit_scope:" + job["id"])
            continue
        expected = expected_coverage(article, evidence, ledger, outputs, job["id"], job["blocks"],
                                     indexed=scope == "indexed", partition=partition if job["id"] == "whole" else None)
        try:
            if receipt_outcome(receipt.get("report"), expected) != "passed":
                failures.append("audit_needs_review:" + job["id"])
        except (ValueError, KeyError, TypeError):
            failures.append("invalid_audit_receipt:" + job["id"])
    editorial = {"writing_revision": writing.REVISION} if writing.revision(article) else {}
    warnings = [{"job": key, **finding} for key, receipt in receipts.items()
                for finding in receipt.get("report", {}).get("findings", []) if finding.get("severity") == "advisory"]
    return {**editorial, "format": FORMAT, "policy_revision": references.policy.POLICY_REVISION,
            "review_revision": REVIEW_REVISION, "warnings": warnings,
            "advisory_candidates": checked["citation_coverage"].get("advisory_candidates", []),
            "article_digest": digest(article), "evidence_digest": evidence["digest"],
            "reference_digest": digest(ledger), "rendered_digest": digest(outputs), "deterministic": checked,
            "manuscript_available": True, "draft_complete": checked["draft_complete"],
            "citation_warnings": checked["citation_coverage"]["advisory_missing"]
                + [row.get("reference_id", row.get("reason", "unresolved lead")) for row in
                   checked["citation_coverage"]["advisory_candidates"]],
            "receipts": receipts, "failures": failures, "formal_certificate": False,
            "state": "ready" if checked["passed"] and not failures else "needs_review",
            "meaning_of_ready": "Automated editorial checks passed; qualified human review is still required."}
