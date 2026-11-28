"""Evidence-bound editorial planning derived from arXiv:2608.13536v2."""
from __future__ import annotations

import copy
import json

from . import evidence, render
from .models import ArticleError, IDENTIFIER, digest, nonempty_text, require_keys, safe_path, string_list, write_json

REVISION = 1
ASSEMBLY_BINDING_REVISION = 1
SUBSECTION_LIMITS = {"preliminaries": 2, "core": 3}
STYLE = """Reference style: arXiv:2608.13536v2, A solution to Banach's isometric conjecture.
Use its expository architecture, never its mathematical claims, bibliography or page count.
Introduction: begin with research context, the supported development of related work with
credit, and the concrete mathematical significance BEFORE formal definitions/theorems;
then precise problem and scope, evidenced prior results, exact main statements,
method-level comparison where verified, then a concrete roadmap. No invented historical
attribution and no vague 'related literature' paragraph pretending to establish novelty.
Preliminaries: first fix global conventions (ambient spaces, norms, measures, parameters),
then topic-specific subsections containing definitions and the exact tools used later.
Explain each tool's hypotheses and later use. Give a short proof from the original material
or a genuinely applicable checked reference; not a catalogue of unexplained theorem names.
The sample groups convex bodies, bundles and degree theory because its proof uses them:
choose the topics needed by THIS paper. Define notation before formal use, including in
main statements; do not defer necessary definitions to scattered proof steps. Avoid a
one-paragraph placeholder preliminaries section or re-proving tools in every later section.
Core section: state its technical objective, develop dependent constructions in named
subsections, pair auxiliary results with local proofs, and explain transitions and uses.
Final section: assemble the main arguments from the established tools, divide genuinely
long proofs into named steps, state where restrictive assumptions enter, and close every
principal result without repeating whole auxiliary proofs. Mathematical remarks only.
Use AMS-style numbered statements, selective numbered displays, consistent symbols and
cross-references, conventional numeric citations and an alphabetic bibliography. No audit
language, internal benchmark/fact IDs, source-map prose, arbitrary page quotas, or padding.
"""
PLAN_CONTRACT = """Return {checked_units,global_conventions,notation,preliminaries,core,main_proofs}.
checked_units lists every supplied original unit ID exactly once. global_conventions is
{text,evidence}; describe the actual ambient objects and assumptions, without adding any.
notation is [{symbol,meaning,evidence}], only nontrivial symbols really needed in this proof.
preliminaries is 1-2 broad topic groups {id,title,evidence,concepts,used_by,citation_obligations}.
concepts is a nonempty list of definitions or tools to explain; used_by is a nonempty list
of original units needing these concepts. citation_obligations contains existing obligation
IDs only (may be empty for internally established material). No generic 'basic facts' filler.
core is 1-3 broad technical groups {id,title,evidence,requires,goal}. Keep at most FIVE
subsections in the entire paper, only in Preliminaries and the core section. These are
upper limits, not quotas. Merge related definitions, estimates and constructions into a
coherent topic instead of a subsection per lemma, calculation or proof ingredient. Use
brief transitions inside each topic; preserve every substantive argument and attribution.
Do not bypass the limit with many tiny Step headings. requires lists preceding
preliminary/core group IDs only, in dependency order. Each goal is a specific contribution
to the main argument, not a claim of a new theorem. All topic IDs are distinct and will be
used literally as subsection block IDs. Titles are short plain text, not TeX commands.
main_proofs is [{target_id,evidence,uses,strategy}], exactly one per supplied target;
uses lists existing topic IDs. Assign EVERY supplied original unit, including statements,
definitions, all proof chunks AND completion:bridge when present, to at least one evidence
or used_by list. checked_units is only a read checklist, NOT an assignment of those units.
The completion bridge belongs to final assembly of the main results: include it in each
main_proofs evidence list. The caller may explicitly bind an otherwise unassigned bridge
there, but never invent a proof or waive a source unit. Read its actual mathematical content
and preserve the bridge's premises and final implications in the manuscript source_map;
project/run-status sentences and internal IDs must not be copied into publication prose.
All evidence/used_by entries are supplied unit IDs. All lists are arrays of distinct IDs
unless the list contains objects. This is an editorial plan, not manuscript text or
permission to discard proof details.
"""


def validate_topic_title(title: str) -> None:
    nonempty_text(title)
    if any(c in title for c in "\\$\r\n\t") or "[[" in title or "]]" in title:
        raise ArticleError("publication_topic_title_must_be_plain_text")


class PlanCoverageError(ArticleError):
    def __init__(self, missing_units: set[str], missing_targets: set[str]):
        self.missing_units = sorted(missing_units)
        self.missing_targets = sorted(missing_targets)
        super().__init__("publication_plan_incomplete_proof_route: " + json.dumps({
            "unassigned_units": self.missing_units, "missing_targets": self.missing_targets}, ensure_ascii=False)
            + "; assign the named units within existing topic groups or main_proofs, without adding subsections "
              "or removing original evidence. checked_units alone does not assign evidence.")


def validate_plan(value: dict, material: dict) -> None:
    require_keys(value, {"checked_units", "global_conventions", "notation", "preliminaries", "core", "main_proofs"})
    units = {u["id"] for u in material["original_units"]}
    checked = set(string_list(value["checked_units"]))
    if checked != units:
        raise ArticleError("publication_plan_missing_original_units: " + json.dumps({
            "missing": sorted(units - checked), "unknown": sorted(checked - units)}, ensure_ascii=False))
    covered = set()

    def check_units(values):
        found = set(string_list(values, nonempty=True))
        if found - units:
            raise ArticleError("publication_plan_unknown_evidence")
        covered.update(found)

    conventions = require_keys(value["global_conventions"], {"text", "evidence"})
    nonempty_text(conventions["text"])
    check_units(conventions["evidence"])
    if not isinstance(value["notation"], list) or len(value["notation"]) > 48:
        raise ArticleError("invalid_publication_notation_plan")
    symbols = set()
    for row in value["notation"]:
        require_keys(row, {"symbol", "meaning", "evidence"})
        symbol = nonempty_text(row["symbol"])
        if symbol in symbols:
            raise ArticleError("duplicate_publication_notation")
        symbols.add(symbol)
        nonempty_text(row["meaning"])
        check_units(row["evidence"])
    topics = set()
    limits = SUBSECTION_LIMITS if material.get("writing_revision", 0) >= 3 else {"preliminaries": 5, "core": 8}
    for role, maximum in limits.items():
        if not isinstance(value[role], list) or not 1 <= len(value[role]) <= maximum:
            raise ArticleError("publication_plan_requires_topic_groups: " + role + " must have 1-" + str(maximum)
                               + " broad groups; merge related topics without omitting content")
        for row in value[role]:
            required = {"id", "title", "evidence"} | ({"concepts", "used_by", "citation_obligations"}
                        if role == "preliminaries" else {"requires", "goal"})
            require_keys(row, required)
            if (not isinstance(row["id"], str) or not IDENTIFIER.fullmatch(row["id"])
                    or row["id"] in topics or row["id"] in ("title", "abstract", "intro", "prelim", "core", "proofs")):
                raise ArticleError("invalid_publication_topic_id")
            validate_topic_title(row["title"])
            check_units(row["evidence"])
            if role == "preliminaries":
                string_list(row["concepts"], nonempty=True)
                check_units(row["used_by"])
                if set(string_list(row["citation_obligations"])) - set(material["obligations"]):
                    raise ArticleError("publication_plan_unknown_citation_need")
            else:
                if set(string_list(row["requires"])) - topics:
                    raise ArticleError("publication_plan_forward_or_unknown_dependency")
                nonempty_text(row["goal"])
            topics.add(row["id"])
    targets = set()
    if not isinstance(value["main_proofs"], list):
        raise ArticleError("invalid_publication_main_proof_plan")
    for row in value["main_proofs"]:
        require_keys(row, {"target_id", "evidence", "uses", "strategy"})
        if row["target_id"] not in material["targets"] or row["target_id"] in targets:
            raise ArticleError("publication_plan_wrong_target")
        targets.add(row["target_id"])
        check_units(row["evidence"])
        if set(string_list(row["uses"], nonempty=True)) - topics:
            raise ArticleError("publication_plan_unknown_proof_tool")
        nonempty_text(row["strategy"])
    if targets != set(material["targets"]) or covered != units:
        raise PlanCoverageError(units - covered, set(material["targets"]) - targets)


def bind_completion_bridge(value: dict, material: dict) -> tuple[dict, list[dict]]:
    try:
        validate_plan(value, material)
        return value, []
    except PlanCoverageError as exc:
        units = {u["id"]: u for u in material["original_units"]}
        bridge = units.get("completion:bridge")
        if (exc.missing_targets or exc.missing_units != ["completion:bridge"] or not bridge
                or bridge["kind"] != "bridge" or bridge["fact_id"] is not None):
            raise
    fixed, changes = copy.deepcopy(value), []
    for index, proof in enumerate(fixed["main_proofs"]):
        proof["evidence"].append("completion:bridge")
        changes.append({"path": ["main_proofs", index, "evidence"], "target_id": proof["target_id"],
                        "added_unit": "completion:bridge", "source_sha256": bridge["sha256"],
                        "purpose": "Route the supplied completion bridge to final assembly; not a proof certificate."})
    validate_plan(fixed, material)
    return fixed, changes


def validate_article(article: dict, plan: dict) -> None:
    sections = {s["role"]: s for s in article["sections"]}
    if article.get("writing_revision", 0) >= 3:
        for role, section in sections.items():
            count = sum(b["kind"] == "subsection" for b in section["blocks"])
            if count > SUBSECTION_LIMITS.get(role, 0):
                raise ArticleError("publication_subsection_limit_exceeded: " + role
                                   + "; merge related topics while preserving the full argument and citations")
    for role in ("preliminaries", "core"):
        blocks = sections[role]["blocks"]
        if not blocks or blocks[0]["kind"] != "paragraph":
            raise ArticleError("publication_section_requires_conventions_or_objective: " + role)
        expected = [row["id"] for row in plan[role]]
        headings = [b for b in blocks if b["kind"] == "subsection"]
        if [b["id"] for b in headings] != expected:
            raise ArticleError("publication_subsections_do_not_follow_plan: " + role)
        if any(b["text"] != row["title"] for b, row in zip(headings, plan[role])):
            raise ArticleError("publication_subsection_title_mismatch")
        for index, block in enumerate(blocks):
            if block["kind"] == "subsection" and (index + 1 == len(blocks) or blocks[index + 1]["kind"] == "subsection"):
                raise ArticleError("empty_publication_topic")
    intro = sections["introduction"]["blocks"]
    if intro[-1]["kind"] != "paragraph" or not set(intro[-1]["crossrefs"]) >= {
            sections[role]["id"] for role in ("preliminaries", "core", "main_proofs")}:
        raise ArticleError("publication_introduction_requires_concrete_roadmap")
    if article.get("writing_revision", 0) >= 2:
        if len(intro) < 3 or any(b["kind"] != "paragraph" for b in intro[:2]):
            raise ArticleError("introduction_requires_background_and_motivation_before_formal_results")
        for section in article["sections"]:
            blocks = section["blocks"]
            if section["role"] == "main_proofs" and blocks[0]["kind"] != "paragraph":
                raise ArticleError("main_proofs_require_strategy_transition")
            for index, block in enumerate(blocks):
                if block["kind"] == "subsection" and (index + 1 == len(blocks) or blocks[index + 1]["kind"] != "paragraph"):
                    raise ArticleError("subsection_requires_explanatory_transition")
                if block["kind"] in ("theorem", "lemma", "fact", "corollary", "definition") and (
                        index == 0 or blocks[index - 1]["kind"] != "paragraph"):
                    raise ArticleError("formal_statement_requires_motivating_transition")
    for block in render.walk_blocks(article):
        if block["kind"] == "subsection":
            validate_topic_title(block["text"])
            if block["citations"] or block["crossrefs"] or not block["evidence"]:
                raise ArticleError("publication_heading_requires_evidence_without_markers")


def layout_severity(report: dict | None) -> str | None:
    if report is None or (report.get("compiled") is True and report.get("max_overfull_pt", 0) <= 2):
        return None
    return "advisory" if report.get("compiled") is True and report.get("max_overfull_pt", 0) <= 12 else "blocking"


def layout_findings(root, report: dict | None) -> list[dict]:
    severity = layout_severity(report)
    if severity is None:
        return []
    error = report.get("error") or "overfull_math_or_text_exceeds_2pt"
    excerpt = ""
    path = root / "compile.log"
    if report.get("log_path") == str(path) and not path.is_symlink() and path.is_file():
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 16000))
            excerpt = handle.read(16000).decode("utf-8", errors="replace")[-6000:]
    return [{"location": "compiled manuscript / TeX diagnostic", "severity": severity,
             "check": "typography_readable", "issue": error + "\n" + excerpt +
             "\nRepair the specific TeX or long display using safe aligned equations. Do not shrink fonts, "
             "change margins, drop mathematical content or fake a successful compilation."}]


def with_layout_findings(report: dict, findings: list[dict]) -> dict:
    blocking = [row for row in findings if row.get("severity", "blocking") != "advisory"]
    if not blocking:
        return report
    failures = [f for f in report.get("failures", []) if f != "typesetting_requires_revision"]
    result = dict(report, state="needs_review", failures=[*failures, "typesetting_requires_revision"],
                  typesetting_findings=blocking)
    if "checks" in report:
        result["checks"] = dict(report["checks"], format=False)
    return result


def review_layout(root, article: dict, outputs: dict, report: dict, compilation: dict | None) -> dict:
    if compilation is None:
        return report
    findings = layout_findings(root, compilation)
    write_json(safe_path(root, "layout-review.json"), {"article_digest": digest(article),
               "rendered_digest": digest(outputs), "findings": findings, "compiled": compilation.get("compiled")})
    return with_layout_findings(report, findings)


def check_internal_ids(article: dict, frozen: dict) -> None:
    import re
    identifiers = set(evidence.units(frozen)) | {f["fact_id"] for f in frozen["facts"]}
    for block in [article["title"], *render.walk_blocks(article)]:
        text = render._MARKER.sub("", block["text"])
        if re.search(r"\bHYM\d{3}\b|\bsource_map\b", text) or any(uid in text for uid in identifiers):
            raise ArticleError("internal_evidence_id_in_publication_prose")
