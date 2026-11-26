"""Validate model contracts before expensive downstream stages or cache reuse."""
from __future__ import annotations

import copy

from . import audit, evidence, literature, references, render, writing_policy as writing
from .models import FORMAT, ArticleError, IDENTIFIER, digest, nonempty_text, require_keys, string_list

RECOVERY_REVISION = 5


def bind_article_metadata(value: dict, material: dict) -> tuple[dict, list[dict]]:
    if not isinstance(value.get("article"), dict):
        return value, []
    frozen, ledger = material["evidence"], material["ledger"]
    evidence.validate(frozen)
    references.validate(ledger, frozen)
    expected = {"format": FORMAT, "evidence_digest": frozen["digest"],
                "reference_digest": digest(ledger), "writing_revision": writing.REVISION}
    if material.get("publication_plan") is not None:
        expected["publication_plan_digest"] = digest(material["publication_plan"])
    changes = [{"field": key, "present": key in value["article"],
                "returned": value["article"].get(key), "bound": bound}
               for key, bound in expected.items()
               if key not in value["article"] or value["article"][key] != bound
               or type(value["article"][key]) is not type(bound)]
    fixed = copy.deepcopy(value)
    fixed["article"].update(expected)
    return fixed, changes


def validate(task: str, value: dict, material: dict) -> None:
    if task in ("detect_proof_gaps", "propose_proof_repair", "verify_proof_repair"):
        from . import proof_repair
        validator = {"detect_proof_gaps": proof_repair.validate_detection,
                     "propose_proof_repair": proof_repair.validate_proposal,
                     "verify_proof_repair": proof_repair.validate_verification}[task]
        validator(value, material)
    elif task == "revise_article_blocks":
        from . import revision_patch
        revision_patch.apply(value, material)
    elif task == "repair_article_fields":
        from . import field_recovery
        field_recovery.apply(material["response"], material["fields"], value)
    elif task == "plan_review_references":
        from .improvement import validate_plan
        validate_plan(value, material)
    elif task == "plan_publication":
        from .publication import validate_plan
        validate_plan(value, material)
    elif task == "plan_references":
        literature.validate_plan(value, material)
    elif task == "select_focused_references":
        literature.validate_selection(value, material)
    elif task == "insert_citations":
        article = literature.apply_citation_patch(material["article"], material["ledger"], value)
        audit.validate_article(article)
        render.render(article, material["ledger"])
    elif task == "standard_reference":
        require_keys(value, {"checks"})
        if not isinstance(value["checks"], list):
            raise ArticleError("standard_reference_checks_must_be_list")
        known = {row["id"] for row in material["obligations"]}
        for row in value["checks"]:
            require_keys(row, {"obligation_id", "reference_id", "rationale", "standard_result", "supported"})
            if (not isinstance(row["obligation_id"], str) or row["obligation_id"] not in known
                    or row["reference_id"] != material["reference_id"]):
                raise ArticleError("standard_reference_check_out_of_scope")
        references.apply_standard_checks(copy.deepcopy(material["ledger"]), value["checks"])
    elif task == "background_references":
        require_keys(value, {"checks"})
        known = {row["id"]: row for row in material["obligations"]}
        if not isinstance(value["checks"], list):
            raise ArticleError("background_checks_must_be_array")
        for row in value["checks"]:
            require_keys(row, {"obligation_id", "reference_id", "rationale", "supported"})
            if (row["obligation_id"] not in known or row["reference_id"] not in material["references"]
                    or row["reference_id"] not in known[row["obligation_id"]]["reference_ids"]
                    or type(row["supported"]) is not bool):
                raise ArticleError("background_reference_out_of_scope")
            nonempty_text(row["rationale"])
    elif task == "citation_inventory":
        require_keys(value, {"checked_units", "materials_complete", "truncated", "candidates", "obligations",
                             "queries", "exclusions", "proof_summaries"})
        if (value["checked_units"] != [u["id"] for u in material["units"]]
                or value["materials_complete"] is not True or value["truncated"] is not False):
            raise ArticleError("article_citation_inventory_incomplete")
        for key in ("candidates", "obligations", "queries", "exclusions", "proof_summaries"):
            if not isinstance(value[key], list):
                raise ArticleError("article_inventory_requires_arrays")
    elif task == "select_related_literature":
        require_keys(value, {"checked_results", "selected", "discarded", "unresolved", "materials_complete", "truncated"})
        if set(string_list(value["checked_results"])) != set(material["results"]):
            raise ArticleError("incomplete_literature_selection: account for every supplied result")
        if value["materials_complete"] is not True or value["truncated"] is not False:
            raise ArticleError("incomplete_literature_selection: materials_complete concerns only the supplied search metadata; "
                               "missing full publications are unresolved citations, not omitted input. Do not claim unseen text was read")
        known = {u["id"] for u in material["units"]}
        handled = []
        for category in ("selected", "discarded", "unresolved"):
            if not isinstance(value[category], list):
                raise ArticleError("literature_selection_requires_arrays")
            for choice in value[category]:
                required = {"arxiv_id", "reason"} | ({"source_units", "kind"} if category == "selected" else set())
                require_keys(choice, required)
                aid = choice["arxiv_id"]
                if aid not in material["results"] or aid in handled:
                    raise ArticleError("untraceable_literature_selection")
                handled.append(aid)
                nonempty_text(choice["reason"])
                if category == "selected":
                    if (set(string_list(choice["source_units"], nonempty=True)) - known
                            or choice["kind"] not in ("mathematical", "historical", "comparison", "background")):
                        raise ArticleError("unknown_literature_selection_evidence")
        if set(handled) != set(material["results"]):
            raise ArticleError("unaccounted_literature_candidates")
    elif task == "classify_source":
        require_keys(value, {"source_id", "source_sha256", "reference_id", "document_type", "identity_supported",
                             "body_quote", "locator", "outside_abstract"})
        if any(value[key] != material[key] for key in ("source_id", "source_sha256", "reference_id")):
            raise ArticleError("source_identity_or_type_mismatch")
        if (value["document_type"] not in ("fulltext", "abstract", "metadata", "unknown")
                or type(value["identity_supported"]) is not bool or type(value["outside_abstract"]) is not bool):
            raise ArticleError("invalid_source_classification")
        if value["document_type"] == "fulltext" and value["identity_supported"] and value["outside_abstract"]:
            if references.source_kind(material["text"], material["origin"]) == "metadata":
                raise ArticleError("metadata_cannot_be_original_body")
            for key in ("body_quote", "locator"):
                references.source_span(material["text"], value[key])
    elif task == "rewrite_proof":
        require_keys(value, {"block"})
        block = value["block"]
        audit._block(block, nested=True)
        if (block["id"] != material["block_id"] or material["unit"]["id"] not in block["evidence"]
                or block["kind"] not in ("paragraph", "step")):
            raise ArticleError("rewritten_proof_unit_not_fully_mapped")
    elif task == "write_front":
        require_keys(value, {"title", "abstract", "keywords", "msc", "sections"})
        require_keys(value["title"], {"text", "evidence"})
        render.safe_tex(value["title"]["text"])
        string_list(value["title"]["evidence"], nonempty=True)
        string_list(value["keywords"], nonempty=True)
        string_list(value["msc"])
        audit._block(value["abstract"])
        if value["abstract"]["id"] != "abstract" or value["abstract"]["kind"] != "paragraph":
            raise ArticleError("invalid_article_abstract")
        sections = value["sections"]
        if (not isinstance(sections, list) or [s.get("role") for s in sections]
                != ["introduction", "preliminaries", "core", "main_proofs"]):
            raise ArticleError("article_section_style_mismatch")
        known = {u["id"] for u in material["source_units"]}
        ids, found = {"title", "abstract"}, {}
        expected = material["statement_ids"]
        for section in sections:
            require_keys(section, {"id", "role", "title", "blocks"})
            sid = section["id"]
            if not isinstance(sid, str) or not IDENTIFIER.fullmatch(sid) or sid in ids:
                raise ArticleError("invalid_or_duplicate_section_id")
            ids.add(sid)
            canonical = material["style"]["section_titles"].get(section["role"])
            if canonical and canonical != section["title"]:
                raise ArticleError("article_section_title_mismatch")
            for block in section["blocks"]:
                audit._block(block)
                if block["kind"] == "proof" or block["id"] in ids or block["id"].startswith(("proof-", "part-")):
                    raise ArticleError("reserved_or_duplicate_article_block_id")
                ids.add(block["id"])
                if set(block["evidence"]) - known:
                    raise ArticleError("unknown_article_evidence")
                if block["kind"] in ("theorem", "lemma", "fact", "corollary"):
                    bound = [fid for fid, bid in expected.items() if bid == block["id"] and fid + ":statement" in block["evidence"]]
                    if len(bound) != 1 or bound[0] in found:
                        raise ArticleError("article_statement_id_mismatch")
                    if (material.get("review_mode", audit.STRICT) == audit.STRICT
                            and bound[0] in material["targets"] and section["role"] != "introduction"):
                        raise ArticleError("principal_result_must_be_in_introduction")
                    found[bound[0]] = block["id"]
        if found != expected:
            raise ArticleError("missing_verified_statement_in_front")
    elif task.startswith("review_"):
        audit.receipt_outcome(value, material["expected_coverage"])
    elif task == "check_reference":
        require_keys(value, {"checked_source", "materials_complete", "truncated", "checks"})
        if (value["checked_source"] != material["expected_source"] or value["materials_complete"] is not True
                or value["truncated"] is not False or not isinstance(value["checks"], list)):
            raise ArticleError("incomplete_reference_content_check")
        for check in value["checks"]:
            require_keys(check, {"obligation_id", "reference_id", "source_id", "quote", "locator", "premise_comparison", "supported"})
            if (check["obligation_id"] != material["obligation"]["id"]
                    or check["reference_id"] != material["reference_id"]
                    or check["source_id"] != material["expected_source"]["source_id"]
                    or type(check["supported"]) is not bool):
                raise ArticleError("invalid_reference_content_binding")
            nonempty_text(check["premise_comparison"])
            for key in ("quote", "locator"):
                references.source_span(material["source_text"], check[key])
    elif task == "compose_paper" or (task == "revise_article" and material.get("writing_revision")):
        require_keys(value, {"article"})
        article = value["article"]
        review_mode = material.get("review_mode", audit.STRICT)
        if writing.revision(article) != writing.REVISION:
            raise ArticleError("publication_writing_revision_required")
        checked = audit.deterministic(article, material["evidence"], material["ledger"], review_mode=review_mode)
        if not checked["draft_complete"]:
            raise ArticleError("publication_source_coverage_incomplete: " + ",".join(checked["missing_evidence"]))
        if review_mode == audit.STRICT and checked["editorial_metrics"]["duplicate_substantive_passages"]:
            raise ArticleError("duplicate_substantive_passages_require_consolidation")
        if material.get("publication_plan") is not None:
            from . import publication
            if article.get("publication_plan_digest") != digest(material["publication_plan"]):
                raise ArticleError("article_publication_plan_binding_mismatch")
            if review_mode == audit.STRICT:
                publication.validate_article(article, material["publication_plan"])
                publication.check_internal_ids(article, material["evidence"])
        render.render(article, material["ledger"])
    elif task == "revise_article":
        require_keys(value, {"article"})
        audit.validate_article(value["article"])
