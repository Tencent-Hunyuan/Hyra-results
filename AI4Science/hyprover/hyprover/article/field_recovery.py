"""Bounded, hash-bound repairs to invalid manuscript fields, never whole drafts."""
from __future__ import annotations

import copy

from . import evidence, render
from .models import ArticleError, digest, require_keys

CONTRACT = """Return exactly {patches:[{path,before_digest,value}]}, one patch for EVERY
supplied field and no other fields. Echo each path and before_digest exactly. Change only
the invalid field, not the mathematics, assumptions, conclusions, evidence IDs or citations.
For kind, use only allowed_values; a block containing proves and proof parts has kind proof.
For text, repair JSON/LaTeX corruption against the supplied original evidence, preserving all
unaffected text. Do not merely remove an unknown control character if its intended meaning
is uncertain. All LaTeX backslashes must be escaped in JSON (e.g. \\frac -> \\\\frac).
No control characters except ordinary whitespace; no macros, files, tools or raw citations.
If a repair cannot be justified from the source, return {patches:[]} rather than invent it.
"""


def plan(value: dict, frozen: dict) -> list[dict]:
    article = value.get("article")
    if not isinstance(article, dict):
        return []
    units = evidence.units(frozen)
    blocks = [(article["title"], ["title"]), (article["abstract"], ["abstract"])]
    top = [(b, ["sections", i, "blocks", j]) for i, s in enumerate(article["sections"])
           for j, b in enumerate(s["blocks"])]
    statements = {b["id"]: b for b, _ in top if b.get("kind") in render._NUMBERED}
    blocks.extend(top)
    blocks.extend((p, path + ["parts", i]) for b, path in top for i, p in enumerate(b.get("parts", [])))
    fields = []
    for block, path in blocks:
        source = [units[uid] for uid in block.get("evidence", []) if uid in units]
        base = {"block_id": block.get("id", "title"), "source_units": source}
        if (len(path) == 4 and block.get("kind") != "proof" and block.get("parts")
                and block.get("proves") in statements):
            fields.append(dict(base, path=path + ["kind"], before_digest=digest(block["kind"]),
                value=block["kind"], allowed_values=["proof"], proves=statements[block["proves"]],
                error="A block with proves and nonempty parts must explicitly have kind proof."))
        try:
            render.safe_tex(block["text"])
        except ArticleError as exc:
            fields.append(dict(base, path=path + ["text"], before_digest=digest(block["text"]),
                               value=block["text"], error=str(exc)))
    if len(fields) > 48:
        raise ArticleError("article_field_repair_budget_exceeded")
    return fields


def apply(value: dict, fields: list[dict], response: dict) -> dict:
    require_keys(response, {"patches"})
    patches = response["patches"]
    if not isinstance(patches, list) or len(patches) != len(fields):
        raise ArticleError("incomplete_article_field_repairs")
    expected = {tuple(field["path"]): field for field in fields}
    if len(expected) != len(fields):
        raise ArticleError("duplicate_article_repair_target")
    fixed = copy.deepcopy(value)
    seen = set()
    for patch in patches:
        require_keys(patch, {"path", "before_digest", "value"})
        if not isinstance(patch["path"], list):
            raise ArticleError("invalid_article_repair_path")
        key = tuple(patch["path"])
        if key not in expected or key in seen:
            raise ArticleError("unrequested_article_field_repair")
        field = expected[key]
        seen.add(key)
        if patch["before_digest"] != field["before_digest"]:
            raise ArticleError("stale_article_field_repair")
        parent = fixed["article"]
        for component in field["path"][:-1]:
            parent = parent[component]
        leaf = field["path"][-1]
        if digest(parent[leaf]) != field["before_digest"]:
            raise ArticleError("article_repair_source_changed")
        if "allowed_values" in field:
            if patch["value"] not in field["allowed_values"]:
                raise ArticleError("invalid_article_repair_kind")
        else:
            render.safe_tex(patch["value"])
            for kind in ("cite", "ref"):
                if render.markers(patch["value"], kind) != render.markers(field["value"], kind):
                    raise ArticleError("article_field_repair_changed_reference_binding")
        parent[leaf] = patch["value"]
    return fixed
