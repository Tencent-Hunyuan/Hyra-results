"""Hash-bound editorial patches followed by full evidence and independent review."""
from __future__ import annotations

import copy

from . import audit, publication, render
from .models import ArticleError, digest, nonempty_text, require_keys, string_list

CONTRACT = """Return {patches,unaddressed}. patches is at most 24 objects
{block_id,before_digest,text,citations,crossrefs}. Use before_digest from block_versions.
Only replace the named block's text/citation/crossref fields to fix concrete supplied
blocking findings. Do not return a whole article, change IDs/kinds/evidence, weaken a
hypothesis/conclusion, remove a proof step, invent a source, or erase a valid credit.
For scope problems explicitly state the original parameter ranges in an appropriate
existing conventions paragraph. For over-specific background, state precisely what the
supplied source actually supports, preserving attribution and the original mathematics.
Reflow long equations using safe aligned environments, not reduced fonts/margins.
Do not repeatedly rewrite unaffected paragraphs for style preferences. You may preserve
existing long citation IDs or use the supplied short handles; never guess an identifier.
A fully proved classical step can retain its proof and an appropriate verified standard
reference; missing necessary external premises cannot be fixed by deleting the citation.
unaddressed=[{location,issue}] records findings needing unavailable evidence or structural
changes outside this patch protocol. Empty patches are valid; do not falsely claim a fix.
Every change will be fully validated and independently reviewed against original evidence.
"""


def apply(value: dict, material: dict) -> dict:
    require_keys(value, {"patches", "unaddressed"})
    if not isinstance(value["patches"], list) or len(value["patches"]) > 24 or not isinstance(value["unaddressed"], list):
        raise ArticleError("editorial_patch_requires_bounded_arrays")
    for issue in value["unaddressed"]:
        require_keys(issue, {"location", "issue"})
        nonempty_text(issue["location"])
        nonempty_text(issue["issue"])
    fixed = copy.deepcopy(material["article"])
    blocks = {b["id"]: b for b in render.walk_blocks(fixed)}
    seen = set()
    for patch in value["patches"]:
        require_keys(patch, {"block_id", "before_digest", "text", "citations", "crossrefs"})
        bid = patch["block_id"]
        if not isinstance(bid, str) or bid not in blocks or bid in seen:
            raise ArticleError("editorial_patch_unknown_or_duplicate_block")
        block = blocks[bid]
        if patch["before_digest"] != digest(block):
            raise ArticleError("editorial_patch_stale_block")
        seen.add(bid)
        nonempty_text(patch["text"])
        string_list(patch["citations"])
        string_list(patch["crossrefs"])
        block.update({k: patch[k] for k in ("text", "citations", "crossrefs")})
    checked = audit.deterministic(fixed, material["evidence"], material["ledger"])
    if not checked["draft_complete"]:
        raise ArticleError("editorial_patch_missing_original_evidence")
    if material.get("publication_plan"):
        publication.validate_article(fixed, material["publication_plan"])
    publication.check_internal_ids(fixed, material["evidence"])
    render.render(fixed, material["ledger"])
    return fixed
