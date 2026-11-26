#!/usr/bin/env python3
"""Role-gated notebook tools; only the verification service publishes facts.

Private local memory remains agent-owned. Shared reads/edits use KnowledgeStore,
and fact_submit forwards a candidate without accepting caller-provided verdicts.
Configuration and direct-call roles come from HYPROVER_* environment variables;
registered MCP tools bind the server role independently of their arguments.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hyprover._mcp import FastMCP, public_error_code, register_tool
from hyprover.core import KnowledgeStore
from hyprover.integrations import search as _arxiv_search

from .roles import tools_for

_PROJECT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_FACT_ID_RE = re.compile(r"[0-9a-f]{16}")
_CALL_ROLE: ContextVar[Optional[str]] = ContextVar("gateway_role", default=None)
_CALL_CONFIG: ContextVar[Optional[dict]] = ContextVar("gateway_config", default=None)
_CONFIG_KEYS = ("HYPROVER_AGENTS_ROOT", "HYPROVER_PROJECT_DIR", "HYPROVER_AUTHOR",
                "HYPROVER_PROBLEM_ID", "HYPROVER_VERIFY_URL", "HYPROVER_VERIFY_TIMEOUT",
                "HYPROVER_LITERATURE_SEARCH")


class SourceRef(BaseModel):
    """A conjecture's internal ID and entry revision from one knowledge_search result."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, revalidate_instances="always")
    id: str = Field(min_length=1, pattern=r"^\S+$", description="Conjecture internal ID, not its display label.")
    revision: int = Field(ge=1, description="Entry revision that was read, not the notebook snapshot revision.")


def _source_pair(source: Optional[SourceRef], source_id: Optional[str],
                 source_revision: Optional[int]) -> tuple[Optional[str], Optional[int]]:
    if source is not None and (source_id is not None or source_revision is not None):
        raise ValueError("source_arguments_conflict")
    if source is None:
        if source_id is None:
            if source_revision is not None:
                raise ValueError("source_revision_without_source")
            return None, None
        if type(source_revision) is not int or source_revision < 1:
            raise ValueError("source_requires_positive_integer_revision")
        source = {"id": source_id, "revision": source_revision}
    try:
        reference = SourceRef.model_validate(source)
    except ValidationError as exc:
        raise ValueError("invalid_source_reference") from exc
    return reference.id, reference.revision


def _config(name: str, default: str = "") -> str:
    captured = _CALL_CONFIG.get()
    return os.environ.get(name, default) if captured is None else captured.get(name, default)


def _author() -> str:
    return _config("HYPROVER_AUTHOR", "unknown")


def _real_directory(value: str) -> Path:
    supplied = Path(value)
    if (not supplied.is_absolute() or ".." in supplied.parts
            or any(part.is_symlink() for part in (supplied, *supplied.parents))):
        raise ValueError("invalid_or_symlink_project_path")
    return supplied.resolve(strict=True)


def _role() -> str:
    return _CALL_ROLE.get() or os.environ.get("HYPROVER_ROLE", "verifier")


def _require_tool(name: str) -> None:
    if name not in tools_for(_role()):
        raise PermissionError(f"role {_role()!r} cannot use {name}")


def _project(project: Optional[str] = None) -> Path:
    """Resolve an existing project, rejecting traversal and escaping symlinks."""
    agents_root = _config("HYPROVER_AGENTS_ROOT", "")
    project_dir = _config("HYPROVER_PROJECT_DIR", "")
    if project is not None:
        if not isinstance(project, str) or not _PROJECT_NAME_RE.fullmatch(project):
            raise RuntimeError(f"invalid project name: {project!r}")
        if _role() not in ("main", "all"):
            raise PermissionError("only main may select another project")
        if not agents_root:
            raise RuntimeError("HYPROVER_AGENTS_ROOT is not set; cannot resolve a project by name")
        root = _real_directory(agents_root)
        pdir = _real_directory(str(root / project))
        if pdir.parent != root:
            raise RuntimeError("project escapes HYPROVER_AGENTS_ROOT")
    else:
        if not project_dir:
            raise RuntimeError("HYPROVER_PROJECT_DIR is not set and no project was given")
        if not Path(project_dir).is_absolute():
            raise RuntimeError("HYPROVER_PROJECT_DIR must be absolute")
        pdir = _real_directory(project_dir)
        if agents_root and pdir.parent != _real_directory(agents_root):
            raise RuntimeError("project escapes HYPROVER_AGENTS_ROOT")
    if not pdir.is_dir():
        raise RuntimeError("project must be an existing directory")
    return pdir


def _knowledge(project: Optional[str] = None) -> KnowledgeStore:
    return KnowledgeStore(_project(project))


def _verify(payload: Dict[str, Any], *, operation: str = "fact") -> Dict[str, Any]:
    """POST to the verification service; never apply a verdict locally."""
    if operation not in ("fact", "review"):
        raise ValueError("invalid_verification_operation")
    verify_url = _config("HYPROVER_VERIFY_URL", "")
    if not verify_url:
        raise RuntimeError("HYPROVER_VERIFY_URL is not set (verify service not wired yet)")
    url = urlsplit(verify_url)
    if url.scheme not in ("http", "https") or not url.netloc or url.query or url.fragment:
        raise RuntimeError("HYPROVER_VERIFY_URL must be an HTTP endpoint without query or fragment")
    path = url.path.rstrip("/")
    if not path.endswith("/verify/fact"):
        path = path + "/fact" if path.endswith("/verify") else path + "/verify/fact"
    if operation == "review":
        path = path.removesuffix("/fact") + "/review"
    endpoint = urlunsplit((url.scheme, url.netloc, path, "", ""))
    try:
        timeout = int(_config("HYPROVER_VERIFY_TIMEOUT", "3600"))
        if timeout == 0:
            timeout = None
        elif timeout < 0:
            timeout = 3600
    except ValueError:
        timeout = 3600
    req = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def knowledge_record(
    kind: str,
    title: str,
    body: str,
    entry_id: Optional[str] = None,
    expected_revision: Optional[int] = None,
    status: Optional[str] = None,
    links: Optional[List[str]] = None,
    details: str = "",
    attributes: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Record non-factual overview, route, conjecture, experience, lead or report.

    Reports share progress, handoffs and audits; feedback remains service-only.
    Experiences require attributes.subtype: method (usage lessons), boundary
    (applicability limits) or review (concrete concerns, with attributes.audits).
    Audit items contain target_id, revision, scope, not_covered; link every target.
    Older baselines require historical=true. Recorded baselines cannot be rewritten.
    Only the report author may edit its content; main may archive others' reports.
    Updates require entry_id + expected_revision. Creation retries may reuse a
    request_id with the identical payload; it cannot be used on updates.
    Links are associations, never proof dependencies. Keep raw drafts private.
    """
    _require_tool("knowledge_record")
    if kind == "feedback":
        raise PermissionError("feedback is written only by the verification service")
    role = "main" if _role() == "all" else _role()
    entry = _knowledge(project).record(
        kind=kind, title=title, body=body, author=_author(), role=role,
        entry_id=entry_id, expected_revision=expected_revision, status=status,
        links=links, details=details, attributes=attributes, request_id=request_id,
    )
    return {"id": entry.entry_id, "entry_id": entry.entry_id, "kind": entry.kind,
            "display_id": entry.display_id, "title": entry.title, "is_fact": False,
            "revision": entry.revision, "status": entry.status}


def knowledge_search(query: str = "", limit: int = 10, kinds: Optional[List[str]] = None,
                     project: Optional[str] = None, include_superseded: bool = False,
                     subtype: Optional[str] = None, author: Optional[str] = None,
                     linked_to: Optional[str] = None, include_archived: bool = False,
                     sort_by: str = "updated", offset: int = 0,
                     expected_snapshot_revision: Optional[int] = None) -> Dict[str, Any]:
    """Browse shared summaries and judge their relevance yourself; no ranking or scores.

    Start with query="" to page through a category, not a natural-language search.
    A nonempty query is an exact ID/label lookup or a case-insensitive literal
    substring filter, not semantic retrieval. No hit does not mean no useful knowledge.
    Reports require kinds=["report"]; archived entries require include_archived.
    For experiences use subtype=method|boundary|review (no fourth subtype).
    linked_to accepts an internal ID or reading label. sort_by is updated (default)
    or created, newest first; legacy relevance is an alias for updated.
    Follow has_more/offset and expected_snapshot_revision; refresh on snapshot conflict.
    Read linked proof/note files only as needed. Private notes are never searched.
    Non-facts and audit warnings never certify mathematics. Conjecture results include
    source_ref={id, revision}; copy it unchanged into fact_submit(source=...). It uses
    the entry revision from this read, not snapshot_revision, and may become stale.
    """
    _require_tool("knowledge_search")
    page = _knowledge(project).search_page(
        query, limit=limit, kinds=kinds, include_superseded=include_superseded,
        subtype=subtype, author=author, linked_to=linked_to, include_archived=include_archived,
        sort_by=sort_by, offset=offset, expected_snapshot_revision=expected_snapshot_revision)
    for row in page["results"]:
        if row["kind"] == "conjecture":
            row["source_ref"] = {"id": row["id"], "revision": row["revision"]}
    return {"query": query, **page}


def fact_submit(
    statement: str,
    proof: str,
    predecessors: Optional[List[str]] = None,
    glossary_introduces: Optional[Dict[str, str]] = None,
    intuition: str = "",
    source_id: Optional[str] = None,
    external_refs: Optional[List[Dict[str, Any]]] = None,
    source_revision: Optional[int] = None,
    title: str = "",
    source: Optional[SourceRef] = None,
) -> Dict[str, Any]:
    """Ask the independent verifier to check and publish a candidate.

    To resolve a conjecture, read it with knowledge_search and copy its source_ref
    into source={id, revision}; BOTH fields are required. The statement must match
    that conjecture. For an independent lemma omit source and both legacy fields.
    Legacy source_id + source_revision remain supported as a complete pair, but
    must not be mixed with source. Versions are never inferred or refreshed.
    The gateway forwards the legacy HTTP pair, so existing verifiers need no restart.
    Only accepted=true AND published=true with a valid fact_id confirms publication.
    This gateway never writes facts or promotes sources itself.
    """
    _require_tool("fact_submit")
    try:
        source_id, source_revision = _source_pair(source, source_id, source_revision)
        result = _verify({
            "project_dir": str(_project()), "author": _author(),
            "problem_id": _config("HYPROVER_PROBLEM_ID") or None,
            "statement": statement, "proof": proof,
            "predecessors": predecessors or [],
            "glossary_introduces": glossary_introduces or {},
            "intuition": intuition, "source_id": source_id,
            "source_revision": source_revision, "external_refs": external_refs or [],
            "title": title,
        })
        if not isinstance(result, dict):
            raise RuntimeError(f"verify service returned a non-dict body ({type(result).__name__})")
        result = dict(result)
        fid = result.get("fact_id")
        confirmed = (result.get("accepted") is True and result.get("published") is True
                     and isinstance(fid, str) and bool(_FACT_ID_RE.fullmatch(fid)))
        if not confirmed:
            if result.get("accepted") is True or result.get("published") is True:
                result.setdefault("write_error", "verify service did not confirm a published fact")
            result.update(accepted=False, published=False, fact_id=None)
        return result
    except Exception as exc:
        error = public_error_code(exc, _PUBLIC_TOOL_ERRORS) if _CALL_ROLE.get() is not None else str(exc)
        return {"accepted": False, "published": False, "fact_id": None, "verdict": "error", "error": error}


def fact_review(fact_id: str, action: str, reason: str, expected_revision: int,
                replacement_id: Optional[str] = None, evidence_fact_ids: Optional[List[str]] = None,
                project: Optional[str] = None) -> Dict[str, Any]:
    """Request independent revoke/supersede review; never mutate facts locally.

    Give a precise defect or a full implication argument, not just a verdict.
    Supersession requires an already published, independent stronger fact.
    Only accepted=true AND applied=true confirms the lifecycle change.
    """
    _require_tool("fact_review")
    result = {"accepted": False, "applied": False, "fact_id": fact_id, "action": action,
              "replacement_id": replacement_id, "revoked": [], "verdict": "error"}
    try:
        store = _knowledge(project)
        target = store.snapshot().facts.get(fact_id)
        if not _FACT_ID_RE.fullmatch(fact_id) or not target:
            raise ValueError("invalid_or_unknown_fact_id")
        if action not in ("revoke", "supersede"):
            raise ValueError("invalid_review_action")
        response = _verify({"project_dir": str(store.root), "author": _author(),
                            "fact_id": fact_id, "action": action, "reason": reason,
                            "expected_revision": expected_revision, "replacement_id": replacement_id,
                            "evidence_fact_ids": evidence_fact_ids or []}, operation="review")
        if not isinstance(response, dict):
            raise ValueError("review_service_returned_non_object")
        result.update(response)
        confirmed = (response.get("accepted") is True and response.get("applied") is True
                     and response.get("fact_id") == fact_id and response.get("action") == action
                     and response.get("replacement_id") == replacement_id
                     and response.get("verdict") == "correct" and bool(response.get("run_id")))
        if confirmed:
            current = store.snapshot()
            revoked = response.get("revoked")
            if action == "revoke":
                confirmed = (isinstance(revoked, list) and fact_id in revoked
                             and all(isinstance(i, str) and i in current.facts and current.facts[i].status == "revoked"
                                     for i in revoked))
                metadata = current.facts[fact_id].revocation
            else:
                confirmed = (replacement_id in current.facts and current.facts[replacement_id].status == "verified"
                             and current.facts[fact_id].status == "verified"
                             and current.facts[fact_id].supersession.get("replacement_id") == replacement_id)
                metadata = current.facts[fact_id].supersession
            stored = metadata.get("review", {})
            confirmed = (confirmed and stored.get("run_id") == response["run_id"]
                         and stored.get("request_id") == response.get("request_id")
                         and stored.get("action") == action and stored.get("reason") == reason
                         and stored.get("expected_revisions", {}).get(fact_id) == expected_revision)
            if action == "revoke":
                confirmed = confirmed and set(revoked or []) == set(stored.get("revoked", ()))
        if not confirmed:
            result.update(accepted=False, applied=False, revoked=[])
            if response.get("applied") or response.get("accepted"):
                result["error"] = "unconfirmed_fact_review_commit"
    except Exception as exc:
        error = public_error_code(exc, _PUBLIC_TOOL_ERRORS) if _CALL_ROLE.get() is not None else str(exc)
        result.update(accepted=False, applied=False, revoked=[], error=error)
    return result


def fact_revoke(fact_id: str, reason: str, project: Optional[str] = None) -> Dict[str, Any]:
    """Main-only cascade revocation of a fact and its proof descendants."""
    _require_tool("fact_revoke")
    if not _FACT_ID_RE.fullmatch(fact_id):
        raise ValueError("invalid fact_id")
    return {"revoked": _knowledge(project).revoke(fact_id, reason=reason)}


def search_arxiv_theorems(query: str, num_results: int = 10) -> Dict[str, Any]:
    """Search arXiv paper metadata/abstracts, not theorem statements or verified facts.

    The historical tool name is retained for callers; no theorem text is returned.
    Use keywords or arXiv syntax such as ti:convex AND au:Gardner. Limit: 1–20.
    Read the source paper separately before relying on a theorem or its hypotheses.
    """
    if _config("HYPROVER_LITERATURE_SEARCH", "1") == "0":
        raise PermissionError("literature_search_disabled_for_this_run")
    _require_tool("search_arxiv_theorems")
    return _arxiv_search(query, num_results=num_results)


_TOOLS = {
    "knowledge_record": knowledge_record,
    "knowledge_search": knowledge_search,
    "fact_submit": fact_submit,
    "fact_revoke": fact_revoke,
    "fact_review": fact_review,
    "search_arxiv_theorems": search_arxiv_theorems,
}


def _bind_role(fn, role: str, config=None):
    captured = ({key: os.environ[key] for key in _CONFIG_KEYS if key in os.environ}
                if config is None else dict(config))
    @wraps(fn)
    def bound(*args, **kwargs):
        role_token = _CALL_ROLE.set(role)
        config_token = _CALL_CONFIG.set(captured)
        try:
            return fn(*args, **kwargs)
        finally:
            _CALL_CONFIG.reset(config_token)
            _CALL_ROLE.reset(role_token)
    return bound


_PUBLIC_TOOL_ERRORS = frozenset({
    "search_snapshot_conflict", "entry_revision_conflict", "document_revision_conflict",
    "record_request_conflict", "record_request_identity_conflict", "request_id_only_for_creation",
    "invalid_record_request_id", "invalid_expected_revision", "unexpected_entry_revision",
    "entry_ownership_conflict", "entry_identity_conflict", "role_cannot_record_kind",
    "resolved_conjecture_is_immutable", "illegal_state_promotion", "only_main_controls_routes",
    "invalid_entry_content", "invalid_entry_identity", "invalid_entry_links", "duplicate_entry_links",
    "reserved_entry_attributes", "unknown_entry", "unknown_link", "report_author_required",
    "invalid_search_query", "invalid_search_limit", "invalid_search_offset", "invalid_search_kind",
    "invalid_search_boolean", "invalid_search_sort", "invalid_search_filter", "invalid_search_revision",
    "unknown_link_filter", "subtype_filter_requires_experience", "invalid_experience_subtype",
    "entry_metadata_too_deep", "entry_metadata_keys_must_be_strings",
    "entry_metadata_must_be_finite_json", "entry_attributes_must_be_object",
    "metadata_must_be_finite_json", "json_metadata_keys_must_be_strings",
    "invalid_json_unicode", "external_refs_must_be_list",
    "audits_must_be_nonempty_list", "invalid_audit_fields", "unknown_audit_target",
    "duplicate_audit_target", "audit_target_requires_link", "invalid_audit_revision",
    "invalid_audit_historical", "invalid_audit_scope", "invalid_audit_not_covered",
    "audit_revision_conflict", "audit_baseline_is_immutable", "review_requires_audits",
    "subtype_only_for_experience", "audits_only_for_research_entries",
    "legacy_experience_requires_explicit_subtype", "legacy_reserved_attribute_conflict",
    "legacy_audits_require_explicit_baseline", "invalid_or_unknown_fact_id", "invalid_review_action",
    "invalid_source_reference", "source_arguments_conflict",
    "source_requires_positive_integer_revision", "source_revision_without_source",
})


def build_app(role: Optional[str] = None) -> FastMCP:
    """Expose and bind exactly the configured role's tools, failing closed."""
    app = FastMCP("hyprover-core")
    selected_role = role if role is not None else _role()
    config = {key: os.environ[key] for key in _CONFIG_KEYS if key in os.environ}
    for name in tools_for(selected_role):
        if name == "search_arxiv_theorems" and config.get("HYPROVER_LITERATURE_SEARCH", "1") == "0":
            continue
        register_tool(app, name, _bind_role(_TOOLS[name], selected_role, config),
                      public_errors=_PUBLIC_TOOL_ERRORS)
    return app
