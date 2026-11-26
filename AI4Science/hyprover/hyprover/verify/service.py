"""hyprover verify service — the sole write-gate's HTTP front.

    POST /verify {statement, proof} -> {verification_report, verdict, repair_hints}
    GET  /health                    -> {status: "ok", pid: <int>}

/verify validates its request schema, cold-starts a fresh codex verifier
(``launcher.run_codex_verification``), and returns a report without modifying
knowledge. ``/verify/fact`` independently
checks and publishes candidates; ``/verify/review`` independently checks reasons
for certification withdrawal or stronger-result supersession and applies them.
The service owns all commits; the verifier model only emits its report. It is an
LLM, NOT a formal proof assistant — see ``agents/contracts/verifier.md``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from hyprover.core import Fact, KnowledgeStore

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .launcher import _allocate_run_id, run_codex_verification


class VerifyRequest(BaseModel):
    statement: str = Field(..., min_length=1)
    proof: str = Field(..., min_length=1)

    @field_validator("statement", "proof")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace text")
        return value


app = FastAPI(title="hyprover verify service", version="0.1.0")


@app.get("/health")
async def health() -> Dict[str, Any]:
    # async on purpose: /health must not queue behind sync /verify threadpool
    # calls, so it responds in ~microseconds regardless of in-flight verifications.
    # `pid` self-identifies this instance: a health probe alone cannot tell OUR
    # verify from another deployment's verify holding the same port on a shared
    # host — callers match this pid against runtime/run/verify.pid to be sure.
    return {"status": "ok", "pid": os.getpid()}


@app.post("/verify")
def verify(request: VerifyRequest) -> Dict[str, Any]:
    run_id = _allocate_run_id(request.statement)
    return run_codex_verification(run_id=run_id, statement=request.statement, proof=request.proof)


class VerifyFactRequest(VerifyRequest):
    model_config = ConfigDict(extra="forbid")
    project_dir: str = Field(..., min_length=1)
    author: str = Field(..., min_length=1)
    problem_id: Optional[str] = None
    predecessors: List[str] = Field(default_factory=list)
    glossary_introduces: Dict[str, str] = Field(default_factory=dict)
    intuition: str = ""
    external_refs: List[Dict[str, Any]] = Field(default_factory=list)
    source_id: Optional[str] = None
    source_revision: Optional[int] = Field(default=None, ge=1, strict=True)
    title: str = Field(default="", max_length=160)


class ReviewFactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_dir: str = Field(..., min_length=1)
    author: str = Field(..., pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    fact_id: str = Field(..., pattern=r"^[0-9a-f]{16}$")
    action: Literal["revoke", "supersede"]
    reason: str = Field(..., min_length=1)
    expected_revision: int = Field(..., ge=1, strict=True)
    replacement_id: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    evidence_fact_ids: List[str] = Field(default_factory=list)


@app.post("/verify/review")
def review(request: ReviewFactRequest) -> Dict[str, Any]:
    from .review import review_fact
    try:
        store = _publication_store(request.project_dir)
        return review_fact(
            store, request.fact_id, action=request.action, reason=request.reason,
            expected_revision=request.expected_revision, replacement_id=request.replacement_id,
            evidence_fact_ids=request.evidence_fact_ids, author=request.author,
            verifier=run_codex_verification, allocator=_allocate_run_id,
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _publication_store(project_dir: str) -> KnowledgeStore:
    configured = os.environ.get("HYPROVER_AGENTS_ROOT")
    if not configured:
        raise HTTPException(503, "HYPROVER_AGENTS_ROOT must be configured for fact publication")
    root_path = Path(configured)
    supplied = Path(project_dir)
    for path in (root_path, supplied):
        if (not path.is_absolute() or ".." in path.parts
                or any(part.is_symlink() for part in (path, *path.parents))):
            raise HTTPException(400, "invalid or symlink project path")
    root = root_path.resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", supplied.name):
        raise HTTPException(400, "invalid project name")
    resolved = supplied.resolve(strict=True)
    if resolved.parent != root or not resolved.is_dir():
        raise HTTPException(400, "project is outside HYPROVER_AGENTS_ROOT")
    store = KnowledgeStore(resolved)
    store.snapshot()
    return store


@app.post("/verify/fact")
def verify_fact(request: VerifyFactRequest) -> Dict[str, Any]:
    from .publication import verify_and_publish
    try:
        store = _publication_store(request.project_dir)
        problem_id = store.snapshot().problem_id
        if request.problem_id is not None and request.problem_id != problem_id:
            raise HTTPException(400, "candidate_project_mismatch")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", request.author):
            raise HTTPException(400, "invalid candidate author")
        candidate = Fact(
            fact_id="", problem_id=problem_id, author=request.author,
            statement=request.statement, proof=request.proof, predecessors=request.predecessors,
            glossary_introduces=request.glossary_introduces, intuition=request.intuition,
            external_refs=request.external_refs, title=request.title,
        )
        return verify_and_publish(
            store, candidate, verifier=run_codex_verification, allocator=_allocate_run_id,
            source_id=request.source_id, source_revision=request.source_revision,
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
