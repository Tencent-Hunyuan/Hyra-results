"""Read-only notebook dashboard. Statements load by default; proofs load on demand."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hyprover.core import KnowledgeSnapshot, KnowledgeStore
from hyprover.core.knowledge_markdown import display_name, prepare_display

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
CHANNELS = [("overview", "strategy"), ("route", "judgment"), ("conjecture", "hypothesis"),
            ("experience", "judgment"), ("lead", "judgment"), ("feedback", "verify"),
            ("report", "non-factual report")]
_CHANNEL_KINDS = {k for k, _ in CHANNELS}


def _project_dir() -> Path:
    p = os.environ.get("HYPROVER_DASHBOARD_PROJECT") or os.environ.get("HYPROVER_PROJECT_DIR")
    if not p:
        raise RuntimeError("no project dir — set --project / HYPROVER_PROJECT_DIR")
    return Path(p)


def _load_facts(project: Path, snapshot: KnowledgeSnapshot | None = None) -> List[dict]:
    snap = snapshot or prepare_display(KnowledgeStore(project).snapshot())
    return [f.to_dict() for f in snap.active_facts().values()]


class Overview(BaseModel):
    project: str
    facts: int
    facts_with_predecessors: int
    facts_by_author: Dict[str, int]
    channel_counts: Dict[str, int]
    verdicts: Dict[str, int]
    updated_at: float


class GraphNode(BaseModel):
    id: str
    display_id: str
    title: str
    display_name: str
    author: str
    problem_id: str
    statement: str
    proof_path: str
    predecessors: List[str]
    depth: int


class GraphEdge(BaseModel):
    source: str
    target: str


class FactGraphResp(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    max_depth: int


class ChannelInfo(BaseModel):
    kind: str
    role: str
    count: int


class ChannelsResp(BaseModel):
    channels: List[ChannelInfo]


class ChannelResp(BaseModel):
    kind: str
    count: int
    entries: List[Dict[str, Any]]
    total: int
    offset: int
    limit: int
    has_more: bool
    snapshot_revision: int


def build_overview(project: Optional[Path] = None) -> Dict[str, Any]:
    project = project or _project_dir()
    snapshot = prepare_display(KnowledgeStore(project).snapshot())
    facts = _load_facts(project, snapshot)
    counts = {k: 0 for k, _ in CHANNELS}
    verdicts: dict[str, int] = {}
    for entry in snapshot.entries.values():
        if entry.status != "archived":
            counts[entry.kind] += 1
        if entry.kind == "feedback":
            if entry.attributes.get("operation") == "fact_review":
                status = "review_applied" if entry.attributes.get("applied") else "review_not_applied"
            else:
                status = "published" if entry.attributes.get("published") else "not_published"
            verdicts[status] = verdicts.get(status, 0) + 1
    by_author: dict[str, int] = {}
    for f in facts:
        by_author[f["author"]] = by_author.get(f["author"], 0) + 1
    return {"project": project.name, "facts": len(facts),
            "facts_with_predecessors": sum(bool(f["predecessors"]) for f in facts),
            "facts_by_author": by_author, "channel_counts": counts, "verdicts": verdicts,
            "updated_at": time.time()}


def build_factgraph(project: Optional[Path] = None) -> Dict[str, Any]:
    project = project or _project_dir()
    snapshot = prepare_display(KnowledgeStore(project).snapshot())
    nodes, edges, depths = [], [], {}
    for fid in snapshot.topological():
        fact = snapshot.facts[fid]
        depths[fid] = 1 + max((depths[pid] for pid in fact.predecessors), default=-1)
        nodes.append({"id": fid, "author": fact.author, "problem_id": fact.problem_id,
                      "display_id": fact.display_id, "title": fact.title, "display_name": display_name(fact),
                      "statement": fact.statement, "proof_path": fact.proof_path,
                      "predecessors": list(fact.predecessors), "depth": depths[fid]})
        edges.extend({"source": pid, "target": fid} for pid in fact.predecessors)
    return {"nodes": nodes, "edges": edges, "max_depth": max(depths.values(), default=0)}


def build_channels(project: Optional[Path] = None) -> Dict[str, Any]:
    project = project or _project_dir()
    snapshot = prepare_display(KnowledgeStore(project).snapshot())
    counts = {k: 0 for k, _ in CHANNELS}
    for entry in snapshot.entries.values():
        if entry.status != "archived":
            counts[entry.kind] += 1
    return {"channels": [{"kind": k, "role": r, "count": counts[k]} for k, r in CHANNELS]}


def build_channel(kind: str, project: Optional[Path] = None, *, query: str = "",
                  subtype: str | None = None, author: str | None = None,
                  linked_to: str | None = None, include_archived: bool = False,
                  sort_by: str = "updated", offset: int = 0, limit: int = 20,
                  expected_snapshot_revision: int | None = None) -> Dict[str, Any]:
    if kind not in _CHANNEL_KINDS:
        raise KeyError(kind)
    store = KnowledgeStore(project or _project_dir())
    snapshot = prepare_display(store.snapshot())
    page = store.search_page(query, limit=limit, kinds=[kind], subtype=subtype,
                             author=author, linked_to=linked_to, include_archived=include_archived,
                             sort_by=sort_by, offset=offset,
                             expected_snapshot_revision=expected_snapshot_revision, snapshot=snapshot)
    return {"kind": kind, "count": page["total"], "entries": page["results"],
            **{key: page[key] for key in ("total", "offset", "limit", "has_more", "snapshot_revision")}}


app = FastAPI(title="hyprover-observability", version="0.1.0")


@app.exception_handler(ValueError)
@app.exception_handler(OSError)
async def invalid_notebook(request: Request, exc: Exception) -> JSONResponse:
    status = 409 if str(exc) == "search_snapshot_conflict" else 400
    detail = "notebook_attachment_unavailable" if isinstance(exc, OSError) else str(exc)
    return JSONResponse({"detail": detail}, status_code=status)


@app.get("/api/overview", response_model=Overview)
def overview() -> JSONResponse:
    return JSONResponse(build_overview())


@app.get("/api/factgraph", response_model=FactGraphResp)
def factgraph() -> JSONResponse:
    return JSONResponse(build_factgraph())


@app.get("/api/fact/{fact_id}")
def fact_detail(fact_id: str) -> dict:
    store = KnowledgeStore(_project_dir())
    snapshot = prepare_display(store.snapshot())
    record = snapshot.active_facts().get(fact_id)
    if record is None:
        raise HTTPException(404, "unknown or revoked fact")
    fact = store.read_fact(fact_id, snapshot=snapshot)
    return {"id": fact.fact_id, "author": fact.author, "problem_id": fact.problem_id,
            "display_id": record.display_id, "title": record.title, "display_name": display_name(record),
            "statement": fact.statement, "proof": fact.proof, "intuition": fact.intuition,
            "predecessors": fact.predecessors,
            "supersession": record.to_dict()["supersession"],
            "preferred_fact_id": snapshot.preferred_fact_id(fact_id),
            "preferred_fact_label": display_name(snapshot.facts[snapshot.preferred_fact_id(fact_id)]),
            "predecessor_labels": {pid: display_name(snapshot.facts[pid]) for pid in fact.predecessors}}


@app.get("/api/channels", response_model=ChannelsResp)
def channels() -> JSONResponse:
    return JSONResponse(build_channels())


@app.get("/api/channel/{kind}", response_model=ChannelResp)
def channel(kind: str, query: str = "", subtype: str | None = None, author: str | None = None,
            linked_to: str | None = None, include_archived: bool = False, sort_by: str = "updated",
            offset: int = 0, limit: int = 20, expected_snapshot_revision: int | None = None) -> JSONResponse:
    try:
        return JSONResponse(build_channel(kind, query=query, subtype=subtype, author=author,
                            linked_to=linked_to, include_archived=include_archived, sort_by=sort_by,
                            offset=offset, limit=limit, expected_snapshot_revision=expected_snapshot_revision))
    except KeyError:
        raise HTTPException(404, f"unknown channel {kind}")


@app.get("/api/entry/{entry_id}")
def entry_detail(entry_id: str) -> dict:
    try:
        return KnowledgeStore(_project_dir()).entry_detail(entry_id)
    except KeyError:
        raise HTTPException(404, "unknown entry")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    ap = argparse.ArgumentParser(description="hyprover read-only research notebook dashboard.")
    ap.add_argument("--project", help="project dir (or set HYPROVER_PROJECT_DIR)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    args = ap.parse_args()
    if args.project:
        os.environ["HYPROVER_DASHBOARD_PROJECT"] = args.project
    project = _project_dir()
    if not project.is_dir():
        raise SystemExit(f"project dir not found: {project}")
    KnowledgeStore(project).snapshot()
    import uvicorn
    print(f"hyprover dashboard: http://{args.host}:{args.port}/  (project: {project})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
