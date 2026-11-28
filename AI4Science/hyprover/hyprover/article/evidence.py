"""Immutable article inputs; this module never writes to the knowledge store."""
from __future__ import annotations

import os
from pathlib import Path

from hyprover.core import KnowledgeStore, compute_fact_id, plain
from hyprover.execution.layout import math_computation_allowed
from hyprover.verify import completion_evidence

from .models import (FORMAT, ArticleConfig, ArticleError, EvidenceChanged, digest,
                     read_json, read_text, require_keys, safe_path, string_list,
                     text_hash, write_json)


def permissions(metadata: dict, environ: dict | None = None) -> dict:
    env = os.environ if environ is None else environ
    return {"research_mode": metadata.get("research_mode"),
            "allow_math_computation": math_computation_allowed(metadata=metadata),
            "literature_search": env.get("HYPROVER_LITERATURE_SEARCH", "1") != "0"}


def _completion(project: Path, result_path: str, completion_root: str | None,
                original_project: Path | None = None) -> tuple[dict, dict]:
    result = read_json(safe_path(project, result_path))
    if result.get("formal_certificate") is not False:
        raise ArticleError("article_requires_nonformal_completion_receipt")
    if result.get("state") != "completed":
        raise ArticleError("article_requires_committed_completion")
    run_id = result.get("full_revalidation_run")
    report_path = result.get("completion_report")
    if (not isinstance(run_id, str) or not run_id.startswith("completion-")
            or Path(run_id).name != run_id or not isinstance(report_path, str)):
        raise ArticleError("article_requires_committed_completion")
    archive_binding = None
    if safe_path(project, "export-manifest.json").exists():
        from .archive import validate as validate_archive
        root, manifest, archive_binding = validate_archive(project, result_path, result, completion_root)
    else:
        root = Path(completion_root) if completion_root else Path(report_path).parent
        if not root.is_absolute() or root.name != run_id or Path(report_path).name != "completion-report.json":
            raise ArticleError("invalid_article_completion_location")
        safe_path(root, "completion-manifest.json")
        manifest = completion_evidence.validate_completion(root, project, result=result,
                                                            original_project=original_project)
    if run_id != "completion-" + manifest["digest"]:
        raise ArticleError("article_completion_run_digest_mismatch")
    receipt = {"result_path": result_path, "root": str(root), "result": result,
               "manifest_digest": manifest["digest"], "archive": archive_binding,
               "report_digest": digest(read_json(root / "completion-report.json")),
               "progress_digest": digest(read_json(root / "completion-progress.json"))}
    return receipt, manifest


def capture(project: Path, targets: list[str] | None = None, *,
            config: ArticleConfig | None = None, result_path: str = "main/result.json",
            completion_root: str | None = None, environ: dict | None = None,
            original_project: Path | None = None) -> dict:
    config = config or ArticleConfig()
    project = Path(project).absolute()
    safe_path(project, "KNOWLEDGE.md")
    metadata_path = safe_path(project, "project.json")
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    completion, summary = None, ""
    if targets is None:
        completion, completed = _completion(project, result_path, completion_root, original_project)
        targets, summary = completed["targets"], completed["summary"]
    targets = sorted(string_list(targets, nonempty=True))
    store = KnowledgeStore(project)
    snapshot = store.snapshot()
    facts = []
    for fid in snapshot.topological(targets):
        record = snapshot.facts[fid]
        if (record.status != "verified" or record.verification.get("verdict") != "correct"
                or not record.verification.get("run_id")
                or any(record.verification.get(k) for k in ("errors", "critical_errors", "gaps"))):
            raise ArticleError("article_requires_published_verified_fact")
        fact = store.read_fact(fid, snapshot=snapshot)
        facts.append({"fact_id": fid, "problem_id": fact.problem_id,
                      "revision": record.revision, "statement": fact.statement,
                      "predecessors": list(fact.predecessors),
                      "definitions": plain(fact.glossary_introduces),
                      "verification": plain(record.verification),
                      "external_refs": plain(fact.external_refs),
                      "proof": fact.proof, "proof_sha256": text_hash(fact.proof),
                      "ranges": completion_evidence.ranges(fact.proof, config.chunk_chars)})
    problem_path = safe_path(project, "PROBLEM.md")
    problem = read_text(problem_path) if problem_path.exists() else ""
    result = {"format": FORMAT, "project_dir": str(project), "problem_id": snapshot.problem_id,
              "mode": "completed" if completion else "selected_facts", "problem": problem,
              "targets": targets, "summary": summary, "permissions": permissions(metadata, environ),
              "chunk_chars": config.chunk_chars, "facts": facts, "completion": completion,
              "formal_certificate": False}
    result["digest"] = digest(result)
    validate(result)
    return result


def validate(evidence: dict) -> None:
    require_keys(evidence, {"format", "project_dir", "problem_id", "mode", "problem", "targets",
                            "summary", "permissions", "chunk_chars", "facts", "completion",
                            "formal_certificate", "digest"})
    if (evidence["format"] != FORMAT or evidence["formal_certificate"] is not False
            or evidence["digest"] != digest({k: v for k, v in evidence.items() if k != "digest"})
            or evidence["mode"] not in ("completed", "selected_facts")):
        raise ArticleError("invalid_article_evidence")
    string_list(evidence["targets"], nonempty=True)
    policy = require_keys(evidence["permissions"], {"research_mode", "allow_math_computation", "literature_search"})
    if any(type(policy[k]) is not bool for k in ("allow_math_computation", "literature_search")):
        raise ArticleError("invalid_article_permissions")
    if type(evidence["chunk_chars"]) is not int or not 256 <= evidence["chunk_chars"] <= 64000:
        raise ArticleError("invalid_article_chunk_limit")
    if not isinstance(evidence["facts"], list) or not evidence["facts"]:
        raise ArticleError("missing_article_facts")
    seen = {}
    for fact in evidence["facts"]:
        require_keys(fact, {"fact_id", "problem_id", "revision", "statement", "predecessors",
                            "definitions", "verification", "external_refs", "proof", "proof_sha256", "ranges"})
        fid = fact["fact_id"]
        if fid in seen or any(p not in seen for p in fact["predecessors"]):
            raise ArticleError("invalid_article_dependency_order")
        if type(fact["revision"]) is not int or fact["revision"] < 1:
            raise ArticleError("invalid_article_fact_revision")
        if (fact["problem_id"] != evidence["problem_id"]
                or fid != compute_fact_id(problem_id=fact["problem_id"],
                                          predecessors=fact["predecessors"],
                                          glossary_introduces=fact["definitions"],
                                          statement=fact["statement"], proof=fact["proof"])):
            raise ArticleError("article_fact_content_address_mismatch")
        if (fact["proof_sha256"] != text_hash(fact["proof"])
                or fact["ranges"] != completion_evidence.ranges(fact["proof"], evidence["chunk_chars"])):
            raise ArticleError("article_proof_or_ranges_changed")
        seen[fid] = fact
    required, pending = set(), list(evidence["targets"])
    while pending:
        fid = pending.pop()
        if fid not in seen:
            raise ArticleError("missing_article_dependency")
        if fid not in required:
            required.add(fid)
            pending.extend(seen[fid]["predecessors"])
    if required != set(seen):
        raise ArticleError("article_closure_mismatch")
    if evidence["mode"] == "selected_facts" and (evidence["completion"] is not None or evidence["summary"]):
        raise ArticleError("manual_article_cannot_claim_project_completion")
    if evidence["mode"] == "completed" and not isinstance(evidence["completion"], dict):
        raise ArticleError("article_completion_receipt_missing")


def assert_current(evidence: dict, *, environ: dict | None = None, project: Path | None = None) -> None:
    validate(evidence)
    receipt = evidence["completion"]
    original = Path(evidence["project_dir"])
    project = original if project is None else Path(project).absolute()
    relocated = original != project
    receipt_root = Path(receipt["root"]) if receipt else None
    try:
        if relocated and receipt_root is not None:
            if receipt_root.is_relative_to(original):
                receipt_root = safe_path(project, str(receipt_root.relative_to(original)))
            else:
                # An external verification run can accompany the copied project here.
                attached = safe_path(project, "verify-runs/" + receipt_root.name)
                if attached.exists():
                    receipt_root = attached
        current = capture(project,
                          None if receipt else evidence["targets"],
                          config=ArticleConfig(chunk_chars=evidence["chunk_chars"]),
                          result_path=receipt["result_path"] if receipt else "main/result.json",
                          completion_root=str(receipt_root) if receipt_root is not None else None,
                          environ=environ, original_project=original if relocated else None)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvidenceChanged("article_evidence_unavailable_or_revoked") from exc
    if relocated:
        # The old paths are evidence identity, not live locations. Re-use them only
        # after comparing the current project and checking the actual receipt files.
        current["project_dir"] = str(original)
        if receipt:
            current["completion"]["root"] = receipt["root"]
        current["digest"] = digest({key: value for key, value in current.items() if key != "digest"})
    if current != evidence:
        raise EvidenceChanged("article_evidence_changed")


def persist(root: Path, evidence: dict) -> None:
    validate(evidence)
    target = safe_path(root, "evidence.json")
    if target.exists():
        if read_json(target) != evidence:
            raise ArticleError("article_evidence_is_immutable")
        return
    root.mkdir(parents=True, exist_ok=True)
    write_json(target, evidence)


def load(root: Path) -> dict:
    evidence = read_json(safe_path(root, "evidence.json"))
    validate(evidence)
    return evidence


def units(evidence: dict) -> dict[str, dict]:
    """Every proof character and definition has a stable, auditable source ID."""
    result = {}
    for fact in evidence["facts"]:
        fid = fact["fact_id"]
        items = [(fid + ":statement", "statement", fact["statement"])]
        items.extend((fid + f":definition:{i:04d}", "definition", name + ": " + value)
                     for i, (name, value) in enumerate(sorted(fact["definitions"].items()), 1))
        items.extend((fid + ":proof:" + r["id"], "proof", fact["proof"][r["start"]:r["end"]])
                     for r in fact["ranges"])
        for identifier, kind, text in items:
            result[identifier] = {"id": identifier, "kind": kind, "fact_id": fid,
                                  "text": text, "sha256": text_hash(text)}
    if evidence["completion"]:
        result["completion:bridge"] = {"id": "completion:bridge", "kind": "bridge", "fact_id": None,
                                       "text": evidence["summary"], "sha256": text_hash(evidence["summary"])}
    return result
