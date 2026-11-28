"""Frozen completion evidence and fail-closed, machine-checkable coverage receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hyprover.core import KnowledgeStore, plain
from hyprover.core._util import atomic_write_text
from hyprover.execution import layout as L
from .publication import accepted_verdict

FORMAT = "hyprover-completion-v1"
DEFAULT_CHUNK_CHARS = 24000


class EvidenceChanged(ValueError):
    """A completion judgement no longer applies to the current project."""


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    safe_path(path.parent, path.name)
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> dict:
    safe_path(path.parent, path.name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("completion_document_must_be_object")
    return value


def safe_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(p in ("..", ".") for p in rel.parts):
        raise ValueError("unsafe_completion_path")
    path = root.absolute() / rel
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink_completion_path")
    return path


def ranges(text: str, limit: int) -> list[dict]:
    """Cover every character once, preferring paragraph boundaries to hard splits."""
    if type(limit) is not int or limit < 1 or not text.strip():
        raise ValueError("invalid_completion_proof_or_chunk_limit")
    result, start = [], 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary > start:
                end = boundary + 2
        result.append({"id": f"part-{len(result) + 1:04d}", "start": start, "end": end,
                       "sha256": text_hash(text[start:end])})
        start = end
    return result


def capture(project: Path, decision: dict, config: dict) -> tuple[dict, dict[str, str]]:
    store = KnowledgeStore(project)
    snapshot = store.snapshot()
    targets = decision.get("target_fact_ids")
    summary = decision.get("summary")
    if (not isinstance(targets, list) or not targets or any(not isinstance(i, str) for i in targets)
            or len(targets) != len(set(targets))):
        raise ValueError("completion_requires_distinct_verified_targets")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("completion_requires_bridge_argument")
    limit = config.get("chunk_chars", DEFAULT_CHUNK_CHARS)
    if type(limit) is not int or limit < 1:
        raise ValueError("invalid_completion_chunk_limit")
    problem = safe_path(project, "PROBLEM.md").read_bytes().decode("utf-8")
    if not problem.strip():
        raise ValueError("completion_requires_original_problem")
    metadata = read_json(project / "project.json")
    facts, bodies = [], {}
    for fid in snapshot.topological(sorted(targets)):
        record = snapshot.facts[fid]
        fact = store.read_fact(fid, snapshot=snapshot)
        path = f"evidence/{fid}.txt"
        bodies[path] = fact.proof
        facts.append({"fact_id": fid, "revision": record.revision, "statement": fact.statement,
                      "predecessors": list(fact.predecessors),
                      "definitions": plain(fact.glossary_introduces),
                      "external_refs": plain(fact.external_refs), "intuition": fact.intuition,
                      "proof_file": path, "proof_sha256": text_hash(fact.proof),
                      "ranges": ranges(fact.proof, limit)})
    manifest = {"format": FORMAT, "project_dir": str(project.resolve()),
                "problem_id": snapshot.problem_id, "problem": problem, "problem_hash": text_hash(problem),
                "research_mode": metadata.get("research_mode"), "targets": sorted(targets),
                "allow_math_computation": L.math_computation_allowed(metadata=metadata),
                "summary": summary, "config": dict(config, chunk_chars=limit), "facts": facts}
    manifest["digest"] = digest(manifest)
    return manifest, bodies


def persist(root: Path, manifest: dict, bodies: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = safe_path(root, "completion-manifest.json")
    if manifest_path.exists():
        if load_manifest(root) != manifest:
            raise ValueError("completion_manifest_is_immutable")
        return
    for relative, body in bodies.items():
        path = safe_path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, body)
    write_json(manifest_path, manifest)


def load_manifest(root: Path) -> dict:
    manifest = read_json(safe_path(root, "completion-manifest.json"))
    if manifest.get("format") != FORMAT or manifest.get("digest") != digest(
            {k: v for k, v in manifest.items() if k != "digest"}):
        raise ValueError("invalid_completion_manifest")
    if "allow_math_computation" not in manifest:
        raise ValueError("completion_computation_policy_missing")
    L.math_computation_allowed(metadata=manifest)
    if manifest.get("problem_hash") != text_hash(manifest["problem"]):
        raise ValueError("invalid_completion_problem_hash")
    facts = manifest.get("facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("missing_completion_facts")
    seen = {}
    for fact in facts:
        fid = fact["fact_id"]
        if fid in seen or any(pid not in seen for pid in fact["predecessors"]):
            raise ValueError("invalid_completion_dependency_order")
        if type(fact["revision"]) is not int or fact["revision"] < 1:
            raise ValueError("invalid_completion_revision")
        text = safe_path(root, fact["proof_file"]).read_bytes().decode("utf-8")
        if text_hash(text) != fact["proof_sha256"] or ranges(text, manifest["config"]["chunk_chars"]) != fact["ranges"]:
            raise ValueError("completion_proof_or_ranges_changed")
        seen[fid] = fact
    required, pending = set(), list(manifest["targets"])
    while pending:
        fid = pending.pop()
        if fid not in seen:
            raise ValueError("completion_target_missing")
        if fid not in required:
            required.add(fid)
            pending.extend(seen[fid]["predecessors"])
    if required != set(seen):
        raise ValueError("completion_closure_mismatch")
    return manifest


def assert_current(project: Path, manifest: dict, *, original_project: Path | None = None) -> None:
    try:
        current, _ = capture(project, {"target_fact_ids": manifest["targets"],
                                      "summary": manifest["summary"]}, manifest["config"])
    except (ValueError, OSError, KeyError) as exc:
        raise EvidenceChanged(f"completion_evidence_changed: {exc}") from exc
    if original_project is not None:
        if not original_project.is_absolute() or manifest["project_dir"] != str(original_project):
            raise EvidenceChanged("completion_original_project_identity_changed")
        current["project_dir"] = str(original_project)
        current["digest"] = digest({key: value for key, value in current.items() if key != "digest"})
    if current != manifest:
        raise EvidenceChanged("completion_evidence_changed")


def jobs(manifest: dict) -> list[dict]:
    result = []
    for fact in manifest["facts"]:
        parts = [r["id"] for r in fact["ranges"]]
        base = {"fact_id": fact["fact_id"], "revision": fact["revision"],
                "proof_sha256": fact["proof_sha256"]}
        if len(parts) > 1:
            for part in parts:
                result.append(dict(base, id=f"{fact['fact_id']}-{part}", kind="chunk", ranges=[part]))
        result.append(dict(base, id=f"{fact['fact_id']}-whole", kind="whole", ranges=parts))
    result.append({"id": "bridge", "kind": "bridge", "fact_id": None, "revision": None,
                   "proof_sha256": text_hash(manifest["summary"]), "ranges": []})
    return result


def expected_coverage(manifest: dict, job: dict) -> dict:
    return {"evidence_digest": manifest["digest"], "job_id": job["id"], "fact_id": job["fact_id"],
            "revision": job["revision"], "proof_sha256": job["proof_sha256"],
            "checked_ranges": job["ranges"],
            "checked_facts": [f["fact_id"] for f in manifest["facts"]] if job["kind"] == "bridge" else []}


def report_outcome(manifest: dict, job: dict, report: dict) -> str:
    if not isinstance(report, dict):
        raise ValueError("invalid_completion_report")
    coverage = report.get("coverage")
    expected = expected_coverage(manifest, job)
    if (not isinstance(coverage, dict) or not expected.keys() <= coverage.keys()
            or digest({k: coverage[k] for k in expected}) != digest(expected)):
        raise ValueError("completion_coverage_mismatch")
    if (coverage.get("uncovered") != [] or coverage.get("materials_complete") is not True
            or coverage.get("truncated") is not False):
        return "incomplete"
    if accepted_verdict(report):
        if report.get("failure_kind") not in (None, ""):
            raise ValueError("contradictory_completion_report")
        return "passed"
    if report.get("failure_kind") == "incomplete":
        return "incomplete"
    details = report.get("verification_report", {})
    if not isinstance(details, dict):
        raise ValueError("invalid_completion_findings")
    errors, gaps = details.get("critical_errors"), details.get("gaps")
    if not isinstance(errors, list) or not isinstance(gaps, list):
        raise ValueError("invalid_completion_findings")
    findings = errors + gaps
    if report.get("verdict") != "wrong" or not findings or any(
            not isinstance(f, dict) or not isinstance(f.get("location"), str) or not f["location"].strip()
            or not isinstance(f.get("issue"), str) or not f["issue"].strip() for f in findings):
        raise ValueError("completion_failure_requires_located_findings")
    if job["kind"] == "bridge" and report.get("failure_kind") == "bridge_gap":
        return "bridge_gap"
    if job["kind"] != "bridge" and report.get("failure_kind") == "fact_defect":
        if any(f.get("range_id") not in job["ranges"] for f in findings):
            raise ValueError("completion_defect_requires_checked_range")
        return "fact_defect"
    raise ValueError("invalid_completion_failure_kind")


def load_progress(root: Path, manifest: dict) -> dict:
    path = safe_path(root, "completion-progress.json")
    if not path.exists():
        return {"format": FORMAT, "evidence_digest": manifest["digest"], "receipts": {}}
    progress = read_json(path)
    if progress.get("format") != FORMAT or progress.get("evidence_digest") != manifest["digest"]:
        raise ValueError("stale_completion_checkpoint")
    expected = jobs(manifest)
    receipts = progress.get("receipts")
    if not isinstance(receipts, dict) or set(receipts) - {job["id"] for job in expected}:
        raise ValueError("invalid_completion_checkpoint")
    missing = False
    for job in expected:
        receipt = receipts.get(job["id"])
        if receipt is None:
            missing = True
            continue
        if missing or not isinstance(receipt, dict) or not isinstance(receipt.get("run_id"), str) or not receipt["run_id"]:
            raise ValueError("out_of_order_completion_checkpoint")
        if report_outcome(manifest, job, receipt.get("report")) != "passed":
            raise ValueError("unaccepted_completion_checkpoint")
    return progress


def validate_completion(root: Path, project: Path, *, result: dict | None = None,
                        original_project: Path | None = None) -> dict:
    manifest = load_manifest(root)
    report = read_json(safe_path(root, "completion-report.json"))
    if (report.get("format") != FORMAT or report.get("evidence_digest") != manifest["digest"]
            or report.get("outcome") != "passed" or report.get("accepted") is not True):
        raise ValueError("completion_full_revalidation_not_accepted")
    progress = load_progress(root, manifest)
    if set(progress["receipts"]) != {job["id"] for job in jobs(manifest)} or progress.get("pending_failure"):
        raise ValueError("completion_checks_missing_or_unresolved")
    bridge = progress["receipts"]["bridge"]
    if report.get("coverage_verification_run") != bridge["run_id"]:
        raise ValueError("completion_bridge_receipt_mismatch")
    if result is not None and (
            result.get("completion_evidence_digest") != manifest["digest"]
            or result.get("coverage_verification_run") != bridge["run_id"]
            or sorted(result.get("target_fact_ids", [])) != manifest["targets"]
            or result.get("summary") != manifest["summary"]):
        raise ValueError("completion_result_evidence_mismatch")
    assert_current(project, manifest, original_project=original_project)
    return manifest
