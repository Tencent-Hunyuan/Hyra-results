"""Full completion revalidation; research rejection is not an execution failure."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from hyprover.core import KnowledgeStore
from . import completion_evidence as e


def _check(check_cancel: Callable) -> None:
    if check_cancel():
        raise InterruptedError("completion_stop_requested")


def _feedback(project: Path, root: Path, result: dict) -> dict:
    stable = {k: v for k, v in result.items() if k not in ("feedback_id", "feedback_error", "deduplicated")}
    details = json.dumps(stable, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    store = KnowledgeStore(project)
    entry = store.record(
        "feedback", "Completion full revalidation", research_feedback(stable), "verifier", "verifier",
        links=([stable["failed_fact_id"]] if stable.get("failed_fact_id") else []), details=details,
        attributes={"operation": "completion_revalidation", "evidence_digest": stable["evidence_digest"],
                    "outcome": stable["outcome"], "run_id": root.name},
        request_id="completion-" + e.digest(stable),
    )
    result["feedback_id"] = entry.entry_id
    return result


def research_feedback(result: dict) -> str:
    if result.get("accepted"):
        return "Full dependency proofs and final original-problem bridge passed fresh LLM revalidation; not a formal certificate."
    return (
        f"Completion NOT accepted: {result['outcome']}. "
        f"Defective/suspect fact: {result.get('failed_fact_id') or 'none identified'}. "
        f"Withdrawn certifications: {', '.join(result.get('revoked', [])) or 'none'}. "
        f"Read {result.get('report_path', '')} for exact locations and repair requirements. "
        + (f"Input/run detail: {str(result['error'])[:1500]}. " if result.get("error") else "")
        + "Continue research; repair or replace defective proofs and the final bridge, update assignments/routes, "
        "and do not use revoked facts. Withdrawal is not a refutation. Do not repeat an unchanged completion claim."
    )


def _finish(project: Path, root: Path, manifest: dict, outcome: str, **fields) -> dict:
    result = {"format": e.FORMAT, "evidence_digest": manifest["digest"], "accepted": outcome == "passed",
              "outcome": outcome, "full_revalidation_run": root.name,
              "report_path": str(root / "completion-report.json"), "revoked": [], **fields}
    if outcome == "incomplete":
        result["retry_after"] = time.time() + 300
    e.write_json(root / "completion-report.json", result)
    try:
        _feedback(project, root, result)
    except Exception as exc:
        result["feedback_error"] = f"{type(exc).__name__}: {exc}"
    e.write_json(root / "completion-report.json", result)
    return result


def _job_context(root: Path, manifest: dict, job: dict, progress: dict) -> dict:
    facts = {f["fact_id"]: f for f in manifest["facts"]}
    target = facts.get(job["fact_id"])
    needed, pending = set(), list(target["predecessors"]) if target else list(facts)
    while pending:
        fid = pending.pop()
        if fid not in needed:
            needed.add(fid)
            pending.extend(facts[fid]["predecessors"])
    required = [fid for fid in facts if fid in needed]
    trusted = []
    for fid in required:
        receipt = progress["receipts"].get(fid + "-whole")
        if receipt is None:
            raise ValueError("completion_predecessor_not_revalidated")
        f = facts[fid]
        trusted.append({"fact_id": fid, "revision": f["revision"], "statement": f["statement"],
                        "definitions": f["definitions"], "external_refs": f["external_refs"],
                        "proof_file": str(e.safe_path(root, f["proof_file"])), "run_id": receipt["run_id"]})
    chunk_reports = {key: value for key, value in progress["receipts"].items()
                     if target and key.startswith(target["fact_id"] + "-part-")}
    frozen_target = dict(target, proof_file=str(e.safe_path(root, target["proof_file"]))) if target else None
    return {"allow_math_computation": manifest["allow_math_computation"],
            "completion_review": {"manifest_path": str(root / "completion-manifest.json"),
            "research_mode": manifest["research_mode"], "problem": manifest["problem"],
            "job": job, "fact": frozen_target, "expected_coverage": e.expected_coverage(manifest, job),
            "trusted_current_run": trusted, "chunk_reports": chunk_reports}}


def review_defect(project: Path, manifest: dict, failure: dict, verifier: Callable,
                  allocator: Callable, check_cancel: Callable) -> dict:
    from .review import review_fact
    _check(check_cancel)
    e.assert_current(project, manifest)
    job, report = failure["job"], failure["report"]
    reason = json.dumps({"completion_evidence": manifest["digest"], "fact_id": job["fact_id"],
                         "revision": job["revision"], "proof_sha256": job["proof_sha256"],
                         "findings": report["verification_report"], "repair_hints": report.get("repair_hints", "")},
                        ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    result = review_fact(KnowledgeStore(project), job["fact_id"], action="revoke", reason=reason,
                         expected_revision=job["revision"], author="main", verifier=verifier, allocator=allocator,
                         check_cancel=check_cancel, before_commit=lambda _: e.assert_current(project, manifest))
    if not result.get("applied") and result.get("error"):
        if any(marker in result["error"] for marker in ("EvidenceChanged", "revision_or_state_conflict", "evidence_changed")):
            raise e.EvidenceChanged(result["error"])
        raise RuntimeError("completion_withdrawal_review_failed: " + result["error"])
    if not result.get("applied") and result.get("verdict") == "error":
        raise RuntimeError("completion_withdrawal_review_failed")
    return result


def run(project: Path, decision: dict, *, results_root: Path, config: dict,
        verifier: Callable, allocator: Callable, check_cancel: Callable = lambda: False,
        update: Callable = lambda **_: None, defect_handler: Callable | None = None) -> dict:
    """Callbacks own child processes; checkpoints only reuse this exact frozen input."""
    _check(check_cancel)
    try:
        manifest, bodies = e.capture(project, decision, config)
    except (ValueError, OSError) as exc:
        return {"accepted": False, "outcome": "incomplete", "revoked": [],
                "error": f"{type(exc).__name__}: {exc}", "report_path": "", "failed_fact_id": None}
    root = e.safe_path(results_root, "completion-" + manifest["digest"])
    e.persist(root, manifest, bodies)
    del bodies
    update(state="reviewing", full_revalidation_run=root.name,
           completion_evidence_digest=manifest["digest"], completion_phase="resume")
    report_path = root / "completion-report.json"
    if report_path.exists():
        cached = e.read_json(report_path)
        if cached.get("evidence_digest") != manifest["digest"]:
            raise ValueError("completion_cached_report_mismatch")
        if cached.get("accepted"):
            e.validate_completion(root, project)
        elif cached.get("outcome") not in ("fact_defect", "disputed", "bridge_gap", "incomplete", "stale"):
            raise ValueError("invalid_completion_cached_outcome")
        _check(check_cancel)
        retry_incomplete = (cached.get("outcome") == "incomplete"
                            and time.time() >= cached.get("retry_after", 0))
        if not retry_incomplete:
            try:
                _feedback(project, root, cached)
            except Exception as exc:
                cached["feedback_error"] = f"{type(exc).__name__}: {exc}"
            return dict(cached, deduplicated=True)
        progress = e.load_progress(root, manifest)
        progress.pop("pending_failure", None)
        e.write_json(root / "completion-progress.json", progress)
        report_path.replace(e.safe_path(root, f"incomplete-{time.time_ns()}.json"))
    progress = e.load_progress(root, manifest)
    try:
        for job in e.jobs(manifest):
            _check(check_cancel)
            if job["id"] in progress["receipts"]:
                continue
            e.assert_current(project, manifest)
            update(completion_phase=job["kind"], completion_fact_id=job["fact_id"], completion_job=job["id"])
            pending = progress.get("pending_failure")
            if pending:
                if pending.get("job") != job:
                    raise ValueError("invalid_pending_completion_failure")
                report, rid = pending["report"], pending["run_id"]
            else:
                rid = allocator("completion " + manifest["digest"] + " " + job["id"])
                context = _job_context(root, manifest, job, progress)
                fact = context["completion_review"]["fact"]
                statement = fact["statement"] if fact else (
                    "The supplied facts establish a complete positive proof under the exact original hypotheses and "
                    "quantifiers. Do not assume the original assertion as a premise. " if manifest["research_mode"] == "proof_only"
                    else "The supplied facts fully settle the exact original research question. "
                ) + manifest["problem"]
                report = verifier(run_id=rid, statement=statement,
                                  proof=manifest["summary"] if fact is None else "Read the frozen proof_file in the manifest.",
                                  context=context)
            _check(check_cancel)
            e.assert_current(project, manifest)
            try:
                outcome = e.report_outcome(manifest, job, report)
            except ValueError as exc:
                return _finish(project, root, manifest, "incomplete", failure={"job": job, "run_id": rid,
                               "error": str(exc), "report": report})
            if outcome == "passed":
                progress["receipts"][job["id"]] = {"run_id": rid, "report": report}
                progress.pop("pending_failure", None)
                e.write_json(root / "completion-progress.json", progress)
                continue
            failure = {"job": job, "run_id": rid, "report": report}
            progress["pending_failure"] = failure
            e.write_json(root / "completion-progress.json", progress)
            if outcome == "fact_defect":
                review = (defect_handler or review_defect)(project, manifest, failure, verifier, allocator, check_cancel)
                applied = review.get("accepted") is True and review.get("applied") is True
                result = _finish(project, root, manifest, "fact_defect" if applied else "disputed",
                                 failed_fact_id=job["fact_id"], failure=failure, review=review,
                                 revoked=review.get("revoked", []) if applied else [])
                _check(check_cancel)
                return result
            return _finish(project, root, manifest, outcome, failure=failure)
        _check(check_cancel)
        e.assert_current(project, manifest)
        return _finish(project, root, manifest, "passed",
                       coverage_verification_run=progress["receipts"]["bridge"]["run_id"],
                       checked_facts=len(manifest["facts"]), checked_jobs=len(progress["receipts"]))
    except e.EvidenceChanged as exc:
        return _finish(project, root, manifest, "stale", error=str(exc))
