"""Bounded, resumable generation for an exported collection of verified projects."""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
import uuid
from pathlib import Path

from . import runtime, literature, improvement, audit, stage_state, citation_policy as policy, writing_policy as writing
from .models import FORMAT, ArticleConfig, ArticleError, digest, read_json, safe_path, write_json
from .response_validation import RECOVERY_REVISION

MAX_CONCURRENCY = 9
_DISPATCH_MARKERS = {
    "policy_upgrade_pending": ("upgrade_attempted", policy.POLICY_REVISION),
    "recovery_pending": ("recovery_attempted", RECOVERY_REVISION),
    "typesetting_review_pending": ("typesetting_review_attempted", None),
    "writing_upgrade_pending": ("writing_attempted", writing.REVISION),
    "references_refresh_pending": ("references_attempted", literature.FOCUS_REVISION),
    "improvement_pending": ("improvement_attempted", improvement.REVISION),
    "reassessment_pending": ("reassessment_attempted", audit.REVIEW_REVISION),
}
_STRICT_REVIEW_DISPATCHES = {"typesetting_review_pending", "improvement_pending", "reassessment_pending"}


def generate(root: Path, *, concurrency: int = 2, timeout: int | None = None, compile_output: bool = False) -> int:
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError("article_collection_concurrency_must_be_1_to_" + str(MAX_CONCURRENCY))
    if timeout is not None and (type(timeout) is not int or not 1 <= timeout <= 10800):
        raise ArticleError("invalid_article_config: timeout_seconds")
    root = root.absolute()
    with runtime.locked(safe_path(root, ".article-collection.lock"), blocking=False, optional=True) as acquired:
        if not acquired:
            raise ArticleError("article_collection_busy")
        return _generate_locked(root, concurrency, timeout, compile_output)


def _generate_locked(root: Path, concurrency: int, timeout: int | None, compile_output: bool) -> int:
    collection = read_json(safe_path(root, "export-manifest.json"))
    rows = collection.get("projects", [])
    ids = [row["benchmark"] for row in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("invalid_article_collection")
    state_path = safe_path(root, "article-generation.json")
    old = read_json(state_path) if state_path.exists() else {}
    state = {"format": FORMAT, "timeout_override_seconds": timeout, "concurrency": concurrency,
             "started_at": old.get("started_at", time.time()), "projects": {}, "complete": False,
             "compile_output": compile_output, "review_mode": audit.FORMAT_COVERAGE}
    order = sorted(rows, key=lambda row: row["benchmark"])
    for row in order:
        project = safe_path(root, row["directory"])
        try:
            config = runtime.resolve_config(project)
            if timeout is not None:
                config = dataclasses.replace(config, timeout_seconds=timeout)
            task = runtime.prepare(project, config=config)
            record = runtime.status(project, task.name)
            previous = old.get("projects", {}).get(row["benchmark"], {})
            inherited = previous if previous.get("run_id") == task.name and previous.get("directory") == row["directory"] else {}
            state["projects"][row["benchmark"]] = {**inherited, "directory": row["directory"], "run_id": task.name,
                "state": record["state"], "error": record.get("error"), "article_directory": str(task.relative_to(root)),
                "review_mode": record.get("review_mode", state["review_mode"])}
        except Exception as exc:
            state["projects"][row["benchmark"]] = {"directory": row["directory"], "state": "failed",
                                                     "error": str(exc), "error_type": type(exc).__name__,
                                                     "review_mode": state["review_mode"]}
    write_json(state_path, state)
    return _drive_locked(root, state)


def resume(root: Path, *, upgrade_policy: bool = False, compile_output: bool = False, retry_failed: bool = False,
           upgrade_writing: bool = False, refresh_references: bool = False, reassess: bool = False) -> int:
    root = root.absolute()
    with runtime.locked(safe_path(root, ".article-collection.lock"), blocking=False, optional=True) as acquired:
        if not acquired:
            raise ArticleError("article_collection_busy")
        return _resume_locked(root, upgrade_policy, compile_output, retry_failed, upgrade_writing, refresh_references, reassess)


def _resume_locked(root: Path, upgrade_policy: bool, compile_output: bool, retry_failed: bool,
                   upgrade_writing: bool, refresh_references: bool, reassess: bool) -> int:
    state = read_json(safe_path(root, "article-generation.json"))
    concurrency = state.get("concurrency")
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError("article_collection_concurrency_must_be_1_to_" + str(MAX_CONCURRENCY))
    collection = read_json(safe_path(root, "export-manifest.json"))
    expected = {row["benchmark"]: row["directory"] for row in collection["projects"]}
    if (state.get("format") != FORMAT or set(state.get("projects", {})) != set(expected)
            or any(row["directory"] != expected[key] for key, row in state["projects"].items())):
        raise ValueError("article_collection_resume_mismatch")
    if "config" in state:
        ArticleConfig.from_dict(state["config"])
    state.setdefault("review_mode", audit.STRICT)
    for record in state["projects"].values():
        record.setdefault("review_mode", state["review_mode"])
        if record.get("run_id"):
            project = safe_path(root, record["directory"])
            runtime.load_manifest(runtime.job_path(project, record["run_id"]))
    if upgrade_policy:
        state["upgrade_policy_revision"] = policy.POLICY_REVISION
    if compile_output:
        state["compile_output"] = True
    if retry_failed:
        state["retry_failed"] = True
    if upgrade_writing:
        state["writing_revision"] = writing.REVISION
    if refresh_references:
        state["references_revision"] = literature.FOCUS_REVISION
    if reassess:
        state["review_revision"] = audit.REVIEW_REVISION
    return _drive_locked(root, state)


def _deliver_pdf(project: Path, record: dict) -> bool:
    from . import delivery, render
    task = runtime.job_path(project, record["run_id"])
    try:
        source_digest = digest(render.pdf_sources(task))
        compiled = render.current_pdf(task)
        if not compiled and record.get("pdf_attempted") == source_digest:
            return False
        if not compiled:
            result = runtime.compile_article(project, record["run_id"])
            if result.get("busy"):
                return True
            compiled = result.get("compilation", {})
            record.update(state=result["state"], error=result.get("error"),
                          review_mode=result.get("review_mode", record.get("review_mode", audit.STRICT)),
                          pdf_attempted=source_digest, compilation=compiled)
        else:
            record["compilation"] = compiled
        audit_path = task / "audit.json"
        audit_digest = digest(read_json(audit_path)) if audit_path.exists() else None
        delivery_digest = digest([compiled.get("pdf_sha256"), audit_digest])
        if compiled.get("compiled") and record.get("delivery_digest") != delivery_digest:
            exported = delivery.export(project, record["run_id"])
            record.update(delivered_pdf_sha256=exported["pdf_sha256"], delivery_digest=delivery_digest,
                          pdf_directory=str(project / "paper"))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        record["pdf_error"] = str(exc)
    return False


def _reconcile_dispatch(record: dict, status: dict) -> bool:
    intent = record.get("dispatch_intent")
    if intent and status.get("dispatch_id") == intent["id"]:
        key, _ = _DISPATCH_MARKERS[intent["action"]]
        record[key] = intent["revision"]
        del record["dispatch_intent"]
        return True
    return False


def _checkpoint(task: Path) -> dict:
    try:
        return stage_state.read_checkpoint(task)
    except FileNotFoundError:
        return {}


def _drive_locked(root: Path, state: dict) -> int:
    state_path = safe_path(root, "article-generation.json")
    concurrency = state["concurrency"]
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError("article_collection_concurrency_must_be_1_to_" + str(MAX_CONCURRENCY))
    while True:
        active = 0
        for identifier, record in state["projects"].items():
            if "run_id" not in record:
                continue
            project = safe_path(root, record["directory"])
            current = runtime.status(project, record["run_id"])
            record.update(state=current["state"], phase=current.get("phase"), error=current.get("error"),
                          review_mode=current.get("review_mode", record.get("review_mode", audit.STRICT)))
            _reconcile_dispatch(record, current)
            running = current["state"] in ("running", "queued", "busy")
            intent = record.get("dispatch_intent")
            if (not running and record["review_mode"] == audit.FORMAT_COVERAGE and intent
                    and intent["action"] in _STRICT_REVIEW_DISPATCHES):
                del record["dispatch_intent"]
            if current.get("compilation"):
                record["compilation"] = current["compilation"]
                if current["compilation"].get("compiled"):
                    record.pop("pdf_error", None)
            if running:
                active += 1
            elif record.get("dispatch_intent") and current["state"] not in ("cancelled", "stale"):
                record["state"] = record["dispatch_intent"]["action"]
            elif (record["review_mode"] == audit.STRICT and state.get("review_revision") == audit.REVIEW_REVISION
                  and current["state"] in ("ready", "needs_review")
                  and record.get("reassessment_attempted") != audit.REVIEW_REVISION):
                task = runtime.job_path(project, record["run_id"])
                report = read_json(task / "audit.json") if (task / "audit.json").exists() else {}
                if (not (task / "cancel.json").exists() and report.get("article_digest")
                        and report.get("review_revision") != audit.REVIEW_REVISION):
                    record["state"] = "reassessment_pending"
            elif (state.get("references_revision") == literature.FOCUS_REVISION
                  and current["state"] in ("ready", "needs_review", "failed", "interrupted")):
                task = runtime.job_path(project, record["run_id"])
                checkpoint = _checkpoint(task)
                if (not (task / "cancel.json").exists() and (task / "article.json").exists()
                        and checkpoint.get("references_revision") != literature.FOCUS_REVISION
                        and record.get("references_attempted") != literature.FOCUS_REVISION):
                    record["state"] = "references_refresh_pending"
            elif (state.get("writing_revision") == writing.REVISION
                  and current["state"] in ("ready", "needs_review", "failed", "interrupted")):
                task = runtime.job_path(project, record["run_id"])
                manuscript = read_json(task / "article.json") if (task / "article.json").exists() else {}
                if (not (task / "cancel.json").exists() and writing.revision(manuscript) < writing.REVISION
                        and record.get("writing_attempted") != writing.REVISION):
                    record["state"] = "writing_upgrade_pending"
            elif state.get("retry_failed") and current["state"] in ("failed", "interrupted"):
                task = runtime.job_path(project, record["run_id"])
                checkpoint = _checkpoint(task)
                if (not (task / "cancel.json").exists()
                        and checkpoint.get("response_validation_revision") != RECOVERY_REVISION
                        and record.get("recovery_attempted") != RECOVERY_REVISION):
                    record["state"] = "recovery_pending"
            elif (record["review_mode"] == audit.STRICT and state.get("compile_output")
                  and current["state"] == "needs_review"
                  and current.get("error") == "typesetting_repaired_requires_review"):
                task = runtime.job_path(project, record["run_id"])
                repair_path = task / "typesetting-repair.json"
                repair = read_json(repair_path) if repair_path.exists() else {}
                if (repair.get("review_required") and not (task / "cancel.json").exists()
                        and record.get("typesetting_review_attempted") != repair.get("article_digest")):
                    record.update(state="typesetting_review_pending", typesetting_digest=repair["article_digest"])
            elif state.get("upgrade_policy_revision") == policy.POLICY_REVISION:
                task = runtime.job_path(project, record["run_id"])
                checkpoint = _checkpoint(task)
                raw = read_json(task / "status.json")
                if (raw.get("state") in ("ready", "needs_review", "failed", "running", "queued")
                        and not (task / "cancel.json").exists()
                        and checkpoint.get("policy_revision") != policy.POLICY_REVISION
                        and record.get("upgrade_attempted") != policy.POLICY_REVISION):
                    record["state"] = "policy_upgrade_pending"
            if current.get("auto_improvement"):
                record["auto_improvement"] = current["auto_improvement"]
            if record["review_mode"] == audit.STRICT and current["state"] == record["state"] == "needs_review":
                task = runtime.job_path(project, record["run_id"])
                checkpoint = _checkpoint(task)
                if (not (task / "cancel.json").exists() and (task / "article.json").exists()
                        and (task / "audit.json").exists() and improvement.eligible(checkpoint)
                        and record.get("improvement_attempted") != improvement.REVISION):
                    record["state"] = "improvement_pending"
        for identifier, record in state["projects"].items():
            if active >= concurrency:
                break
            action = record["state"]
            if action != "pending" and action not in _DISPATCH_MARKERS:
                continue
            try:
                project = safe_path(root, record["directory"])
                options = {}
                if action in _DISPATCH_MARKERS:
                    intent = record.get("dispatch_intent")
                    if intent is None:
                        _, revision = _DISPATCH_MARKERS[action]
                        intent = {"action": action, "id": uuid.uuid4().hex,
                                  "revision": record["typesetting_digest"] if revision is None else revision}
                        record["dispatch_intent"] = intent
                        write_json(state_path, state)
                    options.update(resume=True, dispatch_id=intent["id"])
                if action == "references_refresh_pending":
                    options["refresh_references"] = True
                if action == "writing_upgrade_pending":
                    options["upgrade_writing"] = True
                if state.get("compile_output"):
                    options["compile_output"] = True
                launched = runtime.launch(runtime.job_path(project, record["run_id"]), **options)
                _reconcile_dispatch(record, launched)
                record.update(state=launched["state"], error=launched.get("error"),
                              review_mode=launched.get("review_mode", record["review_mode"]))
                if launched["state"] in ("queued", "running", "busy"):
                    active += 1
            except Exception as exc:
                record.update(state="failed", error=str(exc), error_type=type(exc).__name__)
        pdf_busy = False
        if state.get("compile_output"):
            for record in state["projects"].values():
                if record["state"] in ("ready", "needs_review") and record.get("run_id"):
                    pdf_busy = _deliver_pdf(safe_path(root, record["directory"]), record) or pdf_busy
        pending = pdf_busy or any(r["state"] in {"pending", "queued", "running", "busy", *_DISPATCH_MARKERS}
                                  for r in state["projects"].values())
        state.update(updated_at=time.time(), complete=not pending,
                     ready=sum(r["state"] == "ready" for r in state["projects"].values()),
                     needs_review=sum(r["state"] == "needs_review" for r in state["projects"].values()),
                     pdf_ready=sum(r.get("compilation", {}).get("compiled") is True for r in state["projects"].values()))
        write_json(state_path, state)
        if not pending:
            return 0 if state["ready"] == len(state["projects"]) else 2
        time.sleep(10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--jobs", type=int, choices=range(1, MAX_CONCURRENCY + 1), default=2,
                        help="Maximum simultaneous article workers")
    parser.add_argument("--timeout", type=int, help="Override the per-call timeout for this collection")
    parser.add_argument("--resume", action="store_true", help="Resume the exact existing collection without creating new jobs")
    parser.add_argument("--upgrade-policy", action="store_true", help="Re-review existing jobs after running workers finish, preserving budgets")
    parser.add_argument("--retry-failed", action="store_true", help="Resume old failed jobs once with current response validation")
    parser.add_argument("--upgrade-writing", action="store_true", help="Recompose old drafts as concise papers and obtain fresh reviews")
    parser.add_argument("--compile", action="store_true", help="Compile and export each manuscript PDF locally as it finishes")
    parser.add_argument("--refresh-references", action="store_true", help="Supplement references and minimal context without rewriting existing proofs")
    parser.add_argument("--reassess", action="store_true", help="Re-review existing drafts under current acceptance rules with bounded local repairs")
    args = parser.parse_args()
    try:
        if (args.upgrade_policy or args.retry_failed or args.upgrade_writing or args.refresh_references or args.reassess) and not args.resume:
            raise ValueError("upgrade_or_retry_requires_resume")
        if args.resume:
            return resume(args.root, upgrade_policy=args.upgrade_policy, compile_output=args.compile,
                          retry_failed=args.retry_failed, upgrade_writing=args.upgrade_writing,
                          refresh_references=args.refresh_references, reassess=args.reassess)
        return generate(args.root, concurrency=args.jobs, timeout=args.timeout, compile_output=args.compile)
    except (ValueError, OSError) as exc:
        print(json.dumps({"state": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
