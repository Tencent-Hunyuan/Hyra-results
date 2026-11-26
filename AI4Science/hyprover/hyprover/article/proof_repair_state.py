"""Bind local proof review outcomes to the exact manuscript and frozen evidence."""
from __future__ import annotations

from pathlib import Path

from .models import digest, read_json, safe_path

REVISION = 1
RECEIPT = "proof-repair.json"
READY_MEANING = "Readiness also requires a current, passed independent review of targeted proof repairs."


def read_receipt(root: Path, article: dict, evidence: dict) -> dict | None:
    path = safe_path(root, RECEIPT)
    try:
        receipt = read_json(path)
    except (OSError, ValueError):
        return None
    if (type(receipt.get("revision")) is not int or receipt["revision"] != REVISION
            or receipt.get("article_digest") != digest(article)
            or receipt.get("evidence_digest") != evidence["digest"]
            or receipt.get("targets") != evidence["targets"]
            or receipt.get("outcome") not in ("passed", "needs_review")):
        return None
    expected = {fact["fact_id"] for fact in evidence["facts"]}
    if evidence["completion"]:
        expected.add("completion:bridge")
    checks = receipt.get("checks")
    if not isinstance(checks, list) or len(checks) != len(expected):
        return None
    seen = set()
    for check in checks:
        if not isinstance(check, dict):
            return None
        target = check.get("target")
        if (not isinstance(target, str) or target not in expected or target in seen
                or check.get("outcome") not in ("clear", "repaired", "needs_review")):
            return None
        seen.add(target)
    if receipt["outcome"] == "passed" and any(check["outcome"] == "needs_review" for check in checks):
        return None
    return receipt


def apply_report(root: Path, article: dict, evidence: dict, report: dict, *, required: bool = False) -> dict:
    path = safe_path(root, RECEIPT)
    if not path.exists() and not required and "proof_repair" not in report:
        return report
    receipt = read_receipt(root, article, evidence)
    summary = {"revision": REVISION, "outcome": "needs_review"}
    error = None
    if receipt is None:
        error = "proof_repair_receipt_missing_or_stale"
    else:
        summary.update(outcome=receipt["outcome"], receipt_digest=digest(receipt),
                       repaired_targets=[row["target"] for row in receipt["checks"] if row["outcome"] == "repaired"],
                       unresolved_targets=[row["target"] for row in receipt["checks"] if row["outcome"] == "needs_review"])
        if receipt["outcome"] != "passed":
            error = "proof_repair_requires_review"
    meaning = report.get("meaning_of_ready")
    if isinstance(meaning, str) and not meaning.endswith(READY_MEANING):
        report = dict(report, meaning_of_ready=meaning + " " + READY_MEANING)
    if error is None:
        return dict(report, proof_repair=summary)
    summary["error"] = error
    if report.get("proof_repair") == summary and error in report.get("failures", []):
        return dict(report, state="needs_review")
    failures = [failure for failure in report.get("failures", []) if not failure.startswith("proof_repair_")]
    return dict(report, state="needs_review", failures=[*failures, error], proof_repair=summary)
