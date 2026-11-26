"""Independently review withdrawal evidence; only the service applies changes."""
from __future__ import annotations

import hashlib
import json
from typing import Callable

from hyprover.core import KnowledgeStore, plain
from hyprover.core.knowledge_markdown import FACT_ID, check_text
from hyprover.execution import layout as L
from .publication import accepted_verdict


def _feedback(store: KnowledgeStore, result: dict, reason: str, author: str) -> None:
    try:
        snapshot = store.snapshot()
        links = [i for i in (result["fact_id"], result.get("replacement_id")) if i in snapshot.facts]
        details = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).replace("<", "\\u003c")
        entry = store.record(
            "feedback", "Fact review: " + result["action"],
            ("Review applied. " if result["applied"] else "Review not applied. ") + reason,
            "verifier", "verifier", links=links, details=details,
            attributes={"operation": "fact_review", "action": result["action"],
                        "applied": result["applied"], "run_id": result.get("run_id", ""),
                        "requested_by": author},
        )
        result["feedback_id"] = entry.entry_id
    except Exception as exc:
        result["feedback_error"] = f"{type(exc).__name__}: {exc}"


def review_fact(store: KnowledgeStore, fact_id: str, *, action: str, reason: str,
                expected_revision: int, author: str, verifier: Callable, allocator: Callable,
                replacement_id: str | None = None, evidence_fact_ids: list[str] | None = None,
                check_cancel: Callable | None = None, before_commit: Callable | None = None) -> dict:
    """A correct verdict certifies the review argument, NOT the challenged fact."""
    def guard(snapshot=None):
        if check_cancel is not None and check_cancel():
            raise InterruptedError("fact_review_stop_requested")
        if snapshot is not None and before_commit is not None:
            before_commit(snapshot)
        if check_cancel is not None and check_cancel():
            raise InterruptedError("fact_review_stop_requested")

    guard()
    result = {"accepted": False, "applied": False, "fact_id": fact_id, "action": action,
              "replacement_id": replacement_id, "revoked": [], "verdict": "error", "stage": "precheck"}
    request_id = ""
    try:
        if action not in ("revoke", "supersede"):
            raise ValueError("invalid_review_action")
        check_text(reason)
        check_text(author)
        if not reason.strip() or not author.strip():
            raise ValueError("review_requires_reason_and_author")
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("review_requires_positive_revision")
        if not FACT_ID.fullmatch(fact_id):
            raise ValueError("invalid_fact_id")
        evidence = list(evidence_fact_ids or [])
        if len(evidence) != len(set(evidence)) or any(not FACT_ID.fullmatch(i) for i in evidence):
            raise ValueError("invalid_review_evidence_ids")
        if (action == "supersede") != bool(replacement_id):
            raise ValueError("replacement_required_only_for_supersession")
        snapshot = store.snapshot()
        target = snapshot.facts.get(fact_id)
        if not target:
            raise ValueError("unknown_review_target")
        payload = dict(fact_id=fact_id, action=action, reason=reason, expected_revision=expected_revision,
                       replacement_id=replacement_id, evidence_fact_ids=sorted(evidence), author=author)
        request_id = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        prior = store.review_receipt(fact_id, request_id)
        if prior:
            return dict(prior, deduplicated=True)
        if target.status != "verified" or target.revision != expected_revision:
            raise ValueError("review_target_revision_or_state_conflict")
        challenged = store.read_fact(fact_id, snapshot=snapshot)
        replacement = None
        if replacement_id:
            replacement = store.read_fact(replacement_id, snapshot=snapshot)
        roots = list(dict.fromkeys([*target.predecessors, *evidence,
                                   *([replacement_id] if replacement_id else [])]))
        closure = snapshot.topological(roots)
        forbidden = {fact_id, *store._descendants(snapshot, fact_id)}
        if forbidden.intersection(closure):
            raise ValueError("review_cannot_use_target_or_descendants_as_evidence")
        trusted = [{"fact_id": i, "statement": snapshot.facts[i].statement,
                    "glossary_introduces": dict(snapshot.facts[i].glossary_introduces),
                    "proof_path": str(store.root / snapshot.facts[i].proof_path)} for i in closure]
        versions = {i: snapshot.facts[i].revision for i in [fact_id, *closure]}
        review = {"action": action, "fact_id": fact_id, "request_id": request_id,
                  "replacement_id": replacement_id, "reason": reason, "requested_by": author,
                  "expected_revisions": versions, "evidence_fact_ids": closure,
                  "proof_checks": list(dict.fromkeys([fact_id, *roots]))}
        context = {"allow_math_computation": L.math_computation_allowed(store.root),
                   "problem_id": snapshot.problem_id, "predecessors": trusted,
                   "definitions": {k: v for i in closure for k, v in snapshot.facts[i].glossary_introduces.items()},
                   "fact_review": {"action": action, "target": {
                       "fact_id": fact_id, "statement": challenged.statement, "proof": challenged.proof,
                       "definitions": challenged.glossary_introduces},
                       "replacement": ({"fact_id": replacement_id, "statement": replacement.statement}
                                       if replacement else None)}}
        claim = ("The review argument identifies a concrete mathematical error or material proof gap "
                 f"in fact {fact_id}, warranting withdrawal of its certification (not necessarily refutation)."
                 if action == "revoke" else
                 f"The independently published fact {replacement_id} covers the full conclusion of {fact_id} "
                 "under every original hypothesis and parameter range, and provides a genuine strengthening; "
                 "it may replace the old fact as the preferred result without invalidating the old proof.")
        result.update(stage="verification", request_id=request_id)
        run_id = allocator(claim + "\n" + request_id)
        result["run_id"] = run_id
        guard()
        verdict = verifier(run_id=run_id, statement=claim, proof=reason, context=context)
        guard()
        result["raw_verification"] = plain(verdict)
        if not isinstance(verdict, dict):
            raise ValueError("verification_result_must_be_object")
        result.update({k: verdict[k] for k in ("verdict", "verification_report", "repair_hints") if k in verdict})
        if not accepted_verdict(verdict):
            if verdict.get("verdict") == "correct":
                result["error"] = "inconsistent_or_incomplete_review_report"
        else:
            review.update(run_id=run_id, verification=plain(verdict))
            result["stage"] = "commit"
            result.update(store.apply_review(review, before_commit=guard)
                          if check_cancel is not None or before_commit is not None else store.apply_review(review))
    except InterruptedError:
        raise
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {getattr(exc, 'detail', str(exc))}"
        if result["stage"] == "commit":
            try:
                receipt = store.review_receipt(fact_id, request_id, run_id=result.get("run_id"))
                if receipt:
                    result.update(receipt, durability_warning=result["error"])
            except Exception:
                pass
    _feedback(store, result, reason, author)
    return result
