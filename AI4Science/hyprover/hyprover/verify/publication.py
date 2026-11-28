"""Independent candidate verification followed by publication into one notebook."""
from __future__ import annotations

import json
from typing import Callable

from hyprover.core import Fact, KnowledgeStore, compute_fact_id, plain
from hyprover.execution import layout as L


def accepted_verdict(result: object) -> bool:
    if not isinstance(result, dict) or result.get("verdict") != "correct":
        return False
    report = result.get("verification_report")
    return (
        isinstance(report, dict)
        and isinstance(report.get("summary"), str)
        and report.get("critical_errors") == []
        and report.get("gaps") == []
        and result.get("repair_hints") == ""
        and not any(result.get(key) for key in ("errors", "critical_errors", "gaps"))
    )


def _feedback(store: KnowledgeStore, result: dict, source_id: str | None, run_id: str) -> None:
    """Feedback failure cannot turn an already published fact into a rejection."""
    try:
        details = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).replace("<", "\\u003c")
        entry = store.record(
            kind="feedback", title="Verification submission outcome",
            body=("The independent verification service published this candidate."
                  if result.get("published") else
                  "This candidate was not published. A failed proof is not a refuted proposition."),
            author="verifier", role="verifier",
            links=([source_id] if source_id and source_id in store.snapshot().entries else [])
                  + ([result["fact_id"]] if result.get("published") else []),
            details=details, attributes={"run_id": run_id, "published": bool(result.get("published"))},
        )
        result["feedback_id"] = entry.entry_id
    except Exception as exc:
        result["feedback_error"] = f"{type(exc).__name__}: {exc}"


def verify_and_publish(
    store: KnowledgeStore, candidate: Fact, *, verifier: Callable,
    allocator: Callable, source_id: str | None = None, source_revision: int | None = None,
) -> dict:
    """Only this service path obtains a verdict and calls the store's write gate."""
    candidate = Fact(**plain(candidate))
    result = {"accepted": False, "published": False, "fact_id": None, "verdict": "error"}
    run_id = ""
    try:
        snapshot = store.snapshot()
        if source_id is not None and (type(source_revision) is not int or source_revision < 1):
            raise ValueError("source_requires_positive_integer_revision")
        if candidate.problem_id != snapshot.problem_id:
            raise ValueError("candidate_project_mismatch")
        if not candidate.author.strip():
            raise ValueError("candidate_author_required")
        if not candidate.statement.strip() or not candidate.proof.strip():
            raise ValueError("empty_fact_content")
        fid = compute_fact_id(
            problem_id=candidate.problem_id, statement=candidate.statement, proof=candidate.proof,
            predecessors=candidate.predecessors, glossary_introduces=candidate.glossary_introduces,
        )
        if candidate.fact_id and candidate.fact_id != fid:
            raise ValueError("candidate_content_address_mismatch")
        candidate.fact_id = fid
        if len(set(candidate.predecessors)) != len(candidate.predecessors):
            raise ValueError("duplicate_predecessors")
        closure = snapshot.topological(candidate.predecessors)
        source = snapshot.entries.get(source_id) if source_id else None
        existing = snapshot.facts.get(fid)
        already_resolved = (source and source.status == "proved"
                            and source.attributes.get("resolution_fact_id") == fid)
        if source_id:
            if not source or source.kind != "conjecture":
                raise ValueError("invalid_conjecture_source")
            if " ".join(source.body.split()) != " ".join(candidate.statement.split()):
                raise ValueError("source_statement_mismatch")
            if not already_resolved and (source.revision != source_revision or source.status not in ("open", "reopened")):
                raise ValueError("source_revision_or_state_conflict")
        elif source_revision is not None:
            raise ValueError("source_revision_without_source")
        if existing:
            # The source association can still require a write; retain its commit recovery path.
            result.update(stage="publication", deduplicated=True, verdict="correct")
            run_id = existing.verification.get("run_id", "")
            fid = store.publish(candidate, plain(existing.verification), source_id, source_revision)
            return {"accepted": True, "published": True, "fact_id": fid,
                    "verdict": "correct", "deduplicated": True}
        context = {
            "allow_math_computation": L.math_computation_allowed(store.root),
            "problem_id": snapshot.problem_id,
            "predecessors": [
                {"fact_id": pid, "statement": snapshot.facts[pid].statement,
                 "glossary_introduces": dict(snapshot.facts[pid].glossary_introduces),
                 "proof_path": str(store.root / snapshot.facts[pid].proof_path)}
                for pid in candidate.predecessors
            ],
            "definitions": {k: v for pid in closure for k, v in snapshot.facts[pid].glossary_introduces.items()},
            "candidate_definitions": candidate.glossary_introduces,
        }
        run_id = allocator(candidate.statement)
        verdict = verifier(run_id=run_id, statement=candidate.statement, proof=candidate.proof, context=context)
        result["raw_verification"] = plain(verdict)
        if not isinstance(verdict, dict):
            raise ValueError("verification_result_must_be_object")
        result.update({k: verdict[k] for k in ("verdict", "verification_report", "repair_hints") if k in verdict})
        result.update(run_id=run_id)
        proposals = verdict.get("fact_reviews", [])
        if proposals:
            from .review import review_fact
            if not isinstance(proposals, list) or len(proposals) > 3:
                result["fact_review_warning"] = "fact_reviews must contain at most three proposals"
            else:
                result["fact_reviews"] = []
                for proposal in proposals:
                    if (not isinstance(proposal, dict) or proposal.get("fact_id") not in closure
                            or set(proposal) - {"fact_id", "action", "reason", "replacement_id", "evidence_fact_ids"}):
                        result["fact_reviews"].append({"accepted": False, "applied": False,
                                                      "error": "invalid_or_out_of_scope_review_proposal"})
                        continue
                    reviewed_id = proposal["fact_id"]
                    result["fact_reviews"].append(review_fact(
                        store, reviewed_id, action=proposal.get("action", "revoke"),
                        reason=proposal.get("reason", ""), replacement_id=proposal.get("replacement_id"),
                        evidence_fact_ids=proposal.get("evidence_fact_ids"),
                        expected_revision=snapshot.facts[reviewed_id].revision, author="verifier",
                        verifier=verifier, allocator=allocator))
        if not accepted_verdict(verdict):
            result["stage"] = "verification"
            if verdict.get("verdict") == "correct":
                result["error"] = "inconsistent_or_incomplete_verification_report"
        else:
            result["stage"] = "publication"
            fid = store.publish(candidate, {"verdict": "correct", "run_id": run_id,
                                "verifier": "independent-codex"}, source_id, source_revision)
            result.update(accepted=True, published=True, fact_id=fid, stage="published")
    except Exception as exc:
        detail = getattr(exc, "detail", str(exc))
        result["error"] = f"{type(exc).__name__}: {detail}"
        if result.get("stage") == "publication":
            result["write_error"] = result["error"]
            try:
                current = store.get(candidate.fact_id)
                if current and current.verification.get("run_id") == run_id:
                    persisted = store.read_fact(candidate.fact_id)
                    source = store.snapshot().entries.get(source_id) if source_id else None
                    resolved = not source_id or (source and source.status == "proved"
                               and source.attributes.get("resolution_fact_id") == candidate.fact_id)
                    if persisted.statement == candidate.statement and persisted.proof == candidate.proof and resolved:
                        result.update(accepted=True, published=True, fact_id=candidate.fact_id,
                                      stage="published", durability_warning=result.pop("write_error"))
            except Exception:
                pass
    _feedback(store, result, source_id, run_id)
    return result
