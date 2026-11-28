"""Persistent, bounded review -> evidence -> revision -> independent review loop."""
from __future__ import annotations

import copy
import shutil
import time

from . import audit, literature, references as refs, render, writing_policy as writing
from .models import ArticleError, EvidenceChanged, digest, read_json, safe_path, string_list, require_keys, write_json

REVISION = 1
MAX_ROUNDS = 3
MAX_STAGNANT_ROUNDS = 2
PLAN_CONTRACT = """Return {checked_obligations,queries,candidates}. checked_obligations must
list every supplied obligation ID exactly once. queries is a list of NEW targeted public
bibliographic queries, no more than query_limit, excluding previous_queries. Use the review
findings to seek original sources, precise applicability, and missing attribution, not generic
unrelated papers. candidates is [{metadata,obligation_ids}]; metadata contains identifiable
publication facts (title,authors,year,doi,arxiv as known), never invented identifiers, and
obligation_ids is a nonempty subset of the supplied IDs. Do not create obligations, remove
existing gaps, downgrade criticality, or call metadata/abstracts theorem verification.
Empty queries/candidates are valid when no justified lead exists. Respect candidate_limit.
"""
_FILES = ("article.json", "references.json", "audit.json", "paper.tex", "references.tex", "references.bib",
          "paper.pdf", "compilation.json", "draft-status.json", "source-map.json", "citation-patch.json")


def validate_plan(value: dict, material: dict) -> None:
    require_keys(value, {"checked_obligations", "queries", "candidates"})
    known = set(material["obligations"])
    if set(string_list(value["checked_obligations"])) != known:
        raise ArticleError("incomplete_review_reference_plan")
    queries = string_list(value["queries"])
    if (len(queries) > material["query_limit"] or any(len(q) > 1000 for q in queries)
            or set(queries) & set(material["previous_queries"])):
        raise ArticleError("invalid_review_reference_queries")
    if not isinstance(value["candidates"], list) or len(value["candidates"]) > material["candidate_limit"]:
        raise ArticleError("too_many_review_reference_candidates")
    for row in value["candidates"]:
        require_keys(row, {"metadata", "obligation_ids"})
        if set(string_list(row["obligation_ids"], nonempty=True)) - known:
            raise ArticleError("review_reference_outside_original_obligations")
        if not isinstance(row["metadata"], dict) or not row["metadata"].get("title"):
            raise ArticleError("review_reference_title_required")
        refs.reference_id(row["metadata"])


def findings(report: dict) -> dict:
    result = {key: [finding for finding in row["report"].get("findings", [])
                    if finding.get("severity", "blocking") != "advisory"] for key, row in report.get("receipts", {}).items()}
    result = {key: rows for key, rows in result.items() if rows}
    if report.get("typesetting_findings"):
        result["typesetting"] = report["typesetting_findings"]
    return result


def metrics(article: dict, ledger: dict, report: dict) -> dict:
    used = set().union(*(render.markers(b["text"], "cite") for b in render.walk_blocks(article)))
    coverage = refs.coverage(ledger, used)
    return {"blocking": len(coverage["blocking_missing"]) + len(coverage["blocking_candidates"]),
            "findings": sum(len(rows) for rows in findings(report).values()),
            "review_failures": len(report.get("failures", [])),
            "verified_uses": len({ledger["checks"][cid]["obligation_id"] for cid in ledger["checks"]
                                  if refs.usable_check(ledger, cid)}),
            "missing_evidence": len(report.get("deterministic", {}).get("missing_evidence", []))}


def progressed(before: dict, after: dict) -> bool:
    costs = ("blocking", "findings", "review_failures", "missing_evidence")
    return (all(after[k] <= before[k] for k in costs)
            and (any(after[k] < before[k] for k in costs) or after["verified_uses"] > before["verified_uses"]))


def eligible(checkpoint: dict) -> bool:
    state = checkpoint.get("auto_improvement", {})
    return not state.get("stop_reason") and state.get("completed", 0) < MAX_ROUNDS


class Improvement:
    def __init__(self, pipe):
        self.pipe = pipe
        self.state = pipe.checkpoint.setdefault("auto_improvement", {
            "revision": REVISION, "completed": 0, "stagnant": 0, "phase": "idle", "history": []})
        for key in ("completed", "stagnant"):
            if type(self.state.get(key)) is not int or not 0 <= self.state[key] <= MAX_ROUNDS:
                raise ArticleError("invalid_article_improvement_checkpoint")
        if self.state.get("phase") not in ("idle", "references", "revision", "review", "stopped"):
            raise ArticleError("invalid_article_improvement_phase")

    @property
    def pending(self):
        return self.state["phase"] in ("references", "revision", "review")

    @property
    def directory(self):
        prefix = "improvements/writing-" + str(self.state["writing_revision"]) + "/" if self.state.get("writing_revision") else "improvements/"
        return safe_path(self.pipe.root, prefix + "round-" + str(self.state["completed"] + 1))

    def save(self):
        self.pipe.save_checkpoint()
        write_json(safe_path(self.pipe.root, "improvement-status.json"), {
            **self.state, "max_rounds": self.pipe.config.max_revisions, "max_stagnant_rounds": MAX_STAGNANT_ROUNDS,
            "calls_used": self.pipe.checkpoint["calls"], "call_budget": self.pipe.config.max_calls,
            "elapsed_seconds": self.pipe.checkpoint["elapsed_seconds"],
            "time_budget_seconds": self.pipe.config.total_seconds,
            "source_requests_used": self.pipe.reader.requests, "source_request_budget": self.pipe.config.max_sources,
            "queries_used": literature.focused_queries_used(self.pipe.ledger.get("focus", {})),
            "query_budget": self.pipe.config.max_queries,
            "literature_search_allowed": self.pipe.evidence["permissions"]["literature_search"]})
        self.pipe.update(improvement_round=self.state["completed"] + int(self.pending),
                         improvement_stop_reason=self.state.get("stop_reason"))

    def stop(self, reason: str, detail: str | None = None):
        self.state.update(phase="stopped", stop_reason=reason)
        if detail:
            self.state["detail"] = detail
        self.save()

    def observe(self, article: dict, report: dict):
        if self.state["phase"] == "review":
            current = metrics(article, self.pipe.ledger, report)
            progress = progressed(self.state["before"], current)
            self.state["stagnant"] = 0 if progress else self.state["stagnant"] + 1
            self.state["history"].append({"round": self.state["completed"] + 1,
                "before": self.state["before"], "after": current, "progress": progress,
                "article_digest": digest(article), "review_state": report["state"]})
            self.state["completed"] += 1
            self.state.update(phase="idle", last_article_digest=digest(article), last_report_digest=digest(report))
            self.save()
        if report["state"] == "ready":
            self.stop("ready")

    def start(self, article: dict, report: dict) -> bool:
        p = self.pipe
        if self.state.get("stop_reason"):
            return False
        if not p.config.max_revisions:
            self.stop("automatic_revision_disabled")
        elif self.state["completed"] >= p.config.max_revisions:
            self.stop("round_limit")
        elif self.state["stagnant"] >= MAX_STAGNANT_ROUNDS:
            self.stop("no_progress")
        elif p.checkpoint["calls"] >= p.config.max_calls:
            self.stop("model_call_budget")
        elif p.elapsed_before + time.monotonic() - p.started >= p.config.total_seconds:
            self.stop("time_budget")
        else:
            history = self.directory
            history.mkdir(parents=True, exist_ok=True)
            for name in _FILES:
                path, target = safe_path(p.root, name), history / name
                if path.exists() and not target.exists():
                    shutil.copyfile(path, target)
            write_json(history / "baseline-checkpoint.json", p.checkpoint)
            self.state.update(phase="references", before=metrics(article, p.ledger, report),
                baseline_article_digest=digest(article), baseline_reference_digest=digest(p.ledger),
                plan_applied=False)
            self.save()
            return True
        return False

    def supplement(self, article: dict, report: dict):
        p = self.pipe
        p.update(phase="review_reference_evidence")
        coverage = refs.coverage(p.ledger)
        missing = {oid for oid in coverage["missing"]}
        obligations = {oid: row for oid, row in refs.external_obligations(p.ledger).items()
                       if oid in missing or any(oid in str(rows) for rows in findings(report).values())}
        focus = p.ledger.get("focus", {})
        previous = list(focus.get("searches", {}))
        remaining = max(0, p.config.max_queries - literature.focused_queries_used(focus))
        if not self.state.get("plan_applied"):
            plan = {"checked_obligations": list(obligations), "queries": [], "candidates": []}
            if obligations and p.evidence["permissions"]["literature_search"] and remaining:
                material = {"round": self.state["completed"] + 1, "problem": p.evidence["problem"],
                    "findings": findings(report), "obligations": obligations,
                    "original_units": [p.source_units[uid] for uid in sorted({
                        uid for row in obligations.values() for uid in row["source_units"]})],
                    "references": {rid: row["metadata"] or row["declared"] for rid, row in p.ledger["references"].items()},
                    "unresolved_candidates": coverage["blocking_candidates"], "previous_queries": previous,
                    "query_limit": min(3, remaining), "candidate_limit": min(6, p.config.max_sources)}
                if not p.request_fits("plan_review_references", material, PLAN_CONTRACT):
                    raise ArticleError("article_reference_revision_exceeds_input_budget_review_existing_draft")
                plan = p.ask("plan_review_references", material, PLAN_CONTRACT).value
            write_json(self.directory / "reference-plan.json", plan)
            for candidate in plan["candidates"]:
                rid = refs.add_reference(p.ledger, candidate["metadata"])
                for oid in candidate["obligation_ids"]:
                    refs.bind_reference(p.ledger, oid, rid)
            p.checkpoint["queries"] = list(dict.fromkeys([*previous, *plan["queries"]]))
            if plan["queries"] or plan["candidates"]:
                if focus:
                    focus["selection_done"] = False
                p.checkpoint["references_done"] = False
            self.state["plan_applied"] = True
            p.save_ledger()
            self.save()
        p.check_references()
        self.state["phase"] = "revision"
        self.save()

    def continue_round(self) -> dict:
        p = self.pipe
        baseline = read_json(self.directory / "article.json")
        report = read_json(self.directory / "audit.json")
        ledger = read_json(self.directory / "references.json")
        if (digest(baseline) != self.state["baseline_article_digest"]
                or digest(ledger) != self.state["baseline_reference_digest"]):
            raise EvidenceChanged("article_improvement_baseline_changed")
        refs.validate(ledger, p.evidence)
        if self.state["phase"] == "references":
            self.supplement(baseline, report)
        candidate_path = self.directory / "candidate.json"
        if self.state["phase"] == "revision":
            p.update(phase="review_revision")
            article = copy.deepcopy(baseline)
            article["reference_digest"] = digest(p.ledger)
            from .pipeline import _BLOCK_CONTRACT
            material = {"article": article, "findings": findings(report),
                "review_failures": report.get("failures", []), "round": self.state["completed"] + 1,
                "citation_coverage": refs.coverage(p.ledger), "original_units": list(p.source_units.values()),
                "citations": p.citation_material(), "style": p.style, "publication_plan": p.publication_plan()}
            contract = (_BLOCK_CONTRACT + "Return {article}. Fix the supplied review issues only, preserving all original "
                "mathematics, hypotheses, proof arguments and scope. Insert only supplied verified citation checks "
                "at their actual uses. If evidence is still unavailable, preserve the gap; do not fabricate, "
                "remove obligations, or weaken a claim to force approval. Improve source_map accuracy and remove "
                "only genuine repetition, not distinct proof steps. " + writing.POLICY + writing.CONTRACT)
            if not p.request_fits("revise_article", material, contract):
                raise ArticleError("article_revision_exceeds_input_budget_review_existing_draft")
            reply = p.ask("revise_article", material, contract,
                validation_material={"writing_revision": writing.REVISION, "evidence": p.evidence, "ledger": p.ledger,
                                     "publication_plan": p.publication_plan()})
            write_json(candidate_path, reply.value["article"])
            self.state.update(phase="review", candidate_digest=digest(reply.value["article"]))
            p.checkpoint["review_generation"] = digest(["auto-improvement", REVISION, self.state["completed"] + 1,
                                                        self.state["candidate_digest"], digest(p.ledger)])
            self.save()
        candidate = read_json(candidate_path)
        if digest(candidate) != self.state["candidate_digest"]:
            raise EvidenceChanged("article_improvement_candidate_changed")
        audit.deterministic(candidate, p.evidence, p.ledger)
        render.render(candidate, p.ledger)
        write_json(p.root / "article.json", candidate)
        return candidate

    def abort(self, error: Exception) -> dict | None:
        if not self.pending:
            return None
        p, history = self.pipe, self.directory
        baseline = read_json(history / "article.json")
        ledger = read_json(history / "references.json")
        if digest(baseline) != self.state["baseline_article_digest"] or digest(ledger) != self.state["baseline_reference_digest"]:
            raise EvidenceChanged("article_improvement_baseline_changed")
        refs.validate(ledger, p.evidence)
        checkpoint = read_json(history / "baseline-checkpoint.json")
        for name in _FILES:
            if name == "references.json":
                continue
            source = history / name
            if source.exists():
                shutil.copyfile(source, safe_path(p.root, name))
        p.ledger = ledger
        for name in ("references_done", "queries", "citation_patch_reference_digest", "review_generation"):
            if name in checkpoint:
                p.checkpoint[name] = checkpoint[name]
            else:
                p.checkpoint.pop(name, None)
        p.save_ledger()
        detail = str(error)
        reasons = (("call_budget", "model_call_budget"), ("time_budget", "time_budget"),
                   ("exceeds_input_budget", "revision_input_budget"))
        reason = next((reason for marker, reason in reasons if marker in detail), "requires_intervention")
        self.stop(reason, detail)
        report = read_json(history / "audit.json")
        report.update(state="needs_review", error=str(error), error_type=type(error).__name__)
        if getattr(error, "failures", None):
            report["response_failures"] = error.failures
        write_json(p.root / "audit.json", report)
        return report
