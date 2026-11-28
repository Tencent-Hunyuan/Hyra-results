"""One versioned re-review and bounded local repairs; never reset spent budgets."""
from __future__ import annotations

import shutil

from . import audit, improvement, render, revision_patch
from .models import ArticleError, EvidenceChanged, digest, read_json, safe_path, write_json

MAX_REPAIRS = 2
_FILES = (*improvement._FILES, "status.json", "checkpoint.json", "layout-review.json")


def select(pipe):
    if pipe.upgrade_writing or pipe.refresh_references:
        return None
    state = pipe.checkpoint.get("reassessment")
    if state and state.get("review_revision") == audit.REVIEW_REVISION:
        return Reassessment(pipe)
    path = pipe.root / "audit.json"
    if not path.exists() or not (pipe.root / "article.json").exists():
        return None
    report = read_json(path)
    if report.get("review_revision") == audit.REVIEW_REVISION:
        return None
    if pipe.checkpoint.get("auto_improvement", {}).get("phase") in ("references", "revision", "review"):
        return None
    if not report.get("article_digest"):
        return None
    article = read_json(pipe.root / "article.json")
    if (digest(article) != report["article_digest"] or report["reference_digest"] != digest(pipe.ledger)
            or report["evidence_digest"] != pipe.evidence["digest"]):
        raise EvidenceChanged("reassessment_baseline_binding_changed")
    outputs = {name: (pipe.root / name).read_text() for name in ("paper.tex", "references.tex", "references.bib")}
    if digest(outputs) != report["rendered_digest"]:
        raise EvidenceChanged("reassessment_rendered_baseline_changed")
    history = safe_path(pipe.root, "reassessments/review-" + str(audit.REVIEW_REVISION) + "/original")
    history.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        source = pipe.root / name
        if source.exists() and not (history / name).exists():
            shutil.copyfile(source, history / name)
    pipe.checkpoint["reassessment"] = {"review_revision": audit.REVIEW_REVISION, "phase": "idle", "repairs": 0,
        "original_article_digest": digest(article), "original_calls_spent": pipe.checkpoint["calls"], "history": []}
    pipe.checkpoint["review_generation"] = digest(["reassessment", audit.REVIEW_REVISION, digest(article)])
    pipe.save_checkpoint()
    return Reassessment(pipe)


class Reassessment:
    def __init__(self, pipe):
        self.pipe = pipe
        self.state = pipe.checkpoint["reassessment"]
        if self.state.get("phase") not in ("idle", "patch", "review", "stopped") or not 0 <= self.state.get("repairs", -1) <= MAX_REPAIRS:
            raise ArticleError("invalid_reassessment_checkpoint")

    @property
    def pending(self):
        return self.state["phase"] in ("patch", "review")

    @property
    def directory(self):
        return safe_path(self.pipe.root, "reassessments/review-" + str(audit.REVIEW_REVISION)
                         + "/repair-" + str(self.state["repairs"] + 1))

    @property
    def max_repairs(self):
        return min(MAX_REPAIRS, self.pipe.config.max_revisions)

    def save(self):
        self.pipe.save_checkpoint()
        write_json(self.pipe.root / "reassessment-status.json", dict(self.state, max_repairs=self.max_repairs,
                   calls_used=self.pipe.checkpoint["calls"], call_budget=self.pipe.config.max_calls))

    def stop(self, reason, detail=None):
        self.state.update(phase="stopped", stop_reason=reason)
        if detail:
            self.state["detail"] = detail
        self.save()

    def observe(self, article, report):
        if self.state["phase"] == "review":
            self.state["repairs"] += 1
            self.state["history"].append({"article_digest": digest(article), "review_state": report["state"],
                                           "failures": report.get("failures", [])})
            self.state["phase"] = "idle"
        if report["state"] == "ready":
            self.stop("ready")
        else:
            self.save()

    def start(self, article, report):
        if self.state.get("stop_reason"):
            return False
        if self.state["repairs"] >= self.max_repairs:
            self.stop("targeted_repair_limit")
            return False
        if self.pipe.checkpoint["calls"] + len(audit.review_jobs(article)) + 2 > self.pipe.config.max_calls:
            self.stop("insufficient_remaining_call_budget")
            return False
        if not improvement.findings(report):
            self.stop("requires_source_evidence")
            return False
        history = self.directory
        history.mkdir(parents=True, exist_ok=True)
        for name in _FILES:
            source = self.pipe.root / name
            if source.exists() and not (history / name).exists():
                shutil.copyfile(source, history / name)
        self.state.update(phase="patch", baseline_article_digest=digest(article), baseline_reference_digest=digest(self.pipe.ledger))
        self.save()
        return True

    def continue_round(self):
        p, history = self.pipe, self.directory
        article = read_json(history / "article.json")
        if digest(article) != self.state["baseline_article_digest"] or digest(p.ledger) != self.state["baseline_reference_digest"]:
            raise EvidenceChanged("reassessment_repair_baseline_changed")
        candidate_path = history / "candidate.json"
        if self.state["phase"] == "patch":
            p.update(phase="targeted_revision")
            material = {"article": article, "findings": improvement.findings(read_json(history / "audit.json")),
                "block_versions": {b["id"]: digest(b) for b in render.walk_blocks(article)},
                "original_units": list(p.source_units.values()), "citations": p.citation_material(),
                "citation_coverage": audit.deterministic(article, p.evidence, p.ledger)["citation_coverage"]}
            validation = {"article": article, "evidence": p.evidence, "ledger": p.ledger, "publication_plan": p.publication_plan()}
            if not p.request_fits("revise_article_blocks", material, revision_patch.CONTRACT):
                raise ArticleError("article_targeted_revision_exceeds_input_budget_review_existing_draft")
            reply = p.ask("revise_article_blocks", material, revision_patch.CONTRACT, validation_material=validation)
            candidate = revision_patch.apply(reply.value, validation)
            write_json(history / "patch.json", {"response": reply.value, "session_id": reply.session_id,
                       "before_digest": digest(article), "candidate_digest": digest(candidate)})
            write_json(candidate_path, candidate)
            self.state.update(phase="review", candidate_digest=digest(candidate))
            if candidate == article:
                self.state["stop_reason"] = "no_supported_local_change"
            p.checkpoint["review_generation"] = digest(["targeted-revision", audit.REVIEW_REVISION, self.state["repairs"], candidate])
            self.save()
        candidate = read_json(candidate_path)
        if digest(candidate) != self.state["candidate_digest"]:
            raise EvidenceChanged("reassessment_candidate_changed")
        write_json(p.root / "article.json", candidate)
        return candidate

    def abort(self, error):
        if not self.pending:
            return None
        p, history = self.pipe, self.directory
        article = read_json(history / "article.json")
        ledger = read_json(history / "references.json")
        if digest(article) != self.state["baseline_article_digest"] or digest(ledger) != self.state["baseline_reference_digest"]:
            raise EvidenceChanged("reassessment_fallback_changed")
        for name in improvement._FILES:
            if name == "references.json":
                continue
            source = history / name
            if source.exists():
                shutil.copyfile(source, p.root / name)
        p.ledger = ledger
        p.save_ledger()
        p.compilation = render.current_pdf(p.root)
        p.compilation_sources = render.pdf_sources(p.root)
        report = read_json(history / "audit.json")
        report.update(state="needs_review", error=str(error), error_type=type(error).__name__)
        self.stop("targeted_repair_requires_attention", str(error))
        write_json(p.root / "audit.json", report)
        return report
