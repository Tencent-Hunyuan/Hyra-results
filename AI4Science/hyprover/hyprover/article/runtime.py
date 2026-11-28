"""Detached optional article jobs, bounded checkpoints, and operator controls."""
from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from hyprover.core._util import atomic_write_text

from . import audit, evidence as ev, proof_repair_state, publication, references as refs, render, stage_state
from .agent import backend_profile, validate_backend_profile
from .models import (FORMAT, TERMINAL, ArticleConfig, ArticleError, Cancelled, EvidenceChanged,
                     article_auto_enabled, digest, read_json, read_text, safe_path, text_hash, write_json)
from .pipeline import Pipeline

_RUN = re.compile(r"article-[a-f0-9]{24}\Z")
_DISPATCH = re.compile(r"(?:[a-f0-9]{32}|[a-f0-9]{64})\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")
_OUTPUTS = ("paper.tex", "references.tex", "references.bib")
_REVIEW_MODE_FILE = "review-mode.json"
_LITERATURE_KEY = "HYPROVER_LITERATURE_SEARCH"
_STAGE_KEYS = {"HYPROVER_LITERATURE_SEARCH", "HYPROVER_ARTICLE_AUTO", "HYPROVER_ARTICLE_MODEL",
               "HYPROVER_ARTICLE_EFFORT", "HYPROVER_MAIN_MODEL", "HYPROVER_MAIN_EFFORT",
               "HYPROVER_CODEX_MODEL", "HYPROVER_CODEX_EFFORT"}
_LOG = logging.getLogger(__name__)


def environment(project: Path, overrides: dict | None = None) -> dict:
    env = os.environ.copy()
    literature_disabled = env.get(_LITERATURE_KEY) == "0"
    settings = safe_path(project, "run-settings.json")
    if safe_path(project, ".sequence.json").exists() and settings.exists():
        supplied = read_json(settings)
        env.update({k: v for k, v in supplied.items() if k in _STAGE_KEYS and isinstance(v, str)})
        literature_disabled |= env.get(_LITERATURE_KEY) == "0"
    if overrides:
        env.update({k: v for k, v in overrides.items() if k in _STAGE_KEYS and isinstance(v, str)})
    if literature_disabled:
        env[_LITERATURE_KEY] = "0"
    env.pop("HYPROVER_SEQUENCE_STAGE", None)
    env.pop("HYPROVER_RUN_DEADLINE", None)
    env.pop("HYPROVER_ROLE", None)
    env[_LITERATURE_KEY] = "0" if env.get(_LITERATURE_KEY) == "0" else "1"
    return env


def process_identity(pid: int) -> dict | None:
    if type(pid) is not int or pid < 1:
        return None
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["/bin/ps", "-p", str(pid), "-o", "lstart=", "-o", "stat="],
                capture_output=True, text=True, timeout=2,
                env={"LC_ALL": "C", "TZ": "UTC"},
            )
            started, state = result.stdout.strip().rsplit(None, 1)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None
        if result.returncode or state.startswith("Z"):
            return None
        return {"pid": pid, "start_time": started}
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": stat[19]}
    except (OSError, IndexError, ValueError):
        return None


@contextmanager
def locked(path: Path, *, blocking: bool = True, optional: bool = False):
    path = safe_path(path.parent, path.name)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            if not optional:
                raise
            yield False
            return
        yield True
    finally:
        os.close(fd)


def job_path(project: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not _RUN.fullmatch(run_id):
        raise ArticleError("invalid_article_run_id")
    return safe_path(project, "articles/" + run_id)


def _metadata(project: Path) -> dict:
    path = safe_path(project, "project.json")
    return read_json(path) if path.exists() else {}


def resolve_config(project: Path, *, overrides: dict | None = None) -> ArticleConfig:
    return ArticleConfig.resolve(_metadata(project), environment(project, overrides))


def _job_project(root: Path, run_id: str) -> Path:
    if root.parent.name != "articles" or root != job_path(root.parent.parent, run_id):
        raise ArticleError("invalid_article_manifest")
    return root.parent.parent


def _job_owner(root: Path, manifest: dict, state: dict) -> dict | None:
    historical_root = Path(manifest["project_dir"]) / "articles" / manifest["run_id"]
    return state.get("identity") if state.get("owner_directory", str(historical_root)) == str(root) else None


def _execution_environment(project: Path, manifest: dict, overrides: dict | None = None) -> dict:
    env = environment(project, overrides)
    planned = manifest["policy_environment"][_LITERATURE_KEY]
    if planned != "0" and env[_LITERATURE_KEY] == "0":
        raise ArticleError("article_literature_policy_disallows_frozen_run")
    # A previously disabled job stays disabled even when launched from an enabled shell.
    env[_LITERATURE_KEY] = planned
    return env


def _execution_profile(root: Path, manifest: dict, config: ArticleConfig, env: dict, *, bind_legacy=False) -> dict:
    active = backend_profile(config, env)
    frozen = manifest.get("backend_profile")
    if frozen is None:
        path = safe_path(root, "backend-binding.json")
        if path.exists():
            binding = read_json(path)
            if (binding.get("manifest_fingerprint") != manifest["fingerprint"]
                    or binding.get("historical_routes") != "unverified"):
                raise ArticleError("invalid_article_backend_binding")
            frozen = binding.get("profile")
            validate_backend_profile(config, frozen)
        elif bind_legacy and active["endpoint_sha256"] is not None:
            # Older manifests did not identify the endpoint. Bind future calls without
            # claiming that the endpoint of prior receipts could be reconstructed.
            write_json(path, {"manifest_fingerprint": manifest["fingerprint"], "profile": active,
                             "historical_routes": "unverified"})
    if frozen is not None and frozen != active:
        raise ArticleError("article_backend_route_changed")
    return active


def _validate_review_mode(value: str) -> str:
    if not isinstance(value, str) or value not in audit.REVIEW_MODES:
        raise ArticleError("invalid_article_review_mode")
    return value


def _local_review_mode(root: Path, manifest: dict) -> dict | None:
    path = safe_path(root, _REVIEW_MODE_FILE)
    if not path.exists():
        return None
    selection = read_json(path)
    bindings = selection.get("original_bindings")
    before_status = selection.get("before_status_digest")
    if (selection.get("manifest_fingerprint") != manifest["fingerprint"]
            or selection.get("review_mode") != audit.FORMAT_COVERAGE or not isinstance(bindings, dict)
            or set(bindings) != {"article_digest", "evidence_digest", "reference_digest", "rendered_digest"}
            or bindings["evidence_digest"] != manifest["evidence_digest"]
            or not isinstance(before_status, str) or not _HASH.fullmatch(before_status)
            or any(not isinstance(value, str) or not _HASH.fullmatch(value) for value in bindings.values())):
        raise ArticleError("invalid_article_review_mode_binding")
    return selection


def review_mode(root: Path, manifest: dict | None = None) -> str:
    manifest = manifest or load_manifest(root)
    selection = _local_review_mode(root, manifest)
    return selection["review_mode"] if selection else _validate_review_mode(manifest.get("review_mode", audit.STRICT))


def prepare(project: Path, targets: list[str] | None = None, *, config: ArticleConfig | None = None,
            result_path: str = "main/result.json", completion_root: str | None = None,
            sources: str | None = None, overrides: dict | None = None) -> Path:
    project = Path(project).absolute()
    env = environment(project, overrides)
    config = config or ArticleConfig.resolve(_metadata(project), env)
    frozen = ev.capture(project, targets, config=config, result_path=result_path,
                        completion_root=completion_root, environ=env)
    if sources and not frozen["permissions"]["literature_search"]:
        raise ArticleError("supplied_sources_forbidden_by_literature_policy")
    supplied = refs.supplied_sources(project, sources)
    supplied["local_hashes"] = {}
    for spec in supplied["sources"]:
        if "path" in spec:
            if Path(spec["path"]).parts[:1] != ("literature",):
                raise ArticleError("local_sources_must_be_inside_project_literature")
            supplied["local_hashes"][spec["path"]] = text_hash(
                read_text(safe_path(project, spec["path"]), config.local_source_max_bytes))
    _, style = render.template()
    inputs = {"format": FORMAT, "project_dir": str(project), "evidence_digest": frozen["digest"],
              "config": config.to_dict(), "supplied": supplied, "template": style,
              "policy_environment": {_LITERATURE_KEY: env[_LITERATURE_KEY]},
              "backend_profile": backend_profile(config, env), "review_mode": audit.FORMAT_COVERAGE,
              "proof_repair_revision": proof_repair_state.REVISION}
    fingerprint = digest(inputs)
    run_id = "article-" + fingerprint[:24]
    home = safe_path(project, "articles")
    home.mkdir(exist_ok=True)
    with locked(safe_path(home, ".tasks.lock")):
        root = job_path(project, run_id)
        root.mkdir(exist_ok=True)
        manifest = dict(inputs, run_id=run_id, fingerprint=fingerprint)
        target = safe_path(root, "manifest.json")
        if target.exists():
            if read_json(target) != manifest:
                raise ArticleError("article_manifest_is_immutable")
        else:
            ev.persist(root, frozen)
            write_json(target, manifest)
        if not safe_path(root, "status.json").exists():
            write_json(root / "status.json", {"format": FORMAT, "run_id": run_id, "state": "pending",
                "created_at": time.time(), "updated_at": time.time(), "identity": None, "formal_certificate": False})
    return root


def load_manifest(root: Path) -> dict:
    root = Path(root).absolute()
    manifest = read_json(safe_path(root, "manifest.json"))
    value = {k: v for k, v in manifest.items() if k not in ("run_id", "fingerprint")}
    if (manifest.get("format") != FORMAT or manifest.get("fingerprint") != digest(value)
            or manifest.get("run_id") != "article-" + manifest["fingerprint"][:24]
            or not isinstance(manifest.get("project_dir"), str) or not Path(manifest["project_dir"]).is_absolute()):
        raise ArticleError("invalid_article_manifest")
    _job_project(root, manifest["run_id"])
    _validate_review_mode(manifest.get("review_mode", audit.STRICT))
    if "proof_repair_revision" in manifest and (
            type(manifest["proof_repair_revision"]) is not int
            or manifest["proof_repair_revision"] != proof_repair_state.REVISION):
        raise ArticleError("invalid_article_proof_repair_revision")
    config = ArticleConfig.from_dict(manifest["config"])
    if "backend_profile" in manifest:
        validate_backend_profile(config, manifest["backend_profile"])
    if render.template()[1] != manifest["template"]:
        raise EvidenceChanged("article_template_changed")
    return manifest


def current(root: Path, manifest: dict | None = None, *, mode: str | None = None) -> dict:
    root = Path(root).absolute()
    manifest = manifest or load_manifest(root)
    mode = review_mode(root, manifest) if mode is None else _validate_review_mode(mode)
    frozen = ev.load(root)
    if frozen["digest"] != manifest["evidence_digest"] or frozen["project_dir"] != manifest["project_dir"]:
        raise EvidenceChanged("article_manifest_evidence_changed")
    project = _job_project(root, manifest["run_id"])
    ev.assert_current(frozen, environ=manifest["policy_environment"], project=project)
    for relative, hashed in manifest["supplied"].get("local_hashes", {}).items():
        if text_hash(read_text(safe_path(project, relative))) != hashed:
            raise EvidenceChanged("local_article_source_changed")
    ledger_path = safe_path(root, "references.json")
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        refs.validate(ledger, frozen)
        refs.assert_sources_current(ledger, project, manifest["supplied"])
        plan_path = safe_path(root, "publication-plan.json")
        if plan_path.exists():
            from . import publication
            plan = read_json(plan_path)
            if plan.get("evidence_digest") != frozen["digest"] or plan.get("plan_digest") != digest(plan.get("plan")):
                raise EvidenceChanged("publication_plan_changed")
            publication.validate_plan(plan["plan"], {"original_units": list(ev.units(frozen).values()),
                                      "targets": frozen["targets"], "obligations": ledger["obligations"],
                                      "writing_revision": plan.get("writing_revision", 0)})
            article_path = safe_path(root, "article.json")
            if article_path.exists():
                article = read_json(article_path)
                if article.get("writing_mode") != "segmented":
                    if article.get("publication_plan_digest") != plan["plan_digest"]:
                        raise EvidenceChanged("saved_article_publication_plan_changed")
                    if mode == audit.STRICT:
                        publication.validate_article(article, plan["plan"])
    return frozen


def _manuscript_outputs(root: Path) -> dict:
    return {name: read_text(safe_path(root, name)) for name in _OUTPUTS}


def _report_bindings(article: dict, frozen: dict, ledger: dict, outputs: dict) -> dict:
    return {"article_digest": digest(article), "evidence_digest": frozen["digest"],
            "reference_digest": digest(ledger), "rendered_digest": digest(outputs)}


def _current_review(report: dict, mode: str) -> bool:
    return report.get("review_revision") == audit.REVIEW_REVISION and report.get("review_mode", audit.STRICT) == mode


def _finalize_saved_review(root: Path, article: dict, frozen: dict, ledger: dict, outputs: dict, report: dict,
                           mode: str, *, proof_repair_required: bool = False) -> dict:
    sessions = stage_state.read_checkpoint(root, ledger).get("writer_sessions", []) if mode == audit.STRICT else []
    checked = audit.finalize(article, frozen, ledger, outputs, report.get("receipts", {}), sessions, review_mode=mode)
    if mode == audit.STRICT and report.get("review_mode") == mode:
        checked["review_mode"] = mode
    return proof_repair_state.apply_report(root, article, frozen, checked, required=proof_repair_required)


def _bound_layout(root: Path, report: dict) -> list[dict]:
    path = safe_path(root, "layout-review.json")
    layout = read_json(path) if path.exists() else {}
    if (layout.get("article_digest") == report["article_digest"]
            and layout.get("rendered_digest") == report["rendered_digest"]):
        return layout["findings"]
    return []


def status(project: Path, run_id: str) -> dict:
    root = job_path(Path(project).absolute(), run_id)
    state = read_json(safe_path(root, "status.json"))
    owner = None
    try:
        manifest = load_manifest(root)
        owner = _job_owner(root, manifest, state)
        mode = review_mode(root, manifest)
        state["review_mode"] = mode
        frozen = current(root, manifest, mode=mode)
        ready = state.get("state") == "ready"
        audit_path = safe_path(root, "audit.json")
        draft_review = (mode == audit.FORMAT_COVERAGE and state.get("state") == "needs_review" and audit_path.exists())
        report = read_json(audit_path) if ready or draft_review else {}
        if ready or (draft_review and _current_review(report, mode)):
            article = read_json(safe_path(root, "article.json"))
            ledger = read_json(safe_path(root, "references.json"))
            outputs = _manuscript_outputs(root)
            if not _current_review(report, mode):
                if any(report.get(key) != value for key, value in _report_bindings(article, frozen, ledger, outputs).items()):
                    raise EvidenceChanged("article_outputs_or_audit_changed")
                state.update(state="needs_review", error="review_policy_upgrade_required")
            else:
                required = "proof_repair_revision" in manifest
                checked = _finalize_saved_review(root, article, frozen, ledger, outputs, report, mode,
                    proof_repair_required=required)
                proof_error = checked.get("proof_repair", {}).get("error")
                layout_findings = report.get("typesetting_findings", [])
                recorded = report
                if proof_error:
                    recorded = publication.with_layout_findings(proof_repair_state.apply_report(
                        root, article, frozen, report, required=required), layout_findings)
                if ((ready and checked["state"] != "ready" and not proof_error)
                        or publication.with_layout_findings(checked, layout_findings) != recorded
                        or (mode == audit.STRICT and render.render(article, ledger) != outputs)):
                    raise EvidenceChanged("article_outputs_or_audit_changed")
                layout_report = publication.with_layout_findings(checked, _bound_layout(root, checked))
                if "checks" in report:
                    state["checks"] = layout_report["checks"] if report["state"] == "ready" else report["checks"]
                if "proof_repair" in checked:
                    state["proof_repair"] = checked["proof_repair"]
                if proof_error:
                    state.update(state="needs_review", error=proof_error)
                elif report.get("typesetting_findings") or layout_report.get("typesetting_findings"):
                    state.update(state="needs_review", error="typesetting_requires_revision")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        state.update(state="stale", error=str(exc), formal_certificate=False)
        state.pop("checks", None)
        state.pop("proof_repair", None)
    if state.get("state") in ("queued", "running") and (not owner or process_identity(owner["pid"]) != owner):
        state.update(state="interrupted", identity=None)
    saved_compilation = state.get("compilation") or {}
    compiled = None
    if safe_path(root, "compilation.json").exists() or saved_compilation.get("format") == "hyprover-article-pdf-v1":
        compiled = render.current_pdf(root)
        state["compilation"] = compiled or (saved_compilation if saved_compilation.get("compiled") is False
                                            else {"compiled": False, "error": "stale_or_invalid_pdf"})
    state["pdf_verified"] = compiled is not None and state.get("state") != "stale"
    if state.get("state") == "ready" and publication.layout_severity(state.get("compilation")) == "blocking":
        state.update(state="needs_review", error="typesetting_requires_revision")
        if "checks" in state:
            state["checks"] = dict(state["checks"], format=False)
    improvement_path = safe_path(root, "improvement-status.json")
    if improvement_path.exists():
        state["auto_improvement"] = read_json(improvement_path)
    reassessment_path = safe_path(root, "reassessment-status.json")
    if reassessment_path.exists():
        state["reassessment"] = read_json(reassessment_path)
    return dict(state, output_dir=str(root))


def list_jobs(project: Path) -> list[dict]:
    home = safe_path(Path(project).absolute(), "articles")
    if not home.exists():
        return []
    return [status(project, p.name) for p in sorted(home.iterdir()) if _RUN.fullmatch(p.name) and p.is_dir()]


def _archive_review(root: Path, previous: dict, bindings: dict) -> None:
    history = safe_path(root, "rechecks/" + digest([previous, bindings]))
    history.mkdir(parents=True, exist_ok=True)
    for name in ("audit.json", "article.json", "references.json", *_OUTPUTS):
        source = read_text(safe_path(root, name))
        target = safe_path(history, name)
        if target.exists():
            if read_text(target) != source:
                raise EvidenceChanged("article_recheck_history_changed")
        else:
            atomic_write_text(target, source)
    write_json(safe_path(history, "provenance.json"), {"snapshot_bindings": bindings, "archived_audit_digest": digest(previous),
               "audit_applies_to_snapshot": all(previous.get(key) == value for key, value in bindings.items())})


def _bound_failed_compilation(root: Path, state: dict) -> dict | None:
    current_sources = render.pdf_sources(root)
    path = safe_path(root, "compilation.json")
    entries = [state.get("compilation") or {}]
    if path.exists():
        entries.append(read_json(path))
    return next((entry for entry in entries
                 if entry.get("compiled") is False and entry.get("source_hashes") == current_sources), None)


def _record_compile_sources(root: Path, compilation: dict, source_hashes: dict) -> dict:
    if compilation.get("compiled") is False and source_hashes == render.pdf_sources(root):
        return dict(compilation, source_hashes=source_hashes)
    return compilation


def recheck(project: Path, run_id: str) -> dict:
    """Apply local format and original-source coverage checks to the exact saved manuscript."""
    project = Path(project).absolute()
    root = job_path(project, run_id)
    manifest = load_manifest(root)
    with locked(safe_path(root, ".lock"), blocking=False, optional=True) as acquired:
        if not acquired:
            return dict(read_json(safe_path(root, "status.json")), state="busy", busy=True, output_dir=str(root))
        state = read_json(safe_path(root, "status.json"))
        owner = _job_owner(root, manifest, state)
        if owner and process_identity(owner["pid"]) == owner:
            return dict(state, state="busy", busy=True, output_dir=str(root))
        _local_review_mode(root, manifest)
        return _recheck_locked(root, manifest, state)


def _recheck_locked(root: Path, manifest: dict, state: dict, *, expected_bindings: dict | None = None) -> dict:
    mode = audit.FORMAT_COVERAGE
    frozen = current(root, manifest, mode=mode)
    article = read_json(safe_path(root, "article.json"))
    ledger = read_json(safe_path(root, "references.json"))
    outputs = _manuscript_outputs(root)
    bindings = _report_bindings(article, frozen, ledger, outputs)
    if expected_bindings is not None and expected_bindings != bindings:
        raise EvidenceChanged("article_interrupted_recheck_inputs_changed")
    try:
        report = audit.finalize(article, frozen, ledger, outputs, {}, [], review_mode=mode)
    except (ArticleError, KeyError, TypeError) as exc:
        if str(exc) == "article_rendered_output_mismatch":
            raise EvidenceChanged("article_rendered_sources_changed") from exc
        raise ArticleError(f"article_recheck_format_or_coverage_invalid: {exc}") from exc
    report = proof_repair_state.apply_report(root, article, frozen, report,
        required="proof_repair_revision" in manifest)
    compilation = render.current_pdf(root) or _bound_failed_compilation(root, state)
    layout = publication.layout_findings(root, compilation) if compilation else _bound_layout(root, report)
    report = publication.with_layout_findings(report, layout)
    audit_path = safe_path(root, "audit.json")
    previous = read_json(audit_path) if audit_path.exists() else None
    if previous is not None and previous != report:
        _archive_review(root, previous, bindings)
    write_json(safe_path(root, _REVIEW_MODE_FILE), {"manifest_fingerprint": manifest["fingerprint"],
               "review_mode": mode, "original_bindings": bindings, "before_status_digest": digest(state)})
    if previous != report:
        write_json(audit_path, report)
    draft_path = safe_path(root, "draft-status.json")
    if draft_path.exists():
        draft = read_json(draft_path)
        draft.update(review_state=report["state"], review_mode=mode, checks=report["checks"],
                     draft_complete=report["draft_complete"], article_digest=bindings["article_digest"])
        write_json(draft_path, draft)
    state.update(state=report["state"], phase="finished", identity=None, error=_review_error(report),
                 review_mode=mode, checks=report["checks"], manuscript_available=True,
                 draft_complete=report["draft_complete"], local_recheck_digest=digest(report), updated_at=time.time())
    if "proof_repair" in report:
        state["proof_repair"] = report["proof_repair"]
    if compilation:
        state["compilation"] = compilation
    elif not layout:
        state.pop("compilation", None)
    write_json(root / "status.json", state)
    return status(_job_project(root, manifest["run_id"]), manifest["run_id"])


def _resume_local_review(root: Path, manifest: dict, state: dict) -> bool:
    if not state.get("local_recheck_digest") or state.get("phase") != "finished" or not _local_review_mode(root, manifest):
        return False
    report = read_json(safe_path(root, "audit.json"))
    return _current_review(report, audit.FORMAT_COVERAGE) and digest(report) == state["local_recheck_digest"]


def _resume_rechecked(root: Path, manifest: dict, state: dict, *, compile_output: bool) -> dict | None:
    project = _job_project(root, manifest["run_id"])
    if _resume_local_review(root, manifest, state):
        result = status(project, root.name)
    else:
        selection = _local_review_mode(root, manifest)
        if not selection or digest(state) != selection["before_status_digest"]:
            return None
        result = _recheck_locked(root, manifest, state, expected_bindings=selection["original_bindings"])
    if result.get("proof_repair", {}).get("error") or (
            "proof_repair_revision" in manifest and result["state"] == "stale"):
        return None
    return _compile_locked(project, root, root.name) if compile_output else result


def _repair_typesetting(root: Path, article: dict, frozen: dict, ledger: dict) -> dict:
    import shutil
    fixed = json.loads(json.dumps(article))
    known = ev.units(frozen)
    changes, sources = [], []
    for block in [{"id": "title", **fixed["title"]}, *render.walk_blocks(fixed)]:
        if not any(ord(c) < 32 and c not in "\n\r\t" for c in block["text"]):
            continue
        source = "\n".join(known[uid]["text"] for uid in block["evidence"] if uid in known)
        text, repaired = render.restore_json_escapes(block["text"], source)
        render.safe_tex(text)
        block["text"] = text
        if block["id"] == "title":
            fixed["title"]["text"] = text
        changes.extend(dict(change, block_id=block["id"], source_units=block["evidence"]) for change in repaired)
        sources.append(source)
    if not changes:
        return article
    outputs = render.render(fixed, ledger)
    for name, expected in outputs.items():
        normalized, _ = render.restore_json_escapes(read_text(safe_path(root, name)), "\n".join(sources))
        if normalized != expected:
            raise EvidenceChanged("unrelated_tex_changes_block_typographic_repair")
    history = safe_path(root, "typesetting-revisions/" + digest(article)[:24])
    history.mkdir(parents=True, exist_ok=True)
    for name in ("article.json", "paper.tex", "references.tex", "references.bib", "audit.json", "draft-status.json", "source-map.json"):
        path = safe_path(root, name)
        if path.exists() and not (history / name).exists():
            shutil.copyfile(path, history / name)
    write_json(root / "article.json", fixed)
    render.write(root, outputs)
    if fixed.get("writing_revision"):
        write_json(root / "source-map.json", {"writing_revision": fixed["writing_revision"],
            "article_digest": digest(fixed), "evidence_digest": frozen["digest"], "source_map": fixed["source_map"]})
    repair = {"from_article_digest": digest(article), "article_digest": digest(fixed), "changes": changes,
              "mathematical_text_rewritten": False, "review_required": True}
    write_json(root / "typesetting-repair.json", repair)
    report = read_json(root / "audit.json")
    report.update(state="needs_review", error="typesetting_repaired_requires_review", receipts={}, typesetting_repair=repair)
    write_json(root / "audit.json", report)
    draft_path = safe_path(root, "draft-status.json")
    if draft_path.exists():
        draft = read_json(draft_path)
        draft.update(article_digest=digest(fixed), review_state="needs_review")
        write_json(draft_path, draft)
    state = read_json(root / "status.json")
    state.update(state="needs_review", error="typesetting_repaired_requires_review")
    write_json(root / "status.json", state)
    return fixed


def _review_compilation(root: Path, article: dict, frozen: dict, ledger: dict, outputs: dict,
                        compilation: dict, mode: str, *, proof_repair_required: bool = False) -> dict:
    audit_path = safe_path(root, "audit.json")
    report = read_json(audit_path)
    if _current_review(report, mode) and report.get("typesetting_findings"):
        checked = _finalize_saved_review(root, article, frozen, ledger, outputs, report, mode,
            proof_repair_required=proof_repair_required)
        # A successful recompile only clears a layout blocker whose original
        # audit still reproduces against the exact current manuscript.
        if publication.with_layout_findings(checked, report["typesetting_findings"]) == report:
            report = checked
    report = proof_repair_state.apply_report(root, article, frozen, report, required=proof_repair_required)
    report = publication.review_layout(root, article, outputs, report, compilation)
    write_json(audit_path, report)
    draft_path = safe_path(root, "draft-status.json")
    if draft_path.exists():
        saved = dict(read_json(draft_path), review_state=report["state"])
        if "checks" in report:
            saved["checks"] = report["checks"]
        write_json(draft_path, saved)
    return report


def _review_error(report: dict) -> str | None:
    return (report.get("error") or report.get("proof_repair", {}).get("error")
            or ("typesetting_requires_revision" if report.get("typesetting_findings") else None))


def compile_article(project: Path, run_id: str) -> dict:
    root = job_path(Path(project).absolute(), run_id)
    with locked(safe_path(root, ".lock"), blocking=False, optional=True) as acquired:
        if not acquired:
            return dict(status(project, run_id), busy=True)
        return _compile_locked(project, root, run_id)


def _compile_locked(project: Path, root: Path, run_id: str) -> dict:
    state = status(project, run_id)
    if state["state"] not in ("ready", "needs_review"):
        raise ArticleError("article_manuscript_not_ready_for_compilation")
    manifest = load_manifest(root)
    mode = review_mode(root, manifest)
    saved = read_json(safe_path(root, "status.json"))
    local_review = _resume_local_review(root, manifest, saved)
    frozen = current(root, manifest, mode=mode)
    article = read_json(safe_path(root, "article.json"))
    ledger = read_json(safe_path(root, "references.json"))
    article = _repair_typesetting(root, article, frozen, ledger)
    expected = render.render(article, ledger)
    actual = {name: read_text(safe_path(root, name)) for name in expected}
    if actual != expected:
        legacy = dict(expected)
        legacy["references.tex"] = expected["references.tex"].replace(
            "\n\n\\end{thebibliography}\n", "\n\\end{thebibliography}\n")
        if actual != legacy or actual == expected:
            raise EvidenceChanged("article_rendered_sources_changed")
        history = safe_path(root, "typesetting-revisions/bibliography-paragraph-v1")
        history.mkdir(parents=True, exist_ok=True)
        for name in (*expected, "audit.json", "status.json", "draft-status.json"):
            path = safe_path(root, name)
            if path.exists() and not (history / name).exists():
                atomic_write_text(history / name, read_text(path))
        render.write(root, expected)
        report = read_json(root / "audit.json")
        report.update(state="needs_review", receipts={}, renderer_repair={
            "revision": "bibliography-paragraph-v1", "article_digest": digest(article),
            "mathematical_text_rewritten": False, "review_required": True})
        write_json(root / "audit.json", report)
        saved = read_json(root / "status.json")
        saved.update(state="needs_review", error="bibliography_rendering_repaired_requires_review")
        write_json(root / "status.json", saved)
        draft_path = root / "draft-status.json"
        if draft_path.exists():
            draft = read_json(draft_path)
            draft["review_state"] = "needs_review"
            write_json(draft_path, draft)
    source_hashes = render.pdf_sources(root)
    compilation = render.current_pdf(root) or render.compile_pdf(root)
    compilation = _record_compile_sources(root, compilation, source_hashes)
    saved = read_json(safe_path(root, "status.json"))
    saved.update(compilation=compilation, updated_at=time.time())
    write_json(root / "status.json", saved)
    report = _review_compilation(root, article, frozen, ledger, expected, compilation, mode,
        proof_repair_required="proof_repair_revision" in manifest)
    saved = read_json(safe_path(root, "status.json"))
    if report["state"] == "ready" and state.get("error") == "review_policy_upgrade_required":
        saved.update(state="needs_review", error=state["error"])
    else:
        saved.update(state=report["state"], error=_review_error(report))
    saved.update(compilation=compilation, review_mode=mode, updated_at=time.time(),
                 pdf_verified=render.current_pdf(root) is not None)
    if "checks" in report:
        saved["checks"] = report["checks"]
    if "proof_repair" in report:
        saved["proof_repair"] = report["proof_repair"]
    if local_review:
        saved["local_recheck_digest"] = digest(report)
    write_json(safe_path(root, "status.json"), saved)
    return dict(saved, output_dir=str(root))


def _check_reference_refresh(root: Path) -> None:
    path = safe_path(root, "checkpoint.json")
    if not path.exists():
        return
    checkpoint = stage_state.read_checkpoint(root)
    if (checkpoint.get("auto_improvement", {}).get("phase") in ("references", "revision", "review")
            or checkpoint.get("reassessment", {}).get("phase") in ("patch", "review")):
        raise ArticleError("article_revision_in_progress_resume_without_refresh_references")


def _validate_dispatch_id(dispatch_id: str | None) -> None:
    if dispatch_id is not None and (not isinstance(dispatch_id, str) or not _DISPATCH.fullmatch(dispatch_id)):
        raise ArticleError("invalid_article_dispatch_id")


def launch(root: Path, *, overrides: dict | None = None, resume: bool = False, compile_output: bool = False,
           upgrade_writing: bool = False, refresh_references: bool = False, dispatch_id: str | None = None) -> dict:
    _validate_dispatch_id(dispatch_id)
    root = Path(root).absolute()
    manifest = load_manifest(root)
    project = _job_project(root, manifest["run_id"])
    current(root, manifest)
    with locked(safe_path(root, ".lock"), blocking=False, optional=True) as acquired:
        state = read_json(root / "status.json")
        owner = _job_owner(root, manifest, state)
        alive = bool(owner and process_identity(owner["pid"]) == owner)
        if not acquired:
            return dict(state, state="running" if alive else "busy", output_dir=str(root), busy=True, reused=alive)
        if alive:
            return dict(state, output_dir=str(root), reused=True)
        already_dispatched = dispatch_id is not None and state.get("dispatch_id") == dispatch_id
        if state.get("state") in TERMINAL and (already_dispatched or not resume):
            return status(project, root.name)
        if resume and not (upgrade_writing or refresh_references):
            local_result = _resume_rechecked(root, manifest, state, compile_output=compile_output)
            if local_result is not None:
                return local_result
        if refresh_references:
            _check_reference_refresh(root)
        env = _execution_environment(project, manifest, overrides)
        _execution_profile(root, manifest, ArticleConfig.from_dict(manifest["config"]), env)
        mode = review_mode(root, manifest)
        if resume:
            safe_path(root, "cancel.json").unlink(missing_ok=True)
        source_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        command = [sys.executable, "-m", "hyprover.article", "_run", str(project), "--run-id", root.name]
        if resume:
            command.append("--resume")
        if compile_output:
            command.append("--compile")
        if upgrade_writing:
            command.append("--upgrade-writing")
        if refresh_references:
            command.append("--refresh-references")
        if dispatch_id is not None:
            command.extend(("--dispatch-id", dispatch_id))
        with safe_path(root, "run.log").open("a") as log:
            proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        state.pop("dispatch_id", None)
        state.pop("local_recheck_digest", None)
        state.pop("checks", None)
        state.pop("proof_repair", None)
        if dispatch_id is not None:
            state["dispatch_id"] = dispatch_id
        state.update(state="queued", identity=process_identity(proc.pid), owner_directory=str(root),
                     updated_at=time.time(), error=None, review_mode=mode)
        write_json(root / "status.json", state)
        threading.Thread(target=proc.wait, daemon=True).start()
        return dict(state, output_dir=str(root))


def work(project: Path, run_id: str, *, resume: bool = False, agent=None, compile_output: bool = False,
         wait_for_launch: bool = False, upgrade_writing: bool = False, refresh_references: bool = False,
         dispatch_id: str | None = None) -> dict:
    _validate_dispatch_id(dispatch_id)
    project = Path(project).absolute()
    root = job_path(project, run_id)
    manifest = load_manifest(root)
    interrupted = threading.Event()
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, lambda *_: interrupted.set())
    try:
        with locked(safe_path(root, ".lock"), blocking=wait_for_launch):
            state = read_json(root / "status.json")
            already_dispatched = dispatch_id is not None and state.get("dispatch_id") == dispatch_id
            if state["state"] in TERMINAL and (already_dispatched or not resume):
                return status(project, run_id)
            if resume and not (upgrade_writing or refresh_references):
                local_result = _resume_rechecked(root, manifest, state, compile_output=compile_output)
                if local_result is not None:
                    return local_result
            if refresh_references:
                _check_reference_refresh(root)
            env = _execution_environment(project, manifest)
            config = ArticleConfig.from_dict(manifest["config"])
            profile = _execution_profile(root, manifest, config, env, bind_legacy=agent is None)
            mode = review_mode(root, manifest)
            if resume and not wait_for_launch:
                safe_path(root, "cancel.json").unlink(missing_ok=True)
            def update(**fields):
                state.update(fields, updated_at=time.time())
                write_json(root / "status.json", state)
            if not wait_for_launch or state.get("owner_directory") != str(root) or dispatch_id is not None:
                state.pop("dispatch_id", None)
            if dispatch_id is not None:
                state["dispatch_id"] = dispatch_id
            state.pop("local_recheck_digest", None)
            state.pop("checks", None)
            state.pop("proof_repair", None)
            update(state="running", phase="evidence", identity=process_identity(os.getpid()), owner_directory=str(root),
                   error=None, review_mode=mode)
            try:
                frozen = current(root, manifest, mode=mode)
                pipe = Pipeline(root, frozen, config, supplied=manifest["supplied"],
                    agent=agent, update=update, environ=env, project=project, backend_profile=profile, upgrade_writing=upgrade_writing,
                    refresh_references=refresh_references, compile_output=compile_output, review_mode=mode,
                    proof_repair_required="proof_repair_revision" in manifest,
                    check_cancel=lambda: interrupted.is_set() or safe_path(root, "cancel.json").exists())
                report = pipe.run()
                current(root, manifest, mode=mode)
                if compile_output and report["state"] in ("ready", "needs_review") and (root / "paper.tex").exists():
                    update(phase="compilation")
                    source_hashes = render.pdf_sources(root)
                    compiled = pipe.compilation if pipe.compilation_sources == source_hashes else None
                    if compiled is None:
                        compiled = render.current_pdf(root) or render.compile_pdf(
                            root, deadline=pipe.deadline, cancelled=pipe.check_cancel)
                        compiled = _record_compile_sources(root, compiled, source_hashes)
                        update(compilation=compiled)
                        article = read_json(safe_path(root, "article.json"))
                        ledger = read_json(safe_path(root, "references.json"))
                        report = _review_compilation(root, article, frozen, ledger, render.render(article, ledger), compiled, mode,
                            proof_repair_required="proof_repair_revision" in manifest)
                    else:
                        compiled = _record_compile_sources(root, compiled, source_hashes)
                    update(compilation=compiled, pdf_verified=render.current_pdf(root) is not None)
                if "checks" in report:
                    state["checks"] = report["checks"]
                if "proof_repair" in report:
                    state["proof_repair"] = report["proof_repair"]
                update(state=report["state"], phase="finished", identity=None, error=_review_error(report))
            except Cancelled as exc:
                update(state="cancelled", identity=None, error=str(exc))
            except EvidenceChanged as exc:
                update(state="stale", identity=None, error=str(exc))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                update(state="failed", identity=None, error=str(exc))
            return dict(state, output_dir=str(root))
    except BlockingIOError:
        return dict(read_json(root / "status.json"), output_dir=str(root), busy=True)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def stop(project: Path, run_id: str) -> dict:
    root = job_path(Path(project).absolute(), run_id)
    manifest = load_manifest(root)
    write_json(safe_path(root, "cancel.json"), {"requested_at": time.time()})
    state = read_json(root / "status.json")
    owner = _job_owner(root, manifest, state)
    if (hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
            and owner and owner["pid"] != os.getpid() and process_identity(owner["pid"]) == owner):
        try:
            fd = os.pidfd_open(owner["pid"])
            try:
                if process_identity(owner["pid"]) == owner:
                    signal.pidfd_send_signal(fd, signal.SIGTERM)
            finally:
                os.close(fd)
        except (ProcessLookupError, PermissionError):
            pass
    return {"run_id": run_id, "cancel_requested": True, "previous_state": state.get("state")}


def maybe_start(project: Path, *, stage_environment: dict | None = None, sequence: bool = False) -> dict:
    """Best-effort integration; failure never changes a research completion result."""
    if os.environ.get("HYPROVER_ARTICLE_AUTO") == "0":
        return {"state": "disabled"}
    if os.environ.get("HYPROVER_SEQUENCE_STAGE") and not sequence:
        return {"state": "deferred_to_sequence"}
    if stage_environment and stage_environment.get("HYPROVER_ARTICLE_AUTO") == "0":
        return {"state": "disabled"}
    try:
        metadata = _metadata(Path(project))
        env = environment(Path(project), stage_environment)
        config = ArticleConfig.resolve(metadata, env)
        if not config.auto_generate:
            return {"state": "disabled"}
        root = prepare(Path(project), config=config, overrides=stage_environment)
        return launch(root, overrides=stage_environment)
    except Exception as exc:
        _LOG.warning("Optional article generation failed (%s)", type(exc).__name__)
        return {"state": "failed", "error_type": type(exc).__name__}
