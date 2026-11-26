"""Export only reviewable manuscripts and public provenance, never model sessions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from hyprover.core._util import atomic_write_text, fsync_directory

from . import audit, evidence, proof_repair_state, references, render, runtime, writing_policy as writing
from .models import FORMAT, ArticleError, digest, json_document, read_json, read_text, safe_path, text_hash, write_json

_ARTIFACT_FORMAT = "hyprover-article-artifacts-v1"
_VERSION_ROOT = ".article-versions"
_ARTIFACT_MANIFEST = "artifact-manifest.json"
_CURRENT_POINTER = "current.json"
_TEXT_OUTPUTS = ("paper.tex", "references.tex", "references.bib")
_REQUIRED_ARTIFACTS = frozenset((*_TEXT_OUTPUTS, "article.json", "audit.json", "citation-audit.json", "provenance.json"))
_OPTIONAL_ARTIFACTS = frozenset(("paper.pdf", "compilation.json", "source-map.json"))
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _copy_artifact(source: Path, target: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=".article-copy-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as original:
            shutil.copyfileobj(original, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _verified_version(target: Path, version: str) -> tuple[Path, dict]:
    if not isinstance(version, str) or not _SHA256.fullmatch(version):
        raise ArticleError("invalid_article_artifact_version")
    root = safe_path(target, _VERSION_ROOT + "/" + version)
    manifest = read_json(safe_path(root, _ARTIFACT_MANIFEST))
    files = manifest.get("files")
    if (manifest.get("format") != _ARTIFACT_FORMAT or not isinstance(files, dict)
            or not _REQUIRED_ARTIFACTS <= files.keys() <= _REQUIRED_ARTIFACTS | _OPTIONAL_ARTIFACTS
            or ("paper.pdf" in files) != ("compilation.json" in files)
            or digest(files) != version or set(p.name for p in root.iterdir()) != set(files) | {_ARTIFACT_MANIFEST}):
        raise ArticleError("invalid_article_artifact_manifest")
    if any(_file_hash(safe_path(root, name)) != wanted for name, wanted in files.items()):
        raise ArticleError("article_artifact_snapshot_changed")
    provenance = read_json(safe_path(root, "provenance.json"))
    if (provenance.get("files") != sorted(files) or provenance.get("pdf_compiled") != ("paper.pdf" in files)
            or provenance.get("output_hashes") != {name: files[name] for name in _TEXT_OUTPUTS}
            or ("paper.pdf" in files and provenance.get("pdf_sha256") != files["paper.pdf"])):
        raise ArticleError("article_artifact_provenance_mismatch")
    return root, files


def current_artifacts(target: Path) -> tuple[Path, dict]:
    """Return the verified current snapshot and hashes, independent of flat mirrors."""
    root, files, _, _ = _read_current(target)
    return root, files


def _read_current(target: Path) -> tuple[Path, dict, str, str]:
    pointer_text = read_text(safe_path(target, _CURRENT_POINTER))
    pointer = json.loads(pointer_text)
    if not isinstance(pointer, dict):
        raise ArticleError("invalid_article_artifact_pointer")
    if pointer.get("format") != _ARTIFACT_FORMAT:
        raise ArticleError("invalid_article_artifact_pointer")
    root, files = _verified_version(target, pointer.get("version"))
    manifest_hash = _file_hash(safe_path(root, _ARTIFACT_MANIFEST))
    if manifest_hash != pointer.get("manifest_sha256"):
        raise ArticleError("article_artifact_manifest_changed")
    return root, files, text_hash(pointer_text), manifest_hash


def _archive_compatibility_export(target: Path, replacement: dict) -> None:
    if safe_path(target, _CURRENT_POINTER).exists():
        previous, artifact_hashes = current_artifacts(target)
        filenames = artifact_hashes.keys()
    else:
        previous, filenames = target, _REQUIRED_ARTIFACTS | _OPTIONAL_ARTIFACTS
    old_path = safe_path(previous, "provenance.json")
    if not old_path.exists():
        return
    old = read_json(old_path)
    if all(old.get(key) == replacement.get(key) for key in ("article_digest", "audit_digest")):
        return
    history = safe_path(target, "history/" + digest(old)[:24])
    history.mkdir(parents=True, exist_ok=True)
    for name in sorted(filenames):
        source, archived = safe_path(previous, name), safe_path(history, name)
        if source.is_file() and not archived.exists():
            _copy_artifact(source, archived)


def _publish_artifacts(target: Path, outputs: dict, documents: dict, source_pdf: Path | None) -> str:
    target.mkdir(exist_ok=True)
    fsync_directory(target.parent)
    with runtime.locked(safe_path(target, ".article-delivery.lock")):
        if source_pdf is None and safe_path(target, "paper.pdf").exists():
            raise ArticleError("existing_delivered_pdf_is_not_current")
        versions = safe_path(target, _VERSION_ROOT)
        versions.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".staging-", dir=versions) as work:
            staging = Path(work)
            for name, text in outputs.items():
                atomic_write_text(staging / name, text)
            for name, data in documents.items():
                write_json(staging / name, data)
            if source_pdf is not None:
                _copy_artifact(source_pdf, staging / "paper.pdf")
            files = {p.name: _file_hash(p) for p in staging.iterdir()}
            version = digest(files)
            write_json(staging / _ARTIFACT_MANIFEST, {"format": _ARTIFACT_FORMAT, "files": files})
            target_version = safe_path(versions, version)
            if not target_version.exists():
                os.replace(staging, target_version)
                fsync_directory(versions)
            verified, verified_hashes = _verified_version(target, version)
            if verified_hashes != files:
                raise ArticleError("article_artifact_snapshot_changed")
            _archive_compatibility_export(target, documents["provenance.json"])
            write_json(safe_path(target, _CURRENT_POINTER), {"format": _ARTIFACT_FORMAT, "version": version,
                "manifest_sha256": _file_hash(safe_path(verified, _ARTIFACT_MANIFEST))})
            # Existing consumers still receive ordinary files. The committed snapshot
            # and its pointer are authoritative if these compatibility copies lag.
            for name, wanted in files.items():
                mirror = safe_path(target, name)
                if not mirror.is_file() or _file_hash(mirror) != wanted:
                    _copy_artifact(safe_path(verified, name), mirror)
            return version


def export(project: Path, run_id: str, destination: str = "paper") -> dict:
    project = project.absolute()
    root = runtime.job_path(project, run_id)
    with runtime.locked(safe_path(root, ".lock"), blocking=False, optional=True) as acquired:
        if not acquired:
            raise ArticleError("article_busy_for_delivery")
        return _export(project, root, run_id, destination)


def _export(project: Path, root: Path, run_id: str, destination: str) -> dict:
    state = runtime.status(project, run_id)
    if state["state"] not in ("ready", "needs_review"):
        raise ArticleError("article_not_available_for_delivery")
    manifest = runtime.load_manifest(root)
    review_mode = state.get("review_mode", manifest.get("review_mode", audit.STRICT))
    if not isinstance(review_mode, str) or review_mode not in audit.REVIEW_MODES:
        raise ArticleError("invalid_article_review_mode")
    frozen = evidence.load(root)
    ledger = read_json(safe_path(root, "references.json"))
    report = read_json(safe_path(root, "audit.json"))
    if report.get("review_mode", audit.STRICT) != review_mode:
        raise ArticleError("article_delivery_review_mode_mismatch")
    if report.get("policy_revision") != references.policy.POLICY_REVISION:
        raise ArticleError("article_requires_current_policy_review")
    article = read_json(safe_path(root, "article.json"))
    if writing.revision(article) and report.get("writing_revision") != writing.revision(article):
        raise ArticleError("article_requires_current_writing_review")
    report = proof_repair_state.apply_report(root, article, frozen, report,
        required="proof_repair_revision" in manifest)
    if report["state"] != "ready":
        state["state"] = "needs_review"
    outputs = {name: read_text(safe_path(root, name)) for name in _TEXT_OUTPUTS}
    if render.render(article, ledger) != outputs:
        raise ArticleError("delivered_article_does_not_match_rendered_manuscript")
    compiled = render.current_pdf(root)
    documents = {"article.json": article, "audit.json": report}
    if writing.revision(article):
        documents["source-map.json"] = {"writing_revision": writing.REVISION,
            "article_digest": digest(article), "evidence_digest": frozen["digest"], "source_map": article["source_map"]}
    citations = {cid for block in render.walk_blocks(article) for cid in block["citations"]}
    bibliography = {rid: {key: value for key, value in row.items() if key not in ("source_ids", "metadata_sources")}
                    for rid, row in ledger["references"].items()}
    source_index = {sid: {key: source[key] for key in ("origin", "kind", "version", "sha256", "retrieved_at")}
                    for sid, source in ledger["sources"].items()}
    citation_report = {"format": FORMAT, "review_mode": review_mode, "coverage": references.coverage(ledger, citations),
        "references": bibliography, "obligations": ledger["obligations"], "checks": ledger["checks"],
        "exclusions": ledger["exclusions"], "source_index": source_index}
    documents["citation-audit.json"] = citation_report
    result = {"format": FORMAT, "run_id": run_id, "state": state["state"], "review_mode": review_mode,
        "formal_certificate": False,
        "author": manifest["template"]["author"], "model": manifest["config"]["model"],
        "requested_effort": manifest["config"]["effort"], "template": manifest["template"],
        "evidence_digest": frozen["digest"], "article_digest": digest(article), "target_fact_ids": frozen["targets"],
        "writing_revision": writing.revision(article), "audit_digest": digest(report),
        "references_revision": ledger.get("focus", {}).get("revision", 0),
        "completion_evidence_digest": frozen["completion"]["manifest_digest"] if frozen["completion"] else None,
        "dependency_fact_ids": [fact["fact_id"] for fact in frozen["facts"]],
        "output_hashes": {name: text_hash(text) for name, text in outputs.items()},
        "pdf_compiled": False, "model_sessions_exported": False, "full_external_sources_exported": False,
        "review_meaning": (
            "Ready means automated format and evidence-coverage checks passed; proof fidelity and citation adequacy require editorial review."
            if review_mode == audit.FORMAT_COVERAGE else
            "Ready means automated editorial checks passed; qualified human review is still required.")}
    if "proof_repair" in report:
        result["proof_repair"] = report["proof_repair"]
        result["review_meaning"] += " " + proof_repair_state.READY_MEANING
    files = [*outputs, *documents, "provenance.json"]
    source_pdf = None
    if compiled:
        source_pdf = safe_path(root, "paper.pdf")
        documents["compilation.json"] = {k: v for k, v in compiled.items() if k not in ("path", "log_path")}
        result.update(pdf_compiled=True, pdf_sha256=compiled["pdf_sha256"], pdf_pages=compiled["pages"])
        files.extend(("paper.pdf", "compilation.json"))
    result["files"] = sorted(files)
    documents["provenance.json"] = result
    version = _publish_artifacts(safe_path(project, destination), outputs, documents, source_pdf)
    return dict(result, artifact_version=version)


def collect(root: Path) -> tuple[dict, list[str]]:
    with runtime.locked(safe_path(root, ".article-publication.lock")):
        return _collect(root)


def _batch_artifacts(root: Path, target: Path, version: str) -> dict[str, str]:
    snapshot, files, pointer_hash, manifest_hash = _read_current(target)
    if snapshot.name != version:
        raise ArticleError("article_artifact_changed_during_delivery")
    result = {}
    for name, wanted in files.items():
        mirror = safe_path(target, name)
        if _file_hash(mirror) != wanted:
            raise ArticleError("article_artifact_mirror_changed")
        result[str(mirror.relative_to(root))] = wanted
        result[str(safe_path(snapshot, name).relative_to(root))] = wanted
    result[str(safe_path(snapshot, _ARTIFACT_MANIFEST).relative_to(root))] = manifest_hash
    result[str(safe_path(target, _CURRENT_POINTER).relative_to(root))] = pointer_hash
    return result


def _collect(root: Path) -> tuple[dict, list[str]]:
    generation = read_json(safe_path(root, "article-generation.json"))
    if not generation.get("complete"):
        raise ArticleError("article_collection_still_running")
    inventory = read_json(safe_path(root, "export-manifest.json"))
    expected = {row["benchmark"]: row for row in inventory["projects"]}
    if set(generation["projects"]) != set(expected):
        raise ArticleError("article_collection_members_changed")
    result = {"format": FORMAT, "projects": {}, "formal_certificate": False,
              "review_mode": generation.get("review_mode", audit.STRICT),
              "all_ready": False, "completed_at": time.time()}
    file_hashes = {}
    for identifier, record in generation["projects"].items():
        project = safe_path(root, expected[identifier]["directory"])
        row = {"state": record["state"], "run_id": record.get("run_id"), "manuscript_available": False,
               "review_mode": record.get("review_mode", result["review_mode"])}
        if record["state"] in ("ready", "needs_review"):
            try:
                provenance = export(project, record["run_id"])
                target = safe_path(project, "paper")
                with runtime.locked(safe_path(target, ".article-delivery.lock")):
                    delivered = _batch_artifacts(root, target, provenance["artifact_version"])
                file_hashes.update(delivered)
                row.update(state=provenance["state"], manuscript_available=True,
                           review_mode=provenance["review_mode"], pdf_compiled=provenance["pdf_compiled"],
                           directory=str(target.relative_to(root)), artifact_version=provenance["artifact_version"],
                           evidence_digest=provenance["evidence_digest"], article_digest=provenance["article_digest"])
            except (ValueError, OSError, KeyError, TypeError) as exc:
                row.update(state="delivery_blocked", error_type=type(exc).__name__, error=str(exc))
        else:
            row["error"] = record.get("error")
        result["projects"][identifier] = row
    result["all_ready"] = all(row["state"] == "ready" for row in result["projects"].values())
    result["file_hashes"] = file_hashes
    write_json(safe_path(root, "article-delivery.json"), result)
    return result, [*file_hashes, "article-delivery.json"]


def check_public_files(root: Path, paths: list[str]) -> None:
    secrets = [value for name, value in os.environ.items() if value and len(value) >= 12
               and any(part in name.upper() for part in ("TOKEN", "API_KEY", "PASSWORD", "SECRET"))]
    for name in paths:
        path = safe_path(root, name)
        if path.suffix.lower() == ".pdf":
            if path.stat().st_size > 33554432:
                raise ArticleError("article_pdf_too_large")
            text = path.read_bytes().decode("latin-1")
        else:
            text = read_text(path)
        if any(value in text for value in secrets):
            raise ArticleError("credential_detected_in_article_delivery")
        if re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|Bearer\s+[A-Za-z0-9._-]{16,}", text):
            raise ArticleError("sensitive_material_detected_in_article_delivery")


def commit_and_push(root: Path, paths: list[str], *, file_hashes: dict[str, str] | None = None) -> str:
    def git(*args, check=True, binary=False):
        return subprocess.run(["git", "-C", str(root), *args], check=check, capture_output=True,
                              text=not binary, timeout=120,
                              env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LITERAL_PATHSPECS="1"))
    if git("branch", "--show-current").stdout.strip() != "master":
        raise ArticleError("article_delivery_requires_master_branch")
    if git("diff", "--cached", "--quiet", check=False).returncode:
        raise ArticleError("existing_staged_changes_block_article_delivery")
    check_public_files(root, paths)
    base = git("rev-parse", "HEAD").stdout.strip()
    remote_ref = "refs/heads/master"
    remote = git("ls-remote", "--exit-code", "origin", remote_ref).stdout.split()
    if remote != [base, remote_ref]:
        raise ArticleError("article_delivery_requires_master_at_remote_tip")
    if not file_hashes or set(paths) != set(file_hashes):
        raise ArticleError("article_delivery_requires_artifact_hashes")
    for path, wanted in file_hashes.items():
        if _file_hash(safe_path(root, path)) != wanted:
            raise ArticleError("article_delivery_changed_before_stage")
    git("add", "--", *paths)

    verified_blobs = {}

    def verify_artifacts(tree: str):
        if tree:
            listed = git("ls-tree", "-z", tree, "--", *paths).stdout
        else:
            listed = git("ls-files", "--stage", "-z", "--", *paths).stdout
        index = {}
        for entry in filter(None, listed.split("\0")):
            meta, name = entry.split("\t", 1)
            if tree:
                mode, kind, blob_id = meta.split()
            else:
                mode, blob_id, stage = meta.split()
                kind = "blob" if stage == "0" else "conflict"
            if name in index or mode not in ("100644", "100755") or kind != "blob":
                raise ArticleError("article_delivery_requires_regular_artifacts")
            index[name] = blob_id
        if index.keys() != file_hashes.keys():
            raise ArticleError("article_delivery_changed_stage_paths")
        for path, blob_id in index.items():
            if blob_id not in verified_blobs:
                blob = git("cat-file", "blob", blob_id, binary=True).stdout
                verified_blobs[blob_id] = hashlib.sha256(blob).hexdigest()
            if verified_blobs[blob_id] != file_hashes[path]:
                raise ArticleError("article_delivery_staged_bytes_changed")

    try:
        staged = set(filter(None, git("diff", "--cached", "--name-only", "-z").stdout.split("\0")))
        if staged - set(paths):
            raise ArticleError("unrelated_staged_changes_block_article_delivery")
        verify_artifacts("")
        git("diff", "--cached", "--check")
    except (ArticleError, subprocess.SubprocessError):
        git("reset", "-q", base, "--", *paths, check=False)
        raise
    if staged:
        git("commit", "-m", "Add generated manuscripts and explicit editorial review outcomes")
    commit = git("rev-parse", "HEAD").stdout.strip()
    if staged:
        parent = git("rev-list", "--parents", "-n", "1", commit).stdout.split()
        changed = set(filter(None, git("diff", "--name-only", "-z", base, commit).stdout.split("\0")))
        if parent != [commit, base] or changed != staged:
            raise ArticleError("article_delivery_commit_does_not_match_stage")
        verify_artifacts(commit)
        git("push", "origin", f"{commit}:{remote_ref}")
    return commit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--push", action="store_true", help="Explicitly authorize committing and pushing the master delivery")
    args = parser.parse_args()
    root = args.root.absolute()
    deadline = time.monotonic() + 7 * 86400
    try:
        while args.watch and not read_json(safe_path(root, "article-generation.json")).get("complete"):
            if time.monotonic() >= deadline:
                raise ArticleError("article_delivery_watch_deadline")
            time.sleep(30)
        with runtime.locked(safe_path(root, ".article-publication.lock")):
            result, paths = _collect(root)
            publication = {"state": "local", "all_ready": result["all_ready"],
                           "review_mode": result["review_mode"], "files": paths}
            if args.push:
                hashes = dict(result["file_hashes"])
                hashes["article-delivery.json"] = text_hash(json_document(result))
                publication.update(state="pushed", commit=commit_and_push(root, paths, file_hashes=hashes))
            write_json(safe_path(root, "article-publication.json"), publication)
        print(json.dumps(publication, ensure_ascii=False), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        failure = {"state": "blocked", "error_type": type(exc).__name__,
                   "error": str(exc) if not isinstance(exc, subprocess.SubprocessError) else "git_delivery_command_failed"}
        write_json(safe_path(root, "article-publication.json"), failure)
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
