"""Verify relocated public exports without modifying historical completion evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path

from hyprover.core import KnowledgeStore
from hyprover.verify import completion_evidence as ce

from .models import ArticleError, digest, read_json, safe_path


def _verified_inventory(project: Path, inventory: list, required: set[str] | None = None) -> dict:
    if not isinstance(inventory, list) or not inventory:
        raise ArticleError("archive_file_inventory_missing")
    seen, indexed = set(), {}
    for item in inventory:
        relative = item.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise ArticleError("invalid_archive_inventory")
        seen.add(relative)
        if required is not None and relative not in required:
            continue
        path = safe_path(project, relative)
        if path.stat().st_size != item.get("bytes"):
            raise ArticleError("archive_file_size_changed: " + relative)
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()
        if checksum != item.get("sha256"):
            raise ArticleError("archive_file_content_changed: " + relative)
        indexed[relative] = checksum
    if required is not None and required - indexed.keys():
        raise ArticleError("archive_required_evidence_not_in_inventory")
    return indexed


def _validate_repository_export(project: Path, export: dict, result_path: str, result: dict,
                                root_override: str | None) -> tuple[Path, dict, dict]:
    migration = read_json(safe_path(project, "migration-manifest.json"))
    source = export.get("source_directory")
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise ArticleError("archive_original_project_identity_missing")
    benchmark = migration.get("benchmark")
    inventory = migration.get("files")
    if (migration.get("format") != "hyprover-repository-migration-v1"
            or migration.get("source_project") != export.get("project")
            or export.get("project") != Path(source).name
            or not isinstance(benchmark, str) or not benchmark or Path(benchmark).name != benchmark
            or migration.get("method", {}).get("target_prefix") != benchmark + "/"
            or not isinstance(inventory, list) or len(inventory) != migration.get("integrity", {}).get("tracked_files")):
        raise ArticleError("invalid_archive_inventory")
    completion = export.get("completion", {})
    migrated = migration.get("completion", {})
    report = completion.get("report")
    target = completion.get("target_fact_id")
    if (completion.get("result") != result_path or not isinstance(report, str)
            or not isinstance(target, str) or result.get("target_fact_ids") != [target]
            or completion.get("accepted") is not True or completion.get("formal_certificate") is not False
            or migrated.get("accepted") is not True or migrated.get("formal_certificate") is not False
            or migrated.get("target_fact_id") != target
            or migrated.get("source_export_manifest") != f"{benchmark}/export-manifest.json"
            or migrated.get("result") != f"{benchmark}/{result_path}" or migrated.get("report") != f"{benchmark}/{report}"
            or migrated.get("frozen_evidence_digest") != result.get("completion_evidence_digest")):
        raise ArticleError("archive_completion_binding_mismatch")
    report_path = safe_path(project, report)
    root = report_path.parent
    run_id = result["full_revalidation_run"]
    original_root = export.get("layout", {}).get("original_verification_root")
    if root_override is not None and Path(root_override).absolute() != root:
        raise ArticleError("archive_completion_root_mismatch")
    if (not isinstance(original_root, str) or not Path(original_root).is_absolute()
            or result.get("completion_report") != str(Path(original_root) / run_id / "completion-report.json")
            or root.name != run_id or report_path.name != "completion-report.json"):
        raise ArticleError("archive_completion_path_mismatch")
    root_relative = root.relative_to(project)
    required = {"export-manifest.json", "KNOWLEDGE.md", "PROBLEM.md", "project.json", result_path, report,
                str(root_relative / "completion-manifest.json"), str(root_relative / "completion-progress.json")}
    manifest = ce.validate_completion(root, project, result=result, original_project=Path(source))
    if run_id != "completion-" + manifest["digest"]:
        raise ArticleError("archive_completion_identity_mismatch")
    public_facts = KnowledgeStore(project).snapshot().facts
    for fact in manifest["facts"]:
        required.add(public_facts[fact["fact_id"]].proof_path)
        required.add(str(root_relative / fact["proof_file"]))
    indexed = _verified_inventory(project, inventory, required)
    binding = {"export_digest": digest(export), "migration_digest": digest(migration), "source_directory": source,
               "inventory_digest": digest(indexed), "relocation": "project_directory_only",
               "historical_receipts_modified": False}
    return root, manifest, binding


def validate(project: Path, result_path: str, result: dict, root_override: str | None) -> tuple[Path, dict, dict]:
    export = read_json(safe_path(project, "export-manifest.json"))
    if export.get("format") != "hyprover-original-project-export-v1":
        if "format" not in export and export.get("export_type") == "complete_research_project_snapshot":
            return _validate_repository_export(project, export, result_path, result, root_override)
        raise ArticleError("unsupported_article_archive")
    source = export.get("source_directory")
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise ArticleError("archive_original_project_identity_missing")
    completion = export.get("completion", {})
    if completion.get("result") != result_path or completion.get("accepted") is not True or completion.get("outcome") != "passed":
        raise ArticleError("archive_completion_binding_mismatch")
    report_path = safe_path(project, completion["report"])
    root = report_path.parent
    if root_override is not None and Path(root_override).absolute() != root:
        raise ArticleError("archive_completion_root_mismatch")
    run_id = result["full_revalidation_run"]
    layout = export.get("layout", {})
    original_root = layout.get("original_verification_root")
    inventory = export.get("files")
    if inventory is None and layout.get("file_inventory_hashes_and_exclusions") == "export-inventory.json":
        detached = read_json(safe_path(project, "export-inventory.json"))
        inventory = detached.get("included_source_files")
        if (detached.get("format") != "hyprover-export-file-inventory-v1"
                or detached.get("source_project") != Path(source).name
                or export.get("project") != Path(source).name
                or not isinstance(inventory, list)
                or len(inventory) != export.get("counts", {}).get("included_original_files")):
            raise ArticleError("invalid_archive_inventory")
        if original_root is None:
            original_root = str(Path(source) / "verify-runs")
    if (not isinstance(original_root, str) or not Path(original_root).is_absolute()
            or result.get("completion_report") != str(Path(original_root) / run_id / "completion-report.json")
            or root.name != run_id or report_path.name != "completion-report.json"):
        raise ArticleError("archive_completion_path_mismatch")
    indexed = _verified_inventory(project, inventory)
    for name in ("KNOWLEDGE.md", "PROBLEM.md", "project.json", result_path, completion["report"],
                 str((root / "completion-manifest.json").relative_to(project)),
                 str((root / "completion-progress.json").relative_to(project))):
        if name not in indexed:
            raise ArticleError("archive_required_evidence_not_in_inventory")
    manifest = ce.load_manifest(root)
    if manifest["project_dir"] != source or run_id != "completion-" + manifest["digest"]:
        raise ArticleError("archive_completion_identity_mismatch")
    current, _ = ce.capture(project, {"target_fact_ids": result["target_fact_ids"], "summary": result["summary"]}, manifest["config"])
    current["project_dir"] = source
    current["digest"] = ce.digest({key: value for key, value in current.items() if key != "digest"})
    if current != manifest:
        raise ArticleError("relocated_archive_evidence_changed")
    report = ce.read_json(root / "completion-report.json")
    progress = ce.load_progress(root, manifest)
    if (report.get("format") != ce.FORMAT or report.get("evidence_digest") != manifest["digest"]
            or report.get("outcome") != "passed" or report.get("accepted") is not True
            or set(progress["receipts"]) != {job["id"] for job in ce.jobs(manifest)}
            or progress.get("pending_failure")):
        raise ArticleError("archive_completion_not_fully_accepted")
    bridge = progress["receipts"]["bridge"]["run_id"]
    if (report.get("coverage_verification_run") != bridge or result.get("coverage_verification_run") != bridge
            or result.get("completion_evidence_digest") != manifest["digest"]
            or sorted(result.get("target_fact_ids", [])) != manifest["targets"] or result.get("summary") != manifest["summary"]
            or sorted(completion.get("target_fact_ids", [])) != manifest["targets"]):
        raise ArticleError("archive_result_receipt_mismatch")
    binding = {"export_digest": digest(export), "source_directory": source, "inventory_digest": digest(indexed),
               "relocation": "project_directory_only", "historical_receipts_modified": False}
    return root, manifest, binding
