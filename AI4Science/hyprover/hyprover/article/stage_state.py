"""Recoverable publication of the reference ledger and its execution cursor."""
from __future__ import annotations

from pathlib import Path

from hyprover.core._util import atomic_write_text, fsync_directory

from .models import ArticleError, digest, json_document, read_json, require_keys, safe_path, write_json

_INTENT = "reference-commit.json"


def _read_commit(root: Path, ledger: dict | None = None) -> tuple[Path, dict | None]:
    intent_path = safe_path(root, _INTENT)
    if not intent_path.exists():
        return intent_path, None
    intent = read_json(intent_path)
    require_keys(intent, {"before", "after", "checkpoint", "checkpoint_digest"})
    if digest(intent["checkpoint"]) != intent["checkpoint_digest"]:
        raise ArticleError("invalid_article_reference_commit")
    if ledger is None:
        path = safe_path(root, "references.json")
        ledger = read_json(path) if path.exists() else None
    actual = digest(ledger) if ledger is not None else None
    if actual == intent["after"]:
        return intent_path, intent["checkpoint"]
    if actual != intent["before"]:
        raise ArticleError("unrecognized_article_reference_commit")
    return intent_path, None


def read_checkpoint(root: Path, ledger: dict | None = None) -> dict:
    _, committed = _read_commit(root, ledger)
    return committed if committed is not None else read_json(safe_path(root, "checkpoint.json"))


def recover(root: Path, fingerprint: str) -> None:
    """Resolve an interrupted write while holding the article job lock."""
    path, checkpoint = _read_commit(root)
    if checkpoint is not None:
        if checkpoint.get("fingerprint") != fingerprint:
            raise ArticleError("stale_article_reference_commit")
        write_json(root / "checkpoint.json", checkpoint)
    if path.exists():
        path.unlink()
        fsync_directory(root)


def commit(root: Path, before: str | None, ledger: dict, checkpoint: dict) -> str:
    """Make the next cursor recoverable before atomically replacing its ledger."""
    ledger_document = json_document(ledger)
    checkpoint_document = json_document(checkpoint)
    after = digest(ledger)
    intent = safe_path(root, _INTENT)
    if intent.exists():
        raise ArticleError("article_reference_commit_requires_recovery")
    write_json(intent, {"before": before, "after": after, "checkpoint": checkpoint,
                        "checkpoint_digest": digest(checkpoint)})
    atomic_write_text(safe_path(root, "references.json"), ledger_document)
    atomic_write_text(safe_path(root, "checkpoint.json"), checkpoint_document)
    intent.unlink()
    fsync_directory(root)
    return after
