"""UTC timestamps and atomic shared-document publication."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fsync_directory(path: Path) -> None:
    """Persist a directory rename; failure must remain visible to callers."""
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def sync_existing_file(path: Path) -> None:
    """Finish durability work when retrying a previously visible write."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_directory(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    """Flush a same-directory temporary file and atomically publish it."""
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
