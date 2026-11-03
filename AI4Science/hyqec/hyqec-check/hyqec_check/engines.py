"""Locate wheel-bundled or checkout-built native executables."""

import os
from pathlib import Path


def environment(suffix, default=None):
    """Read renamed test/tool settings; retain the old alias for release 1.3."""
    return os.environ.get(f"HYQEC_CHECK_{suffix}", os.environ.get(f"BNBCHECK_{suffix}", default))


def engine_path(name):
    package = Path(__file__).resolve().parent
    bundled = package / "_native" / name
    return bundled if bundled.is_file() else package.parent / "build" / name
