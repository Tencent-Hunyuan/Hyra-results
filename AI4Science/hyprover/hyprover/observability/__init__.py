"""hyprover.observability — the strictly read-only dashboard.

A FastAPI view of the project's authoritative KNOWLEDGE.md. It reuses core
snapshots, never writes project state, and loads proofs and notes only through
explicit detail requests.

    python -m hyprover.observability --project <dir> [--port 8099]

See ``app.py`` for the server + parsers.
"""

from __future__ import annotations

from .app import (
    CHANNELS,
    app,
    build_channel,
    build_channels,
    build_factgraph,
    build_overview,
    main,
)

__all__ = [
    "app",
    "main",
    "CHANNELS",
    "build_overview",
    "build_factgraph",
    "build_channels",
    "build_channel",
]
