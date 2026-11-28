"""hyprover.verify — the cold-start proof-verifier HTTP service (the sole write-gate).

``service.app`` is the FastAPI app (POST /verify, GET /health). Run as
``python -m hyprover.verify`` (or ``uvicorn hyprover.verify.service:app``).
"""

from __future__ import annotations

__all__ = ["app", "VerifyRequest", "run_codex_verification"]


def __getattr__(name: str):
    # Local completion-evidence consumers do not need the optional HTTP service.
    if name in ("app", "VerifyRequest"):
        from . import service

        return getattr(service, name)
    if name == "run_codex_verification":
        from .launcher import run_codex_verification

        return run_codex_verification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
