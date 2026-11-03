"""Environment variables: ``HYQEC_<NAME>``, with the pre-rename ``STABCERT_<NAME>`` accepted as a
deprecated alias for one release (0.1.x)."""
from __future__ import annotations

import os
import warnings

_warned: set[str] = set()


def getenv(name: str, default: str | None = None) -> str | None:
    """Value of HYQEC_<name>; else of the deprecated STABCERT_<name> (with a one-time warning)."""
    v = os.environ.get("HYQEC_" + name)
    if v is not None:
        return v
    v = os.environ.get("STABCERT_" + name)
    if v is not None:
        if name not in _warned:
            _warned.add(name)
            warnings.warn(f"STABCERT_{name} is deprecated; use HYQEC_{name}", DeprecationWarning, stacklevel=2)
        return v
    return default
