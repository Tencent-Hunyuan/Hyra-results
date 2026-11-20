"""Standard 3D scalp electrode coordinate lookup (10-20, 10-10, 10-05, and bipolar midpoints)."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import torch

_PREFIX_RE = re.compile(r"^(?:EEG[\s_-]+)", flags=re.IGNORECASE)
_SUFFIX_RE = re.compile(r"(?:[\s_-]+(?:REF|LE|AVG|AR|CAR|A1|A2|M1|M2))$", flags=re.IGNORECASE)
_LEGACY_ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


@lru_cache(maxsize=1)
def _load_position_bank() -> tuple[dict[str, tuple[float, float, float]], dict[str, str]]:
    asset = Path(__file__).resolve().parent / "assets" / "positions.json"
    raw = json.loads(asset.read_text(encoding="utf-8"))
    coords: dict[str, tuple[float, float, float]] = {}
    upper_to_key: dict[str, str] = {}
    for name, xyz in raw.items():
        coords[name] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        upper = name.upper()
        if upper not in upper_to_key:
            upper_to_key[upper] = name
    return coords, upper_to_key


def list_standard_channels() -> list[str]:
    """Return all canonical electrode names in the built-in 3D scalp position bank."""
    coords, _ = _load_position_bank()
    return sorted(coords.keys())


def _lookup_single(name: str) -> tuple[float, float, float] | None:
    coords, upper_to_key = _load_position_bank()
    cleaned = name.strip()
    if cleaned in coords:
        return coords[cleaned]
    upper = cleaned.upper()
    if upper in upper_to_key:
        return coords[upper_to_key[upper]]

    stripped = _PREFIX_RE.sub("", cleaned).strip()
    stripped_no_ref = _SUFFIX_RE.sub("", stripped).strip()
    for candidate in (stripped, stripped_no_ref):
        u = candidate.upper()
        if u in upper_to_key:
            return coords[upper_to_key[u]]
        if u in _LEGACY_ALIASES and _LEGACY_ALIASES[u] in upper_to_key:
            return coords[upper_to_key[_LEGACY_ALIASES[u]]]
    return None


def resolve_positions(channels: Sequence[str]) -> torch.Tensor:
    """Resolve channel names (e.g. ``['Fp1', 'C3', 'Cz', 'C4', 'O2']``) to ``(C, 3)`` float32 meters.

    Supports case-insensitive 10-20 / 10-10 / 10-05 labels, optional ``EEG `` prefixes or
    ``-REF`` suffixes, and bipolar pairs such as ``'Fp1-F7'`` (mapped to their 3D midpoint).
    """
    if not channels:
        raise ValueError("channels must be a non-empty sequence of electrode names")

    out = np.zeros((len(channels), 3), dtype=np.float32)
    missing: list[str] = []
    for i, ch in enumerate(channels):
        pos = _lookup_single(ch)
        if pos is None and "-" in ch:
            parts = [p.strip() for p in ch.split("-") if p.strip()]
            if len(parts) == 2:
                p0, p1 = _lookup_single(parts[0]), _lookup_single(parts[1])
                if p0 is not None and p1 is not None:
                    pos = (
                        0.5 * (p0[0] + p1[0]),
                        0.5 * (p0[1] + p1[1]),
                        0.5 * (p0[2] + p1[2]),
                    )
        if pos is None:
            missing.append(ch)
        else:
            out[i] = pos

    if missing:
        raise KeyError(
            f"Unknown EEG channel name(s): {missing}. "
            "Pass custom 3D scalp coordinates (in meters) via `positions=`."
        )
    return torch.from_numpy(out)
