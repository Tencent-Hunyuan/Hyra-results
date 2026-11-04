"""Strict JSON reading and a provisional, checker-owned algebra input format.

This module reads strict JSON and the historical checker-owned reference
fixture format. Coverage certificates use the separate certificate adapter.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .algebra import Code, ValidationError, integer


def strict_json(raw: bytes) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value):
        raise ValidationError(f"nonfinite JSON number {value}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValidationError(f"invalid UTF-8 JSON: {error}") from error


@dataclass
class Instance:
    code: Code
    permutations: list[list[int]]
    orbits: list[list[int]]
    representatives: list[int]
    file_sha256: str


def read_reference_instance(path: str | Path) -> Instance:
    raw = Path(path).read_bytes()
    data = strict_json(raw)
    if not isinstance(data, dict):
        raise ValidationError("input must be an object")
    allowed = {"format", "n", "k", "generators", "permutations", "orbits", "representatives"}
    if set(data) - allowed:
        raise ValidationError(f"unknown input fields: {sorted(set(data) - allowed)}")
    if data.get("format") != "bnbcheck-reference-code/0":
        raise ValidationError("expected checker-owned format bnbcheck-reference-code/0")
    if "n" not in data or "generators" not in data:
        raise ValidationError("missing n or generators")
    code = Code.from_strings(data["n"], data["generators"])
    if "k" in data and integer(data["k"], "k") != code.k:
        raise ValidationError("claimed k differs from independently computed k")
    permutations = data.get("permutations", [])
    actual_orbits = code.orbits(permutations)
    orbits = data.get("orbits", [list(orbit) for orbit in actual_orbits])
    if "representatives" in data:
        representatives = data["representatives"]
    else:
        if (not isinstance(orbits, list)
                or any(not isinstance(orbit, list) or not orbit for orbit in orbits)):
            raise ValidationError("orbits must be nonempty lists")
        representatives = [orbit[0] for orbit in orbits]
    code.validate_orbit_order(permutations, orbits, representatives)
    return Instance(code, permutations, orbits, representatives, hashlib.sha256(raw).hexdigest())
