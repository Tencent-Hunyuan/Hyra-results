"""Adapter to the independent local-policy semantic oracle (not a cert checker)."""

import json
from pathlib import Path
import subprocess

from .algebra import ValidationError, integer, pauli, pauli_string, weight
from .input import Instance
from .engines import engine_path


DEFAULT_ENGINE = engine_path("hyqec-check-reference")


def roots(instance: Instance, domain: str) -> list[str]:
    if domain not in ("P", "X", "Z"):
        raise ValidationError("domain must be P, X or Z")
    # Verify before using any orbit partition, including programmatic callers.
    instance.code.validate_orbit_order(instance.permutations, instance.orbits, instance.representatives)
    excluded = set()
    result = []
    for orbit, rep in zip(instance.orbits, instance.representatives):
        for label in ("XYZ" if domain == "P" else domain):
            state = ["*" if domain == "P" else domain.lower()] * instance.code.n
            for q in excluded:
                state[q] = "I"
            state[rep] = label
            result.append("".join(state))
        excluded.update(orbit)
    return result


def search(instance: Instance, cap: int, domain: str = "P", threads: int = 1,
           engine: str | Path = DEFAULT_ENGINE) -> dict:
    integer(cap, "cap")
    integer(threads, "threads", 1)
    code = instance.code
    if cap > code.n or threads > 192:
        raise ValidationError("require cap <= n and threads <= 192")
    root_states = roots(instance, domain)
    lines = [f"{code.n} {len(code.generators)} {len(code.logicals)} {cap} {threads} {len(root_states)}"]
    lines += [pauli_string(v, code.n) for v in code.generators + code.logicals]
    lines += root_states
    run = subprocess.run([str(engine)], input="\n".join(lines) + "\n", text=True, capture_output=True)
    if run.returncode:
        raise ValidationError(f"native reference failed: {run.stderr.strip()}")
    result = json.loads(run.stdout)
    witnesses = [r["witness"] for r in result["roots"] if r["witness"]]
    for witness in witnesses:
        value = pauli(witness, code.n)
        if not code.is_logical(value) or weight(value, code.n) > cap:
            raise ValidationError("reference search emitted an invalid logical witness")
        if domain != "P" and any(label not in ("I", domain) for label in witness):
            raise ValidationError("reference search witness is outside requested domain")
    result["minimum_found"] = min((weight(pauli(w, code.n), code.n) for w in witnesses), default=None)
    result["nodes"] = sum(r["nodes"] for r in result["roots"])
    result["instance_file_sha256"] = instance.file_sha256
    result["cap"] = cap
    result["domain"] = domain
    result["n"] = code.n
    result["k"] = code.k
    result["statement"] = (
        "Independent reference search executed under local policy; no certificate, frontier, "
        "task coverage, expected count or expected trace hash was verified."
    )
    return result
