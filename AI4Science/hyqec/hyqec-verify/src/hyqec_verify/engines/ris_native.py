"""Wrapper for the native RIS binaries (``stabcert-ris`` CPU, ``stabcert-ris-cuda`` GPU).

Writes the ``PAULIRS2`` input (INTERFACE.md section 3), runs the binary, and
re-verifies every reported operator with the core; ``rejected_by_core`` must be 0.
"""
from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from pathlib import Path

from ..core import CodeError, StabilizerCode, pauli_from_xz, swap_halves, xz_from_pauli
from .ris import _side_basis

__all__ = ["write_input", "run"]

_LOCATED: dict = {}


def write_input(path, code: StabilizerCode, rows: list[int]) -> None:
    n = code.n
    nq = (n + 63) // 64
    L = code.logicals
    lsw = [swap_halves(l, n) for l in L]
    nsw = (len(L) + 63) // 64
    colnz = 0
    for v in rows:
        colnz |= v
    m64 = (1 << 64) - 1
    with open(path, "wb") as f:
        f.write(b"PAULIRS2")
        f.write(struct.pack("<4i", n, nq, len(rows), len(L)))
        f.write(bytes(((colnz >> j) & 1) for j in range(2 * n)))
        for v in rows:
            syn = 0
            for j, l in enumerate(lsw):
                if (v & l).bit_count() & 1:
                    syn |= 1 << j
            x, z = v & ((1 << n) - 1), v >> n
            words = [(x >> (64 * i)) & m64 for i in range(nq)] + [(z >> (64 * i)) & m64 for i in range(nq)]
            words += [(syn >> (64 * i)) & m64 for i in range(nsw)]
            f.write(struct.pack(f"<{len(words)}Q", *words))


def run(code: StabilizerCode, *, seconds: float = 30.0, seed: int = 0, domain: str = "P", target: int | None = None,
        gpu: bool = False, threads: int | None = None, device: int = 0, no_chain: bool = False,
        binary=None) -> dict:
    from .. import native
    code.require_logicals()
    n = code.n
    rows = code.normalizer_basis() if domain == "P" else _side_basis(code, domain)
    if binary is None:
        tgt = "ris-cuda" if gpu else "ris-cpu"
        if tgt not in _LOCATED:          # locating probes the compilers: once per process
            _LOCATED[tgt] = native.locate(tgt)
        binary = _LOCATED[tgt]
    with tempfile.TemporaryDirectory(prefix="hyqec-ris-") as t:
        inp = Path(t) / "input.bin"
        write_input(inp, code, rows)
        cmd = [str(binary), str(inp), "--seconds", str(seconds), "--seed", str(seed)]
        if target is not None:
            cmd += ["--target", str(target)]
        if threads and not gpu:
            cmd += ["--threads", str(threads)]
        if gpu:
            cmd += ["--device", str(device)]
        if no_chain:
            cmd += ["--no-chain"]
        from .. import deadline
        with open(Path(t) / "out.jsonl", "w") as so, open(Path(t) / "err.txt", "w") as se:
            rc, _ = deadline.run(cmd, timeout=seconds + 120, stdout=so, stderr=se)
        stdout = (Path(t) / "out.jsonl").read_text()
        stderr = (Path(t) / "err.txt").read_text()
    if rc is not None and rc != 0:
        raise CodeError(f"native RIS failed (exit {rc}): {stderr.strip()[-500:]}")
    best, best_v, rejected, done = None, None, 0, None
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        ev = json.loads(line)
        if ev.get("event") in ("witness", "ring"):
            v = pauli_from_xz(ev["x"], ev["z"], n, "RIS witness")
            if code.witness_problem(v) is not None or code.weight(v) != ev["w"]:
                rejected += 1
                continue
            if best is None or ev["w"] < best:
                best, best_v = ev["w"], v
        elif ev.get("event") == "done":
            done = ev
    if rejected:
        raise CodeError(f"native RIS reported {rejected} operator(s) that fail the core check")
    return {"upper_bound": best, "witness": xz_from_pauli(best_v, n) if best_v is not None else None,
            "witness_int": best_v, "engine": "ris-cuda" if gpu else "ris-cpu", "rejected_by_core": rejected,
            "search": done}
