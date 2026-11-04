"""Branch-and-bound lower bounds (driver for the native ``stabcert-bnb`` binaries).

The search itself (``native/bnb/core.h``) and its soundness proof are documented
in docs/SOUNDNESS.md. This module writes the instance file from the trusted core
(generators, logical basis, verified orbits), runs the binary
(INTERFACE.md section 2), and re-verifies anything it reports.

Outcomes of a run with cap C (min mode):
* nothing found: no nontrivial logical of weight <= C in the domain  (d >= C+1);
* found f <= C: f is the minimum weight in the domain (exact d = f), with a
  witness verified here.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core import CodeError, StabilizerCode
from ..symmetry import Orbits, trivial_orbits

__all__ = ["write_instance", "fiedler_order", "run_bnb", "BnbOutcome"]

INSTANCE_HEADER = "stabcert-bnb-instance 1"


def _labels(v: int, n: int):
    for q in range(n):
        lab = ((v >> q) & 1) | (((v >> (n + q)) & 1) << 1)
        if lab:
            yield q, lab


def instance_text(code: StabilizerCode, orbits: Orbits, domain: str = "P", legacy: bool = False) -> str:
    """Instance file contents (INTERFACE.md 2.1). legacy=True: the pre-release prototype's format (domain P only)."""
    n = code.n
    if domain not in ("P", "X", "Z"):
        raise CodeError(f"unknown domain {domain!r}")
    if legacy and domain != "P":
        raise CodeError("legacy instances support domain P only")
    gens = [g for g in code.generators if g]
    L = code.logicals
    lines = [] if legacy else [INSTANCE_HEADER]
    lines.append(f"{n} {len(gens)} {len(L)}")
    for v in gens + L:
        lab = list(_labels(v, n))
        lines.append(" ".join([str(len(lab))] + [f"{q} {l}" for q, l in lab]))
    if len(orbits.orbit_of) != n:
        raise CodeError("orbits do not match the code")
    lines.append(" ".join(map(str, orbits.orbit_of)))
    lines.append(" ".join(map(str, [orbits.count] + orbits.reps)))
    if not legacy:
        lines.append(f"domain {domain}")
    return "\n".join(lines) + "\n"


def write_instance(path, code, orbits, domain="P", legacy=False) -> str:
    text = instance_text(code, orbits, domain, legacy).encode()
    Path(path).write_bytes(text)
    return hashlib.sha256(text).hexdigest()


def fiedler_order(code: StabilizerCode) -> list[int]:
    """Qubits sorted by the Fiedler vector of the 'share a generator' graph.

    Only a root-order heuristic (any order is sound): consecutive roots then sit next
    to already excluded qubits (2x fewer nodes on [[485,97,25]] than file order)."""
    import numpy as np
    n = code.n
    if n <= 2:
        return list(range(n))
    A = np.zeros((n, n))
    for g in code.generators:
        sup = [q for q, _ in _labels(g, n)]
        for a in sup:
            A[a, sup] = 1
    np.fill_diagonal(A, 0)
    Lp = np.diag(A.sum(1)) - A
    vals, vecs = np.linalg.eigh(Lp)
    return [int(q) for q in np.lexsort((np.arange(n), np.round(vecs[:, 1], 12)))]


@dataclass
class BnbOutcome:
    status: str                    # "none" (lower bound cap+1), "found" (exact), "upper" (stopped on a
                                   # first find: witness only), "timeout", "error"
    cap: int
    found: int | None = None
    witness: int | None = None
    record: dict = field(default_factory=dict)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _limits_ok(code, cap, limits):
    n = code.n
    gens = [g for g in code.generators if g]
    W = max((sum(1 for _ in _labels(g, n)) for g in gens), default=0)
    col = [0] * n
    for g in gens:
        for q, _ in _labels(g, n):
            col[q] += 1
    problems = []
    if n > limits["NMAX"]:
        problems.append(f"n={n} > NMAX={limits['NMAX']}")
    if len(gens) > limits["MMAX"]:
        problems.append(f"m={len(gens)} > MMAX={limits['MMAX']}")
    if W > limits["WMAX"]:
        problems.append(f"generator weight {W} > WMAX={limits['WMAX']}")
    if max(col, default=0) > limits["CMAX"]:
        problems.append(f"column degree {max(col)} > CMAX={limits['CMAX']}")
    if cap > limits["DMAXW"] - 2:
        problems.append(f"cap {cap} > DMAXW-2={limits['DMAXW'] - 2}")
    return problems


_LOCATED: dict = {}


def generator_shape(code: StabilizerCode) -> tuple[int, int]:
    """(max generator weight, max column degree) over the nonzero supplied generators."""
    n = code.n
    gens = [g for g in code.generators if g]
    col = [0] * n
    W = 0
    for g in gens:
        sup = [q for q, _ in _labels(g, n)]
        W = max(W, len(sup))
        for q in sup:
            col[q] += 1
    return W, max(col, default=0)


def locate_binary(code: StabilizerCode, gpu: bool = False, rules: bool = False):
    """Native B&B binary + its --version (memoized per process: locating probes the compilers).
    rules=True selects the CPU rules build (variant suffix -rules) needed by COUNT_ONLY_POLICIES."""
    from .. import native
    base = 1024 if code.n <= 1024 and len(code.generators) <= 1024 else 4096
    W, C = generator_shape(code)
    if W <= 32 and C <= 64:
        variant = base
    elif gpu:
        raise CodeError(f"B&B GPU engine supports generator weight <= 32 and column degree <= 64 "
                        f"(this code: {W}, {C}); use the CPU engine")
    elif W <= 64 and C <= 128:
        variant = f"{base}-w64"
    elif W <= 128 and C <= 128:
        variant = f"{base}-w128"
    else:
        raise CodeError(f"B&B engine supports generator weight <= 128 and column degree <= 128 (this code: {W}, {C})")
    if rules:
        if gpu:
            raise CodeError("the opt-in S0/S1/S2 policies are CPU-only")
        variant = f"{variant}-rules"
    key = ("bnb-cuda" if gpu else "bnb-cpu", variant)
    if key not in _LOCATED:
        path = native.locate(key[0], variant=variant)
        _LOCATED[key] = (path, native.version(path))
        if rules and _LOCATED[key][1]["limits"].get("RULES") != "s0s1s2/1":
            del _LOCATED[key]
            raise CodeError(f"{path} is not a rules build (no opt-in S0/S1/S2 policies)")
    return _LOCATED[key]


# Opt-in pruning policies (S0, fixed S1, S2; docs/SOUNDNESS.md 5.1). Version-1 coverage certificates admit only
# the section-5 rules; these policies export only coverage certificate version 2 (docs/BNB_CERT_SPEC_v2.md), which
# verify counts only through a v2 companion checker (docs/SOUNDNESS.md 5.2); otherwise their runs stay count-only.
COUNT_ONLY_POLICIES = ("greedy+s0", "greedy+s0s1", "greedy+s0s1s2")


def run_bnb(code: StabilizerCode, cap: int, workdir, *, domain: str = "P", orbits: Orbits | None = None,
            threads: int | None = None, gpus: str | None = None, split: int = -200000,
            delta: int | None = None, budget: int | None = None, cpu_budget: int | None = None,
            chunk: int | None = None, ckpt: bool = False, timeout: float | None = None,
            stop_first: bool = False, binary=None, legacy: bool = False,
            emit_cert: bool = False, root_order: str = "fiedler", bound: str = "greedy",
            slots: int | None = None, cert_version: int | None = None) -> BnbOutcome:
    """Run the B&B search with cap ``cap``; returns a verified outcome.

    cert_version: coverage certificate container version; default 1 under greedy and 2 under the opt-in
    S0/S1/S2 policies, which version 1 cannot express."""
    code.require_logicals()
    if type(cap) is not int or cap < 0:
        raise CodeError("cap must be an int >= 0 (cap 0 proves only the trivial d >= 1)")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if orbits is None:
        # trivial group: any root order is sound; the spectral order only shrinks the trees
        if root_order not in ("fiedler", "file"):
            raise CodeError(f"unknown root order {root_order!r}")
        orbits = trivial_orbits(code.n, fiedler_order(code) if root_order == "fiedler" else None)
    if bound not in ("greedy", "none") + COUNT_ONLY_POLICIES:
        raise CodeError(f"unknown bound {bound!r}")
    if bound == "none" and (emit_cert or ckpt):
        raise CodeError("--bound none is an ablation switch; it cannot be combined with certificates or checkpoints")
    if cert_version is None:
        cert_version = 2 if bound in COUNT_ONLY_POLICIES else 1
    if cert_version not in (1, 2):
        raise CodeError(f"unknown coverage certificate version {cert_version!r}")
    if bound in COUNT_ONLY_POLICIES:
        # Opt-in S0/S1/S2 rules (docs/SOUNDNESS.md 5.1): CPU only, outside coverage certificate v1.
        if emit_cert and cert_version == 1:
            raise CodeError(f"--bound {bound}: coverage certificate v1 admits only the section-5 rules (use version 2)")
        if gpus:
            raise CodeError(f"--bound {bound} is CPU-only")
    if binary is None:
        binary, ver = locate_binary(code, gpu=bool(gpus), rules=bound in COUNT_ONLY_POLICIES)
        problems = _limits_ok(code, cap, ver["limits"])
        if problems:
            raise CodeError("B&B engine limits exceeded: " + "; ".join(problems))
    else:
        ver = None
    inst = workdir / "instance.txt"
    inst_sha = write_instance(inst, code, orbits, domain, legacy)
    bin_sha = _sha256(binary)
    cmd = [str(binary), str(inst), "--cap", str(cap), "--split", str(split)]
    if not legacy:
        # the caller hashes the instance it just wrote and the executable; the engine echoes the
        # instance hash in its done line and binds checkpoints to both (INTERFACE.md 2.5)
        cmd += ["--instance-sha256", inst_sha, "--binary-sha256", bin_sha]
    # never default to every core of a shared machine
    cmd += ["--threads", str(threads or min(16, os.cpu_count() or 1))]
    if gpus:
        cmd += ["--gpu", gpus]
        if slots is not None:
            # kernels in flight per GPU; each slot has a host waiter thread (~1 core while busy)
            cmd += ["--slots", str(slots)]
    for flag, val in (("--delta", delta), ("--budget", budget), ("--cpu-budget", cpu_budget), ("--chunk", chunk)):
        if val is not None:
            cmd += [flag, str(val)]
    if ckpt:
        cmd += ["--ckpt", str(workdir / "ckpt.txt")]
    if stop_first:
        cmd += ["--stop-first"]
    if bound != "greedy":
        cmd += ["--bound", bound]
    cov_raw = workdir / "coverage.sbcov"
    if emit_cert:
        # coverage certificate (docs/BNB_CERT_SPEC.md): the engine needs the group hash and the
        # verified permutations behind the orbits; the independent checker re-verifies both
        if ckpt or stop_first or legacy:
            raise CodeError("coverage export is incompatible with --ckpt, --stop-first and legacy engines")
        meta = workdir / "coverage-meta.json"
        meta.write_text(json.dumps({"group_sha256": code.group_hash(), "permutations": orbits.generators}))
        cov_raw.unlink(missing_ok=True)
        cmd += ["--emit-cert", str(cov_raw), "--cert-metadata", str(meta)]
        if cert_version != 1:
            cmd += ["--cert-version", str(cert_version)]
    rec = {"method": "bnb", "params": {"cap": cap, "domain": domain, "orbit_roots": orbits.to_json(),
                                       "split": split, "stop_first": stop_first, "bound": bound},
           "engine": {"binary": str(binary), "binary_sha256": bin_sha, "version": ver},
           "instance_sha256": inst_sha, "argv": cmd}
    t0 = time.monotonic()
    with open(workdir / "engine.jsonl", "ab") as out, open(workdir / "engine.err", "ab") as err:
        from .. import deadline
        rc, _ = deadline.run(cmd, timeout=timeout, stdout=out, stderr=err)
    rec["seconds"] = round(time.monotonic() - t0, 3)
    rec["returncode"] = rc
    if rc is None:
        return BnbOutcome("timeout", cap, record=rec)
    done = None
    for line in (workdir / "engine.jsonl").read_text().splitlines():
        if line.startswith("{"):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "done":
                done = ev
    if rc != 0 or done is None:
        rec["stderr"] = (workdir / "engine.err").read_text(errors="replace")[-2000:]
        return BnbOutcome("error", cap, record=rec)
    keep = ("cap0", "split", "delta", "rounds", "tasks_total", "aborted_total", "gen_nodes", "task_nodes",
            "nodes", "max_task_nodes", "replay_fail", "final_cap", "found", "executor", "resumed", "sec")
    rec["result"] = {k: done[k] for k in keep if k in done}
    if done.get("replay_fail", 0) != 0:
        rec["error"] = "replay failures"
        return BnbOutcome("error", cap, record=rec)
    if not legacy and done.get("instance_sha256") not in (None, inst_sha):
        rec["error"] = "engine read a different instance"
        return BnbOutcome("error", cap, record=rec)
    if not legacy and done.get("instance_sha256") is None:
        rec["error"] = "engine did not report the instance hash"
        return BnbOutcome("error", cap, record=rec)
    f = done.get("found", -1)
    if f is None or f < 0:
        rec["proves"] = _proves(cap, domain)
        if emit_cert:
            if not cov_raw.exists():
                rec["error"] = "engine finished without publishing the coverage certificate"
                return BnbOutcome("error", cap, record=rec)
            rec["coverage"] = _compress_coverage(cov_raw)
            rec["directory"] = str(workdir)
            rec["coverage"]["header"] = read_coverage_header(workdir / rec["coverage"]["file"])
        return BnbOutcome("none", cap, record=rec)
    n = code.n
    v = 0
    for q, lab in done["witness"]:
        if lab & 1:
            v |= 1 << q
        if lab & 2:
            v |= 1 << (n + q)
    w = code.check_witness(v)                 # raises if not a nontrivial logical
    if w != f or w > cap:
        raise AssertionError(f"B&B witness weight {w} != reported {f}")
    if (domain == "X" and v >> n) or (domain == "Z" and v & ((1 << n) - 1)):
        raise AssertionError("B&B witness outside its domain")
    # a run that stopped on its first find (--stop-first, or any checkpointed run) only gives an upper bound
    stopped = bool(done.get("stopped_on_find", stop_first or ckpt))
    rec["stopped_on_find"] = stopped
    if stopped:
        return BnbOutcome("upper", cap, f, v, rec)
    rec["proves"] = _proves(f - 1, domain)
    return BnbOutcome("found", cap, f, v, rec)


def _compress_coverage(raw: Path) -> dict:
    """gzip the raw SBCOV1/SBCOV2 stream deterministically (mtime 0, no file name), streaming in bounded
    chunks (artifacts reach ~1 GB); the .gz is published atomically. The result is exactly one gzip member with
    its CRC-32/ISIZE trailer and nothing after it (the version-2 transport, BNB_CERT_SPEC_v2.md section 14)."""
    import gzip
    out = raw.with_name(raw.name + ".gz")
    tmp = raw.with_name(raw.name + ".gz.tmp")
    h_raw, size = hashlib.sha256(), 0
    with open(raw, "rb") as src, open(tmp, "wb") as f, \
            gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=6) as g:
        for block in iter(lambda: src.read(1 << 22), b""):
            h_raw.update(block)
            size += len(block)
            g.write(block)
    tmp.replace(out)
    raw.unlink()
    return {"file": out.name, "sha256": h_raw.hexdigest(), "bytes": size,
            "gz_sha256": _sha256(out), "gz_bytes": out.stat().st_size}


def read_coverage_header(path) -> dict:
    """Header JSON of an SBCOV1 or SBCOV2 coverage stream (raw or gzip, chosen by magic)."""
    import gzip
    import struct
    with open(path, "rb") as f:
        magic = f.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as f:
        if f.read(8) not in (b"SBCOV1\r\n", b"SBCOV2\r\n"):
            raise CodeError("not an SBCOV1/SBCOV2 coverage stream")
        (hl,) = struct.unpack("<Q", f.read(8))
        if hl > 1 << 30:
            raise CodeError("implausible coverage header length")
        return json.loads(f.read(hl).decode("utf-8"))


def _proves(W, domain):
    kind = {"P": "nontrivial logical", "X": "X-type nontrivial logical", "Z": "Z-type nontrivial logical"}[domain]
    return f"no {kind} of Pauli weight <= {W}"
