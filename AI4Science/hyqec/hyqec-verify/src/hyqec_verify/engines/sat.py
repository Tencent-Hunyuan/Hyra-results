"""SAT/DRAT lower bounds: "no nontrivial logical of weight <= W" as an UNSAT CNF.

TRUSTED part: ``encode`` (the reduction) — see docs/SOUNDNESS.md. Encoding
``stabcert-pauli-cnf/1`` for a code with n qubits:

* variables ``x_q = q+1``, ``z_q = n+q+1``, ``u_q = 2n+q+1``; clauses
  ``u_q <-> (x_q or z_q)`` (both directions), so sum u = Pauli weight;
* domain X / Z (CSS per-side questions): units ``-z_q`` / ``-x_q``;
* for every supplied generator g: XOR over ``x_q (q in supp_z g)`` and
  ``z_q (q in supp_x g)`` equals 0 (commutes with g);
* for every element l of the core's logical basis: a fresh ``y_l`` with
  ``y_l = <v, l>``, and the clause ``OR_l y_l`` (v is not in S, given v in N(S));
* ``sum_q u_q <= W`` by a sequential counter (Sinz 2005);
* optional orbit roots (verified automorphisms, symmetry.Orbits): fresh ``c_i`` with
  ``c_i -> u_{rep_i}``, ``c_i -> -u_q`` for q in earlier orbits, and ``OR_i c_i``.

Satisfiable (without roots) iff a nontrivial logical of weight <= W exists in the
domain; with roots, satisfiability is unchanged (a minimum-weight logical can be
moved to satisfy them). XORs are cut into chains of ``chunk`` literals with
Tseitin variables, each piece expanded into its exact truth table.

Everything else (running solvers, checking proofs, cube bundles) is plumbing:
an UNSAT answer counts only with a proof that the (vendored) DRAT checker
accepts for the exact CNF bytes, regenerated from the code at verification time.
"""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..core import CodeError, StabilizerCode
from ..symmetry import Orbits
from .. import tools

ENCODING = "stabcert-pauli-cnf/1"

__all__ = ["ENCODING", "Formula", "encode", "prove", "prove_cubes", "check_record", "SatOutcome"]


class Formula:
    def __init__(self, nv: int = 0):
        self.nv = nv
        self.clauses: list[list[int]] = []

    def fresh(self) -> int:
        self.nv += 1
        return self.nv

    def add(self, clause: Sequence[int]) -> None:
        self.clauses.append(list(clause))

    def xor(self, lits: Sequence[int], parity: int = 0, chunk: int = 4) -> None:
        """Clauses for lits[0] ^ lits[1] ^ ... == parity (chained, exact)."""
        lits = list(lits)

        def expand(xs, p):
            for vals in itertools.product((0, 1), repeat=len(xs)):
                if sum(vals) % 2 != p:
                    # forbid this assignment
                    self.clauses.append([-x if v else x for x, v in zip(xs, vals)])

        while len(lits) > chunk:
            y = self.fresh()
            expand(lits[:chunk - 1] + [y], 0)       # y == xor(lits[:chunk-1])
            lits = [y] + lits[chunk - 1:]
        expand(lits, parity)

    def at_most(self, xs: Sequence[int], k: int) -> None:
        """Sequential counter: sum(xs) <= k (Sinz 2005, LT_SEQ)."""
        xs = list(xs)
        n = len(xs)
        if k >= n:
            return
        if k == 0:
            for x in xs:
                self.add([-x])
            return
        s = [[self.fresh() for _ in range(k)] for _ in range(n - 1)]   # s[i][j]: sum(x_0..x_i) >= j+1
        self.add([-xs[0], s[0][0]])
        for j in range(1, k):
            self.add([-s[0][j]])
        for i in range(1, n - 1):
            self.add([-xs[i], s[i][0]])
            self.add([-s[i - 1][0], s[i][0]])
            for j in range(1, k):
                self.add([-xs[i], -s[i - 1][j - 1], s[i][j]])
                self.add([-s[i - 1][j], s[i][j]])
            self.add([-xs[i], -s[i - 1][k - 1]])
        self.add([-xs[n - 1], -s[n - 2][k - 1]])

    def dimacs(self, extra: Sequence[Sequence[int]] = ()) -> bytes:
        extra = [list(c) for c in extra]
        for c in extra:
            # a 0 would terminate the clause early in DIMACS; out-of-range literals would add variables
            if not c or any(type(l) is not int or l == 0 or abs(l) > self.nv for l in c):
                raise CodeError(f"invalid extra clause {c!r} (literals must be nonzero ints within 1..{self.nv})")
        lines = [f"p cnf {self.nv} {len(self.clauses) + len(extra)}"]
        lines += [" ".join(map(str, c)) + " 0" for c in itertools.chain(self.clauses, extra)]
        return ("\n".join(lines) + "\n").encode()


@dataclass
class Encoded:
    formula: Formula
    params: dict
    selectors: list[int]            # y_l
    roots: list[int]                # c_i (empty without orbit roots)


def encode(code: StabilizerCode, W: int, *, domain: str = "P", chunk: int = 4,
           orbits: Orbits | None = None) -> Encoded:
    """CNF satisfiable iff a nontrivial logical of weight <= W exists in the domain."""
    code.require_logicals()
    n = code.n
    if type(W) is not int or not 0 <= W < n:
        raise CodeError(f"bound W must be an int in [0, n), got {W!r}")
    if not 3 <= chunk <= 8:
        raise CodeError("xor chunk must be in [3, 8]")
    if domain not in ("P", "X", "Z"):
        raise CodeError(f"unknown domain {domain!r}")
    X = lambda q: q + 1
    Z = lambda q: n + q + 1
    U = lambda q: 2 * n + q + 1
    f = Formula(3 * n)
    for q in range(n):
        f.add([-X(q), U(q)])
        f.add([-Z(q), U(q)])
        f.add([X(q), Z(q), -U(q)])
    if domain == "X":
        for q in range(n):
            f.add([-Z(q)])
    elif domain == "Z":
        for q in range(n):
            f.add([-X(q)])

    def pairing_lits(p: int) -> list[int]:
        # <v, p> = sum_q x_q p.z_q + z_q p.x_q
        lits = [X(q) for q in range(n) if (p >> (n + q)) & 1]
        lits += [Z(q) for q in range(n) if (p >> q) & 1]
        return sorted(lits)

    for g in code.generators:
        lits = pairing_lits(g)
        if lits:
            f.xor(lits, 0, chunk)
    selectors = []
    for l in code.logicals:
        y = f.fresh()
        selectors.append(y)
        f.xor(pairing_lits(l) + [y], 0, chunk)      # y == <v, l>
    f.add(selectors)
    f.at_most([U(q) for q in range(n)], W)
    roots = []
    if orbits is not None:
        if len(orbits.orbit_of) != n:
            raise CodeError("orbits do not match the code")
        no = len(orbits.reps)
        if no == 0 or sorted(set(orbits.orbit_of)) != list(range(no)) or \
                any(type(r) is not int or not 0 <= r < n for r in orbits.reps):
            raise CodeError("orbit roots: every orbit 0..no-1 needs exactly one representative")
        members = orbits.members()
        for i, rep in enumerate(orbits.reps):
            if orbits.orbit_of[rep] != i:
                raise CodeError("orbit representative not in its orbit")
            c = f.fresh()
            roots.append(c)
            f.add([-c, U(rep)])
            for j in range(i):
                for q in members[j]:
                    f.add([-c, -U(q)])
        f.add(roots)
    params = {"encoding": ENCODING, "bound": W, "domain": domain, "xor_chunk": chunk,
              "cardinality": "seqcounter", "orbit_roots": orbits.to_json() if orbits is not None else None}
    return Encoded(f, params, selectors, roots)


def encode_from_params(code: StabilizerCode, params: dict) -> Encoded:
    """Re-encode from a recorded parameter dict (orbits are re-verified)."""
    from ..symmetry import orbits_from
    if params.get("encoding") != ENCODING or params.get("cardinality") != "seqcounter":
        raise CodeError(f"unsupported encoding {params.get('encoding')!r}")
    orb = None
    o = params.get("orbit_roots")
    if o is not None:
        orb = orbits_from(code, o["generators"])
        # recorded numbering must be an orbit numbering of the verified group
        if _orbit_partition(orb.orbit_of) != _orbit_partition(o["orbit_of"]):
            raise CodeError("recorded orbits differ from the verified group's orbits")
        orb = Orbits(list(o["orbit_of"]), list(o["reps"]), orb.generators)
    return encode(code, params["bound"], domain=params["domain"], chunk=params["xor_chunk"], orbits=orb)


def _orbit_partition(orbit_of):
    groups: dict[int, list[int]] = {}
    for q, o in enumerate(orbit_of):
        groups.setdefault(o, []).append(q)
    return sorted(groups.values())


# ---------------------------------------------------------------------------
# running solvers and the checker
# ---------------------------------------------------------------------------

_LIMIT_WRAPPER = ("import os,resource,sys;"
                  "m=int(sys.argv[1]);"
                  "m>0 and resource.setrlimit(resource.RLIMIT_FSIZE,(m,m));"
                  "os.execv(sys.argv[2],sys.argv[2:])")


def _run(cmd, log_path, timeout, file_limit_bytes=0):
    from .. import deadline
    with open(log_path, "wb") as out:
        rc, sec = deadline.run([sys.executable, "-c", _LIMIT_WRAPPER, str(file_limit_bytes), *map(str, cmd)],
                               timeout=timeout, stdout=out, stderr=subprocess.STDOUT)
    return rc, sec


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_proof(cnf_path, proof_path, workdir, timeout=None, lrat=False) -> dict:
    """Run drat-trim on (cnf, proof); optionally also produce and check LRAT."""
    chk = tools.checker()
    workdir = Path(workdir)
    cmd = [chk, cnf_path, proof_path, "-t", str(int(timeout or 10 ** 7))]
    lrat_path = workdir / "proof.lrat"
    if lrat:
        cmd += ["-L", lrat_path]
    rc, sec = _run(cmd, workdir / "checker.log", timeout)
    text = (workdir / "checker.log").read_text(errors="replace")
    ok = rc == 0 and "s VERIFIED" in text
    res = {"verified": ok, "checker": "drat-trim", "checker_sha256": tools.sha256_file(chk),
           "returncode": rc, "seconds": round(sec, 3)}
    if lrat and ok:
        lc = tools.lrat_checker()
        rc2, sec2 = _run([lc, cnf_path, lrat_path], workdir / "lrat.log", timeout)
        t2 = (workdir / "lrat.log").read_text(errors="replace")
        lok = rc2 == 0 and "c VERIFIED" in t2 and "FAILED" not in t2
        res["lrat"] = {"verified": lok, "checker": "lrat-check", "checker_sha256": tools.sha256_file(lc),
                       "seconds": round(sec2, 3)}
        res["verified"] = ok and lok
        lrat_path.unlink(missing_ok=True)
    return res


@dataclass
class SatOutcome:
    status: str                     # "UNSAT" (checked), "SAT", "TIMEOUT", "UNCHECKED", "ERROR"
    record: dict = field(default_factory=dict)
    model_pauli: int | None = None


def _solve_one(cnf: bytes, outdir: Path, *, solver: str, seed: int, timeout: float | None,
               lrat: bool, file_limit_mb: int, n: int, keep_proof: bool = True,
               check_timeout: float | None = None) -> SatOutcome:
    outdir.mkdir(parents=True, exist_ok=True)
    cnf_path = outdir / "problem.cnf"
    proof_path = outdir / "proof.drat"
    cnf_path.write_bytes(cnf)
    exe = tools.find_solver(solver)
    cmd = [exe, "-q", f"--seed={seed}", cnf_path, proof_path]
    rc, sec = _run(cmd, outdir / "solver.log", timeout, file_limit_mb << 20)
    rec = {"cnf_sha256": sha256_bytes(cnf), "solver": {"name": solver, **tools.tool_info(exe)},
           "seed": seed, "solver_returncode": rc, "solve_seconds": round(sec, 3)}
    try:
        if rc == 20:
            chk = check_proof(cnf_path, proof_path, outdir, timeout=check_timeout, lrat=lrat)
            rec["check"] = chk
            if not chk["verified"]:
                return SatOutcome("UNCHECKED", rec)
            # stream: proofs can be gigabytes
            h, size = hashlib.sha256(), 0
            g = gzip.open(outdir / "proof.drat.gz", "wb", compresslevel=3) if keep_proof else None
            with open(proof_path, "rb") as src:
                for block in iter(lambda: src.read(1 << 22), b""):
                    h.update(block)
                    size += len(block)
                    if g:
                        g.write(block)
            if g:
                g.close()
            rec["proof"] = {"file": "proof.drat.gz", "sha256": h.hexdigest(), "bytes": size}
            return SatOutcome("UNSAT", rec)
        if rc == 10:
            model = set()
            for line in (outdir / "solver.log").read_text().splitlines():
                if line.startswith("v "):
                    model.update(int(t) for t in line[2:].split() if t != "0")
            v = 0
            for q in range(n):
                if (q + 1) in model:
                    v |= 1 << q
                if (n + q + 1) in model:
                    v |= 1 << (n + q)
            return SatOutcome("SAT", rec, v)
        return SatOutcome("TIMEOUT" if rc is None else "ERROR", rec)
    finally:
        proof_path.unlink(missing_ok=True)
        cnf_path.unlink(missing_ok=True)


def prove(code: StabilizerCode, W: int, outdir, *, domain="P", orbits: Orbits | None = None,
          solver="kissat", seed=0, timeout=None, chunk=4, lrat=False, file_limit_mb=4096,
          check_timeout=None) -> SatOutcome:
    """Try to prove d_domain >= W+1 with one solver call. SAT gives a verified witness."""
    enc = encode(code, W, domain=domain, chunk=chunk, orbits=orbits)
    out = _solve_one(enc.formula.dimacs(), Path(outdir), solver=solver, seed=seed, timeout=timeout,
                     lrat=lrat, file_limit_mb=file_limit_mb, n=code.n, check_timeout=check_timeout)
    out.record = {"method": "sat-drat", "params": enc.params, **out.record}
    if out.status == "UNSAT":
        out.record["proves"] = _proves_text(W, domain)
        out.record["directory"] = str(outdir)
    if out.status == "SAT":
        _check_model(code, out.model_pauli, W, domain)
    return out


def _proves_text(W, domain):
    kind = {"P": "nontrivial logical", "X": "X-type nontrivial logical", "Z": "Z-type nontrivial logical"}[domain]
    return f"no {kind} of Pauli weight <= {W}"


def _check_model(code, v, W, domain):
    w = code.check_witness(v)        # raises if the solver model is not a nontrivial logical
    if w > W:
        raise AssertionError("SAT model heavier than the bound")
    n = code.n
    if domain == "X" and v >> n:
        raise AssertionError("SAT model outside the X domain")
    if domain == "Z" and v & ((1 << n) - 1):
        raise AssertionError("SAT model outside the Z domain")


# ---------------------------------------------------------------------------
# cube-and-conquer bundle
# ---------------------------------------------------------------------------

def make_cubes(enc: Encoded, n_first: int, split_labels: bool, n: int, orbits: Orbits) -> list[list[int]]:
    """Cubes on the orbit-root selectors: [c_i] for i < n_first (optionally x 3 labels
    of rep_i), then one tail cube [-c_0, ..., -c_{n_first-1}]."""
    roots = enc.roots
    n_first = min(n_first, len(roots))
    cubes = []
    for i in range(n_first):
        rep = orbits.reps[i]
        x, z = rep + 1, n + rep + 1
        if split_labels and enc.params["domain"] == "P":
            cubes += [[roots[i], x, -z], [roots[i], -x, z], [roots[i], x, z]]
        else:
            cubes.append([roots[i]])
    if n_first < len(roots):
        cubes.append([-c for c in roots[:n_first]])
    return cubes


def prove_cubes(code: StabilizerCode, W: int, outdir, *, orbits: Orbits, domain="P", n_first=None,
                split_labels=False, solver="kissat", workers=8, timeout=None, chunk=4, lrat=False,
                file_limit_mb=4096, seed=0) -> SatOutcome:
    """Cube-and-conquer: every leaf F & cube refuted with a checked proof, plus a checked
    coverage proof of F & (not cube_1) & ... (so the cubes cover every model of F)."""
    outdir = Path(outdir)
    enc = encode(code, W, domain=domain, chunk=chunk, orbits=orbits)
    f = enc.formula
    cubes = make_cubes(enc, n_first if n_first is not None else len(enc.roots), split_labels, code.n, orbits)
    jobs = [("leaf", i, [[l] for l in cube]) for i, cube in enumerate(cubes)]
    jobs.append(("coverage", len(cubes), [[-l for l in cube] for cube in cubes]))
    results: dict[int, SatOutcome] = {}

    def run(job):
        kind, i, extra = job
        return i, _solve_one(f.dimacs(extra), outdir / f"{kind}-{i:05d}", solver=solver, seed=seed,
                             timeout=timeout, lrat=lrat, file_limit_mb=file_limit_mb, n=code.n)

    witness = None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, out in ex.map(run, jobs):
            results[i] = out
            if out.status == "SAT" and i < len(cubes):
                _check_model(code, out.model_pauli, W, domain)
                witness = out.model_pauli
    leaves = []
    for kind, i, _ in jobs:
        r = dict(results[i].record)
        if "proof" in r:
            r["proof"] = {**r["proof"], "file": f"{kind}-{i:05d}/proof.drat.gz"}
        leaves.append({"kind": kind, "index": i, "cube": cubes[i] if kind == "leaf" else None,
                       "status": results[i].status, **r})
    statuses = [results[i].status for i in range(len(jobs))]
    rec = {"method": "sat-drat-cubes", "params": enc.params, "cubes": cubes, "leaves": leaves,
           "directory": str(outdir)}
    if witness is not None:
        return SatOutcome("SAT", rec, witness)
    if all(s == "UNSAT" for s in statuses):
        rec["proves"] = _proves_text(W, domain)
        return SatOutcome("UNSAT", rec)
    bad = [s for s in statuses if s != "UNSAT"]
    return SatOutcome("TIMEOUT" if "TIMEOUT" in bad else ("UNCHECKED" if "UNCHECKED" in bad else "ERROR"), rec)


# ---------------------------------------------------------------------------
# verification of a recorded proof (used by hyqec-verify verify)
# ---------------------------------------------------------------------------

def check_record(code: StabilizerCode, rec: dict, base_dir, *, lrat=False) -> dict:
    """Regenerate the CNF(s) from the code, compare hashes, re-check every proof.

    Returns {"verified": bool, "bound": W, "domain": ..., "details": [...]}."""
    base_dir = Path(base_dir)
    enc = encode_from_params(code, rec["params"])
    W, domain = rec["params"]["bound"], rec["params"]["domain"]
    items = []
    if rec["method"] == "sat-drat":
        items.append((enc.formula.dimacs(), rec))
    elif rec["method"] == "sat-drat-cubes":
        cubes = rec["cubes"]
        leaves = rec["leaves"]
        if not isinstance(cubes, list) or not cubes:
            raise CodeError("cube bundle: no cubes")
        for cube in cubes:
            if not isinstance(cube, list) or not cube or \
                    any(type(l) is not int or l == 0 or abs(l) > enc.formula.nv for l in cube):
                raise CodeError(f"cube bundle: invalid cube {cube!r}")
        if len(leaves) != len(cubes) + 1 or leaves[-1]["kind"] != "coverage":
            raise CodeError("cube bundle: wrong number of leaves")
        for i, cube in enumerate(cubes):
            if leaves[i]["kind"] != "leaf" or leaves[i]["cube"] != cube:
                raise CodeError("cube bundle: leaf/cube mismatch")
            items.append((enc.formula.dimacs([[l] for l in cube]), leaves[i]))
        items.append((enc.formula.dimacs([[-l for l in c] for c in cubes]), leaves[-1]))
    else:
        raise CodeError(f"not a SAT record: {rec.get('method')!r}")
    details = []
    ok = True
    with tempfile.TemporaryDirectory(prefix="hyqec-verify-") as tmp:
        tmp = Path(tmp)
        for j, (cnf, leaf) in enumerate(items):
            d = {"index": j}
            if sha256_bytes(cnf) != leaf.get("cnf_sha256"):
                d["error"] = "regenerated CNF hash differs from the record"
                ok = False
                details.append(d)
                continue
            pf = leaf.get("proof")
            if not pf:
                d["error"] = "no proof recorded"
                ok = False
                details.append(d)
                continue
            path = base_dir / pf["file"]
            try:
                h = hashlib.sha256()
                with gzip.open(path, "rb") as g, open(tmp / "p.drat", "wb") as out:   # stream to disk
                    for block in iter(lambda: g.read(1 << 22), b""):
                        h.update(block)
                        out.write(block)
            except (OSError, EOFError, ValueError) as e:      # missing, not gzip, truncated
                d["error"] = f"cannot read proof: {e}"
                ok = False
                details.append(d)
                continue
            if h.hexdigest() != pf["sha256"]:
                d["error"] = "proof hash mismatch"
                ok = False
                details.append(d)
                continue
            (tmp / "p.cnf").write_bytes(cnf)
            res = check_proof(tmp / "p.cnf", tmp / "p.drat", tmp, lrat=lrat)
            d.update(res)
            ok = ok and res["verified"]
            details.append(d)
    return {"verified": ok, "bound": W, "domain": domain, "proves": _proves_text(W, domain) if ok else None,
            "details": details}
