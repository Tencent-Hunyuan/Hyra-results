"""Orchestration: upper bound search, lower-bound proofs, exact distance, cross-checks.

Rules (never relaxed):
* an upper bound is the weight of a witness verified by the core;
* a lower bound is ``W+1`` for a completed proof "no logical of weight <= W"
  from an engine (brute force, B&B, SAT/DRAT with a checked proof);
* if any lower bound exceeds any verified witness weight, or two engines disagree
  on the same question, the result is ``inconsistent`` (exit code 3 in the CLI)
  and nothing is silently resolved.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .core import CodeError, StabilizerCode, xz_from_pauli
from .symmetry import Orbits, find_automorphisms, orbits_from, trivial_orbits

__all__ = ["DistanceResult", "distance", "ENGINES"]

ENGINES = ("brute", "bnb", "sat")
CURRENT = None          # the _State of the running distance() call (read by the CLI watchdog)


@dataclass
class DistanceResult:
    status: str                         # "exact" | "bounds" | "inconsistent" | "no_logical"
    domain: str
    lower: int
    upper: int | None
    witness: int | None
    evidence: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    seconds: float = 0.0
    workdir: str | None = None          # engine files (SAT proofs) referenced by the evidence
    seconds_to_upper: float | None = None   # when the final witness was found
    seconds_to_lower: float | None = None   # when the final lower bound was proved
    upper_bound_by: str | None = None
    lower_bound_by: str | None = None

    def cleanup(self) -> None:
        """Delete a temporary work directory created by distance() (after writing a certificate)."""
        if self.workdir and Path(self.workdir).name.startswith("hyqec-tmp-"):
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None

    def summary(self, n: int) -> dict:
        out = {"status": self.status, "domain": self.domain, "lower_bound": self.lower, "upper_bound": self.upper}
        if self.status == "exact":
            out["d"] = self.upper
        if self.witness is not None:
            out["witness"] = xz_from_pauli(self.witness, n)
        if self.notes:
            out["notes"] = self.notes
        out["seconds"] = self.seconds
        out["seconds_to_upper"] = self.seconds_to_upper
        out["seconds_to_lower"] = self.seconds_to_lower
        out["upper_bound_by"] = self.upper_bound_by
        out["lower_bound_by"] = self.lower_bound_by if self.lower > 1 else "trivial (d >= 1)"
        return out


def require_side_ok(code: StabilizerCode, domain: str) -> None:
    """Per-side (X/Z) distances are only defined here for CSS input (every given generator
    pure X or pure Z); there the X-only minimum over N(S) minus S equals d_X (SOUNDNESS.md 1)."""
    if domain not in ("P", "X", "Z"):
        raise CodeError(f"unknown side {domain!r}")
    if domain != "P" and not code.is_css:
        raise CodeError(f"--side {domain} needs a CSS presentation (every generator pure X or pure Z); "
                        "for a non-CSS code use the Pauli-weight distance (--side P)")


def _in_domain(code, v, domain):
    n = code.n
    return domain == "P" or (domain == "X" and not v >> n) or (domain == "Z" and not v & ((1 << n) - 1))


class _State:
    def __init__(self, code, domain):
        self.code, self.domain = code, domain
        self.t0 = time.monotonic()
        self.t_upper = None          # seconds until the final upper bound was found
        self.t_lower = None          # seconds until the final lower bound was proved
        self.engines_used = []       # upper-bound search engines in the order they ran (all phases)
        self.upper_by = None         # source of the final witness ("input", "ris-cpu", "bnb", "sat", ...)
        self.lower_by = None         # engine that proved the final lower bound ("brute", "bnb", "sat")
        self.lower = 1
        self.upper = None
        self.witness = None
        self.evidence = []
        self.notes = []
        self.inconsistent = False
        # every completed claim "no logical of weight <= W" per engine, for cross-checks
        self.none_upto: dict[str, int] = {}

    def offer_witness(self, v, source):
        if v is None:
            return
        w = self.code.check_witness(v)
        if not _in_domain(self.code, v, self.domain):
            raise AssertionError(f"{source}: witness outside the domain")
        if self.upper is None or w < self.upper:
            self.upper, self.witness = w, v
            self.t_upper = round(time.monotonic() - self.t0, 3)
            self.upper_by = source
            self.evidence.append({"method": "witness", "source": source, "weight": w,
                                  "witness": xz_from_pauli(v, self.code.n)})
        self._check()

    def offer_lower(self, W, record, engine):
        """record proves: no logical of weight <= W (in the domain)."""
        record = dict(record)
        record["lower_bound"] = W + 1
        self.evidence.append(record)
        self.none_upto[engine] = max(self.none_upto.get(engine, 0), W)
        if W + 1 > self.lower:
            self.lower = W + 1
            self.t_lower = round(time.monotonic() - self.t0, 3)
            self.lower_by = engine
        self._check()

    def _check(self):
        if self.upper is not None and self.lower > self.upper:
            self.inconsistent = True
            self.notes.append(f"INCONSISTENT: lower bound {self.lower} > verified witness weight {self.upper}")

    @property
    def exact(self):
        return self.upper is not None and self.lower == self.upper


def distance(code: StabilizerCode, *, domain: str = "P", engines: Sequence[str] = ("bnb",),
             cross_check: Sequence[str] = (), budget: float = 600.0, ris_seconds: float | None = None,
             witnesses: Sequence[int] = (), threads: int | None = None, gpus: str | None = None,
             symmetry: str = "auto", workdir=None, sat_workers: int = 8, sat_solver: str = "kissat",
             brute_limit: int = 2_000_000, seed: int = 0, ris_engine: str = "auto",
             bnb_cert: bool = False) -> DistanceResult:
    """Best certified bounds on the (domain) distance within the time budget."""
    t0 = time.monotonic()
    deadline = t0 + budget
    if code.k < 1:
        return DistanceResult("no_logical", domain, 0, None, None, notes=[f"k = {code.k}: no logical qubits"])
    require_side_ok(code, domain)
    for e in list(engines) + list(cross_check):
        if e not in ENGINES:
            raise ValueError(f"unknown engine {e!r} (choose from {ENGINES})")
    st = _State(code, domain)
    global CURRENT
    CURRENT = st
    from . import deadline as _dl
    _dl.set_deadline(deadline)
    own_tmp = workdir is None
    workdir = Path(tempfile.mkdtemp(prefix="hyqec-tmp-")) if own_tmp else Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        for v in witnesses:
            if code.is_logical(v) and _in_domain(code, v, domain):
                st.offer_witness(v, "input")
            else:
                st.notes.append("an input witness was rejected (not a nontrivial logical in the domain)")

        # 1. upper bound search (skipped when exhaustive search of everything is cheap)
        from .engines.brute import brute_force, brute_work
        tiny = brute_work(code.n, code.n, domain) <= brute_limit
        if not tiny:
            # an explicit ris_seconds is spent in full; the default stops after 3 s without improvement
            secs = ris_seconds if ris_seconds is not None else min(60.0, 0.1 * budget)
            secs = min(secs, max(0.0, deadline - time.monotonic()))
            patience = None if ris_seconds is not None else 3.0
            _ris(code, st, domain, secs, seed, ris_engine, threads, gpus, patience)

        # 2. exhaustive search when cheap (or requested): exact by itself
        top = st.upper - 1 if st.upper is not None else code.n
        if top >= 1 and ("brute" in engines or brute_work(code.n, top, domain) <= brute_limit):
            r = brute_force(code, top, domain=domain, work_limit=max(brute_limit, brute_work(code.n, top, domain))
                            if "brute" in engines else brute_limit, deadline=deadline)
            if r.found is None and r.max_weight < top:
                st.notes.append(f"brute: time budget exhausted after weight {r.max_weight}")
            if r.found is None and r.max_weight < 1:
                pass
            elif r.found is not None:
                if r.found > 1:
                    st.offer_lower(r.found - 1, {"method": "brute", "examined": r.examined,
                                                 "proves": _proves(r.found - 1, domain)}, "brute")
                st.offer_witness(r.witness, "brute")
            else:
                st.offer_lower(r.max_weight, {"method": "brute", "examined": r.examined,
                                              "proves": _proves(r.max_weight, domain)}, "brute")

        # 3. orbits for symmetry reduction (verified automorphisms)
        orbits = None
        if not st.exact and symmetry != "none" and code.n > 2:
            perms = find_automorphisms(code, symmetry)
            if perms:
                orbits = orbits_from(code, perms)
                st.notes.append(f"symmetry: {len(perms)} verified generators, {orbits.count} qubit orbits")

        # 4. lower bounds: ladder of caps up to upper-1 for each primary engine
        for eng in engines:
            if eng == "brute" or st.exact:
                continue
            _ladder(code, st, eng, domain, orbits, deadline, workdir / eng, threads, gpus,
                    sat_workers, sat_solver, bnb_cert=bnb_cert)

        # 5. cross-check: the other engine(s) on the final question
        for eng in cross_check:
            if st.upper is None or st.lower < 2:
                continue
            W = st.lower - 1
            if st.none_upto.get(eng, 0) >= W:
                continue
            _ladder(code, st, eng, domain, orbits, deadline, workdir / f"x-{eng}", threads, gpus,
                    sat_workers, sat_solver, start=W, stop=W, bnb_cert=bnb_cert)
            if st.none_upto.get(eng, 0) < W:
                st.notes.append(f"cross-check by {eng} did not finish for W={W}")
    except BaseException:
        _dl.clear_deadline()
        if own_tmp:
            shutil.rmtree(workdir, ignore_errors=True)
        raise
    _dl.clear_deadline()
    if time.monotonic() > deadline - 0.05 and not st.exact:
        st.notes.append("time budget reached: results are the bounds proved so far")
    # a temporary work dir is kept only while evidence refers to files in it (call result.cleanup())
    keep = any(e.get("method", "").startswith("sat") or e.get("coverage") for e in st.evidence)
    if own_tmp and not keep:
        shutil.rmtree(workdir, ignore_errors=True)
    status = "inconsistent" if st.inconsistent else ("exact" if st.exact else "bounds")
    return DistanceResult(status, domain, st.lower, st.upper, st.witness, st.evidence, st.notes,
                          round(time.monotonic() - t0, 3), str(workdir) if (keep or not own_tmp) else None,
                          st.t_upper, st.t_lower, st.upper_by, st.lower_by)


def _proves(W, domain):
    kind = {"P": "nontrivial logical", "X": "X-type nontrivial logical", "Z": "Z-type nontrivial logical"}[domain]
    return f"no {kind} of Pauli weight <= {W}"


CPU_STALL_SECONDS = 3.0     # auto policy: the CPU phase ends after this long without improvement,
CPU_FIRST_SECONDS = 5.0     # ... and, when a GPU is offered, after at most this long in any case


def _native_slices(code, st, domain, seconds, seed, gpu, threads, gpus, patience, target):
    """Native RIS in slices of growing length.

    Returns (why it stopped: "target" | "idle" | "time", seconds since the last improvement)."""
    from .engines import ris_native
    dev = int(gpus.split(",")[0]) if gpu and gpus else 0
    th = threads or min(16, os.cpu_count() or 1)
    patience = patience if patience is not None else float("inf")
    t_end = time.monotonic() + seconds
    slice_s, idle, i = 0.5, 0.0, 0
    while time.monotonic() < t_end:
        if idle >= patience:
            return "idle", idle
        sl = min(slice_s, max(0.1, t_end - time.monotonic()))
        before = st.upper
        r = ris_native.run(code, seconds=sl, seed=seed + i, domain=domain, target=target, gpu=gpu,
                           threads=th, device=dev)
        if r["engine"] not in st.engines_used:
            st.engines_used.append(r["engine"])
        if r["witness_int"] is not None:
            st.offer_witness(r["witness_int"], r["engine"])
        idle = 0.0 if st.upper != before else idle + sl
        if target is not None and st.upper is not None and st.upper < target:
            return "target", idle
        slice_s = min(2 * slice_s, 30.0)
        i += 1
    return "time", idle


def _ris(code, st, domain, seconds, seed, engine, threads, gpus, patience=None):
    """Upper-bound search for `seconds` (stops early only on reaching the target, or after
    `patience` idle seconds when a patience is given, as inside `distance`).

    engine "auto": native CPU RIS until it stalls (CPU_STALL_SECONDS without improvement) or, with
    --gpu, for at most CPU_FIRST_SECONDS; then - with --gpu and a CUDA build - the GPU for the rest
    of the time; without a GPU, the CPU keeps going with new seeds. "cpu"/"gpu": that native engine only. Python RIS if nothing native is built."""
    from .engines.ris import ris_upper_bound
    if seconds <= 0:
        return
    target = st.lower + 1 if st.lower else None
    if engine in ("auto", "cpu", "gpu"):
        try:
            from . import native
            from .engines import ris_native
            have = lambda t: t in ris_native._LOCATED or native.available(t)
            if engine == "gpu":
                if not have("ris-cuda"):
                    raise CodeError("native RIS target ris-cuda unavailable")
                _native_slices(code, st, domain, seconds, seed, True, threads, gpus, patience, target)
                return
            if have("ris-cpu"):
                t_end = time.monotonic() + seconds
                stall = CPU_STALL_SECONDS if patience is None else min(CPU_STALL_SECONDS, patience)
                cpu_phase = min(seconds, CPU_FIRST_SECONDS) if (engine == "auto" and gpus) else seconds
                why, idle = _native_slices(code, st, domain, cpu_phase, seed, False, threads, gpus, stall, target)
                left = t_end - time.monotonic()
                if why == "target" or left <= 0.5:
                    return
                rest = None if patience is None else max(0.0, patience - idle)
                if engine == "auto" and gpus and have("ris-cuda"):
                    st.notes.append(f"upper bound search moved to the GPU after {seconds - left:.1f} s on CPU "
                                    f"({'stalled' if why == 'idle' else 'CPU phase limit'})")
                    _native_slices(code, st, domain, left, seed + 1000, True, threads, gpus, rest, target)
                else:
                    _native_slices(code, st, domain, left, seed + 1000, False, threads, gpus, rest, target)
                return
            if engine != "auto":
                raise CodeError("native RIS target ris-cpu unavailable")
        except CodeError:
            if engine != "auto":
                raise
            st.notes.append("native RIS failed; used the Python RIS")
        except Exception as e:          # native build problems: fall back, but say so
            if engine != "auto":
                raise
            st.notes.append(f"native RIS unavailable ({type(e).__name__}); used the Python RIS")
    r = ris_upper_bound(code, seconds=seconds, seed=seed, domain=domain, target=target, patience=patience)
    if r.witness is not None:
        st.offer_witness(r.witness, r.engine)


def _ladder(code, st, eng, domain, orbits, deadline, workdir, threads, gpus, sat_workers, sat_solver,
            start=None, stop=None, bnb_cert=False):
    """Prove 'no logical <= W' for W = start .. stop (default: lower .. upper-1) while time remains."""
    top = (st.upper - 1) if st.upper is not None else code.n - 1
    W = st.lower if start is None else start
    if eng == "bnb":
        # locate (and, on first use, build) the binary before the budget clock is read for each rung:
        # locating probes compilers and must not eat into a rung's time limit
        from .engines.bnb import locate_binary
        locate_binary(code, gpu=bool(gpus))
    stop = top if stop is None else stop
    while W <= stop and not st.inconsistent:
        left = deadline - time.monotonic()
        if left <= 0.05:
            st.notes.append(f"{eng}: time budget exhausted before W={W} ({max(left, 0):.2f} s left)")
            return
        d = workdir / f"W{W}"
        if eng == "brute":
            from .engines.brute import brute_force
            r = brute_force(code, W, domain=domain, work_limit=10 ** 12, deadline=deadline)
            if r.found is not None:
                st.offer_witness(r.witness, "brute")
                if r.found > 1:
                    st.offer_lower(r.found - 1, {"method": "brute", "examined": r.examined,
                                                 "proves": _proves(r.found - 1, domain)}, "brute")
                return
            if r.max_weight < W:
                st.notes.append(f"brute W={W}: time budget exhausted")
                return
            st.offer_lower(W, {"method": "brute", "examined": r.examined, "proves": _proves(W, domain)}, "brute")
            W += 1
            continue
        if eng == "bnb":
            from .engines.bnb import run_bnb
            o = run_bnb(code, W, d, domain=domain, orbits=orbits, threads=threads, gpus=gpus, timeout=left,
                        emit_cert=bnb_cert)
            if o.status == "none":
                st.offer_lower(W, o.record, "bnb")
            elif o.status == "upper":
                st.offer_witness(o.witness, "bnb")
                stop = min(stop, st.upper - 1)
                continue
            elif o.status == "found":
                # min mode: f is the minimum in the domain
                st.offer_witness(o.witness, "bnb")
                if o.found > 1:
                    st.offer_lower(o.found - 1, o.record, "bnb")
                return
            else:
                st.notes.append(f"bnb W={W}: {o.status} {o.record.get('error', '')}".strip())
                return
        elif eng == "sat":
            from .engines import sat
            # leave time for the proof check (checking typically takes about as long as solving)
            o = sat.prove(code, W, d, domain=domain, orbits=orbits, solver=sat_solver,
                          timeout=0.5 * left, check_timeout=max(1.0, deadline - time.monotonic()))
            if o.status == "UNSAT":
                o.record["artifacts"] = str(d)
                st.offer_lower(W, o.record, "sat")
            elif o.status == "SAT":
                st.offer_witness(o.model_pauli, "sat")
                # SAT at W: the minimum is <= weight(model); continue below the new upper bound
                stop = min(stop, st.upper - 1)
                continue
            else:
                st.notes.append(f"sat W={W}: {o.status}")
                return
        W += 1
