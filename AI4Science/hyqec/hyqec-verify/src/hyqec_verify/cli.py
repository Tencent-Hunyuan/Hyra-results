"""Command line interface: ``hyqec-verify <command> ...`` (``hyqec-verify -h`` for the list).

Exit codes: 0 a verdict was produced; 1 usage error or input rejected;
2 inconclusive (bounds only / timeout); 3 inconsistency between engines or
between a lower bound and a verified witness.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ._env import getenv
from .core import CodeError, xz_from_pauli

EXIT_OK, EXIT_INPUT, EXIT_INCONCLUSIVE, EXIT_INCONSISTENT = 0, 1, 2, 3


def _emit(obj, args, human=None):
    if getattr(args, "json", False) or human is None:
        print(json.dumps(obj, sort_keys=True))
    else:
        print(human)


def _load(args):
    from .io import load
    lc = load(args.code, n=getattr(args, "n", None))
    problems = lc.check_claims() if not getattr(args, "ignore_claims", False) else []
    if problems:
        raise CodeError("; ".join(problems))
    return lc


def _code_info(lc):
    c = lc.code
    n = c.n
    gw = [c.weight(g) for g in c.generators if g]
    col = [0] * n
    for g in c.generators:
        for q in range(n):
            if ((g | (g >> n)) >> q) & 1:
                col[q] += 1
    return {"n": n, "k": c.k, "rank": c.rank, "generators": len(c.generators), "is_css": c.is_css,
            "max_generator_weight": max(gw, default=0), "max_column_degree": max(col, default=0),
            "group_sha256": c.group_hash(), "format": lc.fmt, "claimed_d": lc.claimed_d}


def cmd_info(args):
    lc = _load(args)
    info = _code_info(lc)
    _emit(info, args, "\n".join(f"{k:22s} {v}" for k, v in info.items()))
    return EXIT_OK


def cmd_witness(args):
    from .io import parse_witness
    lc = _load(args)
    c = lc.code
    text = args.witness
    if os.path.exists(text):
        text = open(text).read()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = text
    v = parse_witness(obj, c.n)
    prob = c.witness_problem(v)
    w = c.weight(v)
    if prob is None and args.claimed_weight is not None and args.claimed_weight != w:
        prob = "weight_mismatch"
    out = {"status": "valid_witness" if prob is None else "invalid_witness", "weight": w}
    if prob:
        out["reason"] = prob
    _emit(out, args, f"{out['status']} weight={w}" + (f" ({prob})" if prob else ""))
    return EXIT_OK


def _distance_kwargs(args):
    args.engines_explicit = args.engines is not None
    if args.engines is None:
        args.engines = "bnb"
    return dict(domain=args.side, engines=tuple(e for e in args.engines.split(",") if e),
                cross_check=tuple(e for e in (args.cross_check or "").split(",") if e),
                budget=args.budget, ris_seconds=args.ris_seconds, threads=args.threads, gpus=args.gpu,
                symmetry=args.symmetry, sat_workers=args.sat_workers, sat_solver=args.solver, seed=args.seed,
                brute_limit=args.brute_limit, bnb_cert=args.bnb_cert)


WATCHDOG_GRACE = 30.0      # seconds after --budget before children are killed
WATCHDOG_HARD = 90.0       # seconds after --budget before the process exits with the bounds so far


def _watchdog(args, lc):
    """Backstop for --budget: kill engine children, then (if still running) exit 2 with partial bounds."""
    import threading
    import importlib
    from . import deadline
    dmod = importlib.import_module("hyqec_verify.distance")    # (hyqec_verify.distance the attribute is the function)

    def soft():
        n = deadline.kill_children()
        print(f"warning: --budget exceeded by {WATCHDOG_GRACE:.0f} s; killed {n} engine process(es)", file=sys.stderr)

    def hard():
        st = dmod.CURRENT
        out = {"status": "bounds", "inconclusive": "budget", "n": lc.code.n, "k": lc.code.k,
               "lower_bound": st.lower if st else 1, "upper_bound": st.upper if st else None,
               "lower_bound_by": (st.lower_by if st and st.lower > 1 else "trivial (d >= 1)"),
               "upper_bound_by": st.upper_by if st else None,
               "witness": xz_from_pauli(st.witness, lc.code.n) if st and st.witness is not None else None,
               "notes": ["stopped by the --budget watchdog; no certificate written"]}
        print(json.dumps(out, sort_keys=True), flush=True)
        os._exit(EXIT_INCONCLUSIVE)
    t1 = threading.Timer(args.budget + WATCHDOG_GRACE, soft)
    t2 = threading.Timer(args.budget + WATCHDOG_HARD, hard)
    for t in (t1, t2):
        t.daemon = True
        t.start()
    return t1, t2


def _run_distance(args, lc):
    from .distance import distance
    wit = [] if args.no_input_witness else list(lc.claimed_witnesses.values())
    timers = _watchdog(args, lc)
    try:
        return distance(lc.code, witnesses=wit, workdir=args.workdir, **_distance_kwargs(args))
    finally:
        for t in timers:
            t.cancel()


def _result_json(lc, res):
    c = lc.code
    out = res.summary(c.n)
    out.update(n=c.n, k=c.k, rank=c.rank)
    if res.status == "no_logical":
        out["d"] = None
    if res.status == "bounds" and (any("time budget" in n or "TIMEOUT" in n or ": timeout" in n for n in res.notes)):
        out["inconclusive"] = "budget"
    return out


def cmd_distance(args):
    lc = _load(args)
    if lc.code.k < 1:
        _emit({"status": "no_logical", "d": None, "n": lc.code.n, "k": lc.code.k, "rank": lc.code.rank}, args,
              f"k = {lc.code.k}: no logical qubits, distance undefined")
        return EXIT_OK
    import tempfile
    keep = args.certificate is not None
    if keep and args.workdir is None:
        args.workdir = tempfile.mkdtemp(prefix="hyqec-run-")
    res = _run_distance(args, lc)
    out = _result_json(lc, res)
    if args.certificate:
        from .certificate import build_certificate, write_certificate
        from .tools import sha256_file
        src = {"path": os.path.basename(args.code), "sha256": sha256_file(args.code)}
        write_certificate(build_certificate(lc.code, res, src), args.certificate)
        out["certificate"] = args.certificate
    human = (f"[[{lc.code.n},{lc.code.k},{res.upper}]] exact" if res.status == "exact" else
             f"{res.status}: {res.lower} <= d <= {res.upper}")
    human += (f"\n  lower bound {res.lower} by {out['lower_bound_by']}; "
              f"upper bound {res.upper} by {out['upper_bound_by']}")
    asked = [e for e in args.engines.split(",") if e] if args.engines_explicit else []
    if asked:
        # true only if a requested engine proved the final lower bound; the trivial d >= 1 is not
        # proved by any engine (for d = 1 nothing needs proving, so that case gets a note, no warning)
        proved = res.lower > 1 and res.lower_bound_by in asked
        out["requested_engines_proved_lower_bound"] = proved
        if not proved and res.lower <= 1 and res.status == "exact":
            out.setdefault("notes", []).append("d = 1: the lower bound d >= 1 is trivial; no engine was needed")
        elif not proved:
            if res.lower <= 1:
                warn = (f"requested engine(s) {','.join(asked)} proved no lower bound beyond the trivial d >= 1")
            else:
                warn = (f"requested engine(s) {','.join(asked)} did not prove the final lower bound "
                        f"(it came from {res.lower_bound_by}); use --brute-limit 0 to force them")
            out.setdefault("notes", []).append("WARNING: " + warn)
            print("warning: " + warn, file=sys.stderr)
    if res.notes and not args.json:
        human += "\n" + "\n".join("  note: " + s for s in res.notes)
    _emit(out, args, human)
    return {"exact": EXIT_OK, "bounds": EXIT_INCONCLUSIVE, "inconsistent": EXIT_INCONSISTENT}.get(res.status, EXIT_OK)


def cmd_claim(args):
    lc = _load(args)
    D = args.d
    if lc.code.k < 1:
        _emit({"status": "invalid_claim", "d": None, "reason": "k = 0"}, args, "invalid_claim (k = 0)")
        return EXIT_OK
    res = _run_distance(args, lc)
    out = {"lower_bound": res.lower, "upper_bound": res.upper}
    if res.status == "inconsistent":
        out["status"] = "inconsistent"
        code = EXIT_INCONSISTENT
    elif res.status == "exact":
        out.update(status="valid_claim" if res.upper == D else "invalid_claim", d=res.upper)
        code = EXIT_OK
    elif (res.upper is not None and res.upper < D) or res.lower > D:
        out["status"] = "invalid_claim"
        code = EXIT_OK
    else:
        out["status"] = "bounds"
        code = EXIT_INCONCLUSIVE
    if res.witness is not None:
        out["witness"] = xz_from_pauli(res.witness, lc.code.n)
    _emit(out, args, f"{out['status']}: {res.lower} <= d <= {res.upper} (claim {D})")
    return code


def cmd_upper(args):
    from .distance import require_side_ok
    from .engines.ris import ris_upper_bound
    lc = _load(args)
    lc.code.require_logicals()
    require_side_ok(lc.code, args.side)
    if args.no_chain and args.engine not in ("cpu", "gpu"):
        raise CodeError("--no-chain needs --engine cpu or --engine gpu")
    if args.engine == "auto":
        from .distance import _State, _ris
        st = _State(lc.code, args.side)
        _ris(lc.code, st, args.side, args.seconds, args.seed, "auto", args.threads, args.gpu, None)
        out = {"upper_bound": st.upper, "witness": xz_from_pauli(st.witness, lc.code.n) if st.witness else None,
               "engine": st.upper_by, "engines_used": st.engines_used, "seconds_to_upper": st.t_upper,
               "notes": st.notes}
    elif args.engine == "python":
        r = ris_upper_bound(lc.code, seconds=args.seconds, seed=args.seed, domain=args.side, target=args.target)
        out = {"upper_bound": r.weight, "witness": xz_from_pauli(r.witness, lc.code.n) if r.witness else None,
               "trials": r.trials, "seconds": round(r.seconds, 2), "engine": r.engine}
    else:
        from .engines import ris_native
        r = ris_native.run(lc.code, seconds=args.seconds, seed=args.seed, domain=args.side, target=args.target,
                           gpu=args.engine == "gpu", threads=args.threads, device=args.device,
                           no_chain=args.no_chain)
        out = {k: v for k, v in r.items() if k != "witness_int"}
    _emit(out, args, f"upper bound {out['upper_bound']}")
    return EXIT_OK


def cmd_lower(args):
    from .distance import require_side_ok
    lc = _load(args)
    lc.code.require_logicals()
    require_side_ok(lc.code, args.side)
    W = args.bound
    workdir = args.workdir or os.path.join(os.getcwd(), f"hyqec-lower-W{W}")
    if args.engine == "bnb":
        from .engines.bnb import run_bnb
        orb = _orbits(args, lc.code)
        o = run_bnb(lc.code, W, workdir, domain=args.side, orbits=orb, threads=args.threads, gpus=args.gpu,
                    timeout=args.timeout, ckpt=args.checkpoint, emit_cert=args.bnb_cert,
                    root_order=args.root_order, bound=args.packing, slots=args.gpu_slots)
        out = {"status": o.status, "proves": o.record.get("proves"), "record": o.record}
        if o.witness is not None:
            out["witness"] = xz_from_pauli(o.witness, lc.code.n)
        ok = o.status in ("none", "found")
    elif args.engine == "sat":
        from .engines import sat
        orb = _orbits(args, lc.code, sat_default=True)
        if args.cubes:
            from .symmetry import trivial_orbits
            o = sat.prove_cubes(lc.code, W, workdir, orbits=orb or trivial_orbits(lc.code.n),
                                domain=args.side, n_first=args.cubes,
                                split_labels=args.split_labels, solver=args.solver, workers=args.sat_workers,
                                timeout=args.timeout, lrat=args.lrat)
        else:
            o = sat.prove(lc.code, W, workdir, domain=args.side, orbits=orb, solver=args.solver,
                          timeout=args.timeout, lrat=args.lrat)
        out = {"status": o.status, "proves": o.record.get("proves"), "record": o.record}
        if o.model_pauli is not None:
            out["witness"] = xz_from_pauli(o.model_pauli, lc.code.n)
        ok = o.status in ("UNSAT", "SAT")
    else:
        from .engines.brute import brute_force
        r = brute_force(lc.code, W, domain=args.side, work_limit=10 ** 12)
        out = {"status": "none" if r.found is None else "found", "examined": r.examined}
        if r.found is None:
            out["proves"] = f"no {'nontrivial' if args.side == 'P' else args.side + '-type nontrivial'} logical of Pauli weight <= {W}"
        if r.found is not None:
            out["witness"] = xz_from_pauli(r.witness, lc.code.n)
        ok = True
    # one public status vocabulary for every engine; the engine's own word stays in engine_status
    raw = out["status"]
    out["engine_status"] = raw
    out["status"] = {"none": "proved", "UNSAT": "proved", "found": "found", "SAT": "found", "upper": "found",
                     "TIMEOUT": "inconclusive", "timeout": "inconclusive", "UNCHECKED": "inconclusive"}.get(raw, "error")
    if raw == "found" and args.engine == "bnb":
        out["minimum"] = True            # min mode: the found weight is the minimum in the domain
    if args.certificate:
        out["certificate"] = _lower_certificate(args, lc, W, out)
    _emit(out, args, json.dumps({k: v for k, v in out.items() if k != "record"}))
    return EXIT_OK if out["status"] in ("proved", "found") else (
        EXIT_INCONCLUSIVE if out["status"] == "inconclusive" else EXIT_INCONSISTENT)


def _lower_certificate(args, lc, W, out):
    """Certificate from one lower-bound run plus the input file's verified witnesses."""
    from .certificate import build_certificate, write_certificate
    from .distance import DistanceResult
    from .core import pauli_from_xz
    from .tools import sha256_file
    code, n, dom = lc.code, lc.code.n, args.side
    ev, lower = [], 1
    upper = wit = None
    cands = [] if args.no_input_witness else list(lc.claimed_witnesses.values())
    if out.get("witness"):
        cands.append(pauli_from_xz(out["witness"]["X"], out["witness"]["Z"], n))
    for v in cands:
        ok_dom = dom == "P" or (dom == "X" and not v >> n) or (dom == "Z" and not v & ((1 << n) - 1))
        if code.is_logical(v) and ok_dom and (upper is None or code.weight(v) < upper):
            upper, wit = code.weight(v), v
    if wit is not None:
        ev.append({"method": "witness", "source": "input" if out.get("witness") is None else args.engine,
                   "weight": upper, "witness": xz_from_pauli(wit, n)})
    rec = dict(out.get("record") or {})
    if out.get("proves") and out["status"] == "proved":
        rec.setdefault("method", {"bnb": "bnb", "sat": "sat-drat", "brute": "brute"}[args.engine])
        rec["lower_bound"] = W + 1
        rec["proves"] = out["proves"]
        if args.engine == "sat":
            rec.setdefault("directory", rec.get("directory") or args.workdir)
        ev.append(rec)
        lower = W + 1
    elif out.get("minimum") and out.get("witness"):
        # min mode: the run proved nothing lighter than its find
        rec["lower_bound"] = upper
        ev.append(rec)
        lower = upper
    status = "inconsistent" if upper is not None and lower > upper else (
        "exact" if upper is not None and lower == upper else "bounds")
    res = DistanceResult(status, dom, lower, upper, wit, ev)
    src = {"path": os.path.basename(args.code), "sha256": sha256_file(args.code)}
    write_certificate(build_certificate(code, res, src), args.certificate)
    return args.certificate


def _orbits(args, code, sat_default=False):
    from .symmetry import find_automorphisms, orbits_from
    if args.symmetry == "none":
        return None
    perms = find_automorphisms(code, args.symmetry)
    return orbits_from(code, perms) if perms else None


def cmd_verify(args):
    from .certificate import verify_certificate
    r = verify_certificate(args.certificate, rerun_bnb=args.rerun_bnb, lrat=args.lrat, threads=args.threads,
                           bnb_checker=args.bnb_checker or getenv("BNB_CHECKER"))
    human = (f"{r['status']}: verified {r['verified_lower']} <= d <= {r['verified_upper']}"
             f" (claimed {r['claimed'].get('lower_bound')}..{r['claimed'].get('upper_bound')})")
    if r["bnb_records_not_rerun"]:
        human += (f"\n  {r['bnb_records_not_rerun']} B&B record(s) not checked (use --bnb-checker for coverage "
                  f"certificates, --rerun-bnb for run records); accepting them gives d >= {r['lower_with_recorded_bnb']}")
    for c in r["checks"]:
        if c.get("evidence_level"):
            human += f"\n  {c['item']}: {c['evidence_level']}"
        if c.get("ok") is None and c.get("problem"):
            human += f"\n  note: {c['item']}: {c['problem']}"
    for f in r["failed"]:
        human += f"\n  FAILED {f['item']} {f.get('method', '')} {f.get('problem', '')}"
    if r["claim_mismatch"]:
        human += f"\n  CLAIM NOT REPRODUCED: {', '.join(r['claim_mismatch'])} differ from the verified result"
    _emit(r, args, human)
    if r["failed"] or not r["claims_reproduced"]:
        return EXIT_INCONSISTENT
    return EXIT_OK


def cmd_wrap_coverage(args):
    """Certificate from an existing B&B coverage file + the code file's verified witness."""
    from .certificate import build_certificate, coverage_evidence, write_certificate
    from .distance import DistanceResult
    from .tools import sha256_file
    lc = _load(args)
    code, n = lc.code, lc.code.n
    ev_b = coverage_evidence(code, args.coverage)
    wits = [v for v in lc.claimed_witnesses.values() if code.is_logical(v)]
    best = min(wits, key=code.weight) if wits else None
    ev = ([{"method": "witness", "source": "input", "weight": code.weight(best), "witness": xz_from_pauli(best, n)}]
          if best is not None else []) + [ev_b]
    lo, up = ev_b["lower_bound"], (code.weight(best) if best is not None else None)
    st = "inconsistent" if up is not None and lo > up else ("exact" if up == lo else "bounds")
    res = DistanceResult(st, ev_b["params"]["domain"], lo, up, best, ev,
                         ["lower bound claimed by an external coverage file; counted only after "
                          "'hyqec-verify verify --bnb-checker' accepts it"])
    write_certificate(build_certificate(code, res, {"path": os.path.basename(args.code),
                                                    "sha256": sha256_file(args.code)}), args.certificate)
    out = {"certificate": args.certificate, "claimed_lower_bound": lo, "upper_bound": up, "status": st}
    _emit(out, args, f"wrote {args.certificate}: claims {lo} <= d <= {up}; check with hyqec-verify verify --bnb-checker")
    return EXIT_OK


def cmd_tools(args):
    from . import tools
    if args.action == "install":
        names = args.names or ["kissat", "cadical"]
        for nm in names:
            p = tools.install_solver(nm, from_source=args.from_source, jobs=args.jobs)
            print(f"{nm}: {p}")
        return EXIT_OK
    return cmd_doctor(args)


def cmd_doctor(args):
    from . import tools
    out = {"python": sys.version.split()[0]}
    for nm in ("kissat", "cadical"):
        try:
            out[nm] = tools.tool_info(tools.find_solver(nm))
        except tools.ToolMissing as e:
            out[nm] = {"missing": str(e)}
    for nm, fn in (("drat-trim", tools.checker), ("lrat-check", tools.lrat_checker)):
        try:
            p = fn()
            out[nm] = {"path": str(p), "sha256": tools.sha256_file(p)}
        except (tools.ToolMissing, OSError) as e:
            out[nm] = {"missing": str(e)}
    try:
        from . import native
        status = {t: native.availability(t) for t in native.TARGETS}
        out["native"] = {t: ok for t, (ok, reason) in status.items()}
        out["native_unavailable"] = {t: reason for t, (ok, reason) in status.items() if not ok}
    except Exception as e:      # native module optional
        out["native"] = {"error": str(e)}
    try:
        import pynauty  # noqa: F401
        out["pynauty"] = True
    except ImportError:
        out["pynauty"] = False
    _emit(out, args, json.dumps(out, indent=1))
    return EXIT_OK


def cmd_build(args):
    from . import native
    targets = args.targets or ["bnb-cpu", "ris-cpu"]
    for t in targets:
        try:
            p = native.build(t, cuda_arch=args.cuda_arch, force=args.force, verbose=True)
        except native.NativeUnavailable as exc:
            raise CodeError(str(exc)) from exc
        print(f"{t}: {p}")
    return EXIT_OK


def _common_code(p):
    p.add_argument("code", help="code file (challenge JSON, generator JSON, .npz, or text Pauli strings)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--ignore-claims", action="store_true", help="do not reject inputs whose claimed k/rank differ")


def _common_search(p):
    p.add_argument("--side", choices=["P", "X", "Z"], default="P",
                   help="P: Pauli-weight distance (default); X / Z: CSS X- / Z-distance (CSS input only)")
    p.add_argument("--engines", default=None,
                   help="lower-bound engines, comma separated: bnb, sat, brute (default bnb). Small codes are "
                        "answered by exhaustive search first unless --brute-limit 0")
    p.add_argument("--cross-check", default="", help="engines that must independently confirm the final bound")
    p.add_argument("--budget", type=float, default=600.0, help="wall-clock budget in seconds")
    p.add_argument("--ris-seconds", type=float, default=None, help="time for the upper-bound search")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--gpu", default=None, help="CUDA devices for B&B, e.g. 0,1")
    p.add_argument("--symmetry", default="auto", choices=["auto", "nauty", "cyclic", "none"])
    p.add_argument("--sat-workers", type=int, default=8)
    p.add_argument("--solver", default="kissat", choices=["kissat", "cadical"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workdir", default=None, help="keep engine files here")
    p.add_argument("--no-input-witness", action="store_true",
                   help="ignore witnesses in the input file (by default verified input witnesses are upper bounds)")
    p.add_argument("--bnb-cert", action="store_true",
                   help="B&B: also write a coverage certificate (docs/BNB_CERT_SPEC.md) for each proved bound")
    p.add_argument("--brute-limit", type=int, default=2_000_000,
                   help="use exhaustive search when it needs at most this many candidates (0: never)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hyqec-verify", description="Certified distances of quantum stabilizer codes.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="validate a code and print n, k, weights")
    _common_code(p)
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("witness", help="check a claimed logical operator")
    _common_code(p)
    p.add_argument("--witness", required=True, help='JSON {"X":[..],"Z":[..]}, a Pauli string, or a file')
    p.add_argument("--claimed-weight", type=int, default=None)
    p.set_defaults(fn=cmd_witness)

    p = sub.add_parser("distance", help="certified bounds / exact distance")
    _common_code(p)
    _common_search(p)
    p.add_argument("-o", "--certificate", default=None, help="write a certificate (JSON + artifacts dir)")
    p.set_defaults(fn=cmd_distance)

    p = sub.add_parser("claim", help="check a claimed exact distance")
    _common_code(p)
    _common_search(p)
    p.add_argument("--d", type=int, required=True)
    p.set_defaults(fn=cmd_claim)

    p = sub.add_parser("upper", help="upper-bound search (random information sets)")
    _common_code(p)
    p.add_argument("--side", choices=["P", "X", "Z"], default="P")
    p.add_argument("--engine", choices=["auto", "python", "cpu", "gpu"], default="auto",
                   help="auto: native CPU until it stalls (3 s without a lighter logical) or, with --gpu, "
                        "5 s at most; then the GPU (needs --gpu) for the rest of --seconds, else the CPU "
                        "continues; stops early only on --target; Python if nothing native is built")
    p.add_argument("--gpu", default=None, help="CUDA device for --engine auto after the CPU phase, e.g. 0")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", type=int, default=None, help="stop once a logical lighter than this is found")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--no-chain", action="store_true",
                   help="native engines only: fresh eliminations per trial instead of the Markov chain of swaps")
    p.set_defaults(fn=cmd_upper)

    p = sub.add_parser("lower", help="prove d >= W+1 with one engine",
                       description="Prove that no nontrivial logical of weight <= W exists. JSON 'status': "
                                   "proved | found (a logical of weight <= W, see 'witness') | inconclusive | error; "
                                   "'engine_status' keeps the engine's own word.")
    _common_code(p)
    p.add_argument("--bound", type=int, required=True, help="W: prove there is no logical of weight <= W")
    p.add_argument("--engine", choices=["bnb", "sat", "brute"], default="bnb")
    p.add_argument("--side", choices=["P", "X", "Z"], default="P")
    p.add_argument("--symmetry", default="auto", choices=["auto", "nauty", "cyclic", "none"])
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--gpu", default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--checkpoint", action="store_true", help="B&B: checkpoint in the work dir (resumable)")
    p.add_argument("--bnb-cert", action="store_true", help="B&B: write a coverage certificate in the work dir")
    p.add_argument("--gpu-slots", type=int, default=None,
                   help="B&B GPU: kernels in flight per GPU (native default 2; each uses a host thread)")
    p.add_argument("--root-order", choices=["fiedler", "file"], default="fiedler",
                   help="B&B without symmetry: root order (any order is sound; fiedler is faster)")
    p.add_argument("--packing", choices=["greedy", "none", "greedy+s0", "greedy+s0s1", "greedy+s0s1s2"],
                   default="greedy",
                   help="B&B packing bound; 'none' is an ablation (no certificates); greedy+s0* are opt-in "
                        "CPU-only rules (SOUNDNESS.md 5.1); with --bnb-cert they write a coverage certificate version 2 "
                        "(BNB_CERT_SPEC_v2.md) that verify --bnb-checker counts only if a v2 checker (normally hyqec-check >= 2.0.0) accepts it")
    p.add_argument("--solver", default="kissat", choices=["kissat", "cadical"])
    p.add_argument("--cubes", type=int, default=0, help="SAT: split on the first N orbit roots")
    p.add_argument("--split-labels", action="store_true", help="SAT cubes: also split the root's X/Y/Z label")
    p.add_argument("--sat-workers", type=int, default=8)
    p.add_argument("--lrat", action="store_true", help="SAT: also check an LRAT proof (lrat-check)")
    p.add_argument("--workdir", default=None)
    p.add_argument("-o", "--certificate", default=None,
                   help="write a certificate: this run's lower bound + the input file's verified witness")
    p.add_argument("--no-input-witness", action="store_true")
    p.set_defaults(fn=cmd_lower)

    p = sub.add_parser("verify", help="re-check a certificate")
    p.add_argument("certificate")
    p.add_argument("--rerun-bnb", action="store_true", help="re-execute B&B runs and compare node counts")
    p.add_argument("--bnb-checker", default=None,
                   help="independent B&B coverage checker: 'auto' (the companion on PATH: hyqec-check for coverage "
                        "v1, hyqec-check-v2 from hyqec-check >= 2.0.0 for coverage v2), a path to a checker executable or "
                        "source directory (hyqec_check/, hyqec_check_v2/ or the older bnbcheck/), or a command "
                        "template with placeholders {file} {group} {domain} {cap} {instance}; accepted only if it "
                        "exits 0, prints a JSON report with certificate_verified: true and its verdict matches the "
                        "certificate (v2: also the rule set and spec revision); --threads is passed to the "
                        "checker (default min(16, cores)); default $HYQEC_BNB_CHECKER")
    p.add_argument("--lrat", action="store_true")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("wrap-coverage", help="certificate from an existing B&B coverage file (+ input witness)",
                       description="Wrap a coverage file produced outside hyqec-verify (e.g. by the native stabcert-bnb engine directly). "
                                   "Its header must name this code's group. The lower bound counts only after "
                                   "'hyqec-verify verify CERT --bnb-checker CMD' accepts the file.")
    _common_code(p)
    p.add_argument("coverage", help="the coverage file (.gz)")
    p.add_argument("-o", "--certificate", required=True)
    p.set_defaults(fn=cmd_wrap_coverage)

    p = sub.add_parser("tools", help="external SAT solvers: install / list")
    p.add_argument("action", choices=["install", "list"])
    p.add_argument("names", nargs="*")
    p.add_argument("--from-source", default=None, help="build from a local source tree instead of git")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_tools)

    p = sub.add_parser("doctor", help="show available engines and tools")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("build", help="compile the native engines into the cache")
    p.add_argument("targets", nargs="*", help="bnb-cpu bnb-cuda ris-cpu ris-cuda (default: CPU targets)")
    p.add_argument("--cuda-arch", default="native")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_build)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (CodeError, OSError) as e:
        if getattr(args, "json", False):
            print(json.dumps({"status": "rejected", "reason": str(e)}))
        else:
            print(f"error: {e}", file=sys.stderr)
        return EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())
