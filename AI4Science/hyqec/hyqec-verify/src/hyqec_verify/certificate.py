"""Distance certificates: a self-contained JSON file (+ an artifacts directory for proofs).

``stabcert-certificate/1``:
  code        n, k, rank, generators ({"X","Z"} supports), group_sha256 (core.group_hash)
  question    domain: "P" (Pauli weight), "X" or "Z" (CSS per-side, X-/Z-type logicals)
  result      status, lower_bound, upper_bound, d (if exact)
  witness     lightest verified logical ({"X","Z"})
  evidence    list of records; each lower-bound record has "proves" and "lower_bound"
  artifacts   directory (relative to the certificate) holding DRAT proofs

``verify_certificate`` trusts nothing in the file: it rebuilds the code from the
listed generators, re-checks the group hash and k, the witness, every
permutation used for symmetry, regenerates every CNF and re-checks every proof
with the vendored checker. A B&B lower bound counts only if it is re-established:
a run record is re-executed on request (``rerun_bnb``; the run is deterministic and
its node count is compared), and a coverage certificate (docs/BNB_CERT_SPEC.md) is
accepted only through the independent checker (``bnb_checker``). Otherwise the bound
is withheld.
"""
from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

from . import __version__
from .core import CodeError, StabilizerCode, pauli_from_xz, xz_from_pauli

FORMAT = "stabcert-certificate/1"

__all__ = ["FORMAT", "build_certificate", "write_certificate", "verify_certificate"]


def build_certificate(code: StabilizerCode, result, source: dict | None = None) -> dict:
    return {
        "format": FORMAT,
        "stabcert_version": __version__,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code": {"n": code.n, "k": code.k, "rank": code.rank, "name": code.name,
                 "generators": code.to_xz(), "group_sha256": code.group_hash(), "source": source},
        "question": {"domain": result.domain,
                     "definition": "min Pauli weight (Y counts once) over N(S) minus S" if result.domain == "P"
                     else f"min weight over {result.domain}-type operators in N(S) minus S"},
        "result": {"status": result.status, "lower_bound": result.lower, "upper_bound": result.upper,
                   **({"d": result.upper} if result.status == "exact" else {})},
        "witness": xz_from_pauli(result.witness, code.n) if result.witness is not None else None,
        "evidence": _strongest(result.evidence),
        "notes": list(result.notes),
    }


def _strongest(evidence):
    """Per method, keep only the record with the highest lower bound (ladder rungs below it are
    implied) and the lightest witness."""
    best = {}
    for e in evidence:
        key = e.get("method")
        score = -e["weight"] if key == "witness" else e.get("lower_bound", 0)
        if key not in best or score > best[key][0]:
            best[key] = (score, dict(e))
    return [v for _, v in best.values()]


def write_certificate(cert: dict, path) -> Path:
    """Write the JSON and copy every proof artifact next to it (``<stem>.artifacts/``)."""
    path = Path(path)
    art = path.with_name(path.stem + ".artifacts")
    cert = json.loads(json.dumps(cert))
    for i, ev in enumerate(cert["evidence"]):
        src = ev.pop("directory", None)
        ev.pop("artifacts", None)
        if ev.get("method") in ("sat-drat", "sat-drat-cubes") and src:
            dst = art / f"ev{i:03d}"
            dst.mkdir(parents=True, exist_ok=True)
            files = [ev["proof"]["file"]] if ev["method"] == "sat-drat" else \
                [lf["proof"]["file"] for lf in ev["leaves"] if lf.get("proof")]
            for f in files:
                (dst / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(src) / f, dst / f)
            ev["artifact_dir"] = str(dst.relative_to(path.parent))
        if ev.get("method") == "bnb":
            ev.get("engine", {}).pop("binary", None)
            ev.pop("argv", None)
            cov = ev.get("coverage")
            if cov and src:
                dst = art / f"ev{i:03d}"
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(src) / cov["file"], dst / cov["file"])
                ev["artifact_dir"] = str(dst.relative_to(path.parent))
    path.write_text(json.dumps(cert, indent=1) + "\n")
    return path


def coverage_evidence(code: StabilizerCode, coverage_path) -> dict:
    """A B&B evidence record for an existing coverage file (e.g. produced by the native binary directly).

    Only the header is read here; it must name this code's group. The orbit data are parsed from the
    embedded instance so that verify can re-check the permutations; the full proof is checked by the
    independent checker (verify --bnb-checker)."""
    import gzip
    import hashlib
    from .engines.bnb import read_coverage_header
    from .tools import sha256_file
    p = Path(coverage_path)
    h = read_coverage_header(p)
    if h.get("format") not in ("stabcert-bnb-coverage/1", COVERAGE_V2_IDS["format"]):
        raise CodeError("not a stabcert-bnb-coverage/1 or /2 file")
    policy = None
    if h["format"] == COVERAGE_V2_IDS["format"]:
        # a v2 file declares its rule set; the recorded policy is the one that writes it (the checker
        # re-executes with that rule set, and verify binds header, policy and verdict to each other)
        policy = {r: p for p, r in COVERAGE_V2_RULES.items()}.get(h.get("rules"))
        if policy is None or any(h.get(k) != v for k, v in COVERAGE_V2_IDS.items()):
            raise CodeError(f"coverage v2 header: unknown rule set or identifiers ({h.get('rules')!r})")
    if h["group_sha256"] != code.group_hash():
        raise CodeError("coverage file is for a different stabilizer group")
    if hashlib.sha256(h["instance_text"].encode("ascii")).hexdigest() != h["instance_sha256"]:
        raise CodeError("coverage header: embedded instance does not match its hash")
    toks = h["instance_text"].split()
    pos = 2 if toks[0] == "stabcert-bnb-instance" else 0      # header "stabcert-bnb-instance 1"
    n, m, nl = int(toks[pos]), int(toks[pos + 1]), int(toks[pos + 2])
    pos += 3
    for _ in range(m + nl):
        r = int(toks[pos])
        pos += 1 + 2 * r
    orbit_of = [int(t) for t in toks[pos:pos + n]]
    pos += n
    no = int(toks[pos])
    reps = [int(t) for t in toks[pos + 1:pos + 1 + no]]
    if n != code.n:
        raise CodeError("coverage instance has a different qubit count")
    with open(p, "rb") as fh:
        magic = fh.read(2)                       # transport by magic; never load the artifact
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(p, "rb") as f:
        raw = hashlib.sha256()
        size = 0
        for b in iter(lambda: f.read(1 << 22), b""):
            raw.update(b)
            size += len(b)
    return {"method": "bnb", "source": "external coverage file",
            "params": {"cap": h["cap"], "domain": h["domain"], "split": h["split"],
                       "orbit_roots": {"generators": h["permutations"], "reps": reps, "orbit_of": orbit_of},
                       **({"bound": policy} if policy else {})},
            "instance_sha256": h["instance_sha256"], "lower_bound": h["cap"] + 1,
            "proves": f"no {'nontrivial' if h['domain'] == 'P' else h['domain'] + '-type nontrivial'} logical "
                      f"of Pauli weight <= {h['cap']}",
            "result": {"found": -1}, "provenance": h.get("provenance"),
            "coverage": {"file": p.name, "sha256": raw.hexdigest(), "bytes": size,
                         "gz_sha256": sha256_file(p), "gz_bytes": p.stat().st_size},
            "directory": str(p.parent)}


def _code_from_cert(cert) -> StabilizerCode:
    c = cert["code"]
    code = StabilizerCode.from_xz(c["generators"], c["n"])
    if code.group_hash() != c["group_sha256"]:
        raise CodeError("group hash does not match the listed generators")
    for key, val in (("k", code.k), ("rank", code.rank)):
        if type(c.get(key)) is not int or c[key] != val:       # JSON integers only (not 2.0, not true)
            raise CodeError(f"code.{key} is missing, not an integer, or does not match the listed generators")
    return code


def _is_int(x) -> bool:
    return type(x) is int


def _xz_ok(w) -> bool:
    return (isinstance(w, dict) and set(w) == {"X", "Z"}
            and all(isinstance(w[k], list) and all(_is_int(q) for q in w[k]) for k in ("X", "Z")))


# Exact params keys per evidence method: what the writers emit (engines/bnb.py run_bnb; coverage_evidence for
# wrapped coverage files; engines/sat.py encode for sat-drat and sat-drat-cubes). witness and brute evidence
# carries no params. Any other key is rejected, so no field can be added to a record without a parser change.
_SAT_PARAMS = frozenset({"encoding", "bound", "domain", "xor_chunk", "cardinality", "orbit_roots"})
EVIDENCE_PARAMS = {
    "bnb": frozenset({"cap", "domain", "orbit_roots", "split", "stop_first", "bound"}),
    "sat-drat": _SAT_PARAMS,
    "sat-drat-cubes": _SAT_PARAMS,
    "brute": None,
    "witness": None,
}


def load_certificate_json(path) -> dict:
    """Parse a certificate strictly: valid JSON, no duplicate keys, the expected structure and JSON types.
    Anything else is a CodeError (CLI: status 'rejected', exit 1)."""
    def no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise CodeError(f"duplicate JSON keys: {sorted({k for k in keys if keys.count(k) > 1})}")
        return dict(pairs)
    try:
        cert = json.loads(Path(path).read_text(), object_pairs_hook=no_dupes)
    except json.JSONDecodeError as e:
        raise CodeError(f"invalid JSON: {e}") from None
    if not isinstance(cert, dict) or cert.get("format") != FORMAT:
        raise CodeError(f"not a {FORMAT} file")
    c = cert.get("code")
    if not isinstance(c, dict) or not all(_is_int(c.get(k)) for k in ("n", "k", "rank")) \
            or not isinstance(c.get("generators"), list) or not all(_xz_ok(g) for g in c["generators"]) \
            or not isinstance(c.get("group_sha256"), str):
        raise CodeError("certificate 'code' is malformed (n, k, rank must be integers; generators {X: [...], Z: [...]})")
    q = cert.get("question")
    if not isinstance(q, dict) or q.get("domain") not in ("P", "X", "Z"):
        raise CodeError("certificate 'question.domain' must be P, X or Z")
    if cert.get("witness") is not None and not _xz_ok(cert["witness"]):
        raise CodeError("certificate 'witness' must be {X: [...], Z: [...]} with integer qubits")
    ev = cert.get("evidence")
    if not isinstance(ev, list) or not all(isinstance(e, dict) and isinstance(e.get("method"), str) for e in ev):
        raise CodeError("certificate 'evidence' must be a list of objects with a 'method'")
    for e in ev:
        if e["method"] == "witness" and not _xz_ok(e.get("witness")):
            raise CodeError("witness evidence must carry {X: [...], Z: [...]}")
        if e["method"].startswith(("sat", "bnb")) and not isinstance(e.get("params"), dict):
            raise CodeError(f"{e['method']} evidence needs a 'params' object")
        if e["method"] in EVIDENCE_PARAMS:
            allowed = EVIDENCE_PARAMS[e["method"]]
            if allowed is None and "params" in e:
                raise CodeError(f"{e['method']} evidence carries no 'params'")
            unknown = sorted(set(e.get("params") or {}) - (allowed or set()))
            if unknown:
                raise CodeError(f"{e['method']} evidence params has unknown keys {unknown}")
        ints = [("lower_bound", e), ("weight", e)]
        if e["method"].startswith("sat"):
            ints += [("bound", e["params"]), ("xor_chunk", e["params"])]
        elif e["method"] == "bnb":
            ints += [("cap", e["params"]), ("split", e["params"])]
            if not isinstance(e["params"].get("bound", "greedy"), str):
                raise CodeError("bnb evidence field 'bound' must be a JSON string")
        for k, d in ints:
            if d.get(k) is not None and not _is_int(d[k]):
                raise CodeError(f"{e['method']} evidence field '{k}' must be a JSON integer")
    r = cert.get("result")
    if not isinstance(r, dict) or r.get("status") not in ("exact", "bounds", "inconsistent") \
            or not _is_int(r.get("lower_bound")) \
            or not (r.get("upper_bound") is None or _is_int(r.get("upper_bound"))) \
            or ("d" in r and not _is_int(r["d"])):
        raise CodeError("certificate 'result' is malformed (status exact|bounds|inconsistent, integer bounds)")
    return cert


def _in_domain(v: int, n: int, domain: str) -> bool:
    return domain == "P" or (domain == "X" and not v >> n) or (domain == "Z" and not v & ((1 << n) - 1))


def verify_certificate(path, *, rerun_bnb: bool = False, lrat: bool = False, threads: int | None = None,
                       bnb_checker: str | None = None) -> dict:
    """Re-check a certificate from scratch.

    Every bound is DERIVED from re-checked facts, never read from an editable field: an upper
    bound is the weight of a witness that the core re-verifies (inside the question's domain);
    a lower bound is cap+1 of a proof that is re-checked here (SAT/DRAT against a CNF regenerated
    from the code), re-executed (brute force; B&B with --rerun-bnb, whose instance is rebuilt and
    must hash to the recorded value), or accepted by the independent checker (coverage file whose
    header is bound to this code/question). Recorded 'lower_bound' fields must agree with the
    derived value or the item fails. The claimed result (status, d, lower, upper) must equal the
    verified one, else claims_reproduced is false."""
    from .distance import require_side_ok
    path = Path(path)
    cert = load_certificate_json(path)
    code = _code_from_cert(cert)
    domain = cert["question"]["domain"]
    checker_spec = bnb_checker.strip() if isinstance(bnb_checker, str) and bnb_checker.strip() else None
    wide = False
    if checker_spec:
        from .engines.bnb import generator_shape
        W, C = generator_shape(code)
        wide = W > 32 or C > 64
    require_side_ok(code, domain)
    code.require_logicals()
    n = code.n
    checks = []
    upper = None

    def take_witness(xz, item):
        nonlocal upper
        v = pauli_from_xz(xz["X"], xz["Z"], n, "witness")
        prob = code.witness_problem(v)
        if prob is None and not _in_domain(v, n, domain):
            prob = f"witness is not in the question's domain {domain}"
        item["weight"] = code.weight(v)
        item["ok"] = prob is None
        if prob:
            item["problem"] = prob
        else:
            upper = item["weight"] if upper is None else min(upper, item["weight"])
        return item["ok"]

    if cert.get("witness") is not None:
        checks.append({"item": "witness"})
        take_witness(cert["witness"], checks[-1])
    lower = 1
    recorded_bnb = 0

    def claim_matches(ev, derived, item):
        rec = ev.get("lower_bound")
        if rec is not None and rec != derived:
            item["ok"] = False
            item["problem"] = f"recorded lower_bound {rec} != {derived} derived from the re-checked proof"
            return False
        return True

    for i, ev in enumerate(cert["evidence"]):
        m = ev.get("method")
        item = {"item": f"evidence[{i}]", "method": m}
        try:
            if m == "witness":
                take_witness(ev["witness"], item)
                if item["ok"] and ev.get("weight") not in (None, item["weight"]):
                    item["ok"] = False
                    item["problem"] = f"recorded weight {ev.get('weight')} != {item['weight']}"
            elif m in ("sat-drat", "sat-drat-cubes"):
                from .engines import sat
                if ev["params"]["domain"] != domain:
                    raise CodeError("evidence for a different question")
                r = sat.check_record(code, ev, path.parent / ev["artifact_dir"], lrat=lrat)
                item["ok"] = r["verified"]
                item["proves"] = r["proves"]
                if r["verified"] and claim_matches(ev, r["bound"] + 1, item):
                    lower = max(lower, r["bound"] + 1)
            elif m == "brute":
                from .engines.brute import brute_force
                W = ev["lower_bound"] - 1
                if type(W) is not int or W < 0:
                    raise CodeError("invalid brute-force bound")
                r = brute_force(code, W, domain=domain, work_limit=10 ** 9)
                item["ok"] = r.found is None
                if not item["ok"]:
                    item["problem"] = f"brute force found a logical of weight {r.found} <= {W}"
                else:
                    lower = max(lower, W + 1)
            elif m == "bnb":
                from .symmetry import Orbits
                o = ev["params"]["orbit_roots"]
                orb = Orbits(o["orbit_of"], o["reps"], o["generators"])
                _check_orbit_numbering(code, orb)          # re-verifies every permutation too
                if ev["params"]["domain"] != domain:
                    raise CodeError("evidence for a different question")
                cap = ev["params"]["cap"]
                if type(cap) is not int or cap < 0:
                    raise CodeError("invalid cap")
                cov = ev.get("coverage")
                # The evidence level is derived from method + params.bound, never read from a field.
                from .engines.bnb import COUNT_ONLY_POLICIES
                policy = ev["params"].get("bound", "greedy")
                if policy not in ("greedy", "none") + COUNT_ONLY_POLICIES:
                    raise CodeError(f"unknown B&B policy {policy!r}")
                item["policy"] = policy
                cov_version = _coverage_version(path.parent / ev.get("artifact_dir", "."), cov) if cov else None
                if cov:
                    item["coverage_version"] = cov_version
                if cov and cov_version == 1 and policy != "greedy":
                    # coverage v1 admits only the section-5 rules with greedy packing (the only policy that can
                    # emit it); any other recorded policy on a coverage attachment is an inconsistent record
                    item["ok"] = False
                    item["problem"] = f"coverage evidence with policy {policy}: coverage v1 admits only greedy"
                elif cov and cov_version == 2 and policy not in COVERAGE_V2_RULES:
                    item["ok"] = False
                    item["problem"] = f"coverage v2 evidence with policy {policy}: no v2 rule set"
                elif cov and checker_spec:
                    r = {"ok": False, "problem": "independent checker unavailable"}
                    try:
                        cmd = resolve_checker(checker_spec, threads, wide=wide, version=cov_version)
                    except CodeError as e:               # missing / old checker: never counts
                        cmd, r = None, {"ok": False, "problem": f"independent checker unavailable: {e}"}
                    if cmd is not None:
                        r = _check_coverage(code, ev, path.parent / ev.get("artifact_dir", "."), domain, cmd,
                                            version=cov_version, policy=policy)
                    item.update(r)
                    if item["ok"] and claim_matches(ev, r["header_cap"] + 1, item):
                        lower = max(lower, r["header_cap"] + 1)
                        if cov_version == 2:
                            item["evidence_level"] = (
                                f"lower bound by bnb coverage certificate v2 (rules {r['rules']}, spec "
                                f"{r['spec_revision']}), policy {policy}, independently checked by "
                                f"{r['checker_name']} (reported version {r['checker_version']})")
                        else:
                            item["evidence_level"] = (f"lower bound by bnb coverage certificate, policy {policy}, "
                                                      "independently checked")
                elif rerun_bnb and ev.get("source") == "external coverage file":
                    item["ok"] = None
                    item["problem"] = ("external coverage file: its instance was not written by hyqec-verify, so "
                                       "--rerun-bnb cannot reproduce it; use --bnb-checker")
                    recorded_bnb = max(recorded_bnb, cap + 1)
                elif rerun_bnb:
                    import tempfile
                    from .engines.bnb import instance_text, run_bnb
                    import hashlib
                    inst = hashlib.sha256(instance_text(code, orb, domain).encode()).hexdigest()
                    if inst != ev.get("instance_sha256"):
                        raise CodeError("recorded instance_sha256 differs from the instance rebuilt from the "
                                        "code and the recorded orbits")
                    with tempfile.TemporaryDirectory() as t:
                        # re-run with exactly the recorded packing policy
                        out = run_bnb(code, cap, t, domain=domain, orbits=orb, threads=threads,
                                      split=ev["params"].get("split", -200000), bound=policy)
                    item["rerun_status"] = out.status
                    item["rerun_nodes"] = out.record.get("result", {}).get("nodes")
                    if out.record.get("instance_sha256") != inst:
                        raise CodeError("re-run used a different instance")
                    if out.status == "none":
                        derived = cap + 1                  # no logical of weight <= cap in the domain
                        rec_nodes = (ev.get("result") or {}).get("nodes")
                        item["ok"] = rec_nodes in (None, item["rerun_nodes"])
                        if not item["ok"]:
                            item["problem"] = f"node count {item['rerun_nodes']} != recorded {rec_nodes}"
                    elif out.status == "found":
                        derived = out.found                # min mode: the minimum, witness re-verified
                        item["ok"] = True
                        take_witness(xz_from_pauli(out.witness, n), {})
                    else:
                        derived = None
                        item["ok"] = False
                        item["problem"] = f"re-run did not complete: {out.status}"
                    if item["ok"] and claim_matches(ev, derived, item):
                        lower = max(lower, derived)
                        item["evidence_level"] = (f"lower bound by bnb rerun, policy {policy}, same-prover "
                                                  "re-execution, not independently checked")
                else:
                    item["ok"] = None            # recorded, not re-executed
                    item["evidence_level"] = (f"bnb coverage certificate, policy {policy}, not checked "
                                              "(no --bnb-checker)" if cov else
                                              f"bnb run record, policy {policy}, not re-executed, "
                                              "not independently checked")
                    # what accepting the record would give: its recorded minimum if it found one, else cap+1
                    rf = (ev.get("result") or {}).get("found", -1)
                    recorded_bnb = max(recorded_bnb, rf if _is_int(rf) and 0 < rf <= cap else cap + 1)
            else:
                item["ok"] = False
                item["problem"] = "unknown evidence method"
        except Exception as e:                      # fail closed: any malformed artifact or field
            item["ok"] = False
            item["problem"] = f"{type(e).__name__}: {e}"
        checks.append(item)
    res = cert.get("result") or {}
    status = "inconsistent" if upper is not None and lower > upper else \
        ("exact" if upper is not None and lower == upper else "bounds")
    verified = {"status": status, "lower_bound": lower, "upper_bound": upper}
    if status == "exact":
        verified["d"] = upper
    def same(a, b):
        # exact type match: 2.0 or true must not pass for 2 or 1
        return type(a) is type(b) and a == b
    mismatch = [k for k in ("status", "lower_bound", "upper_bound", "d") if not same(res.get(k), verified.get(k))]
    if mismatch:
        checks.append({"item": "result", "ok": False,
                       "problem": f"claimed {', '.join(mismatch)} not reproduced by the verified evidence"})
    unverified_bnb = [c for c in checks if c.get("method") == "bnb" and c.get("ok") is None]
    return {"certificate": str(path), "n": n, "k": code.k, "domain": domain,
            "verified_lower": lower, "verified_upper": upper, "status": status,
            "claimed": res, "claims_reproduced": not mismatch and not any(c.get("ok") is False for c in checks),
            "claim_mismatch": mismatch,
            "bnb_records_not_rerun": len(unverified_bnb),
            # lower bound if the recorded (not re-executed) B&B runs are accepted
            "lower_with_recorded_bnb": max(lower, recorded_bnb),
            "failed": [c for c in checks if c.get("ok") is False], "checks": checks}


CHECKER_NAME = "hyqec-check"
WIDE_CHECKER_NAME = "hyqec-check-wide"       # coverage of generators heavier than 32 or columns above 64
V2_CHECKER_NAME = "hyqec-check-v2"           # coverage v2 (hyqec-check 2.0.0), every rule set and both limits
# coverage v2 (BNB_CERT_SPEC_v2 2.0-rc1): header ids, and the rule set each B&B policy must declare
COVERAGE_V2_IDS = {"format": "stabcert-bnb-coverage/2", "search": "irreducible-pauli-dfs/2",
                   "trace": "poly64-state-rules/2", "rules_version": 1}
COVERAGE_V2_RULES = {"greedy": "greedy", "greedy+s0": "s0", "greedy+s0s1": "s0s1", "greedy+s0s1s2": "s0s1s2"}
V2_SPEC_REVISION = "2.0-rc1"
_CHECKER_ARGS = ("verify {file} --expected-group {group} --expected-domain {domain} --expected-cap {cap} "
                 "--expected-instance {instance}")


def resolve_checker(spec: str | None, threads: int | None = None, wide: bool = False,
                    version: int | None = 1) -> str | None:
    """Turn a --bnb-checker value into a command template with {file} etc. for a coverage file of
    container ``version`` (1 or 2; None only validates the value).

    * a template containing ``{file}`` is used as given (for every version);
    * ``auto``, ``hyqec-check`` or ``hyqec-check-v2``: the companion checker on PATH: for version 1
      ``hyqec-check`` (or ``hyqec-check-wide`` when ``wide``: generators heavier than 32 or column degree
      above 64), for version 2 ``hyqec-check-v2`` (hyqec-check >= 2.0.0);
    * a path to an executable (``hyqec-check``, or a pre-rename ``bnbcheck`` script);
    * a path to a checker source directory holding ``hyqec_check/`` or the pre-rename ``bnbcheck/``
      package, or ``hyqec_check_v2/`` for version 2 (run as ``python -m <package>`` with that directory on
      the module path).
    The checker is never run implicitly: without a value, coverage bounds are withheld."""
    import shlex
    import sys
    if spec is None or not spec.strip():
        return None
    if "{file}" in spec:
        return spec
    import os
    # the companion checker requires --threads; default as for B&B runs: min(16, cores)
    tail = _CHECKER_ARGS + f" --threads {int(threads) if threads else min(16, os.cpu_count() or 1)}"
    if spec in ("auto", CHECKER_NAME, V2_CHECKER_NAME):
        if version is None:
            return None
        name = V2_CHECKER_NAME if version == 2 else WIDE_CHECKER_NAME if wide else CHECKER_NAME
        exe = shutil.which(name)
        if exe is None:
            raise CodeError(f"--bnb-checker {spec}: {name} is not on PATH (install the companion "
                            "package hyqec-check, or pass a command template or a path)")
        return f"{shlex.quote(exe)} {tail}"
    p = Path(spec).expanduser()
    if p.is_dir():
        if version is None:
            return None
        pkgs = (("hyqec_check_v2",) if version == 2 else
                ("hyqec_check_wide", "bnbcheck_wide") if wide else ("hyqec_check", "bnbcheck"))
        for pkg in pkgs:
            if (p / pkg).is_dir():
                boot = (f"import runpy,sys; sys.path.insert(0, {str(p.resolve())!r}); "
                        f"runpy.run_module({pkg!r}, run_name='__main__', alter_sys=True)")
                return f"{shlex.quote(sys.executable)} -c {shlex.quote(boot)} {tail}"
        raise CodeError(f"--bnb-checker {spec}: directory holds no checker package for coverage v{version} "
                        f"({' or '.join(p + '/' for p in pkgs)})")
    exe = str(p.resolve()) if p.is_file() else shutil.which(spec)        # keep ./relative paths runnable
    if exe is None:
        raise CodeError(f"--bnb-checker {spec}: not a command template, executable or checker directory")
    return f"{shlex.quote(exe)} {tail}"


def _coverage_version(base, cov) -> int:
    """Container version of a coverage attachment, from its header's format id (1 or 2; else CodeError)."""
    from .engines.bnb import read_coverage_header
    fmt = read_coverage_header(Path(base) / cov["file"]).get("format")
    v = {"stabcert-bnb-coverage/1": 1, "stabcert-bnb-coverage/2": 2}.get(fmt)
    if v is None:
        raise CodeError(f"unknown coverage format {fmt!r}")
    return v


def _check_coverage(code, ev, base, domain, checker_cmd, version: int = 1, policy: str = "greedy") -> dict:
    """Bind a B&B coverage file to this certificate's question, then run the independent
    checker (command template with {file}) and bind its verdict to the same question.

    Accepted only if the checker exits 0 and reports certificate_verified true with status accepted for
    the complete file, and every question field it reports (group, instance, domain, cap; for v2 also
    the rule set, rules_version and spec revision, which must be present) equals the certificate's."""
    import shlex
    import subprocess
    from .engines.bnb import read_coverage_header
    from .tools import sha256_file
    cov = ev["coverage"]
    f = Path(base) / cov["file"]
    out = {"coverage_file": str(f)}
    if sha256_file(f) != cov["gz_sha256"]:
        return {**out, "ok": False, "problem": "coverage file hash mismatch"}
    h = read_coverage_header(f)
    want = {"group_sha256": code.group_hash(), "domain": domain, "cap": ev["params"]["cap"],
            "instance_sha256": ev["instance_sha256"],
            # the resolved split (the engine's done line) if recorded, else the requested one
            "split": (ev.get("result") or {}).get("split", ev["params"].get("split"))}
    if version == 2:
        # the rule set is the one the recorded policy implies; never read a policy from the header alone
        want.update(COVERAGE_V2_IDS, rules=COVERAGE_V2_RULES[policy])
    bad = [k for k, v in want.items() if h.get(k) != v]
    if bad:
        return {**out, "ok": False, "problem": f"coverage header does not match the question: {bad}"}
    out["header_cap"] = h["cap"]
    subst = {"{file}": str(f), "{group}": want["group_sha256"], "{domain}": domain, "{cap}": str(want["cap"]),
             "{instance}": want["instance_sha256"]}

    def fill(a):
        for k, v in subst.items():
            a = a.replace(k, v)
        return a
    cmd = [fill(a) for a in shlex.split(checker_cmd)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:                     # checker missing / not executable: never a success
        return {**out, "ok": False, "problem": f"independent checker could not be run: {e}"}
    # accepted only with exit 0 AND an explicit JSON report that the whole certificate was verified
    # (a partial replay, or a checker that prints nothing, is not an acceptance)
    ok, rep = False, None
    try:
        rep = json.loads(p.stdout.strip().splitlines()[-1])
        ok = p.returncode == 0 and isinstance(rep, dict) and rep.get("certificate_verified") is True
    except (json.JSONDecodeError, IndexError):
        ok = False
    out.update(checker=" ".join(cmd), checker_returncode=p.returncode, checker_output=(p.stdout + p.stderr)[-2000:])
    if not ok:
        return {**out, "ok": False, "problem": "independent coverage checker did not accept the complete file"}
    # the verdict must be about this question (an older checker may omit fields; v2 fields are required)
    claim = {"group_sha256": want["group_sha256"], "instance_sha256": want["instance_sha256"], "domain": domain,
             "cap": want["cap"], "status": "accepted", "selection": "all"}
    required = set()
    if version == 2:
        claim.update(rules=want["rules"], rules_version=COVERAGE_V2_IDS["rules_version"], spec_revision=V2_SPEC_REVISION)
        required = set(claim)
    # exact JSON types, as in the strict certificate parser (True == 1 and 3.0 == 3 do not count)
    same = lambda a, b: type(a) is type(b) and a == b
    wrong = sorted(k for k, v in claim.items() if (k in rep or k in required) and not same(rep.get(k), v))
    if wrong:
        return {**out, "ok": False, "problem": f"checker verdict does not match the certificate claim: {wrong}"}
    ver = rep.get("checker_version")
    if version == 2 and not _version_at_least(ver, (2, 0, 0)):
        # reported metadata, not authentication: the label repeats it, and v2 needs a v2-capable checker
        return {**out, "ok": False, "problem": f"checker_version {ver!r}: coverage v2 needs a checker reporting "
                                               "a version string >= 2.0.0"}
    out.update(ok=True, checker_version=ver, checker_name=_checker_name(cmd), spec_revision=rep.get("spec_revision"),
               rules=rep.get("rules", "greedy" if version == 1 else None))
    return out


def _version_at_least(v, least) -> bool:
    """True iff v is a version string 'X.Y.Z...' (numeric leading components) that is >= least."""
    import re
    if not isinstance(v, str) or not re.fullmatch(r"\d+(\.\d+)*", v):
        return False
    parts = tuple(int(x) for x in v.split("."))
    return parts + (0,) * (len(least) - len(parts)) >= least


def _checker_name(cmd) -> str:
    """The configured checker's name for the evidence label: the module of a source-directory checker
    (``python -c ... run_module('pkg')``), else the executable's file name."""
    import os
    import re
    if len(cmd) > 2 and cmd[1] == "-c":
        m = re.search(r"run_module\('([\w.]+)'", cmd[2])
        if m:
            return m.group(1)
    return os.path.basename(cmd[0]) if cmd else "the configured checker"


def _check_orbit_numbering(code, orb):
    from .symmetry import orbits_from
    ref = orbits_from(code, orb.generators)
    groups = lambda of: sorted(sorted(q for q in range(len(of)) if of[q] == o) for o in set(of))
    if groups(ref.orbit_of) != groups(orb.orbit_of) or any(orb.orbit_of[r] != i for i, r in enumerate(orb.reps)):
        raise CodeError("recorded orbits are not the orbits of the verified group")
