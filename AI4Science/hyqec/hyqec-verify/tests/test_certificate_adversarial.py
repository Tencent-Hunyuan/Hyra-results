"""Forged / edited certificates must not verify (the independent reviewer's reproducers, plus field-by-field
edits). Every bound must be derived from re-checked facts, never read from an editable field."""
import copy
import gzip
import json

import pytest

from hyqec_verify.certificate import build_certificate, verify_certificate, write_certificate
from hyqec_verify.core import CodeError, StabilizerCode
from hyqec_verify.distance import DistanceResult
from hyqec_verify.engines import sat


def _cert(code, domain, lower, upper, witness, evidence):
    return build_certificate(code, DistanceResult("exact" if upper == lower else "bounds", domain, lower, upper,
                                                  witness, evidence))


def _verify(tmp_path, doc, name="c", **kw):
    p = tmp_path / f"{name}.json"
    write_certificate(doc, p)
    return verify_certificate(p, **kw)


def _native():
    try:
        from hyqec_verify.engines.bnb import locate_binary
        locate_binary(StabilizerCode(4, [15, 15 << 4]))
        return True
    except Exception:
        return False


C4 = StabilizerCode(4, [15, 15 << 4])            # [[4,2,2]]: XXXX, ZZZZ


@pytest.fixture
def bnb_good(tmp_path):
    if not _native():
        pytest.skip("native B&B unavailable")
    from hyqec_verify.engines import bnb
    o = bnb.run_bnb(C4, 1, tmp_path / "run", threads=2, root_order="file")
    assert o.status == "none"
    ev = {**o.record, "lower_bound": 2}
    return _cert(C4, "P", 2, 2, 3, [ev])          # witness XXII (int 3)


def test_bnb_valid_rerun(tmp_path, bnb_good):
    v = _verify(tmp_path, bnb_good, rerun_bnb=True)
    assert v["status"] == "exact" and v["verified_lower"] == 2 and v["claims_reproduced"]


def test_bnb_inflated_lower_bound_rejected(tmp_path, bnb_good):
    """Review finding #1: rerun at cap 1 proves d >= 2 only; an edited lower_bound=4 must not count."""
    bad = copy.deepcopy(bnb_good)
    bad["evidence"][0]["lower_bound"] = 4
    bad["result"] = {"status": "exact", "lower_bound": 4, "upper_bound": 4, "d": 4}
    bad["witness"] = {"X": [0, 3], "Z": [1, 2]}             # a weight-4 logical
    v = _verify(tmp_path, bad, rerun_bnb=True)
    assert v["verified_lower"] < 4 and not v["claims_reproduced"] and v["failed"]


def test_bnb_wrong_instance_hash_rejected(tmp_path, bnb_good):
    """Review finding #3"""
    bad = copy.deepcopy(bnb_good)
    bad["evidence"][0]["instance_sha256"] = "0" * 64
    v = _verify(tmp_path, bad, rerun_bnb=True)
    assert v["verified_lower"] == 1 and v["failed"] and not v["claims_reproduced"]


def test_bnb_edited_cap_cannot_inflate(tmp_path, bnb_good):
    """Editing the cap only changes what is re-executed: at cap 3 the rerun finds the true d=2."""
    bad = copy.deepcopy(bnb_good)
    bad["evidence"][0]["params"]["cap"] = 3
    v = _verify(tmp_path, bad, rerun_bnb=True)
    assert v["verified_lower"] == 2 and v["verified_upper"] == 2
    bad["evidence"][0]["lower_bound"] = 4                     # and a claimed 4 is refused
    v = _verify(tmp_path, bad, "b2", rerun_bnb=True)
    assert v["verified_lower"] <= 2 and v["failed"]


def test_witness_outside_domain_rejected(tmp_path):
    """Review finding #2: S=<ZZI,IZZ>; question X (d_X=3); a Z-type witness must not give d_X <= 1."""
    rep = StabilizerCode.css([], [[0, 1], [1, 2]], 3)
    doc = _cert(rep, "X", 1, 1, None, [{"method": "witness", "weight": 1, "witness": {"X": [], "Z": [0]}}])
    v = _verify(tmp_path, doc)
    assert v["verified_upper"] is None and v["failed"] and not v["claims_reproduced"]


def test_witness_recorded_weight_must_match(tmp_path):
    doc = _cert(C4, "P", 1, 2, 3, [{"method": "witness", "weight": 1, "witness": {"X": [0, 1], "Z": []}}])
    v = _verify(tmp_path, doc)
    assert any("recorded weight" in (f.get("problem") or "") for f in v["failed"])


def test_sat_zero_literal_cube_rejected(tmp_path):
    """Review finding: a cube literal 0 writes a DIMACS clause terminator (empty clause => trivial UNSAT)."""
    free = StabilizerCode(3, [])
    enc = sat.encode(free, 1)
    d = tmp_path / "cube"
    d.mkdir()
    with gzip.open(d / "proof.drat.gz", "wb") as f:
        f.write(b"")
    pf = {"file": "proof.drat.gz", "sha256": sat.sha256_bytes(b""), "bytes": 0}
    with pytest.raises(CodeError):
        enc.formula.dimacs([[0]])
    h = "0" * 64
    rec = {"method": "sat-drat-cubes", "params": enc.params, "cubes": [[0]],
           "leaves": [{"kind": "leaf", "index": 0, "cube": [0], "proof": pf, "cnf_sha256": h},
                      {"kind": "coverage", "index": 1, "cube": None, "proof": pf, "cnf_sha256": h}],
           "directory": str(d), "lower_bound": 2}
    v = _verify(tmp_path, _cert(free, "P", 2, 2, 3, [rec]))
    assert v["verified_lower"] == 1 and v["failed"]


@pytest.mark.parametrize("field,value", [("d", 999), ("lower_bound", 999), ("upper_bound", 999),
                                         ("status", "exact")])
def test_edited_result_fields_not_reproduced(tmp_path, field, value):
    """Review finding #4: the claimed result must equal the verified one."""
    free = StabilizerCode(3, [])
    doc = _cert(free, "P", 1, 1, 1, [])
    assert _verify(tmp_path, doc, "ok")["claims_reproduced"]
    bad = copy.deepcopy(doc)
    if field == "status":
        bad["result"] = {"status": "exact", "lower_bound": 1, "upper_bound": 2, "d": 2}
    else:
        bad["result"][field] = value
    v = _verify(tmp_path, bad, "bad")
    assert not v["claims_reproduced"] and v["claim_mismatch"]


def test_cli_exit_nonzero_on_claim_mismatch(tmp_path, capsys):
    from hyqec_verify.cli import main
    free = StabilizerCode(3, [])
    doc = _cert(free, "P", 1, 1, 1, [])
    doc["result"]["d"] = 999
    p = tmp_path / "x.json"
    write_certificate(doc, p)
    assert main(["verify", str(p), "--json"]) != 0
    assert json.loads(capsys.readouterr().out)["claims_reproduced"] is False


def test_brute_bound_is_reexecuted(tmp_path):
    doc = _cert(C4, "P", 3, 3, None, [{"method": "brute", "lower_bound": 3}])   # false: d=2
    v = _verify(tmp_path, doc)
    assert v["verified_lower"] == 1 and v["failed"]


@pytest.mark.parametrize("field,value", [("k", 3.0), ("k", True), ("rank", 0.0)])
def test_code_integer_fields_must_be_ints(tmp_path, field, value):
    """Review finding: k/rank as equal floats or bools must be rejected."""
    free = StabilizerCode(3, [])
    doc = _cert(free, "P", 1, 1, 1, [])
    doc["code"][field] = value
    p = tmp_path / "t.json"
    write_certificate(doc, p)
    with pytest.raises(CodeError):
        verify_certificate(p)


@pytest.mark.parametrize("field,value", [("d", 1.0), ("lower_bound", True), ("upper_bound", 1.0)])
def test_result_fields_must_be_ints(tmp_path, field, value):
    free = StabilizerCode(3, [])
    doc = _cert(free, "P", 1, 1, 1, [])
    doc["result"][field] = value
    p = tmp_path / "r.json"
    write_certificate(doc, p)
    with pytest.raises(CodeError):          # malformed result: rejected outright
        verify_certificate(p)


def test_rerun_uses_recorded_packing_policy(tmp_path):
    """Review finding: a run made with --packing none must be re-executed with the same policy (node counts)."""
    if not _native():
        pytest.skip("native B&B unavailable")
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from known_codes import toric
    from hyqec_verify.engines import bnb
    c = toric(4)
    o = bnb.run_bnb(c, 3, tmp_path / "run", threads=2, bound="none")
    assert o.status == "none" and o.record["params"]["bound"] == "none"
    doc = _cert(c, "P", 4, 4, None, [{**o.record, "lower_bound": 4}])
    doc["witness"] = None
    doc["result"] = {"status": "bounds", "lower_bound": 4, "upper_bound": None}
    v = _verify(tmp_path, doc, rerun_bnb=True)
    assert v["verified_lower"] == 4 and not v["failed"], v["checks"]


@pytest.mark.parametrize("orbit_roots", [
    {"orbit_of": [-1, 0, 1], "reps": [1, 2], "generators": []},     # Review finding: orbit -1 gets no root
    {"orbit_of": [0, 1, 2], "reps": [0, 1], "generators": []},      # last orbit has no root
    {"orbit_of": [0, 1, 2], "reps": [], "generators": []},
])
def test_orbit_roots_must_cover_every_orbit(orbit_roots):
    """An orbit without a root selector is silently excluded from the search: for S=<IZI,IIZ>
    (d=1, XII logical) omitting qubit 0's orbit made "no logical of weight 1" UNSAT."""
    code = StabilizerCode(3, [2 << 3, 4 << 3])
    params = sat.encode(code, 1).params
    with pytest.raises(CodeError):
        sat.encode_from_params(code, {**params, "orbit_roots": orbit_roots})


@pytest.mark.parametrize("field,value", [("lower_bound", 2.0), ("lower_bound", True), ("weight", True),
                                         ("weight", 1.0)])
def test_evidence_integer_fields_must_be_ints(tmp_path, field, value):
    """Review finding: recorded evidence integers given as equal floats or bools must be rejected."""
    free = StabilizerCode(3, [])
    doc = _cert(free, "P", 1, 1, 1, [{"method": "witness", "weight": 1, "witness": {"X": [0], "Z": []}}])
    assert _verify(tmp_path, doc, "ok")["claims_reproduced"]
    doc["evidence"][0][field] = value
    with pytest.raises(CodeError):
        _verify(tmp_path, doc, "bad")


@pytest.mark.parametrize("method,key", [("sat-drat", "bound"), ("sat-drat", "xor_chunk"), ("bnb", "cap"),
                                        ("bnb", "split")])
def test_evidence_param_integers_must_be_ints(tmp_path, method, key):
    free = StabilizerCode(3, [])
    doc = _cert(free, "P", 1, 1, 1, [{"method": method, "params": {key: 1.0}}])
    with pytest.raises(CodeError):
        _verify(tmp_path, doc)


_PARAMS = {
    "bnb": {"cap": 1, "domain": "P", "orbit_roots": {"orbit_of": [0, 1, 2, 3], "reps": [0, 1, 2, 3], "generators": []},
            "split": 1, "stop_first": False, "bound": "greedy"},
    "sat-drat": {"encoding": "x", "bound": 1, "domain": "P", "xor_chunk": 4, "cardinality": "seqcounter",
                 "orbit_roots": None},
}
_PARAMS["sat-drat-cubes"] = dict(_PARAMS["sat-drat"])


def _evidence(method, params):
    ev = {"method": method, "lower_bound": 2}
    if method == "witness":
        ev = {"method": "witness", "source": "input", "weight": 2, "witness": {"X": [0, 1], "Z": []}}
    if params is not None:
        ev["params"] = params
    return ev


@pytest.mark.parametrize("method", ["bnb", "sat-drat", "sat-drat-cubes", "brute", "witness"])
def test_strict_parser_params_allowlist(tmp_path, method):
    """Every evidence method has an exact params allowlist (what its writer emits); anything else is rejected."""
    from hyqec_verify.certificate import EVIDENCE_PARAMS, load_certificate_json

    def parse(ev):
        doc = _cert(C4, "P", 2, 2, 0b11, [ev])
        p = tmp_path / "c.json"
        write_certificate(doc, p)
        return load_certificate_json(p)
    base = _PARAMS.get(method)
    assert EVIDENCE_PARAMS[method] == (None if base is None else set(base))
    parse(_evidence(method, base))                                   # exactly the writer's keys: accepted
    for extra in ("count_only", "evidence_level", "lower_bound", "x"):
        with pytest.raises(CodeError, match="unknown keys|carries no 'params'"):
            parse(_evidence(method, {**(base or {}), extra: True}))
    if base is not None:
        for missing in base:                                         # dropping keys stays legal (older writers)
            parse(_evidence(method, {k: v for k, v in base.items() if k != missing}))
    else:
        with pytest.raises(CodeError, match="carries no 'params'"):
            parse(_evidence(method, {}))


def test_strict_parser_rejects_unknown_keys_via_cli(tmp_path, capsys):
    """CLI: a certificate with an unknown params key is 'rejected' with exit status 1."""
    from hyqec_verify.cli import main
    doc = _cert(C4, "P", 2, 2, 0b11, [_evidence("bnb", {**_PARAMS["bnb"], "count_only": True})])
    p = tmp_path / "c.json"
    write_certificate(doc, p)
    rc = main(["verify", str(p), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["status"] == "rejected" and "unknown keys" in out["reason"], out


@pytest.mark.parametrize("policy", ["greedy+s0", "greedy+s0s1", "greedy+s0s1s2"])
def test_rule_policy_evidence_level(tmp_path, policy):
    """Opt-in S0/S1/S2 records: the evidence level is derived from method + params.bound. A run record is
    never labelled independently checked (only a v2 coverage certificate accepted by hyqec-check 2.0.0 is:
    tests/test_verify_coverage_v2.py); --rerun-bnb re-runs with exactly the recorded policy (same prover)."""
    if not _native():
        pytest.skip("native B&B unavailable")
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from known_codes import toric
    from hyqec_verify.engines import bnb
    c = toric(4)
    # version 1 cannot express these rules; the default export is version 2 (tests/native/test_coverage_v2.py)
    with pytest.raises(CodeError, match="coverage certificate v1"):
        bnb.run_bnb(c, 3, tmp_path / "cert", threads=2, bound=policy, emit_cert=True, cert_version=1)
    with pytest.raises(CodeError, match="CPU-only"):
        bnb.run_bnb(c, 3, tmp_path / "gpu", threads=2, bound=policy, gpus="0")
    o = bnb.run_bnb(c, 3, tmp_path / "run", threads=2, bound=policy)
    assert o.status == "none" and o.record["params"]["bound"] == policy
    assert set(o.record["params"]) == {"cap", "domain", "orbit_roots", "split", "stop_first", "bound"}
    greedy = bnb.run_bnb(c, 3, tmp_path / "greedy", threads=2)
    assert greedy.record["result"]["nodes"] != o.record["result"]["nodes"]

    def doc_for(record):
        doc = _cert(c, "P", 4, 4, None, [{**record, "lower_bound": 4}])
        doc["witness"] = None
        doc["result"] = {"status": "bounds", "lower_bound": 4, "upper_bound": None}
        return doc
    independent = "lower bound by bnb coverage certificate, policy greedy, independently checked"
    v = _verify(tmp_path, doc_for(o.record))
    item = next(ch for ch in v["checks"] if ch.get("method") == "bnb")
    assert v["verified_lower"] < 4 and v["lower_with_recorded_bnb"] == 4
    assert item["evidence_level"] == f"bnb run record, policy {policy}, not re-executed, not independently checked"
    v = _verify(tmp_path, doc_for(o.record), rerun_bnb=True)
    item = next(ch for ch in v["checks"] if ch.get("method") == "bnb")
    assert v["verified_lower"] == 4 and not v["failed"], v["checks"]
    assert item["rerun_nodes"] == o.record["result"]["nodes"]          # re-run with exactly this policy
    assert item["evidence_level"] == (f"lower bound by bnb rerun, policy {policy}, same-prover re-execution, "
                                      "not independently checked")
    # The same record relabelled greedy is re-run as greedy: the node count no longer matches.
    v = _verify(tmp_path, doc_for({**o.record, "params": {**o.record["params"], "bound": "greedy"}}), rerun_bnb=True)
    assert v["verified_lower"] < 4 and "node count" in v["failed"][0]["problem"]
    # A coverage attachment that is not a checker-accepted v2 file never makes a rule-policy record
    # independently checked (here the file does not exist; a v1 file: test_acceptance_referee.py).
    v = _verify(tmp_path, doc_for({**o.record, "coverage": {"file": "proof.cov.gz"}}), rerun_bnb=True,
                bnb_checker="false {file}")
    assert v["verified_lower"] < 4 and v["failed"]
    assert all(ch.get("evidence_level") != independent for ch in v["checks"])
    # The strict parser rejects unknown keys: an evidence-level field cannot be smuggled in.
    for extra in ({"count_only": True}, {"count_only": False}, {"evidence_level": "independently checked"}):
        with pytest.raises(CodeError, match="unknown keys"):
            _verify(tmp_path, doc_for({**o.record, "params": {**o.record["params"], **extra}}), rerun_bnb=True)
    # Unknown policy names are rejected, not silently treated as greedy.
    v = _verify(tmp_path, doc_for({**o.record, "params": {**o.record["params"], "bound": "greedy+s9"}}))
    assert v["failed"] and "unknown B&B policy" in v["failed"][0]["problem"]
