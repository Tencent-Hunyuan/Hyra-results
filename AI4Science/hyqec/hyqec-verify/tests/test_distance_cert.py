import json
import shutil
from pathlib import Path

import pytest

from known_codes import known, toric, rotated_surface
from hyqec_verify import tools
from hyqec_verify.certificate import build_certificate, verify_certificate, write_certificate
from hyqec_verify.cli import main
from hyqec_verify.distance import distance


def have_sat():
    try:
        tools.find_solver("kissat")
        tools.checker()
        return True
    except tools.ToolMissing:
        return False


needs_sat = pytest.mark.skipif(not have_sat(), reason="kissat / drat-trim not available")


def test_brute_path_small_codes():
    for c, k, d, dx, dz in known():
        if c.n > 9:
            continue
        r = distance(c, engines=(), budget=30, ris_seconds=0.5)
        assert r.status == "exact" and r.upper == d, c.name
        if dx:
            assert distance(c, domain="X", engines=(), budget=30, ris_seconds=0.5).upper == dx


@needs_sat
@pytest.mark.sat
def test_sat_distance_certificate_roundtrip(tmp_path):
    c = toric(4)
    r = distance(c, engines=("sat",), budget=300, ris_seconds=3, workdir=tmp_path / "w", brute_limit=0)
    assert r.status == "exact" and r.upper == 4, r.notes
    cert = build_certificate(c, r, None)
    p = write_certificate(cert, tmp_path / "toric4.json")
    shutil.rmtree(tmp_path / "w")                    # the certificate must be self-contained
    v = verify_certificate(p)
    assert v["status"] == "exact" and v["verified_lower"] == 4 and v["verified_upper"] == 4
    assert not v["failed"]
    # tamper: a different group (toric generators are dependent, so drop three)
    doc = json.loads(p.read_text())
    doc["code"]["generators"] = doc["code"]["generators"][3:]
    q = tmp_path / "tampered.json"
    q.write_text(json.dumps(doc))
    with pytest.raises(Exception):
        verify_certificate(q)                         # group hash mismatch
    doc = json.loads(p.read_text())
    side = "X" if doc["witness"]["X"] else "Z"
    doc["witness"][side] = doc["witness"][side][1:]
    doc["evidence"] = [e for e in doc["evidence"] if e["method"] != "witness"]
    q.write_text(json.dumps(doc))
    shutil.copytree(tmp_path / "toric4.artifacts", tmp_path / "tampered.artifacts")
    v = verify_certificate(q)
    assert any(f["item"] == "witness" for f in v["failed"]) and v["verified_upper"] is None
    assert v["status"] == "bounds"


@needs_sat
@pytest.mark.sat
def test_cli_distance_and_verify(tmp_path, capsys):
    c = rotated_surface(5)
    path = tmp_path / "s5.json"
    path.write_text(json.dumps({"n": c.n, "generators": c.to_xz()}))
    rc = main(["distance", str(path), "--engines", "sat", "--budget", "300", "--ris-seconds", "3",
               "--json", "-o", str(tmp_path / "cert.json")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["status"] == "exact" and out["d"] == 5
    rc = main(["verify", str(tmp_path / "cert.json"), "--json"])
    v = json.loads(capsys.readouterr().out)
    assert rc == 0 and v["status"] == "exact" and v["verified_lower"] == 5
    rc = main(["distance", str(path), "--side", "X", "--engines", "sat", "--budget", "300", "--ris-seconds", "2",
               "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["d"] == 5 and out["domain"] == "X"


def test_side_rejected_for_non_css(tmp_path, capsys):
    from hyqec_verify.core import CodeError
    c = known()[0][0]                      # five-qubit code: non-CSS
    with pytest.raises(CodeError, match="CSS"):
        distance(c, domain="X", budget=10)
    path = tmp_path / "five.json"
    path.write_text(json.dumps({"n": 5, "generators": c.to_xz()}))
    for cmd in (["distance", str(path), "--side", "Z"], ["lower", str(path), "--side", "X", "--bound", "2"],
                ["upper", str(path), "--side", "X", "--seconds", "1"]):
        rc = main(cmd + ["--json"])
        out = json.loads(capsys.readouterr().out)
        assert rc == 1 and out["status"] == "rejected" and "CSS" in out["reason"]
    # a CSS code is accepted
    s = rotated_surface(3)
    assert distance(s, domain="X", engines=(), budget=10, ris_seconds=0.5).upper == 3


def _bnb_native():
    try:
        from hyqec_verify.engines.bnb import locate_binary
        locate_binary(rotated_surface(3))
        return True
    except Exception:
        return False


@pytest.mark.native
@pytest.mark.skipif(not _bnb_native(), reason="native B&B unavailable")
def test_bnb_coverage_certificate_binding(tmp_path):
    from hyqec_verify.engines.bnb import read_coverage_header
    c = toric(3)
    r = distance(c, engines=("bnb",), budget=120, ris_seconds=1, brute_limit=0, bnb_cert=True,
                 workdir=tmp_path / "w", threads=2)
    assert r.status == "exact" and r.upper == 3
    p = write_certificate(build_certificate(c, r, None), tmp_path / "t3.json")
    shutil.rmtree(tmp_path / "w")
    doc = json.loads(p.read_text())
    ev = next(e for e in doc["evidence"] if e["method"] == "bnb")
    h = read_coverage_header(p.parent / ev["artifact_dir"] / ev["coverage"]["file"])
    assert h["group_sha256"] == c.group_hash() and h["cap"] == 2 and h["domain"] == "P"
    # a stub checker that exits 0 but reports nothing is NOT an acceptance
    assert verify_certificate(p, bnb_checker="true {file}")["status"] == "bounds"
    ok_stub = "python3 -c \"print('{\\\"certificate_verified\\\": true}')\" {file}"
    assert verify_certificate(p, bnb_checker=ok_stub)["status"] == "exact"
    assert verify_certificate(p, bnb_checker="false {file}")["failed"]
    assert verify_certificate(p)["verified_lower"] == 1          # not counted without a checker or rerun
    # binding: a coverage file for another cap must be refused even if the checker accepts it
    doc["evidence"] = [dict(e, params=dict(e["params"], cap=1)) if e["method"] == "bnb" else e
                       for e in doc["evidence"]]
    q = tmp_path / "t3b.json"
    q.write_text(json.dumps(doc))
    shutil.copytree(tmp_path / "t3.artifacts", tmp_path / "t3b.artifacts")
    v = verify_certificate(q, bnb_checker=ok_stub)
    assert any("does not match" in (f.get("problem") or "") for f in v["failed"])


@pytest.mark.parametrize("engine", ["sat", "bnb"])
def test_budget_is_a_wall_clock_limit(tmp_path, capsys, engine):
    """A hard code with a tiny budget must come back within the budget (+ start-up), exit 2,
    with partial bounds and inconclusive='budget'."""
    import time
    from pathlib import Path
    import os
    # HYQEC_TEST_UPSTREAM: a checkout of the qldpc-challenge repository (codes/452-115-18.json)
    code = Path(os.environ.get("HYQEC_TEST_UPSTREAM", "/nonexistent")) / "codes" / "452-115-18.json"
    if not code.exists():
        if engine == "bnb":                  # B&B proves every small sparse fallback within the budget
            pytest.skip("set HYQEC_TEST_UPSTREAM to a qldpc-challenge checkout (codes/452-115-18.json)")
        c = toric(12)                        # fallback: too big for SAT to finish in 3 s
        code = tmp_path / "t12.json"
        code.write_text(json.dumps({"n": c.n, "generators": c.to_xz()}))
    if engine == "bnb" and not _bnb_native():
        pytest.skip("native B&B unavailable")
    if engine == "sat" and not have_sat():
        pytest.skip("no SAT tools")
    t0 = time.monotonic()
    rc = main(["distance", str(code), "--engines", engine, "--budget", "3", "--brute-limit", "0",
               "--no-input-witness", "--threads", "2", "--json"])
    wall = time.monotonic() - t0
    out = json.loads(capsys.readouterr().out)
    assert rc == 2 and out["status"] == "bounds" and out.get("inconclusive") == "budget", out
    assert wall < 3 + 8, wall                 # 3 s budget + binary lookup / start-up


@pytest.mark.native
@pytest.mark.skipif(not _bnb_native(), reason="native B&B unavailable")
def test_missing_checker_never_counts(tmp_path):
    c = toric(3)
    r = distance(c, engines=("bnb",), budget=60, ris_seconds=1, brute_limit=0, bnb_cert=True,
                 workdir=tmp_path / "w", threads=2)
    p = write_certificate(build_certificate(c, r, None), tmp_path / "t.json")
    v = verify_certificate(p, bnb_checker="/nonexistent/bnbcheck {file}")
    assert v["verified_lower"] == 1 and v["failed"]
    assert "could not be run" in v["failed"][0]["problem"]
    v = verify_certificate(p)                      # no checker at all: recorded, not counted
    assert v["verified_lower"] == 1 and v["bnb_records_not_rerun"] == 1


@pytest.mark.native
@pytest.mark.skipif(not _bnb_native(), reason="native B&B unavailable")
def test_wrap_coverage_of_a_hyqec_made_file(tmp_path, capsys):
    """wrap-coverage must parse the v1 instance header (regression: n was read from the wrong token)."""
    from hyqec_verify.certificate import coverage_evidence
    c = toric(3)
    r = distance(c, engines=("bnb",), budget=60, ris_seconds=1, brute_limit=0, bnb_cert=True,
                 workdir=tmp_path / "w", threads=2)
    ev = next(e for e in r.evidence if e.get("coverage"))
    cov = Path(ev["directory"]) / ev["coverage"]["file"]
    w = coverage_evidence(c, cov)
    assert w["params"]["orbit_roots"]["orbit_of"] == ev["params"]["orbit_roots"]["orbit_of"]
    assert w["params"]["orbit_roots"]["reps"] == ev["params"]["orbit_roots"]["reps"]
    assert w["lower_bound"] == ev["lower_bound"] and w["instance_sha256"] == ev["instance_sha256"]
    code = tmp_path / "t3.json"
    code.write_text(json.dumps({"n": c.n, "generators": c.to_xz()}))
    assert main(["wrap-coverage", str(code), str(cov), "-o", str(tmp_path / "wc.json"), "--json"]) == 0
    capsys.readouterr()


def test_requested_engine_flag_false_for_trivial_lower_bound(tmp_path, capsys):
    """Acceptance finding: an engine that proved nothing beyond d >= 1 must not be reported as having proved it."""
    import json
    from hyqec_verify.cli import main
    p = tmp_path / "c.txt"
    p.write_text("XXXX\nZZZZ\n")                                    # [[4,2,2]]
    import hyqec_verify.cli as cli
    from hyqec_verify.distance import DistanceResult
    real = cli._run_distance
    def fake(args, lc):                                                  # SAT timed out at W=1
        args.engines_explicit = True
        return DistanceResult("bounds", "P", 1, 2, 3, [])
    cli._run_distance = fake
    try:
        main(["distance", str(p), "--engines", "sat", "--brute-limit", "0", "--no-input-witness", "--json"])
    finally:
        cli._run_distance = real
    cap = capsys.readouterr()
    out = json.loads(cap.out.strip().splitlines()[-1])
    assert out["lower_bound"] == 1 and out["requested_engines_proved_lower_bound"] is False
    assert "trivial" in cap.err
    p1 = tmp_path / "d1.txt"
    p1.write_text("ZZI\nIZZ\n")                                      # repetition code, d = 1
    main(["distance", str(p1), "--engines", "sat", "--json"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["status"] == "exact" and out["upper_bound"] == 1
    assert out["requested_engines_proved_lower_bound"] is False
    assert any("trivial" in n for n in out.get("notes", []))


def test_unchecked_sat_proof_does_not_count(tmp_path, capsys, monkeypatch):
    """Lead (rc2): an UNCHECKED SAT result (proof not verified) raises no bound and is not attributed."""
    import json
    from hyqec_verify.cli import main
    from hyqec_verify.engines import sat
    monkeypatch.setattr(sat, "prove", lambda *a, **k: sat.SatOutcome("UNCHECKED", {"method": "sat-drat"}))
    p = tmp_path / "c.txt"
    p.write_text("XXXX\nZZZZ\n")                                    # [[4,2,2]]
    main(["distance", str(p), "--engines", "sat", "--brute-limit", "0", "--no-input-witness", "--json"])
    cap = capsys.readouterr()
    out = json.loads(cap.out.strip().splitlines()[-1])
    assert out["lower_bound"] == 1 and out["status"] == "bounds"
    assert out["requested_engines_proved_lower_bound"] is False and "trivial" in cap.err
    assert any("UNCHECKED" in n for n in out["notes"])
