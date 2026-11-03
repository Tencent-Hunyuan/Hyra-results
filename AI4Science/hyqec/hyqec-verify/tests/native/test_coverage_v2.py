"""Coverage certificate version 2 exporter (docs/BNB_CERT_SPEC_v2.md): producer-side self-consistency, not a checker.

* no behaviour change: node counts and answers with and without export, for every policy; version 1 under greedy
  unchanged; version 1 refused under the S0/S1/S2 policies;
* structure, counts, traces and |PK| = h_field from the records (cov2_ref.consistency), strict gzip transport;
* every prune node's PK equals an independent re-implementation of PACKING_SELECT (spec 6.1, cov2_ref) evaluated on
  the node state dumped by a test-only build (cov2_pk_dump.cpp), and every recorded PK equals the dumped one.
"""
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest
from hyqec_verify import native
from hyqec_verify.core import StabilizerCode
from hyqec_verify.engines.bnb import read_coverage_header, run_bnb
from hyqec_verify.symmetry import orbits_from, trivial_orbits
from known_codes import rotated_surface
from . import cov2_ref
from .helpers import run, write_instance
from .test_bnb import five

pytestmark = pytest.mark.native
POLICIES = ("greedy+s0", "greedy+s0s1", "greedy+s0s1s2")
HERE = Path(__file__).parent


def golay():
    """[[23,1,7]]: X rows = Z rows = the 23 cyclic shifts of one weight-8 word of the dual Golay code (dependent)."""
    word = 0b1010010011111   # qubits 0,1,2,3,4,7,10,12: a weight-8 word of the [23,11,8] even-weight Golay subcode (checked below)
    gen = sum(1 << i for i, c in enumerate([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1]) if c)   # g(x) of the [23,12,7] code
    code_words = [((gen << s) | (gen >> (23 - s))) & ((1 << 23) - 1) for s in range(12)]
    assert bin(word).count("1") == 8 and all(bin(word & c).count("1") % 2 == 0 for c in code_words)
    rows = [[q for q in range(23) if (word >> ((q - s) % 23)) & 1] for s in range(23)]
    return StabilizerCode.css(rows, rows, 23, name="golay-23")


@pytest.fixture(scope="module")
def rules():
    return native.build("bnb-cpu", variant="1024-rules")


@pytest.fixture(scope="module")
def dumper(tmp_path_factory):
    target = tmp_path_factory.mktemp("pkdump") / "pk-dump"
    flags = ["-O2", "-std=c++17", "-x", "c++", "-pthread", "-DNMAX=1024", "-DMMAX=1024",
             '-DSTABCERT_SOURCE_SHA256="test-only-pk-dump"']
    cmd = shlex.split(os.environ.get("CXX", "c++")) + flags + [str(HERE / "cov2_pk_dump.cpp"), "-lz", "-o", str(target)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    return target


def emit(binary, code, tmp_path, cap, split, *args, domain="P", detail="frontier", orbits=None, name="proof", env=None):
    orbits = orbits or trivial_orbits(code.n)
    from hyqec_verify.engines.bnb import write_instance as package_instance
    instance = tmp_path / f"{name}.inst"
    package_instance(instance, code, orbits, domain)
    meta = tmp_path / f"{name}.json"
    meta.write_text(json.dumps({"group_sha256": code.group_hash(), "permutations": orbits.generators}))
    path = tmp_path / f"{name}.cov"
    path.unlink(missing_ok=True)
    old = dict(os.environ)
    os.environ.update(env or {})
    try:
        done = run(binary, instance, cap, "--split", split, "--emit-cert", path, "--cert-metadata", meta,
                   "--cert-detail", detail, *args)
    finally:
        os.environ.clear()
        os.environ.update(old)
    return done, path, instance


def test_v2_counts_traces_and_v1_identity(rules, tmp_path):
    code = rotated_surface(5)
    plain = native.build("bnb-cpu")
    for domain in ("P", "X", "Z"):
        for split in (1, 2, 3):
            for detail in ("frontier", "full"):
                v1, p1, inst = emit(plain, code, tmp_path, 4, split, domain=domain, detail=detail, name="v1")
                v1r, p1r, _ = emit(rules, code, tmp_path, 4, split, domain=domain, detail=detail, name="v1r")
                v2, p2, _ = emit(rules, code, tmp_path, 4, split, "--cert-version", 2, domain=domain, detail=detail,
                                 name="v2")
                h1, r1 = cov2_ref.header_bytes(p1.read_bytes())
                h1r, r1r = cov2_ref.header_bytes(p1r.read_bytes())
                h2, r2 = cov2_ref.header_bytes(p2.read_bytes())
                # the rules build writes the same version 1 artifact; version 2 greedy has the version-1 records
                assert r1 == r1r == r2 and p1.read_bytes()[:8] == b"SBCOV1\r\n" and p2.read_bytes()[:8] == b"SBCOV2\r\n"
                j1, j2 = json.loads(h1[16:]), json.loads(h2[16:])
                assert j2["rules"] == "greedy" and j2["rules_version"] == 1 and "rules" not in j1
                for j in (j1, j2):
                    j.pop("provenance")
                assert sorted(k for k in j2 if j1.get(k) != j2[k]) == ["format", "rules", "rules_version", "search",
                                                                        "trace"]
                assert v1["global_trace"] == v2["global_trace"] and v1["nodes"] == v1r["nodes"] == v2["nodes"]
                cov2_ref.consistency(p2.read_bytes(), v2)
            for policy in POLICIES:
                count = run(rules, inst, 4, "--split", split, "--bound", policy)
                for detail in ("frontier", "full"):
                    for extra in ((), ("--cpu-budget", 1, "--delta", 1, "--chunk", 2)):
                        done, path, _ = emit(rules, code, tmp_path, 4, split, "--bound", policy, "--cert-version", 2,
                                             *extra, domain=domain, detail=detail)
                        assert done["found"] == -1 and done["nodes"] == count["nodes"]
                        c = cov2_ref.consistency(path.read_bytes(), done)
                        assert c["rules"] == policy.split("+")[1]
                        gz = cov2_ref.gzip_v2(path.read_bytes())
                        assert cov2_ref.transport_ok(gz) == (True, path.read_bytes())


def test_v1_refused_and_find_unpublished(rules, tmp_path):
    code = five()
    instance = write_instance(code, tmp_path / "five.inst")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"group_sha256": code.group_hash(), "permutations": []}))
    out = tmp_path / "proof.cov"
    for policy in POLICIES:
        for version in ((), ("--cert-version", 1)):
            p = run(rules, instance, 2, "--bound", policy, "--emit-cert", out, "--cert-metadata", meta, *version,
                    check=False)
            assert p.returncode == 2 and "incompatible" in p.stderr and not out.exists()
        # a find under version 2: the witness, and no published artifact
        done = run(rules, instance, 3, "--bound", policy, "--emit-cert", out, "--cert-metadata", meta,
                   "--cert-version", 2)
        assert done["found"] == 3 and not done["certificate_complete"] and not out.exists()
    for bad in (("--cert-version", 3), ("--cert-version", 2, "--bound", "none")):
        p = run(rules, instance, 2, "--emit-cert", out, "--cert-metadata", meta, *bad, check=False)
        assert p.returncode == 2 and not out.exists()
    p = run(rules, instance, 2, "--cert-version", 2, check=False)
    assert p.returncode == 2 and "require --emit-cert" in p.stderr


def test_pk_matches_spec_reimplementation(dumper, tmp_path):
    code = golay()
    cyc = orbits_from(code, [[(q + 1) % 23 for q in range(23)]])
    census, recorded = {}, 0
    cov2_ref.PATHS.clear()
    for code, orbits, domain, cap in ((code, None, "P", 6), (code, cyc, "P", 6), (code, None, "X", 6),
                                      (rotated_surface(5), None, "P", 4)):
        for policy in POLICIES:
            rule_set = policy.split("+")[1]
            for split in (3, 4):
                dump = tmp_path / "pk.dump"
                dump.unlink(missing_ok=True)
                done, path, _ = emit(dumper, code, tmp_path, cap, split, "--bound", policy, "--cert-version", 2,
                                     domain=domain, detail="full", orbits=orbits, env={"X1_DUMP": str(dump)})
                raw = path.read_bytes()
                c = cov2_ref.consistency(raw, done)
                stats, bad = cov2_ref.check_dump(cov2_ref.Inst(read_coverage_header(path)["instance_text"]), dump,
                                                 rule_set)
                assert not bad, bad[:3]
                for k, v in stats.items():
                    census[k] = census.get(k, 0) + v
                states = {}
                for line in dump.read_text().splitlines():
                    p = line.split()
                    states.setdefault((int(p[3], 16), cov2_ref.RULE_OF[p[0]]), set()).add(tuple(map(int, p[5:])))
                for r in cov2_ref.decode(raw)[2]:
                    if r[0] == 0x10 and r[3] in (2, 5, 6, 7):
                        assert states[(r[4], r[3])] == {tuple(r[7])}
                        recorded += 1
                if rule_set == "s0s1s2" and split == 3 and cap == 6 and domain == "P" and orbits is None:
                    assert {2, 5, 6, 7} <= set(c["task"]) and {5, 6, 7} <= set(c["top"])
    assert {2, 5, 6, 7} <= set(census) and recorded > 1000
    # every branch of PACKING_SELECT that these instances reach
    assert {"greedy", "mis-a", "mis-b-stop", "mis-c", "mis-c-none"} <= set(cov2_ref.PATHS), cov2_ref.PATHS


def test_wide_pk_truncation(tmp_path):
    """More than 96 active generators: the greedy list is the first 96 of A in index order, then sorted (6.1)."""
    target = tmp_path / "pk-dump-w128"
    flags = ["-O2", "-std=c++17", "-x", "c++", "-pthread", "-DNMAX=1024", "-DMMAX=1024", "-DWMAX=128", "-DCMAX=128",
             '-DSTABCERT_SOURCE_SHA256="test-only-pk-dump"']
    subprocess.run(shlex.split(os.environ.get("CXX", "c++")) + flags + [str(HERE / "cov2_pk_dump.cpp"), "-lz", "-o",
                   str(target)], check=True, capture_output=True, text=True, timeout=300)
    base = golay()
    code = StabilizerCode(23, [g for g in base.generators for _ in range(8)])   # column degree 128
    dump = tmp_path / "pk.dump"
    cov2_ref.PATHS.clear()
    done, path, _ = emit(target, code, tmp_path, 6, 3, "--bound", "greedy+s0s1s2", "--cert-version", 2, domain="X",
                         detail="full", env={"X1_DUMP": str(dump)})
    cov2_ref.consistency(path.read_bytes(), done)
    stats, bad = cov2_ref.check_dump(cov2_ref.Inst(read_coverage_header(path)["instance_text"]), dump, "s0s1s2")
    assert not bad and {"R>96", "greedy-R>64"} <= set(cov2_ref.PATHS), (bad[:3], cov2_ref.PATHS)


def test_run_bnb_writes_v2(tmp_path):
    code = rotated_surface(3)
    o = run_bnb(code, 2, tmp_path / "w", threads=2, split=2, emit_cert=True, bound="greedy+s0s1s2")
    assert o.status == "none" and o.record["coverage"]["header"]["format"] == "stabcert-bnb-coverage/2"
    assert o.record["coverage"]["header"]["rules"] == "s0s1s2"
    gz = (tmp_path / "w" / o.record["coverage"]["file"]).read_bytes()
    ok, raw = cov2_ref.transport_ok(gz)
    assert ok and cov2_ref.consistency(raw)["nodes"] == o.record["result"]["nodes"]
    from hyqec_verify.core import CodeError
    with pytest.raises(CodeError, match="v1 admits only"):
        run_bnb(code, 2, tmp_path / "x", threads=2, emit_cert=True, bound="greedy+s0", cert_version=1)
