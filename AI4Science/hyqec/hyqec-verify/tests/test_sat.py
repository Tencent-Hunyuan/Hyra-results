import gzip
import itertools
import json
import random

import pytest

import oracle
from hyqec_verify import tools
from hyqec_verify.core import StabilizerCode
from hyqec_verify.engines import sat
from hyqec_verify.io import parse_pauli
from hyqec_verify.symmetry import orbits_from, trivial_orbits


def dpll(clauses, assign):
    """Tiny DPLL with unit propagation (tests only). assign: dict var -> bool."""
    assign = dict(assign)
    while True:
        changed = False
        for c in clauses:
            unassigned, sat_ = [], False
            for l in c:
                v = assign.get(abs(l))
                if v is None:
                    unassigned.append(l)
                elif v == (l > 0):
                    sat_ = True
                    break
            if sat_:
                continue
            if not unassigned:
                return False
            if len(unassigned) == 1:
                l = unassigned[0]
                assign[abs(l)] = l > 0
                changed = True
        if not changed:
            break
    for c in clauses:
        for l in c:
            if abs(l) not in assign:
                for val in (True, False):
                    if dpll(clauses, {**assign, abs(l): val}):
                        return True
                return False
    return True


def code_from(gens, n):
    return StabilizerCode(n, [oracle.to_int(g, n) for g in gens])


@pytest.mark.parametrize("seed", range(12))
def test_encoding_semantics_exact(seed):
    """For every Pauli v and every W: CNF + units(v) satisfiable iff v is a
    nontrivial logical of weight <= W (in the domain)."""
    rng = random.Random(seed)
    n = rng.randrange(2, 5)
    k = rng.randrange(1, n)
    gens = oracle.random_code(n, k, rng, extra_rows=1)
    c = code_from(gens, n)
    _, _, d, N, G = oracle.analyse(gens, n)
    Nset = set(N)
    domain = rng.choice(["P", "P", "X", "Z"])
    for W in range(0, n):
        enc = sat.encode(c, W, domain=domain, chunk=3)
        cl = enc.formula.clauses
        for p in itertools.product(range(4), repeat=n):
            in_dom = domain == "P" or all(l in (0, 1 if domain == "X" else 2) for l in p)
            expect = p in Nset and p not in G and oracle.wt(p) <= W and in_dom
            units = {}
            for q, l in enumerate(p):
                units[q + 1] = bool(l & 1)
                units[n + q + 1] = bool(l & 2)
            assert dpll(cl, units) == expect, (W, p, domain)


@pytest.mark.parametrize("seed", range(6))
def test_orbit_roots_preserve_existence(seed):
    """With verified orbit roots, the CNF is satisfiable iff d <= W."""
    rng = random.Random(100 + seed)
    n = 5
    base = "XZZXI"
    gens = [parse_pauli(base[-i:] + base[:-i])[0] for i in range(4)]
    c = StabilizerCode(n, gens)
    orb = orbits_from(c, [[(q + 1) % n for q in range(n)]])
    assert orb.count == 1
    for W in range(0, 4):
        enc = sat.encode(c, W, orbits=orb)
        assert dpll(enc.formula.clauses, {}) == (W >= 3)


def test_xor_and_counter_small():
    for k in range(0, 5):
        for m in range(1, 6):
            f = sat.Formula(m)
            f.at_most(list(range(1, m + 1)), k)
            for vals in itertools.product((0, 1), repeat=m):
                a = {i + 1: bool(v) for i, v in enumerate(vals)}
                assert dpll(f.clauses, a) == (sum(vals) <= k)
    for chunk in (3, 4, 5):
        for m in range(1, 9):
            for par in (0, 1):
                f = sat.Formula(m)
                f.xor(list(range(1, m + 1)), par, chunk)
                for vals in itertools.product((0, 1), repeat=m):
                    a = {i + 1: bool(v) for i, v in enumerate(vals)}
                    assert dpll(f.clauses, a) == (sum(vals) % 2 == par)


def have_solver():
    try:
        tools.find_solver("kissat")
        tools.checker()
        return True
    except tools.ToolMissing:
        return False


needs_sat = pytest.mark.skipif(not have_solver(), reason="kissat / drat-trim not available")


@needs_sat
@pytest.mark.sat
@pytest.mark.parametrize("seed", range(8))
def test_prove_and_witness(seed, tmp_path):
    rng = random.Random(seed)
    n = rng.randrange(4, 8)
    gens = oracle.random_code(n, rng.randrange(1, 3), rng)
    c = code_from(gens, n)
    _, _, d, _, _ = oracle.analyse(gens, n)
    if d > 1:
        out = sat.prove(c, d - 1, tmp_path / "lb", lrat=True)
        assert out.status == "UNSAT"
        chk = sat.check_record(c, out.record, tmp_path / "lb")
        assert chk["verified"]
    out = sat.prove(c, d, tmp_path / "ub")
    assert out.status == "SAT"
    assert c.check_witness(out.model_pauli) <= d


@needs_sat
@pytest.mark.sat
def test_cubes_and_tamper(tmp_path):
    n = 5
    base = "XZZXI"
    c = StabilizerCode(n, [parse_pauli(base[-i:] + base[:-i])[0] for i in range(4)])
    orb = trivial_orbits(n)
    out = sat.prove_cubes(c, 2, tmp_path / "b", orbits=orb, n_first=3, split_labels=True, workers=4)
    assert out.status == "UNSAT", out.record
    rec = out.record
    assert len(rec["cubes"]) == 3 * 3 + 1
    assert sat.check_record(c, rec, tmp_path / "b")["verified"]
    # tamper: a different bound regenerates a different CNF
    bad = json.loads(json.dumps(rec))
    bad["params"]["bound"] = 3
    assert not sat.check_record(c, bad, tmp_path / "b")["verified"]
    # tamper: drop a cube (coverage CNF changes, leaf count wrong)
    bad = json.loads(json.dumps(rec))
    del bad["cubes"][0]
    with pytest.raises(Exception):
        sat.check_record(c, bad, tmp_path / "b")
    # tamper: corrupt a proof file
    pf = tmp_path / "b" / rec["leaves"][0]["proof"]["file"]
    raw = gzip.open(pf).read()
    with gzip.open(pf, "wb") as g:
        g.write(raw[: len(raw) // 2])
    assert not sat.check_record(c, rec, tmp_path / "b")["verified"]
    # a different code (one generator dropped) must not verify
    c2 = StabilizerCode(n, c.generators[:3])
    assert not sat.check_record(c2, rec, tmp_path / "b")["verified"]
