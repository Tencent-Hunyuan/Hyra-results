import itertools
import random

import pytest

import oracle
from hyqec_verify.core import CodeError, GF2Basis, StabilizerCode, kernel, symplectic, weight


def code_of(gens, n):
    return StabilizerCode(n, [oracle.to_int(g, n) for g in gens])


def test_gf2basis_against_enumeration():
    rng = random.Random(1)
    for _ in range(200):
        w = rng.randrange(1, 9)
        rows = [rng.randrange(1 << w) for _ in range(rng.randrange(0, 7))]
        B = GF2Basis(rows)
        span = {0}
        for r in rows:
            span |= {s ^ r for s in span}
        assert 1 << len(B) == len(span)
        for v in range(1 << w):
            assert B.contains(v) == (v in span)
            assert B.reduce(v) == B.reduce(B.reduce(v))
        # canonical form depends only on the subspace
        assert B.canonical() == GF2Basis(sorted(span)).canonical()
        K = kernel(rows, w)
        ker = {v for v in range(1 << w) if all(bin(v & r).count("1") % 2 == 0 for r in rows)}
        assert len(K) == w - len(B)
        assert all(v in ker for v in K)
        assert 1 << len(GF2Basis(K)) == len(ker)


@pytest.mark.parametrize("seed", range(60))
def test_random_codes_match_oracle(seed):
    rng = random.Random(seed)
    n = rng.randrange(2, 8)
    k = rng.randrange(0, n)
    gens = oracle.random_code(n, k, rng, extra_rows=rng.randrange(0, 3))
    iso, k_or, d_or, N, G = oracle.analyse(gens, n)
    assert iso and k_or == k
    c = code_of(gens, n)
    assert c.k == k
    assert len(c.normalizer_basis()) == n + k
    if k == 0:
        with pytest.raises(CodeError):
            c.require_logicals()
        assert c.logicals == []
        return
    assert len(c.logicals) == 2 * k
    Nset = set(N)
    best = None
    for p in itertools.product(range(4), repeat=n):
        v = oracle.to_int(p, n)
        expect = p in Nset and p not in G
        assert c.is_logical(v) == expect, p
        if p in Nset:
            assert c.is_nontrivial_via_logicals(v) == (p not in G)
        if expect:
            best = weight(v, n) if best is None else min(best, weight(v, n))
    assert best == d_or


def test_anticommuting_rejected():
    with pytest.raises(CodeError, match="anticommute"):
        StabilizerCode(2, [0b0001, 0b0100])   # X0 and Z0


def test_bad_rows_rejected():
    with pytest.raises(CodeError):
        StabilizerCode(2, [1 << 4])
    with pytest.raises(CodeError):
        StabilizerCode(0, [])
    with pytest.raises(CodeError):
        StabilizerCode.from_xz([{"X": [0, 0]}], 2)
    with pytest.raises(CodeError):
        StabilizerCode.from_xz([{"X": [2]}], 2)
    with pytest.raises(CodeError):
        StabilizerCode.from_xz([{"X": [True]}], 2)
    with pytest.raises(CodeError):
        StabilizerCode.from_xz([{"Y": [0]}], 2)


def test_group_hash_invariant_under_generator_choice():
    rng = random.Random(5)
    n = 6
    gens = oracle.random_code(n, 2, rng)
    c1 = code_of(gens, n)
    gens2 = list(gens)
    gens2[0] = oracle.mul(gens2[0], gens2[1])
    gens2.append(oracle.mul(gens2[2], gens2[3]))
    rng.shuffle(gens2)
    c2 = code_of(gens2, n)
    assert c1.group_hash() == c2.group_hash()
    # a different group (drop a generator) hashes differently
    c3 = code_of(gens[1:], n)
    assert c3.group_hash() != c1.group_hash()


def test_automorphism_check():
    # 5-qubit code: cyclic shifts of XZZXI
    n = 5
    base = "XZZXI"
    gens = [base[-i:] + base[:-i] for i in range(4)]
    from hyqec_verify.io import parse_pauli
    c = StabilizerCode(n, [parse_pauli(g)[0] for g in gens])
    shift = [(q + 1) % n for q in range(n)]
    assert c.is_automorphism(shift)
    assert not c.is_automorphism([1, 0, 2, 3, 4])
    assert not c.is_automorphism([0, 0, 1, 2, 3])
    assert not c.is_automorphism([0, 1, 2, 3])


def test_symplectic_basics():
    n = 1
    X, Z, Y = 0b01, 0b10, 0b11
    assert symplectic(X, Z, n) == 1 and symplectic(X, Y, n) == 1 and symplectic(Y, Y, n) == 0
    assert weight(Y, n) == 1
