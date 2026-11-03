import random

import pytest

import oracle
from hyqec_verify.core import StabilizerCode
from hyqec_verify.engines.brute import brute_force
from hyqec_verify.engines.ris import ris_upper_bound


def random_case(seed, nmin=3, nmax=8):
    rng = random.Random(seed)
    n = rng.randrange(nmin, nmax)
    k = rng.randrange(1, n)
    gens = oracle.random_code(n, k, rng, extra_rows=rng.randrange(0, 2))
    c = StabilizerCode(n, [oracle.to_int(g, n) for g in gens])
    return gens, n, c


@pytest.mark.parametrize("seed", range(40))
def test_brute_matches_oracle(seed):
    gens, n, c = random_case(seed)
    _, _, d, _, _ = oracle.analyse(gens, n)
    r = brute_force(c, n)
    assert r.found == d
    assert c.check_witness(r.witness) == d
    if d > 1:
        assert brute_force(c, d - 1).found is None


@pytest.mark.parametrize("seed", range(40))
def test_ris_finds_distance_small(seed):
    gens, n, c = random_case(seed)
    _, _, d, _, _ = oracle.analyse(gens, n)
    r = ris_upper_bound(c, trials=200, seed=seed)
    assert r.weight == d          # tiny codes: RIS must hit the minimum quickly
    assert c.check_witness(r.witness) == d


def css_case(seed):
    """Random CSS code: HX random, HZ in the kernel of HX."""
    from hyqec_verify.core import GF2Basis
    rng = random.Random(seed)
    while True:
        n = rng.randrange(4, 9)
        rx = rng.randrange(1, n - 1)
        hx = [rng.randrange(1, 1 << n) for _ in range(rx)]
        ker = GF2Basis(hx).kernel(n)
        if len(ker) < 2:
            continue
        hz = [v for v in ker if rng.random() < 0.5][: len(ker) - 1]
        hxl = [[q for q in range(n) if (v >> q) & 1] for v in hx]
        hzl = [[q for q in range(n) if (v >> q) & 1] for v in hz]
        c = StabilizerCode.css(hxl, hzl, n)
        if c.k >= 1:
            return c, hxl, hzl, n


@pytest.mark.parametrize("seed", range(25))
def test_side_distances(seed):
    c, hxl, hzl, n = css_case(seed)
    gens = [oracle.from_int(g, n) for g in c.generators]
    dx = oracle.side_distance(gens, n, 1)
    dz = oracle.side_distance(gens, n, 2)
    _, _, d, _, _ = oracle.analyse(gens, n)
    assert d == min(x for x in (dx, dz) if x is not None)
    for dom, ds in (("X", dx), ("Z", dz)):
        r = brute_force(c, n, domain=dom)
        assert r.found == ds
        rr = ris_upper_bound(c, trials=100, seed=seed, domain=dom)
        assert rr.weight == ds


def test_env_new_name_and_deprecated_alias(monkeypatch):
    import warnings
    from hyqec_verify import _env
    monkeypatch.delenv("HYQEC_SOME_OPTION", raising=False)
    monkeypatch.setenv("STABCERT_SOME_OPTION", "old")
    _env._warned.discard("SOME_OPTION")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert _env.getenv("SOME_OPTION") == "old"
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    monkeypatch.setenv("HYQEC_SOME_OPTION", "new")
    assert _env.getenv("SOME_OPTION") == "new"                     # the new name wins
    monkeypatch.delenv("HYQEC_SOME_OPTION")
    monkeypatch.delenv("STABCERT_SOME_OPTION")
    assert _env.getenv("SOME_OPTION", "d") == "d"
