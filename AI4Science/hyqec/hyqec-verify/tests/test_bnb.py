"""B&B wrapper tests. Runs against the native build (if it builds here) and, when
HYQEC_LEGACY_BNB points at the pre-release prototype's pauli_cert_cpu, against that too."""
import os
import random

import pytest

import oracle
from known_codes import known
from hyqec_verify.core import StabilizerCode
from hyqec_verify.engines.bnb import instance_text, run_bnb, locate_binary
from hyqec_verify.symmetry import find_automorphisms, orbits_from, trivial_orbits


def _native():
    try:
        locate_binary(StabilizerCode(1, [0b10]))
        return True
    except Exception:
        return False


ENGINES = []
if os.environ.get("HYQEC_LEGACY_BNB"):
    ENGINES.append(("legacy", dict(binary=os.environ["HYQEC_LEGACY_BNB"], legacy=True)))
if _native():
    ENGINES.append(("native", {}))

pytestmark = [pytest.mark.native,
              pytest.mark.skipif(not ENGINES, reason="no B&B binary available")]


def random_codes(count, dmin=2):
    out = []
    seed = 0
    while len(out) < count:
        rng = random.Random(seed)
        seed += 1
        n = rng.randrange(4, 9)
        k = rng.randrange(1, max(2, n // 3))
        gens = oracle.random_code(n, k, rng, depth=n * n)
        _, _, d, _, _ = oracle.analyse(gens, n)
        if d >= dmin:
            out.append((StabilizerCode(n, [oracle.to_int(g, n) for g in gens]), d))
    return out


@pytest.mark.parametrize("engine", ENGINES, ids=[e[0] for e in ENGINES])
def test_random_codes_exact(engine, tmp_path):
    _, kw = engine
    for i, (c, d) in enumerate(random_codes(25)):
        o = run_bnb(c, c.n - 1, tmp_path / f"m{i}", threads=2, **kw)
        assert o.status == "found" and o.found == d, (i, d, o.status, o.found, o.record)
        o = run_bnb(c, d - 1, tmp_path / f"l{i}", threads=2, **kw)
        assert o.status == "none", (i, d, o.record)


@pytest.mark.parametrize("engine", ENGINES, ids=[e[0] for e in ENGINES])
def test_known_codes_with_symmetry(engine, tmp_path):
    _, kw = engine
    for c, k, d, dx, dz in known():
        perms = find_automorphisms(c, "cyclic")
        orb = orbits_from(c, perms) if perms else trivial_orbits(c.n)
        o = run_bnb(c, d, tmp_path / f"{c.name}-min", orbits=orb, threads=4, **kw)
        assert o.status == "found" and o.found == d, (c.name, o.record)
        o = run_bnb(c, d - 1, tmp_path / f"{c.name}-lb", orbits=orb, threads=4, **kw)
        assert o.status == "none", c.name


@pytest.mark.skipif(not _native(), reason="native build unavailable")
def test_side_domains(tmp_path):
    for c, k, d, dx, dz in known():
        for dom, ds in (("X", dx), ("Z", dz)):
            if ds is None:
                continue
            o = run_bnb(c, c.n - 1 if c.n <= 40 else ds + 1, tmp_path / f"{c.name}-{dom}", domain=dom, threads=4)
            assert o.status == "found" and o.found == ds, (c.name, dom, o.record)
            o = run_bnb(c, ds - 1, tmp_path / f"{c.name}-{dom}-lb", domain=dom, threads=4)
            assert o.status == "none", (c.name, dom)


def test_instance_format():
    c, *_ = known()[0]
    t = instance_text(c, trivial_orbits(5), "P").splitlines()
    assert t[0] == "stabcert-bnb-instance 1"
    assert t[1] == "5 4 2"
    assert t[2] == "4 0 1 1 2 2 2 3 1"          # XZZXI
    assert t[-1] == "domain P"


@pytest.mark.skipif(not _native(), reason="native build unavailable")
def test_wide_generator_variant(tmp_path):
    """Generators heavier than 32 select the CPU-only wide build; GPU refuses clearly."""
    from hyqec_verify.core import CodeError
    from hyqec_verify.engines.bnb import generator_shape, locate_binary
    c = StabilizerCode.css([list(range(48))], [list(range(48))], 48)      # [[48,46,2]]
    assert generator_shape(c) == (48, 2)
    try:
        _, ver = locate_binary(c)
    except Exception as e:
        pytest.skip(f"wide variant not available: {e}")
    assert ver["limits"]["WMAX"] >= 64
    assert run_bnb(c, 1, tmp_path / "lb", threads=2).status == "none"
    o = run_bnb(c, 2, tmp_path / "min", threads=2)
    assert o.status == "found" and o.found == 2
    with pytest.raises(CodeError, match="CPU engine"):
        locate_binary(c, gpu=True)
