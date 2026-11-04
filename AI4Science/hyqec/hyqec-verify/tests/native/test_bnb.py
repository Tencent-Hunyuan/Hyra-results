import random
import pytest
from hyqec_verify import native
from hyqec_verify.core import StabilizerCode
from oracle import analyse, random_code, to_int, side_distance
from .helpers import write_instance, run, witness, python_nodes

pytestmark = pytest.mark.native

@pytest.fixture(scope="module")
def binary():
    return native.build("bnb-cpu")


def five():
    rows = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    return StabilizerCode(5, [to_int(tuple("IXZY".index(c) for c in row), 5) for row in rows])


def test_five_and_python_tree(binary, tmp_path):
    code = five()
    path = write_instance(code, tmp_path / "five.inst")
    for cap in (0, 1, 2):
        expected = python_nodes(code, cap)
        for split in (1, 2, 3, -10):
            done = run(binary, path, cap, "--split", split)
            assert done["found"] == -1
            assert done["nodes"] == expected
    done = run(binary, path, 5)
    assert done["found"] == 3
    witness(code, done)


def test_random_vs_oracle(binary, tmp_path):
    rng = random.Random(8164)
    for i in range(30):
        n = rng.randrange(2, 8)
        gens = random_code(n, rng.randrange(1, n), rng, extra_rows=2)
        code = StabilizerCode(n, [to_int(g, n) for g in gens])
        d = analyse(gens, n)[2]
        path = write_instance(code, tmp_path / "random.inst")
        done = run(binary, path, n, "--split", 1 + i % 3)
        assert done["found"] == d
        witness(code, done)
        below = run(binary, path, d - 1)
        assert below["found"] == -1
        assert below["nodes"] == python_nodes(code, d - 1)
        for domain, label in (("X", 1), ("Z", 2)):
            path = write_instance(code, tmp_path / "side.inst", domain)
            dside = side_distance(gens, n, label)
            done = run(binary, path, n)
            assert done["found"] == (-1 if dside is None else dside)
            if dside:
                v = witness(code, done)
                assert (v >> n == 0) if label == 1 else (v & ((1 << n) - 1) == 0)


def test_budget_resplit(binary, tmp_path):
    # A tensor product of five-qubit blocks has the same d with a larger tree.
    base = five()
    n = 10
    gens = []
    for offset in (0, 5):
        for row in base.generators:
            x, z = row & 31, row >> 5
            gens.append((x << offset) | (z << (n + offset)))
    code = StabilizerCode(n, gens)
    path = write_instance(code, tmp_path / "ten.inst")
    expected = run(binary, path, 2)
    done = run(binary, path, 2, "--cpu-budget", 1, "--delta", 1, "--chunk", 2)
    assert done["aborted_total"] > 0
    assert done["nodes"] == expected["nodes"]


def test_input_rejection(binary, tmp_path):
    code = five()
    path = write_instance(code, tmp_path / "bad.inst")
    for args, message in [(("--cap", 47), "DMAXW"), (("--threads", 0), "threads"), (("--delta", 0), "delta"), (("--gpu", 0), "without CUDA")]:
        p = run(binary, path, 2, *args, check=False)
        assert p.returncode == 2 and message in p.stderr
    path.write_text("1025 0 0\n")
    p = run(binary, path, 2, check=False)
    assert p.returncode == 2 and "NMAX" in p.stderr
    path = write_instance(code, path)
    path.write_text(path.read_text() + "5 7 7 7 7 7\n")
    p = run(binary, path, 2, check=False)
    assert p.returncode == 2 and "rootmask" in p.stderr


def test_each_input_limit(binary, tmp_path):
    path=tmp_path/'limits.inst'
    cases=[('1 1025 0\n','MMAX'),
           ('33 1 0\n33 '+' '.join(f'{q} 1' for q in range(33))+'\n','WMAX'),
           ('2 65 0\n'+('1 0 2\n'*65),'CMAX')]
    for text,limit in cases:
        path.write_text(text)
        result=run(binary,path,2,check=False)
        assert result.returncode==2 and limit in result.stderr
