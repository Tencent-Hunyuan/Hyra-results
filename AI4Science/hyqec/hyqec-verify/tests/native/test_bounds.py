"""Experimental residual-syndrome bounds, cross-checked against exhaustive codes."""
import os
import random

import pytest
from hyqec_verify import native
from hyqec_verify.core import StabilizerCode
from oracle import analyse, random_code, side_distance, to_int
from .helpers import run, write_instance, witness
from .test_bnb import five

pytestmark = pytest.mark.native


def test_extra_bounds_match_exhaustive(tmp_path):
    binary = native.build('bnb-cpu')
    rng = random.Random(530192)
    for case in range(40):
        n = rng.randrange(2, 8)
        rows = random_code(n, rng.randrange(1, n), rng, extra_rows=2)
        code = StabilizerCode(n, [to_int(row, n) for row in rows] + [0])
        for domain, label in (('P', 0), ('X', 1), ('Z', 2)):
            expected = analyse(rows, n)[2] if not label else side_distance(rows, n, label)
            path = write_instance(code, tmp_path/'input.inst', domain)
            found = run(binary, path, n, '--bound', 'greedy+b1b3')
            assert found['found'] == (-1 if expected is None else expected)
            if expected is not None:
                witness(code, found)
            cap = n if expected is None else expected-1
            normal = run(binary, path, cap)
            extra = run(binary, path, cap, '--bound', 'greedy+b1b3')
            assert extra['found'] == -1
            assert extra['nodes'] <= normal['nodes']
            split = run(binary, path, cap, '--bound', 'greedy+b1b3', '--split', 2,
                        '--cpu-budget', 1, '--delta', 1)
            assert split['nodes'] == extra['nodes']


def test_extra_bounds_cannot_claim_v1_certificate(tmp_path):
    binary = native.build('bnb-cpu')
    path = write_instance(five(), tmp_path/'five.inst')
    for arguments in (('--ckpt', tmp_path/'checkpoint'),
                      ('--emit-cert', tmp_path/'proof', '--cert-metadata', tmp_path/'metadata')):
        result = run(binary, path, 2, '--bound', 'greedy+b1b3', *arguments, check=False)
        assert result.returncode == 2 and 'incompatible' in result.stderr


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('HYQEC_TEST_GPU') != '1', reason='GPU opt-in required')
def test_extra_bounds_cpu_gpu_identity(tmp_path):
    cpu, gpu = native.build('bnb-cpu'), native.build('bnb-cuda')
    cases = [(five(), domain, cap) for domain, cap in (('P', 2), ('X', 4), ('Z', 4))]
    rng = random.Random(830751)
    for _ in range(12):
        n = rng.randrange(3, 9)
        rows = random_code(n, rng.randrange(1, n), rng, extra_rows=2)
        code = StabilizerCode(n, [to_int(row, n) for row in rows] + [0])
        for domain, label in (('P', 0), ('X', 1), ('Z', 2)):
            d = analyse(rows, n)[2] if not label else side_distance(rows, n, label)
            cases.append((code, domain, n if d is None else d - 1))
    for case, (code, domain, cap) in enumerate(cases):
        path = write_instance(code, tmp_path/'input.inst', domain)
        a = run(cpu, path, cap, '--bound', 'greedy+b1b3')
        # The fixed example exercises all split depths; random examples vary
        # the split and force nested budget refinements whenever possible.
        for split in ((1, 2, 3) if case < 3 else (1 + case % 3,)):
            b = run(gpu, path, cap, '--bound', 'greedy+b1b3', '--gpu', 0, '--split', split,
                    '--budget', 1, '--delta', 1)
            assert a['nodes'] == b['nodes'] and b['found'] == -1
