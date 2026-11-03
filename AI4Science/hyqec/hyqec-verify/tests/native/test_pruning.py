"""Cut-level differential audit and regressions from the independent S1 review."""
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess

import pytest

from hyqec_verify import native
from hyqec_verify.core import StabilizerCode
from oracle import analyse, random_code, to_int
from .helpers import run, witness, write_instance

pytestmark = pytest.mark.native
# Every new shipped pruning policy must be added here AND to cut_audit.cpp.
POLICIES = ('none', 'greedy', 'greedy+b1b3', 'greedy+s0', 'greedy+s0s1', 'greedy+s0s1s2')
# Count-only parity rules (docs/SOUNDNESS.md section 5.1): CPU only.
GPU_POLICIES = ('none', 'greedy', 'greedy+b1b3')
# Cut reasons each policy must exercise on the random corpus.
RULE_CUTS = {'greedy+s0': ('s0',), 'greedy+s0s1': ('s0', 's1'), 'greedy+s0s1s2': ('s0', 's1', 's2')}
FIXTURES = Path(__file__).parent / 'fixtures' / 'pruning'
SOURCE = Path(__file__).with_name('cut_audit.cpp')


def read_code(path):
    """Read the preserved old input generators, including dependent rows."""
    tokens = iter(path.read_text().split())
    n, m, nl = (int(next(tokens)) for _ in range(3))
    def row():
        result = 0
        for _ in range(int(next(tokens))):
            q, label = int(next(tokens)), int(next(tokens))
            result |= (label & 1) << q | (label >> 1) << (n + q)
        return result
    generators = [row() for _ in range(m)]
    for _ in range(nl):
        row()
    assert [int(next(tokens)) for _ in range(n)] == list(range(n))
    assert int(next(tokens)) == n
    assert [int(next(tokens)) for _ in range(n)] == list(range(n))
    assert next(tokens, None) is None
    return StabilizerCode(n, generators)


def random_manifest(directory, cases=2000):
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(17820261022)
    lines = []
    for i in range(cases):
        n = 5 + i % 10  # Every size receives 200 independently sampled codes.
        k = rng.randint(1, min(4, n-1))
        rows = random_code(n, k, rng, depth=rng.randint(4*n, 16*n), extra_rows=i % 4)
        code = StabilizerCode(n, [to_int(row, n) for row in rows] + ([0] if i % 5 == 0 else []))
        domain = rng.choice(('P', 'X', 'Z'))
        path = write_instance(code, directory / f'random-{i:04d}.inst', domain)
        lines.append(f'{path} {rng.randrange(n+1)}\n')
    # These exact generator presentations exercise the historical cache failures.
    for name in ('lifted-cache-counterexample', 'cache-repro', 'chain'):
        code = read_code(FIXTURES / f'{name}.inst')
        for domain in ('P', 'X', 'Z'):
            path = write_instance(code, directory / f'{name}-{domain}.inst', domain)
            lines.append(f'{path} 3\n')
    # Preserved review inputs (the S1 reviewer differential trials, the S0 reviewer GAC cycle), every domain.
    provenance = {r['name']: r['sha256'] for r in json.loads((FIXTURES / 'provenance.json').read_text())}
    for source in sorted((FIXTURES / 'review').glob('*.inst')):
        assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance[f'review/{source.stem}']
        code = read_code(source)
        for domain in ('P', 'X', 'Z'):
            path = write_instance(code, directory / f'{source.stem}-{domain}.inst', domain)
            lines.append(f'{path} {rng.randrange(code.n+1)}\n')
    manifest = directory / 'manifest.txt'
    manifest.write_text(''.join(lines))
    return manifest


# 'capacity' shrinks every rule table (choices, local generators, occurrences, the anticommuting list and the S2
# hash table) so that each overflow / give-up path is taken on these small codes. With the scratch hook this is how
# a cleanup that misses entries on overflow shows up at n <= 14 (review finding 01 needed n > 2048 at full size).
AUDITOR_CONFIG = {}
CAPACITY_FLAGS = ['-DCSP_MAXCH=6', '-DCSP_MAXLG=8', '-DCSP_MAXOCC=16', '-DAMAX=6', '-DS2_HT=64']


@pytest.fixture(scope='module', params=['default', 'capacity'])
def auditor(tmp_path_factory, request):
    target = tmp_path_factory.mktemp('cut-audit') / 'audit'
    # The rules build (-DSTABCERT_RULES) contains every policy, including the default ones.
    flags = ['-O2', '-std=c++17', '-pthread', '-DNMAX=32', '-DMMAX=128', '-DSTABCERT_RULES=1']
    if request.param == 'capacity':
        flags += CAPACITY_FLAGS
    if os.environ.get('HYQEC_AUDIT_SANITIZE') == '1':
        flags += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    cmd = shlex.split(os.environ.get('CXX', 'c++')) + flags + [str(SOURCE), '-lz', '-o', str(target)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    AUDITOR_CONFIG[str(target)] = request.param
    return target


def test_differential_every_cut_on_2000_random_codes(auditor, tmp_path):
    manifest = random_manifest(tmp_path / 'instances')
    result = subprocess.run([str(auditor), str(manifest)], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (result.stdout, result.stderr)
    stats = json.loads(result.stdout)
    assert stats['cases'] == 2009 + 3 * len(list((FIXTURES / 'review').glob('*.inst')))
    assert stats['baseline_nodes'] > 0 and stats['scratch_checks'] > 0
    for policy in POLICIES:
        for reason in ('weight', 'dead', 'dead-suffix') + (() if policy == 'none' else ('packing',)):
            assert stats['cuts'].get(f'{policy}/{reason}', 0) > 0, stats
        for reason in RULE_CUTS.get(policy, ()):
            assert stats['cuts'].get(f'{policy}/{reason}', 0) > 0, stats
    (tmp_path / 'cut-summary.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, sort_keys=True))


def test_cut_auditor_detects_injected_unsafe_cut(auditor, tmp_path):
    path = write_instance(read_code(FIXTURES / 'lifted-cache-counterexample.inst'), tmp_path / 'lifted.inst')
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text(f'{path} 3\n')
    result = subprocess.run([str(auditor), str(manifest), '--inject-unsafe-cut'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and 'unsafe self-test' in result.stderr
    assert 'witness=' in result.stderr


@pytest.mark.parametrize('name', ['lifted-cache-counterexample', 'cache-repro', 'chain'])
def test_s1_reproducer_cpu(tmp_path, name):
    check_reproducer(tmp_path, name, native.build('bnb-cpu'), rules=native.build('bnb-cpu', variant='1024-rules'))


def check_reproducer(tmp_path, name, binary, gpu=False, rules=None):
    source = FIXTURES / f'{name}.inst'
    provenance = {r['name']: r for r in json.loads((FIXTURES / 'provenance.json').read_text())}
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance[name]['sha256']
    code = read_code(source)
    if name == 'lifted-cache-counterexample':
        # Independently described nontrivial weight-three operator Z15 X16 Z17.
        assert code.check_witness((1 << 16) | (1 << (code.n+15)) | (1 << (code.n+17))) == 3
        expected = 3  # Also proved by complete independent enumeration in the cut auditor.
    else:
        rows = [tuple(((g >> q) & 1) | (((g >> (code.n+q)) & 1) << 1)
                      for q in range(code.n)) for g in code.generators]
        expected = analyse(rows, code.n)[2]
    # The small original reproducers have nl=0; fill the validated normalizer
    # quotient basis while preserving every generator and the trivial root order.
    path = write_instance(code, tmp_path / f'{name}.inst')
    for policy in GPU_POLICIES if gpu else POLICIES:
        for split in (1, 2):
            args = ['--bound', policy, '--split', split, '--delta', 1]
            args += ['--gpu', 0, '--budget', 1] if gpu else ['--cpu-budget', 1]
            engine = rules if policy in RULE_CUTS else binary   # the S0/S1/S2 policies need the rules build
            found = run(engine, path, 3, *args)
            assert found['found'] == expected
            witness(code, found)
            lower = run(engine, path, expected-1, *args)
            assert lower['found'] == -1


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('HYQEC_TEST_GPU') != '1', reason='GPU opt-in required')
def test_s1_reproducers_gpu(tmp_path):
    binary = native.build('bnb-cuda')
    for name in ('lifted-cache-counterexample', 'cache-repro', 'chain'):
        check_reproducer(tmp_path, name, binary, gpu=True)
