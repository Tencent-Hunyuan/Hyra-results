"""Opt-in count-only parity rules S0/S1/S2 (--bound greedy+s0, greedy+s0s1, greedy+s0s1s2).

Soundness of every individual cut is audited in test_pruning.py; this file checks the end-to-end
behaviour, the node-level review reproducers, and the policy plumbing (CPU only, no coverage
certificate, checkpoint binding)."""
import json
import os
from pathlib import Path
import random
import shlex
import sys
import subprocess

import pytest
from hyqec_verify import native
from hyqec_verify.core import StabilizerCode
from oracle import analyse, random_code, side_distance, to_int
from .helpers import run, write_instance, witness
from .test_bnb import five
from .test_checkpoint import assert_rejected_untouched

pytestmark = pytest.mark.native
RULES = ('greedy+s0', 'greedy+s0s1', 'greedy+s0s1s2')
SOURCE = Path(__file__).with_name('rule_cases.cpp')
CASES = {'s2-extra-only-rows-pair', 's2-extra-only-rows-blocked', 's2-triple-anticommuting', 's2-triple-blocked',
         's1-stale-cache-node', 's0-gac-incomplete-packing', 's0-gac-incomplete', 's0-mis-threshold'}


@pytest.fixture(scope='module')
def binary():
    return native.build('bnb-cpu', variant='1024-rules')


def test_review_node_cases(tmp_path):
    target = tmp_path / 'rule-cases'
    flags = ['-O2', '-std=c++17', '-pthread', '-DNMAX=32', '-DMMAX=128', '-DSTABCERT_RULES=1']
    if os.environ.get('HYQEC_AUDIT_SANITIZE') == '1':
        flags += ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    cmd = shlex.split(os.environ.get('CXX', 'c++')) + flags + [str(SOURCE), '-lz', '-o', str(target)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    result = subprocess.run([str(target), str(tmp_path)], capture_output=True, text=True, timeout=60)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 0 and all(r['ok'] for r in rows), (result.stdout, result.stderr)
    assert {r['case'] for r in rows} == CASES
    by = {r['case']: r for r in rows}
    # Documented limits (docs/SOUNDNESS.md 5.1): GAC misses the XOR cycle, mis_pack is a threshold.
    assert by['s0-gac-incomplete']['completions'] == 0 and by['s0-gac-incomplete']['why'] == [0, 0, 0]
    assert (by['s0-mis-threshold']['mis_pack'], by['s0-mis-threshold']['maximum']) == (3, 4)
    assert by['s1-stale-cache-node']['completions'] > 0 and by['s1-stale-cache-node']['s1_calls'] > 0


def test_rules_match_exhaustive(binary, tmp_path):
    rng = random.Random(260261026)
    for case in range(40):
        n = rng.randrange(3, 9)
        rows = random_code(n, rng.randrange(1, n), rng, extra_rows=2)
        code = StabilizerCode(n, [to_int(row, n) for row in rows] + [0])
        for domain, label in (('P', 0), ('X', 1), ('Z', 2)):
            expected = analyse(rows, n)[2] if not label else side_distance(rows, n, label)
            path = write_instance(code, tmp_path / 'input.inst', domain)
            cap = n if expected is None else expected - 1
            greedy = run(binary, path, cap)
            for policy in RULES:
                found = run(binary, path, n, '--bound', policy)
                assert found['found'] == (-1 if expected is None else expected)
                if expected is not None:
                    witness(code, found)
                proof = run(binary, path, cap, '--bound', policy)
                assert proof['found'] == -1 and proof['nodes'] <= greedy['nodes']
                split = run(binary, path, cap, '--bound', policy, '--split', 1 + case % 3,
                            '--cpu-budget', 1, '--delta', 1)
                assert split['found'] == -1 and split['nodes'] == proof['nodes']


def test_rules_only_in_rules_build(binary, tmp_path):
    # Default builds compile without the rules (they preprocess to the unchanged search) and refuse them.
    default = native.build('bnb-cpu')
    assert 'RULES' not in native.version(default)['limits']
    assert native.version(binary)['limits']['RULES'] == 's0s1s2/1'
    path = write_instance(five(), tmp_path / 'five.inst')
    for policy in RULES:
        refused = run(default, path, 2, '--bound', policy, check=False)
        assert refused.returncode == 2 and '--bound must be' in refused.stderr
    for target in ('bnb-cuda', 'ris-cpu'):
        with pytest.raises(ValueError, match='-rules'):
            native.build(target, variant='1024-rules')
    greedy = run(default, path, 2)
    assert run(binary, path, 2)['nodes'] == greedy['nodes']


@pytest.mark.parametrize('policy', RULES)
def test_rules_are_cpu_only_and_count_only(binary, tmp_path, policy):
    code = five()
    path = write_instance(code, tmp_path / 'five.inst')
    gpu = run(binary, path, 2, '--bound', policy, '--gpu', 0, check=False)
    assert gpu.returncode == 2 and 'CPU-only' in gpu.stderr
    meta = tmp_path / 'meta.json'
    meta.write_text(json.dumps({'group_sha256': code.group_hash(), 'permutations': []}))
    cert = run(binary, path, 2, '--bound', policy, '--emit-cert', tmp_path / 'proof.cov',
               '--cert-metadata', meta, check=False)
    assert cert.returncode == 2 and 'incompatible' in cert.stderr
    assert not (tmp_path / 'proof.cov').exists()


@pytest.mark.parametrize('policy', RULES)
def test_checkpoint_binds_rule_policy(binary, tmp_path, policy):
    path = write_instance(five(), tmp_path / 'five.inst')
    plain = tmp_path / 'greedy.ckpt'
    run(binary, path, 2, '--ckpt', plain)
    assert b'bound=' not in plain.read_bytes().split(b'\n', 1)[0]   # default header unchanged
    assert_rejected_untouched(binary, path, 2, plain, '--bound', policy)
    ckpt = tmp_path / 'rule.ckpt'
    a = run(binary, path, 2, '--ckpt', ckpt, '--bound', policy, '--cpu-budget', 1, '--delta', 1, '--chunk', 1)
    assert f' bound={policy}'.encode() in ckpt.read_bytes().split(b'\n', 1)[0]
    b = run(binary, path, 2, '--ckpt', ckpt, '--bound', policy, '--cpu-budget', 1, '--delta', 1, '--chunk', 1)
    assert b['resumed']
    for key in ('nodes', 'gen_nodes', 'task_nodes', 'rounds', 'tasks_total', 'aborted_total'):
        assert a[key] == b[key]
    for other in ('greedy',) + tuple(p for p in RULES if p != policy):
        assert_rejected_untouched(binary, path, 2, ckpt, '--cpu-budget', 1, '--delta', 1, '--chunk', 1,
                                  *(() if other == 'greedy' else ('--bound', other)))


@pytest.mark.parametrize('variant,n', [('1024-w64', 64), ('1024-w128', 96), ('4096-w128', 129)])
def test_rules_on_wide_variants(tmp_path, variant, n):
    # Dense rows give up to 2*WMAX values per packed variable (the [port] uint16 width of na[]).
    from .test_wide import dense_five
    binary = native.build('bnb-cpu', variant=f'{variant}-rules')
    code = dense_five(n)
    for domain, d in (('P', 3), ('X', 5), ('Z', 5)):
        path = write_instance(code, tmp_path / 'wide.inst', domain)
        greedy = run(binary, path, d - 1, '--split', 1)
        for policy in RULES:
            proof = run(binary, path, d - 1, '--split', 1, '--bound', policy)
            split = run(binary, path, d - 1, '--split', 2, '--cpu-budget', 1, '--delta', 1, '--bound', policy)
            assert proof['found'] == split['found'] == -1
            assert split['nodes'] == proof['nodes'] <= greedy['nodes']
            found = run(binary, path, d, '--split', 2, '--bound', policy)
            assert found['found'] == d
            witness(code, found)


def test_rules_on_large_variant(tmp_path):
    # n > 1365 also exercises the S2 size guard (it returns "no prune" instead of using its table).
    n = 1500
    rows = [(g & 31) << (n - 5) | (g >> 5) << (2 * n - 5) for g in five().generators] + [1 << (n + q) for q in range(n - 5)]
    code = StabilizerCode(n, rows)
    binary = native.build('bnb-cpu', variant='4096-rules')
    path = write_instance(code, tmp_path / 'large.inst')
    greedy = run(binary, path, 2)
    for policy in RULES:
        proof = run(binary, path, 2, '--bound', policy, '--threads', 4)
        assert proof['found'] == -1 and proof['nodes'] <= greedy['nodes']
        found = run(binary, path, 3, '--bound', policy, '--threads', 4)
        assert found['found'] == 3
        witness(code, found)


def test_rules_build_checkpoint_diagnostics(binary, tmp_path):
    # The rules build names the bound field and the policies its version-2 checkpoints accept;
    # the default build keeps its messages (and executable) unchanged.
    path = write_instance(five(), tmp_path / 'five.inst')
    ckpt = tmp_path / 'greedy.ckpt'
    run(binary, path, 2, '--ckpt', ckpt)
    for policy in ('none', 'greedy+b1b3'):
        fresh = run(binary, path, 2, '--ckpt', tmp_path / f'{policy}.ckpt', '--bound', policy, check=False)
        assert fresh.returncode == 2 and 'none and greedy+b1b3 are incompatible' in fresh.stderr
        resumed = run(binary, path, 2, '--ckpt', ckpt, '--bound', policy, check=False)
        assert resumed.returncode == 6
        assert 'requires greedy or greedy+s0/greedy+s0s1/greedy+s0s1s2' in resumed.stderr
    mismatch = run(binary, path, 2, '--ckpt', ckpt, '--bound', 'greedy+s0', check=False)
    assert mismatch.returncode == 6 and 'gen_tasks/bound)' in mismatch.stderr
    default = native.build('bnb-cpu')
    plain = tmp_path / 'default.ckpt'
    run(default, path, 2, '--ckpt', plain)
    refused = run(default, path, 2, '--ckpt', plain, '--bound', 'none', check=False)
    assert refused.returncode == 6 and 'bound policy mismatch (version 2 requires greedy)' in refused.stderr


@pytest.mark.parametrize('wmax,blocks,width', [(128, 17, 128), (128, 16, 128), (128, 31, 128),
                                               (64, 32, 64), (64, 33, 64), (128, 33, 64)])
def test_wide_s1_filter_cleanup_regression(tmp_path, wmax, blocks, width):
    """review finding 01 and its boundary sweep: at the 4096-w64/w128 limits the S1 filter must clean every touched
    entry, with exactly 2048 touched qubits (16x128, 32x64) and beyond (33x64, 17x128, 31x128)."""
    target = tmp_path / 'wide-s1'
    cmd = shlex.split(os.environ.get('CXX', 'c++')) + [
        '-O2', '-std=c++17', '-pthread', '-DNMAX=4096', '-DMMAX=4096', f'-DWMAX={wmax}', '-DCMAX=128',
        str(Path(__file__).with_name('wide_s1_regression.cpp')), '-lz', '-o', str(target)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    result = subprocess.run([str(target), str(tmp_path / 'wide-s1.inst'), str(blocks), str(width)],
                            capture_output=True, text=True, timeout=120)
    row = json.loads(result.stdout)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert row['n'] == 1 + blocks * width and row['W'] == width and row['commutes']
    assert row['second_cut'] == row['fresh_cut'] == 0 and row['stale_after_first'] == row['stale_after_second'] == 0


WIDE_FLAGS = ['-O2', '-std=c++17', '-pthread', '-DNMAX=4096', '-DMMAX=4096', '-DWMAX=128', '-DCMAX=128']


def _build_wide(source, target):
    cmd = shlex.split(os.environ.get('CXX', 'c++')) + WIDE_FLAGS + [str(source), '-lz', '-o', str(target)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    return target


def test_large_n_rule_state_reuse(tmp_path):
    """Large-n stratum (n up to 2401, rows of weight 60-128): one reused search state vs a fresh state on random
    node sequences; decisions must agree and the cross-call scratch must be clean after every call."""
    binary = _build_wide(Path(__file__).with_name('rule_reuse_fuzz.cpp'), tmp_path / 'fuzz')
    total = {}
    # Seeds fixed; S2 prunes are rare in this generator, and 14 / 29 give several hundred (all decisions S0-S2 run).
    for seed in (7, 14, 29):
        result = subprocess.run([str(binary), str(tmp_path), str(seed), '3000'], capture_output=True, text=True,
                                timeout=600)
        stats = json.loads(result.stdout)
        assert result.returncode == 0 and stats['mismatches'] == 0 and stats['dirty_calls'] == 0, (seed, stats,
                                                                                                    result.stderr)
        for k, v in stats['results'].items():
            total[k] = total.get(k, 0) + v
    assert sum(total.values()) > 200000 and all(total[k] > 0 for k in ('packing', 's0', 's1', 's2')), total


def test_cut_audit_large_codes(tmp_path):
    """Large-n stratum of the cut audit (n > 2048, no enumeration oracle): every cut is re-searched without the
    rule from rebuilt state, and each policy's distance equals the reviewed greedy search's."""
    from known_codes import toric
    binary = _build_wide(Path(__file__).with_name('cut_audit.cpp'), tmp_path / 'audit-large')
    blocks, width = 17, 128
    n = 1 + blocks * width
    z = lambda qs: sum(1 << (n + q) for q in qs)
    rows = [z([0] + [1 + i * width for i in range(blocks)])] * 2
    rows += [z(range(1 + i * width, 1 + (i + 1) * width)) for i in range(blocks)]
    family = StabilizerCode(n, rows)
    lines = []
    for name, code in (('family', family), ('toric33', toric(33))):
        assert code.n > 2048
        for domain in ('P', 'X'):
            path = write_instance(code, tmp_path / f'{name}-{domain}.inst', domain)
            lines += [f'{path} 2\n', f'{path} 3\n']
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text(''.join(lines))
    result = subprocess.run([str(binary), str(manifest), '--large'], capture_output=True, text=True, timeout=1200)
    assert result.returncode == 0, (result.stdout, result.stderr[-2000:])
    stats = json.loads(result.stdout)
    assert stats['cases'] == len(lines) and stats['scratch_checks'] > 0
    assert any(k.endswith('/s0') for k in stats['cuts']), stats


# ---- every buffer-size regime of every rules build variant (owner requirement 3; docs/SOUNDNESS.md 5.1) ----
VARIANTS = [(1024, 32, 64), (1024, 64, 128), (1024, 128, 128), (4096, 32, 64), (4096, 64, 128), (4096, 128, 128)]
# Capacity limits of s012.inc / s2.inc (STABCERT_RULE_LIMIT names) that can occur in a variant. Not reachable:
# NMAX 1024 cannot exceed 2048 touched qubits or the S2 table (n <= 1365), MMAX 1024 cannot exceed CSP_MAXLG = 2048
# local generators; s2_h_range (h <= 46 < DMAX) and s2_empty_domain (a live packed row has a candidate) never.
LIMITS_1024 = {'gac_maxch', 'gac_maxocc', 'greedy_amax', 'kil_maxch', 'mis_nA_gt_64', 'need_amax',
               's1_bitmask_h_gt_31', 's1_k2_maxocc', 's1_records_maxch', 'zl_amax'}
LIMITS_4096 = LIMITS_1024 | {'gac_maxlg', 's1_touched_gt_2048', 's2_table'}


def _variant_id(v):
    return f'{v[0]}-w{v[1]}'


def _build_variant(source, target, variant):
    nmax, wmax, cmax = variant
    flags = ['-O2', '-std=c++17', '-pthread', f'-DNMAX={nmax}', f'-DMMAX={nmax}', f'-DWMAX={wmax}', f'-DCMAX={cmax}']
    cmd = shlex.split(os.environ.get('CXX', 'c++')) + flags + [str(source), '-lz', '-o', str(target)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    return target


def _planted(tmp_path, variant, seeds, cases):
    binary = _build_variant(Path(__file__).with_name('rule_planted.cpp'), tmp_path / 'planted', variant)
    hit, calls = set(), 0
    for seed in seeds:
        result = subprocess.run([str(binary), str(tmp_path), str(seed), str(cases)], capture_output=True, text=True,
                                timeout=1800)
        stats = json.loads(result.stdout)
        assert result.returncode == 0, (seed, stats, result.stderr[-2000:])
        assert stats['false_prunes'] == stats['mismatches'] == stats['dirty_calls'] == 0, stats
        hit |= set(stats['limits']); calls += stats['calls']
    return hit, calls


@pytest.mark.parametrize('variant', VARIANTS, ids=_variant_id)
def test_planted_stratum_quick(tmp_path, variant):
    """Planted-completion nodes at full buffer sizes (no prune allowed), reused vs fresh state, clean scratch."""
    hit, calls = _planted(tmp_path, variant, [11], 60)
    assert calls > 400 and hit


@pytest.mark.slow
@pytest.mark.parametrize('variant', VARIANTS, ids=_variant_id)
def test_planted_stratum_every_limit(tmp_path, variant):
    """As above, three seeds: every capacity limit reachable in the variant must be hit at full size."""
    hit, calls = _planted(tmp_path, variant, [11, 12, 13], 400)
    assert hit == (LIMITS_4096 if variant[0] == 4096 else LIMITS_1024), sorted(hit)
    assert calls > 8000


def _block_family(blocks, width):
    n = 1 + blocks * width
    z = lambda qs: sum(1 << (n + q) for q in qs)
    rows = [z([0] + [1 + i * width for i in range(blocks)])] * 2
    rows += [z(range(1 + i * width, 1 + (i + 1) * width)) for i in range(blocks)]
    return StabilizerCode(n, rows)


LARGE_CASES = {   # (blocks, width) of the rules reviewer's family that fit the variant's WMAX, and a toric code near the size limit
    (1024, 32, 64): ([(31, 32)], 22), (1024, 64, 128): ([(15, 64)], 22), (1024, 128, 128): ([(7, 128)], 22),
    (4096, 32, 64): ([], 33), (4096, 64, 128): ([(33, 64)], 33), (4096, 128, 128): ([(17, 128), (31, 128)], 33),
}


@pytest.mark.slow
@pytest.mark.parametrize('variant', VARIANTS, ids=_variant_id)
def test_cut_audit_large_every_variant(tmp_path, variant):
    """Search-level large stratum in every variant: every cut of greedy+s0/s0s1/s0s1s2 re-searched without the rule
    from rebuilt state, distances equal to the reviewed greedy search, scratch clean after every rule pass."""
    from known_codes import toric
    binary = _build_variant(Path(__file__).with_name('cut_audit.cpp'), tmp_path / 'audit', variant)
    families, L = LARGE_CASES[variant]
    codes = [(f'family{b}x{w}', _block_family(b, w)) for b, w in families] + [(f'toric{L}', toric(L))]
    lines = []
    for name, code in codes:
        assert code.n <= variant[0]
        for domain in ('P', 'X'):
            path = write_instance(code, tmp_path / f'{name}-{domain}.inst', domain)
            lines += [f'{path} 2\n', f'{path} 3\n']
    manifest = tmp_path / 'manifest.txt'
    manifest.write_text(''.join(lines))
    result = subprocess.run([str(binary), str(manifest), '--large'], capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, (result.stdout, result.stderr[-2000:])
    stats = json.loads(result.stdout)
    assert stats['cases'] == len(lines) and stats['scratch_checks'] > 0, stats


def test_scratch_inventory_covers_every_member():
    """Owner requirement 1: every data member of s012.inc / S2Scratch is covered by SCRATCH_AUDIT.md."""
    script = Path(__file__).parent / 'port' / 'scratch_inventory.py'
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout


def test_high_guard_cases(tmp_path):
    """the rules reviewer's deterministic boundary cases (h = 32, |A_v| = 97, S2 at n = 1365 / 1366, full need / zero lists):
    witnessed nodes must not be pruned, repeated calls leave clean scratch, the named guard fires (or not)."""
    binary = _build_variant(Path(__file__).with_name('high_guard_cases.cpp'), tmp_path / 'high-guard', (4096, 128, 128))
    result = subprocess.run([str(binary), str(tmp_path)], capture_output=True, text=True, timeout=300)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 0 and len(rows) == 7 and all(r['ok'] for r in rows), result.stdout
