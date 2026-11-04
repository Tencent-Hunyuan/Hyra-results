"""Crash exactly at find-persistence boundaries, then resume the SAME binary."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

import pytest

from .helpers import witness
from .test_pruning import FIXTURES, read_code

pytestmark = pytest.mark.native
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src/hyqec_verify/native/bnb/bnb.cu'
HOOKS = Path(__file__).with_name('checkpoint_crash_hooks.h')
INSTANCE = FIXTURES / 'lifted-cache-counterexample.inst'
# HYQEC_CHECKPOINT_BOUND=greedy+s0s1s2 (etc.) reruns the whole suite under an opt-in pruning policy,
# with the crash binary compiled as the rules build.
BOUND = os.environ.get('HYQEC_CHECKPOINT_BOUND')
POINTS = (
    'find-detected', 'witness-captured', 'cpu-find-returned',
    'cpu-cap-published', 'cpu-find-copied', 'before-find-consume',
    'find-adopted', 'before-find-write', 'after-find-write',
    'after-find-flush', 'before-find-fsync', 'fsync-called',
    'after-find-fsync', 'find-committed', 'before-find-progress',
    'after-find-progress', 'before-find-done',
)


def build_crash_binary(directory, rules):
    binary = directory / 'bnb'
    command = shlex.split(os.environ.get('CXX', 'c++')) + [
        '-x', 'c++', '-O2', '-std=c++17', '-fopenmp', '-pthread',
        # The rules build also runs toric(5) (n = 50): refinement survives the S0/S1/S2 pruning there.
        f'-DNMAX={64 if rules else 32}', '-DMMAX=128', '-include', str(HOOKS),
        *(['-DSTABCERT_RULES=1'] if rules else []), str(SOURCE), '-lz', '-o', str(binary),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    return binary


@pytest.fixture(scope='module')
def crash_binary(tmp_path_factory):
    return build_crash_binary(tmp_path_factory.mktemp('checkpoint-crash'), bool(BOUND))


@pytest.fixture(scope='module')
def rules_crash_binary(tmp_path_factory):
    return build_crash_binary(tmp_path_factory.mktemp('checkpoint-crash-rules'), True)


@pytest.fixture(scope='module')
def gpu_crash_binary(tmp_path_factory):
    if os.environ.get('HYQEC_TEST_GPU') != '1':
        pytest.skip('GPU opt-in required')
    binary = tmp_path_factory.mktemp('checkpoint-crash-gpu') / 'bnb'
    command = shlex.split(os.environ.get('NVCC', 'nvcc')) + [
        '-O2', '-std=c++17', '-arch=native', '--expt-relaxed-constexpr', '-Xcompiler=-fopenmp,-pthread',
        '-DNMAX=32', '-DMMAX=128', '-include', str(HOOKS),
        str(SOURCE), '-lz', '-o', str(binary),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
    return binary


def execute(binary, ckpt, *, fault=None, threads=1, chunk=1, cap=3, split=1, budget=None, gpu=False, bound=None,
            instance=INSTANCE):
    # Only the runtime fault injection changes on resume. Both caller hashes
    # and the executable remain identical; a mismatch cannot mask a lost find.
    command = [str(binary), str(instance), '--cap', str(cap), '--split', str(split),
               '--threads', str(threads), '--chunk', str(chunk), '--ckpt', str(ckpt),
               '--instance-sha256', hashlib.sha256(Path(instance).read_bytes()).hexdigest(),
               '--binary-sha256', hashlib.sha256(binary.read_bytes()).hexdigest()]
    if budget is not None:
        command += ['--cpu-budget', str(budget), '--delta', '3']
    if bound or BOUND:
        command += ['--bound', bound or BOUND]
    if gpu:
        command += ['--gpu', '0', '--slots', '2']
    environment = {k: v for k, v in os.environ.items() if not k.startswith('STABCERT_TEST_')}
    environment.update(fault or {})
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
    if directory := os.environ.get('HYQEC_CHECKPOINT_AUDIT_DIR'):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / f'{time.monotonic_ns()}.json').write_text(json.dumps({
            'command': command, 'fault': fault, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr,
            'checkpoint': ckpt.read_text() if ckpt.exists() else None,
        }, indent=2) + '\n')
    return result


def records(result):
    return [json.loads(line) for line in result.stdout.splitlines()]


def assert_found(result, *, resumed=None):
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    done = [r for r in records(result) if r['event'] == 'done']
    assert len(done) == 1 and done[0]['found'] == 3 and done[0]['replay_fail'] == 0
    assert done[0]['stopped_on_find']
    if resumed is not None:
        assert done[0]['resumed'] is resumed
    witness(read_code(INSTANCE), done[0])
    assert sorted(done[0]['witness']) == [[15, 2], [16, 1], [17, 2]]


def checkpoint_lines(ckpt):
    data = ckpt.read_bytes()
    # A tail without a newline is uncommitted, even if it contains all numbers.
    return data, data.split(b'\n')[:-1]


def assert_no_finding_completion(ckpt, *, late=True):
    data, lines = checkpoint_lines(ckpt)
    completed = [line.split() for line in lines if line.startswith(b'C ')]
    if late:
        assert completed, data
    # The only weight-three logical has smallest support qubit15, labelZ.
    # At split1 its root task is 15*3+(2-1)=46. It must never receive C.
    assert not any(int(row[2]) <= 46 < int(row[3]) for row in completed), data
    finds = [i for i, line in enumerate(lines) if line.startswith(b'F ')]
    assert not finds or finds == [len(lines)-1], data


@pytest.mark.parametrize('threads,chunk', [(1, 1), (4, 8)])
@pytest.mark.parametrize('point', POINTS)
def test_kill_during_late_find(crash_binary, tmp_path, point, threads, chunk):
    ckpt = tmp_path / 'killed.ckpt'
    failed = execute(crash_binary, ckpt, threads=threads, chunk=chunk,
                     fault={'STABCERT_TEST_KILL_POINT': point})
    assert failed.returncode == -signal.SIGKILL, (point, failed.stdout, failed.stderr)
    assert f'CKPT_TEST {point}\n' in failed.stderr
    assert not any(r['event'] == 'done' for r in records(failed))
    assert_no_finding_completion(ckpt)
    resumed = execute(crash_binary, ckpt, threads=threads, chunk=chunk)
    assert_found(resumed, resumed=True)
    assert_no_finding_completion(ckpt)


@pytest.mark.parametrize('point', POINTS)
def test_kill_during_late_find_rule_policy(rules_crash_binary, tmp_path, point):
    # The policy is bound in the header, so the resume must repeat it (docs/SOUNDNESS.md 5.1).
    ckpt = tmp_path / 'killed.ckpt'
    failed = execute(rules_crash_binary, ckpt, threads=4, chunk=8, bound='greedy+s0s1s2',
                     fault={'STABCERT_TEST_KILL_POINT': point})
    assert failed.returncode == -signal.SIGKILL, (point, failed.stdout, failed.stderr)
    assert_no_finding_completion(ckpt)
    assert_found(execute(rules_crash_binary, ckpt, threads=4, chunk=8, bound='greedy+s0s1s2'), resumed=True)
    assert_no_finding_completion(ckpt)


def test_every_partial_find_write(crash_binary, tmp_path):
    baseline = tmp_path / 'complete.ckpt'
    assert_found(execute(crash_binary, baseline))
    final = baseline.read_bytes().splitlines(keepends=True)[-1]
    assert final.startswith(b'F 3 ') and final.endswith(b'\n')
    for count in range(len(final)+1):
        ckpt = tmp_path / f'partial-{count}.ckpt'
        result = execute(crash_binary, ckpt, fault={
            'STABCERT_TEST_KILL_POINT': 'partial-find-write',
            'STABCERT_TEST_PARTIAL_FIND_BYTES': str(count),
        })
        assert result.returncode == -signal.SIGKILL, (count, result.stderr)
        assert ckpt.read_bytes() == baseline.read_bytes()[:-len(final)] + final[:count]
        assert_no_finding_completion(ckpt)
        assert_found(execute(crash_binary, ckpt), resumed=True)
        assert ckpt.read_bytes() == baseline.read_bytes()


def test_find_fsync_order_and_failure(crash_binary, tmp_path):
    ckpt = tmp_path / 'find.ckpt'
    result = execute(crash_binary, ckpt)
    assert_found(result)
    events = [line.removeprefix('CKPT_TEST ') for line in result.stderr.splitlines()]
    assert [e for e in events if e in POINTS] == list(POINTS)
    assert_no_finding_completion(ckpt)
    for saved in (False, True):
        target = tmp_path / f'failure-{saved}.ckpt'
        if saved:
            target.write_bytes(ckpt.read_bytes())
        failure = execute(crash_binary, target, fault={'STABCERT_TEST_FSYNC_FAULT': 'EIO'})
        assert failure.returncode == 1 and 'checkpoint fsync failed' in failure.stderr
        assert not any(r['event'] == 'done' or r.get('found', -1) > 0 for r in records(failure))
        assert 'CKPT_TEST after-find-fsync' not in failure.stderr
        assert_found(execute(crash_binary, target), resumed=True)
    interrupted = execute(crash_binary, tmp_path / 'eintr.ckpt',
                          fault={'STABCERT_TEST_FSYNC_FAULT': 'EINTR'})
    assert_found(interrupted)
    assert interrupted.stderr.count('CKPT_TEST fsync-called\n') == 2


def test_complete_find_rejects_later_completion(crash_binary, tmp_path):
    ckpt = tmp_path / 'find-then-completion.ckpt'
    assert_found(execute(crash_binary, ckpt))
    invalid = ckpt.read_bytes() + b'C 0 46 47 1 1 0 0\n'
    ckpt.write_bytes(invalid)
    result = execute(crash_binary, ckpt)
    assert result.returncode == 6 and 'records after final find' in result.stderr
    assert ckpt.read_bytes() == invalid
    assert not any(r['event'] == 'done' for r in records(result))


@pytest.mark.parametrize('split,budget', [(4, None), (1, 1)])
@pytest.mark.parametrize('point', ['find-detected', 'before-find-write', 'after-find-flush', 'after-find-fsync'])
def test_generation_and_refinement_find_crashes(crash_binary, tmp_path, split, budget, point):
    ckpt = tmp_path / 'expanded.ckpt'
    result = execute(crash_binary, ckpt, cap=4, split=split, budget=budget,
                     fault={'STABCERT_TEST_KILL_POINT': point})
    assert result.returncode == -signal.SIGKILL, (result.stdout, result.stderr)
    assert_found(execute(crash_binary, ckpt, cap=4, split=split, budget=budget))


@pytest.mark.gpu
def test_gpu_checkpoint_find_crashes(gpu_crash_binary, tmp_path):
    # Device findings are copied and independently reproduced on the CPU before
    # consume(). Exercise that handoff and the common locked persistence path.
    points = ('gpu-find-returned', 'gpu-cap-published', 'gpu-find-copied') + POINTS
    for point in points:
        ckpt = tmp_path / f'{point}.ckpt'
        result = execute(gpu_crash_binary, ckpt, gpu=True,
                         fault={'STABCERT_TEST_KILL_POINT': point})
        assert result.returncode == -signal.SIGKILL, (point, result.stdout, result.stderr)
        assert_no_finding_completion(ckpt, late=False)
        assert_found(execute(gpu_crash_binary, ckpt, gpu=True), resumed=True)
        assert_no_finding_completion(ckpt, late=False)


IO_FAULTS = ('short-write', 'write-ENOSPC', 'flush-EIO', 'flush-ENOSPC',
             'fsync-EIO', 'fsync-ENOSPC', 'close-EIO', 'close-ENOSPC')
CHUNK_POINTS = ('before-chunk-write', 'after-chunk-write', 'after-chunk-flush',
                'before-chunk-fsync', 'chunk-fsync-called', 'after-chunk-fsync')


def assert_no_find(result, expected=None):
    assert result.returncode == 0, (result.stdout, result.stderr)
    done, = [r for r in records(result) if r['event'] == 'done']
    assert done['found'] == -1 and done['replay_fail'] == 0
    if expected is not None:
        for key in ('nodes', 'gen_nodes', 'task_nodes', 'rounds', 'tasks_total', 'aborted_total'):
            assert done[key] == expected[key], (key, done, expected)
    return done


@pytest.mark.parametrize('kind,cap,fault', [
    (kind, cap, fault) for kind, cap in [('C', 2), ('C', 3), ('F', 3)] for fault in IO_FAULTS
    if not (kind == 'C' and cap == 3 and fault.startswith('close-'))
])
def test_record_io_errors(crash_binary, tmp_path, kind, cap, fault):
    ckpt = tmp_path / 'io-error.ckpt'
    baseline = execute(crash_binary, tmp_path / 'baseline.ckpt', cap=cap)
    result = execute(crash_binary, ckpt, cap=cap, fault={
        'STABCERT_TEST_IO_KIND': kind, 'STABCERT_TEST_IO_FAULT': fault})
    assert result.returncode == 1, (fault, result.stdout, result.stderr)
    operation = 'write' if fault == 'short-write' else fault.split('-')[0]
    assert f'checkpoint {operation} failed' in result.stderr
    assert not any(r['event'] == 'done' for r in records(result))
    if not fault.startswith('close-'):
        assert not any(r.get('found', -1) > 0 for r in records(result))
    resumed = execute(crash_binary, ckpt, cap=cap)
    if cap == 3:
        assert_found(resumed, resumed=True)
        assert_no_finding_completion(ckpt)
    else:
        assert_no_find(resumed, assert_no_find(baseline))


def refinement_case(tmp_path, bound):
    """(instance, cap) with budget-driven refinement and no find. The S0/S1/S2 policies prune the
    [[18,1,3]] fixture so far that no cap-2 task exceeds budget 1; toric(5) at cap 4 still refines."""
    if not bound:
        return INSTANCE, 2
    from known_codes import toric
    from .helpers import write_instance
    return write_instance(toric(5), tmp_path / 'toric5.inst'), 4


def chunk_crash_with_refinement(binary, tmp_path, point, bound=None):
    instance, cap = refinement_case(tmp_path, bound or BOUND)
    run = dict(cap=cap, budget=1, chunk=8, bound=bound, instance=instance)
    baseline = execute(binary, tmp_path / 'baseline.ckpt', **run)
    expected = assert_no_find(baseline)
    assert expected['aborted_total'] > 0
    ckpt = tmp_path / 'chunk.ckpt'
    result = execute(binary, ckpt, **run, fault={'STABCERT_TEST_KILL_POINT': point})
    assert result.returncode == -signal.SIGKILL, result.stderr
    assert_no_find(execute(binary, ckpt, **run), expected)


@pytest.mark.parametrize('point', CHUNK_POINTS)
def test_chunk_crashes_with_refinement(crash_binary, tmp_path, point):
    chunk_crash_with_refinement(crash_binary, tmp_path, point)


@pytest.mark.parametrize('point', CHUNK_POINTS)
def test_chunk_crashes_with_refinement_rule_policy(rules_crash_binary, tmp_path, point):
    chunk_crash_with_refinement(rules_crash_binary, tmp_path, point, bound='greedy+s0s1s2')


def test_chunk_sync_count_retry_and_resume_failure(crash_binary, tmp_path):
    ckpt = tmp_path / 'chunks.ckpt'
    first = execute(crash_binary, ckpt, cap=2, budget=1, chunk=8,
                    fault={'STABCERT_TEST_IO_KIND': 'C', 'STABCERT_TEST_IO_FAULT': 'fsync-EINTR'})
    expected = assert_no_find(first)
    count = sum(line.startswith(b'C ') for line in ckpt.read_bytes().splitlines())
    assert first.stderr.count('CKPT_TEST chunk-fsync-called\n') == count + 1
    assert first.stderr.count('CKPT_TEST after-chunk-fsync\n') == count
    original = ckpt.read_bytes()
    for fault in ('fsync-EIO', 'fsync-ENOSPC', 'close-EIO', 'close-ENOSPC'):
        failed = execute(crash_binary, ckpt, cap=2, budget=1, chunk=8,
                         fault={'STABCERT_TEST_IO_KIND': 'C', 'STABCERT_TEST_IO_FAULT': fault})
        assert failed.returncode == 1, failed.stderr
        assert not any(r['event'] == 'done' for r in records(failed))
        assert ckpt.read_bytes() == original
    assert_no_find(execute(crash_binary, ckpt, cap=2, budget=1, chunk=8), expected)


def test_every_partial_chunk_write(crash_binary, tmp_path):
    baseline = tmp_path / 'complete.ckpt'
    expected = assert_no_find(execute(crash_binary, baseline, cap=2, budget=1, chunk=8))
    header, first, *_ = baseline.read_bytes().splitlines(keepends=True)
    assert first.startswith(b'C ')
    for count in range(len(first) + 1):
        ckpt = tmp_path / f'partial-chunk-{count}.ckpt'
        result = execute(crash_binary, ckpt, cap=2, budget=1, chunk=8, fault={
            'STABCERT_TEST_KILL_POINT': 'partial-chunk-write',
            'STABCERT_TEST_PARTIAL_CHUNK_BYTES': str(count)})
        assert result.returncode == -signal.SIGKILL, (count, result.stderr)
        assert ckpt.read_bytes() == header + first[:count]
        assert_no_find(execute(crash_binary, ckpt, cap=2, budget=1, chunk=8), expected)
        assert ckpt.read_bytes() == baseline.read_bytes()


def test_every_header_prefix_restarts(crash_binary, tmp_path):
    baseline = tmp_path / 'complete.ckpt'
    expected = assert_no_find(execute(crash_binary, baseline, cap=2, chunk=8))
    header = baseline.read_bytes().splitlines(keepends=True)[0]
    for count in range(len(header)):
        ckpt = tmp_path / f'partial-header-{count}.ckpt'
        result = execute(crash_binary, ckpt, cap=2, chunk=8, fault={
            'STABCERT_TEST_KILL_POINT': 'partial-header-write',
            'STABCERT_TEST_PARTIAL_HEADER_BYTES': str(count)})
        assert result.returncode == -signal.SIGKILL, (count, result.stderr)
        assert ckpt.read_bytes() == header[:count]
        done = assert_no_find(execute(crash_binary, ckpt, cap=2, chunk=8), expected)
        assert done['resumed'] is False
        assert ckpt.read_bytes() == baseline.read_bytes()
        # A non-prefix must stay untouched even if its line is incomplete.
        invalid = header[:count] + b'?'
        ckpt.write_bytes(invalid)
        failed = execute(crash_binary, ckpt, cap=2, chunk=8)
        assert failed.returncode == 6 and ckpt.read_bytes() == invalid


@pytest.mark.gpu
@pytest.mark.parametrize('kind,cap', [('C', 2), ('F', 3)])
def test_gpu_checkpoint_io_errors(gpu_crash_binary, tmp_path, kind, cap):
    expected = execute(gpu_crash_binary, tmp_path / 'baseline.ckpt', cap=cap, gpu=True)
    for fault in IO_FAULTS:
        ckpt = tmp_path / f'{fault}.ckpt'
        result = execute(gpu_crash_binary, ckpt, cap=cap, gpu=True, fault={
            'STABCERT_TEST_IO_KIND': kind, 'STABCERT_TEST_IO_FAULT': fault})
        assert result.returncode == 1, (fault, result.stdout, result.stderr)
        assert not any(r['event'] == 'done' for r in records(result))
        resumed = execute(gpu_crash_binary, ckpt, cap=cap, gpu=True)
        if cap == 3:
            assert_found(resumed, resumed=True)
            assert_no_finding_completion(ckpt, late=False)
        else:
            assert_no_find(resumed, assert_no_find(expected))
