import hashlib
import json
import subprocess
import zlib
from pathlib import Path
import pytest
from hyqec_verify import native
from hyqec_verify.core import StabilizerCode
from .helpers import write_instance, run, witness
from .test_bnb import five

pytestmark = pytest.mark.native

@pytest.fixture(scope="module")
def binary():
    return native.build("bnb-cpu")


def assert_rejected_untouched(binary, instance, cap, ckpt, *args):
    original = ckpt.read_bytes()
    result = run(binary, instance, cap, '--ckpt', ckpt, *args, check=False)
    assert result.returncode == 6, (result.returncode, result.stdout, result.stderr)
    assert 'checkpoint rejected' in result.stderr
    assert not any(json.loads(line).get('event') == 'done' for line in result.stdout.splitlines())
    assert ckpt.read_bytes() == original


@pytest.mark.parametrize('cap', [3, 4])
def test_adversarial_five_qubit_cap_escalation(binary, tmp_path, cap):
    """The legacy cap-2 resume counterexample must never certify d >= 4 or 5."""
    path = Path(__file__).parent / 'fixtures' / 'legacy-resume-five.inst'
    ckpt = tmp_path / 'cap2.ckpt'
    first = run(binary, path, 2, '--split', 1, '--ckpt', ckpt)
    assert first['found'] == -1 and first['nodes'] == 46
    # A fresh search finds the weight-three logical that a stale resume skipped.
    assert run(binary, path, cap, '--split', 1)['found'] == 3
    assert_rejected_untouched(binary, path, cap, ckpt, '--split', 1)


@pytest.mark.parametrize('cap', [2, 3, 4])
def test_adversarial_legacy_headerless_checkpoint(binary, tmp_path, cap):
    fixture = Path(__file__).parent / 'fixtures'
    ckpt = tmp_path / 'legacy.ckpt'
    ckpt.write_bytes((fixture / 'legacy-resume-cap2.ckpt').read_bytes())
    assert_rejected_untouched(binary, fixture / 'legacy-resume-five.inst', cap, ckpt, '--split', 1)


@pytest.mark.parametrize('change', ['domain-X', 'domain-Z', 'orbits', 'root-order'])
def test_adversarial_instance_tree_change(binary, tmp_path, change):
    path = write_instance(five(), tmp_path / 'five.inst')
    ckpt = tmp_path / 'original.ckpt'
    run(binary, path, 2, '--split', 1, '--ckpt', ckpt)
    lines = path.read_text().splitlines()
    if change.startswith('domain-'):
        lines[-1] = 'domain ' + change[-1]
    elif change == 'orbits':
        # The cyclic five-qubit group is transitive; the new tree has one root orbit.
        lines[-3], lines[-2] = '0 0 0 0 0', '1 0'
    else:
        lines[-3], lines[-2] = '4 3 2 1 0', '5 4 3 2 1 0'
    path.write_text('\n'.join(lines) + '\n')
    # run() rehashes the actual changed instance bytes, as the public wrapper does.
    assert_rejected_untouched(binary, path, 2, ckpt, '--split', 1)


@pytest.mark.parametrize('domain', ['X', 'Z'])
def test_adversarial_domain_header_binding(binary, tmp_path, domain):
    path = write_instance(five(), tmp_path / 'five.inst')
    ckpt = tmp_path / 'original.ckpt'
    run(binary, path, 2, '--ckpt', ckpt)
    # Isolate the explicit domain field, independently of the instance hash check.
    ckpt.write_bytes(ckpt.read_bytes().replace(b'domain=P', ('domain=' + domain).encode()))
    assert_rejected_untouched(binary, path, 2, ckpt)


@pytest.mark.parametrize('policy', ['none', 'greedy+b1b3'])
def test_adversarial_checkpoint_bound_policy(binary, tmp_path, policy):
    path = write_instance(five(), tmp_path / 'five.inst')
    ckpt = tmp_path / 'greedy.ckpt'
    run(binary, path, 2, '--ckpt', ckpt)
    assert_rejected_untouched(binary, path, 2, ckpt, '--bound', policy)
    fresh = tmp_path / 'unsupported.ckpt'
    result = run(binary, path, 2, '--ckpt', fresh, '--bound', policy, check=False)
    assert result.returncode == 2 and 'incompatible' in result.stderr
    assert not fresh.exists()


@pytest.mark.parametrize('variant', ['4096', '1024-w64', '1024-w128'])
def test_adversarial_checkpoint_build_variant(binary, tmp_path, variant):
    path = write_instance(five(), tmp_path / 'five.inst')
    ckpt = tmp_path / 'base.ckpt'
    run(binary, path, 2, '--ckpt', ckpt)
    other = native.build('bnb-cpu', variant=variant)
    assert native.version(binary)['limits'] != native.version(other)['limits']
    assert_rejected_untouched(other, path, 2, ckpt)


@pytest.mark.parametrize('existing', [False, True])
def test_adversarial_emit_certificate_with_checkpoint(binary, tmp_path, existing):
    code = five()
    path = write_instance(code, tmp_path / 'five.inst')
    ckpt = tmp_path / 'search.ckpt'
    if existing:
        run(binary, path, 2, '--ckpt', ckpt)
    original = ckpt.read_bytes() if existing else None
    meta = tmp_path / 'meta.json'
    meta.write_text(json.dumps({'group_sha256': code.group_hash(), 'permutations': []}))
    coverage = tmp_path / 'proof.cov'
    result = run(binary, path, 2, '--ckpt', ckpt, '--emit-cert', coverage,
                 '--cert-metadata', meta, check=False)
    assert result.returncode == 2 and 'incompatible' in result.stderr
    assert not coverage.exists()
    if existing:
        assert ckpt.read_bytes() == original
    else:
        assert not ckpt.exists()


def test_resume_and_reject(binary, tmp_path):
    code = five()
    path = write_instance(code, tmp_path / "five.inst")
    ckpt = tmp_path / "search.ckpt"
    a = run(binary, path, 2, "--ckpt", ckpt, "--cpu-budget", 1, "--delta", 1, "--chunk", 1)
    assert ckpt.read_text().startswith("H stabcert-bnb-ckpt 2 ")
    b = run(binary, path, 2, "--ckpt", ckpt, "--cpu-budget", 30, "--delta", 1, "--chunk", 9, "--threads", 1)
    for key in ("nodes", "gen_nodes", "task_nodes", "rounds", "tasks_total", "aborted_total"):
        assert a[key] == b[key]
    assert b["resumed"] and b["instance_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    original = ckpt.read_bytes()
    for args in [("--cap", 1), ("--split", 2), ("--delta", 2), ("--stop-first",),
                 ("--instance-sha256", "a"*64), ("--binary-sha256", "b"*64)]:
        p = run(binary, path, 2, "--ckpt", ckpt, "--delta", 1, *args, check=False)
        assert p.returncode == 6, (p.returncode, p.stderr)
        assert ckpt.read_bytes() == original
    header, body = original.split(b"\n", 1)
    for broken in [header.replace(b"source_sha256=", b"source_sha256=0") + b"\n" + body,
                   original + b"C 0 0 1 0 0 0 0\n", original + b"C 99 0 1 0 0 0 0\n",
                   original + b"C 1 0 999999 0 0 0 0\n"]:
        ckpt.write_bytes(broken)
        p = run(binary, path, 2, "--ckpt", ckpt, "--delta", 1, check=False)
        assert p.returncode == 6
        assert ckpt.read_bytes() == broken
    ckpt.write_bytes(original + b"C 2 0")
    c = run(binary, path, 2, "--ckpt", ckpt, "--delta", 1)
    assert c["nodes"] == a["nodes"]
    assert ckpt.read_bytes() == original


def test_find_reemitted(binary, tmp_path):
    code = five()
    path = write_instance(code, tmp_path / "five.inst")
    ckpt = tmp_path / "find.ckpt"
    a = run(binary, path, 5, "--ckpt", ckpt, "--chunk", 1)
    original = ckpt.read_bytes()
    assert a["stopped_on_find"] and b"\nF " in original
    b = run(binary, path, 5, "--ckpt", ckpt)
    assert b["stopped_on_find"] and b["resumed"]
    assert a["found"] == b["found"]
    witness(code, b)
    assert ckpt.read_bytes() == original


def test_kill_resume(binary, tmp_path):
    base = five()
    copies, n = 30, 150
    rows = [(g & 31) << off | (g >> 5) << (n + off) for off in range(0, n, 5) for g in base.generators]
    code = StabilizerCode(n, rows)
    path = write_instance(code, tmp_path / "many.inst")
    ckpt = tmp_path / "interrupted.ckpt"
    command = [str(binary), str(path), "--cap", "2", "--threads", "1", "--chunk", "1", "--ckpt", str(ckpt),
               "--instance-sha256", hashlib.sha256(path.read_bytes()).hexdigest(),
               "--binary-sha256", hashlib.sha256(binary.read_bytes()).hexdigest()]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for line in proc.stdout:
            if json.loads(line)["event"] == "progress":
                proc.kill()
                break
        proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    expected = run(binary, path, 2)
    assert len(ckpt.read_text().splitlines()) < expected["tasks_total"] + 1
    resumed = run(binary, path, 2, "--ckpt", ckpt, "--threads", 4, "--chunk", 17)
    assert resumed["resumed"]
    assert resumed["nodes"] == expected["nodes"]


def chained(header, payloads):
    """Independent format encoder for exercising validation beyond the CRC."""
    result = header + b'\n'
    crc = zlib.crc32(result)
    for payload in payloads:
        crc = zlib.crc32(payload + b'\n', crc)
        result += payload + f' crc32={crc:08x}\n'.encode()
    return result


def test_integrity_chain_and_valid_aborted_digit_corruption(binary, tmp_path):
    path = write_instance(five(), tmp_path / 'five.inst')
    ckpt = tmp_path / 'aborted.ckpt'
    expected = run(binary, path, 2, '--ckpt', ckpt, '--cpu-budget', 1, '--chunk', 100, '--delta', 1)
    assert expected['aborted_total'] == 9
    original = ckpt.read_bytes()
    header, *lines = original.splitlines()
    payloads = [line.rsplit(b' crc32=', 1)[0] for line in lines]
    assert original == chained(header, payloads)
    # 8 -> 9 is a single digit change: still in [0,15), unique, sorted,
    # and with unchanged aborted count. Retain only round 0 so later task
    # counts cannot be responsible for rejecting this corrupted prefix.
    assert payloads[0].endswith(b' 6 7 8')
    changed = lines[0].replace(b' 6 7 8 ', b' 6 7 9 ')
    ckpt.write_bytes(header + b'\n' + changed + b'\n')
    result = run(binary, path, 2, '--ckpt', ckpt, '--delta', 1, check=False)
    assert result.returncode == 6 and 'record checksum mismatch' in result.stderr
    assert_rejected_untouched(binary, path, 2, ckpt, '--delta', 1)
    # Unmodified prefix is accepted and recovers all work.
    ckpt.write_bytes(header + b'\n' + lines[0] + b'\n')
    recovered = run(binary, path, 2, '--ckpt', ckpt, '--delta', 1)
    assert recovered['nodes'] == expected['nodes'] == 46


@pytest.mark.parametrize('change', ['node-digit', 'checksum-digit', 'delete-middle', 'swap', 'duplicate', 'find-label'])
def test_integrity_chain_rejects_record_corruption(binary, tmp_path, change):
    path = write_instance(five(), tmp_path / 'five.inst')
    ckpt = tmp_path / 'integrity.ckpt'
    cap = 3 if change == 'find-label' else 2
    run(binary, path, cap, '--ckpt', ckpt, '--chunk', 1)
    header, *lines = ckpt.read_bytes().splitlines()
    assert ckpt.read_bytes() == chained(header, [line.rsplit(b' crc32=', 1)[0] for line in lines])
    if change == 'node-digit':
        row = lines[0].split(); row[4] = str(int(row[4]) + 1).encode(); lines[0] = b' '.join(row)
    elif change == 'checksum-digit':
        lines[0] = lines[0][:-1] + (b'0' if lines[0][-1:] != b'0' else b'1')
    elif change == 'delete-middle':
        del lines[1]
    elif change == 'swap':
        lines[0], lines[1] = lines[1], lines[0]
    elif change == 'duplicate':
        lines.insert(1, lines[0])
    else:
        assert lines[-1].startswith(b'F ')
        row = lines[-1].split(); row[3] = str(int(row[3]) % 3 + 1).encode(); lines[-1] = b' '.join(row)
    ckpt.write_bytes(b'\n'.join([header, *lines]) + b'\n')
    assert_rejected_untouched(binary, path, cap, ckpt)


@pytest.mark.parametrize('payloads', [
    [b'C 0 0 1 1 1 0 1 1'],                # aborted index outside its chunk
    [b'C 0 0 2 1 1 0 2 0 0'],              # duplicate aborted index
    [b'C 0 0 2 1 1 0 2 1 0'],              # unsorted aborted indices
    [b'C 0 0 16 1 1 0 0'],                 # chunk exceeds generated task count
    [b'C 0 0 1 1 1 0 0', b'C 0 0 1 1 1 0 0'],  # overlapping chunks
    [b'C 0 0 1 1 1 0 0', b'C 1 0 1 1 1 0 0'],  # next round before completion
    [b'F 3 0 1 1 1 2 1'],                 # weight exceeds cap
])
def test_valid_crc_does_not_bypass_semantic_validation(binary, tmp_path, payloads):
    path = write_instance(five(), tmp_path / 'five.inst')
    ckpt = tmp_path / 'semantics.ckpt'
    run(binary, path, 2, '--ckpt', ckpt)
    header = ckpt.read_bytes().splitlines()[0]
    ckpt.write_bytes(chained(header, payloads))
    assert_rejected_untouched(binary, path, 2, ckpt)


def test_version_one_is_explicitly_refused(binary, tmp_path):
    path = write_instance(five(), tmp_path / 'five.inst')
    ckpt = tmp_path / 'old.ckpt'
    run(binary, path, 2, '--ckpt', ckpt)
    header, *lines = ckpt.read_bytes().splitlines()
    ckpt.write_bytes(header.replace(b'stabcert-bnb-ckpt 2 ', b'stabcert-bnb-ckpt 1 ') + b'\n' +
                     b'\n'.join(line.rsplit(b' crc32=', 1)[0] for line in lines) + b'\n')
    assert_rejected_untouched(binary, path, 2, ckpt)
    result = run(binary, path, 2, '--ckpt', ckpt, check=False)
    assert 'version-1 checkpoints lack integrity checks' in result.stderr
