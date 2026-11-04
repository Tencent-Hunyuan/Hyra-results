"""Exporter format/CPU-CUDA/refinement regressions; not the independent checker."""
import io
import json
import os
from pathlib import Path
import struct

import pytest
from hyqec_verify import native
from hyqec_verify.core import StabilizerCode
from .helpers import write_instance, run, python_nodes
from .test_bnb import five

pytestmark = pytest.mark.native


def decode(path):
    raw = path.read_bytes()
    f = io.BytesIO(raw)
    assert f.read(8) == b'SBCOV1\r\n'
    size, = struct.unpack('<Q', f.read(8))
    header = json.loads(f.read(size))
    payload = raw[f.tell():]
    records = []
    def byte():
        b = f.read(1)
        assert b, 'truncated byte'
        return b[0]
    def fixed():
        return struct.unpack('<Q', f.read(8))[0]
    def u():
        result = shift = 0
        while True:
            b = byte(); result |= (b & 127) << shift
            if not b & 128:
                assert not shift or b
                return result
            shift += 7
            assert shift <= 63
    while True:
        tag = byte()
        if tag == 0x10:
            parent, edge, rule, fp, w, active = u(), u(), byte(), fixed(), u(), u()
            hint = [u() for _ in range(u())] if rule == 2 else ([u()] if rule == 3 else [])
            records.append((tag, parent, edge, rule, fp, w, active, hint))
        elif tag in (0x11, 0x12, 0x13):
            records.append((tag, u(), u(), u(), fixed()))
        elif tag == 0x1f:
            records.append((tag, u(), u(), u(), fixed()))
        elif tag == 0x20:
            task, root, length = u(), u(), u()
            path_bytes = f.read(length)
            assert len(path_bytes) == length
            records.append((tag, task, root, path_bytes, u(), fixed()))
        elif tag in (0x21, 0x22):
            records.append((tag, u()))
        elif tag == 0x7f:
            records.append((tag, u(), u(), u(), u(), fixed()))
            assert not f.read()
            break
        else:
            raise AssertionError(f'unknown tag {tag:x}')
    return header, payload, records


def emit(binary, code, tmp_path, cap, split, *args, domain='P', detail='frontier', name='proof'):
    instance = write_instance(code, tmp_path / f'{name}.inst', domain)
    metadata = tmp_path / f'{name}.json'
    metadata.write_text(json.dumps({'group_sha256': code.group_hash(), 'permutations': []}))
    path = tmp_path / f'{name}.cov'
    done = run(binary, instance, cap, '--split', split, '--emit-cert', path,
               '--cert-metadata', metadata, '--cert-detail', detail, *args)
    return done, path


def test_published_vector(tmp_path):
    binary = native.build('bnb-cpu')
    code = StabilizerCode(4, [15, 15 << 4])
    vectors = json.loads((Path(__file__).parents[2] / 'docs/BNB_CERT_VECTORS.json').read_text())
    for split in (1, 2):
        done, path = emit(binary, code, tmp_path, 1, split)
        header, _, records = decode(path)
        assert header['group_sha256'] == code.group_hash()
        assert done['nodes'] == vectors['nodes'] == 9
        assert done['global_trace'] == vectors['global_trace']
        assert records[-1][-1] == int(vectors['global_trace'], 16)
        if split == 1:
            tasks = [r for r in records if r[0] == 0x20]
            assert len(tasks) == 12
            for task, expected in zip(tasks, vectors['records']):
                assert task[4] == expected['node_count']
                assert task[5] == int(expected['trace'], 16)
        else:
            assert sum(r[0] == 0x10 for r in records) == 9
            assert sum(r[0] == 0x11 for r in records) == 3
    done, path = emit(binary, code, tmp_path, 0, 1)
    assert done['nodes'] == 0 and done['certificate_complete']
    assert decode(path)[2][-1] == (0x7f, 0, 0, 0, 0, 0)


def test_refinement_and_full_transcript(tmp_path):
    binary = native.build('bnb-cpu')
    code = five()
    for domain in ('P', 'X', 'Z'):
        for split in (1, 2, 3):
            done, path = emit(binary, code, tmp_path, 2, split, domain=domain)
            expected = decode(path)
            assert done['nodes'] == python_nodes(code, 2, {'P': 0, 'X': 1, 'Z': 2}[domain])
            refined, path = emit(binary, code, tmp_path, 2, split, '--cpu-budget', 1,
                                 '--delta', 1, '--chunk', 2, domain=domain, name='refined')
            assert decode(path)[1] == expected[1]
            assert refined['global_trace'] == done['global_trace']
            if split == 1 and domain == 'P':
                assert refined['aborted_total'] > 0
            full, path = emit(binary, code, tmp_path, 2, split, domain=domain, detail='full', name='full')
            assert full['global_trace'] == done['global_trace']
            assert decode(path)[2][-1] == expected[2][-1]


def test_found_and_invalid_options(tmp_path):
    binary = native.build('bnb-cpu')
    code = five()
    done, path = emit(binary, code, tmp_path, 3, 1)
    assert done['found'] == 3 and not done['certificate_complete']
    assert not path.exists()
    instance = write_instance(code, tmp_path / 'input.inst')
    meta = tmp_path / 'meta.json'
    meta.write_text(json.dumps({'group_sha256': code.group_hash(), 'permutations': []}))
    for args in (('--stop-first',), ('--bound', 'none'), ('--ckpt', tmp_path/'ckpt')):
        result = run(binary, instance, 2, '--emit-cert', path, '--cert-metadata', meta, *args, check=False)
        assert result.returncode == 2 and 'incompatible' in result.stderr
        assert not path.exists()
    # Ablation changes only pruning, and still reaches the same exact witness.
    standard = run(binary, instance, 2)
    without = run(binary, instance, 2, '--bound', 'none')
    assert without['nodes'] >= standard['nodes'] and without['found'] == -1
    assert run(binary, instance, 3, '--bound', 'none')['found'] == 3


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('HYQEC_TEST_GPU') != '1', reason='GPU opt-in required')
def test_gpu_payload_identity(tmp_path):
    cpu, gpu = native.build('bnb-cpu'), native.build('bnb-cuda')
    code = five()
    for domain in ('P', 'X', 'Z'):
        for split in (1, 2):
            done, path = emit(cpu, code, tmp_path, 2, split, domain=domain)
            expected = decode(path)
            # Five-task batches also exercise ragged strided pieces, including
            # restoring their results before budget-driven refinement.
            for slots in (1, 2, 3):
                cuda, path = emit(gpu, code, tmp_path, 2, split, '--gpu', 0, '--budget', 1,
                                  '--delta', 1, '--chunk', 5, '--slots', slots, domain=domain, name='cuda')
                actual = decode(path)
                assert actual[1] == expected[1]
                assert cuda['global_trace'] == done['global_trace']
                assert actual[0]['provenance']['binary_sha256'] != expected[0]['provenance']['binary_sha256']
    instance = write_instance(code, tmp_path/'none.inst')
    assert run(cpu, instance, 2, '--bound', 'none')['nodes'] == run(gpu, instance, 2, '--bound', 'none', '--gpu', 0)['nodes']
