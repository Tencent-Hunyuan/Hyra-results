"""CPU-only wide variants preserve the tree and undo high support positions."""
import pytest

from hyqec_verify import native
from hyqec_verify.core import CodeError, StabilizerCode
from .helpers import python_nodes, run, witness, write_instance
from .test_bnb import five
from .test_coverage import decode, emit

pytestmark = pytest.mark.native


def dense_five(n):
    """Five-qubit code on the final qubits, with |0> ancillas before it.

    Multiply every code generator by the product of ancilla Z stabilizers.
    This changes the supplied presentation, not the group or distance. Its
    late core qubits force branch decisions above positions 32 and 64.
    """
    shift = n - 5
    ancillary = ((1 << shift) - 1) << n
    rows = [((g & 31) << shift) | ((g >> 5) << (n + shift)) for g in five().generators]
    rows = [g ^ ancillary for g in rows]
    rows += [1 << (n + q) for q in range(shift)]
    return StabilizerCode(n, rows)


@pytest.mark.parametrize('variant,n', [('1024-w64', 64), ('1024-w128', 96), ('4096-w128', 129)])
def test_dense_known_distance_and_reference_tree(tmp_path, variant, n):
    binary = native.build('bnb-cpu', variant=variant)
    limits = native.version(binary)['limits']
    assert limits['WMAX'] == (64 if variant.endswith('w64') else 128)
    assert limits['CMAX'] == 128
    code = dense_five(n)
    for domain, label, d in (('P', 0, 3), ('X', 1, 5), ('Z', 2, 5)):
        path = write_instance(code, tmp_path/'wide.inst', domain)
        expected = python_nodes(code, d - 1, label)
        direct = run(binary, path, d - 1, '--split', 1)
        split = run(binary, path, d - 1, '--split', 2, '--cpu-budget', 1, '--delta', 1)
        assert direct['nodes'] == split['nodes'] == expected
        found = run(binary, path, d, '--split', 2)
        assert found['found'] == d
        witness(code, found)


def test_wide_and_narrow_share_existing_tree(tmp_path):
    code = five()
    for domain, cap in (('P', 2), ('X', 4), ('Z', 4)):
        path = write_instance(code, tmp_path/'small.inst', domain)
        counts = [run(native.build('bnb-cpu', variant=v), path, cap)['nodes']
                  for v in (1024, '1024-w64', '1024-w128')]
        assert len(set(counts)) == 1


def test_wide_column_degree_and_limits(tmp_path):
    # 127 dependent dense rows plus one ancilla stabilizer: degree exactly128.
    base = dense_five(6)
    code = StabilizerCode(6, [base.generators[i % 4] for i in range(127)] + [1 << 6])
    path = write_instance(code, tmp_path/'degree.inst')
    narrow = native.build('bnb-cpu')
    wide = native.build('bnb-cpu', variant='1024-w64')
    rejected = run(narrow, path, 2, check=False)
    assert rejected.returncode == 2 and 'CMAX' in rejected.stderr
    assert run(wide, path, 2)['nodes'] == python_nodes(code, 2)
    witness(code, run(wide, path, 3))
    too_many = StabilizerCode(6, list(code.generators) + [1 << 6])
    path = write_instance(too_many, path)
    rejected = run(wide, path, 2, check=False)
    assert rejected.returncode == 2 and 'CMAX' in rejected.stderr
    path = write_instance(dense_five(66), path)
    rejected = run(wide, path, 2, check=False)
    assert rejected.returncode == 2 and 'WMAX' in rejected.stderr
    widest = native.build('bnb-cpu', variant='1024-w128')
    path = write_instance(dense_five(130), path)
    rejected = run(widest, path, 2, check=False)
    assert rejected.returncode == 2 and 'WMAX' in rejected.stderr


def test_wide_variant_is_cpu_bnb_only():
    for target in ('bnb-cuda', 'ris-cpu', 'ris-cuda'):
        with pytest.raises(ValueError, match='only for bnb-cpu'):
            native.build(target, variant='1024-w64')


@pytest.mark.parametrize('case', ['weight64', 'weight128', 'degree128'])
def test_public_wrapper_selects_wide_cpu(tmp_path, case):
    from hyqec_verify.engines.bnb import run_bnb
    if case == 'degree128':
        base = dense_five(6)
        code = StabilizerCode(6, [base.generators[i % 4] for i in range(127)] + [1 << 6])
    else:
        code = dense_five(64 if case == 'weight64' else 96)
    lower = run_bnb(code, 2, tmp_path / 'lower', threads=2, root_order='file', split=1)
    assert lower.status == 'none', lower.record
    assert lower.record['result']['nodes'] == python_nodes(code, 2)
    limits = lower.record['engine']['version']['limits']
    assert limits['WMAX'] == (128 if case == 'weight128' else 64)
    assert limits['CMAX'] == 128
    exact = run_bnb(code, 3, tmp_path / 'exact', threads=2, root_order='file', split=1)
    assert exact.status == 'found' and exact.found == 3
    assert code.check_witness(exact.witness) == 3
    # Unsupported GPU shapes fail before probing CUDA or launching a device job.
    with pytest.raises(CodeError, match='use the CPU engine'):
        run_bnb(code, 2, tmp_path / 'gpu', threads=2, gpus='0', root_order='file')


def test_original_limit_payload_identity_and_wide_resume(tmp_path):
    code = dense_five(33)  # weight32, exercising the final packed position
    narrow = native.build('bnb-cpu')
    wide = native.build('bnb-cpu', variant='1024-w128')
    for domain, cap in (('P', 2), ('X', 4), ('Z', 4)):
        _, original = emit(narrow, code, tmp_path, cap, 2, domain=domain, name='narrow')
        _, expanded = emit(wide, code, tmp_path, cap, 2, domain=domain, name='wide')
        assert decode(original)[1] == decode(expanded)[1]
    code = dense_five(129)
    path = write_instance(code, tmp_path/'resume.inst', 'X')
    ckpt = tmp_path/'resume.ckpt'
    arguments = ('--split', 2, '--cpu-budget', 1, '--delta', 1, '--ckpt', ckpt)
    first = run(wide, path, 4, *arguments)
    resumed = run(wide, path, 4, *arguments, '--threads', 1)
    assert resumed['resumed'] and resumed['nodes'] == first['nodes']
