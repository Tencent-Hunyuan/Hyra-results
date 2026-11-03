import concurrent.futures
import hashlib
import json
import os
import shutil
import pytest
from hyqec_verify import native

pytestmark = pytest.mark.native


@pytest.mark.parametrize('compiler',['g++','clang++'])
@pytest.mark.parametrize('openmp',['0','auto'])
def test_cpu_compiler_variants(compiler,openmp,monkeypatch,tmp_path):
    if not shutil.which(compiler):pytest.skip(f'{compiler} unavailable')
    monkeypatch.setenv('CXX',compiler)
    monkeypatch.setenv('HYQEC_OPENMP',openmp)
    monkeypatch.setenv('HYQEC_CACHE',str(tmp_path))
    for target in ('bnb-cpu','ris-cpu'):
        path=native.build(target,variant=4096)
        ver=native.version(path)
        assert ver['interface']==1 and not ver['cuda']
        if openmp=='0':assert not ver['openmp']
        if target=='bnb-cpu':assert ver['limits']['NMAX']==4096
        monkeypatch.setenv('HYQEC_NO_BUILD','1')
        assert native.locate(target,variant=4096)==path
        monkeypatch.delenv('HYQEC_NO_BUILD')


def test_concurrent_cache_and_no_build(monkeypatch,tmp_path):
    monkeypatch.setenv('HYQEC_CACHE',str(tmp_path))
    monkeypatch.setenv('HYQEC_OPENMP','0')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a,b=list(pool.map(lambda _:native.build('ris-cpu'),range(2)))
    assert a==b
    meta=json.loads((a.parent/'build.json').read_text())
    assert meta['binary_sha256']==hashlib.sha256(a.read_bytes()).hexdigest()
    monkeypatch.setenv('HYQEC_NO_BUILD','1')
    monkeypatch.setenv('CXX','nonexistent-hyqec-compiler')
    assert native.locate('ris-cpu')==a
    with pytest.raises(native.NativeUnavailable):native.locate('bnb-cpu')


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('HYQEC_TEST_GPU')!='1',reason='set HYQEC_TEST_GPU=1')
def test_cuda_fat_binary(monkeypatch,tmp_path):
    monkeypatch.setenv('HYQEC_CACHE',str(tmp_path))
    for target in ('bnb-cuda','ris-cuda'):
        binary=native.build(target,cuda_arch='70,80,86,89,90')
        assert native.version(binary)['arch']=='70,80,86,89,90'


@pytest.mark.parametrize('failure', ['header', 'library'])
def test_missing_zlib_diagnostics(failure, monkeypatch, tmp_path, capsys):
    """The real probe/CLI diagnose both absent headers and a failed -lz link."""
    from hyqec_verify import cli
    import sys
    compiler = tmp_path / 'cxx'
    compiler.write_text(f'#!{sys.executable}\n' + '''
import sys
if '--version' in sys.argv:
    print('zlib-missing-test-compiler 1')
    raise SystemExit(0)
if '-lz' in sys.argv:
    print(ERROR, file=sys.stderr)
    raise SystemExit(1)
raise SystemExit(0)
'''.replace('ERROR', repr('fatal error: zlib.h: No such file or directory' if failure == 'header'
                          else 'ld: cannot find -lz')))
    compiler.chmod(0o755)
    monkeypatch.setenv('CXX', str(compiler))
    monkeypatch.setenv('NVCC', str(compiler))
    monkeypatch.setenv('HYQEC_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setenv('HYQEC_OPENMP', '0')
    ok, reason = native.availability('bnb-cpu')
    assert not ok and 'zlib development headers and library' in reason
    assert 'zlib1g-dev' in reason and 'zlib-devel' in reason and 'conda install zlib' in reason
    assert ('zlib.h' if failure == 'header' else 'cannot find -lz') in reason
    assert not native.availability('bnb-cuda')[0]
    assert native.available('ris-cpu')  # zlib is not a RIS dependency
    with pytest.raises(native.NativeUnavailable, match='zlib1g-dev'):
        native.build('bnb-cpu')
    with pytest.raises(native.NativeUnavailable, match='zlib1g-dev'):
        native.build('bnb-cuda', cuda_arch='90')
    assert cli.main(['doctor', '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert not report['native']['bnb-cpu']
    assert 'zlib1g-dev' in report['native_unavailable']['bnb-cpu']
    assert 'zlib1g-dev' in report['native_unavailable']['bnb-cuda']
    assert cli.main(['build', 'bnb-cpu']) == 1
    assert 'zlib1g-dev' in capsys.readouterr().err
