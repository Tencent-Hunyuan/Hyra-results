import os
import json
import subprocess
import pytest
from hyqec_verify import native
from hyqec_verify.core import StabilizerCode, pauli_from_xz
from .helpers import write_ris
from .test_bnb import five
from oracle import side_distance, from_int

pytestmark = pytest.mark.native


def call(binary, path, *args):
    run = subprocess.run([str(binary),str(path),"--seconds","10","--seed","17",*map(str,args)],capture_output=True,text=True,timeout=30)
    assert run.returncode == 0, (run.returncode,run.stdout,run.stderr)
    return [json.loads(line) for line in run.stdout.splitlines()]


def verify(code, records, hamming=False):
    count = 0
    for r in records:
        if r['event'] in ('witness','ring'):
            v=pauli_from_xz(r['x'],r['z'],code.n)
            w=code.check_witness(v)
            assert (len(r['x'])+len(r['z']) if hamming else w) == r['w']
            count+=1
    assert count
    assert records[-1]['event']=='done' and records[-1]['interface']==1


def steane():
    rows = [[0,2,4,6],[1,2,5,6],[3,4,5,6]]
    return StabilizerCode.css(rows,rows,7)


@pytest.mark.parametrize('chain',[True,False])
def test_cpu_witnesses_and_determinism(tmp_path,chain):
    binary=native.build('ris-cpu')
    for code in (five(),steane()):
        for domain in ('P','X','Z'):
            path=write_ris(code,tmp_path/'input.prs',domain)
            args=['--threads',1,'--max-steps',100,'--steps-per-round',20,'--report-le',4,'--ring-cap',32]
            if not chain:args+=['--no-chain']
            if domain!='P':args+=['--hamming']
            a=call(binary,path,*args);b=call(binary,path,*args)
            verify(code,a,domain!='P')
            assert a[-1]['best'] == (3 if domain == 'P' else side_distance([from_int(g,code.n) for g in code.generators], code.n, 1 if domain == 'X' else 2))
            for records in (a,b):records[-1].pop('seconds')
            assert a==b


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('HYQEC_TEST_GPU')!='1',reason='set HYQEC_TEST_GPU=1 and CUDA_VISIBLE_DEVICES')
def test_gpu_shared_global_and_witnesses(tmp_path):
    binary=native.build('ris-cuda')
    for code in (five(),steane()):
        for domain in ('P','X','Z'):
            path=write_ris(code,tmp_path/'gpu.prs',domain)
            for chain in (True,False):
                args=['--blocks',4,'--threads',32,'--max-steps',32,'--steps-per-round',16,'--report-le',4,'--ring-cap',16]
                if not chain:args+=['--no-chain']
                if domain!='P':args+=['--hamming']
                for mem in ([],['--gmem']):
                    records=call(binary,path,*args,*mem)
                    verify(code,records,domain!='P')
                    assert records[-1]['best'] == (3 if domain == 'P' else side_distance([from_int(g,code.n) for g in code.generators], code.n, 1 if domain == 'X' else 2))


def test_malformed_ris(tmp_path):
    binary=native.build('ris-cpu');code=five()
    path=write_ris(code,tmp_path/'bad.prs')
    data=path.read_bytes()
    for bad in (data[:-1],data+b'x',data[:24]+b'\x02'+data[25:]):
        path.write_bytes(bad)
        run=subprocess.run([str(binary),str(path),'--seconds','0'],capture_output=True,text=True)
        assert run.returncode==2


def test_cpu_no_swappable_qubits_fixed_budget(tmp_path):
    # The full Pauli space has no free qubit: successful swaps stay zero.
    code=StabilizerCode(2,[])
    path=write_ris(code,tmp_path/'full.prs')
    records=call(native.build('ris-cpu'),path,'--max-steps',17,'--threads',1)
    verify(code,records)
    assert records[-1]['attempted_steps']==17 and records[-1]['steps']==0
    assert records[-1]['best']==1


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('HYQEC_TEST_GPU')!='1',reason='set HYQEC_TEST_GPU=1')
def test_gpu_automatic_global_memory(tmp_path):
    n=1000
    code=StabilizerCode(n,[1 << (n+q) for q in range(n-1)])
    path=write_ris(code,tmp_path/'large.prs')
    records=call(native.build('ris-cuda'),path,'--blocks',2,'--threads',32,'--no-chain','--max-steps',2)
    assert records[0]['gmem']==1
    assert records[-1]['best']==1
    verify(code,records)
