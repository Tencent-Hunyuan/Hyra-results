#!/usr/bin/env python3
"""Direct native-engine adapter for the independent reference-corpus harness.

Uses the package's input validation, writer and witness checker. It does not
read reference distances and is limited by the native B&B implementation's
published limits. Package CLI/certificate acceptance is tested separately.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from hyqec_verify import native
from hyqec_verify.core import CodeError, xz_from_pauli
from hyqec_verify.engines.bnb import write_instance
from hyqec_verify.engines.ris_native import run as upper
from hyqec_verify.io import load
from hyqec_verify.symmetry import trivial_orbits


def main():
    code=load(sys.argv[1]).code
    if code.k==0:
        print(json.dumps({'status':'no_logical','n':code.n,'k':0,'d':None}));return
    ub=upper(code,seconds=0.05,threads=1,seed=7)
    cap=min(46,ub['upper_bound'] or code.n)
    binary=native.locate('bnb-cpu',variant=1024 if max(code.n,len(code.generators))<=1024 else 4096)
    with tempfile.TemporaryDirectory(prefix='hyqec-native-reference-') as tmp:
        instance=Path(tmp)/'instance.txt'
        digest=write_instance(instance,code,trivial_orbits(code.n))
        cmd=[str(binary),str(instance),'--cap',str(cap),'--split','2','--threads','2',
             '--instance-sha256',digest,'--binary-sha256',hashlib.sha256(binary.read_bytes()).hexdigest()]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
        if p.returncode:raise RuntimeError(p.stderr)
        done=json.loads(p.stdout.splitlines()[-1])
        assert done['replay_fail']==0 and done['instance_sha256']==digest
        assert done['nodes']==done['gen_nodes']+done['task_nodes']
        result={'n':code.n,'k':code.k,'rank':code.rank}
        if done['found']>=0:
            v=0
            for q,l in done['witness']:
                if l&1:v|=1<<q
                if l&2:v|=1<<(code.n+q)
            weight=code.check_witness(v)
            assert weight==done['found'] and not done['stopped_on_find']
            result.update(status='exact',d=weight,witness=xz_from_pauli(v,code.n))
        else:
            result.update(status='bounds',lower_bound=cap+1)
        print(json.dumps(result))


if __name__=='__main__':
    try:main()
    except CodeError as error:
        print(json.dumps({'status':'rejected','reason':str(error)}))
