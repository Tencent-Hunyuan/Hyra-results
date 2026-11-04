import hashlib
import json
import os
import subprocess
import pytest
from hyqec_verify import native
from .helpers import write_instance, run, witness
from .test_bnb import five

pytestmark = [pytest.mark.native, pytest.mark.gpu,
              pytest.mark.skipif(os.environ.get("HYQEC_TEST_GPU") != "1", reason="set HYQEC_TEST_GPU=1 and CUDA_VISIBLE_DEVICES")]


def test_cpu_gpu_tree(tmp_path):
    cpu = native.build("bnb-cpu")
    gpu = native.build("bnb-cuda")
    code = five()
    for domain in ("P", "X", "Z"):
        inst = write_instance(code, tmp_path / "five.inst", domain)
        for cap in (0, 1, 2):
            a = run(cpu, inst, cap)
            for split in (1, 2, 3):
                b = run(gpu, inst, cap, "--gpu", 0, "--split", split, "--budget", 1, "--delta", 1)
                assert b["nodes"] == a["nodes"]
                assert b["found"] == -1
        done = run(gpu, inst, 5, "--gpu", 0)
        witness(code, done)


def test_large_variant_executes(tmp_path):
    from hyqec_verify.core import StabilizerCode
    # Both dimensions exceed the default build; ten unconstrained qubits give d=1.
    n=1100
    rows=[1 << (n+q) for q in range(1090)]
    code=StabilizerCode(n,rows+rows[:20])
    path=write_instance(code,tmp_path/'large.inst')
    for target,args in (('bnb-cpu',[]),('bnb-cuda',['--gpu',0])):
        binary=native.build(target,variant=4096)
        done=run(binary,path,1,*args)
        assert done['found']==1
        witness(code,done)


def gpu_queue_run(binary, instance, *args):
    command = [str(binary), str(instance), '--cap', '2', '--split', '1',
               '--threads', '2', '--gpu', '0', '--delta', '1',
               '--instance-sha256', hashlib.sha256(instance.read_bytes()).hexdigest(),
               '--binary-sha256', hashlib.sha256(binary.read_bytes()).hexdigest(),
               *map(str, args)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    done = next(event for event in events if event['event'] == 'done')
    assert done['found'] == -1 and done['replay_fail'] == 0
    return events, done


@pytest.mark.parametrize('slots', [1, 2, 3])
def test_small_gpu_round_uses_multiple_chunks(tmp_path, slots):
    """A large configured chunk must not give a whole small round to one slot."""
    instance = write_instance(five(), tmp_path/'five.inst')
    cpu = run(native.build('bnb-cpu'), instance, 2)
    events, done = gpu_queue_run(native.build('bnb-cuda'), instance, '--slots', slots)
    tasks = next(event['ntasks'] for event in events if event['event'] == 'tasks')
    ranges = sorted((event['task_start'], event['task_end']) for event in events
                    if event['event'] == 'progress' and event['round'] == 0)
    assert len(ranges) > 1
    cursor = 0
    for start, end in ranges:
        assert start == cursor and end > start
        cursor = end
    assert cursor == tasks
    assert done['nodes'] == cpu['nodes']


def test_gpu_resume_with_changed_queue_shape(tmp_path):
    """Repartitioning must cover exactly the unfinished tasks after a crash."""
    instance = write_instance(five(), tmp_path/'five.inst')
    binary = native.build('bnb-cuda')
    ckpt = tmp_path/'queue.ckpt'
    _, completed = gpu_queue_run(binary, instance, '--slots', 1, '--budget', 1,
                                '--ckpt', ckpt)
    lines = ckpt.read_bytes().splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines) if line.startswith(b'C '))
    assert sum(line.startswith(b'C ') for line in lines) > 1
    # Retain a valid durable prefix, as if interrupted after this completion.
    ckpt.write_bytes(b''.join(lines[:first+1]))
    _, resumed = gpu_queue_run(binary, instance, '--slots', 3, '--chunk', 3,
                              '--budget', 1, '--ckpt', ckpt)
    cpu = run(native.build('bnb-cpu'), instance, 2)
    assert resumed['resumed']
    assert resumed['nodes'] == completed['nodes'] == cpu['nodes']
