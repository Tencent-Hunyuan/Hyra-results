"""Data-parallel fine-tuning on two GPUs: the summed gradient equals the gradient of one batch holding both
GPUs' structures (skipped without two CUDA devices or installed presto weights)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason='two CUDA devices are required')

from hyff.registry import installed  # noqa: E402

if installed('presto') is None:
    pytestmark = [pytestmark, pytest.mark.skip(reason='presto weights are not installed (hyff download presto)')]

SCRIPT = r'''
import os, sys
from pathlib import Path
import numpy as np
import torch
from hyff import finetuning as ft
root = Path(sys.argv[1])
world = int(os.environ.get('WORLD_SIZE', '1'))
data = root / 'train.npz'
if not data.exists():
    rng = np.random.default_rng(0)
    water = np.array([[0., 0., 0.], [0.9572, 0., 0.], [-0.24, 0.927, 0.]])
    clusters = [np.concatenate([water + rng.uniform(0, 5, 3) + 3.0 * i for i in range(1 + k % 3)]) for k in range(40)]
    sizes = [len(c) for c in clusters]
    np.savez(data, positions=np.concatenate(clusters), numbers=np.concatenate([[8, 1, 1] * (n // 3) for n in sizes]),
             offsets=np.concatenate([[0], np.cumsum(sizes)]), energies=rng.normal(size=40),
             forces=rng.normal(scale=0.3, size=(sum(sizes), 3)))
run = ft.FineTune(ft.FineTuneConfig(train=str(data), model='presto', output=str(root / f'run{world}'), batch_atoms=96,
                                    gpus=world, valid_fraction=0.0), verbose=False)
groups = ft._batches(run.data, run.train_indices, 96, None)[:2]
loss = run.train_step(groups[run.rank] if world > 1 else groups[0] + groups[1])
if run.rank == 0:
    torch.save(dict(loss=loss, grads={n: None if p.grad is None else p.grad.double().cpu()
                                      for n, p in run.model.named_parameters()}), root / f'grads{world}.pt')
run.close()
'''


def test_two_gpu_gradient_equals_the_union_batch_gradient(tmp_path):
    import hyff
    package_root = str(Path(hyff.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [package_root, os.environ.get('PYTHONPATH')])))
    script = tmp_path / 'step.py'
    script.write_text(SCRIPT)
    single = subprocess.run([sys.executable, str(script), str(tmp_path)], env=dict(env, CUDA_VISIBLE_DEVICES='0'),
                            capture_output=True, text=True, timeout=900)
    assert single.returncode == 0, single.stderr[-3000:]
    double = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                             str(script), str(tmp_path)], env=env, capture_output=True, text=True, timeout=900)
    assert double.returncode == 0, double.stderr[-3000:]
    a, b = torch.load(tmp_path / 'grads2.pt'), torch.load(tmp_path / 'grads1.pt')
    assert abs(a['loss'] - b['loss']) <= 1e-5 * abs(b['loss'])
    assert {n for n, g in a['grads'].items() if g is None} == {n for n, g in b['grads'].items() if g is None}
    used = [n for n, g in b['grads'].items() if g is not None]
    difference = sum(float((a['grads'][n] - b['grads'][n]).square().sum()) for n in used)
    norm = sum(float(b['grads'][n].square().sum()) for n in used)
    assert (difference / norm) ** 0.5 < 1e-4
