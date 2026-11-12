"""Small test fixtures: randomly initialized checkpoints and molecules, without downloaded weights."""
import json
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[2] / 'src' / 'hyff' / 'checkpoints'


def random_checkpoint(directory, name='presto', seed=0):
    """Write a randomly initialized model of the released architecture ``name`` as weights.safetensors/.json.

    The charge/spin projection is made nonzero so that charge and multiplicity change the energy.
    """
    from safetensors.torch import save_file
    from hyff.models.factory import build_model
    spec = json.loads((PACKAGE / name / 'weights.json').read_text())
    spec.pop('weights_sha256', None)
    torch.manual_seed(seed)
    model = build_model(spec)
    with torch.no_grad():
        model.state_projection.weight.normal_(0, 0.1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().clone() for k, v in model.state_dict().items()}, str(directory / 'weights.safetensors'))
    (directory / 'weights.json').write_text(json.dumps(spec))
    return directory / 'weights.safetensors'


def random_molecules(count, seed=0, elements=(1, 6, 7, 8)):
    """Molecule-like clusters: atoms on a jittered grid 1.3 A apart; returns [(positions, numbers)]."""
    rng = np.random.default_rng(seed)
    molecules = []
    for k in range(count):
        n = int(rng.integers(3, 9))
        grid = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing='ij'), -1).reshape(-1, 3)
        positions = grid[rng.permutation(len(grid))[:n]] * 1.3 + rng.normal(scale=0.05, size=(n, 3))
        numbers = rng.choice(elements, size=n)
        molecules.append((positions.astype(np.float64), numbers.astype(np.int64)))
    return molecules
