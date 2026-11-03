import numpy as np
import pytest
import torch

import hycell

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope='module')
def base(weights):
    return hycell.load(weights, device=DEVICE)


def test_finetune_returns_new_model(base, lung, expected):
    before = base.model.proj.weight.detach().clone()
    tuned = base.finetune(lung, 'cell_type', steps=4, cells_per_step=8, validation_fraction=0, progress=False)
    assert torch.equal(base.model.proj.weight, before)
    assert tuned.meta['finetune']['steps_run'] == 4
    z = tuned.embed(lung[:32].copy())
    assert z.shape == (32, 512) and np.isfinite(z).all()
    assert not np.allclose(z, expected['embedding'][:32])
    with pytest.raises(ValueError):
        base.predict(lung[:4].copy())


def test_finetune_errors(base, lung):
    with pytest.raises(KeyError):
        base.finetune(lung, 'not_a_column', steps=1)
    one = lung.copy()
    one.obs['one'] = 'x'
    with pytest.raises(ValueError):
        base.finetune(one, 'one', steps=1)


def test_save_and_load(base, lung, tmp_path):
    tuned = base.finetune(lung, 'cell_type', steps=4, cells_per_step=8, validation_fraction=0.1, eval_every=2,
                          progress=False)
    assert tuned.classes == sorted(lung.obs.cell_type.astype(str).unique())
    pred = tuned.predict(lung[:16].copy())
    proba = tuned.predict(lung[:16].copy(), return_probabilities=True)
    assert list(proba.columns) == tuned.classes and np.allclose(proba.sum(1), 1, atol=1e-5)
    tuned.save(tmp_path / 'tuned')
    again = hycell.load(tmp_path / 'tuned', device=DEVICE)
    assert again.classes == tuned.classes
    assert np.array_equal(again.embed(lung[:16].copy()), tuned.embed(lung[:16].copy()))
    assert (again.predict(lung[:16].copy()) == pred).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='training to convergence needs a GPU')
def test_annotator_learns_cell_types(base, lung):
    tuned = base.finetune(lung, 'cell_type', steps=200, cells_per_step=32, eval_every=50, progress=False)
    h = tuned.meta['finetune']['history']
    assert min(x['val_loss'] for x in h[1:]) < h[0]['val_loss'] and max(x['val_accuracy'] for x in h) > 0.75
