import h5py
import numpy as np
import pytest
import torch

import hypde
from hypde.layers import Geometry, merge_tokens, patch_merges, unmerge_tokens
from hypde.model import HyPDE, HyPDEConfig
from hypde.api import WindowNorm

TINY = HyPDEConfig(d=64, depth=2, heads=2, mlp_hidden=176, c_stem=32, d_frame=32, refiner_hidden=8, n_registers=2)


def _randomise(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(0.05 * torch.randn(p.shape, generator=g))
    return model


def test_merge_roundtrip():
    t = torch.randn(2, 8, 4, 5)
    for axes in ((0,), (1,), (0, 1)):
        assert torch.equal(unmerge_tokens(merge_tokens(t, axes), axes, 5), t)


def test_patch_plan():
    assert patch_merges((256, 256)) == (1, 1)       # 8 x 8 patches, 1024 tokens
    assert patch_merges((512, 128)) == (1, 1)       # 8 x 8 (1024 tokens) beats the anisotropic 8 x 4 (2048)
    assert patch_merges((128, 128)) == (0, 0)       # 4 x 4, 1024 tokens


def test_forward_shapes_and_history_padding():
    m = _randomise(HyPDE(TINY)).eval()
    geom = Geometry((32, 64), ((2, 2), (0, 1)), (0, 0))
    ids = hypde.layers.field_ids(["pressure", "velocity_x"])
    z = torch.randn(1, 6, 2, 32, 64)
    with torch.no_grad():
        y = m(z, ids, geom, 3)
        y2 = m(z[:, -3:], ids, geom, 3, k=2)
    assert y.shape == (1, 4, 2, 32, 64) and y2.shape == (1, 2, 2, 32, 64)
    assert torch.isfinite(y).all()


def test_periodic_translation_equivariance():
    m = _randomise(HyPDE(TINY), 1).eval()
    geom = Geometry((32, 32), ((2, 2), (2, 2)), (0, 0))
    ids = hypde.layers.field_ids(["concentration"])
    z = torch.randn(1, 6, 1, 32, 32)
    shift = 8                                       # two tokens of 4 cells
    with torch.no_grad():
        a = torch.roll(m(z, ids, geom, 0), shift, dims=-1)
        b = m(torch.roll(z, shift, dims=-1), ids, geom, 0)
    assert torch.allclose(a, b, atol=1e-5)


def test_window_norm():
    u = torch.randn(2, 6, 3, 8, 8) + 5.0
    n = WindowNorm.of(u)
    assert torch.allclose(n.sigma, u.square().mean((1, 3, 4)).sqrt())
    y = torch.zeros(2, 4, 3, 8, 8)
    assert torch.equal(n.step(u[:, -1], y), u[:, -1:].expand(2, 4, 3, 8, 8))


def test_load_trajectory(tmp_path):
    f = tmp_path / "toy.hdf5"
    T, X, Y = 102, 16, 16
    with h5py.File(f, "w") as h:
        h.attrs["dataset_name"] = "acoustic_scattering_discontinuous"
        h.create_group("dimensions").attrs["spatial_dims"] = ["x", "y"]
        g0 = h.create_group("t0_fields")
        g0["pressure"] = np.random.rand(2, T, X, Y).astype(np.float32)
        g0["density"] = np.random.rand(2, X, Y).astype(np.float32)
        g0["density"].attrs["sample_varying"] = True
        g0["speed_of_sound"] = np.random.rand(2, X, Y).astype(np.float32)
        g0["speed_of_sound"].attrs["sample_varying"] = True
        h.create_group("t1_fields")["velocity"] = np.random.rand(2, T, X, Y, 2).astype(np.float32)
        h.create_group("t2_fields")
    tr = hypde.load_trajectory(str(f), 1)
    assert tr.fields.shape == (T, 3, X, Y)
    with h5py.File(f, "r") as h:
        assert np.array_equal(tr.fields[:, 2].numpy(), h["t1_fields/velocity"][1, ..., 1])
        assert np.array_equal(tr.constants[1].numpy(), h["t0_fields/speed_of_sound"][1])
    assert float(tr.constants[0].abs().max()) == 0.0          # zeros_like_density


@pytest.mark.skipif(not (hypde.api.DEFAULT_WEIGHTS / "model.safetensors").exists(), reason="weights not downloaded")
def test_released_weights_load():
    import json
    cfg = json.loads((hypde.api.DEFAULT_WEIGHTS / "config.json").read_text())
    fc = hypde.load(device="cpu")
    assert fc.model.num_params() == cfg["parameters"]
