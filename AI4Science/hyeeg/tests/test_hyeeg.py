"""Unit tests for the standalone hyeeg inference package."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from hyeeg import HyEEGConfig, HyEEGEncoder, HyEEGModel, resolve_positions, window_normalize

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"

CHANNELS_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]


def test_resolve_positions_standard_and_bipolar():
    pos = resolve_positions(CHANNELS_19)
    assert pos.shape == (19, 3)
    assert pos.dtype == torch.float32
    assert torch.isfinite(pos).all()

    # Case-insensitivity, EEG prefix, -REF suffix, and legacy 10-20 aliases
    aliased = resolve_positions(["EEG fp1-REF", "t3", "T5", "cz"])
    canonical = resolve_positions(["Fp1", "T7", "P7", "Cz"])
    assert torch.allclose(aliased, canonical, atol=1e-6)

    # Bipolar pair maps to midpoint of constituent electrodes
    bipolar = resolve_positions(["Fp1-F7"])[0]
    expected_mid = 0.5 * (resolve_positions(["Fp1"])[0] + resolve_positions(["F7"])[0])
    assert torch.allclose(bipolar, expected_mid, atol=1e-6)


def test_window_normalize_flat_channel_and_clip():
    rng = np.random.default_rng(0)
    eeg = rng.normal(loc=5.0, scale=2.0, size=(6, 800)).astype(np.float32)
    eeg[0] = 3.14  # flat channel
    eeg[1, 10] = 1e5  # extreme outlier

    z = window_normalize(eeg)
    assert z.shape == (6, 800)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()
    assert float(z.max()) <= 15.0 and float(z.min()) >= -15.0
    # Flat channel stays zero instead of exploding
    assert float(z[0].abs().max()) < 1e-5


@pytest.mark.parametrize(
    "scale,expected_depth,expected_params",
    [
        ("small", 1, 3_247_616),
        ("base", 3, 9_527_808),
        ("large", 8, 25_228_288),
    ],
)
def test_model_configs_and_shapes(scale: str, expected_depth: int, expected_params: int):
    cfg = HyEEGConfig.from_json(WEIGHTS_DIR / scale / "config.json")
    assert cfg.depth == expected_depth
    model = HyEEGModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == expected_params

    eeg = torch.randn(2, 19, 800)  # 4 s at 200 Hz -> 4 overlapping 1 s patches
    pos = resolve_positions(CHANNELS_19).unsqueeze(0).expand(2, -1, -1)
    tokens = model(eeg, pos)
    assert tokens.shape == (2, 19, 4, 512)
    pooled = model.attention_pooling(tokens)
    assert pooled.shape == (2, 512)

    all_layers = model(eeg, pos, return_all_layers=True)
    assert len(all_layers) == expected_depth + 1
    assert torch.allclose(all_layers[-1], tokens)


@pytest.mark.parametrize("scale", ["small", "base", "large"])
def test_load_released_weights_if_present(scale: str):
    weights_file = WEIGHTS_DIR / scale / "model.safetensors"
    if not weights_file.is_file():
        pytest.skip(f"{weights_file} not downloaded; run `bash weights/download.sh {scale}`")
    enc = HyEEGEncoder.from_pretrained(WEIGHTS_DIR / scale, device="cpu", precision="fp32")
    eeg = np.random.default_rng(42).normal(size=(2, 19, 800)).astype(np.float32)
    out = enc.encode(eeg, channels=CHANNELS_19, pooling="attention")
    assert out.shape == (2, 512)
    assert torch.isfinite(out).all()
