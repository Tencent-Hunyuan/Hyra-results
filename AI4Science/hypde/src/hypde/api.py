"""Loading the released weights and forecasting in physical units.

Normalisation (Walrus's RMS normalisation, recomputed at every model call). For a call whose input window is
u (B, T, C, X, Y), per sample and channel over time and all cells:

    sigma   = max(sqrt(mean u^2), 1e-5)                         model input z = u / sigma (not centred)
    s_delta = max(sqrt(mean (u[t+1] - u[t])^2), 1e-5)           (1 for a single-frame window)
    next frames u[T-1] + s_delta * y_k, k = 1..K

During a rollout the window contains the model's own predictions, and the statistics are recomputed from it at
every call. Constant inputs are divided by their own spatial RMS (floored at 1e-5); the obstacle mask and
`zeros_like_*` stay as they are.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from .data import DatasetInfo, dataset as dataset_info
from .layers import Geometry, field_ids, patch_merges, tf32 as tf32_flags
from .model import HyPDE, HyPDEConfig

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "weights"
EPS = 1e-5


def sha256(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


@dataclass
class WindowNorm:
    sigma: torch.Tensor      # (B, C)
    s_delta: torch.Tensor    # (B, C)

    @staticmethod
    def of(u: torch.Tensor) -> "WindowNorm":
        with torch.no_grad(), torch.autocast(u.device.type, enabled=False):
            x = u.float()
            dims = (1, 3, 4)
            eps = torch.tensor(EPS, dtype=x.dtype, device=x.device)
            sigma = torch.maximum(x.square().mean(dims).sqrt(), eps)
            if x.shape[1] > 1:
                s_delta = torch.maximum((x[:, 1:] - x[:, :-1]).square().mean(dims).sqrt(), eps)
            else:
                s_delta = torch.ones_like(sigma)
        return WindowNorm(sigma, s_delta)

    def z(self, u: torch.Tensor) -> torch.Tensor:
        """(B, T, C, X, Y) physical -> normalised model input."""
        return u.float() / self.sigma[:, None, :, None, None]

    def step(self, last: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """last (B, C, X, Y) physical, y (B, K, C, X, Y) model output -> (B, K, C, X, Y) physical frames."""
        return last.float()[:, None] + self.s_delta[:, None, :, None, None] * y.float()


def normalize_constants(const: torch.Tensor, names) -> torch.Tensor:
    """(B, Cc, X, Y) constant inputs -> model inputs (computed in fp64, returned in fp32)."""
    out = const.float().clone()
    for i, n in enumerate(names):
        if n == "mask" or n.startswith("zeros_like"):
            continue
        c = const[:, i].double()
        r = c.square().mean((1, 2)).sqrt().clamp_min(EPS)
        out[:, i] = (c / r.view(-1, 1, 1)).float()
    return out


class Forecaster:
    """HyPDE with its normalisation: physical frames in, physical frames out.

    tf32: allow TF32 in the processor on GPU (the recorded results were measured this way). The patch embedding and
    the refiner, which see the uncentred input, always run with TF32 off.
    """

    def __init__(self, model: HyPDE, device: torch.device, tf32: bool = True):
        self.model = model.to(device).eval()
        self.device = device
        self.tf32 = bool(tf32)

    @torch.no_grad()
    def predict(self, history: torch.Tensor, dataset: Union[str, DatasetInfo], constants: torch.Tensor = None,
                k: Optional[int] = None) -> torch.Tensor:
        """Next k (default 4) frames after `history`.

        history (T, C, X, Y) or (B, T, C, X, Y), physical units, channels in the order of `dataset.fields`; the last
        6 frames are used. constants (Cc, X, Y) or (B, Cc, X, Y) in the order of `dataset.constants`.
        Returns (k, C, X, Y) or (B, k, C, X, Y) float32. Cells inside an obstacle mask are set to zero.
        """
        info = dataset if isinstance(dataset, DatasetInfo) else dataset_info(dataset)
        batched = history.dim() == 5
        u = (history if batched else history[None]).to(self.device, torch.float32)
        u = u[:, -self.model.cfg.history:]
        B, _, C = u.shape[:3]
        if C != len(info.fields):
            raise ValueError(f"{info.name}: {C} channels given, the dataset has {len(info.fields)} {info.fields}")
        const, mask = self._constants(constants, info, B)
        grid = tuple(u.shape[3:])
        geom = Geometry(grid, info.bcs, patch_merges(grid))
        norm = WindowNorm.of(u)
        ids = field_ids(info.fields, self.device)
        cids = field_ids(info.constants, self.device) if const is not None else None
        with tf32_flags(self.tf32 and self.device.type == "cuda"):
            y = self.model(norm.z(u), ids, geom, info.dataset_id, const, cids, k)
        out = norm.step(u[:, -1], y)
        if mask is not None:
            out = out.masked_fill(mask[:, None, None], 0.0)
        return out if batched else out[0]

    def _constants(self, constants, info: DatasetInfo, B: int):
        if not info.constants:
            return None, None
        if constants is None:
            raise ValueError(f"{info.name} needs the constant inputs {info.constants}")
        c = constants.to(self.device, torch.float32)
        c = c if c.dim() == 4 else c[None].expand(B, *c.shape)
        mask = c[:, info.constants.index("mask")] > 0.5 if "mask" in info.constants else None
        return normalize_constants(c, info.constants), mask

    @torch.no_grad()
    def rollout(self, history: torch.Tensor, dataset: Union[str, DatasetInfo], steps: int,
                constants: torch.Tensor = None) -> torch.Tensor:
        """Autoregressive forecast of `steps` frames: each call predicts 4 frames from the last 6 (true frames
        first, then the model's own), and the window slides by the frames predicted. Returns (steps, C, X, Y) or
        (B, steps, C, X, Y)."""
        batched = history.dim() == 5
        window = (history if batched else history[None]).to(self.device, torch.float32)
        window = window[:, -self.model.cfg.history:]
        preds, n = [], 0
        while n < steps:
            out = self.predict(window, dataset, constants)[:, :steps - n]
            preds.append(out)
            n += out.shape[1]
            window = torch.cat([window, out], 1)[:, -self.model.cfg.history:]
        out = torch.cat(preds, 1)
        return out if batched else out[0]


def load(weights: Union[str, os.PathLike] = DEFAULT_WEIGHTS, device=None, tf32: bool = True,
         verify: bool = False) -> Forecaster:
    """The released HyPDE forecaster. weights: a directory holding model.safetensors and config.json (default: the
    package's `weights/`, filled by `weights/download.sh`). verify: check the file's sha256 against config.json."""
    from safetensors.torch import load_file
    weights = Path(weights)
    cfg = json.loads((weights / "config.json").read_text())
    st = weights / "model.safetensors"
    if not st.exists():
        raise FileNotFoundError(f"{st} is missing; run weights/download.sh first")
    if verify and sha256(st) != cfg["sha256"]:
        raise ValueError(f"{st}: sha256 differs from config.json")
    model = HyPDE(HyPDEConfig(**cfg["architecture"]))
    model.load_state_dict(load_file(str(st)), strict=True)
    return Forecaster(model, torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")), tf32)
