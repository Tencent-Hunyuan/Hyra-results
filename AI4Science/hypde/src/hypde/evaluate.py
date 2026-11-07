"""Evaluation on the test panel: one-step, T[1:20] and T[21:60] VRMSE, as in Walrus's Table 1.

Per trajectory and channel, VRMSE = sqrt(mean_x (pred - ref)^2 / (var_x(ref) + 1e-7)) with the unbiased spatial
variance, computed in float64 (the_well's VRMSE as Walrus's evaluation calls it). Cells inside an obstacle mask are
zero in both prediction and reference.

  one-step  every start s >= 6: predict frame s from the true frames s-6..s-1 (only the first predicted frame is
            scored), mean over channels, then over starts
  T[a:b]    one autoregressive forecast from the true frames t0-6..t0-1, t0 = 17; frames a..b of it (1-based,
            inclusive), mean over channels, then over those frames; a trajectory shorter than a is not scored

The dataset score is the lower median over trajectories (torch.median's convention), a diverged (NaN) trajectory
counting as +inf.
"""
from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from .api import Forecaster
from .data import Trajectory, load_trajectory

EPS = 1e-7
T0 = 17
BRACKETS = {"t1_20": (1, 20), "t21_60": (21, 60)}
ROLLOUT_STEPS = 60


def vrmse(pred: torch.Tensor, ref: torch.Tensor) -> np.ndarray:
    """pred, ref (T, C, X, Y) -> (T, C)."""
    p, r = pred.double(), ref.double()
    mse = ((p - r) ** 2).mean(dim=(-2, -1))
    var = torch.std(r, dim=(-2, -1)) ** 2
    return torch.sqrt(mse / (var + EPS)).cpu().numpy()


def _ref(traj: Trajectory, lo: int, hi: int) -> torch.Tensor:
    r = traj.fields[lo:hi]
    return r if traj.mask is None else r.masked_fill(traj.mask, 0.0)


def forecast(fc: Forecaster, traj: Trajectory, t0: int = T0, steps: int = ROLLOUT_STEPS):
    """Autoregressive forecast from the true frames t0-6..t0-1 and its reference: (pred, ref), each (n, C, X, Y),
    n = min(steps, T - t0), obstacle cells zero in both."""
    n = min(steps, traj.fields.shape[0] - t0)
    pred = fc.rollout(traj.fields[t0 - 6:t0], traj.info, n, traj.constants).cpu()
    return pred, _ref(traj, t0, t0 + n)


def rollout_vrmse(fc: Forecaster, traj: Trajectory, t0: int = T0, steps: int = ROLLOUT_STEPS) -> np.ndarray:
    """(n, C) VRMSE of the forecast from t0."""
    return vrmse(*forecast(fc, traj, t0, steps))


def onestep_vrmse(fc: Forecaster, traj: Trajectory, first: int = 6) -> np.ndarray:
    """(n_starts, C) VRMSE of the one-frame forecast from every start s >= first. The model is called as in the
    rollout (4 frames per call) and only the first frame is scored."""
    T = traj.fields.shape[0]
    if first >= T:
        return np.zeros((0, len(traj.info.fields)))
    out = []
    for s in range(first, T):
        pred = fc.predict(traj.fields[s - 6:s], traj.info, traj.constants)[:1].cpu()
        out.append(vrmse(pred, _ref(traj, s, s + 1)))
    return np.concatenate(out)


def bracket(v: np.ndarray, first: int, last: int) -> float:
    if v.shape[0] < first:
        return math.nan
    return float(v[first - 1:last].mean(axis=-1).mean())


def lower_median(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    a = np.sort(np.where(np.isnan(a), np.inf, a))
    return float(a[(a.size - 1) // 2]) if a.size else math.nan


def test_panel() -> Dict:
    return json.loads(resources.files("hypde").joinpath("assets/test_panel.json").read_text())["datasets"]


def evaluate(fc: Forecaster, data_root: str, datasets: Optional[Iterable[str]] = None, max_traj: int = 0,
             onestep: bool = True, log=print) -> Dict:
    """Score the test panel. data_root holds The Well's `<dataset>/data/test/*.hdf5`. Returns
    {dataset: {"onestep", "t1_20", "t21_60": median, "per_trajectory": [...]}}."""
    panel = test_panel()
    results = {}
    for ds in datasets or panel:
        rows = []
        trajs = panel[ds]["trajectories"][:max_traj or None]
        for i, (file, idx) in enumerate(trajs):
            traj = load_trajectory(str(Path(data_root) / ds / "data" / "test" / file), idx, ds)
            v = rollout_vrmse(fc, traj)
            row = {"file": file, "index": idx, **{b: bracket(v, *w) for b, w in BRACKETS.items()}}
            if onestep:
                row["onestep"] = float(onestep_vrmse(fc, traj).mean(axis=-1).mean())
            rows.append(row)
            log(f"{ds} {i + 1}/{len(trajs)} {file}:{idx} " +
                " ".join(f"{k}={row[k]:.4f}" for k in ("onestep", *BRACKETS) if k in row))
        res = {b: lower_median(r[b] for r in rows if not math.isnan(r[b])) for b in BRACKETS}
        if onestep:
            res["onestep"] = lower_median(r["onestep"] for r in rows)
        res["n_trajectories"] = len(rows)
        res["per_trajectory"] = rows
        results[ds] = res
    return results
