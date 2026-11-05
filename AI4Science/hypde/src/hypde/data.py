"""The 13 datasets HyPDE was trained and evaluated on, and reading their trajectories from The Well's HDF5 files.

`load_trajectory(path, index, dataset)` returns the predicted fields as a (T, C, X, Y) float32 tensor in the model's
channel order, the constant inputs (Cc, X, Y), the obstacle mask (X, Y) or None, and the dataset description.
Field names follow Walrus's renaming of The Well's fields: a vector field `velocity` on axes (x, y) becomes
`velocity_x`, `velocity_y`; a tensor field `D` becomes `D_xx`, `D_xy`, `D_yx`, `D_yy`; the acoustic sets'
constant `density` enters as `zeros_like_density` (all zeros), as in Walrus's released configuration. Trajectories
are cut to the dataset's evaluation length (`DatasetInfo.n_frames`), the shortest stored length of the split, as
Walrus's evaluation loader does.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class DatasetInfo:
    """What the model needs to know about a dataset besides its fields.

    name        directory name in The Well
    dataset_id  index of the dataset-id embedding
    fields      predicted fields, in the model's channel order
    constants   constant inputs (time-independent fields), in the model's order
    bcs         ((x_lo, x_hi), (y_lo, y_hi)) boundary codes: 0 wall, 1 open, 2 periodic
    grid        spatial size (X, Y)
    n_frames    frames per trajectory used for evaluation (the shortest stored length of the split)
    """
    name: str
    dataset_id: int
    fields: tuple
    constants: tuple
    bcs: tuple
    grid: tuple
    n_frames: int


def _load_datasets():
    raw = json.loads(resources.files("hypde").joinpath("assets/datasets.json").read_text())
    return {k: DatasetInfo(k, v["dataset_id"], tuple(v["fields"]), tuple(v["constants"]),
                           tuple(tuple(r) for r in v["bcs"]), tuple(v["grid"]), int(v["n_frames"]))
            for k, v in raw.items()}


DATASETS = _load_datasets()


def dataset(name: str) -> DatasetInfo:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}") from None


@dataclass
class Trajectory:
    fields: torch.Tensor                 # (T, C, X, Y) float32
    constants: Optional[torch.Tensor]    # (Cc, X, Y) float32, or None
    mask: Optional[torch.Tensor]         # (X, Y) bool, True inside obstacles, or None
    info: DatasetInfo


def _read_field(h5, name: str, index: int, n_frames: int, axes) -> np.ndarray:
    """One named component of a time-varying field: (T, X, Y)."""
    g0 = h5["t0_fields"]
    if name in g0:
        return g0[name][index, :n_frames]
    base, _, comp = name.rpartition("_")
    if base in h5["t1_fields"] and comp in axes:
        return h5["t1_fields"][base][index, :n_frames, ..., axes.index(comp)]
    if base in h5["t2_fields"] and len(comp) == 2 and comp[0] in axes and comp[1] in axes:
        return h5["t2_fields"][base][index, :n_frames, ..., axes.index(comp[0]), axes.index(comp[1])]
    raise KeyError(f"field {name!r} not found in {h5.filename}")


def _read_constant(h5, name: str, index: int) -> np.ndarray:
    src = name[len("zeros_like_"):] if name.startswith("zeros_like_") else name
    d = h5["t0_fields"][src]
    a = d[index] if d.attrs.get("sample_varying", False) else d[()]
    return np.zeros_like(a) if name.startswith("zeros_like_") else a


def load_trajectory(path: str, index: int, name: Optional[str] = None) -> Trajectory:
    """Trajectory `index` of a Well HDF5 file. `name` (the dataset) defaults to the file's `dataset_name`."""
    import h5py
    with h5py.File(path, "r") as h5:
        info = dataset(name or str(h5.attrs["dataset_name"]))
        axes = [a.decode() if isinstance(a, bytes) else str(a) for a in h5["dimensions"].attrs["spatial_dims"]]
        u = np.stack([_read_field(h5, n, index, info.n_frames, axes) for n in info.fields], axis=1)
        const = [_read_constant(h5, n, index) for n in info.constants]
    if u.shape[0] != info.n_frames:
        raise ValueError(f"{path}:{index} has {u.shape[0]} frames, {info.name} is evaluated on {info.n_frames}")
    c = torch.from_numpy(np.stack(const).astype(np.float32)) if const else None
    mask = None
    if "mask" in info.constants:
        mask = c[info.constants.index("mask")] != 0
    return Trajectory(torch.from_numpy(u.astype(np.float32)), c, mask, info)
