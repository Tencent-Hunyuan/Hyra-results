from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

PHYSICAL_KEYS = ("positions", "atomic_numbers", "batch", "ptr")
OPTIONAL_PHYSICAL_KEYS = ("cell", "pbc", "total_charge", "formal_charges", "atom_mask")
PER_ATOM_LABELS = ("formal_charges", "mbis_charges", "mbis_dipoles", "mbis_quadrupoles")
EXTRA_FRAME_FIELDS = ("total_charge", "molecule_id", "total_energy_eV", "conformation_index")


class PackedDataset:
    """Read an existing split without modifying or copying source data."""

    def __init__(self, root: str | Path, domain: str, split: str):
        self.root, self.domain, self.split = Path(root), domain, split
        directory = self.root / domain / split
        self.arrays = {
            key: np.load(directory / f"{key}.npy", mmap_mode="r")
            for key in ("positions", "atomic_numbers", "atom_ptr", "energies", "forces")
        }
        for key in ("n_c", "angle_deg", "config_code", "triplet_id", "pair_id", "family_code", "component_code", "distance_factor", "min_intermonomer_distance_A", *PER_ATOM_LABELS, *EXTRA_FRAME_FIELDS):
            path = directory / f"{key}.npy"
            if path.exists():
                self.arrays[key] = np.load(path, mmap_mode="r")
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        self.reference = np.zeros(119, dtype=np.float64)
        for z, energy in self.metadata["domains"][domain]["element_reference_eV"].items():
            self.reference[int(z)] = float(energy)
        self.counts = np.diff(self.arrays["atom_ptr"])

    def __len__(self) -> int:
        return len(self.counts)

    def frame(self, index: int) -> dict:
        ptr = self.arrays["atom_ptr"]
        begin, end = int(ptr[index]), int(ptr[index + 1])
        record = {
            "positions": np.array(self.arrays["positions"][begin:end], dtype=np.float32),
            "atomic_numbers": np.array(self.arrays["atomic_numbers"][begin:end], dtype=np.int64),
            "forces": np.array(self.arrays["forces"][begin:end], dtype=np.float32),
            "energy": float(self.arrays["energies"][index]),
            "domain": self.domain,
            "index": int(index),
        }
        for key, array in self.arrays.items():
            if key not in ("positions", "atomic_numbers", "atom_ptr", "energies", "forces"):
                value = np.array(array[begin:end]) if key in PER_ATOM_LABELS else array[index]
                if isinstance(value,(bytes,np.bytes_)):
                    value=value.decode()
                record[key] = value.item() if isinstance(value, np.generic) else value
        return record

    def batch(self, indices, device="cpu") -> dict:
        return collate([self.frame(int(index)) for index in indices], device)

    def subset(self, count: int | None, seed: int) -> np.ndarray:
        """Random structure subset for a seed; smaller counts are prefixes of larger ones. Sampling is per structure, not per molecule."""
        order = np.random.default_rng(seed).permutation(len(self))
        return order[:len(self) if not count else min(count, len(self))]

    def triplets(self) -> np.ndarray:
        if self.domain != "interaction":
            raise ValueError("A/AB/B triplets are only defined for interaction data")
        result = []
        for value in np.unique(self.arrays["triplet_id"]):
            indices = np.flatnonzero(self.arrays["triplet_id"] == value)
            indices = indices[np.argsort(self.arrays["component_code"][indices])]
            if len(indices) != 3:
                raise ValueError(f"Incomplete triplet {value}")
            result.append(indices)
        return np.asarray(result, dtype=np.int64)


def collate(records: list[dict], device="cpu") -> dict:
    counts = np.asarray([len(record["positions"]) for record in records], dtype=np.int64)
    ptr = np.concatenate(([0], np.cumsum(counts)))
    result = {
        "positions": torch.tensor(np.concatenate([record["positions"] for record in records]), dtype=torch.float32, device=device),
        "atomic_numbers": torch.tensor(np.concatenate([record["atomic_numbers"] for record in records]), dtype=torch.long, device=device),
        "batch": torch.tensor(np.repeat(np.arange(len(records)), counts), dtype=torch.long, device=device),
        "ptr": torch.tensor(ptr, dtype=torch.long, device=device),
        "n_atoms": torch.tensor(counts, dtype=torch.float32, device=device),
        "records": records,
    }
    if all("energy" in record for record in records):
        result["target_energy"] = torch.tensor([record["energy"] for record in records], dtype=torch.float64, device=device)
    if all("forces" in record for record in records):
        result["target_forces"] = torch.tensor(np.concatenate([record["forces"] for record in records]), dtype=torch.float32, device=device)
    if any("cell" in record or "pbc" in record for record in records):
        if not all("cell" in record and "pbc" in record for record in records):
            raise ValueError("Supply cell and pbc together for every record in a periodic batch")
        result["cell"] = torch.as_tensor(np.stack([r["cell"] for r in records]), dtype=torch.float32, device=device)
        result["pbc"] = torch.as_tensor(np.stack([r["pbc"] for r in records]), dtype=torch.bool, device=device)
    if any("total_charge" in r for r in records):
        result["total_charge"] = torch.as_tensor([float(r.get("total_charge",0.)) for r in records], dtype=torch.float32, device=device)
    if any("formal_charges" in r for r in records):
        if any("formal_charges" not in r and r.get("total_charge",0)!=0 for r in records):
            raise ValueError("A nonzero total charge requires formal atomic charge sources")
        # Records without formal charges are neutral (checked above); their
        # atoms get zero formal charge.
        result["formal_charges"] = torch.as_tensor(np.concatenate([r.get("formal_charges",np.zeros(len(r["positions"]))) for r in records]), dtype=torch.float32, device=device)
    for name,shape in (("mbis_charges",()),("mbis_dipoles",(3,)),("mbis_quadrupoles",(3,3))):
        if any(name in r for r in records):
            values=np.concatenate([r.get(name,np.full((len(r["positions"]),*shape),np.nan,dtype=np.float32)) for r in records])
            result["target_"+name]=torch.as_tensor(values,dtype=torch.float32,device=device)
    if any("teacher_atomic_energy" in r for r in records):
        values=np.concatenate([r.get("teacher_atomic_energy",np.full(len(r["positions"]),np.nan,dtype=np.float32)) for r in records])
        result["target_teacher_atomic_energy"]=torch.as_tensor(values,dtype=torch.float32,device=device)
        weights={r["teacher_weight"] for r in records if "teacher_atomic_energy" in r}
        if len(weights)!=1:raise ValueError("Teacher weight must be constant in a training run")
        result["teacher_weight"]=torch.tensor(weights.pop(),dtype=torch.float32,device=device)
    return result


def physical(batch: dict) -> dict:
    return {key: batch[key] for key in (*PHYSICAL_KEYS, *OPTIONAL_PHYSICAL_KEYS) if key in batch}


def subset_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
