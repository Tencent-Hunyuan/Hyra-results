"""Lattice translations of bonded groups with unwrapped-coordinate bookkeeping.

Used when recentering is enabled; the barostat scales the same groups. Groups are connected
components of the constraints plus physical bonds (an OpenMM system.xml, or ``bond_pairs`` in the
system file), never spatial proximity. Constrained atoms always move together, so RATTLE is
unaffected; the periodic force is invariant to the translations.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch


def pairs_from_xml(path, n_atoms):
    """(B, 2) atom pairs of the constraints and harmonic bonds of an OpenMM system.xml."""
    root = ET.parse(path).getroot()
    if len(root.findall('./Particles/Particle')) != n_atoms:
        raise ValueError('Molecular connectivity and MD atom counts differ')
    edges = list(root.findall('./Constraints/Constraint'))
    edges += root.findall('./Forces/Force[@type="HarmonicBondForce"]/Bonds/Bond')
    if not edges:
        raise ValueError('No physical bonds or constraints found; refusing to infer molecules')
    pairs = np.array([(int(edge.attrib['p1']), int(edge.attrib['p2'])) for edge in edges], dtype=np.int64)
    if not ((pairs >= 0) & (pairs < n_atoms)).all():
        raise ValueError('Bond atom index outside the MD system')
    return pairs


def molecule_ids_from_xml(path, n_atoms):
    return molecule_ids_from_pairs(n_atoms, pairs_from_xml(path, n_atoms))


def molecule_ids_from_pairs(n_atoms, *pair_arrays):
    """Connected components of the given atom pairs (constraints, bonds); unpaired atoms are their own group."""
    parent = np.arange(n_atoms)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pairs in pair_arrays:
        if pairs is None:
            continue
        for i, j in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
            if not 0 <= i < n_atoms or not 0 <= j < n_atoms:
                raise ValueError('Bond atom index outside the MD system')
            parent[find(j)] = find(i)
    _, ids = np.unique([find(i) for i in range(n_atoms)], return_inverse=True)
    return ids.astype(np.int64)


class PeriodicMolecules:
    def __init__(self, molecule_ids, cell_A, device):
        ids = np.asarray(molecule_ids, dtype=np.int64)
        unique, first = np.unique(ids, return_index=True)
        if not np.array_equal(unique, np.arange(len(unique))):
            raise ValueError('Molecule ids must be consecutive from zero')
        self.ids = torch.as_tensor(ids, device=device, dtype=torch.long)
        self.anchors = torch.as_tensor(first, device=device, dtype=torch.long)
        # Use exactly the periodic cell seen by FP32 force kernels. Promoting
        # this rounded cell to FP64 makes the lattice translation reproducible.
        self.cell = torch.as_tensor(cell_A, dtype=torch.float32, device=device).double().reshape(3,3)
        self.inverse = torch.linalg.inv(self.cell)
        self.images = torch.zeros((len(unique),3), dtype=torch.long, device=device)

    def set_cell(self, cell_A):
        """Constant-pressure MD: later translations and unwrapping use the current cell."""
        self.cell = torch.as_tensor(cell_A, dtype=torch.float32, device=self.cell.device).double().reshape(3, 3)
        self.inverse = torch.linalg.inv(self.cell)

    @torch.no_grad()
    def recenter(self, positions):
        if positions.dtype != torch.float64:
            raise TypeError('Recenter integrator coordinates in FP64 before force conversion')
        before = float(positions.abs().max())
        shifts = torch.floor(positions[self.anchors] @ self.inverse + .5).to(torch.long)
        positions.sub_((shifts.double() @ self.cell)[self.ids])
        self.images.add_(shifts)
        return dict(molecules_shifted=int(shifts.ne(0).any(1).sum()),
                    max_abs_coordinate_before_A=before,
                    max_abs_coordinate_after_A=float(positions.abs().max()))

    def unwrapped(self, positions):
        return positions + (self.images.double() @ self.cell)[self.ids]
