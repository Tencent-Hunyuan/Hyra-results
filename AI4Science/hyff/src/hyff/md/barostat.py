"""Monte Carlo barostat: constant-pressure MD by molecular-scaling volume moves at chunk boundaries.

A move scales the cell and the mass-weighted centres of the bonded groups (connected components of the
constraints plus physical bonds), so every group keeps its internal geometry and the constraints stay
exact. It is accepted with probability min(1, exp(-w / kT)),

    w = dE + P dV - N kT ln(V'/V),

N being the number of scaled groups. Velocities are unchanged. Isotropic moves scale the whole cell;
semi-isotropic moves scale either x and y together or z; anisotropic moves scale one of x, y or z
(orthorhombic cells only). Each coupled set of axes adapts its own maximum volume change towards an
acceptance rate between 25% and 75%, as in OpenMM's MonteCarloBarostat.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch

from hyff.md.units import KB_EV_K

BAR_A3_EV = 1e5 * 1e-30 / 1.602176634e-19        # eV per bar A^3
AMU_A3_G_CM3 = 1.66053906660                      # g/cm^3 per amu/A^3
MODES = {'isotropic': ((0, 1, 2),), 'semi-isotropic': ((0, 1), (2,)), 'anisotropic': ((0,), (1,), (2,))}


def engine_cell(cell):
    """The cell as the FP32 force kernels hold it, promoted to FP64."""
    return np.asarray(cell, dtype=np.float32).astype(np.float64).reshape(3, 3)


def is_orthorhombic(cell):
    cell = np.asarray(cell, dtype=np.float64)
    return bool(np.all(cell[~np.eye(3, dtype=bool)] == 0))


def make_whole(positions, cell, n_atoms, pair_arrays):
    """Move atoms by lattice vectors so that every bonded group is contiguous (minimum image along each bond).

    The periodic energy is unchanged. Returns the new positions and the number of atoms moved.
    """
    x = np.array(positions, dtype=np.float64)
    inverse = np.linalg.inv(cell)
    neighbours = [[] for _ in range(n_atoms)]
    for pairs in pair_arrays:
        if pairs is None:
            continue
        for i, j in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
            neighbours[i].append(j)
            neighbours[j].append(i)
    seen = np.zeros(n_atoms, dtype=bool)
    for root in range(n_atoms):
        if seen[root] or not neighbours[root]:
            continue
        seen[root] = True
        queue = deque([root])
        while queue:
            i = queue.popleft()
            for j in neighbours[i]:
                if seen[j]:
                    continue
                d = x[j] - x[i]
                x[j] = x[i] + d - np.round(d @ inverse) @ cell
                seen[j] = True
                queue.append(j)
    moved = int(np.any(np.abs(x - positions) > 1e-9, axis=1).sum())
    return x, moved


class MonteCarloBarostat:
    def __init__(self, pressure_bar, temperature_K, mode, group_ids, masses, cell, rng, device):
        if mode not in MODES:
            raise ValueError(f'barostat must be one of {", ".join(MODES)}')
        if mode != 'isotropic' and not is_orthorhombic(cell):
            raise ValueError(f'the {mode} barostat needs an orthorhombic (diagonal) box; use isotropic')
        ids = np.asarray(group_ids, dtype=np.int64)
        self.pressure_bar, self.temperature_K, self.mode, self.rng = float(pressure_bar), float(temperature_K), mode, rng
        self.kT = KB_EV_K * self.temperature_K
        self.ids = torch.as_tensor(ids, device=device)
        self.n_groups = int(ids.max()) + 1
        self.atom_mass = torch.as_tensor(masses, dtype=torch.float64, device=device)
        self.group_mass = torch.zeros(self.n_groups, dtype=torch.float64, device=device).index_add_(
            0, self.ids, self.atom_mass)
        volume = abs(float(np.linalg.det(cell)))
        self.axis_sets = MODES[mode]
        self.max_dV = [0.01 * volume] * len(self.axis_sets)
        self.window = [[0, 0] for _ in self.axis_sets]       # attempts, acceptances since the last adaptation
        self.attempts = self.acceptances = 0

    def propose(self, cell):
        """A trial cell (FP32-representable) and the linear map M with new = old @ M."""
        k = int(self.rng.integers(len(self.axis_sets)))
        volume = abs(float(np.linalg.det(cell)))
        dV = self.max_dV[k] * (2 * self.rng.random() - 1)
        factor = ((volume + dV) / volume) ** (1 / len(self.axis_sets[k]))
        scale = np.ones(3)
        scale[list(self.axis_sets[k])] = factor
        if self.mode == 'isotropic':
            new = engine_cell(cell * factor)
        else:
            new = engine_cell(cell * scale[None, :])
        return k, new, np.linalg.solve(cell, new)

    @torch.no_grad()
    def scaled_positions(self, positions, M):
        """Positions with the group centres mapped by M (row vectors: x' = x M) and internal geometry kept."""
        M = torch.as_tensor(M, dtype=torch.float64, device=positions.device)
        weighted = torch.zeros((self.n_groups, 3), dtype=torch.float64, device=positions.device)
        weighted.index_add_(0, self.ids, positions * self.atom_mass[:, None])
        centres = weighted / self.group_mass[:, None]
        return positions + (centres @ M - centres)[self.ids]

    def work(self, dE, old_volume, new_volume):
        return dE + self.pressure_bar * BAR_A3_EV * (new_volume - old_volume) - \
            self.n_groups * self.kT * math.log(new_volume / old_volume)

    def decide(self, k, dE, old_volume, new_volume):
        w = self.work(dE, old_volume, new_volume)
        accepted = bool(math.isfinite(w) and (w <= 0 or self.rng.random() < math.exp(-w / self.kT)))
        self.attempts += 1
        self.acceptances += accepted
        window = self.window[k]
        window[0] += 1
        window[1] += accepted
        if window[0] >= 10:
            if window[1] < 0.25 * window[0]:
                self.max_dV[k] /= 1.1
            elif window[1] > 0.75 * window[0]:
                self.max_dV[k] = min(self.max_dV[k] * 1.1, 0.3 * new_volume)
            self.window[k] = [0, 0]
        return accepted

    def description(self):
        return dict(kind='Monte Carlo, molecular scaling', mode=self.mode, pressure_bar=self.pressure_bar,
                    temperature_K=self.temperature_K, scaled_groups=self.n_groups, attempts=self.attempts,
                    acceptances=self.acceptances, max_volume_change_A3=[float(v) for v in self.max_dV])
