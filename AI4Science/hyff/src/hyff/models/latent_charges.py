"""Latent-charge long-range term: learned, charge-conserving atomic charges with a screened Coulomb energy."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class LatentChargeEnergy(nn.Module):
    def __init__(self, feature_dim: int, sigma_initial: float = 1.0,
                 sigma_min: float = 1e-3, dense_limit: int = 4096,
                 block_rows: int = 256, coupling_ev_angstrom: float = 14.399645):
        super().__init__()
        self.charge_head = nn.Linear(feature_dim, 1, bias=False, dtype=torch.float32)
        # Only the energy scale is zero. A zero charge head would trap neutral frames.
        nn.init.normal_(self.charge_head.weight, std=0.01 / math.sqrt(feature_dim))
        # Dimensionless s: k = coupling_ev_angstrom * s; s=1 is physical Coulomb.
        self.scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.coupling_ev_angstrom = float(coupling_ev_angstrom)
        self.sigma_raw = nn.Parameter(torch.tensor(
            math.log(math.expm1(sigma_initial - sigma_min)), dtype=torch.float32))
        self.sigma_min = sigma_min
        self.dense_limit = dense_limit
        self.block_rows = block_rows

    @property
    def sigma(self):
        return F.softplus(self.sigma_raw) + self.sigma_min

    def conserved_charges(self, scalars, atom_batch, total_charge):
        raw = self.charge_head(scalars.float()).squeeze(-1)
        groups = total_charge.numel()
        sums = raw.new_zeros(groups).index_add(0, atom_batch, raw)
        counts = raw.new_zeros(groups).index_add(0, atom_batch, torch.ones_like(raw))
        shifts = (total_charge.float().reshape(-1) - sums) / counts.clamp_min(1)
        return raw + shifts[atom_batch]

    def _screened_kernel(self, distance_squared):
        sigma = self.sigma
        x = distance_squared / (2 * sigma.square())
        # Smooth coincident limit, with finite first AND second derivatives.
        series = (math.sqrt(2 / math.pi) / sigma) * (
            1 - x / 3 + x.square() / 10 - x.pow(3) / 42)
        u = x.clamp_min(1e-3).sqrt()
        regular = torch.erf(u) / (math.sqrt(2) * sigma * u)
        return torch.where(x < 1e-3, series, regular)

    def forward(self, scalars, positions, atom_batch, total_charge):
        with torch.autocast(device_type=positions.device.type, enabled=False):
            positions = positions.float()
            atom_batch = atom_batch.long()
            charge = self.conserved_charges(scalars, atom_batch, total_charge)
            atoms = positions.shape[0]
            columns = torch.arange(atoms, device=positions.device)
            energy = positions.new_zeros(total_charge.numel())
            rows_per_block = atoms if atoms <= self.dense_limit else self.block_rows
            for begin in range(0, atoms, rows_per_block):
                end = min(begin + rows_per_block, atoms)
                rows = columns[begin:end]
                delta = positions[begin:end, None, :] - positions[None, :, :]
                kernel = self._screened_kernel(delta.square().sum(dim=-1))
                pair_mask = ((rows[:, None] < columns[None, :]) &
                             (atom_batch[begin:end, None] == atom_batch[None, :]))
                pair_energy = (charge[begin:end, None] * charge[None, :] *
                               kernel * pair_mask.to(kernel.dtype))
                energy = energy.index_add(0, atom_batch[begin:end], pair_energy.sum(dim=1))
            return (self.coupling_ev_angstrom * self.scale) * energy

