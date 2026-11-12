from __future__ import annotations

from dataclasses import dataclass
import importlib.util  # required before importing nvalchemiops 0.3.1
import math

import torch


NEIGHBOR_BACKEND_VERSION = "growing_spatial_bins_v2_shared_cell_geometry"


@torch.no_grad()
def dense_reference_edges(batch: dict, cutoff: float, block_size: int = 256) -> torch.Tensor:
    """Small-system nonperiodic O(N^2) correctness reference.

    Edges are directed sender -> receiver. Distinct physical samples never interact.
    The block size changes memory use only, not which interactions are modeled.
    """
    positions = batch["positions"].detach()
    sample = batch["batch"]
    n_atoms = len(positions)
    columns = torch.arange(n_atoms, device=positions.device)
    edges = []
    for begin in range(0, n_atoms, block_size):
        end = min(begin + block_size, n_atoms)
        displacement = positions[begin:end, None] - positions[None]
        distance2 = displacement.square().sum(-1)
        rows = torch.arange(begin, end, device=positions.device)
        keep = (distance2 < cutoff * cutoff) & (rows[:, None] != columns[None])
        keep &= sample[begin:end, None] == sample[None]
        receiver, sender = torch.where(keep)
        edges.append(torch.stack((sender, receiver + begin)))
    return torch.cat(edges, dim=1)


@dataclass
class PeriodicEdges:
    """Directed sender -> receiver with a lattice translation of the sender.

    Nonzero self images are physical neighbors; zero self images are reserved
    for inactive padding. Unit shifts remain integers even when a cell changes.
    """

    edge_index: torch.Tensor
    unit_shifts: torch.Tensor

    @property
    def shape(self):
        return self.edge_index.shape


def edge_geometry(batch, graph):
    edges = graph.edge_index if isinstance(graph, PeriodicEdges) else graph
    sender, receiver = edges
    displacement = batch["positions"][sender] - batch["positions"][receiver]
    valid = sender != receiver
    if isinstance(graph, PeriodicEdges):
        displacement = displacement + lattice_translation(batch,graph.unit_shifts,receiver,displacement.dtype)
        valid = valid | (graph.unit_shifts != 0).any(-1)
    return edges, displacement, valid


def lattice_translation(batch,unit_shifts,receiver,dtype):
    shifts=unit_shifts.to(dtype)
    if len(batch["ptr"])==2:
        # A physical MD box has one cell. Broadcasting it to millions of 3x3
        # matrices costs both memory and a pathological batched GEMV launch.
        # This shared matrix product keeps exact fixed-shift cell gradients.
        return shifts@batch["cell"][0]
    cells=batch["cell"][batch["batch"][receiver]]
    return torch.einsum("ei,eij->ej",shifts,cells)


@torch.no_grad()
def cell_list_edges(batch: dict, cutoff: float, initial_capacity: int | None = None):
    """GPU spatial cells with checked capacity, O(N + E) at bounded density.

    The library records the actual degree even when its buffer overflows. We
    retry with sufficient capacity before converting to COO; no edge is dropped.
    The spatial-bin budget grows with atom count, so a single growing physical
    system cannot run into the library's fixed 8192-bin default.
    Allocation, count synchronization, retries and COO conversion are included
    in callers' construction time. On the CPU, neighbor_pairs (numpy binning) is used.
    """
    positions = batch["positions"].detach().contiguous()
    periodic = "pbc" in batch and bool(batch["pbc"].any())
    if positions.device.type != "cuda":
        return _cpu_cell_list_edges(batch, cutoff, periodic)
    from nvalchemiops.torch.neighbors import neighbor_list
    from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes
    from nvalchemiops.torch.neighbors.batch_cell_list import estimate_batch_cell_list_sizes
    from nvalchemiops.torch.neighbors.neighbor_utils import (
        allocate_cell_list, get_neighbor_list_from_neighbor_matrix,
    )

    multiple = len(batch["ptr"]) > 2
    kwargs = {"method": "batch_cell_list" if multiple else "cell_list"}
    if multiple:
        kwargs.update(batch_idx=batch["batch"].int(), batch_ptr=batch["ptr"].int())
    if periodic:
        kwargs.update(cell=batch["cell"].detach().contiguous(), pbc=batch["pbc"].contiguous())
    else:
        # Version 0.3.1's convenience wrapper uses an int32 scatter index in
        # its batched bounding-box branch. Construct bounds with int64 here.
        groups = len(batch["ptr"]) - 1
        index = batch["batch"][:, None].expand_as(positions)
        lower = positions.new_full((groups, 3), float("inf")).scatter_reduce_(0, index, positions, reduce="amin")
        upper = positions.new_full((groups, 3), -float("inf")).scatter_reduce_(0, index, positions, reduce="amax")
        positions = (positions - lower[batch["batch"]]).contiguous()
        kwargs.update(cell=torch.diag_embed(upper - lower + 0.1 * cutoff),
                      pbc=torch.zeros((groups, 3), dtype=torch.bool, device=positions.device))
    # Passing every spatial buffer avoids the high-level wrapper's fixed bin
    # cap. The estimator still allocates only the bins the geometry requires.
    # A constant cap would eventually grow occupancy and approach all-pair
    # search, even at fixed density and cutoff.
    estimate = estimate_batch_cell_list_sizes if multiple else estimate_cell_list_sizes
    # The single-system estimator's Warp kernel expects a one-dimensional PBC
    # array, unlike the batched estimator (and despite its public docstring).
    estimate_pbc = kwargs["pbc"] if multiple else kwargs["pbc"].squeeze(0)
    total_cells, search_radius = estimate(
        kwargs["cell"], estimate_pbc, cutoff, max_nbins=max(8192, len(positions)),
    )
    storage = allocate_cell_list(len(positions), total_cells, search_radius, positions.device)
    kwargs.update(zip((
        "cells_per_dimension", "neighbor_search_radius", "atom_periodic_shifts",
        "atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_start_indices", "cell_atom_list",
    ), storage))
    capacity = initial_capacity or max(32, int(math.ceil(0.15 * 4 * math.pi / 3 * cutoff**3 / 32)) * 32)
    while True:
        matrix, counts, shifts = neighbor_list(positions, cutoff, max_neighbors=capacity, **kwargs)
        required = int(counts.max()) if len(counts) else 0
        if required <= capacity:
            break
        capacity = max(capacity * 2, ((required + 31) // 32) * 32)
        del matrix, counts, shifts
    native_edges, _, native_shifts = get_neighbor_list_from_neighbor_matrix(
        matrix, num_neighbors=counts, neighbor_shift_matrix=shifts,
        fill_value=len(positions),
    )
    # The library's row is central i and column is neighbor j. Our convention
    # is sender j -> receiver i; unit shifts translate j, so their sign stays.
    edges = native_edges.flip(0).long()
    return PeriodicEdges(edges, native_shifts) if periodic else edges


_BIN_OFFSETS = None


def neighbor_pairs(positions, cutoff, cell=None, pbc=(False, False, False)):
    """All pairs closer than ``cutoff`` on the CPU (numpy): receiver ``i``, sender ``j`` and integer
    lattice shift ``S`` with displacement ``x[j] - x[i] + S @ cell``.

    Periodic axes may be any length, including cells shorter than the cutoff (several images). Atoms
    are wrapped into the cell, periodic images within the cutoff of the cell are generated, and the
    pairs are found by binning at the cutoff length (O(N) at fixed density). The self pair with a zero
    shift is excluded; pairs are sorted by (i, j, S).
    """
    import numpy as np
    global _BIN_OFFSETS
    x = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = len(x)
    pbc = np.asarray(pbc, dtype=bool).reshape(3)
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3), np.int64))
    if n == 0:
        return empty
    if not (np.isfinite(cutoff) and cutoff > 0):
        raise ValueError('cutoff must be positive')
    if pbc.any():
        cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        volume = abs(float(np.linalg.det(cell)))
        if not np.isfinite(cell).all() or volume <= 0:
            raise ValueError('A periodic cell must be finite with nonzero volume')
        frac = x @ np.linalg.inv(cell)
        s0 = np.where(pbc, np.floor(frac), 0.).astype(np.int64)
        frac = frac - s0
        receivers = frac @ cell
        # Distance between lattice planes along each axis: |det| / |a_j x a_k|.
        heights = volume / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
        reach = np.where(pbc, np.ceil(cutoff / heights), 0).astype(np.int64)
        grids = np.meshgrid(*[np.arange(-r, r + 1) for r in reach], indexing='ij')
        images = np.stack(grids, -1).reshape(-1, 3)
        margin = cutoff / heights
        inside = np.ones((len(images), n), dtype=bool)
        for axis in np.flatnonzero(pbc):
            f = frac[None, :, axis] + images[:, None, axis]
            inside &= (f >= -margin[axis]) & (f <= 1 + margin[axis])
        image_index, sender_atom = np.nonzero(inside)
        sender_shift = images[image_index]
        senders = receivers[sender_atom] + sender_shift @ cell
    else:
        s0 = np.zeros((n, 3), np.int64)
        receivers = senders = x
        sender_atom = np.arange(n)
        sender_shift = np.zeros((n, 3), np.int64)
    lower = np.minimum(senders.min(0), receivers.min(0))
    bins_s = np.floor((senders - lower) / cutoff).astype(np.int64)
    bins_r = np.floor((receivers - lower) / cutoff).astype(np.int64)
    dims = np.maximum(bins_s.max(0), bins_r.max(0)) + 1
    key = (bins_s[:, 0] * dims[1] + bins_s[:, 1]) * dims[2] + bins_s[:, 2]
    order = np.argsort(key, kind='stable')
    sorted_key = key[order]
    if _BIN_OFFSETS is None:
        _BIN_OFFSETS = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing='ij'), -1).reshape(-1, 3)
    found_i, found_s = [], []
    for offset in _BIN_OFFSETS:
        neighbor = bins_r + offset
        valid = ((neighbor >= 0) & (neighbor < dims)).all(1)
        k = (neighbor[:, 0] * dims[1] + neighbor[:, 1]) * dims[2] + neighbor[:, 2]
        start = np.searchsorted(sorted_key, k, 'left')
        count = np.where(valid, np.searchsorted(sorted_key, k, 'right') - start, 0)
        total = int(count.sum())
        if not total:
            continue
        receiver = np.repeat(np.arange(n), count)
        first = np.repeat(np.cumsum(count) - count, count)
        sender = order[np.repeat(start, count) + np.arange(total) - first]
        d = senders[sender] - receivers[receiver]
        keep = np.einsum('ij,ij->i', d, d) < cutoff * cutoff
        keep &= ~((sender_atom[sender] == receiver) & (sender_shift[sender] == 0).all(1))
        found_i.append(receiver[keep])
        found_s.append(sender[keep])
    if not found_i:
        return empty
    i = np.concatenate(found_i)
    sender = np.concatenate(found_s)
    j = sender_atom[sender]
    shift = sender_shift[sender] - s0[j] + s0[i]
    order = np.lexsort((shift[:, 2], shift[:, 1], shift[:, 0], j, i))
    return i[order], j[order], shift[order]


def _cpu_cell_list_edges(batch, cutoff, periodic):
    import numpy as np

    ptr = batch["ptr"].cpu().numpy()
    positions = batch["positions"].detach().cpu().numpy()
    parts, shifts = [], []
    for g, (a, b) in enumerate(zip(ptr[:-1], ptr[1:])):
        pbc = batch["pbc"][g].cpu().numpy() if periodic else np.zeros(3, bool)
        cell = batch["cell"][g].detach().cpu().numpy() if periodic else None
        i, j, s = neighbor_pairs(positions[a:b], cutoff, cell, pbc)
        parts.append(np.stack((j + a, i + a)))
        shifts.append(s)
    edges = torch.as_tensor(np.concatenate(parts, axis=1), dtype=torch.long, device=batch["positions"].device)
    if periodic:
        return PeriodicEdges(edges, torch.as_tensor(np.concatenate(shifts), dtype=torch.int32, device=edges.device))
    return edges


@torch.no_grad()
def radius_edges(batch: dict, cutoff: float, block_size: int = 256, method="auto"):
    periodic = "pbc" in batch and bool(batch["pbc"].any())
    if method == "dense" and periodic:
        raise ValueError("The dense reference is nonperiodic; use the cell list for periodic images")
    if method == "cell_list" or periodic or (method == "auto" and len(batch["positions"]) >= 2048):
        return cell_list_edges(batch, cutoff)
    if method not in ("auto", "dense"):
        raise ValueError(method)
    return dense_reference_edges(batch, cutoff, block_size)


class NeighborCache:
    """Checked Verlet cache; periodic trajectories use unwrapped coordinates."""

    def __init__(self, cutoff: float, skin: float = 0.6):
        if skin <= 0:
            raise ValueError("skin must be positive")
        self.cutoff, self.skin = cutoff, skin
        self.reference = None
        self.edges = None
        self.signature = None
        self.rebuilds = 0
        self.cell = None

    @torch.no_grad()
    def get(self, batch: dict) -> torch.Tensor:
        if len(batch["ptr"]) != 2:
            raise ValueError("An MD neighbor cache requires one physical sample")
        signature = (tuple(batch["positions"].shape), batch["positions"].device)
        if self.signature is not None and signature != self.signature:
            raise ValueError("A neighbor cache belongs to one fixed system")
        # A switch between periodic and isolated (same atom count) also invalidates the edges.
        cell_changed = ("cell" in batch) != (self.cell is not None) or (
            "cell" in batch and not torch.equal(self.cell, batch["cell"]))
        moved = cell_changed or self.reference is None or bool(
            (batch["positions"] - self.reference).square().sum(-1).max() > (self.skin / 2) ** 2
        )
        if moved:
            self.edges = radius_edges(batch, self.cutoff + self.skin)
            self.reference = batch["positions"].detach().clone()
            self.signature = signature
            self.cell = batch["cell"].detach().clone() if "cell" in batch else None
            self.rebuilds += 1
        return self.edges


def energy_forces(model, batch: dict, edges=None, create_graph: bool = False):
    positions = batch["positions"]
    if not positions.requires_grad:
        positions = positions.detach().requires_grad_(True)
    inputs = {**batch, "positions": positions}
    with torch.enable_grad():
        energy = model.energy(inputs, edges)
        force = -torch.autograd.grad(energy.sum(), positions, create_graph=create_graph)[0]
    return energy, force
