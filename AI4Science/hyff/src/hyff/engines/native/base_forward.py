"""Native-engine forward of the state models: general pair-term stride, optional owned-atom energy weights."""
"""The global state enters only through the invariant initial scalar (``initial_scalar``)."""
import math
from typing import Dict
import torch
from hyff.models.l2_potential import _quadrupole, smooth_switch


def reference_forward(self, batch: Dict[str, torch.Tensor], csr=None, stable_sum=False,
            prefix_depth: int | None = None,
            return_prefix_depth: int | None = None):
    # An optional block prefix is an integrator energy group. It uses the same
    # readout scale and pair term; without a prefix the full energy is returned.
    if prefix_depth is not None and (not isinstance(prefix_depth, int) or
                                    not 1 <= prefix_depth <= len(self.blocks)):
        raise ValueError('The energy prefix must contain 1 through all interaction blocks')
    if return_prefix_depth is not None:
        if prefix_depth is not None:
            raise ValueError('Request a prefix or the full energy with a prefix, not both')
        if not isinstance(return_prefix_depth, int) or not 1 <= return_prefix_depth <= len(self.blocks):
            raise ValueError('Invalid jointly evaluated prefix depth')
    positions = batch["positions"]
    if csr is not None:
        csr.refresh_subgraphs(positions, prefix_depth or len(self.blocks))
    edge_index = batch["edge_index"]
    source, destination = edge_index[0], edge_index[1]
    if csr is None:
        displacement = positions[source] - positions[destination]
        if "shifts" in batch:
            displacement = displacement + batch["shifts"]
        # Inactive padded edges have a nonzero harmless displacement so the
        # norm Hessian remains defined during force-loss differentiation.
        valid = source != destination
        if 'edge_valid' in batch:
            valid = batch['edge_valid']
        elif 'shifts' in batch:
            valid = valid | (batch['shifts'] != 0).any(-1)
        displacement = torch.where(valid[:, None], displacement, self.padding_displacement)
        distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-9)
        direction = displacement / distance[:, None]
        quadrupole = _quadrupole(direction)
        radial = self.rbf(distance)
        envelope = smooth_switch(distance, self.cutoff, self.config.switch_width) * valid.to(distance.dtype)
    else:
        # Prepared kernels consume current coordinates directly. In this
        # path no E-by-radial or E-by-angular PyTorch arrays are needed.
        radial = direction = quadrupole = envelope = None

    atomic_numbers = batch["atomic_numbers"]
    scalar = self.initial_scalar(batch)
    vector = scalar.new_zeros((scalar.shape[0], 3, self.tensor_hidden))
    tensor = scalar.new_zeros((scalar.shape[0], 5, self.tensor_hidden))
    atomic_energy = scalar.new_zeros((scalar.shape[0],))
    prefix_atomic_energy = None
    readout_scale = 1.0 / math.sqrt(len(self.blocks))
    for block_index, (block, norm, readout) in enumerate(zip(
        self.blocks, self.readout_norms, self.readouts
    )):
        if prefix_depth is not None and block_index >= prefix_depth:
            break
        scalar, vector, tensor = block(
            scalar,
            vector,
            tensor,
            radial,
            direction,
            quadrupole,
            source,
            destination,
            (envelope if self.layer_cutoffs[block_index] == self.cutoff else
             smooth_switch(distance, self.layer_cutoffs[block_index], self.config.switch_width) * valid)
                if csr is None else None,
            self.radial_tables[block_index](distance) if self.radial_tables is not None and csr is None and not self.reference_chunk_size else None,
            (positions, csr.radius_view(self.layer_cutoffs[block_index]), self.radial_tables[block_index]) if csr is not None else None,
            (displacement,valid,self.radial_tables[block_index] if self.radial_tables is not None else None,
             self.rbf,self.layer_cutoffs[block_index],self.config.switch_width,self.reference_chunk_size)
            if self.reference_chunk_size and csr is None else None,
            self.frozen_first_messages[atomic_numbers] if block_index == 0 and
                hasattr(self, 'frozen_first_messages') else None,
        )
        if block_index < len(self.refinements):
            for refinement in self.refinements[block_index]:
                scalar, vector, tensor = refinement(scalar, vector, tensor)
        atomic_energy = atomic_energy + readout_scale * readout(
            norm(scalar)
        ).squeeze(-1)
        if block_index + 1 == return_prefix_depth:
            prefix_atomic_energy = atomic_energy

    elemental = self.element_offset(atomic_numbers).squeeze(-1)
    atomic_energy = atomic_energy + elemental
    if prefix_atomic_energy is not None:
        prefix_atomic_energy = prefix_atomic_energy + elemental
    if stable_sum:
        # Extensive energies can sum hundreds of thousands of signed pair
        # terms. Accumulate them in FP64; keep learned operations/forces at
        # the model dtype. Without stable_sum the sum keeps the training order.
        atomic_energy = atomic_energy.double()
        if prefix_atomic_energy is not None:
            prefix_atomic_energy = prefix_atomic_energy.double()

    graph_ids = batch["batch"]
    n_graphs = batch["ptr"].numel() - 1
    elements = self.element_lut[atomic_numbers]
    if csr is not None:
        from hyff.kernels.pair_energy import pair_energy
        atomic_pair = pair_energy(positions, csr.radius_view(self.cutoff), elements, self.radial_tables[-1])
        atomic_energy = atomic_energy + atomic_pair
        if prefix_atomic_energy is not None:
            prefix_atomic_energy = prefix_atomic_energy + atomic_pair
    else:
        low = torch.minimum(elements[source], elements[destination])
        high = torch.maximum(elements[source], elements[destination])
        pair_index = low * self.n_elements + high
        pair_values = 0.5 * (self.pair_coeff(pair_index) * radial).sum(dim=-1) * envelope
        if hasattr(self, "owned_energy_weights"):
            pair_values = pair_values * self.owned_energy_weights[source]
    if hasattr(self, "owned_energy_weights"):
        atomic_energy = atomic_energy * self.owned_energy_weights
        if prefix_atomic_energy is not None:
            prefix_atomic_energy = prefix_atomic_energy * self.owned_energy_weights
    def reduce_energy(atom_values):
        if stable_sum and n_graphs == 1:
            # Parallel tree reduction avoids contention on one global
            # accumulator for the large single-system MD case.
            energy = atom_values.sum().reshape(1)
            if csr is None:
                energy = energy + pair_values.double().sum().reshape(1)
            return energy
        energy = atom_values.new_zeros((n_graphs,))
        energy.index_add_(0, graph_ids, atom_values)
        if csr is None:
            energy.index_add_(0, graph_ids[source], pair_values.to(energy.dtype))
        return energy

    energy = reduce_energy(atomic_energy)
    return energy if prefix_atomic_energy is None else (energy, reduce_energy(prefix_atomic_energy))
