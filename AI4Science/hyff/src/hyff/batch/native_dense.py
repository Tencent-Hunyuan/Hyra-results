"""Native fixed-capacity batching for presto, vivace and moderato.

presto delegates to native_presto. vivace and moderato keep the native
single-system operators and replace only the first-message species lookup
with the gathered graph-conditioned per-atom cache.
"""
from __future__ import annotations
import math
from typing import Dict
from types import MethodType
import torch
from hyff.batch.base import BatchedPotential, _reference_products
from hyff.batch.native_presto import NativeBatchedPotential as PrestoNativeBatchedPotential
from hyff.models.l2_potential import _quadrupole, smooth_switch
from hyff.engines.native import moderato as full
from hyff.engines.native import frozen_nodes as large
from hyff.models.base_model import forward_fp64

def full_batched_forward(self, batch: Dict[str, torch.Tensor], csr=None, stable_sum=False,
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
            batch['_engine_first_message_atoms'] if block_index == 0 else None,
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
        from hyff.kernels.pair_energy_species import pair_energy
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


def _initial_scalar(self, batch):
    return batch['_engine_initial_scalar']


def _full_forward(self, batch, **kwargs):
    for index in self.config.expert_blocks:
        self.blocks[index].update_mlp.partition = batch['_engine_partition']
    return full_batched_forward(self, batch, **kwargs)


def prepare_full(model, tier, precision='tf32x3', statistics='fp64', compiled_nodes=True):
    """Apply the native single-system setup to a freshly loaded model; no config or export file is read."""
    if tier not in {'vivace', 'moderato'} or precision != 'tf32x3' or statistics != 'fp64':
        raise ValueError('vivace/moderato native batching here requires tf32x3 products and fp64 statistics')
    expected = {'vivace': (64, 32, 8, 3), 'moderato': (128, 64, 16, 4)}[tier]
    actual = (model.config.hidden, model.tensor_hidden, model.config.edge_hidden, len(model.blocks))
    if actual != expected:
        raise ValueError('Raw model widths differ from the declared native tier')
    model.eval().requires_grad_(False)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('Native operators require FP32 learned tensors')
    if model.radial_tables is not None or model.fused_messages or model.reference_chunk_size:
        raise ValueError('A fresh raw model is required')
    model._engine_global_context = None
    model._engine_expert_block = 64
    model.reference_chunk_size = 4096
    model.compile_radial_tables(200, order=5)
    model.fused_messages = model.fused = True
    model.fused_message_settings = (16, 8, 4, True)
    model.filter_skin_neighbors = True
    full.base.configure_neighbor_reserve(model, 1.25)
    if model.config.edge_hidden == 8:
        from hyff.kernels.compact_pair_messages import _compact_message_forward
        for block in model.blocks:
            block.forward = MethodType(_compact_message_forward, block)
    full.enable_joined_nodes(model, fold_first=True)
    delattr(model, 'frozen_first_messages')
    large._convert_selected(model, precision)
    full.enable_fast_nodes(model, fuse_activations=True, strided_inputs=True, precision=precision)
    for module in model.modules():
        if isinstance(module, full.JoinedLinear):
            module.fast, module.precision = True, precision
    full.enable_frozen_layernorm(model, atom_tile=8)
    full.enable_frozen_local_features(model, atom_tile=4)
    full.enable_grouped_experts(model, precision)
    model.initial_scalar = MethodType(_initial_scalar, model)
    model.forward = MethodType(_full_forward, model)
    if compiled_nodes:
        model.enable_compiled_nodes()
    return model


class NativeBatchedPotential(PrestoNativeBatchedPotential):
    """Same flat-array API with each tier's native precision policy.

    presto defaults to TF32x3/FP32-LN; vivace and moderato here use TF32x3/FP64-LN
    (their FP32-LN default is set up in hyff.batch.native).
    Each object owns one raw model and one fixed captured storage layout.
    """
    def __init__(self, model, atom_capacity, edge_capacity, graph_capacity, *, device='cuda',
                 backend='cuda_graph', mode='energy_forces', tier='presto', precision='tf32x3',
                 statistics=None, neighbor_capacity=128, compiled_nodes=True):
        statistics = statistics if statistics is not None else ('fp32' if tier == 'presto' else 'fp64')
        if tier == 'presto':
            super().__init__(model, atom_capacity, edge_capacity, graph_capacity, device=device,
                backend=backend, mode=mode, tier=tier, precision=precision, statistics=statistics,
                neighbor_capacity=neighbor_capacity, compiled_nodes=compiled_nodes)
            return
        if tier not in {'vivace', 'moderato'}:
            raise NotImplementedError('Native batching here supports presto, vivace and moderato; no reference fallback')
        if torch.device(device).type != 'cuda':
            raise ValueError('The native prepared kernels require CUDA')
        if type(neighbor_capacity) is not int or neighbor_capacity < 8 or neighbor_capacity % 8:
            raise ValueError('neighbor_capacity must be a positive multiple of eight, at least eight')
        BatchedPotential.__init__(self, model, atom_capacity, edge_capacity, graph_capacity,
            device=device, backend=backend, mode=mode, max_cached_signatures=1)
        with _reference_products():
            prepare_full(self.model, tier, precision, statistics, compiled_nodes)
        self.tier, self.precision, self.statistics = tier, precision, statistics
        self.neighbor_capacity = neighbor_capacity
        self._fixed_entry = None

    def _stage_native(self, entry, values):
        if self.tier == 'presto':
            return super()._stage_native(entry, values)
        # The single-system native _prepare uses this centered FP64 setup helper,
        # even after force-time norms are replaced by frozen kernels. Use the same
        # arithmetic for each graph x species table and restore the native
        # force-time forward before any eager evaluation or capture/replay.
        norm = self.model.blocks[0].scalar_norm
        original = norm.forward
        norm.forward = MethodType(forward_fp64, norm)
        try:
            return super()._stage_native(entry, values)
        finally:
            norm.forward = original

    @property
    def metadata(self):
        value = super().metadata
        if self.tier != 'presto':
            value.update(execution_route=self.tier + '_native_batched',
                native_math='CSR messages/joined projections/TF32x3 grouped experts/FP64 norms; graph-conditioned per-atom first messages',
                setup_math='per-graph FP32 Q/N,(M-1)/N, FP32 state/first-message products, explicit centered FP64 setup norm')
        return value
