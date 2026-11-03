"""CSR/fused-message execution of the two-block isotropic-outer backbone.

Block 0 is the `L2Interaction` scalar-source transport (8 scalar + 8 vector + 8 tensor message
channels, `_finish_fresh` closure), block 1 the `IsotropicOuterInteraction` (8 scalar channels
only). Both class forwards refuse `csr`/fused execution, so this module binds an instance-level
forward that runs

  * block 0 through the unchanged `L2Interaction.forward` CSR branch, i.e. the Triton fused
    scalar-source message kernel (`hyff.kernels.fused_messages.messages`, `first=True`) with the
    block's quintic radial table (3 x 8 channels) on the 3.5 A radius view;
  * block 1 through the same kernel on the 4.5 A view with the node message and the radial table
    zero-padded to the kernel's 3 x C layout (vector/tensor outputs are exactly zero and discarded;
    the scalar aggregate is `sum_j m0 * w0 * envelope`, the block's own message);
  * the symmetric pair term through `hyff.kernels.pair_energy_species.pair_energy`
    (NE = n_elements) on the 4.5 A view;
  * atom-local nodes (FP64-internal LayerNorm policy, element experts, readouts) as ordinary FP32
    torch operations, optionally `torch.compile`d (`compiled_nodes`).

Neighbor preparation is `hyff.engines.native.base_adapter.PreparedGeneralNeighbors` (CSR matrices
from the checked skin graph, current-coordinate radius views), which is also what the device MD
driver (`hyff.md.device.matrix_phases.NativeMatrixPhases`) rebuilds on device. The presto engine
reuses this module's edge-geometry helpers.

With `csr=None` the bound forward is the unmodified training forward (analytic radial filters on
the COO skin graph); `ForceEngine` compares every capture against it. No tensor is cast, rounded
or edited; the weights are verified bytewise after preparation.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn

from hyff.engines.native.base_adapter import PreparedGeneralNeighbors
from hyff.models.isotropic_outer import IsotropicOuterInteraction
from hyff.engines import reference
from hyff.kernels.pair_energy_species import pair_energy
from hyff.models.scalar_refresh import _finish_fresh
from hyff.models.element_model import prepare_batch
from hyff.engines.force_engine import ForceEngine
from hyff.models.experts import ElementPartition
from hyff.kernels.fused_messages import configure_neighbor_reserve, messages
from hyff.models.l2_potential import L2Interaction, _RowSelectedLinear, _quadrupole, smooth_switch
from hyff.kernels.radial_table import QuinticRadialTable

IMPLEMENTATION = 'hyff.engines.native.fast_csr'
BACKEND = 'isotropic_outer_csr'
DEFAULT_SETTINGS = (16, 8, 4, True)
EXPECTED_BACKBONE = dict(hidden=32, edge_hidden=8, tensor_hidden=16, interactions=2, cutoff=4.5,
                         layer_cutoffs=(3.5, 4.5), scalar_source_blocks=(1,), isotropic_outer=True,
                         refresh_angular=True, radial_conditioning='shared', expert_routing='element')

digest = reference.digest
read_export = reference.read_export
state_features = reference.state_features
system_batch = reference.system_batch


def check_export(saved):
    if reference.family(saved) != 'fast':
        raise ValueError('This adapter executes isotropic-outer models only')
    backbone = saved['model_config']['backbone_config']
    actual = {key: (tuple(backbone[key]) if isinstance(backbone[key], list) else backbone[key]) for key in EXPECTED_BACKBONE}
    if actual != EXPECTED_BACKBONE:
        raise ValueError(f'Backbone differs from the validated isotropic-outer execution: {actual} != {EXPECTED_BACKBONE}')
    return backbone


def check_blocks(model):
    """The two blocks must have exactly the structures that the kernels assume."""
    first, outer = model.blocks
    if (len(model.blocks) != 2 or type(first).forward is not L2Interaction.forward or not isinstance(first, L2Interaction)
            or 'forward' in first.__dict__):
        raise TypeError('Block 0 must execute the unchanged L2Interaction forward (CSR branch)')
    if getattr(first._finish_messages, '__func__', None) is not _finish_fresh:
        raise TypeError('Block 0 must carry the scalar-refresh closure (_finish_fresh)')
    if not (first.input_is_scalar and first.terminal and first.bottleneck and first.tensor_bottleneck):
        raise ValueError('Block 0 is not the scalar-source bottleneck transport')
    if not isinstance(outer, IsotropicOuterInteraction) or not outer.isotropic_outer:
        raise TypeError('Block 1 must be the IsotropicOuterInteraction')
    if first.message_hidden != 8 or outer.message_hidden != 8:
        raise ValueError('Message width 8 expected')
    if float(first.cutoff) != float(model.layer_cutoffs[0]) or float(outer.cutoff) != float(model.layer_cutoffs[1]):
        raise ValueError('Block cutoffs must equal the layer cutoffs')
    if float(outer.cutoff) != float(model.cutoff):
        raise ValueError('The outer block must use the model cutoff (skin view filter)')
    if first.switch_width != model.config.switch_width or outer.switch_width != model.config.switch_width:
        raise ValueError('Switch widths differ')
    if first.message_scale != .25 or outer.message_scale != .25:
        raise ValueError('The fused kernel folds message_scale = 0.25')
    for block in (first, outer):
        network = block.radial_filter
        if (not isinstance(network, nn.Sequential) or len(network) != 3 or not _is_affine(network[0])
                or not isinstance(network[1], nn.SiLU) or not _is_affine(network[2])):
            raise TypeError('Radial filters must be affine/SiLU/affine (shared conditioning)')
    if _out_features(first.radial_filter[2]) != 3 * first.message_hidden or _out_features(outer.radial_filter[2]) != outer.message_hidden:
        raise ValueError('Radial filter widths differ from the 3x8 / 8 layout')
    if _out_features(first.scalar_message[-1]) != 3 * first.message_hidden or _out_features(outer.scalar_message[-1]) != outer.message_hidden:
        raise ValueError('Node message widths differ from the 3x8 / 8 layout')
    return first, outer


def _is_affine(layer):
    """nn.Linear or the project's _RowSelectedLinear (F.linear on weight/bias); what filter_jet accepts."""
    return isinstance(layer, (nn.Linear, _RowSelectedLinear)) and hasattr(layer, 'weight')


def _out_features(layer):
    return int(layer.weight.shape[0])


def _padded_linear(layer, out_features):
    """A derived (not learned) copy of `layer` with zero rows appended; the original tensors are untouched."""
    weight = layer.weight.detach()
    extra = out_features - int(weight.shape[0])
    if extra < 0:
        raise ValueError('Cannot shrink a filter')
    padded = nn.Linear(int(weight.shape[1]), out_features, bias=layer.bias is not None, device='meta')
    padded.weight = nn.Parameter(torch.cat((weight, weight.new_zeros((extra, weight.shape[1])))), requires_grad=False)
    if layer.bias is not None:
        bias = layer.bias.detach()
        padded.bias = nn.Parameter(torch.cat((bias, bias.new_zeros(extra))), requires_grad=False)
    return padded


def compile_tables(model, intervals=200):
    """Quintic tables: block 0 (3x8 channels, 3.5 A), block 1 zero-padded to 3x8 (4.5 A), pair (NE^2 rows, 4.5 A)."""
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze the model before compiling radial tables')
    first, outer = model.blocks
    parameter = next(model.parameters())
    table0 = QuinticRadialTable.compile(model.rbf, first.radial_filter, intervals, table_cutoff=first.cutoff)
    padded = [outer.radial_filter[0], outer.radial_filter[1], _padded_linear(outer.radial_filter[2], 3 * outer.message_hidden)]
    table1 = QuinticRadialTable.compile(model.rbf, padded, intervals, table_cutoff=outer.cutoff)
    pair_linear = nn.Linear(model.config.n_rbf, model.n_elements ** 2, bias=False, device='meta')
    pair_linear.weight = nn.Parameter(model.pair_coeff.weight.detach(), requires_grad=False)
    if pair_linear.weight.shape != (model.n_elements ** 2, model.config.n_rbf):
        raise ValueError('Pair coefficient table shape differs from n_elements^2 x n_rbf')
    pair = QuinticRadialTable.compile(model.rbf, [pair_linear], intervals)
    if float(pair.cutoff) != float(model.cutoff):
        raise ValueError('The pair table domain must be the model cutoff')
    tables = nn.ModuleList([table0, table1, pair]).to(parameter)
    if tuple(tables[0].coefficients.shape) != (intervals, 6, 3 * first.message_hidden):
        raise ValueError('Block 0 table layout differs from the FIRST kernel layout')
    if tuple(tables[1].coefficients.shape) != (intervals, 6, 3 * outer.message_hidden):
        raise ValueError('Block 1 padded table layout differs from the FIRST kernel layout')
    if not bool((tables[1].coefficients[:, :, outer.message_hidden:] == 0).all()):
        raise ValueError('Padded block 1 channels must be exactly zero')
    model.radial_tables = tables
    model.radial_table_order = 5
    return tables


# ---------------------------------------------------------------------------------------------
# Batch preparation outside every CUDA graph: species schedule and state-conditioned initial scalar.

@torch.no_grad()
def _prepare(self, batch):
    if '_engine_initial_scalar' in batch:
        return batch
    if len(batch['ptr']) != 2:
        raise ValueError('The engine forward represents one physical system')
    completed = prepare_batch(reference.complete_state_batch(self, batch))
    part = completed['_hyff_atomic_number_partition']
    unknown = set(part.groups).difference(self.atomic_numbers)
    if unknown:
        raise ValueError(f'Unsupported atomic numbers: {sorted(unknown)}')
    bound = ElementPartition(tuple(part.groups.get(z, part.empty) for z in self.atomic_numbers),
                             len(completed['atomic_numbers']), part.empty, part.empty, part.empty)
    # The training initial_scalar (embedding + state_projection([Q/N, (M-1)/N])[batch]); host checks here only.
    initial = self._original_initial_scalar(completed).detach()
    completed['_engine_partition'] = bound
    completed['_engine_initial_scalar'] = initial
    return completed


def _edge_geometry(self, batch):
    positions = batch['positions']
    source, destination = batch['edge_index'][0], batch['edge_index'][1]
    displacement = positions[source] - positions[destination]
    if 'shifts' in batch:
        displacement = displacement + batch['shifts']
    valid = source != destination
    if 'edge_valid' in batch:
        valid = batch['edge_valid']
    elif 'shifts' in batch:
        valid = valid | (batch['shifts'] != 0).any(-1)
    displacement = torch.where(valid[:, None], displacement, self.padding_displacement)
    distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-9)
    direction = displacement / distance[:, None]
    return source, destination, distance, direction, _quadrupole(direction), valid


def fast_forward(self, batch, csr=None, stable_sum=False, prefix_depth=None, return_prefix_depth=None):
    """Energy of one system; `csr` selects the fused path, otherwise the unmodified training arithmetic."""
    if prefix_depth is not None or return_prefix_depth is not None:
        raise ValueError('Energy prefixes are not supported by this engine')
    batch = _prepare(self, batch)
    partition = batch['_engine_partition']
    for block in self.blocks:
        block.update_mlp.partition = partition
    positions = batch['positions']
    numbers = batch['atomic_numbers']
    scalar = batch['_engine_initial_scalar']
    n = scalar.shape[0]
    first, outer = self.blocks
    vector = scalar.new_zeros((n, 3, self.tensor_hidden))
    tensor = scalar.new_zeros((n, 5, self.tensor_hidden))
    readout_scale = 1.0 / math.sqrt(len(self.blocks))
    count = (batch['ptr'][1:] - batch['ptr'][:-1])
    if csr is not None:
        csr.refresh_subgraphs(positions, len(self.blocks))
        scalar, vector, tensor = first(scalar, vector, tensor, None, None, None, None, None, None,
                                       csr=(positions, csr.radius_view(first.cutoff), self.radial_tables[0]))
        atomic_energy = readout_scale * self.readouts[0](self.readout_norms[0](scalar)).squeeze(-1)
        node_message = outer.scalar_message(outer.scalar_norm(scalar))
        width = outer.message_hidden
        padded = torch.cat((node_message, node_message.new_zeros((n, 2 * width))), dim=-1)
        view = csr.radius_view(outer.cutoff)
        aggregate_s, _, _ = messages(positions, view.cell, view.neighbors, view.shifts, view.counts, padded,
                                     scalar.new_zeros((n, 3, width)), scalar.new_zeros((n, 5, width)),
                                     self.radial_tables[1], outer.cutoff, outer.switch_width, True,
                                     self.fused_message_settings)
        scalar, vector, tensor = outer._finish_messages(scalar, vector, tensor, aggregate_s)
        atomic_energy = atomic_energy + readout_scale * self.readouts[1](self.readout_norms[1](scalar)).squeeze(-1)
        atomic_energy = atomic_energy + self.element_offset(numbers).squeeze(-1)
        if stable_sum:
            atomic_energy = atomic_energy.double()
        elements = self.element_lut[numbers]
        atomic_energy = atomic_energy + pair_energy(positions, csr.radius_view(self.cutoff), elements, self.radial_tables[-1])
        energy = atomic_energy.sum().reshape(1)
    else:
        source, destination, distance, direction, quadrupole, valid = _edge_geometry(self, batch)
        radial = self.rbf(distance)
        width = self.config.switch_width
        envelope = smooth_switch(distance, self.cutoff, width) * valid.to(distance.dtype)
        inner_envelope = envelope if float(first.cutoff) == float(self.cutoff) else smooth_switch(distance, first.cutoff, width) * valid
        scalar, vector, tensor = first(scalar, vector, tensor, radial, direction, quadrupole, source, destination, inner_envelope)
        atomic_energy = readout_scale * self.readouts[0](self.readout_norms[0](scalar)).squeeze(-1)
        scalar, vector, tensor = outer(scalar, vector, tensor, radial, direction, quadrupole, source, destination, envelope)
        atomic_energy = atomic_energy + readout_scale * self.readouts[1](self.readout_norms[1](scalar)).squeeze(-1)
        atomic_energy = atomic_energy + self.element_offset(numbers).squeeze(-1)
        if stable_sum:
            atomic_energy = atomic_energy.double()
        elements = self.element_lut[numbers]
        low = torch.minimum(elements[source], elements[destination])
        high = torch.maximum(elements[source], elements[destination])
        pair_values = 0.5 * (self.pair_coeff(low * self.n_elements + high) * radial).sum(dim=-1) * envelope
        if stable_sum:
            energy = atomic_energy.sum().reshape(1) + pair_values.double().sum().reshape(1)
        else:
            energy = atomic_energy.new_zeros((1,)).index_add_(0, batch['batch'], atomic_energy)
            energy = energy.index_add_(0, batch['batch'][source], pair_values.to(energy.dtype))
    # Affine energy scale and shift (FP64 buffers, set from the training data), as in the model's own forward.
    return energy * self.output_scale + count.to(energy) * self.output_shift


def _prepare_energy(self, batch, edges):
    prepared = _prepare(self, batch)
    return PreparedGeneralNeighbors(self, prepared, edges)


def bind_context(model, n_atoms, total_charge=0., multiplicity=1):
    context = dict(n_atoms=int(n_atoms), total_charge=float(total_charge), multiplicity=int(multiplicity),
                   state_features=state_features(n_atoms, total_charge, multiplicity))
    model._engine_global_context = context
    return context


def load(checkpoint, device='cuda', *, global_context=None, reserve=1.25, compiled_nodes=False,
         settings=DEFAULT_SETTINGS, intervals=200):
    """Checked isotropic-outer model on the CSR/fused path; FP32 learned tensors unchanged and verified bytewise."""
    path = Path(checkpoint).resolve()
    saved = read_export(path)
    backbone = check_export(saved)
    model, saved = reference.load(path, device)          # factory build, FP64-internal LayerNorm policy, strict load, bytewise check
    first, outer = check_blocks(model)
    if model.radial_tables is not None or model.fused_messages or model.reference_chunk_size:
        raise ValueError('Fresh export model expected')
    if settings[0] not in (8, 16, 32, 64) or settings[1] not in (4, 8, 16, 32) or settings[2] not in (4, 8):
        raise ValueError('Unsupported fused message tiling')
    compile_tables(model, intervals)
    model.fused_message_settings = tuple(settings)
    model.filter_skin_neighbors = True
    model.fused_messages = model.fused = True
    configure_neighbor_reserve(model, reserve)
    model._original_initial_scalar = model.initial_scalar
    model._engine_global_context = dict(global_context) if global_context is not None else None
    model.forward = MethodType(fast_forward, model)
    model.prepare_energy = MethodType(_prepare_energy, model)
    if compiled_nodes:
        model.enable_compiled_nodes()
    verified = reference.verify_state(model, saved)
    model.inference_backend = BACKEND
    model.deployment_metadata = dict(model.export_metadata,
        implementation=IMPLEMENTATION, inference_backend=BACKEND, backend=BACKEND, precision='fp32',
        numerics=dict(label='fp32', tensor_core_products=False,
                      node_linears='torch FP32 (cuBLAS; TF32 disabled by the driver)',
                      element_experts='per-species FP32 loop (ElementExpertMLP)',
                      layernorm='centered FP64 internal, FP32 affine (export policy)',
                      messages=f'Triton CSR fused scalar-source messages, FP32, quintic radial tables ({intervals} intervals); '
                               'block 1 zero-padded to the 3x8 layout',
                      pair=f'Triton symmetric pair, FP32, quintic table ({intervals} intervals, NE = n_elements)'),
        compiled_nodes=bool(compiled_nodes), fused_message_settings=list(settings), neighbor_reserve=float(reserve),
        filter_skin_neighbors=True, radial_table_cutoffs_A=[float(t.cutoff) for t in model.radial_tables],
        radial_table_channels=[int(t.coefficients.shape[2]) for t in model.radial_tables],
        block_layout=dict(block0='L2Interaction scalar-source CSR branch (8 s + 8 v + 8 t), _finish_fresh closure, 3.5 A view',
                          block1='IsotropicOuterInteraction via padded FIRST kernel (8 s), _finish_scalar, 4.5 A view',
                          pair='symmetric pair energy, 4.5 A view'),
        layer_cutoffs_A=[float(r) for r in model.layer_cutoffs], cutoff_A=float(model.cutoff),
        dependency_radius_A=max(float(model.cutoff), float(sum(model.layer_cutoffs))),
        backbone=backbone, n_elements=int(model.n_elements), condition_spin=bool(model.condition_spin),
        state_features=list(reference.STATE_FEATURES), global_context=model._engine_global_context,
        export_tensors_verified_after_preparation=verified, learned_weights_changed=False,
        export=dict(architecture=saved['architecture'], model_factory=saved['model_factory'],
                    selection_kind=saved.get('selection_kind'), step=saved.get('step'),
                    frame_epochs=saved.get('frame_epochs'), energy_output=saved.get('energy_output'),
                    numerical_execution=saved.get('numerical_execution')))
    return model, saved


def create_engine(positions, atomic_numbers, cell, checkpoint, device='cuda', total_charge=0., multiplicity=1,
                  skin=1.5, reserve=1.25, compiled_nodes=False, settings=DEFAULT_SETTINGS):
    """Checked CUDA-graph ForceEngine on the CSR path; same call shape as the other native engines."""
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    n = len(numbers)
    context = dict(n_atoms=n, total_charge=float(total_charge), multiplicity=int(multiplicity),
                   state_features=state_features(n, total_charge, multiplicity))
    model, _ = load(checkpoint, device, global_context=context, reserve=reserve, compiled_nodes=compiled_nodes, settings=settings)
    batch = system_batch(positions, numbers, cell, total_charge, multiplicity, device)
    engine = ForceEngine(model, batch, 'cuda_graph' if torch.device(device).type == 'cuda' else 'eager', skin=skin)
    try:
        energy, force = engine(batch)
        if not bool(torch.isfinite(energy).all() & torch.isfinite(force).all()):
            raise FloatingPointError('Nonfinite prepared energy or force')
    except BaseException:
        engine.close()
        raise
    metadata = dict(model.deployment_metadata, global_context=context, skin_A=float(skin),
                    neighbor_construction='hyff.graph.NeighborCache/radius_edges (cutoff + skin) -> PreparedGeneralNeighbors CSR')
    return dict(model=model, batch=batch, engine=engine, metadata=metadata)


__all__ = ['load', 'create_engine', 'bind_context', 'fast_forward', 'compile_tables', 'check_export', 'check_blocks',
           'read_export', 'state_features', 'system_batch', 'digest', 'BACKEND', 'IMPLEMENTATION', 'DEFAULT_SETTINGS']
