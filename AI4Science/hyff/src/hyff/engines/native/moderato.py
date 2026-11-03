"""Native moderato engine: frozen-node execution of the l4a0 L2 model with element experts.

For models built by `hyff.models.base_model:build_model` (StatePotential, l4a0 backbone,
hidden 128 / tensor 64 / edge 16, four blocks at 3.5/3.5/6/8 A, element experts, any species
list, charge and spin conditioning). The execution path: joined/folded projections, a
batch-owned state-conditioned first-message cache, frozen LayerNorm (FP64 or FP32 statistics),
fused local features, grouped element experts, compiled node graphs, CSR/fused Triton messages
with quintic radial tables and TF32x3 tensor-core products on the unchanged FP32 weights.

The global state enters as the training features [Q/N, (M-1)/N]
(`hyff.models.state_model.StatePotential.initial_scalar`), computed from the bound
whole-system context (N, Q, M) outside every CUDA graph. The pair term uses
`hyff.kernels.pair_energy_species` (NE = n_elements); the block forward is
`hyff.engines.native.moderato_forward`. The base loader and CSR neighbour preparation come from
`hyff.engines.native.base_adapter`, frozen-node helpers from `hyff.engines.native.frozen_nodes`
and weight reading, state features and batches from `hyff.engines.reference`.

Precision variants
  tf32x3   tensor-core TF32x3 node products, grouped experts (default)
  fp16x3   same path with the residual FP16 three-product decomposition
  fp32     no tensor-core products: node linears stay ordinary FP32 cuBLAS (TF32 disabled),
           element experts stay the per-species loop; joined/folded projections,
           frozen LayerNorm, fused local features, compiled nodes, CSR fused messages and radial
           tables are kept. No learned tensor is changed in any variant.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from hyff.engines.native import base_adapter as base
from hyff.engines.native import frozen_nodes
from hyff.engines.native.centered_fp32 import check_fp32_statistics, normalization_installers
from hyff.kernels.layernorm_fp64 import FrozenLayerNorm
from hyff.kernels.grouped_experts import (
    FrozenElementExpertMLP, bind_partition, enable_grouped_experts,
)
from hyff.engines import reference
from hyff.models.element_model import ElementExpertMLP, prepare_batch
from hyff.models.base_model import POLICY as LAYERNORM_POLICY
from hyff.models.base_model import forward_fp64
from hyff.data import collate, physical
from hyff.engines.force_engine import ForceEngine
from hyff.kernels.fused_linear import FastLinear, enable_fast_nodes
from hyff.kernels.joined_linear import JoinedLinear, enable_joined_nodes

PRECISIONS = ('tf32x3', 'fp16x3', 'fp32')
IMPLEMENTATION = 'hyff.engines.native.moderato'
EXPECTED_BACKBONE = dict(backbone='l4a0', hidden=128, tensor_hidden=64, edge_hidden=16, interactions=4,
                         layer_cutoffs=(3.5, 3.5, 6.0, 8.0))

digest = reference.digest
read_export = reference.read_export
state_features = reference.state_features
system_batch = reference.system_batch


def execution_context():
    return nullcontext()


def dependency_radius(model):
    return max(float(model.cutoff), float(sum(model.layer_cutoffs)))


def check_export(saved):
    """The moderato path applies to the l4a0 backbone only; species and state width are free."""
    if reference.family(saved) != 'large':
        raise ValueError('This engine runs L2Interaction models only')
    config = saved['model_config']
    actual = {key: (tuple(config[key]) if key == 'layer_cutoffs' else config[key]) for key in EXPECTED_BACKBONE}
    if actual != EXPECTED_BACKBONE:
        raise ValueError(f'Backbone differs from the moderato execution: {actual} != {EXPECTED_BACKBONE}')
    policy = saved.get('numerical_execution') or {}
    if policy.get('policy') != LAYERNORM_POLICY:
        raise ValueError(f'Export records an unknown LayerNorm policy: {policy!r}')
    numbers = tuple(config['atomic_numbers'])
    if len(numbers) != len(set(numbers)) or not numbers:
        raise ValueError('Distinct nonempty species list required')
    return dict(atomic_numbers=numbers, n_elements=len(numbers), condition_spin=bool(config.get('condition_spin', False)))


# ---------------------------------------------------------------------------------------------
# Whole-system state and batch-owned first-message cache.

def _global_state(self, batch):
    """Intensive whole-system features in training order; a halo's local count never replaces N.

    Returns (1, F) with F = state_projection.in_features: [Q/N] for charge-only checkpoints,
    [Q/N, (M-1)/N] for `condition_spin`. Arithmetic (float32 division of float32 tensors) is
    the arithmetic of `StatePotential.initial_scalar`.
    """
    numbers = batch['atomic_numbers']
    if batch['ptr'].numel() != 2:
        raise ValueError('An engine represents one physical system')
    local_count = (batch['ptr'][1:] - batch['ptr'][:-1]).to(self.embedding.weight)
    if len(numbers) == 0 or float(local_count[0]) != len(numbers):
        raise ValueError('The single-system ptr must cover all local atoms')
    if batch['batch'].shape != numbers.shape or bool((batch['batch'] != 0).any()):
        raise ValueError('Every local atom must belong to the single physical system')
    charge = batch.get('total_charge', torch.zeros_like(local_count)).to(local_count)
    if charge.shape != local_count.shape:
        raise ValueError('Supply one total charge for the physical system')
    count = local_count
    multiplicity = batch.get('multiplicity', torch.ones_like(local_count)).to(local_count)
    if multiplicity.shape != count.shape:
        raise ValueError('Supply one multiplicity M=2S+1 for the physical system')
    context = self._engine_global_context
    if context is not None:
        n_atoms = context['n_atoms']
        if (not math.isfinite(float(n_atoms)) or int(n_atoms) != n_atoms or n_atoms < len(numbers)):
            raise ValueError('The full system atom count must be integral and cover every halo atom')
        count = torch.full_like(count, n_atoms)
        charge = torch.full_like(charge, context['total_charge'])
        multiplicity = torch.full_like(count, context.get('multiplicity', 1))
    if not bool(torch.isfinite(charge).all() & (count > 0).all()):
        raise ValueError('Finite total charge and nonempty global system required')
    if not bool((torch.isfinite(multiplicity) & (multiplicity >= 1) & (multiplicity == multiplicity.round())).all()):
        raise ValueError('Multiplicity M=2S+1 must be a positive integer')
    features = [charge / count]
    if self.condition_spin:
        features.append((multiplicity - 1.) / count)
    elif not bool((multiplicity == 1).all()):
        raise ValueError('A charge-only checkpoint has no trained nonsinglet response')
    return torch.stack(features, dim=-1).reshape(1, -1)


@torch.no_grad()
def _prepare(self, batch):
    """Prepare species schedule, state-conditioned initial scalar and first messages outside every graph."""
    batch = prepare_batch(batch)
    partition = bind_partition(batch['atomic_numbers'], self.atomic_numbers)
    state_features = _global_state(self, batch)
    if state_features.shape[1] != self.state_projection.weight.shape[1]:
        raise ValueError('State feature count differs from the checkpoint state projection')
    state = F.linear(state_features, self.state_projection.weight, self.state_projection.bias)
    # embedding.weight + state, then a row lookup: elementwise identical to the training
    # `embedding(Z) + state[batch]` (both FP32 additions of the same two numbers).
    initial_table = self.embedding.weight + state
    first = self.blocks[0]
    # Setup-only products in ordinary FP32 with the centered FP64 normalization of the export
    # policy. Force-time TF32x3/FP16x3 never rounds a stored checkpoint tensor.
    normalized = forward_fp64(first.scalar_norm, initial_table)
    source = first.scalar_message
    messages = F.linear(F.silu(F.linear(normalized, source[0].weight, source[0].bias)),
                        source[2].weight, source[2].bias)
    batch['_engine_partition'] = partition
    batch['_engine_initial_scalar'] = initial_table[batch['atomic_numbers']].detach()
    batch['_engine_first_messages'] = messages.detach()
    batch['_engine_state_feature'] = state_features.detach()
    for index in self.config.expert_blocks:
        self.blocks[index].update_mlp.partition = partition
    return batch


def _initial_scalar(self, batch):
    return batch['_engine_initial_scalar']


def _forward(self, batch, **kwargs):
    from hyff.engines.native.moderato_forward import reference_forward
    if '_engine_initial_scalar' not in batch:
        batch = _prepare(self, batch)
    # Bind the schedule belonging to this batch even after another batch was prepared on the
    # same model. CUDA graphs hold their own captured arrays.
    for index in self.config.expert_blocks:
        self.blocks[index].update_mlp.partition = batch['_engine_partition']
    return reference_forward(self, batch, **kwargs)


def _prepare_energy(self, batch, edges):
    prepared = _prepare(self, batch)
    return base.PreparedGeneralNeighbors(self, prepared, edges)


def bind(model, global_context=None):
    """Install the batch-owned state path on a model (surrogates and the native engine)."""
    model._engine_global_context = dict(global_context) if global_context is not None else None
    model.forward = MethodType(_forward, model)
    model.initial_scalar = MethodType(_initial_scalar, model)
    model.prepare_energy = MethodType(_prepare_energy, model)
    return model


# ---------------------------------------------------------------------------------------------
# Native engine.

def load(checkpoint, device='cuda', *, global_context=None, owned_weights=None, reserve=1.25,
         precision='tf32x3', compiled_nodes=True, layernorm_atom_tile=8, statistics='fp64'):
    if precision not in PRECISIONS:
        raise ValueError(f'precision must be one of {PRECISIONS}')
    enable_norm, enable_features = normalization_installers(statistics, precision)
    path = Path(checkpoint).resolve()
    saved = read_export(path)
    species = check_export(saved)
    if global_context is not None:
        global_context = dict(global_context)
    # Base loader: factory build, strict load without cast, quintic radial tables (200 intervals per
    # layer cutoff plus the n_elements^2 pair table), CSR settings, neighbor reserve, tensor check.
    model, saved = base.load(path, device, global_context=global_context, owned_weights=owned_weights,
                             reserve=reserve, node_mode='torch')
    if tuple(model.atomic_numbers) != species['atomic_numbers'] or bool(model.condition_spin) != species['condition_spin']:
        raise ValueError('Built model differs from the export config')
    if model.state_projection.weight.shape != (model.config.hidden, 2 if model.condition_spin else 1):
        raise ValueError('Unexpected state projection shape')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('The full adapter requires FP32 learned weights')
    if model.radial_tables[-1].coefficients.shape[2] != model.n_elements ** 2:
        raise ValueError('Pair radial table rows do not match n_elements^2')
    original_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    original_parameters = tuple(model.named_parameters())

    # Join while modules are ordinary torch operations so CPU setup is valid. The helper's
    # embedding-only lookup is discarded; _prepare creates the state-conditioned replacement.
    enable_joined_nodes(model, fold_first=True)
    delattr(model, 'frozen_first_messages')
    if precision != 'fp32':
        frozen_nodes._convert_selected(model, precision)
        enable_fast_nodes(model, fuse_activations=True, strided_inputs=True, precision=precision)
        for module in model.modules():
            if isinstance(module, JoinedLinear):
                module.fast, module.precision = True, precision
    enable_norm(model, atom_tile=layernorm_atom_tile)
    enable_features(model, atom_tile=4)
    if precision != 'fp32':
        enable_grouped_experts(model, precision)
    bind(model, global_context)
    if compiled_nodes:
        model.enable_compiled_nodes()

    unchanged_count = frozen_nodes._verify_tensors(model, original_state, 'Engine preparation')
    actual = model.state_dict()
    for name, value in saved['model_state'].items():
        if (name not in actual or actual[name].dtype != value.dtype
                or not torch.equal(actual[name].detach().cpu(), value)):
            raise ValueError(f'Checkpoint tensor changed: {name}')
    parameters = tuple(model.named_parameters())
    if (len(parameters) != len(original_parameters)
            or any(new_name != old_name or new_value is not old_value
                   for (new_name, new_value), (old_name, old_value) in zip(parameters, original_parameters))):
        raise ValueError('Engine preparation changed the learned parameter namespace or objects')
    grouped = precision != 'fp32'
    expert_modules = [model.blocks[i].update_mlp for i in model.config.expert_blocks]
    if grouped and not all(isinstance(m, FrozenElementExpertMLP) and m.count == model.n_elements for m in expert_modules):
        raise ValueError('Grouped experts were not installed for every expert block')
    if not grouped and not all(isinstance(m, ElementExpertMLP) and len(m.experts) == model.n_elements for m in expert_modules):
        raise ValueError('The FP32 variant requires plain element experts in every expert block')
    model.inference_backend = 'moderato_native'
    model.deployment_metadata = dict(model.deployment_metadata,
        implementation=IMPLEMENTATION, inference_backend=model.inference_backend, node_mode='full', precision=precision,
        numerics=numerics_label(precision), compiled_nodes=bool(compiled_nodes),
        joined_projections=True, folded_first_projections=True,
        fused_local_features=True, fused_local_features_atom_tile=4,
        grouped_experts=grouped, active_experts_per_atom=1,
        expert_execution='grouped_triton_' + precision if grouped else 'per_species_loop_fp32',
        node_products='triton_' + precision if grouped else 'torch_fp32_cublas',
        frozen_layernorm=True, frozen_layernorm_atom_tile=layernorm_atom_tile,
        frozen_layernorm_modules=sum(isinstance(m, FrozenLayerNorm) for m in model.modules()),
        prepared_state_tensors_verified=unchanged_count,
        export_tensors_verified=len(saved['model_state']),
        original_parameter_objects_and_namespace_verified=True,
        first_message_cache='batch_owned_species_lookup_conditioned_on_global_[Q/N,(M-1)/N]',
        first_message_cache_arithmetic='setup_FP32_linear_centered_FP64_LayerNorm',
        normalization_policy='centered_FP64_internal_same_epsilon',
        fused_feature_normalization_precision='FP64',
        one_physical_system_per_engine=True, topology_input=False,
        atomic_numbers=list(model.atomic_numbers), n_elements=model.n_elements,
        condition_spin=bool(model.condition_spin),
        state_features=list(reference.STATE_FEATURES) if model.condition_spin else ['Q/N'],
        pair_kernel='hyff.kernels.pair_energy_species.pair_energy (NE = n_elements)',
        pair_table_rows=int(model.radial_tables[-1].coefficients.shape[2]),
        radial_table_cutoffs_A=[float(t.cutoff) for t in model.radial_tables],
        dependency_radius_A=dependency_radius(model),
        export=dict(architecture=saved['architecture'], model_factory=saved['model_factory'],
                    selection_kind=saved.get('selection_kind'), step=saved.get('step'),
                    frame_epochs=saved.get('frame_epochs'), energy_output=saved.get('energy_output'),
                    numerical_execution=saved.get('numerical_execution')))
    if statistics == 'fp32':
        check_fp32_statistics(model)
    return model, saved


def numerics_label(precision):
    if precision == 'fp32':
        return dict(label='fp32', tensor_core_products=False,
                    node_linears='torch FP32 (cuBLAS; TF32 disabled by the driver)',
                    element_experts='per-species FP32 loop',
                    layernorm='centered FP64 internal, FP32 affine', local_features='fused, FP64 statistics',
                    messages='Triton CSR fused, FP32, quintic radial tables (200 intervals)',
                    pair='Triton symmetric pair, FP32, quintic table (200 intervals, NE = n_elements)')
    return dict(label=precision, tensor_core_products=True,
                node_linears=f'Triton {precision} on unchanged FP32 weights, FP32 accumulation',
                element_experts=f'grouped Triton {precision}, one expert per atom',
                layernorm='centered FP64 internal, FP32 affine', local_features='fused, FP64 statistics',
                messages='Triton CSR fused, FP32, quintic radial tables (200 intervals)',
                pair='Triton symmetric pair, FP32, quintic table (200 intervals, NE = n_elements)')


def create(record, owned_weights=None, *, checkpoint, global_context=None, device='cuda', skin=1.5,
           reserve=1.25, precision='tf32x3', compiled_nodes=True, layernorm_atom_tile=8, capture_branch_policy='first',
           statistics='fp64'):
    record = {**record, 'total_charge': record.get('total_charge', 0.)}
    context = dict(global_context) if global_context is not None else dict(
        n_atoms=len(record['positions']), total_charge=record['total_charge'],
        multiplicity=record.get('multiplicity', 1))
    if owned_weights is not None and len(owned_weights) != len(record['positions']):
        raise ValueError('One final atomic energy weight is required for every local atom')
    model, _ = load(checkpoint, device, global_context=context, owned_weights=owned_weights, reserve=reserve,
                    precision=precision, compiled_nodes=compiled_nodes, layernorm_atom_tile=layernorm_atom_tile,
                    statistics=statistics)
    batch = physical(collate([record], device))
    backend = 'cuda_graph' if torch.device(device).type == 'cuda' else 'eager'
    engine = ForceEngine(model, batch, backend, skin=skin, capture_branch_policy=capture_branch_policy)
    try:
        energy, force = engine(batch)
        if not bool(torch.isfinite(energy).all() & torch.isfinite(force).all()):
            raise ValueError('Nonfinite fully prepared energy or force')
    except BaseException:
        engine.close()
        raise
    return dict(model=model, batch=batch, engine=engine, metadata=model.deployment_metadata)


def factory(record, owned_weights=None, *, checkpoint, global_context=None, device='cuda', skin=1.5,
            reserve=1.25, precision='tf32x3', compiled_nodes=True, layernorm_atom_tile=8, capture_branch_policy='first'):
    prepared = create(record, owned_weights, checkpoint=checkpoint, global_context=global_context, device=device,
                      skin=skin, reserve=reserve, precision=precision, compiled_nodes=compiled_nodes,
                      layernorm_atom_tile=layernorm_atom_tile, capture_branch_policy=capture_branch_policy)
    return prepared['engine'], prepared['metadata']


def create_full_engine(positions, atomic_numbers, cell, checkpoint, device='cuda', total_charge=0., multiplicity=1,
                       skin=1.5, reserve=1.25, precision='tf32x3', compiled_nodes=True, layernorm_atom_tile=8,
                       *, capture_branch_policy='first', statistics='fp64'):
    """Create the moderato engine for one system; same call shape as the other native engines."""
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    n = len(numbers)
    context = dict(n_atoms=n, total_charge=float(total_charge), multiplicity=int(multiplicity),
                   state_features=state_features(n, total_charge, multiplicity))
    record = dict(positions=np.asarray(positions, dtype=np.float32), atomic_numbers=numbers, total_charge=float(total_charge))
    if cell is not None:
        record.update(cell=np.asarray(cell, dtype=np.float32), pbc=np.ones(3, dtype=bool))
    made = create(record, checkpoint=str(Path(checkpoint).resolve()), global_context=context, device=device, skin=skin,
                  reserve=reserve, precision=precision, compiled_nodes=compiled_nodes, layernorm_atom_tile=layernorm_atom_tile,
                  capture_branch_policy=capture_branch_policy, statistics=statistics)
    made['metadata'] = dict(made['metadata'], backend='native_nodes_csr_fused_messages',
                            global_context=context, skin_A=float(skin))
    return made


# ---------------------------------------------------------------------------------------------
# CPU-executable surrogate of the native path (ordinary torch operators; no Triton).

def load_torch_surrogate(checkpoint, device='cpu', *, radial_tables=False, global_context=None):
    """The native path's torch-executable subset for CPU checks.

    Exactly the adapter logic that does not need Triton: export build (FP64-internal LayerNorm
    policy), joined + folded first projections (`enable_joined_nodes(fold_first=True)`), the
    discarded embedding-only lookup, the batch-owned state-conditioned initial scalar and first
    messages (`_prepare`), the species schedule (`bind_partition`) bound to the plain element
    experts, and `hyff.engines.native.moderato_forward` (general pair stride in its PyTorch
    branch). With `radial_tables` the quintic 200-interval tables of the four layer cutoffs
    replace the analytic radial filters through the memory-bounded chunked reference
    (`hyff.models.l2_chunked_reference`), as the engine's own eager reference does. Node kernels
    (TF32x3 products, frozen FP64 LayerNorm, fused local features, grouped experts), the CSR fused
    messages and the Triton pair kernel are not exercised.
    """
    saved = read_export(checkpoint)
    check_export(saved)
    model = reference.build(saved).to(device)
    original_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    enable_joined_nodes(model, fold_first=True)
    delattr(model, 'frozen_first_messages')
    bind(model, global_context)
    if radial_tables:
        model.compile_radial_tables(200, order=5)
        model.reference_chunk_size = 4096
    else:
        model.reference_chunk_size = 0
    actual = model.state_dict()
    changed = [name for name, value in original_state.items()
               if name not in actual or not torch.equal(value, actual[name].detach().cpu())]
    if changed:
        raise ValueError('Surrogate changed checkpoint tensors: ' + str(changed))
    model.surrogate_metadata = dict(kind='torch_surrogate_of_native_path', radial_tables=bool(radial_tables),
                                    reference_chunk_size=model.reference_chunk_size, joined_projections=True,
                                    folded_first_projections=True, first_message_cache=True,
                                    triton_components_exercised=False)
    return model, saved


__all__ = ['load', 'create', 'factory', 'create_full_engine', 'load_torch_surrogate', 'bind', 'check_export',
           'numerics_label', 'dependency_radius', 'read_export', 'state_features', 'system_batch', 'digest',
           'PRECISIONS', 'IMPLEMENTATION', '_prepare', '_global_state']
