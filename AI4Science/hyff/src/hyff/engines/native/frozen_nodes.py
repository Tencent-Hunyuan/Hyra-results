"""Frozen-node helpers shared by the native engines (tensor checks, precision conversion, global state).

This is an inference adapter, not a different potential or a training recipe.
The one-system contract makes the initial species lookup depend on the full
physical system's Q/N even when only an owned-plus-halo atom subset is present.
First-message tables belong to the prepared batch, never mutable global state.
"""
from contextlib import nullcontext
import math
from types import MethodType

import torch
from torch.nn import functional as F

from hyff.engines.native import base_adapter as base
from hyff.models.element_model import prepare_batch
from hyff.data import collate, physical
from hyff.engines.force_engine import ForceEngine
from hyff.kernels.fused_linear import FastLinear, enable_fast_nodes
from hyff.kernels.layernorm_fp64 import FrozenLayerNorm, enable_frozen_layernorm
from hyff.kernels.local_features_fp64 import enable_frozen_local_features
from hyff.kernels.joined_linear import JoinedLinear, enable_joined_nodes
from hyff.models.l2_potential import _RowSelectedLinear

from hyff.kernels.grouped_experts import bind_partition, enable_grouped_experts
from hyff.models.base_model import POLICY


checkpoint_path = base.checkpoint_path
digest = base.digest


def execution_context():
    return nullcontext()


def dependency_radius(checkpoint='large'):
    radius = base.dependency_radius(checkpoint)
    if radius != 21.0:
        raise ValueError('This adapter requires the 21.0 A dependency radius')
    return radius


def _global_state(self, batch):
    """Return global intensive state; a halo's local atom count never replaces N."""
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
    context = self._engine_global_context
    if context is not None:
        n_atoms = context['n_atoms']
        if (not math.isfinite(float(n_atoms)) or int(n_atoms) != n_atoms
                or n_atoms < len(numbers)):
            raise ValueError('The full system atom count must be integral and cover every halo atom')
        count = torch.full_like(count, n_atoms)
        charge = torch.full_like(charge, context['total_charge'])
        multiplicity = torch.full_like(count, context.get('multiplicity', 1))
    if not bool(torch.isfinite(charge).all() & (count > 0).all()):
        raise ValueError('Finite total charge and nonempty global system required')
    if multiplicity.shape != count.shape or not bool((multiplicity == 1).all()):
        raise ValueError('This model has no trained nonsinglet response')
    return (charge / count).reshape(1, 1)


@torch.no_grad()
def _prepare(self, batch):
    """Prepare state/species-only data outside every force graph and derivative."""
    batch = prepare_batch(batch)
    partition = bind_partition(batch['atomic_numbers'], self.atomic_numbers)
    state_features = _global_state(self, batch)
    state = F.linear(state_features, self.state_projection.weight,
                     self.state_projection.bias)
    initial_table = self.embedding.weight + state
    first = self.blocks[0]
    # These small setup-only products use ordinary FP32 math on both CPU and
    # GPU. Force-time TF32x3/FP16x3 never rounds any stored checkpoint tensor.
    from hyff.models.base_model import forward_fp64
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
    from hyff.engines.native.frozen_nodes_forward import reference_forward
    if '_engine_initial_scalar' not in batch:
        batch = _prepare(self, batch)
    # Bind the schedule belonging to this batch even after another batch was
    # prepared on the same model. CUDA graphs hold their own captured arrays.
    for index in self.config.expert_blocks:
        self.blocks[index].update_mlp.partition = batch['_engine_partition']
    return reference_forward(self, batch, **kwargs)


def _prepare_energy(self, batch, edges):
    prepared = _prepare(self, batch)
    return base.PreparedGeneralNeighbors(self, prepared, edges)


def _convert_selected(module, precision):
    for name, child in list(module.named_children()):
        if isinstance(child, _RowSelectedLinear):
            child.in_features = child.weight.shape[1]
            child.out_features = child.weight.shape[0]
            converted = FastLinear(child, precision)
            converted.strided_inputs = True
            setattr(module, name, converted)
        else:
            _convert_selected(child, precision)


def _verify_tensors(model, expected, label):
    actual = model.state_dict()
    if set(actual) != set(expected):
        raise ValueError(f'{label} state keys changed: missing={sorted(set(expected)-set(actual))}, '
                         f'added={sorted(set(actual)-set(expected))}')
    changed = [name for name, value in expected.items()
               if value.dtype != actual[name].dtype or value.shape != actual[name].shape
               or not torch.equal(value.detach().cpu(), actual[name].detach().cpu())]
    if changed:
        raise ValueError(f'{label} tensors changed: {changed}')
    return len(actual)


def load(checkpoint='large', device='cuda', *, global_context=None, owned_weights=None,
         reserve=1.25, precision='tf32x3', compiled_nodes=True, layernorm_atom_tile=8):
    if precision not in ('tf32x3', 'fp16x3'):
        raise ValueError('Use TF32x3 or FP16x3 without checkpoint rounding')
    if global_context is not None:
        global_context = dict(global_context)
    model, saved = base.load(checkpoint, device, global_context=global_context,
        owned_weights=owned_weights, reserve=reserve, node_mode='torch')
    config = model.config
    if (config.backbone != 'l4a0' or config.hidden != 128 or config.tensor_hidden != 64
            or config.edge_hidden != 16 or config.interactions != 4
            or tuple(config.layer_cutoffs) != (3.5, 3.5, 6.0, 8.0)
            or model.condition_spin or saved.get('numerical_execution', {}).get('policy') != POLICY):
        raise ValueError('This adapter is for FP32 charge-only models')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('The frozen-node adapter requires FP32 learned weights')
    original_state = {name: value.detach().cpu().clone()
                      for name, value in model.state_dict().items()}
    original_parameters = tuple(model.named_parameters())

    # Join while modules are ordinary torch operations so CPU setup is valid.
    # The helper's embedding-only lookup is explicitly discarded; _prepare
    # creates the conditioned replacement belonging to each prepared batch.
    enable_joined_nodes(model, fold_first=True)
    delattr(model, 'frozen_first_messages')
    _convert_selected(model, precision)
    enable_fast_nodes(model, fuse_activations=True, strided_inputs=True, precision=precision)
    for module in model.modules():
        if isinstance(module, JoinedLinear):
            module.fast, module.precision = True, precision
    enable_frozen_layernorm(model, atom_tile=layernorm_atom_tile)
    enable_frozen_local_features(model, atom_tile=4)
    enable_grouped_experts(model, precision)
    model.forward = MethodType(_forward, model)
    model.initial_scalar = MethodType(_initial_scalar, model)
    model.prepare_energy = MethodType(_prepare_energy, model)
    if compiled_nodes:
        model.enable_compiled_nodes()

    unchanged_count = _verify_tensors(model, original_state, 'Engine preparation')
    actual = model.state_dict()
    for name, value in saved['model_state'].items():
        if (name not in actual or actual[name].dtype != value.dtype
                or not torch.equal(actual[name].detach().cpu(), value)):
            raise ValueError(f'Checkpoint tensor changed: {name}')
    parameters = tuple(model.named_parameters())
    if (len(parameters) != len(original_parameters)
            or any(new_name != old_name or new_value is not old_value
                   for (new_name, new_value), (old_name, old_value)
                   in zip(parameters, original_parameters))):
        raise ValueError('Engine preparation changed the learned parameter namespace or objects')
    model.inference_backend = 'frozen_nodes'
    model.deployment_metadata = dict(model.deployment_metadata,
        node_mode='full', precision=precision, compiled_nodes=bool(compiled_nodes),
        joined_projections=True, folded_first_projections=True,
        fused_local_features=True, fused_local_features_atom_tile=4,
        grouped_experts=True, active_experts_per_atom=1,
        frozen_layernorm=True, frozen_layernorm_atom_tile=layernorm_atom_tile,
        frozen_layernorm_modules=sum(isinstance(m, FrozenLayerNorm) for m in model.modules()),
        prepared_state_tensors_verified=unchanged_count,
        original_parameter_objects_and_namespace_verified=True,
        first_message_cache='batch_owned_species_lookup_conditioned_on_global_Q_over_N',
        first_message_cache_arithmetic='setup_FP32_linear_centered_FP64_LayerNorm',
        normalization_policy='centered_FP64_internal_same_epsilon',
        fused_feature_normalization_precision='FP64',
        one_physical_system_per_engine=True, topology_input=False,
        dependency_radius_A=dependency_radius(checkpoint),
        implementation='hyff.engines.native.frozen_nodes')
    return model, saved


def create(record, owned_weights=None, *, checkpoint='large', global_context=None,
           device='cuda', skin=1.5, reserve=1.25, precision='tf32x3',
           compiled_nodes=True, layernorm_atom_tile=8):
    record = {**record, 'total_charge': record.get('total_charge', 0.)}
    context = dict(global_context) if global_context is not None else dict(
        n_atoms=len(record['positions']), total_charge=record['total_charge'],
        multiplicity=record.get('multiplicity', 1))
    if owned_weights is not None and len(owned_weights) != len(record['positions']):
        raise ValueError('One final atomic energy weight is required for every local atom')
    model, _ = load(checkpoint, device, global_context=context, owned_weights=owned_weights,
        reserve=reserve, precision=precision, compiled_nodes=compiled_nodes,
        layernorm_atom_tile=layernorm_atom_tile)
    batch = physical(collate([record], device))
    engine = ForceEngine(model, batch, 'cuda_graph', skin=skin)
    try:
        energy, force = engine(batch)
        if not bool(torch.isfinite(energy).all() & torch.isfinite(force).all()):
            raise ValueError('Nonfinite prepared energy or force')
    except BaseException:
        engine.close()
        raise
    return dict(model=model, batch=batch, engine=engine, metadata=model.deployment_metadata)


def factory(record, owned_weights=None, *, checkpoint='large', global_context=None,
            device='cuda', skin=1.5, reserve=1.25, precision='tf32x3',
            compiled_nodes=True, layernorm_atom_tile=8):
    prepared = create(record, owned_weights, checkpoint=checkpoint, global_context=global_context,
        device=device, skin=skin, reserve=reserve, precision=precision,
        compiled_nodes=compiled_nodes, layernorm_atom_tile=layernorm_atom_tile)
    return prepared['engine'], prepared['metadata']
