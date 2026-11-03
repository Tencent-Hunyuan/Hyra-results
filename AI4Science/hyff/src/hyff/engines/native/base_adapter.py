"""Shared native-engine adapter: CSR neighbours, quintic radial tables, species and global state.

No checkpoint tensor is rounded or edited. The quintic radial interpolation
approximates the analytic radial filters and is checked against them separately.
Owned energy weights mask final atomic energies only, including symmetric
half-pair terms, and keep coordinate derivatives on every local/ghost atom.
"""
from contextlib import nullcontext
import hashlib
import importlib
from pathlib import Path
from types import MethodType

import torch

from hyff.models.element_model import prepare_batch
from hyff.data import collate, physical
from hyff.engines.force_engine import ForceEngine
from hyff.models.experts import ElementPartition
from hyff.kernels.fused_messages import PreparedL2Neighbors, configure_neighbor_reserve
from hyff.weights import read_adapter_export



def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_path(checkpoint):
    return Path(checkpoint).resolve()


def dependency_radius(checkpoint):
    saved = read_adapter_export(checkpoint_path(checkpoint))
    config = saved['model_config']
    return max(float(config['cutoff']), sum(config['layer_cutoffs']))


def execution_context():
    return nullcontext()


def _prepare(self, batch):
    batch = prepare_batch(batch)
    partition = batch['_hyff_atomic_number_partition']
    unsupported = set(partition.groups).difference(self.atomic_numbers)
    if unsupported:
        raise ValueError(f'Unsupported checkpoint elements: {sorted(unsupported)}')
    bound = ElementPartition(tuple(partition.groups.get(z, partition.empty) for z in self.atomic_numbers),
                             len(batch['positions']), partition.empty, partition.empty, partition.empty)
    for index in self.config.expert_blocks:
        self.blocks[index].update_mlp.partition = bound
    # This calculation is outside capture and independent of coordinates.
    scalar = self.embedding(batch['atomic_numbers'])
    counts = (batch['ptr'][1:] - batch['ptr'][:-1]).to(scalar)
    charge = batch.get('total_charge', torch.zeros_like(counts)).to(scalar)
    context = self._engine_global_context
    if context is not None:
        if len(counts) != 1:
            raise ValueError('A global partition context represents one physical system')
        counts = torch.full_like(counts, context['n_atoms'])
        charge = torch.full_like(charge, context['total_charge'])
    if not bool(torch.isfinite(charge).all() & (counts > 0).all()):
        raise ValueError('Finite total charge and nonempty global system required')
    features = [charge / counts]
    if self.condition_spin:
        multiplicity = torch.full_like(counts, 1 if context is None else context.get('multiplicity', 1))
        features.append((multiplicity - 1) / counts)
    elif context is not None and context.get('multiplicity', 1) != 1:
        raise ValueError('Charge-only checkpoint has no trained nonsinglet response')
    state = self.state_projection(torch.stack(features, dim=-1))
    batch['_engine_initial_scalar'] = (scalar + state[batch['batch']]).detach()
    return batch


def _initial_scalar(self, batch):
    return batch['_engine_initial_scalar']


def _forward(self, batch, **kwargs):
    from hyff.engines.native.base_forward import reference_forward
    if '_engine_initial_scalar' not in batch:
        batch = _prepare(self, batch)
    return reference_forward(self, batch, **kwargs)


class PreparedGeneralNeighbors(PreparedL2Neighbors):
    def __init__(self, model, batch, edges):
        self.elements = model.element_lut[batch['atomic_numbers']].contiguous()
        super().__init__(model, batch, edges)
        for view in self.radius_views.values():
            view.elements = self.elements


def _prepare_energy(self, batch, edges):
    batch = _prepare(self, batch)
    return PreparedGeneralNeighbors(self, batch, edges)


def load(checkpoint, device='cuda', *, global_context=None, owned_weights=None,
         reserve=1.25, node_mode='torch'):
    path = checkpoint_path(checkpoint)
    saved = read_adapter_export(path)
    from hyff.models.factory import build_model
    model = build_model(dict(model_id=saved['model_factory'], model_config=saved['model_config']))
    model.to(dtype=saved['model_state']['embedding.weight'].dtype)
    model.load_state_dict(saved['model_state'], strict=True)
    model.to(device).eval().requires_grad_(False)
    model._engine_global_context = global_context
    model.forward = MethodType(_forward, model)
    model.initial_scalar = MethodType(_initial_scalar, model)
    model.prepare_energy = MethodType(_prepare_energy, model)
    if owned_weights is not None:
        weights = torch.as_tensor(owned_weights, device=device, dtype=torch.float32).detach().clone()
        if weights.ndim != 1 or not bool(torch.isfinite(weights).all()):
            raise ValueError('Finite per-local-atom final energy weights required')
        model.register_buffer('owned_energy_weights', weights, persistent=False)
    model.reference_chunk_size = 4096
    model.compile_radial_tables(200, order=5)
    model.fused_messages = model.fused = True
    model.fused_message_settings = (16, 8, 4, True)
    model.filter_skin_neighbors = True
    model.inference_backend = 'native_csr_quintic'
    configure_neighbor_reserve(model, reserve)
    if model.config.edge_hidden == 8:
        from hyff.kernels.compact_pair_messages import _compact_message_forward
        for block in model.blocks:
            block.forward = MethodType(_compact_message_forward, block)
    if node_mode in ('tf32x3', 'fp16x3'):
        from hyff.kernels.fused_linear import enable_fast_nodes, FastLinear
        from hyff.models.l2_potential import _RowSelectedLinear
        def convert_selected(module):
            for name, child in list(module.named_children()):
                if isinstance(child, _RowSelectedLinear):
                    child.in_features = child.weight.shape[1]
                    child.out_features = child.weight.shape[0]
                    setattr(module, name, FastLinear(child, node_mode))
                else:
                    convert_selected(child)
        convert_selected(model)
        # Deliberately call only linear conversion. L2Potential's convenience
        # wrapper also replaces LayerNorm, which would violate the model's FP64
        # numerical policy. FP32 learned tensors stay exactly unchanged here.
        enable_fast_nodes(model, fuse_activations=True, strided_inputs=True, precision=node_mode)
    if node_mode in ('joined', 'compiled', 'tf32x3', 'fp16x3'):
        from hyff.kernels.joined_linear import enable_joined_nodes
        enable_joined_nodes(model, fold_first=False)
        # The generic helper caches embedding-only messages. State conditioning
        # must remain visible, including nonzero charge in validation checks.
        delattr(model, 'frozen_first_messages')
    if node_mode == 'compiled':
        model.enable_compiled_nodes()
    elif node_mode not in ('torch', 'joined', 'tf32x3', 'fp16x3'):
        raise ValueError(node_mode)
    actual = model.state_dict()
    changed = [name for name, value in saved['model_state'].items()
               if name not in actual or not torch.equal(value, actual[name].detach().cpu())]
    if changed:
        raise ValueError('Checkpoint tensors changed: ' + str(changed))
    model.deployment_metadata = dict(checkpoint=str(path), checkpoint_sha256=digest(path),
        checkpoint_tensors_verified=len(saved['model_state']), learned_weights_changed=False,
        atomic_numbers=list(model.atomic_numbers), total_charge_feature='global_Q/global_N',
        global_context=global_context, node_mode=node_mode,
        layer_cutoffs=list(model.layer_cutoffs), dependency_radius_A=dependency_radius(path),
        numerical_execution=saved.get('numerical_execution'),
        radial_table=dict(order=5, intervals=200, requires_independent_analytic_parity_gate=True))
    return model, saved


def create(record, owned_weights=None, *, checkpoint, global_context=None,
            device='cuda', skin=1.5, reserve=1.25, node_mode='torch'):
    record = {**record, 'total_charge': record.get('total_charge', 0.)}
    model, _ = load(checkpoint, device, global_context=global_context,
                    owned_weights=owned_weights, reserve=reserve, node_mode=node_mode)
    data = physical(collate([record], device))
    engine = ForceEngine(model, data, 'cuda_graph', skin=skin)
    energy, force = engine(data)
    if not bool(torch.isfinite(energy).all() & torch.isfinite(force).all()):
        engine.close()
        raise ValueError('Nonfinite native prepared force')
    return dict(model=model, batch=data, engine=engine, metadata=model.deployment_metadata)


def factory(record, owned_weights=None, *, checkpoint, global_context=None,
            device='cuda', skin=1.5, reserve=1.25, node_mode='torch'):
    prepared = create(record, owned_weights, checkpoint=checkpoint,
                      global_context=global_context, device=device, skin=skin,
                      reserve=reserve, node_mode=node_mode)
    return prepared['engine'], prepared['metadata']
