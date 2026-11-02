"""Native vivace engine: the m26 L2 model (hidden 64, edge width 8, three blocks) on the C = 8 path.

Accepts tensors-only weights plus their architecture JSON. The expected vivace
architecture is an eligibility guard, never a replacement for the supplied
architecture. The node path matches hyff.engines.native.moderato (joined/folded
projections, frozen LayerNorm with FP64 or FP32 statistics, fused local
features, grouped element experts, compiled nodes); messages use the compact
eight-channel pair kernels (hyff.kernels.compact_pair_messages).
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import struct
import sys
from types import MethodType
from pathlib import Path

import numpy as np
import torch

from hyff.engines.native import base_adapter as base
from hyff.engines.native import frozen_nodes
from hyff.engines.native.centered_fp32 import check_fp32_statistics, normalization_installers
from hyff.kernels.layernorm_fp64 import FrozenLayerNorm
from hyff.kernels.grouped_experts import FrozenElementExpertMLP, enable_grouped_experts
from hyff.engines.native import moderato
from hyff.engines import reference
from hyff.models.element_model import ElementExpertMLP
from hyff.models.base_model import POLICY as LAYERNORM_POLICY
from hyff.data import collate, physical
from hyff.engines.force_engine import ForceEngine
from hyff.kernels.fused_linear import enable_fast_nodes
from hyff.kernels.joined_linear import JoinedLinear, enable_joined_nodes
from hyff.kernels.fused_messages import configure_neighbor_reserve

from hyff.weights import config_path, load_model, read_adapter_export, read_config, read_weights

PRECISIONS = ('tf32x3', 'fp32')
IMPLEMENTATION = 'hyff.engines.native.vivace'
EXPECTED_MODEL_CONFIG_SHA256 = 'a70ca3744ac1accb9bbca98152a8719b49acddddb58f45a003b3e19ba26beb0a'
STATE_FACTORY = 'hyff.models.base_model:build_model'
VIVACE_BACKBONE = dict(backbone='m26', hidden=64, tensor_hidden=32, edge_hidden=8,
                       interactions=3, layer_cutoffs=(3.5, 3.5, 4.5))
MESSAGE_CHANNEL_TILES = (16,)  # the C = 8 compact dispatcher requires settings (16, 8, 4, True).
ATOM_TILES = (1, 4, 8, 16)
C16_MESSAGE_SETTINGS = (16, 8, 4, True)

digest = reference.digest
state_features = reference.state_features
system_batch = reference.system_batch
bind = moderato.bind
numerics_label = moderato.numerics_label
_prepare = moderato._prepare
_global_state = moderato._global_state


def execution_context():
    return nullcontext()


def dependency_radius(model):
    return max(float(model.cutoff), float(sum(model.layer_cutoffs)))


def resolve_checkpoint(checkpoint):
    """Absolute path of a weight file."""
    return Path(checkpoint).resolve()


def tensor_bytes_sha256(state):
    """Canonical sha256_framed_model_state schema and contiguous CPU bytes."""
    if sys.byteorder != 'little':
        raise RuntimeError('unsupported_byte_order')
    result = hashlib.sha256(b'HyFF model_state tensor schema and bytes v1\0')
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        for label in (name, str(value.dtype)):
            raw = label.encode('utf-8')
            result.update(struct.pack('<Q', len(raw)))
            result.update(raw)
        result.update(struct.pack('<Q', value.ndim))
        for dimension in value.shape:
            result.update(struct.pack('<Q', dimension))
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        result.update(struct.pack('<Q', len(raw)))
        result.update(raw)
    return result.hexdigest()


def expected_model_config():
    """The vivace (m26) architecture configuration."""
    return dict(hidden=64, n_rbf=32, interactions=3, cutoff=4.5, switch_width=1.,
        edge_hidden=8, tensor_hidden=32, atomic_refinements=0,
        layer_cutoffs=[3.5, 3.5, 4.5], mlp_hidden=None, scalar_source_blocks=[],
        higher_invariants=True, tensor_mixing_rank=None, backbone='m26',
        atomic_numbers=list(range(1, 84)), expert_routing='element', expert_count=83,
        expert_hidden=64, expert_blocks=[0, 1, 2], radial_conditioning='shared',
        radial_hidden=64, condition_spin=True)


def read_export(path):
    """Checkpoint dict from hyff.weights.read_adapter_export, after the architecture JSON passed the vivace checks."""
    _, config, _ = _clean_configuration(path)
    saved = read_adapter_export(path)
    check_export(saved)
    if saved['model_config'] != config['model_config']:
        raise ValueError('Architecture changed during loading')
    return saved


def check_export(saved):
    """Require the architecture to match the vivace (m26) configuration."""
    if saved.get('model_factory') != STATE_FACTORY:
        raise ValueError('Expected the hyff.models.base_model StatePotential constructor')
    policy = saved.get('numerical_execution')
    if not isinstance(policy, dict) or policy.get('policy') != LAYERNORM_POLICY:
        raise ValueError('Expected the centered FP64 LayerNorm policy')
    config = saved['model_config']
    expected = expected_model_config()
    projected = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if (type(config) is not dict or config != expected
            or hashlib.sha256(projected.encode()).hexdigest() != EXPECTED_MODEL_CONFIG_SHA256):
        raise ValueError('Architecture differs from the vivace architecture')
    actual = {key: tuple(config[key]) if key == 'layer_cutoffs' else config[key]
              for key in VIVACE_BACKBONE}
    if actual != VIVACE_BACKBONE or float(config['cutoff']) != 4.5:
        raise ValueError('Expected the vivace (m26) architecture')
    if config['atomic_refinements'] or config['tensor_mixing_rank'] is not None or config['scalar_source_blocks']:
        raise ValueError('Unsupported refinement or scalar-source options')
    numbers = tuple(config['atomic_numbers'])
    if len(numbers) != len(set(numbers)) or not numbers:
        raise ValueError('Distinct nonempty source species list required')
    cutoffs = tuple(float(r) for r in config['layer_cutoffs'])
    if len(cutoffs) != config['interactions'] or any(not 0 < r <= config['cutoff'] for r in cutoffs):
        raise ValueError('One valid layer cutoff per source block is required')
    if config['radial_conditioning'] != 'shared' or config['expert_routing'] != 'element':
        raise ValueError('Element experts with shared radial filters required')
    return dict(atomic_numbers=numbers, n_elements=len(numbers), condition_spin=config['condition_spin'],
                hidden=config['hidden'], tensor_hidden=config['tensor_hidden'], edge_hidden=config['edge_hidden'],
                interactions=config['interactions'], cutoff=config['cutoff'], layer_cutoffs=cutoffs,
                n_rbf=config['n_rbf'], mlp_hidden=config['mlp_hidden'], expert_hidden=config['expert_hidden'],
                expert_blocks=tuple(config['expert_blocks']), radial_hidden=config['radial_hidden'],
                higher_invariants=config['higher_invariants'], switch_width=config['switch_width'],
                tier='vivace', vivace_backbone=True, architecture='vivace_m26',
                selection_kind=None, speed_only_artifact=False)


def _clean_configuration(checkpoint):
    """Validate factory, explicit architecture and policy before construction."""
    path = resolve_checkpoint(checkpoint)
    _, config = read_config(path)
    # Check eligibility before constructing a model; no architecture is retagged.
    from hyff.models.factory import factory_id
    factory = factory_id(config['model_id'])
    guard = dict(model_factory=factory, model_config=config['model_config'],
                 numerical_execution=config.get('numerical_execution'))
    # A JSON without a policy is checked with the default policy; load_source_weights
    # then checks the policy of the constructed model.
    if guard['numerical_execution'] is None:
        guard['numerical_execution'] = dict(policy=LAYERNORM_POLICY)
    check_export(guard)
    return path, config, factory


def load_source_weights(checkpoint):
    """Strictly load the weight file and its architecture JSON and verify every tensor byte.

    Only the weight file and its architecture JSON are read.
    """
    path, config, factory = _clean_configuration(checkpoint)
    model, loaded_config = load_model(path, 'cpu')
    if loaded_config != config:
        raise ValueError('Architecture changed during loading')
    state = read_weights(path)
    saved = dict(model_state=state, model_config=config['model_config'],
                 model_factory=factory, numerical_execution=model.numerical_execution,
                 architecture=getattr(model, 'architecture', 'hyff'),
                 selection_kind='standalone_weights_only')
    check_export(saved)
    expected = model.state_dict()
    if (set(expected) != set(state)
            or any(expected[name].shape != state[name].shape
                   or expected[name].dtype != state[name].dtype for name in expected)):
        raise ValueError('Architecture and weight tensor schema differ')
    before = tensor_bytes_sha256(state)
    if (tensor_bytes_sha256(expected) != before
            or any(not torch.equal(expected[name], state[name]) for name in expected)):
        raise ValueError('Loaded weight tensor bytes differ')
    provenance = dict(
        weight_source_only=True, weight_source_block='tensors_only_weights',
        architecture_model_id=config['model_id'], architecture_format=config['format'],
        architecture_json_sha256=digest(config_path(path)),
        weights_file_sha256=digest(path),
        model_preset='m26', model='vivace',
        model_state_tensor_count=len(state), model_state_canonical_sha256=before,
        expected_model_config_sha256=EXPECTED_MODEL_CONFIG_SHA256)
    return model, saved, provenance


def _load_source_base(path, device, *, global_context, owned_weights, reserve):
    """The native preparation of base_adapter.load with node_mode='torch', on weights from load_source_weights."""
    model, saved, provenance = load_source_weights(path)
    model.to(device).eval().requires_grad_(False)
    model._engine_global_context = global_context
    model.forward = MethodType(base._forward, model)
    model.initial_scalar = MethodType(base._initial_scalar, model)
    model.prepare_energy = MethodType(base._prepare_energy, model)
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
    actual = model.state_dict()
    if any(name not in actual or actual[name].dtype != value.dtype
           or not torch.equal(value, actual[name].detach().cpu())
           for name, value in saved['model_state'].items()):
        raise ValueError('Native preparation changed model_state tensors')
    model.deployment_metadata = dict(provenance,
        checkpoint=str(path), checkpoint_sha256=digest(path),
        checkpoint_tensors_verified=len(saved['model_state']), learned_weights_changed=False,
        atomic_numbers=list(model.atomic_numbers), total_charge_feature='global_Q/global_N',
        global_context=global_context, node_mode='torch',
        layer_cutoffs=list(model.layer_cutoffs), dependency_radius_A=dependency_radius(model),
        normalization_policy=LAYERNORM_POLICY,
        radial_table=dict(order=5, intervals=200, requires_independent_analytic_parity_gate=True))
    return model, saved


def _next_pow2(value):
    return 1 << (int(value) - 1).bit_length()


def _cdiv(a, b):
    return -(-int(a) // int(b))


def default_feature_atom_tile(widths):
    """Four atoms per program up to the measured 128-channel tile, one above (register pressure)."""
    return 4 if _next_pow2(max(widths['hidden'], widths['tensor_hidden'])) <= 128 else 1


def _node_macs_per_atom(widths, first, terminal):
    """Node-side multiply-accumulates per atom and block of the native forward (edge kernels excluded)."""
    h, t, e = widths['hidden'], widths['tensor_hidden'], widths['edge_hidden']
    m = widths['mlp_hidden'] or h
    invariants = (5 if widths['higher_invariants'] else 2) * t
    macs = h * m + m * (3 if first else 7) * e                       # scalar_message
    macs += 0 if e == h else e * h                                    # edge_scalar_out
    if e != t:
        macs += 24 * e * t if first else 16 * e * t                    # folded first [3T, E] on 3+5 components / edge in+out
    macs += 0 if first else 16 * t * t                                # joined_vector, joined_tensor (first block folded)
    macs += 0 if t == h else 2 * t * h                                # joined_invariants
    macs += (h + invariants) * widths['expert_hidden'] + widths['expert_hidden'] * (h + (2 if terminal else 4) * t)
    macs += h * m + m                                                 # readout
    return int(macs)


def execution_geometry(widths, *, message_channel_tile=16, layernorm_atom_tile=8, feature_atom_tile=None, precision='tf32x3'):
    """Width-derived launch facts of the native kernels for this model (diagnostics)."""
    h, t, e, n = widths['hidden'], widths['tensor_hidden'], widths['edge_hidden'], widths['n_elements']
    if feature_atom_tile is None:
        feature_atom_tile = default_feature_atom_tile(widths)
    settings = (int(message_channel_tile), 8, 4, True)
    tuned = e == 16 and settings == C16_MESSAGE_SETTINGS
    blocks = []
    for index, cutoff in enumerate(widths['layer_cutoffs']):
        first = index == 0
        terminal = index == widths['interactions'] - 1
        expert_in = h + (5 if widths['higher_invariants'] else 2) * t
        expert_out = h + (2 if terminal else 4) * t
        expert_hidden = widths['expert_hidden']
        blocks.append(dict(
            block=index, cutoff_A=cutoff, scalar_source=first, terminal=terminal,
            message_table_channels=(3 if first else 7) * e, message_channel_tile=settings[0],
            message_programs_per_atom=_cdiv(e, settings[0]),
            message_forward_kernel='compact_dispatch._forward_pair (C8, BC8, BM16)',
            message_backward_kernel=('compact_dispatch._backward_pair_scalar (C8)' if first else
                                     'compact_dispatch._backward_pair_tensor (C8)'),
            first_message_table=[119, 3 * e] if first else None,
            folded_first_projection=[3 * t, e] if first else None,
            joined_vector_tensor=[2 * t, t], joined_invariants=[h, 2 * t],
            expert=dict(inputs=expert_in, hidden=expert_hidden, outputs=expert_out, count=n,
                        grouped_grid=[f'ceil(N/64) + {n}', _cdiv(expert_hidden, 64), _cdiv(expert_out, 64)],
                        k_steps=[_cdiv(expert_in, 32), _cdiv(expert_hidden, 32)],
                        stacked_weight_bytes=4 * n * (expert_in * expert_hidden + expert_hidden * expert_out)),
            node_macs_per_atom_forward=_node_macs_per_atom(widths, first, terminal)))
    return dict(
        tier=widths['tier'], hidden=h, tensor_hidden=t, edge_hidden=e, blocks=widths['interactions'], n_elements=n,
        fused_message_settings=list(settings), message_kernels='compact_pair_c8',
        layernorm=dict(kernel='frozen_layernorm64 (FP64 statistics)', channels=h, channel_tile=_next_pow2(h), atom_tile=layernorm_atom_tile,
                       modules=3 * widths['interactions']),
        local_features=dict(kernel='frozen_local_features64 (FP64 statistics)', scalar=h, tensor=t,
                            channel_tile=_next_pow2(max(h, t)), atom_tile=feature_atom_tile,
                            output_width=h + (5 if widths['higher_invariants'] else 2) * t),
        node_products=dict(kernel='fused_linear._mm/_mm_grouped/_mm_silu', precision=precision,
                           tiles='BM 64, BN 32 (N<=32) or 64, BK 32; N and K masked' if precision != 'fp32' else 'torch FP32 cuBLAS'),
        grouped_experts=dict(kernel='grouped_experts._grouped_product', expert_search_width=_next_pow2(n),
                             row_tile=64, column_tile=64, k_tile=32, stacked_weight_bytes_total=sum(b['expert']['stacked_weight_bytes'] for b in blocks)),
        radial_tables=dict(intervals=200, order=5, per_block_channels=[b['message_table_channels'] for b in blocks],
                           pair_rows=n * n, n_rbf=widths['n_rbf']),
        pair_kernel=f'hyff.kernels.pair_energy_species.pair_energy (NE = {n})',
        per_block=blocks, node_macs_per_atom_forward_total=sum(b['node_macs_per_atom_forward'] for b in blocks),
        dependency_radius_A=max(widths['cutoff'], sum(widths['layer_cutoffs'])))


def load(checkpoint, device='cuda', *, global_context=None, owned_weights=None, reserve=1.25,
         precision='tf32x3', compiled_nodes=True, layernorm_atom_tile=8, feature_atom_tile=None,
         message_channel_tile=16, frozen_layernorm=True, fused_features=True, grouped_experts=True,
         statistics='fp64'):
    if precision not in PRECISIONS:
        raise ValueError(f'precision must be one of {PRECISIONS}')
    enable_norm, enable_features = normalization_installers(statistics, precision)
    if statistics == 'fp32':
        # FP32 statistics are validated with these construction settings only.
        if feature_atom_tile is None:
            feature_atom_tile = 4
        required = dict(layernorm_atom_tile=8, feature_atom_tile=4, message_channel_tile=16,
                        compiled_nodes=True, frozen_layernorm=True, fused_features=True,
                        grouped_experts=True)
        given = dict(layernorm_atom_tile=layernorm_atom_tile, feature_atom_tile=feature_atom_tile,
                     message_channel_tile=message_channel_tile, compiled_nodes=compiled_nodes,
                     frozen_layernorm=frozen_layernorm, fused_features=fused_features,
                     grouped_experts=grouped_experts)
        for key, value in required.items():
            if type(given[key]) is not type(value) or given[key] != value:
                raise ValueError(f'FP32 statistics require {key}={value}')
    if message_channel_tile not in MESSAGE_CHANNEL_TILES:
        raise ValueError(f'message_channel_tile must be one of {MESSAGE_CHANNEL_TILES}')
    if layernorm_atom_tile not in ATOM_TILES:
        raise ValueError(f'layernorm_atom_tile must be one of {ATOM_TILES}')
    path = resolve_checkpoint(checkpoint)
    saved = read_export(path)
    widths = check_export(saved)
    if feature_atom_tile is None:
        feature_atom_tile = default_feature_atom_tile(widths)
    if feature_atom_tile not in ATOM_TILES:
        raise ValueError(f'feature_atom_tile must be one of {ATOM_TILES}')
    if global_context is not None:
        global_context = dict(global_context)
    # Base loader: factory build, strict load without cast, quintic radial tables (200 intervals per
    # layer cutoff plus the n_elements^2 pair table), CSR settings, neighbor reserve, tensor check.
    model, saved = _load_source_base(path, device, global_context=global_context,
                                      owned_weights=owned_weights, reserve=reserve)
    # base.load fixes the C = 16 message tiling (16, 8, 4, True). The channel tile is its only
    # width-dependent entry: the general kernels launch ceil(edge_hidden / tile) programs per atom.
    settings = (int(message_channel_tile), 8, 4, True)
    model.fused_message_settings = settings
    if tuple(model.atomic_numbers) != widths['atomic_numbers'] or bool(model.condition_spin) != widths['condition_spin']:
        raise ValueError('Built model differs from the export config')
    if (model.config.hidden, model.tensor_hidden, model.blocks[0].message_hidden, len(model.blocks)) != (
            widths['hidden'], widths['tensor_hidden'], widths['edge_hidden'], widths['interactions']):
        raise ValueError('Built widths differ from the export config')
    if model.state_projection.weight.shape != (model.config.hidden, 2 if model.condition_spin else 1):
        raise ValueError('Unexpected state projection shape')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('The full adapter requires FP32 learned weights')
    if len(model.radial_tables) != len(model.blocks) + 1 or model.radial_tables[-1].coefficients.shape[2] != model.n_elements ** 2:
        raise ValueError('One radial table per block plus the n_elements^2 pair table is required')
    for block, table, cutoff in zip(model.blocks, model.radial_tables, model.layer_cutoffs):
        if float(table.cutoff) != float(cutoff) or table.coefficients.shape[2] != (3 if block.input_is_scalar else 7) * block.message_hidden:
            raise ValueError('Radial table does not match its block')
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
    if frozen_layernorm:
        enable_norm(model, atom_tile=layernorm_atom_tile)
    if fused_features:
        enable_features(model, atom_tile=feature_atom_tile)
    grouped = bool(grouped_experts) and precision != 'fp32'
    if grouped:
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
    expert_modules = [model.blocks[i].update_mlp for i in model.config.expert_blocks]
    if grouped and not all(isinstance(m, FrozenElementExpertMLP) and m.count == model.n_elements for m in expert_modules):
        raise ValueError('Grouped experts were not installed for every expert block')
    if not grouped and not all(isinstance(m, ElementExpertMLP) and len(m.experts) == model.n_elements for m in expert_modules):
        raise ValueError('The ungrouped variants require plain element experts in every expert block')
    frozen_count = sum(isinstance(m, FrozenLayerNorm) for m in model.modules())
    if frozen_layernorm and frozen_count != 3 * len(model.blocks):
        raise ValueError(f'Expected {3 * len(model.blocks)} frozen LayerNorms, found {frozen_count}')
    if precision == 'fp32':
        expert_execution = 'per_species_loop_fp32'
    elif grouped:
        expert_execution = 'grouped_triton_' + precision
    else:
        expert_execution = 'per_species_loop_triton_' + precision
    geometry = execution_geometry(widths, message_channel_tile=message_channel_tile, layernorm_atom_tile=layernorm_atom_tile,
                                  feature_atom_tile=feature_atom_tile, precision=precision)
    loaded_state = {name: actual[name] for name in saved['model_state']}
    if tensor_bytes_sha256(loaded_state) != model.deployment_metadata['model_state_canonical_sha256']:
        raise ValueError('Engine preparation changed model_state tensor bytes')
    model.inference_backend = 'vivace_native'
    model.deployment_metadata = dict(model.deployment_metadata,
        implementation=IMPLEMENTATION, inference_backend=model.inference_backend, node_mode='full', precision=precision,
        numerics=numerics_label(precision), compiled_nodes=bool(compiled_nodes),
        tier=widths['tier'], architecture=widths['architecture'], speed_only_artifact=widths['speed_only_artifact'],
        widths={k: widths[k] for k in ('hidden', 'tensor_hidden', 'edge_hidden', 'interactions', 'layer_cutoffs', 'n_rbf',
                                       'expert_hidden', 'radial_hidden', 'mlp_hidden', 'higher_invariants')},
        parameters=sum(p.numel() for p in model.parameters()),
        joined_projections=True, folded_first_projections=True,
        fused_local_features=bool(fused_features), fused_local_features_atom_tile=feature_atom_tile if fused_features else None,
        grouped_experts=grouped, active_experts_per_atom=1, expert_execution=expert_execution,
        node_products='triton_' + precision if precision != 'fp32' else 'torch_fp32_cublas',
        frozen_layernorm=bool(frozen_layernorm), frozen_layernorm_atom_tile=layernorm_atom_tile if frozen_layernorm else None,
        frozen_layernorm_modules=frozen_count,
        layernorm_execution='frozen_triton_fp64_internal' if frozen_layernorm else 'torch_forward_fp64_policy',
        local_feature_execution='fused_triton_fp64_statistics' if fused_features else 'torch_reference_node_update',
        fused_message_settings=list(settings), message_kernels=geometry['message_kernels'],
        execution_geometry=geometry,
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
        pair_kernel=geometry['pair_kernel'],
        pair_table_rows=int(model.radial_tables[-1].coefficients.shape[2]),
        radial_table_cutoffs_A=[float(t.cutoff) for t in model.radial_tables],
        radial_table_channels=[int(t.coefficients.shape[2]) for t in model.radial_tables],
        dependency_radius_A=dependency_radius(model),
        weight_source_only=True, weight_source_block='tensors_only_weights')
    if statistics == 'fp32':
        if len(model.blocks) != 3 or model.config.hidden != 64 or model.tensor_hidden != 32:
            raise ValueError('FP32 statistics are validated for the vivace architecture only')
        check_fp32_statistics(model, expected_norms=9)
    return model, saved


LOAD_OPTIONS = ('reserve', 'precision', 'compiled_nodes', 'layernorm_atom_tile', 'feature_atom_tile',
                'message_channel_tile', 'frozen_layernorm', 'fused_features', 'grouped_experts', 'statistics')


def create(record, owned_weights=None, *, checkpoint, global_context=None, device='cuda', skin=1.5, **options):
    capture_branch_policy = options.pop('capture_branch_policy', 'first')
    unknown = set(options) - set(LOAD_OPTIONS)
    if unknown:
        raise TypeError(f'Unknown load options: {sorted(unknown)}')
    record = {**record, 'total_charge': record.get('total_charge', 0.)}
    context = dict(global_context) if global_context is not None else dict(
        n_atoms=len(record['positions']), total_charge=record['total_charge'],
        multiplicity=record.get('multiplicity', 1))
    if owned_weights is not None and len(owned_weights) != len(record['positions']):
        raise ValueError('One final atomic energy weight is required for every local atom')
    model, _ = load(checkpoint, device, global_context=context, owned_weights=owned_weights, **options)
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
            capture_branch_policy='first', **options):
    prepared = create(record, owned_weights, checkpoint=checkpoint, global_context=global_context, device=device,
                      skin=skin, capture_branch_policy=capture_branch_policy, **options)
    return prepared['engine'], prepared['metadata']


def create_full_engine(positions, atomic_numbers, cell, checkpoint, device='cuda', total_charge=0., multiplicity=1,
                       skin=1.5, *, capture_branch_policy='first', **options):
    """Create the vivace engine for one system; same call shape as moderato.create_full_engine plus width options."""
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    n = len(numbers)
    context = dict(n_atoms=n, total_charge=float(total_charge), multiplicity=int(multiplicity),
                   state_features=state_features(n, total_charge, multiplicity))
    record = dict(positions=np.asarray(positions, dtype=np.float32), atomic_numbers=numbers, total_charge=float(total_charge))
    if cell is not None:
        record.update(cell=np.asarray(cell, dtype=np.float32), pbc=np.ones(3, dtype=bool))
    made = create(record, checkpoint=str(resolve_checkpoint(checkpoint)), global_context=context, device=device,
                  skin=skin, capture_branch_policy=capture_branch_policy, **options)
    made['metadata'] = dict(made['metadata'], backend='vivace_native_csr_c8',
                            global_context=context, skin_A=float(skin))
    return made


# ---------------------------------------------------------------------------------------------
# No CPU surrogate for this engine.
