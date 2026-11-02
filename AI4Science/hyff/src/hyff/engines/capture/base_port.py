"""Base-node preparation of an already-built EFM model for the captured engines.

No factory, configuration file, or checkpoint is loaded here. ``bind_base``
mutates its argument; keep a separately built or deep-copied model for the
independent eager reference. Saving the old bound forward alone is insufficient
because the node modules and execution attributes change in place.

Existing EFM block forwards are intentionally left intact: install the EFM execution adapter before evaluating this model.
Node compilation and CUDA graph capture belong to that subsequent adapter.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from hyff.kernels.layernorm_fp64 import FrozenLayerNorm
from hyff.kernels.local_features_fp64 import enable_frozen_local_features
from hyff.kernels.grouped_experts import FrozenElementExpertMLP
from hyff.engines.native import moderato as native_moderato
from hyff.models.element_model import ElementExpertMLP
from hyff.models.base_model import POLICY
from hyff.models.stabilized import StabilizedUpdateMLP
from hyff.kernels.fused_linear import FastLinear
from hyff.kernels.fused_messages import configure_neighbor_reserve
from hyff.kernels.joined_linear import JoinedLinear, enable_joined_nodes
from hyff.models.l2_potential import L2Interaction, L2Potential, _RowSelectedLinear


PRECISIONS = ('fp32', 'tf32x3', 'fp16x3')
ATOM_TILES = (1, 4, 8, 16)
MESSAGE_CHANNEL_TILES = (8, 16, 32, 64)


def _inner(module):
    return module.inner if isinstance(module, StabilizedUpdateMLP) else module


def _context(value):
    if value is None:
        return None
    # Keep only the physical state inputs: atom count, total charge, multiplicity.
    n = value['n_atoms']
    q = float(value.get('total_charge', 0.))
    m = value.get('multiplicity', 1)
    if not math.isfinite(float(n)) or int(n) != n or n < 1:
        raise ValueError('Global atom count must be a positive integer')
    if not math.isfinite(q):
        raise ValueError('Global total charge must be finite')
    if not math.isfinite(float(m)) or int(m) != m or m < 1:
        raise ValueError('Global multiplicity must be a positive integer')
    return dict(n_atoms=int(n), total_charge=q, multiplicity=int(m))


def _validate(model):
    if getattr(model, '_xl_capture_base_bound', False):
        raise ValueError('Base execution is already bound')
    if (getattr(model, 'compiled_nodes', False) or getattr(model, 'fast_nodes', False)
            or getattr(model, 'joined_nodes', False) or getattr(model, 'frozen_local_features', False)):
        raise ValueError('Pass an unoptimized model before node compilation')
    if getattr(model, 'radial_tables', None) is not None:
        raise ValueError('Pass a model without precompiled radial tables')
    if getattr(model, 'numerical_execution', {}).get('policy') != POLICY:
        raise ValueError('Centered FP64-internal LayerNorm policy is required')
    if not isinstance(model, L2Potential):
        raise ValueError('The captured EFM base adapter requires an L2Potential model')
    cfg = model.config
    if cfg.tensor_mixing_rank is not None:
        raise ValueError('Low-rank tensor mixing is not supported')
    if cfg.atomic_refinements or any(len(group) for group in model.refinements):
        raise ValueError('Atomic refinements require an explicit capture port')
    if tuple(cfg.scalar_source_blocks):
        raise ValueError('Later scalar-source blocks are not supported')
    if cfg.radial_conditioning != 'shared' or cfg.expert_routing != 'element':
        raise ValueError('Shared radial filters and element experts are required')
    if (not model.blocks or len(model.layer_cutoffs) != len(model.blocks)
            or len(model.blocks) != cfg.interactions):
        raise ValueError('One layer cutoff per block is required')
    if any(not 0. < float(r) <= float(model.cutoff) for r in model.layer_cutoffs):
        raise ValueError('Layer cutoffs must lie within the model cutoff')
    numbers = tuple(model.atomic_numbers)
    if not numbers or len(set(numbers)) != len(numbers) or model.n_elements != len(numbers):
        raise ValueError('Distinct nonempty model species are required')
    if model.state_projection.weight.shape != (cfg.hidden, 2 if model.condition_spin else 1):
        raise ValueError('Unexpected state projection width')
    params = tuple(model.parameters())
    if not params or any(p.dtype != torch.float32 for p in params):
        raise ValueError('All parameters must be FP32')
    if len({p.device for p in params}) != 1:
        raise ValueError('Place all parameters on one device before binding')
    # ``backbone`` selects factory defaults; ``custom`` constructs these same
    # L2Interaction blocks. Gate the message layout consumed by the kernels,
    # not the name used to obtain the widths.
    hidden, tensor = cfg.hidden, cfg.tensor_hidden or cfg.hidden
    edge = cfg.edge_hidden or cfg.hidden
    if model.hidden != hidden or model.tensor_hidden != tensor or model.cutoff != cfg.cutoff:
        raise ValueError('Model dimensions differ from the declared L2 structure')
    if len(model.readouts) != len(model.blocks) or len(model.readout_norms) != len(model.blocks):
        raise ValueError('One scalar readout per interaction block is required')
    for index, block in enumerate(model.blocks):
        if not isinstance(block, L2Interaction):
            raise ValueError('Every base block must be an L2Interaction')
        if (block.scalar_hidden, block.tensor_hidden, block.message_hidden) != (hidden, tensor, edge):
            raise ValueError('Interaction widths differ from the model L2 structure')
        if (block.input_is_scalar != (index == 0) or block.cross_order != (index > 0)
                or block.terminal != (index == len(model.blocks) - 1)):
            raise ValueError('Unsupported scalar-first, cross-order or terminal message layout')
        if (block.tensor_mixing_rank is not None
                or block.bottleneck != (edge != hidden)
                or block.tensor_bottleneck != (edge != tensor)
                or block.narrow_tensors != (tensor != hidden)):
            raise ValueError('Unsupported L2 projection structure')
        if (block.cutoff != float(model.layer_cutoffs[index])
                or block.switch_width != float(cfg.switch_width)):
            raise ValueError('Block cutoff or switch differs from the model geometry')
        message_channels = (3 if index == 0 else 7) * edge
        if (block.scalar_message[-1].weight.shape[0] != message_channels
                or block.radial_filter[-1].weight.shape[0] != message_channels):
            raise ValueError('Scalar and radial messages must use the same L2 channel layout')
        spec = block.__dict__.get('_efm_block_spec', {})
        if any(spec.get(key, False) for key in ('angular', 'receiver_filter', 'edge_mlp')):
            raise ValueError('Angular, receiver-filter and edge-MLP branches are not ported')
        if any(getattr(block, key, None) is not None for key in ('angular', 'receiver_filter', 'edge_mlp')):
            raise ValueError('Additional message branches are not ported')
        if (spec.get('aggregate_normalization') is not None
                or getattr(block, 'stab_aggregate_normalization', None) is not None):
            raise ValueError('Aggregate normalization requires an explicit CSR port')
        if float(block.message_scale) != 0.25:
            raise ValueError('The fused base messages require message_scale=0.25')
        if getattr(block, 'compiled_nodes', False):
            raise ValueError('Install the base before block compilation')
    for index in cfg.expert_blocks:
        module = _inner(model.blocks[index].update_mlp)
        if not isinstance(module, ElementExpertMLP) or len(module.experts) != model.n_elements:
            raise ValueError('Expected one element expert per species inside any stabilizer')
    return params[0].device


def _convert_nodes(module, *, precision, frozen_layernorm, layernorm_atom_tile, excluded):
    """Use the existing replacements while keeping EFM descendants untouched."""
    for name, child in list(module.named_children()):
        if id(child) in excluded:
            continue
        if precision != 'fp32' and isinstance(child, (nn.Linear, _RowSelectedLinear)):
            if isinstance(child, _RowSelectedLinear):
                child.in_features = child.weight.shape[1]
                child.out_features = child.weight.shape[0]
            replacement = FastLinear(child, precision)
            replacement.strided_inputs = True
            setattr(module, name, replacement)
        elif frozen_layernorm and isinstance(child, nn.LayerNorm):
            setattr(module, name, FrozenLayerNorm(child, layernorm_atom_tile))
        else:
            _convert_nodes(child, precision=precision, frozen_layernorm=frozen_layernorm,
                           layernorm_atom_tile=layernorm_atom_tile, excluded=excluded)
    if precision != 'fp32' and isinstance(module, nn.Sequential):
        for index in range(len(module) - 1):
            if isinstance(module[index], FastLinear) and isinstance(module[index + 1], nn.SiLU):
                module[index].silu = True
                module[index + 1] = nn.Identity()


def bind_base(model, *, global_context=None, reserve=1.25, precision='fp32',
              radial_intervals=200, message_channel_tile=16, joined_projections=False,
              fold_first=False, frozen_layernorm=False, fused_features=False,
              grouped_experts=False, layernorm_atom_tile=8, feature_atom_tile=None):
    """Prepare and return the same frozen model, ready for EFM installation.

    Default node operations keep the native operation order, including each
    stabilizer's ``transform_input -> inner -> output * gate`` sequence.
    Radial tables are an approximation and need a separate energy/force guard.
    Joined/folded products and Triton options are explicit numerical variants;
    they preserve learned tensors but may change floating-point operation order.
    CUDA-only options may be installed on CPU for structural inspection, but
    their forward requires CUDA FP32. Bind again only on a fresh model copy.
    """
    if precision not in PRECISIONS:
        raise ValueError('Unsupported node precision')
    if grouped_experts and precision == 'fp32':
        raise ValueError('Grouped experts require tf32x3 or fp16x3')
    if fold_first and not joined_projections:
        raise ValueError('First-block folding requires joined projections')
    if message_channel_tile not in MESSAGE_CHANNEL_TILES:
        raise ValueError('Unsupported message channel tile')
    if layernorm_atom_tile not in ATOM_TILES:
        raise ValueError('Unsupported LayerNorm atom tile')
    if feature_atom_tile is None:
        feature_atom_tile = 4 if max(model.config.hidden, model.tensor_hidden) <= 128 else 1
    if feature_atom_tile not in ATOM_TILES:
        raise ValueError('Unsupported local-feature atom tile')
    if type(radial_intervals) is not int or radial_intervals < 2:
        raise ValueError('At least two radial intervals are required')
    reserve = float(reserve)
    if not math.isfinite(reserve) or reserve < 1.:
        raise ValueError('Neighbor reserve must be finite and at least one')
    context = _context(global_context)
    _validate(model)
    if context is not None and not model.condition_spin and context['multiplicity'] != 1:
        raise ValueError('A charge-only model has no nonsinglet response')

    model.eval().requires_grad_(False)
    original_parameters = tuple(model.named_parameters())
    original_state = model.state_dict(keep_vars=True)
    original_versions = {name: value._version for name, value in original_state.items()}
    wrappers = {i: block.update_mlp for i, block in enumerate(model.blocks)
                if isinstance(block.update_mlp, StabilizedUpdateMLP)}
    efms = {id(block.efm) for block in model.blocks if getattr(block, 'efm', None) is not None}

    # Compile before replacing any radial SiLU by a fused linear activation.
    model.compile_radial_tables(radial_intervals, order=5)
    for table in model.radial_tables:
        # These are derived execution caches, not new checkpoint state.
        table._non_persistent_buffers_set.add('coefficients')
    if len(model.radial_tables) != len(model.blocks) + 1:
        raise ValueError('One radial table per block plus the pair table is required')
    for block, table, cutoff in zip(model.blocks, model.radial_tables, model.layer_cutoffs):
        channels = (3 if block.input_is_scalar else 7) * block.message_hidden
        if table.cutoff != float(cutoff) or table.coefficients.shape[2] != channels:
            raise ValueError('Radial table does not match its block')
    if model.radial_tables[-1].coefficients.shape[2] != model.n_elements ** 2:
        raise ValueError('Pair table must cover the configured species pairs')

    if joined_projections:
        enable_joined_nodes(model, fold_first=fold_first)
        # Replace the helper's embedding-only cache with Q/M-aware preparation.
        delattr(model, 'frozen_first_messages')
    _convert_nodes(model, precision=precision, frozen_layernorm=frozen_layernorm,
                   layernorm_atom_tile=layernorm_atom_tile, excluded=efms)
    if precision != 'fp32':
        for module in model.modules():
            if isinstance(module, JoinedLinear):
                module.fast, module.precision = True, precision
        model.fast_nodes = True
        model.fast_node_precision = precision
        model.fast_strided_inputs = True
        model.fused_node_activations = True
    if fused_features:
        enable_frozen_local_features(model, atom_tile=feature_atom_tile)
    if grouped_experts:
        for index in model.config.expert_blocks:
            wrapper = model.blocks[index].update_mlp
            replacement = FrozenElementExpertMLP(_inner(wrapper), precision)
            if isinstance(wrapper, StabilizedUpdateMLP):
                wrapper.inner = replacement
            else:
                model.blocks[index].update_mlp = replacement
        model.sparse_expert_nodes = True
        model.grouped_expert_precision = precision

    model.reference_chunk_size = 0
    model.fused_messages = model.fused = True
    model.fused_message_settings = (int(message_channel_tile), 8, 4, True)
    model.filter_skin_neighbors = True
    configure_neighbor_reserve(model, reserve)
    # Includes prepare_energy -> PreparedGeneralNeighbors, with batch-owned
    # species schedules and first-message caches conditioned on global Q/M.
    native_moderato.bind(model, global_context=context)

    actual_parameters = tuple(model.named_parameters())
    if (len(actual_parameters) != len(original_parameters)
            or any(a != b or p is not q for (a, p), (b, q) in zip(original_parameters, actual_parameters))):
        raise RuntimeError('Base adaptation changed parameter names or objects')
    actual_state = model.state_dict(keep_vars=True)
    if list(actual_state) != list(original_state):
        raise RuntimeError('Base adaptation changed the persistent tensor namespace')
    if any(actual_state[name] is not value or value._version != original_versions[name]
           for name, value in original_state.items()):
        raise RuntimeError('Base adaptation replaced or mutated an original tensor')
    if any(model.blocks[index].update_mlp is not wrapper for index, wrapper in wrappers.items()):
        raise RuntimeError('Base adaptation replaced a stabilizer wrapper')
    model._xl_capture_base_bound = True
    model.inference_backend = 'captured_efm_base'
    model.deployment_metadata = dict(
        implementation=__name__, inference_backend=model.inference_backend,
        global_context=context, precision=precision, compiled_nodes=False,
        joined_projections=bool(joined_projections), folded_first_projections=bool(fold_first),
        frozen_layernorm=bool(frozen_layernorm), fused_local_features=bool(fused_features),
        grouped_experts=bool(grouped_experts), stabilizer_wrappers=len(wrappers),
        original_parameter_objects_and_namespace_verified=True,
        original_tensor_versions_verified=True, learned_weights_changed=False,
        radial_table=dict(order=5, intervals=radial_intervals, persistent=False,
                          requires_independent_analytic_parity_gate=True),
        first_message_cache='batch_owned_global_Q_over_N_and_spin_over_N',
        fused_message_settings=list(model.fused_message_settings),
        requires_efm_install=bool(efms), reference_requires_separate_model=True)
    return model
