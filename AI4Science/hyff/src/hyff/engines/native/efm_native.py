"""Width-general native execution of l4a0 L2 models (any hidden/tensor/edge width and block count).

The moderato engine (hyff.engines.native.moderato) fixes the moderato widths (hidden 128,
tensor 64, edge 16, four blocks at 3.5/3.5/6/8 A). This module runs the same node path
(joined/folded projections, batch-owned state-conditioned first-message cache, FP64-internal
frozen LayerNorm, fused local invariant features, grouped element experts, compiled node graphs,
CSR/Triton fused messages with quintic radial tables, TF32x3 tensor-core products on unchanged
FP32 weights) for any l4a0 StatePotential model and exposes the width-dependent execution
choices. The presto engine reuses its weight reading and architecture checks.

Width dependence. Every kernel is written with constexpr widths and masked tiles; the
width-specific places are guards, tuning defaults and tile sizes:

  exposed here
    fused_message_settings (16, 8, 4, True) -> `message_channel_tile` (programs per atom = ceil(edge/tile))
    local-feature atom tile 4 (measured at S=128, T=64; BC = next_pow2(max(S, T)))
                                            -> `feature_atom_tile`, default 4 up to 128 channels, 1 above
  width-general by construction (masked constexpr tiles), reused unchanged
    hyff.kernels.layernorm_fp64 (H constexpr, BH = next_pow2(H), rows tiled by atom_tile)
    hyff.kernels.local_features_fp64 (S, T constexpr, HIGHER flag)
    hyff.kernels.fused_linear._mm/_mm_grouped/_mm_silu (N, K constexpr; BM 64, BN 32|64, BK 32 for tf32x3)
    hyff.kernels.grouped_experts._grouped_product (K, N constexpr; BE = next_pow2(n_elements); row tiles of 64)
    hyff.kernels.fused_messages._forward/_backward/_backward_scalar (C constexpr, channel tiles of BC)
    hyff.kernels.joined_linear.enable_joined_nodes(fold_first=True) (folded first projections [3T, E])
    radial tables (one per block, 3C or 7C channels; pair table n_elements^2 rows), pair kernel (NE = n_elements)
    hyff.engines.native.frozen_nodes_forward (loops over blocks; readout scale 1/sqrt(blocks))
  tuned only for edge width 16 (other widths use the general kernels above)
    hyff.kernels.message_tiles._forward_atoms and hyff.kernels.tensor_vjp._backward_tensor are dispatched
    only for C == 16 with settings (16, 8, 4, True).
  not supported
    the angular-moment branch (hyff.models.angular_branch.build_angular_model) adds l=3/l=4 edge moments
    whose block forward refuses csr execution; it is rejected with an explicit message.

Fallback switches (`frozen_layernorm`, `fused_features`, `grouped_experts`) return the corresponding
component to its ordinary torch execution while everything else stays native. No learned tensor
is changed in any variant; `load` verifies every weight tensor bytewise.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from hyff.engines.native import base_adapter as base
from hyff.engines.native import frozen_nodes
from hyff.kernels.layernorm_fp64 import FrozenLayerNorm, enable_frozen_layernorm
from hyff.kernels.local_features_fp64 import enable_frozen_local_features
from hyff.kernels.grouped_experts import FrozenElementExpertMLP, enable_grouped_experts
from hyff.engines.native import moderato
from hyff.engines import reference
from hyff.models.element_model import ElementExpertMLP
from hyff.models.base_model import POLICY as LAYERNORM_POLICY
from hyff.data import collate, physical
from hyff.engines.force_engine import ForceEngine
from hyff.kernels.fused_linear import enable_fast_nodes
from hyff.kernels.joined_linear import JoinedLinear, enable_joined_nodes
from hyff.weights import read_adapter_export

PRECISIONS = moderato.PRECISIONS
IMPLEMENTATION = 'hyff.engines.native.efm_native'
MODERATO_BACKBONE = moderato.EXPECTED_BACKBONE
MESSAGE_CHANNEL_TILES = (8, 16, 32, 64)
ATOM_TILES = (1, 4, 8, 16)
C16_MESSAGE_SETTINGS = (16, 8, 4, True)
EAGER_ONLY_FACTORIES = {
    'hyff.models.angular_branch:build_angular_model':
        'Angular-moment branch: per-block l=3/l=4 angular moments; its block forward refuses csr execution '
        'and no fused message kernel evaluates the harmonics. Use the reference engine.',
    'hyff.models.angular_branch:build_film_model':
        'State FiLM branch: per-atom state FiLM of the block LayerNorms; the compiled node graphs do not '
        'bind the state features. Use the reference engine.',
}

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


def read_export(path):
    """`hyff.engines.reference.read_export` with an explicit message for constructors without a native path."""
    saved = read_adapter_export(path)
    factory = saved.get('model_factory')
    if factory in EAGER_ONLY_FACTORIES:
        raise ValueError(f'Constructor {factory!r} has no native engine. {EAGER_ONLY_FACTORIES[factory]}')
    missing = [key for key in reference.EXPORT_KEYS if key not in saved]
    if missing:
        raise KeyError(f'Not a HyFF weight export; missing {missing}')
    if factory not in reference.FACTORIES:
        raise ValueError(f'Unsupported model factory {factory!r}; expected one of {sorted(reference.FACTORIES)}')
    return saved


def check_export(saved):
    """Accept any l4a0 StatePotential export; record the widths the execution depends on.

    Required: the L2Interaction/StatePotential constructor and the FP64-internal LayerNorm policy.
    Rejected because untested on the native path, not because a kernel forbids them: atom-only
    refinements, low-rank tensor mixing and later scalar-source blocks (the first-block folding
    assumes block 0 is the only scalar source).
    """
    if reference.family(saved) != 'large':
        raise ValueError('This engine runs L2Interaction models only')
    config = saved['model_config']
    policy = saved.get('numerical_execution') or {}
    if policy.get('policy') != LAYERNORM_POLICY:
        raise ValueError(f'Export records an unknown LayerNorm policy: {policy!r}')
    if config.get('backbone', 'm26') != 'l4a0':
        raise ValueError(f'Expected the l4a0 (L2Interaction) backbone, got {config.get("backbone")!r}')
    if config.get('atomic_refinements', 0):
        raise ValueError('Atom-only refinements are not part of the native execution')
    if config.get('tensor_mixing_rank') is not None:
        raise ValueError('Low-rank tensor mixing is not part of the native execution')
    if tuple(config.get('scalar_source_blocks', ())):
        raise ValueError('Later scalar-source blocks are not part of the native execution (first-block folding)')
    numbers = tuple(config['atomic_numbers'])
    if len(numbers) != len(set(numbers)) or not numbers:
        raise ValueError('Distinct nonempty species list required')
    hidden = int(config['hidden'])
    tensor = int(config.get('tensor_hidden') or hidden)
    edge = int(config.get('edge_hidden') or hidden)
    interactions = int(config['interactions'])
    cutoff = float(config['cutoff'])
    cutoffs = tuple(float(r) for r in (config.get('layer_cutoffs') or (cutoff,) * interactions))
    if len(cutoffs) != interactions or any(not 0 < r <= cutoff for r in cutoffs):
        raise ValueError('One layer cutoff per block, each within the model cutoff, is required')
    expert_blocks = tuple(config.get('expert_blocks') or range(interactions))
    if config.get('radial_conditioning', 'shared') != 'shared' or config.get('expert_routing', 'element') != 'element':
        raise ValueError('Element experts with shared radial filters are required')
    backbone = dict(backbone='l4a0', hidden=hidden, tensor_hidden=tensor, edge_hidden=edge, interactions=interactions,
                    layer_cutoffs=cutoffs)
    return dict(atomic_numbers=numbers, n_elements=len(numbers), condition_spin=bool(config.get('condition_spin', False)),
                hidden=hidden, tensor_hidden=tensor, edge_hidden=edge, interactions=interactions, cutoff=cutoff,
                layer_cutoffs=cutoffs, n_rbf=int(config.get('n_rbf', 32)), mlp_hidden=config.get('mlp_hidden'),
                expert_hidden=int(config.get('expert_hidden', 128)), expert_blocks=expert_blocks,
                radial_hidden=config.get('radial_hidden'), higher_invariants=bool(config.get('higher_invariants', False)),
                switch_width=float(config.get('switch_width', 1.)),
                tier='moderato' if backbone == MODERATO_BACKBONE else 'wide', moderato_backbone=backbone == MODERATO_BACKBONE,
                architecture=saved.get('architecture'), selection_kind=saved.get('selection_kind'),
                speed_only_artifact=saved.get('selection_kind') == 'seeded_init_for_speed_timing')


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
            message_forward_kernel='l2_forward_tiles._forward_atoms (8 atoms/program, tuned C=16)' if tuned
                                   else 'l2_fused_messages._forward (1 atom x channel tile/program, general)',
            message_backward_kernel=('l2_fused_messages._backward_scalar (general)' if first else
                                     'l2_tensor_vjp._backward_tensor (tuned C=16)' if tuned else 'l2_fused_messages._backward (general)'),
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
        fused_message_settings=list(settings), message_kernels='tuned_c16_atom_tiled' if tuned else 'general_channel_tiled',
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
         message_channel_tile=16, frozen_layernorm=True, fused_features=True, grouped_experts=True):
    if precision not in PRECISIONS:
        raise ValueError(f'precision must be one of {PRECISIONS}')
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
    model, saved = base.load(path, device, global_context=global_context, owned_weights=owned_weights,
                             reserve=reserve, node_mode='torch')
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
        enable_frozen_layernorm(model, atom_tile=layernorm_atom_tile)
    if fused_features:
        enable_frozen_local_features(model, atom_tile=feature_atom_tile)
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
    model.inference_backend = 'width_general_native'
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
        export=dict(architecture=saved['architecture'], model_factory=saved['model_factory'],
                    selection_kind=saved.get('selection_kind'), step=saved.get('step'), seed=saved.get('seed'),
                    frame_epochs=saved.get('frame_epochs'), energy_output=saved.get('energy_output'),
                    numerical_execution=saved.get('numerical_execution'), source_model_config=saved.get('source_model_config')))
    return model, saved


LOAD_OPTIONS = ('reserve', 'precision', 'compiled_nodes', 'layernorm_atom_tile', 'feature_atom_tile',
                'message_channel_tile', 'frozen_layernorm', 'fused_features', 'grouped_experts')


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
    """Create the engine for one system; same call shape as moderato.create_full_engine plus width options."""
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    n = len(numbers)
    context = dict(n_atoms=n, total_charge=float(total_charge), multiplicity=int(multiplicity),
                   state_features=state_features(n, total_charge, multiplicity))
    record = dict(positions=np.asarray(positions, dtype=np.float32), atomic_numbers=numbers, total_charge=float(total_charge))
    if cell is not None:
        record.update(cell=np.asarray(cell, dtype=np.float32), pbc=np.ones(3, dtype=bool))
    made = create(record, checkpoint=str(resolve_checkpoint(checkpoint)), global_context=context, device=device,
                  skin=skin, capture_branch_policy=capture_branch_policy, **options)
    made['metadata'] = dict(made['metadata'], backend='native_nodes_csr_fused_messages',
                            global_context=context, skin_A=float(skin))
    return made


# ---------------------------------------------------------------------------------------------
# CPU-executable surrogate of the native path (ordinary torch operators; no Triton).

def load_torch_surrogate(checkpoint, device='cpu', *, radial_tables=False, global_context=None):
    """The native path's torch-executable subset for CPU checks (as in hyff.engines.native.moderato).

    Export build (FP64-internal LayerNorm policy), joined + folded first projections
    (`enable_joined_nodes(fold_first=True)`), the discarded embedding-only lookup, the batch-owned
    state-conditioned initial scalar and first messages (`_prepare`), the species schedule bound to
    the plain (ungrouped) element experts, and the block forward of hyff.engines.native.moderato_forward
    (general pair stride). With
    `radial_tables` the quintic 200-interval tables of every layer cutoff replace the analytic
    radial filters through the memory-bounded chunked reference. Triton components (TF32x3
    products, frozen LayerNorm, fused features, grouped experts, CSR messages, pair kernel) are
    not exercised.
    """
    path = resolve_checkpoint(checkpoint)
    saved = read_export(path)
    widths = check_export(saved)
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
                                    triton_components_exercised=False, tier=widths['tier'], architecture=widths['architecture'])
    return model, saved


__all__ = ['load', 'create', 'factory', 'create_full_engine', 'load_torch_surrogate', 'bind', 'check_export',
           'execution_geometry', 'default_feature_atom_tile', 'resolve_checkpoint', 'read_export', 'numerics_label',
           'dependency_radius', 'state_features', 'system_batch', 'digest', 'PRECISIONS', 'IMPLEMENTATION',
           'MESSAGE_CHANNEL_TILES', 'ATOM_TILES', 'LOAD_OPTIONS', '_prepare', '_global_state']
