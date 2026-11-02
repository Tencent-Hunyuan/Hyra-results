"""Native presto engine: L2Interaction blocks with edge width C = 8.

One force evaluation runs inside one CUDA graph:

  * radius views of the CSR skin matrices at the current coordinates (``_select_two``: the 3.5 A and
    4.5 A views in one launch; a three-block model shares its 3.5 A view between blocks 0 and 1);
  * block 0 (scalar source, 8 s + 8 v + 8 t): ``FirstMessages8``, the eight-lane launch of the
    atom-tiled first-block kernels. Its node message depends only on species and the global state,
    so it is a constant per-atom tensor prepared outside the graph (no head launch, no source
    adjoint in the backward);
  * blocks 1.. (full L2Interaction, seven paths at edge width 8): fused head (norm -> Linear/SiLU ->
    Linear, [N, 56]), ``edge_vector_in`` / ``edge_tensor_in`` projections fused into the previous
    block's compiled closure, ``L2Messages8`` (eight-lane launch of ``_forward_atoms`` /
    ``_backward_tensor``, which ``hyff.kernels.fused_messages.messages`` dispatches only for C = 16);
  * node updates: joined and folded projections (``enable_joined_nodes(fold_first=True)``), fused
    local features (FP64 or FP32 statistics), grouped element experts (``FrozenElementExperts``),
    all inside one ``torch.compile``d closure per block; readouts as fused heads;
  * the symmetric pair term (NE = number of elements) on the 4.5 A view.

Numerics (``precision``, ``layernorm_statistics``):
  tf32x3 / fp64   TF32x3 tensor-core node products, FP64 LayerNorm statistics;
  tf32x3 / fp32   centred two-pass FP32 LayerNorm statistics in the heads and local features
                  (the presto default);
  fp32            IEEE products, for comparison.
No learned tensor is changed; the prepared state is verified bytewise against the weights.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hyff.engines.native.base_adapter import PreparedGeneralNeighbors
from hyff.engines.native.fast_csr import _edge_geometry, _is_affine
from hyff.kernels.fused_element_experts import schedule
from hyff.engines.native.moderato import _global_state
from hyff.engines import reference
from hyff.kernels.pair_energy_species import pair_energy
from hyff.engines.native.radius_variants import refresh_two
from hyff.engines.native import efm_native
from hyff.engines.force_engine import ForceEngine
from hyff.kernels.fused_messages import configure_neighbor_reserve
from hyff.kernels.joined_linear import JoinedLinear, enable_joined_nodes
from hyff.models.l2_potential import L2Interaction, smooth_switch
from hyff.kernels.radial_table import QuinticRadialTable

from hyff.engines.native.presto_messages import (
    C8, PRECISIONS, FirstMessages8, FrozenElementExperts, L2Messages8, check_layout, fp16_representable,
)
from hyff.engines.native.presto_nodes import STATISTICS, DualHead, FusedHead, enable_local_features

for _name in ('recompile_limit', 'cache_size_limit'):
    if hasattr(torch._dynamo.config, _name):
        setattr(torch._dynamo.config, _name, 256)

IMPLEMENTATION = 'hyff.engines.native.presto'
BACKEND = 'presto_csr_c8_messages_fused_nodes'
# (atoms per program, neighbor slots per iteration, warps), measured inside the captured graph on a water box:
# the C = 16 layout (8, 1, 4) costs 240 / 341 us forward / backward at C = 8, (1, 4, 1) costs 120 / 179 us.
DEFAULT_LAYOUTS = dict(first_forward=(8, 1, 1), first_backward=(4, 4, 2), l2_forward=(1, 4, 1), l2_backward=(1, 4, 1))
SETTINGS = (8, 8, 4, True)      # consumed by the pair kernel (fma flag) and recorded in metadata only
digest = reference.digest
state_features = reference.state_features
system_batch = reference.system_batch
read_export = efm_native.read_export
check_export = efm_native.check_export


def resolve_checkpoint(checkpoint):
    """Absolute path of a weight file."""
    return Path(checkpoint).resolve()


def check_layouts(layouts):
    result = dict(DEFAULT_LAYOUTS)
    result.update(layouts or {})
    for name in result:
        if name not in DEFAULT_LAYOUTS:
            raise ValueError(f'Unknown layout {name}')
        result[name] = check_layout(result[name])
    return result


def check_blocks(model):
    """Every block must have the L2Interaction structure that the C = 8 kernels assume."""
    blocks = list(model.blocks)
    if len(blocks) < 2 or len(model.refinements):
        raise ValueError('Two or more L2Interaction blocks without atom-only refinements expected')
    for index, block in enumerate(blocks):
        if not isinstance(block, L2Interaction) or type(block).forward is not L2Interaction.forward or 'forward' in block.__dict__:
            raise TypeError(f'Block {index} must execute the unchanged L2Interaction forward')
        first, last = index == 0, index == len(blocks) - 1
        if block.message_hidden != C8:
            raise ValueError('Edge width 8 expected')
        if block.input_is_scalar != first or block.cross_order == first:
            raise ValueError('Block 0 must be the only scalar-source block; later blocks carry the seven cross-order paths')
        if block.terminal != last or block.tensor_mixing_rank is not None:
            raise ValueError('Only the last block may be terminal; low-rank mixing is not supported')
        if not (block.narrow_tensors and block.bottleneck and block.tensor_bottleneck):
            raise ValueError('The path assumes tensor_hidden != hidden and edge width 8 below both')
        if float(block.cutoff) != float(model.layer_cutoffs[index]) or block.switch_width != model.config.switch_width:
            raise ValueError('Block cutoffs / switch widths must equal the model configuration')
        if block.message_scale != .25:
            raise ValueError('The fused kernels fold message_scale = 0.25')
        paths = 3 if first else 7
        for name, network in (('radial_filter', block.radial_filter), ('scalar_message', block.scalar_message)):
            if (not isinstance(network, nn.Sequential) or len(network) != 3 or not _is_affine(network[0])
                    or not isinstance(network[1], nn.SiLU) or not _is_affine(network[2])):
                raise TypeError(f'Block {index} {name} must be affine/SiLU/affine')
            if int(network[2].weight.shape[0]) != paths * C8:
                raise ValueError(f'Block {index} {name} width differs from the {paths} x 8 layout')
        if not first and not (hasattr(block, 'edge_vector_in') and hasattr(block, 'edge_tensor_in')):
            raise ValueError('Later blocks must carry the edge_vector_in / edge_tensor_in projections')
    if float(blocks[-1].cutoff) != float(model.cutoff):
        raise ValueError('The last block must use the model cutoff (skin view filter)')
    return blocks


def compile_tables(model, intervals=200):
    """One quintic table per block (3 x 8 or 7 x 8 channels at the block cutoff) plus the NE^2-row pair table."""
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze the model before compiling radial tables')
    parameter = next(model.parameters())
    tables = [QuinticRadialTable.compile(model.rbf, block.radial_filter, intervals, table_cutoff=block.cutoff) for block in model.blocks]
    pair_linear = nn.Linear(model.config.n_rbf, model.n_elements ** 2, bias=False, device='meta')
    pair_linear.weight = nn.Parameter(model.pair_coeff.weight.detach(), requires_grad=False)
    if pair_linear.weight.shape != (model.n_elements ** 2, model.config.n_rbf):
        raise ValueError('Pair coefficient table shape differs from n_elements^2 x n_rbf')
    pair = QuinticRadialTable.compile(model.rbf, [pair_linear], intervals)
    if float(pair.cutoff) != float(model.cutoff):
        raise ValueError('The pair table domain must be the model cutoff')
    tables = nn.ModuleList([*tables, pair]).to(parameter)
    for index, block in enumerate(model.blocks):
        expected = (intervals, 6, (3 if block.input_is_scalar else 7) * C8)
        if tuple(tables[index].coefficients.shape) != expected or float(tables[index].cutoff) != float(block.cutoff):
            raise ValueError(f'Block {index} table layout differs from the kernel layout {expected}')
    model.radial_tables = tables
    model.radial_table_order = 5
    return tables


# ---------------------------------------------------------------------------------------------
# Node preparation (derived buffers only; parameters untouched).

def prepare_nodes(model, precision='tf32x3', statistics='fp64', feature_atom_tile=16, head_layout=(16, 2), expert_block=64, joined_fast=False,
                  dual_heads=True):
    if precision not in PRECISIONS or statistics not in STATISTICS:
        raise ValueError(f'precision must be one of {PRECISIONS} and statistics one of {STATISTICS}')
    if any(p.requires_grad for p in model.parameters()) or getattr(model, 'compiled_nodes', False):
        raise ValueError('Freeze the model and prepare nodes before compilation')
    if joined_fast:
        if precision == 'fp32':
            raise ValueError('joined_fast selects the Triton node products (tf32x3 / fp16x3 / fp16w2)')
        model.fast_nodes, model.fast_node_precision = True, precision
    enable_joined_nodes(model, fold_first=True)
    delattr(model, 'frozen_first_messages')          # embedding-only lookup; the state-conditioned message is prepared per batch
    first = model.blocks[0]
    for name in ('folded_first_vector', 'folded_first_tensor'):
        if not isinstance(getattr(first, name, None), JoinedLinear):
            raise RuntimeError(f'Block 0 lacks the folded projection {name}')
    if tuple(first.folded_first_vector.weight.shape) != (3 * model.tensor_hidden, C8):
        raise RuntimeError('Folded first vector projection has an unexpected shape')
    for block in model.blocks:
        for name in ('joined_vector', 'joined_tensor', 'joined_invariants'):
            if not isinstance(getattr(block, name, None), JoinedLinear):
                raise RuntimeError(f'A block lacks the joined projection {name}')
    enable_local_features(model, feature_atom_tile, statistics)
    rows, warps = head_layout
    heads = {}
    for index, block in enumerate(model.blocks):
        if index:
            heads[f'message{index}'] = FusedHead(block.scalar_norm, block.scalar_message[0], block.scalar_message[2], rows, warps, precision, statistics)
        readout = model.readouts[index]
        if not (isinstance(readout, nn.Sequential) and len(readout) == 3 and _is_affine(readout[0]) and isinstance(readout[1], nn.SiLU) and _is_affine(readout[2])):
            raise TypeError('Readouts must be affine/SiLU/affine')
        heads[f'readout{index}'] = FusedHead(model.readout_norms[index], readout[0], readout[2], rows, warps, precision, statistics)
        block.update_mlp = FrozenElementExperts(block.update_mlp, precision)
    if dual_heads:
        # readout of block i and message head of block i+1 read the same scalar: one launch, shared statistics.
        for index in range(len(model.blocks) - 1):
            heads[f'dual{index}'] = DualHead(heads[f'readout{index}'], heads[f'message{index + 1}'])
    hidden, tensor = model.config.hidden, model.tensor_hidden
    for index, block in enumerate(model.blocks):
        inputs = hidden + (5 if block.higher_invariants else 2) * tensor
        outputs = hidden + (2 if block.terminal else 4) * tensor
        if block.update_mlp.weight1.shape != (model.n_elements, model.config.expert_hidden, inputs) or block.update_mlp.weight2.shape != (model.n_elements, outputs, model.config.expert_hidden):
            raise RuntimeError(f'Block {index} expert bank shapes differ from the configuration')
    model._engine_heads = heads
    model._engine_dual_heads = bool(dual_heads)
    model._engine_precision = precision
    model._engine_statistics = statistics
    model._engine_expert_block = int(expert_block)
    model._engine_joined_fast = bool(joined_fast)
    model.sparse_expert_nodes = True
    return model


def _closure(block, next_block):
    """Node update of `block` (+ the next block's edge-width source projections) as one compiled function."""
    finish = block._finish_messages
    if next_block is None:
        def run(scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t):
            return finish(scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t)
    else:
        def run(scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t):
            scalar, vector, tensor = finish(scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t)
            return scalar, vector, tensor, next_block.edge_vector_in(vector), next_block.edge_tensor_in(tensor)
    return run


INDUCTOR_GEMM_MODES = ('aten', 'triton', 'autotune')


def configure_inductor_gemm(mode):
    """Process-wide inductor GEMM choice for the compiled closures: cuBLAS (`aten`), Triton templates only
    (`triton`, epilogue fusion of the small node projections), or autotuned among both (`autotune`)."""
    if mode not in INDUCTOR_GEMM_MODES:
        raise ValueError(f'inductor_gemm must be one of {INDUCTOR_GEMM_MODES}')
    config = torch._inductor.config
    if mode == 'aten':
        config.max_autotune_gemm = False
        config.max_autotune_gemm_backends = 'ATEN,TRITON,CPP'
    else:
        config.max_autotune_gemm = True
        config.max_autotune_gemm_backends = 'TRITON' if mode == 'triton' else 'ATEN,TRITON'
    return mode


def install_closures(model, compiled=True, inductor_gemm='aten'):
    configure_inductor_gemm(inductor_gemm)
    model._engine_inductor_gemm = inductor_gemm
    blocks = list(model.blocks)
    closures = []
    for index, block in enumerate(blocks):
        run = _closure(block, blocks[index + 1] if index + 1 < len(blocks) else None)
        if compiled:
            run = torch.compile(run, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
        closures.append(run)
    model._engine_closures = closures
    model.compiled_nodes = bool(compiled)
    return closures


# ---------------------------------------------------------------------------------------------
# Batch preparation outside every CUDA graph.

@torch.no_grad()
def _prepare(self, batch):
    if '_engine_partition' in batch:
        return batch
    if len(batch['ptr']) != 2:
        raise ValueError('The engine forward represents one physical system')
    completed = dict(reference.complete_state_batch(self, batch))
    numbers = completed['atomic_numbers']
    features = _global_state(self, completed)                          # (1, 2): [Q/N, (M-1)/N] of the bound system
    if features.shape[1] != self.state_projection.weight.shape[1]:
        raise ValueError('State feature count differs from the checkpoint state projection')
    state = F.linear(features, self.state_projection.weight, self.state_projection.bias)
    initial_table = self.embedding.weight + state                       # training arithmetic: embedding(Z) + state[batch]
    first = self.blocks[0]
    normalized = first.scalar_norm(initial_table)                       # the export's FP64-internal LayerNorm
    head = first.scalar_message
    messages = F.linear(F.silu(F.linear(normalized, head[0].weight, head[0].bias)), head[2].weight, head[2].bias)
    completed['_engine_state_feature'] = features
    completed['_engine_initial_scalar'] = initial_table[numbers].contiguous()
    completed['_engine_first_messages'] = messages
    completed['_engine_first_message_atoms'] = messages[numbers].contiguous()
    completed['_engine_partition'] = schedule(numbers, self.atomic_numbers, self._engine_expert_block)
    completed['_engine_elements'] = self.element_lut[numbers].contiguous()
    completed['_engine_offset'] = self.element_offset(numbers).squeeze(-1).detach()
    return completed


def fast_forward(self, batch, csr=None, stable_sum=False, prefix_depth=None, return_prefix_depth=None):
    """Energy of one system; `csr` selects the deployed path, otherwise eager messages with the same node modules."""
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
    blocks = list(self.blocks)
    vector = scalar.new_zeros((n, 3, self.tensor_hidden))
    tensor = scalar.new_zeros((n, 5, self.tensor_hidden))
    readout_scale = 1.0 / math.sqrt(len(blocks))
    heads = self._engine_heads
    if csr is not None:
        layouts = self._engine_layouts
        if self._engine_select_two:
            refresh_two(csr, positions, len(blocks), self._engine_select_tile)
        else:
            csr.refresh_subgraphs(positions, len(blocks))
        message_vector = message_tensor = node_message = None
        atomic_energy = None
        for index, block in enumerate(blocks):
            view = csr.radius_view(block.cutoff)
            table = self.radial_tables[index].coefficients
            if index == 0:
                aggregates = FirstMessages8.apply(positions, view.cell, view.neighbors, view.shifts, view.counts,
                                                  batch['_engine_first_message_atoms'], table, float(block.cutoff), float(block.switch_width),
                                                  layouts['first_forward'], layouts['first_backward'])
            else:
                if node_message is None:
                    node_message = heads[f'message{index}'](scalar)
                aggregates = L2Messages8.apply(positions, view.cell, view.neighbors, view.shifts, view.counts, node_message,
                                               message_vector, message_tensor, table, float(block.cutoff), float(block.switch_width),
                                               layouts['l2_forward'], layouts['l2_backward'])
            outputs = self._engine_closures[index](scalar, vector, tensor, *aggregates)
            scalar, vector, tensor = outputs[:3]
            if len(outputs) == 5:
                message_vector, message_tensor = outputs[3:]
            dual = heads.get(f'dual{index}')
            if dual is not None:
                readout, node_message = dual(scalar)          # readout_i and message_{i+1} in one launch
            else:
                readout, node_message = heads[f'readout{index}'](scalar), None
            contribution = readout_scale * readout.squeeze(-1)
            atomic_energy = contribution if atomic_energy is None else atomic_energy + contribution
        atomic_energy = atomic_energy + batch['_engine_offset']
        if stable_sum:
            atomic_energy = atomic_energy.double()
        atomic_energy = atomic_energy + pair_energy(positions, csr.radius_view(self.cutoff), batch['_engine_elements'], self.radial_tables[-1])
        return atomic_energy.sum().reshape(1)
    source, destination, distance, direction, quadrupole, valid = _edge_geometry(self, batch)
    radial = self.rbf(distance)
    width = self.config.switch_width
    envelope = smooth_switch(distance, self.cutoff, width) * valid.to(distance.dtype)
    atomic_energy = None
    for index, block in enumerate(blocks):
        block_envelope = envelope if float(block.cutoff) == float(self.cutoff) else smooth_switch(distance, block.cutoff, width) * valid
        scalar, vector, tensor = block(scalar, vector, tensor, radial, direction, quadrupole, source, destination, block_envelope)
        contribution = readout_scale * self.readouts[index](self.readout_norms[index](scalar)).squeeze(-1)
        atomic_energy = contribution if atomic_energy is None else atomic_energy + contribution
    atomic_energy = atomic_energy + self.element_offset(numbers).squeeze(-1)
    if stable_sum:
        atomic_energy = atomic_energy.double()
    elements = self.element_lut[numbers]
    low = torch.minimum(elements[source], elements[destination])
    high = torch.maximum(elements[source], elements[destination])
    pair_values = 0.5 * (self.pair_coeff(low * self.n_elements + high) * radial).sum(dim=-1) * envelope
    if stable_sum:
        return atomic_energy.sum().reshape(1) + pair_values.double().sum().reshape(1)
    energy = atomic_energy.new_zeros((1,)).index_add_(0, batch['batch'], atomic_energy)
    return energy.index_add_(0, batch['batch'][source], pair_values.to(energy.dtype))


def _prepare_energy(self, batch, edges):
    prepared = _prepare(self, batch)
    return PreparedGeneralNeighbors(self, prepared, edges)


# ---------------------------------------------------------------------------------------------
# Loading.

def load(checkpoint, device='cuda', *, global_context=None, reserve=1.5, compiled_nodes=True, precision='tf32x3',
         layernorm_statistics='fp64', layouts=None, select_two=True, select_tile=4, intervals=200,
         feature_atom_tile=16, head_layout=(16, 2), expert_block=64, joined_fast=True, inductor_gemm='autotune', dual_heads=True):
    """Checked L2 export on the C = 8 path; learned tensors of the loaded export unchanged and verified bytewise.

    Defaults are the fastest measured configuration: tuned C = 8 layouts, dual heads,
    Triton joined projections (`joined_fast`), inductor-autotuned GEMMs in the compiled closures (`inductor_gemm`)."""
    if precision not in PRECISIONS or layernorm_statistics not in STATISTICS:
        raise ValueError(f'precision must be one of {PRECISIONS} and layernorm_statistics one of {STATISTICS}')
    path = resolve_checkpoint(checkpoint)
    saved = read_export(path)
    widths = check_export(saved)
    if widths['edge_hidden'] != C8:
        raise ValueError('This path executes edge width 8 only')
    rounded = saved.get('fp16_rounding')
    if precision == 'fp16w2' and not rounded:
        raise ValueError('fp16w2 requires an FP16-rounded weight file; weights are never rounded silently')
    model, saved = reference.load(path, device)          # factory build, FP64-internal LayerNorm policy, strict load, bytewise check
    check_blocks(model)
    if model.radial_tables is not None or model.fused_messages or model.reference_chunk_size:
        raise ValueError('Fresh export model expected')
    original_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    original_parameters = tuple(model.named_parameters())
    compile_tables(model, intervals)
    model.fused_message_settings = SETTINGS
    model.filter_skin_neighbors = True
    model.fused_messages = model.fused = True
    configure_neighbor_reserve(model, reserve)
    model._engine_global_context = dict(global_context) if global_context is not None else None
    prepare_nodes(model, precision, layernorm_statistics, feature_atom_tile, tuple(head_layout), expert_block, joined_fast, dual_heads)
    model._engine_layouts = check_layouts(layouts)
    model._engine_select_two = bool(select_two)
    model._engine_select_tile = int(select_tile)
    model.forward = MethodType(fast_forward, model)
    model.prepare_energy = MethodType(_prepare_energy, model)
    install_closures(model, compiled_nodes, inductor_gemm)
    verified = reference.verify_state(model, saved)
    actual = model.state_dict()
    added = sorted(set(actual).difference(original_state))
    if (set(original_state).difference(actual) or any(not k.startswith('radial_tables.') for k in added)
            or any(not torch.equal(actual[k].detach().cpu(), v) for k, v in original_state.items())):
        raise ValueError(f'The prepared model state differs from the loaded export state (added {added})')
    parameters = tuple(model.named_parameters())
    if len(parameters) != len(original_parameters) or any(a[0] != b[0] or a[1] is not b[1] for a, b in zip(parameters, original_parameters)):
        raise ValueError('Engine preparation changed the learned parameter namespace or objects')
    model.inference_backend = BACKEND
    blocks = list(model.blocks)
    model.deployment_metadata = dict(model.export_metadata,
        implementation=IMPLEMENTATION, inference_backend=BACKEND, backend=BACKEND, architecture=widths['architecture'],
        precision=precision, layernorm_statistics=layernorm_statistics,
        policy_label={('tf32x3', 'fp64'): 'i_tf32x3_fp64ln', ('fp32', 'fp64'): 'fp32_fp64ln', ('tf32x3', 'fp32'): 'ii_tf32x3_fp32ln',
                      ('fp32', 'fp32'): 'fp32_fp32ln', ('fp16w2', 'fp64'): 'iii_fp16w2_fp64ln', ('fp16w2', 'fp32'): 'iii_fp16w2_fp32ln'}.get((precision, layernorm_statistics), f'{precision}_{layernorm_statistics}ln'),
        numerics=dict(node_products=f'fused heads and grouped experts: {precision} tl.dot (ieee when fp32; fp16w2 = FP16 weight copies, split FP32 activation)',
                      joined_projections=('Triton ' + precision) if joined_fast else 'torch FP32 cuBLAS (TF32 disabled)',
                      edge_projections='torch FP32 cuBLAS (edge_vector_in/out, edge_tensor_in/out, edge_scalar_out; widths below 16)',
                      layernorm=f'{layernorm_statistics} statistics (centred two-pass), FP32 affine; heads and local features fused',
                      local_features=f'fused Triton local features, {layernorm_statistics} statistics, atom tile {feature_atom_tile}',
                      messages=f'block 0: atom-tiled scalar-source kernels, 8-lane tile; blocks 1..: atom-tiled seven-path L2 kernels, 8-lane tile; '
                               f'FP32, quintic radial tables ({intervals} intervals)',
                      pair=f'Triton symmetric pair, FP32, quintic table ({intervals} intervals, NE = {model.n_elements})',
                      first_message='constant per-atom tensor prepared outside the graph (species + global state)'),
        fp16_rounded_export=rounded, speed_only_artifact=True,
        compiled_nodes=bool(compiled_nodes), inductor_gemm=inductor_gemm, joined_fast=bool(joined_fast), dual_heads=bool(dual_heads),
        message_layouts={k: list(v) for k, v in model._engine_layouts.items()},
        head_layout=list(head_layout), local_features_atom_tile=int(feature_atom_tile), expert_block=int(expert_block),
        radius_selection='_select_two (both views, one launch)' if select_two else '_radius_select per view',
        fused_message_settings=list(SETTINGS), neighbor_reserve=float(reserve), filter_skin_neighbors=True,
        radial_table_cutoffs_A=[float(t.cutoff) for t in model.radial_tables],
        radial_table_channels=[int(t.coefficients.shape[2]) for t in model.radial_tables],
        block_layout=[dict(block=i, cutoff_A=float(b.cutoff), scalar_source=b.input_is_scalar, terminal=b.terminal, local_cross=b.local_cross,
                           messages='FirstMessages8 (8 s + 8 v + 8 t)' if i == 0 else 'L2Messages8 (seven paths, 8 lanes)',
                           expert=[int(v) for v in (b.update_mlp.weight1.shape[2], b.update_mlp.weight1.shape[1], b.update_mlp.weight2.shape[1])])
                      for i, b in enumerate(blocks)],
        widths={k: widths[k] for k in ('hidden', 'tensor_hidden', 'edge_hidden', 'interactions', 'layer_cutoffs', 'n_rbf', 'expert_hidden', 'higher_invariants')},
        parameters=sum(p.numel() for p in model.parameters()),
        layer_cutoffs_A=[float(r) for r in model.layer_cutoffs], cutoff_A=float(model.cutoff),
        dependency_radius_A=max(float(model.cutoff), float(sum(model.layer_cutoffs))),
        n_elements=int(model.n_elements), condition_spin=bool(model.condition_spin),
        state_features=list(reference.STATE_FEATURES), global_context=model._engine_global_context,
        export_tensors_verified_after_preparation=verified, learned_weights_changed=False,
        export=dict(architecture=saved['architecture'], model_factory=saved['model_factory'], selection_kind=saved.get('selection_kind'),
                    step=saved.get('step'), numerical_execution=saved.get('numerical_execution')))
    return model, saved


def create_engine(positions, atomic_numbers, cell, checkpoint, device='cuda', total_charge=0., multiplicity=1,
                  skin=1.5, reserve=1.5, compiled_nodes=True, precision='tf32x3', layouts=None, select_two=True, select_tile=4,
                  *, capture_branch_policy='first', **node_options):
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    n = len(numbers)
    context = dict(n_atoms=n, total_charge=float(total_charge), multiplicity=int(multiplicity),
                   state_features=state_features(n, total_charge, multiplicity))
    model, _ = load(checkpoint, device, global_context=context, reserve=reserve, compiled_nodes=compiled_nodes, precision=precision,
                    layouts=layouts, select_two=select_two, select_tile=select_tile, **node_options)
    batch = system_batch(positions, numbers, cell, total_charge, multiplicity, device)
    engine = ForceEngine(model, batch, 'cuda_graph' if torch.device(device).type == 'cuda' else 'eager',
                         skin=skin, capture_branch_policy=capture_branch_policy)
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


__all__ = ['load', 'create_engine', 'fast_forward', 'prepare_nodes', 'compile_tables', 'check_blocks', 'check_export', 'read_export',
           'resolve_checkpoint', 'state_features', 'system_batch', 'digest', 'BACKEND', 'IMPLEMENTATION',
           'DEFAULT_LAYOUTS', 'PRECISIONS', 'STATISTICS']
