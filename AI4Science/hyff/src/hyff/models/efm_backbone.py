"""EFM backbone constructor: stabilizer + per-block edge-frame mixing (stacked layers) + receiver
conditioning (T7) + edge MLP (T8) + optional angular moments, with ONE block forward.

`build_model(config)` takes a plain model dict plus the ingredient keys below. Every ingredient is off unless
its keys are present; with none present the model is the base StatePotential
(hyff.models.base_model.build_model) itself, bitwise.

Ingredients (every one is identity at initialization except angular, whose projection starts at 0.1 xavier):
  stab_*          stabilizer keys (hyff.models.stabilized), applied by STABILIZED.build_stab_model (asinh on |vv|, |tv| of the
                  expert-MLP input, per-column output gates at 1.0; stab_aggregate_normalization is re-implemented in
                  the unified forward with the same operation order).
  efm_*           edge-frame mixing (hyff.models.edge_frame_mixing). efm_blocks: list (construction order) or absent;
                  efm_channels / efm_mmax / efm_layers: scalar (all EFM blocks), list of length n_blocks with null
                  exactly on the non-EFM blocks, or dict {block: value}. efm_layers (1-4) = number of gated mixing
                  layers in the edge frame; layer 1 is EFM's (m=0 block W0 over [s, r0, a0, c0], gates, |m|=1 and
                  |m|=2 blocks on the radial-weighted frame components); every further layer is an EdgeFrameLayer
                  with its own weights and gates (residual, see below). The zero exits stay at the end.
                  efm_aggregation / efm_complex / efm_frame are global as in EFM (layers > 1 or a receiver need the
                  O(3) projector form).
  receiver_filter T7: edge_message *= (1 + W_r LN(s))[dst] on the P*M radial-gated message paths, W_r zero,
                  bias-free (hidden -> P*M). bool | list of blocks | list of n bools | dict {block: bool}.
  efm_receiver    T7 in EFM: the receiver's projected scalar node_scalar(LN(s))[dst] enters the first EFM m=0 block
                  through a zero-initialised bias-free Linear(C -> 3C) (the same as zero input columns of W0; kept
                  as its own matrix, routed to Muon on its own). Same block-set syntax; must be a subset of the EFM
                  blocks.
  edge_mlp        T8: e = W2 SiLU(A LN(s)[dst] + B LN(s)[src] + Cr f_e + b), W2 zero (edge_mlp_hidden -> P*M),
                  f_e = the block's radial-filter hidden activation (edge_mlp_radial 'hidden', default) or the rbf
                  ('rbf'); edge_message = node_message[src] * radial + e, before the envelope weight.
                  edge_mlp_hidden default = the block's message width. [A B Cr] is one xavier matrix drawn from a
                  dedicated generator (seed 20_260_932) consumed in ascending block order; b = 0.
  angular_*       angular moments (hyff.models.angular_branch) (angular_channels, angular_degrees, angular_init_scale,
                  angular_blocks, default all): scalar term added before the node update.

Parameter creation order (reproducible, base tensors bitwise the plain/stab build with the same seed):
  base build (global RNG) -> stab wrappers (no draw) -> EFM per block in efm_blocks order (global RNG; layer 1 draws
  exactly as EdgeFrameMixing, then receiver_in (zero, no draw), then the extra layers) -> angular per block
  (global RNG, as build_angular_model) -> T7 matrices (zero, no draw) -> T8 (own generator).
  A config with only EFM keys (layers 1, no receiver) builds EFM.EdgeFrameMixing itself; only-stab is
  build_stab_model; only-angular reproduces build_angular_model.

Unified block forward (_unified_block_forward), order of operations:
   1. count_i = sum_e env_e                                  (stab aggregate normalization only; first, as in STABILIZED)
   2. if EFM or angular:  n = LN(s); node_message = scalar_message(n); efm_term = EFM(n, v, T, ...);
                          angular_term = Angular(n, ...)      (EFM / angular reference order: before the bottleneck)
   3. bottleneck: message_vector/tensor = edge_vector_in(v) / edge_tensor_in(T) (zeros in block 0)
   4. otherwise:          n = LN(s); node_message = scalar_message(n)
   5. m = node_message[src]; radial = radial_filter(rbf) (layers 0-2 called one by one when T8 reads the hidden)
   6. m = m * radial
   7. T8: m = m + edge_mlp(n, f, src, dst)
   8. T7: m = m * (1 + receiver_filter(n))[dst]
   9. m = m * (0.25 * env[:, None])
  10. paths (3 / 7 / 5) and the three index_add_ aggregates, verbatim from L2Interaction.forward
  11. stab aggregate normalization: aggregates * rsqrt(1 + count) ('sqrt') or / (1 + count) ('mean')
  12. angular: s = s + angular_term
  13. edge_scalar_out / edge_vector_out / edge_tensor_out; s, v, T = s + A_s, v + A_v, T + A_T
  14. EFM: s, v, T = s + ds, v + dv, T + dt
  15. L2Interaction._node_update(s, v, T)
With only one ingredient on, steps 1-15 create the same autograd operations in the same order as that
ingredient's own constructor (bitwise on energies and forces).

EdgeFrameLayer (layer k >= 2; h, a, c = m=0 scalar / l=1 / l=2 coefficients, xv, xw = |m|=1 in-plane l=1 / l=2
vectors, z = |m|=2 in-plane STF part; all per active edge, C channels):
  [s2, a2, c2] = W0 [h, a, c] + b0;  g = sigmoid(Wg s2 + bg) (Wg xavier x 0.1)
  h += SiLU(s2);  a += g_a a2;  c += g_c c2;  [yv, yw] = W1 [xv, xw];  xv += g_v yv;  xw += g_w yw;  z += g_z W2 z
  (dense over l inside each |m|, shared by the in-plane Cartesian components; no inter-layer norm; exactly O(3)).
Rotate back after the last layer: v_e = a u + xv, T_e = c sqrt(3/2) Q(u) + sqrt2 ST(xw (x) u) + z.

Eager reference geometry only (no radial tables, CSR, chunking, frozen first messages); no fused MD backend.
"""
from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn

from hyff.models.base_model import build_model as build_plain
from hyff.models import stabilized as STABILIZED
from hyff.models import edge_frame_mixing as EFM
from hyff.models import angular_branch as ANGULAR
from hyff.models.l2_potential import _linear, _tensor_dot_direction, _vector_tensor_product

VARIANT = 'efm_backbone'
STAB_KEYS = ('stab_invariant_transform', 'stab_output_gates', 'stab_gate_init', 'stab_residual_scale',
             'stab_aggregate_normalization')
EFM_PER_BLOCK = dict(efm_channels=16, efm_mmax=1, efm_layers=1)  # defaults of EFM.build_model, plus efm_layers
EFM_GLOBAL = dict(efm_aggregation='sum', efm_complex=False, efm_frame='projector')
ANGULAR_KEYS = dict(angular_channels=16, angular_degrees=(3, 4), angular_init_scale=0.1, angular_blocks=None)
BACKBONE_DEFAULTS = dict(receiver_filter=False, efm_receiver=False, edge_mlp=False, edge_mlp_hidden=None,
                     edge_mlp_radial='hidden')
EDGE_MLP_SEED = 20_260_932
MAX_EFM_LAYERS = 4
SQRT2 = EFM.SQRT2
SQRT_3_2 = EFM.SQRT_3_2


# --------------------------------------------------------------------------------------------------
# small helpers (no global RNG draw)
# --------------------------------------------------------------------------------------------------
def _zero_linear(in_features, out_features, bias=False):
    layer = nn.Linear(in_features, out_features, bias=bias, device='meta')
    layer.weight = nn.Parameter(torch.zeros(out_features, in_features))
    if bias:
        layer.bias = nn.Parameter(torch.zeros(out_features))
    return layer


def _xavier_(weight, generator, gain=1.0):
    fan_out, fan_in = weight.shape
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight.uniform_(-bound, bound, generator=generator)
    return weight


def paths(block):
    return 3 if block.input_is_scalar else (7 if block.cross_order else 5)


# --------------------------------------------------------------------------------------------------
# per-block option parsing
# --------------------------------------------------------------------------------------------------
def block_set(value, n, name):
    """bool | list of block indices | list of n booleans | dict {block: bool} -> tuple of block indices (ascending)."""
    if value is None or value is False:
        return ()
    if value is True:
        return tuple(range(n))
    if isinstance(value, dict):
        items = {int(k): v for k, v in value.items()}
        if any(type(v) is not bool for v in items.values()) or any(not 0 <= k < n for k in items):
            raise ValueError(f'{name}: dict must map valid block indices to booleans')
        return tuple(sorted(k for k, v in items.items() if v))
    if isinstance(value, (list, tuple)):
        if value and all(type(x) is bool for x in value):
            if len(value) != n:
                raise ValueError(f'{name}: a boolean list needs one entry per block ({n})')
            return tuple(i for i, x in enumerate(value) if x)
        if any(type(x) is not int for x in value):
            raise ValueError(f'{name}: list entries must be block indices (int) or one boolean per block')
        if len(set(value)) != len(value) or any(not 0 <= i < n for i in value):
            raise ValueError(f'{name}: block indices must be distinct and in [0, {n})')
        return tuple(sorted(value))
    raise ValueError(f'{name}: expected bool, list or dict, got {value!r}')


def _structured_blocks(value, n, name):
    if isinstance(value, dict):
        keys = sorted(int(k) for k in value)
        if any(not 0 <= k < n for k in keys):
            raise ValueError(f'{name}: invalid block index')
        return tuple(keys)
    if len(value) != n:
        raise ValueError(f'{name}: a per-block list needs one entry per block ({n}); use null for blocks without EFM')
    return tuple(i for i, v in enumerate(value) if v is not None)


def per_block(value, blocks, n, name, default):
    """scalar | list (length n, null exactly off the EFM blocks) | dict {block: value} -> {block: int}."""
    if value is None:
        return {b: int(default) for b in blocks}
    if isinstance(value, dict):
        items = {int(k): v for k, v in value.items()}
        if set(items) != set(blocks):
            raise ValueError(f'{name}: dict keys {sorted(items)} must be exactly the EFM blocks {sorted(blocks)}')
        return {b: int(items[b]) for b in blocks}
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f'{name}: a per-block list needs one entry per block ({n})')
        for i, v in enumerate(value):
            if (v is None) != (i not in blocks):
                raise ValueError(f'{name}: entries must be non-null exactly on the EFM blocks {sorted(blocks)}')
        return {b: int(value[b]) for b in blocks}
    if type(value) is bool:
        raise ValueError(f'{name}: expected an integer')
    return {b: int(value) for b in blocks}


def parse(config):
    """Split a model dict into (plain base keys, stab keys or None, EFM spec or None, angular spec or None, backbone keys)."""
    values = dict(config)
    stab = {k: values.pop(k) for k in STAB_KEYS if k in values}
    efm_present = any(k in values for k in ('efm_blocks', *EFM_PER_BLOCK))
    efm = None
    if efm_present:
        efm = dict(blocks=values.pop('efm_blocks', None),
                   **{k: values.pop(k, None) for k in EFM_PER_BLOCK},
                   **{k: values.pop(k, d) for k, d in EFM_GLOBAL.items()})
    elif any(k in values for k in EFM_GLOBAL):
        raise ValueError('efm_aggregation / efm_complex / efm_frame given without EFM blocks or settings')
    angular = None
    if any(k in values for k in ANGULAR_KEYS):
        angular = {k: values.pop(k, d) for k, d in ANGULAR_KEYS.items()}
    backbone = {k: values.pop(k, d) for k, d in BACKBONE_DEFAULTS.items()}
    return values, (stab if stab else None), efm, angular, backbone


def resolve_efm(efm, n):
    """EFM spec -> (construction-ordered blocks, {block: dict(channels, mmax, layers)})."""
    if efm['blocks'] is not None:
        if any(type(b) is not int for b in efm['blocks']):
            raise ValueError('efm_blocks entries must be integers')
        blocks = [int(b) for b in efm['blocks']]
        if not blocks or len(set(blocks)) != len(blocks) or any(not 0 <= b < n for b in blocks):
            raise ValueError('efm_blocks must be distinct valid block indices')
    else:
        structured = [(k, efm[k]) for k in EFM_PER_BLOCK if isinstance(efm[k], (list, tuple, dict))]
        if structured:
            sets = {_structured_blocks(v, n, k) for k, v in structured}
            if len(sets) != 1:
                raise ValueError(f'per-block EFM settings disagree on the EFM blocks: {sorted(sets)}')
            blocks = list(sets.pop())
        else:
            blocks = list(range(n))
        if not blocks:
            raise ValueError('no EFM block selected')
    settings = {k: per_block(efm[k], blocks, n, k, d) for k, d in EFM_PER_BLOCK.items()}
    out = {}
    for b in blocks:
        row = dict(channels=settings['efm_channels'][b], mmax=settings['efm_mmax'][b], layers=settings['efm_layers'][b])
        if row['channels'] < 1 or row['mmax'] not in (1, 2) or not 1 <= row['layers'] <= MAX_EFM_LAYERS:
            raise ValueError(f'block {b}: efm_channels >= 1, efm_mmax in (1, 2), efm_layers in [1, {MAX_EFM_LAYERS}]')
        out[b] = row
    return blocks, out


# --------------------------------------------------------------------------------------------------
# modules
# --------------------------------------------------------------------------------------------------
class EdgeFrameLayer(nn.Module):
    """One further gated edge-frame mixing layer (dense over l inside each |m| block, its own gates)."""

    def __init__(self, channels, m_blocks, n_gates):
        super().__init__()
        C = int(channels)
        self.channels, self.m_blocks = C, tuple(m_blocks)
        self.m0 = _linear(3 * C, 3 * C)
        self.gate = _linear(C, n_gates * C)
        with torch.no_grad():
            self.gate.weight.mul_(0.1)
        if 1 in self.m_blocks:
            self.m1 = _linear(2 * C, 2 * C, bias=False)
        if 2 in self.m_blocks:
            self.m2 = _linear(C, C, bias=False)

    def forward(self, h, a, c, xv, xw, z):
        C = self.channels
        s2, a2, c2 = self.m0(torch.cat((h, a, c), -1)).split(C, -1)
        g = torch.sigmoid(self.gate(s2)).split(C, -1)
        h = h + F.silu(s2)
        a = a + a2 * g[0]
        c = c + c2 * g[1]
        if 1 in self.m_blocks:
            yv, yw = self.m1(torch.cat((xv, xw), -1)).split(C, -1)
            xv = xv + yv * g[2][:, None, :]
            xw = xw + yw * g[3][:, None, :]
        if 2 in self.m_blocks:
            z = z + self.m2(z) * g[4][:, None, :]
        return h, a, c, xv, xw, z


class GeneralEdgeFrameMixing(EFM.EdgeFrameMixing):
    """EdgeFrameMixing (projector frame, O(3)) with stacked layers and an optional receiver input of the m=0 block."""

    def __init__(self, scalar_hidden, tensor_hidden, n_rbf, channels, mmax, input_is_scalar, message_scale,
                 aggregation='sum', layers=1, receiver=False):
        super().__init__(scalar_hidden, tensor_hidden, n_rbf, channels, mmax, input_is_scalar, message_scale,
                         aggregation, False, 'projector')
        if not 1 <= int(layers) <= MAX_EFM_LAYERS:
            raise ValueError(f'efm_layers must be in [1, {MAX_EFM_LAYERS}]')
        self.n_layers = int(layers)
        self.receiver_in = _zero_linear(self.channels, 3 * self.channels) if receiver else None
        self.layers = nn.ModuleList(EdgeFrameLayer(self.channels, self.m_blocks, self.n_gates)
                                    for _ in range(self.n_layers - 1))

    def forward(self, normalized_scalar, vector, tensor, radial, direction, quadrupole, source, destination, envelope):
        if self.frame != 'projector' or self.complex_structure:
            raise ValueError('stacked / receiver EFM supports the O(3) projector form only')
        n_atoms = normalized_scalar.shape[0]
        C = self.channels
        active = torch.nonzero(envelope != 0, as_tuple=False).squeeze(-1)
        src = source.index_select(0, active)
        dst = destination.index_select(0, active)
        u = direction.index_select(0, active)
        quad = quadrupole.index_select(0, active)
        env = envelope.index_select(0, active)
        rad = self.radial(radial.index_select(0, active))
        r0, rad = rad[:, :C], rad[:, C:]
        projected = self.node_scalar(normalized_scalar)
        s = projected.index_select(0, src)
        inputs = [s, r0]
        if self.m_blocks:
            comps = self._to_frame(self.node_vector(vector).index_select(0, src),
                                   self.node_tensor(tensor).index_select(0, src), u, quad, None, None)
            inputs += [comps['a0'], comps['c0']]
        mixed = self.m0(torch.cat(inputs, -1))
        if self.receiver_in is not None:
            mixed = mixed + self.receiver_in(projected).index_select(0, dst)
        s1, a1, c1 = mixed.split(C, -1)
        gates = torch.sigmoid(self.gate(s1)).split(C, -1)
        h = F.silu(s1)
        a = a1 * gates[0]
        c = c1 * gates[1]
        xv = xw = z = None
        if 1 in self.m_blocks:
            yv, yw = self.m1_a(comps['X1'] * rad[:, None, :2 * C]).split(C, -1)
            xv = yv * gates[2][:, None, :]
            xw = yw * gates[3][:, None, :]
        if 2 in self.m_blocks:
            z = self.m2_a(comps['Z2'] * rad[:, None, 2 * C:3 * C]) * gates[4][:, None, :]
        for layer in self.layers:
            h, a, c, xv, xw, z = layer(h, a, c, xv, xw, z)
        v_e = a[:, None, :] * u[:, :, None]
        t_e = c[:, None, :] * (SQRT_3_2 * quad)[:, :, None]
        if 1 in self.m_blocks:
            v_e = v_e + xv
            t_e = t_e + SQRT2 * _vector_tensor_product(xw, u)[0]
        if 2 in self.m_blocks:
            t_e = t_e + z
        if self.aggregation == 'sum':
            weight = self.message_scale * env
        else:
            degree = env.new_zeros((n_atoms,)).index_add_(0, dst, env)
            weight = env / torch.sqrt(1. + degree.index_select(0, dst))
        agg_s = h.new_zeros((n_atoms, C)).index_add_(0, dst, h * weight[:, None])
        agg_v = h.new_zeros((n_atoms, 3, C)).index_add_(0, dst, v_e * weight[:, None, None])
        agg_t = h.new_zeros((n_atoms, 5, C)).index_add_(0, dst, t_e * weight[:, None, None])
        return self.exit_scalar(agg_s), self.exit_vector(agg_v), self.exit_tensor(agg_t)


class EdgeMLP(nn.Module):
    """T8: e = W2 SiLU(A LN(s)[dst] + B LN(s)[src] + Cr f + b); W2 zero so e = 0 at initialization."""

    def __init__(self, hidden, radial_width, width, out_features, generator):
        super().__init__()
        self.dst_proj = _zero_linear(hidden, width, bias=True)   # A, b
        self.src_proj = _zero_linear(hidden, width)              # B
        self.radial_proj = _zero_linear(radial_width, width)     # Cr
        joint = _xavier_(torch.empty(width, 2 * hidden + radial_width), generator)
        with torch.no_grad():
            self.dst_proj.weight.copy_(joint[:, :hidden])
            self.src_proj.weight.copy_(joint[:, hidden:2 * hidden])
            self.radial_proj.weight.copy_(joint[:, 2 * hidden:])
        self.exit = _zero_linear(width, out_features)            # W2

    def forward(self, normalized, radial_features, source, destination):
        pre = (self.dst_proj(normalized).index_select(0, destination)
               + self.src_proj(normalized).index_select(0, source)
               + self.radial_proj(radial_features))
        return self.exit(F.silu(pre))


# --------------------------------------------------------------------------------------------------
# the unified block forward
# --------------------------------------------------------------------------------------------------
def _unified_block_forward(self, scalar, vector, tensor, radial, direction, quadrupole, source, destination,
                        edge_envelope, radial_output=None, csr=None, reference=None, node_message=None):
    if csr is not None or reference is not None or radial_output is not None or node_message is not None:
        raise ValueError('backbone blocks support the eager reference geometry only (no radial tables, CSR, chunking or frozen messages)')
    spec = self.__dict__['_efm_block_spec']
    n_atoms = scalar.shape[0]
    hidden = self.message_hidden
    # 1. smooth neighbour count (stab aggregate normalization)
    count = None
    if spec['aggregate_normalization'] is not None:
        count = scalar.new_zeros((n_atoms,)).index_add(0, destination, edge_envelope)
    # 2. EFM / angular reference order: LN and node message before the bottleneck
    normalized = efm_term = angular_term = None
    if spec['efm'] or spec['angular']:
        normalized = self.scalar_norm(scalar)
        node_message = self.scalar_message(normalized)
        if spec['efm']:
            efm_term = self.efm(normalized, vector, tensor, radial, direction, quadrupole, source, destination, edge_envelope)
        if spec['angular']:
            angular_term = self.angular(normalized, radial, direction, source, destination, edge_envelope)
    # 3. bottleneck
    if self.tensor_bottleneck:
        if self.input_is_scalar:
            message_vector = vector.new_zeros((n_atoms, 3, hidden))
            message_tensor = tensor.new_zeros((n_atoms, 5, hidden))
        else:
            message_vector = self.edge_vector_in(vector)
            message_tensor = self.edge_tensor_in(tensor)
    else:
        message_vector, message_tensor = vector, tensor
    # 4. base order
    if normalized is None:
        normalized = self.scalar_norm(scalar)
        node_message = self.scalar_message(normalized)
    # 5.-9. edge message
    edge_message = node_message[source]
    radial_features = None
    if spec['edge_mlp'] and spec['edge_mlp_radial'] == 'hidden':
        radial_features = self.radial_filter[1](self.radial_filter[0](radial))
        radial_out = self.radial_filter[2](radial_features)
    else:
        radial_out = self.radial_filter(radial)
    edge_message = edge_message * radial_out
    if spec['edge_mlp']:
        edge_message = edge_message + self.edge_mlp(normalized, radial if radial_features is None else radial_features,
                                                    source, destination)
    if spec['receiver_filter']:
        edge_message = edge_message * (1. + self.receiver_filter(normalized)).index_select(0, destination)
    edge_message = edge_message * (self.message_scale * edge_envelope[:, None])
    # 10. paths and aggregates (verbatim L2Interaction.forward)
    if self.input_is_scalar:
        msg_s, msg_vr, msg_tr = edge_message.chunk(3, dim=-1)
        edge_vector = msg_vr[:, None, :] * direction[:, :, None]
        edge_tensor = msg_tr[:, None, :] * quadrupole[:, :, None]
    elif self.cross_order:
        source_vector = message_vector[source]
        source_tensor = message_tensor[source]
        msg_s, msg_vr, msg_v, msg_tv, msg_tr, msg_t, msg_vt = edge_message.chunk(7, dim=-1)
        tensor_vector, tensor_projection = _tensor_dot_direction(source_tensor, direction)
        vector_tensor, vector_projection = _vector_tensor_product(source_vector, direction)
        msg_s = msg_s + msg_vt * vector_projection + msg_tv * tensor_projection
        edge_vector = (msg_v[:, None, :] * source_vector + msg_vr[:, None, :] * direction[:, :, None]
                       + msg_tv[:, None, :] * tensor_vector)
        edge_tensor = (msg_t[:, None, :] * source_tensor + msg_tr[:, None, :] * quadrupole[:, :, None]
                       + msg_vt[:, None, :] * vector_tensor)
    else:
        source_vector = message_vector[source]
        source_tensor = message_tensor[source]
        msg_s, msg_vr, msg_v, msg_tr, msg_t = edge_message.chunk(5, dim=-1)
        edge_vector = msg_v[:, None, :] * source_vector + msg_vr[:, None, :] * direction[:, :, None]
        edge_tensor = msg_t[:, None, :] * source_tensor + msg_tr[:, None, :] * quadrupole[:, :, None]
    aggregate_s = scalar.new_zeros((n_atoms, hidden))
    aggregate_v = vector.new_zeros((n_atoms, 3, hidden))
    aggregate_t = tensor.new_zeros((n_atoms, 5, hidden))
    aggregate_s.index_add_(0, destination, msg_s)
    aggregate_v.index_add_(0, destination, edge_vector)
    aggregate_t.index_add_(0, destination, edge_tensor)
    # 11. stab aggregate normalization
    if count is not None:
        scale = torch.rsqrt(1. + count) if spec['aggregate_normalization'] == 'sqrt' else 1. / (1. + count)
        aggregate_s = aggregate_s * scale[:, None]
        aggregate_v = aggregate_v * scale[:, None, None]
        aggregate_t = aggregate_t * scale[:, None, None]
    # 12. angular scalar term
    if angular_term is not None:
        scalar = scalar + angular_term
    # 13. output projections and residual sums (L2Interaction._finish_messages without the folded branch)
    if self.bottleneck:
        aggregate_s = self.edge_scalar_out(aggregate_s)
    if self.tensor_bottleneck:
        aggregate_v = self.edge_vector_out(aggregate_v)
        aggregate_t = self.edge_tensor_out(aggregate_t)
    scalar, vector, tensor = scalar + aggregate_s, vector + aggregate_v, tensor + aggregate_t
    # 14. EFM exits
    if efm_term is not None:
        ds, dv, dt = efm_term
        scalar, vector, tensor = scalar + ds, vector + dv, tensor + dt
    # 15. node update
    return self._node_update(scalar, vector, tensor)


PATCH_ATTRIBUTES = ('forward', '_finish_messages', '_stab_base_forward', '_stab_base_finish_messages')  # stab aggnorm only


def install_unified_forward(block, spec):
    """Replace any per-instance patch of the block by the one unified forward driven by `spec`."""
    if hasattr(block, 'folded_first_vector'):
        raise ValueError('backbone does not support folded first-block projections')
    for name in PATCH_ATTRIBUTES:
        block.__dict__.pop(name, None)
    for name in ('forward', '_finish_messages', '_node_update'):
        if name in block.__dict__:
            raise RuntimeError(f'unexpected instance override {name}')
    block.__dict__['_efm_block_spec'] = dict(spec)
    block.forward = MethodType(_unified_block_forward, block)
    return block


def base_spec():
    return dict(efm=False, angular=False, receiver_filter=False, edge_mlp=False, edge_mlp_radial='hidden',
                aggregate_normalization=None)


# --------------------------------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------------------------------
def build_model(config, force_unified=False):
    values, stab, efm, angular, backbone = parse(config)
    model = STABILIZED.build_stab_model({**values, **stab}) if stab is not None else build_plain(values)
    n = len(model.blocks)
    n_rbf = model.config.n_rbf
    rows = [dict(block=i, cutoff_A=float(b.cutoff), paths=paths(b), message_width=b.message_hidden) for i, b in enumerate(model.blocks)]
    # --- EFM (global RNG, construction order = efm_blocks order) ---
    efm_blocks, efm_settings = (), {}
    efm_receiver = block_set(backbone['efm_receiver'], n, 'efm_receiver')
    if efm is not None:
        efm_blocks, efm_settings = resolve_efm(efm, n)
        if efm['efm_aggregation'] not in EFM.AGGREGATIONS or efm['efm_frame'] not in EFM.FRAMES:
            raise ValueError('efm_aggregation / efm_frame invalid')
        if type(efm['efm_complex']) is not bool:
            raise ValueError('efm_complex must be a boolean')
        if not set(efm_receiver) <= set(efm_blocks):
            raise ValueError('efm_receiver blocks must be EFM blocks')
        for index in efm_blocks:
            block, row = model.blocks[index], efm_settings[index]
            general = row['layers'] > 1 or index in efm_receiver
            if general and (efm['efm_complex'] or efm['efm_frame'] != 'projector'):
                raise ValueError('stacked layers / efm_receiver need efm_complex false and efm_frame projector')
            if general:
                block.efm = GeneralEdgeFrameMixing(block.scalar_hidden, block.tensor_hidden, n_rbf, row['channels'], row['mmax'],
                                                   block.input_is_scalar, block.message_scale, efm['efm_aggregation'],
                                                   row['layers'], index in efm_receiver)
            else:
                block.efm = EFM.EdgeFrameMixing(block.scalar_hidden, block.tensor_hidden, n_rbf, row['channels'], row['mmax'],
                                                block.input_is_scalar, block.message_scale, efm['efm_aggregation'],
                                                efm['efm_complex'], efm['efm_frame'])
            rows[index]['efm'] = dict(row, receiver=index in efm_receiver, m_blocks=list(block.efm.m_blocks),
                                      module=type(block.efm).__name__, parameters=sum(p.numel() for p in block.efm.parameters()))
    elif efm_receiver:
        raise ValueError('efm_receiver needs EFM')
    # --- angular (global RNG, as build_angular_model) ---
    angular_blocks = ()
    if angular is not None:
        channels, degrees, init_scale = int(angular['angular_channels']), tuple(angular['angular_degrees']), float(angular['angular_init_scale'])
        if channels < 1 or not 0. <= init_scale <= 1.:
            raise ValueError('angular_channels must be positive and angular_init_scale in [0, 1]')
        angular_blocks = tuple(range(n)) if angular['angular_blocks'] is None else block_set(angular['angular_blocks'], n, 'angular_blocks')
        for index in angular_blocks:
            block = model.blocks[index]
            block.angular = ANGULAR.AngularMomentBranch(block.scalar_hidden, n_rbf, channels, degrees, init_scale, block.message_scale)
            rows[index]['angular'] = dict(channels=channels, degrees=list(degrees), init_scale=init_scale,
                                          parameters=sum(p.numel() for p in block.angular.parameters()))
    # --- T7 receiver filter (zero, no draw) ---
    receiver_blocks = block_set(backbone['receiver_filter'], n, 'receiver_filter')
    for index in receiver_blocks:
        block = model.blocks[index]
        block.receiver_filter = _zero_linear(block.scalar_hidden, paths(block) * block.message_hidden)
        rows[index]['receiver_filter_parameters'] = block.receiver_filter.weight.numel()
    # --- T8 edge MLP (own generator, ascending block order) ---
    edge_blocks = block_set(backbone['edge_mlp'], n, 'edge_mlp')
    if backbone['edge_mlp_radial'] not in ('hidden', 'rbf'):
        raise ValueError("edge_mlp_radial must be 'hidden' or 'rbf'")
    if edge_blocks:
        generator = torch.Generator().manual_seed(EDGE_MLP_SEED)
        for index in edge_blocks:
            block = model.blocks[index]
            width = int(backbone['edge_mlp_hidden'] or block.message_hidden)
            radial_width = block.radial_filter[0].weight.shape[0] if backbone['edge_mlp_radial'] == 'hidden' else n_rbf
            block.edge_mlp = EdgeMLP(block.scalar_hidden, radial_width, width, paths(block) * block.message_hidden, generator)
            rows[index]['edge_mlp'] = dict(hidden=width, radial=backbone['edge_mlp_radial'], radial_width=int(radial_width),
                                           parameters=sum(p.numel() for p in block.edge_mlp.parameters()))
    elif backbone['edge_mlp_hidden'] is not None:
        raise ValueError('edge_mlp_hidden given without edge_mlp')
    # --- one forward per block ---
    for index, block in enumerate(model.blocks):
        spec = dict(efm=index in efm_blocks, angular=index in angular_blocks, receiver_filter=index in receiver_blocks,
                    edge_mlp=index in edge_blocks, edge_mlp_radial=backbone['edge_mlp_radial'],
                    aggregate_normalization=getattr(block, 'stab_aggregate_normalization', None))
        needed = force_unified or spec['efm'] or spec['angular'] or spec['receiver_filter'] or spec['edge_mlp'] or spec['aggregate_normalization'] is not None
        if needed:
            install_unified_forward(block, spec)
        rows[index]['block_forward'] = 'unified' if needed else 'base'
        rows[index]['spec'] = spec
    # --- LayerNorm policy: backbone creates no LayerNorm, so the FP64-internal policy list of the base stays exact ---
    names = [name for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    if names != list(model.numerical_execution['layernorm_modules']):
        raise RuntimeError('LayerNorm set changed; the stable-LayerNorm policy record would be stale')
    new = [(nm, p) for nm, p in model.named_parameters() if any(t in nm for t in NEW_MODULE_TAGS)]
    model.efm_backbone_spec = dict(
        variant=VARIANT, stab=getattr(model, 'stabilized_spec', None) and {k: v for k, v in model.stabilized_spec.items() if k != 'blocks'},
        efm=None if efm is None else dict(blocks=list(efm_blocks), aggregation=efm['efm_aggregation'], complex_structure=efm['efm_complex'],
                                           frame=efm['efm_frame'], receiver_blocks=list(efm_receiver)),
        angular_blocks=list(angular_blocks), receiver_filter_blocks=list(receiver_blocks), edge_mlp_blocks=list(edge_blocks),
        edge_mlp_seed=EDGE_MLP_SEED if edge_blocks else None, blocks=rows,
        new_parameter_tensors=len(new), new_parameters=sum(p.numel() for _, p in new),
        initialization='base tensors bitwise the plain/stab build; EFM, T7, T8 exits zero (identity at init); angular projection xavier x init_scale',
        implementation='eager reference geometry; one unified block forward; per-block active-edge selection in EFM/angular; no fused MD backend')
    return model


NEW_MODULE_TAGS = ('.efm.', '.angular.', '.receiver_filter.', '.edge_mlp.')


def new_parameters(model):
    return [(n, p) for n, p in model.named_parameters() if any(t in n for t in NEW_MODULE_TAGS)]
