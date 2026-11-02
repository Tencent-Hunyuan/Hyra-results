"""Stabilized node-update MLP for the EFM models.

build_stab_model builds the base StatePotential chain (hyff.models.base_model ->
hyff.models.state_model -> hyff.models.element_model -> hyff.models.l2_potential)
from the plain model keys and then wraps every interaction block's node-update
MLP (the 83 element experts) in a StabilizedUpdateMLP. All parameters created
here are drawn without the global RNG, so every base tensor is bitwise the
initialization of the plain model with the same model keys.

Why the update MLP. In L2Interaction._node_update the expert MLP input is
cat(LN(s), |vv|, |tv|, tr(t^3), v.t.v, |t v|^2). The last three are formed from
the smoothly normalized v = vv / sqrt(1 + |vv|^2), t = tv / sqrt(1 + |tv|^2) and are
bounded by construction. The two norms |vv|, |tv| are the raw, unbounded magnitudes
of the equivariant residual streams. The MLP output (scalar update and the
vector / tensor gates upd_v, upd_t) is asymptotically linear in them (Linear-SiLU-
Linear), and the block adds upd_v * vu and upd_t * tu to the streams, so the
per-block map is QUADRATIC in the stream magnitude: v_{k+1} ~ v_k + (a + b|v_k|) U v_k.
Four blocks (as in moderato) tolerate this; each extra block compounds it, and the failure
appears as a single-batch, single-atom overflow (a nonfinite loss with a normal
gradient-norm history right before it). The scalar
stream is only ever consumed through LayerNorms (FP64-internal) and is not the
limiting quantity.

Stabilizer (all optional, model-config keys with the stab_ prefix):

  stab_invariant_transform   'asinh' (default): |vv| -> asinh(|vv|), |tv| -> asinh(|tv|)
                             on exactly the two norm columns. Smooth, monotone,
                             identity to first order (asinh x = x - x^3/6 + ...),
                             logarithmic for large arguments, so the MLP output and
                             hence the residual gates grow at most logarithmically
                             with the stream magnitude and the per-block recurrence
                             becomes (essentially) linear with a bounded factor.
                             Magnitude information is kept (monotone), unlike a
                             LayerNorm. At initialization the norms are about 0.1-0.2
                             (relative change below 1 percent), so the initial
                             function is the base's up to that compression.
                             'layernorm': one FP64-internal LayerNorm (same policy as
                             hyff.models.base_model) over the whole 5 x tensor_hidden
                             invariant vector instead (scale-free, bounded; not
                             identity at init). None: off.
  stab_output_gates          true (default): one learnable gate per output column of
                             the update MLP (scalar update, vector-dot and tensor-dot
                             closures, vector and tensor gates), initialized at
                             stab_gate_init (default 1.0 -> exactly the base
                             function at init; 0.0 would be ReZero). Routed to AdamW
                             by the HybridMuon adapter (1-D parameter), so the update
                             amplitude has a fast per-channel path that does not go
                             through Muon's orthogonalized matrix updates.
  stab_residual_scale        null (default) | 'inv_sqrt_blocks' | float: multiplies the
                             gate initialization (depth scaling of every residual
                             update produced by the node update). Not identity at init.
  stab_aggregate_normalization  null (default) | 'sqrt' | 'mean': divide each block's
                             per-atom message aggregates (before the bottleneck
                             projections) by sqrt(1 + n_i) or (1 + n_i), where
                             n_i = sum of the block's C2 envelope over incoming edges
                             (a smooth neighbour count). Not identity at init.

Eager reference geometry only (like the base StatePotential). The fused MD kernels
of the native engines would need the same input compression in the node stage.
"""
from __future__ import annotations

import math
from types import MethodType

import torch
from torch import nn

from hyff.models.base_model import build_model as build_base, forward_fp64

VARIANT = 'hyff_xl_stabilized_update_mlp'
TRANSFORMS = ('asinh', 'layernorm', None)
AGGREGATE_NORMALIZATIONS = ('sqrt', 'mean', None)


class StabilizedUpdateMLP(nn.Module):
    """Wraps a block's update MLP: compresses the two unbounded invariant columns of its input, gates its output.

    Exposes partition / experts / out_features of the wrapped element-expert MLP so the
    StatePotential forward (which binds the species partition on block.update_mlp) and
    the capacity accounting keep working unchanged.
    """

    def __init__(self, inner, hidden, tensor_hidden, n_invariants, output_layout, transform, gates, gate_init):
        super().__init__()
        if transform not in TRANSFORMS:
            raise ValueError(f'stab_invariant_transform must be one of {TRANSFORMS}')
        self.inner = inner
        self.hidden = int(hidden)
        self.tensor_hidden = int(tensor_hidden)
        self.n_invariants = int(n_invariants)
        self.output_layout = tuple((str(n), int(w)) for n, w in output_layout)
        self.transform = transform
        if transform == 'layernorm':
            norm = nn.LayerNorm(self.n_invariants * self.tensor_hidden)  # ones / zeros: no RNG draw
            norm.forward = MethodType(forward_fp64, norm)  # base-model LayerNorm policy (FP64-internal, same epsilon)
            self.invariant_norm = norm
        if gates:
            width = sum(w for _, w in self.output_layout)
            if width != inner.out_features:
                raise ValueError('output layout does not match the wrapped MLP')
            self.gate = nn.Parameter(torch.full((width,), float(gate_init)))  # no RNG draw
        else:
            self.register_parameter('gate', None)

    @property
    def partition(self):
        return self.inner.partition

    @partition.setter
    def partition(self, value):
        self.inner.partition = value

    @property
    def experts(self):
        return self.inner.experts

    @property
    def out_features(self):
        return self.inner.out_features

    def transform_input(self, values):
        if values.shape[1] != self.hidden + self.n_invariants * self.tensor_hidden:
            raise ValueError('unexpected update-MLP input width')
        if self.transform is None:
            return values
        scalar_part, invariants = values.split((self.hidden, self.n_invariants * self.tensor_hidden), dim=1)
        if self.transform == 'asinh':
            if self.n_invariants < 2:
                raise ValueError('asinh transform expects the |v|, |T| norm columns first')
            norms, rest = invariants.split((2 * self.tensor_hidden, (self.n_invariants - 2) * self.tensor_hidden), dim=1)
            invariants = torch.cat((torch.asinh(norms), rest), dim=1)
        else:
            invariants = self.invariant_norm(invariants)
        return torch.cat((scalar_part, invariants), dim=1)

    def forward(self, values):
        output = self.inner(self.transform_input(values))
        if self.gate is not None:
            output = output * self.gate
        return output


def _stab_block_forward(self, scalar, vector, tensor, radial, direction, quadrupole, source, destination,
                        edge_envelope, radial_output=None, csr=None, reference=None, node_message=None):
    if csr is not None or reference is not None:
        raise ValueError('The stabilized block supports the eager reference geometry only')
    if '_stab_count' in self.__dict__:
        raise RuntimeError('Neighbour count from a previous forward was never consumed')
    # Smooth neighbour count inside this block's C2 envelope (differentiable, rotation invariant).
    self.__dict__['_stab_count'] = scalar.new_zeros((scalar.shape[0],)).index_add(0, destination, edge_envelope)
    return self._stab_base_forward(scalar, vector, tensor, radial, direction, quadrupole, source, destination,
                                   edge_envelope, radial_output, None, None, node_message)


def _stab_finish_messages(self, scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t):
    count = self.__dict__.pop('_stab_count')
    if self.stab_aggregate_normalization == 'sqrt':
        scale = torch.rsqrt(1. + count)
    else:
        scale = 1. / (1. + count)
    return self._stab_base_finish_messages(scalar, vector, tensor, aggregate_s * scale[:, None],
                                           aggregate_v * scale[:, None, None], aggregate_t * scale[:, None, None])


def output_layout(block):
    hidden, tw = block.scalar_hidden, block.tensor_hidden
    if block.terminal:
        return (('s', hidden), ('vdot', tw if block.narrow_tensors else hidden), ('tdot', tw if block.narrow_tensors else hidden))
    w = tw if block.narrow_tensors else hidden
    return (('s', hidden), ('vdot', w), ('v', w), ('tdot', w), ('t', w))


def build_stab_model(config):
    values = dict(config)
    transform = values.pop('stab_invariant_transform', 'asinh')
    gates = bool(values.pop('stab_output_gates', True))
    gate_init = float(values.pop('stab_gate_init', 1.0))
    residual_scale = values.pop('stab_residual_scale', None)
    aggregate_norm = values.pop('stab_aggregate_normalization', None)
    if transform not in TRANSFORMS:
        raise ValueError(f'stab_invariant_transform must be one of {TRANSFORMS}')
    if aggregate_norm not in AGGREGATE_NORMALIZATIONS:
        raise ValueError(f'stab_aggregate_normalization must be one of {AGGREGATE_NORMALIZATIONS}')
    if not values.get('higher_invariants', False):
        raise ValueError('The stabilizer targets the five-invariant node update (higher_invariants true)')
    model = build_base(values)
    blocks = len(model.blocks)
    if residual_scale is None:
        scale = 1.0
    elif residual_scale == 'inv_sqrt_blocks':
        scale = 1.0 / math.sqrt(blocks)
    else:
        scale = float(residual_scale)
        if not 0. < scale <= 1.:
            raise ValueError('stab_residual_scale must be in (0, 1]')
    if not gates and (scale != 1.0 or gate_init != 1.0):
        raise ValueError('residual scaling needs the output gates')
    installed = []
    for index, block in enumerate(model.blocks):
        if isinstance(block.update_mlp, StabilizedUpdateMLP):
            raise ValueError('Stabilizer already installed')
        n_inv = 5 if block.higher_invariants else 2
        layout = output_layout(block)
        block.update_mlp = StabilizedUpdateMLP(block.update_mlp, block.scalar_hidden, block.tensor_hidden, n_inv, layout,
                                               transform, gates, gate_init * scale)
        if aggregate_norm is not None:
            block.stab_aggregate_normalization = aggregate_norm
            block._stab_base_forward = block.forward
            block._stab_base_finish_messages = block._finish_messages
            block.forward = MethodType(_stab_block_forward, block)
            block._finish_messages = MethodType(_stab_finish_messages, block)
        installed.append(dict(block=index, cutoff_A=block.cutoff, terminal=block.terminal, output_layout=list(layout),
                              gate_parameters=0 if not gates else block.update_mlp.gate.numel(),
                              layernorm_parameters=0 if transform != 'layernorm' else sum(p.numel() for p in block.update_mlp.invariant_norm.parameters())))
    names = [name for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    model.numerical_execution = dict(model.numerical_execution, layernorm_modules=names)
    model.stabilized_spec = dict(
        variant=VARIANT, invariant_transform=transform, output_gates=gates, gate_init=gate_init, residual_scale=residual_scale,
        effective_gate_init=gate_init * scale, aggregate_normalization=aggregate_norm, blocks=installed,
        transformed_columns='|vv| and |tv| (the two raw stream norms) of the expert-MLP input' if transform == 'asinh'
        else ('all five invariant blocks (5 x tensor_hidden) through one FP64-internal LayerNorm' if transform == 'layernorm' else 'none'),
        identity_at_init=(transform is None and (not gates or gate_init * scale == 1.0) and aggregate_norm is None),
        initialization='no global RNG draw for any stabilizer parameter (base tensors bitwise the plain model)',
        implementation='eager reference geometry; the fused MD node stage would need the same input compression')
    return model
