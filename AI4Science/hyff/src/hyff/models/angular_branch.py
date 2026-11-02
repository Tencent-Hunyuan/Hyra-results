"""Optional per-block angular-moment and state-FiLM components of the EFM backbone.

Both constructors build the base StatePotential chain (hyff.models.base_model ->
hyff.models.state_model -> hyff.models.element_model -> hyff.models.l2_potential)
from the plain model keys and then add one component whose parameters are created
AFTER the base build, so every base tensor is bitwise the initialization of the
plain model with the same model keys.

1. build_angular_model: l=3 and l=4 angular moment powers in EVERY interaction
   block. Each block owns its edge coefficients (a node projection of the
   block's normalized scalar times a small radial MLP of the master RBF, gated by
   the block's own C2 envelope and message scale) and its own projection of the
   2 x channels power invariants into the scalar state, added BEFORE the node
   update so the element experts of the same block process the invariants. The
   projection starts small and nonzero. The real harmonics are written out
   explicitly (no e3nn dependency) with the 'norm' normalization
   (sum_m Y_lm^2 = 1). Eager reference geometry only; no fused MD backend.

2. build_film_model: per-block FiLM of the message-input LayerNorm and the
   update-input LayerNorm by the intensive state features [Q/N, (M-1)/N]
   (the same two features hyff.models.state_model injects into the initial scalar):
   y = LN(x) * (1 + gamma(s)) + beta(s), gamma/beta = Linear(2 -> hidden) with
   zero weights, no bias, so the initial function equals the base model and the
   modulation is exactly inert wherever Q = 0 and M = 1.
"""
from __future__ import annotations

import math
from types import MethodType

import torch
from torch import nn

from hyff.models.base_model import build_model as build_base
from hyff.models.l2_potential import _linear

ANGULAR_VARIANT = 'all_block_l3l4_moment_powers'
FILM_VARIANT = 'per_block_state_film'

_SQ10_4 = math.sqrt(10.) / 4.
_SQ15 = math.sqrt(15.)
_SQ15_2 = math.sqrt(15.) / 2.
_SQ6_4 = math.sqrt(6.) / 4.
_SQ35_2 = math.sqrt(35.) / 2.
_SQ35_8 = math.sqrt(35.) / 8.
_SQ70_4 = math.sqrt(70.) / 4.
_SQ5_2 = math.sqrt(5.) / 2.
_SQ5_4 = math.sqrt(5.) / 4.
_SQ10_4b = math.sqrt(10.) / 4.


def harmonics_l3(u):
    """Seven real l=3 harmonics of unit vectors, 'norm' normalization (sum of squares 1)."""
    x, y, z = u.unbind(-1)
    x2, y2, z2 = x * x, y * y, z * z
    return torch.stack((
        _SQ10_4 * y * (3. * x2 - y2),
        _SQ15 * x * y * z,
        _SQ6_4 * y * (5. * z2 - 1.),
        0.5 * z * (5. * z2 - 3.),
        _SQ6_4 * x * (5. * z2 - 1.),
        _SQ15_2 * (x2 - y2) * z,
        _SQ10_4 * x * (x2 - 3. * y2)), dim=-1)


def harmonics_l4(u):
    """Nine real l=4 harmonics of unit vectors, 'norm' normalization (sum of squares 1)."""
    x, y, z = u.unbind(-1)
    x2, y2, z2 = x * x, y * y, z * z
    return torch.stack((
        _SQ35_2 * x * y * (x2 - y2),
        _SQ70_4 * y * z * (3. * x2 - y2),
        _SQ5_2 * x * y * (7. * z2 - 1.),
        _SQ10_4b * y * z * (7. * z2 - 3.),
        0.125 * (35. * z2 * z2 - 30. * z2 + 3.),
        _SQ10_4b * x * z * (7. * z2 - 3.),
        _SQ5_4 * (x2 - y2) * (7. * z2 - 1.),
        _SQ70_4 * x * z * (x2 - 3. * y2),
        _SQ35_8 * (x2 * x2 - 6. * x2 * y2 + y2 * y2)), dim=-1)


HARMONICS = {3: harmonics_l3, 4: harmonics_l4}


def legendre(degree, t):
    """P_l(t) for the addition-theorem check: sum_m Y_lm(u) Y_lm(v) = P_l(u . v)."""
    if degree == 3:
        return 0.5 * (5. * t ** 3 - 3. * t)
    if degree == 4:
        return 0.125 * (35. * t ** 4 - 30. * t ** 2 + 3.)
    raise ValueError(degree)


class AngularMomentBranch(nn.Module):
    """Per-block l=3/l=4 moment powers with the block's own edge coefficients.

    coefficient_e = node(LN(scalar))[source_e] * radial(rbf_e) * message_scale * envelope_e
    A_l[i] = sum_{e -> i} coefficient_e (x) Y_l(u_e)            (linear in the edge count)
    feature_l[i, c] = p / (sqrt(1 + p) + 1),  p = sum_m A_l[i, m, c]^2  (bounded, smooth)
    output = projection(cat_l feature_l)  added to the scalar state before the node update.
    """

    def __init__(self, hidden, n_rbf, channels, degrees, init_scale, message_scale):
        super().__init__()
        self.degrees = tuple(int(d) for d in degrees)
        if any(d not in HARMONICS for d in self.degrees) or len(set(self.degrees)) != len(self.degrees):
            raise ValueError('Angular degrees must be distinct and in {3, 4}')
        self.channels = int(channels)
        self.message_scale = float(message_scale)
        self.node = _linear(hidden, self.channels)
        self.radial = nn.Sequential(_linear(n_rbf, self.channels), nn.SiLU(), _linear(self.channels, self.channels))
        self.projection = _linear(len(self.degrees) * self.channels, hidden, bias=False)
        with torch.no_grad():
            self.projection.weight.mul_(float(init_scale))

    def forward(self, normalized_scalar, radial, direction, source, destination, envelope):
        n_atoms = normalized_scalar.shape[0]
        # Only edges inside this block's C2 envelope contribute; at zero envelope the
        # branch and its first two radial derivatives vanish, so the selection is exact.
        active = torch.nonzero(envelope != 0, as_tuple=False).squeeze(-1)
        src = source.index_select(0, active)
        dst = destination.index_select(0, active)
        coefficient = (self.node(normalized_scalar).index_select(0, src)
                       * self.radial(radial.index_select(0, active))
                       * (self.message_scale * envelope.index_select(0, active))[:, None])
        u = direction.index_select(0, active)
        features = []
        for degree in self.degrees:
            harmonics = HARMONICS[degree](u)
            aggregate = coefficient.new_zeros((n_atoms, 2 * degree + 1, self.channels))
            aggregate.index_add_(0, dst, harmonics[:, :, None] * coefficient[:, None, :])
            power = aggregate.square().sum(1)
            features.append(power / (torch.sqrt(1. + power) + 1.))
        return self.projection(torch.cat(features, dim=-1))


def _angular_block_forward(self, scalar, vector, tensor, radial, direction, quadrupole,
                           source, destination, edge_envelope, radial_output=None,
                           csr=None, reference=None, node_message=None):
    if csr is not None or reference is not None:
        raise ValueError('The angular branch supports the eager reference geometry only')
    normalized = self.scalar_norm(scalar)
    if node_message is None:
        node_message = self.scalar_message(normalized)
    if '_angular_term' in self.__dict__:
        raise RuntimeError('Angular term from a previous forward was never consumed')
    self.__dict__['_angular_term'] = self.angular(normalized, radial, direction, source, destination, edge_envelope)
    return self._base_forward(scalar, vector, tensor, radial, direction, quadrupole, source, destination,
                                 edge_envelope, radial_output, None, None, node_message)


def _angular_finish_messages(self, scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t):
    term = self.__dict__.pop('_angular_term')
    return self._angular_base_finish_messages(scalar + term, vector, tensor, aggregate_s, aggregate_v, aggregate_t)


def build_angular_model(config):
    values = dict(config)
    channels = int(values.pop('angular_channels', 16))
    degrees = tuple(values.pop('angular_degrees', (3, 4)))
    init_scale = float(values.pop('angular_init_scale', 0.1))
    if channels < 1 or not 0. <= init_scale <= 1.:
        raise ValueError('angular_channels must be positive and angular_init_scale in [0, 1]')
    model = build_base(values)
    n_rbf = model.config.n_rbf
    installed = []
    for index, block in enumerate(model.blocks):
        if hasattr(block, 'angular'):
            raise ValueError('Angular branch already installed')
        block.angular = AngularMomentBranch(block.scalar_hidden, n_rbf, channels, degrees, init_scale, block.message_scale)
        block._base_forward = block.forward
        block._angular_base_finish_messages = block._finish_messages
        block.forward = MethodType(_angular_block_forward, block)
        block._finish_messages = MethodType(_angular_finish_messages, block)
        installed.append(dict(block=index, cutoff_A=block.cutoff, channels=channels, degrees=list(degrees),
                              parameters=sum(p.numel() for p in block.angular.parameters())))
    model.angular_spec = dict(variant=ANGULAR_VARIANT, blocks=installed, channels=channels, degrees=list(degrees),
                            projection_init='xavier_uniform x %g (nonzero: every angular tensor receives gradient at update 1)' % init_scale,
                            injection='scalar state before the node update of the same block',
                            harmonics='explicit real l=3 (7) and l=4 (9) harmonics, norm normalization, no e3nn',
                            persistent_high_degree_state=False, fixed_covalent_topology=False,
                            implementation='eager reference geometry; no fused MD backend')
    return model


class StateFiLM(nn.Module):
    """LayerNorm followed by a state-conditioned affine map, identity at zero weights or zero state."""

    def __init__(self, norm, hidden):
        super().__init__()
        if not isinstance(norm, nn.LayerNorm):
            raise TypeError('FiLM wraps an existing LayerNorm')
        self.norm = norm
        film = nn.Linear(2, 2 * hidden, bias=False, device='meta')
        film.weight = nn.Parameter(torch.zeros(2 * hidden, 2))  # no global RNG draw
        self.film = film
        self.state = None

    def forward(self, value):
        normalized = self.norm(value)
        state = self.state
        if state is None or state.shape[0] != value.shape[0]:
            raise ValueError('Bind the per-atom state features before a FiLM LayerNorm')
        gamma, beta = self.film(state.to(normalized.dtype)).chunk(2, dim=-1)
        return normalized * (1. + gamma) + beta


def state_features(batch, dtype):
    count = (batch['ptr'][1:] - batch['ptr'][:-1]).to(dtype)
    charge = batch['total_charge'].to(dtype)
    multiplicity = batch['multiplicity'].to(dtype)
    if charge.shape != count.shape or multiplicity.shape != count.shape:
        raise ValueError('Supply one total_charge and one multiplicity per system')
    if not bool(torch.isfinite(charge).all() & (count > 0).all() & (multiplicity >= 1).all()
                & (multiplicity == multiplicity.round()).all()):
        raise ValueError('State FiLM requires finite charge, positive integer multiplicity and nonempty systems')
    return torch.stack((charge / count, (multiplicity - 1.) / count), dim=-1)


def _film_forward(self, batch, **kwargs):
    state = state_features(batch, self.embedding.weight.dtype)[batch['batch']]
    for module in self._film_modules:
        module.state = state
    try:
        return self._base_forward(batch, **kwargs)
    finally:
        for module in self._film_modules:
            module.state = None


def build_film_model(config):
    values = dict(config)
    norms = tuple(values.pop('film_norms', ('scalar_norm', 'update_norm')))
    if not values.get('condition_spin', False):
        raise ValueError('State FiLM expects the spin-conditioned StatePotential (condition_spin true)')
    if not norms or any(name not in ('scalar_norm', 'update_norm') for name in norms) or len(set(norms)) != len(norms):
        raise ValueError('film_norms must be a distinct subset of scalar_norm, update_norm')
    model = build_base(values)
    modules = []
    installed = []
    for index, block in enumerate(model.blocks):
        for name in norms:
            wrapped = StateFiLM(getattr(block, name), block.scalar_hidden)
            setattr(block, name, wrapped)
            modules.append(wrapped)
            installed.append(dict(block=index, norm=name, parameters=wrapped.film.weight.numel()))
    model.__dict__['_film_modules'] = modules
    model._base_forward = model.forward
    model.forward = MethodType(_film_forward, model)
    # The stable-LayerNorm policy record lists module names; refresh it for the wrapped norms.
    names = [name for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)]
    model.numerical_execution = dict(model.numerical_execution, layernorm_modules=names)
    model.film_spec = dict(variant=FILM_VARIANT, norms=list(norms), blocks=installed,
                         state_features=['Q/N', '(M-1)/N'], initialization='zero weights (identity at start)',
                         inert_when='total_charge == 0 and multiplicity == 1 for every system in the batch',
                         implementation='per-atom affine modulation; no fused MD backend, adapter extension needed')
    return model
