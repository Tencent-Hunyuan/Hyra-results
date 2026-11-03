"""Edge-frame mixing (EFM): a bounded Cartesian analogue of an SO(2) edge-frame convolution block.

`build_model` builds the base StatePotential chain (hyff.models.base_model ->
hyff.models.state_model -> hyff.models.element_model -> hyff.models.l2_potential)
from the plain model keys and then installs one EdgeFrameMixing module per
selected block. Every EFM parameter is created AFTER the base build, so every base
tensor is bitwise the initialization of the plain model with the same keys, and
the three exit projections are zero, so the initial function equals the base model.

SO(2) edge-frame blocks (as in eSCN-style models) rotate the sender's l<=lmax
node tensor into the edge frame keeping only |m| <= mmax, multiply by radial
features, apply linear layers that are block-diagonal over |m| (m=0 dense over l,
|m|>0 a complex-structured 2x2 coupling [[W_u, -W_v], [W_v, W_u]] of the (-m, +m)
pair) with gated activations between them, rotate back and aggregate. EFM writes
the same frame components as Cartesian projections, so it needs no Wigner-D
matrices and no explicit frame axes.

What EFM does per active edge e = (j -> i), with C = efm_channels, u = unit
direction, P = I - u u^T the projector onto the plane normal to u:

  node projections (per atom, then gathered):  s = W_s LN(scalar) + b,  v = W_v vector,  T = W_T tensor
  frame components (exact for l <= 2; the Wigner-D of the edge frame written as Cartesian projections):
      l=0 m=0     s                                     (C)
      l=1 m=0     a0 = u . v                            (C)
      l=1 |m|=1   v_perp = P v = v - a0 u               (3, C)  the (v_x, v_y) pair as an in-plane Cartesian vector
      l=2 m=0     c0 = sqrt(3/2) u^T T u                (C)     orthonormal coordinate of the m=0 l=2 component
      l=2 |m|=1   w_perp = sqrt(2) P T u                (3, C)  the (sqrt2 T_xz, sqrt2 T_yz) pair as an in-plane vector
      l=2 |m|=2   T_pp = T - (3/2)(u^T T u) Q(u) - (w_perp u^T + u w_perp^T)/sqrt2    (5, C)  dropped for mmax 1
  radial: r = MLP(rbf) split into r0 (C, additive input of the m=0 block) and r1 (2C, multiplicative on the |m|=1 inputs)
  m=0 block (dense over the stacked channels, one gated layer):
      [s', a', c'] = W0 [s, r0, a0, c0] + b0
      s'' = SiLU(s'),   g = sigmoid(W_g s' + b_g),   a'' = g_a a',   c'' = g_c c'
  |m|=1 block (dense over the stacked (v_perp, w_perp) channels, shared by the in-plane components):
      X = [v_perp r1_v, w_perp r1_w]                     (3, 2C)
      Y = W_a X                    (+ W_b (u x X) when efm_complex; u x . is the complex unit i of the SO(2) block, see below)
      [v'_perp, p'] = Y,   v'_perp *= g_1v,   p' *= g_1w
  rotate back (nothing to rotate: everything is already in lab coordinates):
      s_e = s'',   v_e = a'' u + v'_perp,   T_e = c'' sqrt(3/2) Q(u) + sqrt(2) ST(p' (x) u)  [+ T'_pp for mmax 2]
  aggregation:  A_i = sum_e w_e (s_e, v_e, T_e),  w_e = 0.25 env_e  ('sum', the base weight)  or  env_e / sqrt(1 + sum_e env_e)  ('degree')
  exit (zero-initialised, bias-free):  scalar += W_xs A_s,  vector += W_xv A_v,  tensor += W_xT A_T,  before the node update.

Parity. In the edge frame the SO(2) coupling W_v acts as the complex unit on the
(-m, +m) pair, i.e. a rotation by 90 degrees about u: i . X = u x X. The cross
product with the polar vector u is a pseudo-vector operation, so with W_v != 0 the
block is SO(3)- but not O(3)-equivariant (mirror-image structures get different
energies). The default EFM (efm_complex false) keeps only W_u and is exactly
O(3)-equivariant; efm_complex true reproduces the complex SO(2) structure (SO(3) only).

Roll. The projector form never constructs x, y axes, so it is trivially exact
under any roll and smooth at every direction (no pole). The module also carries
an explicit-frame implementation (frame='explicit': two rotation charts, a hard
switch at u_z = 0, an optional roll angle) that computes the same quantities from
(e_x, e_y) components; both forms agree to FP64 rounding for every roll and chart
choice. The 'explicit' form is for verification only; configs use 'projector'.
"""
from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn

from hyff.models.base_model import build_model as build_base
from hyff.models.l2_potential import _linear, _tensor_dot_direction, _vector_tensor_product

VARIANT = 'edge_frame_mixing_cartesian_l2'
SQRT2 = math.sqrt(2.)
SQRT6 = math.sqrt(6.)
SQRT_3_2 = math.sqrt(1.5)
FRAMES = ('projector', 'explicit')
AGGREGATIONS = ('sum', 'degree')


# --------------------------------------------------------------------------------------------------
# Cartesian helpers (five orthonormal l=2 coordinates as in hyff.models.l2_potential._quadrupole)
# --------------------------------------------------------------------------------------------------
def tensor5_to_matrix(t):
    """(E, 5, C) orthonormal STF coordinates -> (E, 3, 3, C) symmetric traceless matrices."""
    a, b, c, d, e = t.unbind(1)
    xx = a / SQRT2 + e / SQRT6
    yy = -a / SQRT2 + e / SQRT6
    zz = -2. * e / SQRT6
    xy, xz, yz = b / SQRT2, c / SQRT2, d / SQRT2
    return torch.stack((torch.stack((xx, xy, xz), 1), torch.stack((xy, yy, yz), 1), torch.stack((xz, yz, zz), 1)), 1)


def matrix_to_tensor5(m):
    """(E, 3, 3, C) matrices -> (E, 5, C) coordinates of their symmetric traceless part."""
    xx, yy, zz = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    xy = 0.5 * (m[:, 0, 1] + m[:, 1, 0])
    xz = 0.5 * (m[:, 0, 2] + m[:, 2, 0])
    yz = 0.5 * (m[:, 1, 2] + m[:, 2, 1])
    return torch.stack(((xx - yy) / SQRT2, SQRT2 * xy, SQRT2 * xz, SQRT2 * yz, (xx + yy - 2. * zz) / SQRT6), 1)


def cross_matrix(u):
    """(E, 3) -> (E, 3, 3) with K v = u x v."""
    x, y, z = u.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((torch.stack((o, -z, y), -1), torch.stack((z, o, -x), -1), torch.stack((-y, x, o), -1)), -2)


def explicit_frame(u, chart='auto', roll=None):
    """Orthonormal completion (e_x, e_y) of the edge direction u with e_y = u x e_x.

    Two rotation charts (+Z / -Z charts written as Rodrigues rotations taking
    z to u): 'pos' is regular except at u = -z, 'neg' except at u = +z, 'auto' switches
    at u_z = 0 (the denominators are floored at 1e-12 so the unselected branch stays
    finite for autograd exactly at its pole; away from the pole the chart is unchanged).
    `roll` rotates the pair about u: a scalar, an (E,) tensor, or a callable of u.
    The EFM output does not depend on the chart or the roll; this function exists to
    check that statement numerically.
    """
    x, y, z = u.unbind(-1)
    one = torch.ones_like(z)
    # +Z chart: R = I + [k]x + [k]x^2 / (1 + z), k = z x u
    dp = (1. + z).clamp_min(1e-12)
    ex_p = torch.stack((one - x * x / dp, -x * y / dp, -x), -1)
    ey_p = torch.stack((-x * y / dp, one - y * y / dp, -y), -1)
    # -Z chart: R' = R_pos(-u) R_x(pi); e_x' = R_pos(-u) e_x, e_y' = -R_pos(-u) e_y evaluated at -u
    dn = (1. - z).clamp_min(1e-12)
    ex_n = torch.stack((one - x * x / dn, -x * y / dn, x), -1)
    ey_n = -torch.stack((-x * y / dn, one - y * y / dn, y), -1)
    if chart == 'pos':
        ex, ey = ex_p, ey_p
    elif chart == 'neg':
        ex, ey = ex_n, ey_n
    elif chart == 'auto':
        pick = (z >= 0.)[:, None]
        ex, ey = torch.where(pick, ex_p, ex_n), torch.where(pick, ey_p, ey_n)
    else:
        raise ValueError(f'unknown chart {chart!r}')
    if roll is not None:
        gamma = roll(u) if callable(roll) else torch.as_tensor(roll, dtype=u.dtype, device=u.device)
        if gamma.ndim == 0:
            gamma = gamma.expand(u.shape[0])
        cg, sg = torch.cos(gamma)[:, None], torch.sin(gamma)[:, None]
        ex, ey = cg * ex + sg * ey, -sg * ex + cg * ey
    return ex, ey


def _zero_linear(in_features, out_features):
    layer = nn.Linear(in_features, out_features, bias=False, device='meta')
    layer.weight = nn.Parameter(torch.zeros(out_features, in_features))  # no global RNG draw
    return layer


# --------------------------------------------------------------------------------------------------
# The module
# --------------------------------------------------------------------------------------------------
class EdgeFrameMixing(nn.Module):
    """One gated edge-frame mixing layer (l <= 2, |m| <= mmax) for one interaction block."""

    def __init__(self, scalar_hidden, tensor_hidden, n_rbf, channels, mmax, input_is_scalar, message_scale,
                 aggregation='sum', complex_structure=False, frame='projector'):
        super().__init__()
        C = int(channels)
        if C < 1 or int(mmax) not in (1, 2):
            raise ValueError('efm_channels must be positive and efm_mmax 1 or 2')
        if aggregation not in AGGREGATIONS or frame not in FRAMES:
            raise ValueError(f'efm_aggregation in {AGGREGATIONS}, efm_frame in {FRAMES}')
        self.channels, self.mmax = C, int(mmax)
        self.input_is_scalar = bool(input_is_scalar)
        self.message_scale = float(message_scale)
        self.aggregation, self.frame = aggregation, frame
        self.complex_structure = bool(complex_structure)
        # explicit-frame verification controls (ignored by the projector form)
        self.chart, self.roll = 'auto', None
        # |m| > 0 blocks exist only when the sender carries vectors/tensors (block 0 is scalar-source:
        # its v and T inputs are exactly zero, so the |m|>0 blocks would output exactly zero).
        self.m_blocks = () if self.input_is_scalar else ((1, 2) if self.mmax == 2 else (1,))
        self.node_scalar = _linear(scalar_hidden, C)
        if not self.input_is_scalar:
            self.node_vector = _linear(tensor_hidden, C, bias=False)
            self.node_tensor = _linear(tensor_hidden, C, bias=False)
        radial_out = C + (2 * C if 1 in self.m_blocks else 0) + (C if 2 in self.m_blocks else 0)
        self.radial = nn.Sequential(_linear(n_rbf, C), nn.SiLU(), _linear(C, radial_out))
        self.m0 = _linear(2 * C + (0 if self.input_is_scalar else 2 * C), 3 * C)
        self.n_gates = 2 + (2 if 1 in self.m_blocks else 0) + (1 if 2 in self.m_blocks else 0)
        self.gate = _linear(C, self.n_gates * C)
        with torch.no_grad():
            self.gate.weight.mul_(0.1)  # gates start near 1/2
        if 1 in self.m_blocks:
            self.m1_a = _linear(2 * C, 2 * C, bias=False)
            self.m1_b = _linear(2 * C, 2 * C, bias=False) if self.complex_structure else None
        if 2 in self.m_blocks:
            self.m2_a = _linear(C, C, bias=False)
            self.m2_b = _linear(C, C, bias=False) if self.complex_structure else None
        if self.complex_structure:
            with torch.no_grad():  # the doubled parameter count of the 2x2 block is compensated by 1/sqrt2
                for layer in (getattr(self, 'm1_a', None), getattr(self, 'm1_b', None),
                              getattr(self, 'm2_a', None), getattr(self, 'm2_b', None)):
                    if layer is not None:
                        layer.weight.mul_(1. / SQRT2)
        self.exit_scalar = _zero_linear(C, scalar_hidden)
        self.exit_vector = _zero_linear(C, tensor_hidden)
        self.exit_tensor = _zero_linear(C, tensor_hidden)

    # ---- frame algebra -----------------------------------------------------------------------
    def _times_i(self, X, u, ex, m):
        """The complex unit of the |m| block: rotation by 90/m degrees of the pair about u."""
        if self.frame == 'projector':
            if m == 1:
                return torch.cross(u[:, :, None].expand_as(X), X, dim=1)
            M = tensor5_to_matrix(X)
            return matrix_to_tensor5(torch.einsum('eij,ejkc->eikc', cross_matrix(u), M))
        return torch.stack((-X[:, 1], X[:, 0]), 1)

    def _to_frame(self, v, t, u, quad, ex, ey):
        a0 = (u[:, :, None] * v).sum(1)
        w, tzz = _tensor_dot_direction(t, u)
        c0 = SQRT_3_2 * tzz
        out = dict(a0=a0, c0=c0)
        if self.frame == 'projector':
            v_perp = v - a0[:, None, :] * u[:, :, None]
            w_perp = SQRT2 * (w - tzz[:, None, :] * u[:, :, None])
            out['X1'] = torch.cat((v_perp, w_perp), -1)                       # (E, 3, 2C) in-plane vectors
            if 2 in self.m_blocks:
                out['Z2'] = (t - 1.5 * tzz[:, None, :] * quad[:, :, None]
                             - SQRT2 * _vector_tensor_product(w_perp, u)[0])  # (E, 5, C) in-plane STF part
        else:
            vx, vy = (ex[:, :, None] * v).sum(1), (ey[:, :, None] * v).sum(1)
            wx, wy = SQRT2 * (ex[:, :, None] * w).sum(1), SQRT2 * (ey[:, :, None] * w).sum(1)
            out['X1'] = torch.stack((torch.cat((vx, wx), -1), torch.cat((vy, wy), -1)), 1)   # (E, 2, 2C)
            if 2 in self.m_blocks:
                M = tensor5_to_matrix(t)
                txx = torch.einsum('ei,eijc,ej->ec', ex, M, ex)
                tyy = torch.einsum('ei,eijc,ej->ec', ey, M, ey)
                txy = torch.einsum('ei,eijc,ej->ec', ex, M, ey)
                out['Z2'] = torch.stack(((txx - tyy) / SQRT2, SQRT2 * txy), 1)               # (E, 2, C)
        return out

    def _plane_vector(self, Y, ex, ey):
        """|m|=1 output pair -> in-plane Cartesian vector (E, 3, C)."""
        if self.frame == 'projector':
            return Y
        return Y[:, 0, None, :] * ex[:, :, None] + Y[:, 1, None, :] * ey[:, :, None]

    def _plane_tensor(self, Y2, ex, ey):
        """|m|=2 output pair -> in-plane STF tensor in the five coordinates (E, 5, C)."""
        if self.frame == 'projector':
            return Y2
        B1 = matrix_to_tensor5(((ex[:, :, None] * ex[:, None, :] - ey[:, :, None] * ey[:, None, :]) / SQRT2)[..., None])[..., 0]
        B2 = matrix_to_tensor5(((ex[:, :, None] * ey[:, None, :] + ey[:, :, None] * ex[:, None, :]) / SQRT2)[..., None])[..., 0]
        return Y2[:, 0, None, :] * B1[:, :, None] + Y2[:, 1, None, :] * B2[:, :, None]

    # ---- forward -----------------------------------------------------------------------------
    def forward(self, normalized_scalar, vector, tensor, radial, direction, quadrupole, source, destination, envelope):
        n_atoms = normalized_scalar.shape[0]
        C = self.channels
        # Only edges inside this block's C2 envelope contribute; at zero envelope the term and its
        # first two radial derivatives vanish, so the selection is exact (same argument as the angular branch).
        active = torch.nonzero(envelope != 0, as_tuple=False).squeeze(-1)
        src = source.index_select(0, active)
        dst = destination.index_select(0, active)
        u = direction.index_select(0, active)
        quad = quadrupole.index_select(0, active)
        env = envelope.index_select(0, active)
        rad = self.radial(radial.index_select(0, active))
        r0, rad = rad[:, :C], rad[:, C:]
        s = self.node_scalar(normalized_scalar).index_select(0, src)
        inputs = [s, r0]
        ex = ey = None
        if self.m_blocks:
            if self.frame == 'explicit':
                ex, ey = explicit_frame(u, self.chart, self.roll)
            comps = self._to_frame(self.node_vector(vector).index_select(0, src),
                                   self.node_tensor(tensor).index_select(0, src), u, quad, ex, ey)
            inputs += [comps['a0'], comps['c0']]
        s1, a1, c1 = self.m0(torch.cat(inputs, -1)).split(C, -1)
        gates = torch.sigmoid(self.gate(s1)).split(C, -1)
        s_e = F.silu(s1)
        v_e = (a1 * gates[0])[:, None, :] * u[:, :, None]
        t_e = (c1 * gates[1])[:, None, :] * (SQRT_3_2 * quad)[:, :, None]
        if 1 in self.m_blocks:
            X = comps['X1'] * rad[:, None, :2 * C]
            Y = self.m1_a(X)
            if self.m1_b is not None:
                Y = Y + self.m1_b(self._times_i(X, u, ex, 1))
            Yv, Yw = Y.split(C, -1)
            v_e = v_e + self._plane_vector(Yv * gates[2][:, None, :], ex, ey)
            p = self._plane_vector(Yw * gates[3][:, None, :], ex, ey)
            t_e = t_e + SQRT2 * _vector_tensor_product(p, u)[0]
        if 2 in self.m_blocks:
            Z = comps['Z2'] * rad[:, None, 2 * C:3 * C]
            Y2 = self.m2_a(Z)
            if self.m2_b is not None:
                Y2 = Y2 + self.m2_b(self._times_i(Z, u, ex, 2))
            t_e = t_e + self._plane_tensor(Y2 * gates[4][:, None, :], ex, ey)
        if self.aggregation == 'sum':
            weight = self.message_scale * env
        else:
            degree = env.new_zeros((n_atoms,)).index_add_(0, dst, env)
            weight = env / torch.sqrt(1. + degree.index_select(0, dst))
        agg_s = s.new_zeros((n_atoms, C)).index_add_(0, dst, s_e * weight[:, None])
        agg_v = s.new_zeros((n_atoms, 3, C)).index_add_(0, dst, v_e * weight[:, None, None])
        agg_t = s.new_zeros((n_atoms, 5, C)).index_add_(0, dst, t_e * weight[:, None, None])
        return self.exit_scalar(agg_s), self.exit_vector(agg_v), self.exit_tensor(agg_t)


# --------------------------------------------------------------------------------------------------
# Installation into the L2Interaction blocks
# --------------------------------------------------------------------------------------------------
def _efm_block_forward(self, scalar, vector, tensor, radial, direction, quadrupole, source, destination,
                       edge_envelope, radial_output=None, csr=None, reference=None, node_message=None):
    if csr is not None or reference is not None:
        raise ValueError('Edge-frame mixing supports the eager reference geometry only')
    normalized = self.scalar_norm(scalar)
    if node_message is None:
        node_message = self.scalar_message(normalized)
    if '_efm_term' in self.__dict__:
        raise RuntimeError('EFM term from a previous forward was never consumed')
    self.__dict__['_efm_term'] = self.efm(normalized, vector, tensor, radial, direction, quadrupole,
                                          source, destination, edge_envelope)
    return self._efm_base_forward(scalar, vector, tensor, radial, direction, quadrupole, source, destination,
                                  edge_envelope, radial_output, None, None, node_message)


def _efm_node_update(self, scalar, vector, tensor, projections=None):
    # Called by _finish_messages with the aggregated messages already added (after the edge_*_out
    # projections); the EFM exits are added here, so they enter the invariants of the node update in
    # every block, including the terminal one.
    ds, dv, dt = self.__dict__.pop('_efm_term')
    return self._efm_base_node_update(scalar + ds, vector + dv, tensor + dt, projections)


def build_model(config):
    values = dict(config)
    channels = int(values.pop('efm_channels', 16))
    mmax = int(values.pop('efm_mmax', 1))
    blocks = values.pop('efm_blocks', None)
    aggregation = values.pop('efm_aggregation', 'sum')
    complex_structure = bool(values.pop('efm_complex', False))
    frame = values.pop('efm_frame', 'projector')
    model = build_base(values)
    n_blocks = len(model.blocks)
    blocks = tuple(range(n_blocks)) if blocks is None else tuple(int(b) for b in blocks)
    if not blocks or len(set(blocks)) != len(blocks) or any(not 0 <= b < n_blocks for b in blocks):
        raise ValueError('efm_blocks must be distinct valid block indices')
    n_rbf = model.config.n_rbf
    installed = []
    for index in blocks:
        block = model.blocks[index]
        if hasattr(block, 'efm'):
            raise ValueError('EFM already installed')
        block.efm = EdgeFrameMixing(block.scalar_hidden, block.tensor_hidden, n_rbf, channels, mmax,
                                    block.input_is_scalar, block.message_scale, aggregation, complex_structure, frame)
        block._efm_base_forward = block.forward
        block._efm_base_node_update = block._node_update
        block.forward = MethodType(_efm_block_forward, block)
        block._node_update = MethodType(_efm_node_update, block)
        installed.append(dict(block=index, cutoff_A=block.cutoff, scalar_source=block.input_is_scalar,
                              m_blocks=list(block.efm.m_blocks), parameters=sum(p.numel() for p in block.efm.parameters()),
                              exit_parameters=sum(p.numel() for n, p in block.efm.named_parameters() if n.startswith('exit_'))))
    model.efm = dict(variant=VARIANT, blocks=installed, channels=channels, mmax=mmax, aggregation=aggregation,
                     complex_structure=complex_structure, frame=frame,
                     equivariance='O(3) (rotations, reflections, translations, permutations)' if not complex_structure
                     else 'SO(3) only: the W_v (u x) coupling is parity-odd',
                     initialization='exit projections zero (initial function equals the base model); interior xavier; gate weights xavier x 0.1',
                     injection='scalar/vector/tensor state after the aggregated messages, before the node update of the same block',
                     implementation='eager reference geometry; per-block active-edge selection; no fused MD backend')
    return model


def efm_modules(model):
    return [(index, block.efm) for index, block in enumerate(model.blocks) if hasattr(block, 'efm')]


# --------------------------------------------------------------------------------------------------
# Analytic per-edge cost (multiply-accumulates per ACTIVE edge of one block; one MAC = 2 flops)
# --------------------------------------------------------------------------------------------------
def base_message_macs(edge_hidden, n_rbf, radial_hidden, cross_order=True):
    """The 7-path (5-path for the scalar-source block) message of L2Interaction.forward."""
    M = edge_hidden
    if cross_order:
        geometric = dict(radial_and_envelope_product=14 * M, tensor_dot_direction=20 * M, vector_tensor_product=15 * M,
                         scalar_closures=2 * M, edge_vector=9 * M, edge_tensor=15 * M)
        paths = 7
        gather, scatter = 15 * M, 9 * M
    else:
        geometric = dict(radial_and_envelope_product=6 * M, edge_vector=3 * M, edge_tensor=5 * M)
        paths = 3
        gather, scatter = 3 * M, 9 * M
    radial_mlp = n_rbf * radial_hidden + radial_hidden * paths * M
    return dict(geometric=geometric, geometric_total=sum(geometric.values()), radial_mlp=radial_mlp,
                total_training=sum(geometric.values()) + radial_mlp, total_md_radial_tabulated=sum(geometric.values()),
                gather_floats=gather, scatter_floats=scatter)


def efm_macs(channels, n_rbf, mmax=1, input_is_scalar=False, complex_structure=False):
    C = channels
    m1 = not input_is_scalar
    m2 = (not input_is_scalar) and mmax == 2
    radial_out = C + (2 * C if m1 else 0) + (C if m2 else 0)
    n_gates = 2 + (2 if m1 else 0) + (1 if m2 else 0)
    parts = dict(
        radial_mlp=n_rbf * C + C * radial_out,
        frame_components=(33 * C if m1 else 0) + (25 * C if m2 else 0),
        radial_weights=(6 * C if m1 else 0) + (5 * C if m2 else 0),
        m0_linear=(2 * C + (2 * C if m1 else 0)) * 3 * C,
        gates=n_gates * C * C + n_gates * C + 2 * C + (6 * C if m1 else 0) + (5 * C if m2 else 0),
        m1_linear=(3 * (2 * C) ** 2 * (2 if complex_structure else 1) + (18 * C if complex_structure else 0)) if m1 else 0,
        m2_linear=(5 * C * C * (2 if complex_structure else 1) + (30 * C if complex_structure else 0)) if m2 else 0,
        rotate_back=8 * C + (23 * C if m1 else 0) + (5 * C if m2 else 0),
        weight_and_scatter=18 * C,
    )
    total = sum(parts.values())
    return dict(parts=parts, total_training=total, total_md_radial_tabulated=total - parts['radial_mlp'],
                geometric_total=total - parts['radial_mlp'], radial_mlp=parts['radial_mlp'],
                gather_floats=(9 * C if m1 else C), scatter_floats=9 * C)


def analytic_cost(model_dict):
    """Per-active-edge MACs of the base message and of EFM for every block of a model dict."""
    n_blocks = int(model_dict['interactions'])
    blocks = model_dict.get('efm_blocks') or list(range(n_blocks))
    scalar_source = set(model_dict.get('scalar_source_blocks') or ()) | {0}
    rows = []
    for index in range(n_blocks):
        base = base_message_macs(model_dict['edge_hidden'], model_dict['n_rbf'], model_dict['radial_hidden'],
                                 cross_order=index not in scalar_source)
        row = dict(block=index, cutoff_A=model_dict['layer_cutoffs'][index], base=base)
        if index in blocks:
            row['efm'] = efm_macs(model_dict.get('efm_channels', 16), model_dict['n_rbf'], model_dict.get('efm_mmax', 1),
                                  index in scalar_source, bool(model_dict.get('efm_complex', False)))
            row['efm_over_base_geometric'] = row['efm']['geometric_total'] / base['geometric_total']
            row['efm_over_base_training'] = row['efm']['total_training'] / base['total_training']
            row['traffic_over_base'] = (row['efm']['gather_floats'] + row['efm']['scatter_floats']) / (base['gather_floats'] + base['scatter_floats'])
        rows.append(row)
    return rows
