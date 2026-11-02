"""Rotated-frame form of the O(3) projector EFM (inference only; exact reformulation, no new parameters).

Why: the projector form carries the |m|=1 channels as 3 Cartesian components and the |m|=2 channels as 5 STF
coordinates, although both live in 2-dimensional in-plane subspaces. Every per-edge map after the rotate-in is
channel-linear per component with rotation-invariant scalar gates, so writing the components in an orthonormal edge
frame is the same function: |m|=1 needs 2 components instead of 3, |m|=2 2 instead of 5 (-22% per-edge GEMM FLOPs for
the K arms, -36% per-edge state). This module is the PyTorch reference of that form; hyff.kernels.efm_blocked
has the same function in a layout for torch.compile.

Per edge e = (j -> i) with unit direction u and the explicit frame (e_x, e_y) of edge_frame_mixing.explicit_frame:
  R (3x3) rows u, e_x, e_y                         vector frame coordinates  f = R v      (f0 = a0, f1, f2 = in-plane pair)
  D (5x5) rows = five-coordinates of the STF basis  tensor frame coordinates tt = D T     (tt0 = c0, tt1-2 = |m|=1 pair,
     B0 = sqrt(3/2)(u u^T - I/3), B1,2 = (e u^T + u e^T)/sqrt2 for e = e_x, e_y,           tt3-4 = |m|=2 pair)
     B3 = (e_x e_x^T - e_y e_y^T)/sqrt2, B4 = (e_x e_y^T + e_y e_x^T)/sqrt2
Both are orthogonal, so rotating back is R^T / D^T. The projector module's quantities are exactly these:
  a0 = u.v, c0 = sqrt(3/2) u^T T u, v_perp = f1 e_x + f2 e_y, w_perp = tt1 e_x + tt2 e_y, T_pp = tt3 B3 + tt4 B4,
  sqrt(3/2) Q(u) = B0, sqrt2 ST(p (x) u) = p_x B1 + p_y B2 for in-plane p.
The frame has a chart switch at u_z = 0 (edge_frame_mixing.explicit_frame 'auto'); the output and its derivative do
not depend on the frame.

The quadrupole argument of the module forward is not used: the tensor frame is built from u, so the whole
dependence on the edge direction (including the one the projector form routes through Q(u)) flows through u.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from hyff.models.edge_frame_mixing import SQRT2, SQRT_3_2, explicit_frame, matrix_to_tensor5


def check_module(efm):
    """The fused / rotated forms cover the O(3) projector EFM with sum aggregation (single layer or stacked layers)."""
    if efm.frame != 'projector' or efm.complex_structure:
        raise ValueError('rotated EFM covers the O(3) projector form only')
    if efm.aggregation != 'sum':
        raise ValueError('rotated EFM covers sum aggregation only')
    if getattr(efm, 'receiver_in', None) is not None:
        raise ValueError('rotated EFM does not cover the receiver input (T7)')


def frame_matrices(u):
    """(E, 3) unit directions -> R (E, 3, 3) and D (E, 5, 5), both orthogonal, rows as in the module docstring."""
    ex, ey = explicit_frame(u, 'auto')
    R = torch.stack((u, ex, ey), 1)
    eye = torch.eye(3, dtype=u.dtype, device=u.device)

    def outer(p, q):
        return p[:, :, None] * q[:, None, :]

    basis = (SQRT_3_2 * (outer(u, u) - eye / 3.),
             (outer(ex, u) + outer(u, ex)) / SQRT2,
             (outer(ey, u) + outer(u, ey)) / SQRT2,
             (outer(ex, ex) - outer(ey, ey)) / SQRT2,
             (outer(ex, ey) + outer(ey, ex)) / SQRT2)
    D = torch.stack([matrix_to_tensor5(b[..., None])[..., 0] for b in basis], 1)
    return R, D


def node_inputs(efm, normalized_scalar, vector, tensor, radial, direction, envelope):
    """Everything before the per-edge chain: node projections, radial MLP, frames, edge weights."""
    S = efm.node_scalar(normalized_scalar)
    V = T = None
    if efm.m_blocks:
        V = efm.node_vector(vector)
        T = efm.node_tensor(tensor)
    rad = efm.radial(radial)
    R, D = frame_matrices(direction)
    w = efm.message_scale * envelope
    return S, V, T, rad, R, D, w


def rotated_core(efm, S, V, T, rad, R, D, w, source, destination, n_atoms):
    """The per-edge chain in frame coordinates and the sum aggregation (reference for the fused kernel)."""
    C = efm.channels
    s = S.index_select(0, source)
    inputs = [s, rad[:, :C]]
    if efm.m_blocks:
        f = torch.einsum('eij,ejc->eic', R, V.index_select(0, source))
        tt = torch.einsum('emk,ekc->emc', D, T.index_select(0, source))
        inputs += [f[:, 0], tt[:, 0]]
    s1, a1, c1 = efm.m0(torch.cat(inputs, -1)).split(C, -1)
    gates = torch.sigmoid(efm.gate(s1)).split(C, -1)
    h = F.silu(s1)
    a = a1 * gates[0]
    c = c1 * gates[1]
    xv = xw = z = None
    if 1 in efm.m_blocks:
        X = torch.cat((f[:, 1:3], tt[:, 1:3]), -1) * rad[:, None, C:3 * C]
        yv, yw = efm.m1_a(X).split(C, -1)
        xv = yv * gates[2][:, None, :]
        xw = yw * gates[3][:, None, :]
    if 2 in efm.m_blocks:
        z = efm.m2_a(tt[:, 3:5] * rad[:, None, 3 * C:4 * C]) * gates[4][:, None, :]
    for layer in getattr(efm, 'layers', ()):
        h, a, c, xv, xw, z = layer(h, a, c, xv, xw, z)
    fv = [a[:, None]] + ([xv] if xv is not None else [])
    ft = [c[:, None]] + ([xw] if xw is not None else []) + ([z] if z is not None else [])
    fv, ft = torch.cat(fv, 1), torch.cat(ft, 1)
    v_e = torch.einsum('eic,eij->ejc', fv, R[:, :fv.shape[1]])
    t_e = torch.einsum('emc,emk->ekc', ft, D[:, :ft.shape[1]])
    agg_s = h.new_zeros((n_atoms, C)).index_add_(0, destination, h * w[:, None])
    agg_v = h.new_zeros((n_atoms, 3, C)).index_add_(0, destination, v_e * w[:, None, None])
    agg_t = h.new_zeros((n_atoms, 5, C)).index_add_(0, destination, t_e * w[:, None, None])
    return agg_s, agg_v, agg_t


def rotated_forward(efm, normalized_scalar, vector, tensor, radial, direction, quadrupole, source, destination, envelope,
                    core=None):
    """Drop-in for the module forward on a fixed edge set (all given edges; zero-envelope edges contribute zero).

    `core` defaults to the PyTorch reference rotated_core; hyff.kernels.efm_blocked.blocked_core has the same signature.
    """
    check_module(efm)
    S, V, T, rad, R, D, w = node_inputs(efm, normalized_scalar, vector, tensor, radial, direction, envelope)
    agg_s, agg_v, agg_t = (core or rotated_core)(efm, S, V, T, rad, R, D, w, source, destination,
                                                 normalized_scalar.shape[0])
    return efm.exit_scalar(agg_s), efm.exit_vector(agg_v), efm.exit_tensor(agg_t)
