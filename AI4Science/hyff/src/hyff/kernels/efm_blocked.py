"""Rotated-frame EFM core with per-|m| contiguous blocks and explicit elementwise frame rotations (for torch.compile).

Same function as hyff.kernels.efm_rotated.rotated_core (and hence the projector module). Differences are layout only:
  - frame rotations are written as broadcast multiply-adds over the 3 / 5 components (the einsum form lowers to
    batched 3x3 / 5x5 matmuls with batch E, which makes the compiled rotated form slower than the projector form);
  - the |m|=1 state is one tensor X1 = [xv | xw] of shape (E, 2, 2C), so every stacked layer's m1 map is one plain GEMM
    on a contiguous (2E, 2C) view and the two gates act as one (E, 2C) broadcast; |m|=2 is Z (E, 2, C).
Stacked layer k >= 2 (EdgeFrameLayer): [s2 a2 c2] = W0 [h a c] + b0, g = sigmoid(Wg s2 + bg),
  h += SiLU(s2), a += g_a a2, c += g_c c2, X1 += W1(X1) [g_v g_w], Z += W2(Z) g_z.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def blocked_core(efm, S, V, T, rad, R, D, w, source, destination, n_atoms):
    C = efm.channels
    s = S.index_select(0, source)
    inputs = [s, rad[:, :C]]
    if efm.m_blocks:
        Vs = V.index_select(0, source)
        Ts = T.index_select(0, source)
        f = [R[:, i, 0, None] * Vs[:, 0] + R[:, i, 1, None] * Vs[:, 1] + R[:, i, 2, None] * Vs[:, 2] for i in range(3)]
        tt = [sum(D[:, m, k, None] * Ts[:, k] for k in range(5)) for m in range(5)]
        inputs += [f[0], tt[0]]
    s1, a1, c1 = efm.m0(torch.cat(inputs, -1)).split(C, -1)
    gates = torch.sigmoid(efm.gate(s1)).split(C, -1)
    h = F.silu(s1)
    a = a1 * gates[0]
    c = c1 * gates[1]
    X1 = Z = None
    if 1 in efm.m_blocks:
        X = torch.stack((torch.cat((f[1], tt[1]), -1), torch.cat((f[2], tt[2]), -1)), 1) * rad[:, None, C:3 * C]
        X1 = efm.m1_a(X) * torch.cat((gates[2], gates[3]), -1)[:, None, :]
    if 2 in efm.m_blocks:
        Z = efm.m2_a(torch.stack((tt[3], tt[4]), 1) * rad[:, None, 3 * C:4 * C]) * gates[4][:, None, :]
    for layer in getattr(efm, 'layers', ()):
        s2, a2, c2 = layer.m0(torch.cat((h, a, c), -1)).split(C, -1)
        g = torch.sigmoid(layer.gate(s2)).split(C, -1)
        h = h + F.silu(s2)
        a = a + a2 * g[0]
        c = c + c2 * g[1]
        if X1 is not None:
            X1 = X1 + layer.m1(X1) * torch.cat((g[2], g[3]), -1)[:, None, :]
        if Z is not None:
            Z = Z + layer.m2(Z) * g[4][:, None, :]
    fv = [a] + ([X1[:, 0, :C], X1[:, 1, :C]] if X1 is not None else [])
    ft = [c] + ([X1[:, 0, C:], X1[:, 1, C:]] if X1 is not None else []) + ([Z[:, 0], Z[:, 1]] if Z is not None else [])
    v_e = torch.stack([sum(R[:, i, j, None] * fv[i] for i in range(len(fv))) for j in range(3)], 1)
    t_e = torch.stack([sum(D[:, m, k, None] * ft[m] for m in range(len(ft))) for k in range(5)], 1)
    agg_s = h.new_zeros((n_atoms, C)).index_add_(0, destination, h * w[:, None])
    agg_v = h.new_zeros((n_atoms, 3, C)).index_add_(0, destination, v_e * w[:, None, None])
    agg_t = h.new_zeros((n_atoms, 5, C)).index_add_(0, destination, t_e * w[:, None, None])
    return agg_s, agg_v, agg_t
