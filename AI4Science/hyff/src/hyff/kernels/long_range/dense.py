"""Exact latent-charge pair term: FP64 torch reference and fused FP32 Triton dense kernel.

Convention (hyff.models.latent_charges): E = sum_{i<j} q_i q_j K(r_ij), K = erf(r/(sqrt2 sigma))/r,
isolated (no cell), positions as given, no self term; the caller multiplies by the coupling
14.399645*scale. Every evaluator here returns the UNSCALED potential and field-like terms

    phi_i = sum_{j!=i} q_j K_ij          (= dE/dq_i)
    g_i   = sum_{j!=i} q_j h_ij (x_i-x_j) (dE/dx_i = q_i g_i),   h = 2 dK/d(r^2)

so E = 0.5 sum_i q_i phi_i. The Triton kernel is owner-computes (each pair evaluated once per
direction), deterministic, O(N) memory, no host synchronisation, sigma read from a device scalar.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
SERIES_T = 0.5
# K/a = sum_k (-1)^k t^k/(k!(2k+1)),  h/(a/sigma^2) = sum_k (-1)^(k+1) t^k/(k!(2k+3)), t<=0.5.
# Degree 10/9 truncation at t=.5 is < 2e-11 relative: far below FP32 rounding.
K_COEF = tuple((-1) ** k / (math.factorial(k) * (2 * k + 1)) for k in range(11))
H_COEF = tuple((-1) ** (k + 1) / (math.factorial(k) * (2 * k + 3)) for k in range(10))


def radial_reference(r2, sigma):
    """FP64-capable K and h, using the same series/regular split (any dtype)."""
    sigma = torch.as_tensor(sigma, dtype=r2.dtype, device=r2.device)
    a = SQRT_2_OVER_PI / sigma
    t = r2 / (2 * sigma * sigma)
    stable = t <= SERIES_T
    tm = torch.where(stable, t, torch.zeros_like(t))
    sk = torch.full_like(t, K_COEF[-1])
    for c in reversed(K_COEF[:-1]):
        sk = c + tm * sk
    sh = torch.full_like(t, H_COEF[-1])
    for c in reversed(H_COEF[:-1]):
        sh = c + tm * sh
    rr2 = torch.where(stable, torch.ones_like(r2), r2)
    r = rr2.sqrt()
    k_reg = torch.erf(r / (math.sqrt(2.0) * sigma)) / r
    h_reg = (a * torch.exp(-t) - k_reg) / rr2
    return torch.where(stable, a * sk, k_reg), torch.where(stable, a / (sigma * sigma) * sh, h_reg)


def dense_reference(positions, charges, sigma, targets=None, block=1024):
    """Exact phi, g for `targets` (default all) against all sources; dtype of the inputs.

    Memory O(block*N). Use float64 inputs for the FP64 reference."""
    n = positions.shape[0]
    idx = torch.arange(n, device=positions.device) if targets is None else targets
    phi = positions.new_zeros(idx.numel())
    g = positions.new_zeros(idx.numel(), 3)
    for b in range(0, idx.numel(), block):
        rows = idx[b:b + block]
        d = positions[rows, None, :] - positions[None, :, :]
        r2 = d.square().sum(-1)
        k, h = radial_reference(r2, sigma)
        w = charges[None, :].expand_as(k).clone()
        w[torch.arange(rows.numel(), device=w.device), rows] = 0
        phi[b:b + block] = (w * k).sum(1)
        g[b:b + block] = ((w * h)[..., None] * d).sum(1)
    return phi, g


@triton.jit
def radial_terms(r2, inv_2s2, a, a_over_s2):
    """K and h=2dK/d(r^2) in FP32: series through t=.5 (no erf cancellation), erf beyond."""
    t = r2 * inv_2s2
    stable = t <= 0.5
    tm = tl.where(stable, t, 0.0)
    sk = tm * (-1.0 / 76204800.0) + (1.0 / 6894720.0)   # k=10, 9
    sk = sk * tm + (-1.0 / 685440.0)
    sk = sk * tm + (1.0 / 75600.0)
    sk = sk * tm + (-1.0 / 9360.0)
    sk = sk * tm + (1.0 / 1320.0)
    sk = sk * tm + (-1.0 / 216.0)
    sk = sk * tm + (1.0 / 42.0)
    sk = sk * tm + (-1.0 / 10.0)
    sk = sk * tm + (1.0 / 3.0)
    sk = 1.0 - sk * tm
    sh = tm * (1.0 / 7620480.0) + (-1.0 / 766080.0)      # k=9, 8
    sh = sh * tm + (1.0 / 85680.0)
    sh = sh * tm + (-1.0 / 10800.0)
    sh = sh * tm + (1.0 / 1560.0)
    sh = sh * tm + (-1.0 / 264.0)
    sh = sh * tm + (1.0 / 54.0)
    sh = sh * tm + (-1.0 / 14.0)
    sh = sh * tm + (1.0 / 5.0)
    sh = sh * tm + (-1.0 / 3.0)
    rr2 = tl.where(stable, 1.0, r2)
    inv_r = libdevice.rsqrt(rr2)
    k_reg = libdevice.erf(rr2 * inv_r * tl.sqrt(inv_2s2)) * inv_r
    h_reg = (a * tl.exp(-t) - k_reg) * (inv_r * inv_r)
    k = tl.where(stable, a * sk, k_reg)
    h = tl.where(stable, a_over_s2 * sh, h_reg)
    return k, h


@triton.jit
def _dense_kernel(X, Q, SIGMA, PHI, G, N, BI: tl.constexpr, BJ: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.load(SIGMA).to(tl.float32)
    inv_2s2 = 0.5 / (s * s)
    a = 0.7978845608028654 / s
    a_over_s2 = a / (s * s)
    i = pid * BI + tl.arange(0, BI)
    mi = i < N
    xi = tl.load(X + 3 * i, mask=mi, other=0.0)
    yi = tl.load(X + 3 * i + 1, mask=mi, other=0.0)
    zi = tl.load(X + 3 * i + 2, mask=mi, other=0.0)
    phi = tl.zeros((BI,), tl.float32)
    gx = tl.zeros((BI,), tl.float32)
    gy = tl.zeros((BI,), tl.float32)
    gz = tl.zeros((BI,), tl.float32)
    for j0 in range(0, N, BJ):
        j = j0 + tl.arange(0, BJ)
        mj = j < N
        xj = tl.load(X + 3 * j, mask=mj, other=0.0)
        yj = tl.load(X + 3 * j + 1, mask=mj, other=0.0)
        zj = tl.load(X + 3 * j + 2, mask=mj, other=0.0)
        qj = tl.load(Q + j, mask=mj, other=0.0)
        dx = xi[:, None] - xj[None, :]
        dy = yi[:, None] - yj[None, :]
        dz = zi[:, None] - zj[None, :]
        r2 = dx * dx + dy * dy + dz * dz
        k, h = radial_terms(r2, inv_2s2, a, a_over_s2)
        w = tl.where(i[:, None] != j[None, :], qj[None, :], 0.0)
        phi += tl.sum(w * k, 1)
        wh = w * h
        gx += tl.sum(wh * dx, 1)
        gy += tl.sum(wh * dy, 1)
        gz += tl.sum(wh * dz, 1)
    tl.store(PHI + i, phi, mask=mi)
    tl.store(G + 3 * i, gx, mask=mi)
    tl.store(G + 3 * i + 1, gy, mask=mi)
    tl.store(G + 3 * i + 2, gz, mask=mi)


def dense_triton(positions, charges, sigma, out=None, block_i=64, block_j=64, num_warps=4):
    """phi (N,), g (N,3) in FP32. positions (N,3) and charges (N,) are cast to contiguous FP32."""
    x = positions.detach().to(torch.float32).contiguous()
    q = charges.detach().to(torch.float32).contiguous()
    s = torch.as_tensor(sigma, device=x.device).detach().reshape(1).to(torch.float32)
    n = x.shape[0]
    if out is None:
        phi = torch.empty(n, device=x.device, dtype=torch.float32)
        g = torch.empty(n, 3, device=x.device, dtype=torch.float32)
    else:
        phi, g = out
    _dense_kernel[(triton.cdiv(n, block_i),)](x, q, s, phi, g, n, BI=block_i, BJ=block_j,
                                             num_warps=num_warps)
    return phi, g
