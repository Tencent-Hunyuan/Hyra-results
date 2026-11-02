"""Fused atom-local node kernels for the presto engine (frozen weights, exact adjoints).

* `FusedHead`: LayerNorm -> Linear(H -> H) + SiLU -> Linear(H -> O) in one Triton launch per direction,
  for any channel count H (masked power-of-two tile BH, so hidden 48 runs on a 64-lane tile) and a selectable
  LayerNorm statistics precision (`fp64`: the export policy, `fp32`: centred two-pass FP32). Products are
  IEEE FP32 / TF32x3 / FP16x3 on the FP32 weights, or fp16w2 on FP16 copies of FP16-representable weights.
  Saved-statistics backward (mean, rstd, pre-activation stored by the forward).
* `local_features`: as `hyff.kernels.local_features_fp64` (affine LayerNorm of the scalar
  state, |v|, |T|, the three higher invariants and the two dot closures) with the same selectable
  statistics precision; `enable_local_features` installs the node update on every block.

Parameters stay the model's `nn.Parameter` objects; nothing is copied or rounded except the optional
FP16 weight copies of the fp16w2 policy, which require exact representability.
"""
from __future__ import annotations

from types import MethodType

import torch
from torch import nn
import triton as tr
import triton.language as tl

from hyff.kernels.eight_channel_messages import _dot
from hyff.kernels.local_features_fp64 import _load_tensor, _load_vector, _square_order2
from hyff.kernels.fused_messages import _matvec, _q, _stv

from hyff.engines.native.presto_messages import PRECISIONS, fp16_representable

STATISTICS = ('fp64', 'fp32')


# ---------------------------------------------------------------------------------------------
# Fused norm -> Linear/SiLU/Linear heads.

@tr.jit
def _head_forward(X, NW, NB, W1, B1, W2, B2, Out, Mean, Rstd, Pre, N, H: tl.constexpr, BH: tl.constexpr, O: tl.constexpr,
                  BO: tl.constexpr, EPS: tl.constexpr, BM: tl.constexpr, FP64: tl.constexpr, PRECISION: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    rmask = row < N
    ch = tl.arange(0, BH)
    cmask = ch < H
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(X + row[:, None] * H + ch[None, :], mask, 0.)
    weight = tl.load(NW + ch, cmask, 0.)
    bias = tl.load(NB + ch, cmask, 0.)
    if FP64:
        x = x.to(tl.float64)
        weight = weight.to(tl.float64)
        bias = bias.to(tl.float64)
    mean = tl.sum(x, 1) / H
    deviation = tl.where(cmask[None, :], x - mean[:, None], 0.)
    variance = tl.sum(deviation * deviation, 1) / H
    rstd = tl.rsqrt(variance + EPS)
    xn = (deviation * rstd[:, None] * weight[None, :] + bias[None, :]).to(tl.float32)
    k = tl.arange(0, BH)
    n = tl.arange(0, BH)
    w1t = tl.load(W1 + n[None, :] * H + k[:, None], (n[None, :] < H) & (k[:, None] < H), 0.)       # [H_in, H_out] = W1^T
    pre = _dot(xn, w1t, tl.zeros((BM, BH), tl.float32), PRECISION) + tl.load(B1 + n, n < H, 0.)[None, :]
    sigmoid = 1. / (1. + tl.exp(-pre))
    hidden = pre * sigmoid
    tl.store(Mean + row, mean, rmask)
    tl.store(Rstd + row, rstd, rmask)
    tl.store(Pre + row[:, None] * H + ch[None, :], pre, mask)
    o = tl.arange(0, BO)
    w2t = tl.load(W2 + o[None, :] * H + k[:, None], (o[None, :] < O) & (k[:, None] < H), 0.)       # [H, BO] = W2^T padded
    out = _dot(hidden, w2t, tl.zeros((BM, BO), tl.float32), PRECISION) + tl.load(B2 + o, o < O, 0.)[None, :]
    tl.store(Out + row[:, None] * O + o[None, :], out, rmask[:, None] & (o[None, :] < O))


@tr.jit
def _head_backward(X, NW, W1, W2, Mean, Rstd, Pre, GY, GX, N, H: tl.constexpr, BH: tl.constexpr, O: tl.constexpr,
                   BO: tl.constexpr, BM: tl.constexpr, FP64: tl.constexpr, PRECISION: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    rmask = row < N
    ch = tl.arange(0, BH)
    cmask = ch < H
    mask = rmask[:, None] & cmask[None, :]
    o = tl.arange(0, BO)
    k = tl.arange(0, BH)
    n = tl.arange(0, BH)
    gy = tl.load(GY + row[:, None] * O + o[None, :], rmask[:, None] & (o[None, :] < O), 0.)
    w2 = tl.load(W2 + o[:, None] * H + k[None, :], (o[:, None] < O) & (k[None, :] < H), 0.)          # [BO, H]
    dh = _dot(gy, w2, tl.zeros((BM, BH), tl.float32), PRECISION)
    pre = tl.load(Pre + row[:, None] * H + ch[None, :], mask, 0.)
    sigmoid = 1. / (1. + tl.exp(-pre))
    dpre = dh * (sigmoid + pre * sigmoid * (1. - sigmoid))
    w1 = tl.load(W1 + n[:, None] * H + k[None, :], (n[:, None] < H) & (k[None, :] < H), 0.)          # [H_out, H_in]
    dxn = _dot(dpre, w1, tl.zeros((BM, BH), tl.float32), PRECISION)
    x = tl.load(X + row[:, None] * H + ch[None, :], mask, 0.)
    weight = tl.load(NW + ch, cmask, 0.)
    if FP64:
        x = x.to(tl.float64)
        weight = weight.to(tl.float64)
        dxn = dxn.to(tl.float64)
    mean = tl.load(Mean + row, rmask, 0.)
    rstd = tl.load(Rstd + row, rmask, 0.)
    normalized = tl.where(cmask[None, :], (x - mean[:, None]) * rstd[:, None], 0.)
    scaled = dxn * weight[None, :]
    mean_scaled = tl.sum(scaled, 1) / H
    mean_projected = tl.sum(scaled * normalized, 1) / H
    dx = (scaled - mean_scaled[:, None] - normalized * mean_projected[:, None]) * rstd[:, None]
    tl.store(GX + row[:, None] * H + ch[None, :], dx, mask)


class _FusedHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nw, nb, w1, b1, w2, b2, eps, rows, warps, precision, fp64):
        x = x.contiguous()
        n, h = x.shape
        o = w2.shape[0]
        bh = max(16, tr.next_power_of_2(h))
        bo = max(32, tr.next_power_of_2(o))
        out = x.new_empty(n, o)
        statistics = torch.float64 if fp64 else torch.float32
        mean, rstd, pre = x.new_empty(n, dtype=statistics), x.new_empty(n, dtype=statistics), torch.empty_like(x)
        if n:
            _head_forward[(tr.cdiv(n, rows),)](x, nw, nb, w1, b1, w2, b2, out, mean, rstd, pre, n, h, bh, o, bo, eps, rows, fp64, precision,
                                               num_warps=warps, enable_fp_fusion=False)
        ctx.save_for_backward(x, nw, w1, w2, mean, rstd, pre)
        ctx.settings = (rows, warps, precision, fp64, o, bo, bh)
        return out

    @staticmethod
    def backward(ctx, gy):
        if torch.is_grad_enabled():
            raise RuntimeError('Force training needs the differentiable reference head')
        rows, warps, precision, fp64, o, bo, bh = ctx.settings
        x, nw, w1, w2, mean, rstd, pre = ctx.saved_tensors
        n, h = x.shape
        gx = torch.empty_like(x)
        if n:
            _head_backward[(tr.cdiv(n, rows),)](x, nw, w1, w2, mean, rstd, pre, gy.contiguous(), gx, n, h, bh, o, bo, rows, fp64, precision,
                                                num_warps=warps, enable_fp_fusion=False)
        return gx, None, None, None, None, None, None, None, None, None, None, None


class FusedHead(nn.Module):
    """norm -> linear1 -> SiLU -> linear2 with the modules' own parameter objects (any H, any O)."""

    def __init__(self, norm, linear1, linear2, rows=16, warps=2, precision='fp32', statistics='fp64'):
        super().__init__()
        if precision not in PRECISIONS:
            raise ValueError(f'precision must be one of {PRECISIONS}')
        if statistics not in STATISTICS:
            raise ValueError(f'statistics must be one of {STATISTICS}')
        if not (len(norm.normalized_shape) == 1 and norm.weight is not None and norm.bias is not None):
            raise ValueError('Affine LayerNorm over the channel dimension required')
        h = int(norm.normalized_shape[0])
        if tuple(linear1.weight.shape) != (h, h) or linear1.bias is None:
            raise ValueError('The fused head expects a square first linear with bias')
        if linear2.weight.shape[1] != h or linear2.bias is None:
            raise ValueError('The second linear must consume the hidden width and carry a bias')
        for parameter in (norm.weight, norm.bias, linear1.weight, linear1.bias, linear2.weight, linear2.bias):
            if parameter.requires_grad or parameter.dtype != torch.float32:
                raise ValueError('Freeze FP32 parameters before fusing the head')
        if rows not in (16, 32, 64, 128) or warps not in (1, 2, 4, 8):
            raise ValueError('Unsupported launch layout')
        self.norm, self.linear1, self.linear2 = norm, linear1, linear2
        self.rows, self.warps, self.precision, self.fp64 = rows, warps, precision, statistics == 'fp64'
        self.out_features = int(linear2.weight.shape[0])
        self.channels = h
        if precision == 'fp16w2':
            for name, layer in (('w1', linear1), ('w2', linear2)):
                if not fp16_representable(layer.weight):
                    raise ValueError('fp16w2 requires FP16-representable head weights; the export is not rounded')
                self.register_buffer(name, layer.weight.detach().half().contiguous(), persistent=False)

    def weights(self):
        if self.precision == 'fp16w2':
            return self.w1, self.w2
        return self.linear1.weight.contiguous(), self.linear2.weight.contiguous()

    def forward(self, values):
        if values.device.type != 'cuda' or values.dtype != torch.float32 or values.ndim != 2 or values.shape[1] != self.channels:
            raise ValueError('Fused heads consume CUDA FP32 [atoms, channels] features of the declared width')
        w1, w2 = self.weights()
        return _FusedHead.apply(values, self.norm.weight, self.norm.bias, w1, self.linear1.bias, w2, self.linear2.bias,
                                float(self.norm.eps), self.rows, self.warps, self.precision, self.fp64)


# ---------------------------------------------------------------------------------------------
# Dual head: readout of block i and scalar-message head of block i+1 share the input scalar and its
# LayerNorm statistics (both norms are LayerNorm(hidden) with the same epsilon; only the affines differ).

@tr.jit
def _affine_mlp(deviation, rstd, cmask, NW, NB, W1, B1, W2, B2, Out, Pre, row, rmask, ch, mask, H: tl.constexpr, BH: tl.constexpr,
                O: tl.constexpr, BO: tl.constexpr, FP64: tl.constexpr, PRECISION: tl.constexpr):
    weight = tl.load(NW + ch, cmask, 0.)
    bias = tl.load(NB + ch, cmask, 0.)
    if FP64:
        weight = weight.to(tl.float64)
        bias = bias.to(tl.float64)
    xn = (deviation * rstd[:, None] * weight[None, :] + bias[None, :]).to(tl.float32)
    k = tl.arange(0, BH)
    n = tl.arange(0, BH)
    w1t = tl.load(W1 + n[None, :] * H + k[:, None], (n[None, :] < H) & (k[:, None] < H), 0.)
    pre = _dot(xn, w1t, tl.zeros((xn.shape[0], BH), tl.float32), PRECISION) + tl.load(B1 + n, n < H, 0.)[None, :]
    sigmoid = 1. / (1. + tl.exp(-pre))
    hidden = pre * sigmoid
    tl.store(Pre + row[:, None] * H + ch[None, :], pre, mask)
    o = tl.arange(0, BO)
    w2t = tl.load(W2 + o[None, :] * H + k[:, None], (o[None, :] < O) & (k[:, None] < H), 0.)
    out = _dot(hidden, w2t, tl.zeros((xn.shape[0], BO), tl.float32), PRECISION) + tl.load(B2 + o, o < O, 0.)[None, :]
    tl.store(Out + row[:, None] * O + o[None, :], out, rmask[:, None] & (o[None, :] < O))


@tr.jit
def _mlp_backward(W1, W2, Pre, GY, row, rmask, ch, mask, H: tl.constexpr, BH: tl.constexpr, O: tl.constexpr, BO: tl.constexpr, PRECISION: tl.constexpr):
    o = tl.arange(0, BO)
    k = tl.arange(0, BH)
    n = tl.arange(0, BH)
    gy = tl.load(GY + row[:, None] * O + o[None, :], rmask[:, None] & (o[None, :] < O), 0.)
    w2 = tl.load(W2 + o[:, None] * H + k[None, :], (o[:, None] < O) & (k[None, :] < H), 0.)
    dh = _dot(gy, w2, tl.zeros((gy.shape[0], BH), tl.float32), PRECISION)
    pre = tl.load(Pre + row[:, None] * H + ch[None, :], mask, 0.)
    sigmoid = 1. / (1. + tl.exp(-pre))
    dpre = dh * (sigmoid + pre * sigmoid * (1. - sigmoid))
    w1 = tl.load(W1 + n[:, None] * H + k[None, :], (n[:, None] < H) & (k[None, :] < H), 0.)
    return _dot(dpre, w1, tl.zeros((gy.shape[0], BH), tl.float32), PRECISION)


@tr.jit
def _dual_forward(X, NWA, NBA, NWB, NBB, W1A, B1A, W2A, B2A, W1B, B1B, W2B, B2B, OutA, OutB, Mean, Rstd, PreA, PreB, N,
                  H: tl.constexpr, BH: tl.constexpr, OA: tl.constexpr, BOA: tl.constexpr, OB: tl.constexpr, BOB: tl.constexpr,
                  EPS: tl.constexpr, BM: tl.constexpr, FP64: tl.constexpr, PRECISION: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    rmask = row < N
    ch = tl.arange(0, BH)
    cmask = ch < H
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(X + row[:, None] * H + ch[None, :], mask, 0.)
    if FP64:
        x = x.to(tl.float64)
    mean = tl.sum(x, 1) / H
    deviation = tl.where(cmask[None, :], x - mean[:, None], 0.)
    variance = tl.sum(deviation * deviation, 1) / H
    rstd = tl.rsqrt(variance + EPS)
    tl.store(Mean + row, mean, rmask)
    tl.store(Rstd + row, rstd, rmask)
    _affine_mlp(deviation, rstd, cmask, NWA, NBA, W1A, B1A, W2A, B2A, OutA, PreA, row, rmask, ch, mask, H, BH, OA, BOA, FP64, PRECISION)
    _affine_mlp(deviation, rstd, cmask, NWB, NBB, W1B, B1B, W2B, B2B, OutB, PreB, row, rmask, ch, mask, H, BH, OB, BOB, FP64, PRECISION)


@tr.jit
def _dual_backward(X, NWA, NWB, W1A, W2A, W1B, W2B, Mean, Rstd, PreA, PreB, GYA, GYB, GX, N,
                   H: tl.constexpr, BH: tl.constexpr, OA: tl.constexpr, BOA: tl.constexpr, OB: tl.constexpr, BOB: tl.constexpr,
                   BM: tl.constexpr, FP64: tl.constexpr, PRECISION: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    rmask = row < N
    ch = tl.arange(0, BH)
    cmask = ch < H
    mask = rmask[:, None] & cmask[None, :]
    dxna = _mlp_backward(W1A, W2A, PreA, GYA, row, rmask, ch, mask, H, BH, OA, BOA, PRECISION)
    dxnb = _mlp_backward(W1B, W2B, PreB, GYB, row, rmask, ch, mask, H, BH, OB, BOB, PRECISION)
    x = tl.load(X + row[:, None] * H + ch[None, :], mask, 0.)
    wa = tl.load(NWA + ch, cmask, 0.)
    wb = tl.load(NWB + ch, cmask, 0.)
    if FP64:
        x = x.to(tl.float64)
        wa = wa.to(tl.float64)
        wb = wb.to(tl.float64)
        dxna = dxna.to(tl.float64)
        dxnb = dxnb.to(tl.float64)
    mean = tl.load(Mean + row, rmask, 0.)
    rstd = tl.load(Rstd + row, rmask, 0.)
    normalized = tl.where(cmask[None, :], (x - mean[:, None]) * rstd[:, None], 0.)
    scaled = dxna * wa[None, :] + dxnb * wb[None, :]
    mean_scaled = tl.sum(scaled, 1) / H
    mean_projected = tl.sum(scaled * normalized, 1) / H
    dx = (scaled - mean_scaled[:, None] - normalized * mean_projected[:, None]) * rstd[:, None]
    tl.store(GX + row[:, None] * H + ch[None, :], dx, mask)


class _DualHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nwa, nba, nwb, nbb, w1a, b1a, w2a, b2a, w1b, b1b, w2b, b2b, eps, rows, warps, precision, fp64):
        x = x.contiguous()
        n, h = x.shape
        oa, ob = w2a.shape[0], w2b.shape[0]
        bh = max(16, tr.next_power_of_2(h))
        boa, bob = max(32, tr.next_power_of_2(oa)), max(32, tr.next_power_of_2(ob))
        out_a, out_b = x.new_empty(n, oa), x.new_empty(n, ob)
        statistics = torch.float64 if fp64 else torch.float32
        mean, rstd = x.new_empty(n, dtype=statistics), x.new_empty(n, dtype=statistics)
        pre_a, pre_b = torch.empty_like(x), torch.empty_like(x)
        if n:
            _dual_forward[(tr.cdiv(n, rows),)](x, nwa, nba, nwb, nbb, w1a, b1a, w2a, b2a, w1b, b1b, w2b, b2b, out_a, out_b, mean, rstd, pre_a, pre_b, n,
                                               h, bh, oa, boa, ob, bob, eps, rows, fp64, precision, num_warps=warps, enable_fp_fusion=False)
        ctx.save_for_backward(x, nwa, nwb, w1a, w2a, w1b, w2b, mean, rstd, pre_a, pre_b)
        ctx.settings = (rows, warps, precision, fp64, oa, boa, ob, bob, bh)
        ctx.set_materialize_grads(False)
        return out_a, out_b

    @staticmethod
    def backward(ctx, gya, gyb):
        if torch.is_grad_enabled():
            raise RuntimeError('Force training needs the differentiable reference heads')
        rows, warps, precision, fp64, oa, boa, ob, bob, bh = ctx.settings
        x, nwa, nwb, w1a, w2a, w1b, w2b, mean, rstd, pre_a, pre_b = ctx.saved_tensors
        n, h = x.shape
        gya = gya.contiguous() if gya is not None else x.new_zeros(n, oa)
        gyb = gyb.contiguous() if gyb is not None else x.new_zeros(n, ob)
        gx = torch.empty_like(x)
        if n:
            _dual_backward[(tr.cdiv(n, rows),)](x, nwa, nwb, w1a, w2a, w1b, w2b, mean, rstd, pre_a, pre_b, gya, gyb, gx, n,
                                                h, bh, oa, boa, ob, bob, rows, fp64, precision, num_warps=warps, enable_fp_fusion=False)
        return (gx,) + (None,) * 17


class DualHead(nn.Module):
    """Two norm -> Linear/SiLU -> Linear heads of one input (shared statistics) in one launch per direction."""

    def __init__(self, head_a: FusedHead, head_b: FusedHead):
        super().__init__()
        for head in (head_a, head_b):
            if not isinstance(head, FusedHead):
                raise TypeError('DualHead combines two FusedHead modules')
        if (head_a.channels != head_b.channels or float(head_a.norm.eps) != float(head_b.norm.eps) or head_a.precision != head_b.precision
                or head_a.fp64 != head_b.fp64 or (head_a.rows, head_a.warps) != (head_b.rows, head_b.warps)):
            raise ValueError('Both heads must share channels, epsilon, precision, statistics and launch layout')
        self.a, self.b = head_a, head_b

    def forward(self, values):
        a, b = self.a, self.b
        if values.device.type != 'cuda' or values.dtype != torch.float32 or values.ndim != 2 or values.shape[1] != a.channels:
            raise ValueError('Dual heads consume CUDA FP32 [atoms, channels] features of the declared width')
        w1a, w2a = a.weights()
        w1b, w2b = b.weights()
        return _DualHead.apply(values, a.norm.weight, a.norm.bias, b.norm.weight, b.norm.bias, w1a, a.linear1.bias, w2a, a.linear2.bias,
                               w1b, b.linear1.bias, w2b, b.linear2.bias, float(a.norm.eps), a.rows, a.warps, a.precision, a.fp64)


# ---------------------------------------------------------------------------------------------
# Fused local features (frozen_local_features64 with a statistics-precision flag).

@tr.jit
def _features_forward(X, W, B, VU, VV, TU, TV, Out, Dots, Mean, Rstd,
                      N, S: tl.constexpr, T: tl.constexpr, EPS: tl.constexpr, HIGHER: tl.constexpr, FP64: tl.constexpr,
                      VUS0: tl.constexpr, VUS1: tl.constexpr, VUS2: tl.constexpr,
                      VVS0: tl.constexpr, VVS1: tl.constexpr, VVS2: tl.constexpr,
                      TUS0: tl.constexpr, TUS1: tl.constexpr, TUS2: tl.constexpr,
                      TVS0: tl.constexpr, TVS1: tl.constexpr, TVS2: tl.constexpr,
                      BM: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    ch = tl.arange(0, BC)
    valid_s = (row[:, None] < N) & (ch[None, :] < S)
    valid_t = (row[:, None] < N) & (ch[None, :] < T)
    x = tl.load(X + row[:, None] * S + ch[None, :], valid_s, 0.)
    weight, bias = tl.load(W + ch, ch < S, 0.), tl.load(B + ch, ch < S, 0.)
    if FP64:
        x = x.to(tl.float64)
        weight = weight.to(tl.float64)
        bias = bias.to(tl.float64)
    mean = tl.sum(x, 1) / S
    deviation = tl.where(ch[None, :] < S, x - mean[:, None], 0.)
    rstd = tl.rsqrt(tl.sum(deviation * deviation, 1) / S + EPS)
    normalized = deviation * rstd[:, None] * weight[None, :] + bias[None, :]
    WIDTH: tl.constexpr = S + (5 if HIGHER else 2) * T
    base = row[:, None] * WIDTH + ch[None, :]
    tl.store(Out + base, normalized, valid_s)
    tl.store(Mean + row, mean, row < N)
    tl.store(Rstd + row, rstd, row < N)
    ux, uy, uz = _load_vector(VU, row, ch, N, T, VUS0, VUS1, VUS2)
    vx, vy, vz = _load_vector(VV, row, ch, N, T, VVS0, VVS1, VVS2)
    u0, u1, u2, u3, u4 = _load_tensor(TU, row, ch, N, T, TUS0, TUS1, TUS2)
    t0, t1, t2, t3, t4 = _load_tensor(TV, row, ch, N, T, TVS0, TVS1, TVS2)
    v2 = vx * vx + vy * vy + vz * vz
    t2norm = t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3 + t4 * t4
    vn, tn = tl.sqrt(v2 + 1.e-8), tl.sqrt(t2norm + 1.e-8)
    tl.store(Out + base + S, vn, valid_t)
    tl.store(Out + base + S + T, tn, valid_t)
    dot_base = row[:, None] * 2 * T + ch[None, :]
    tl.store(Dots + dot_base, ux * vx + uy * vy + uz * vz, valid_t)
    tl.store(Dots + dot_base + T, u0 * t0 + u1 * t1 + u2 * t2 + u3 * t3 + u4 * t4, valid_t)
    if HIGHER:
        vr, trstd = tl.rsqrt(1. + v2), tl.rsqrt(1. + t2norm)
        x, y, z = vx * vr, vy * vr, vz * vr
        a, b, c, d, e = t0 * trstd, t1 * trstd, t2 * trstd, t3 * trstd, t4 * trstd
        tx, ty, tz = _matvec(a, b, c, d, e, x, y, z)
        q0, q1, q2, q3, q4 = _square_order2(a, b, c, d, e)
        tl.store(Out + base + S + 2 * T, q0 * a + q1 * b + q2 * c + q3 * d + q4 * e, valid_t)
        tl.store(Out + base + S + 3 * T, x * tx + y * ty + z * tz, valid_t)
        tl.store(Out + base + S + 4 * T, tx * tx + ty * ty + tz * tz, valid_t)


@tr.jit
def _features_backward(X, W, VU, VV, TU, TV, Features, Mean, Rstd, GOut, GDots,
                       DX, DVU, DVV, DTU, DTV, N, S: tl.constexpr, T: tl.constexpr, HIGHER: tl.constexpr, FP64: tl.constexpr,
                       VUS0: tl.constexpr, VUS1: tl.constexpr, VUS2: tl.constexpr,
                       VVS0: tl.constexpr, VVS1: tl.constexpr, VVS2: tl.constexpr,
                       TUS0: tl.constexpr, TUS1: tl.constexpr, TUS2: tl.constexpr,
                       TVS0: tl.constexpr, TVS1: tl.constexpr, TVS2: tl.constexpr,
                       BM: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    ch = tl.arange(0, BC)
    valid_s = (row[:, None] < N) & (ch[None, :] < S)
    valid_t = (row[:, None] < N) & (ch[None, :] < T)
    x = tl.load(X + row[:, None] * S + ch[None, :], valid_s, 0.)
    WIDTH: tl.constexpr = S + (5 if HIGHER else 2) * T
    base = row[:, None] * WIDTH + ch[None, :]
    weight = tl.load(W + ch, ch < S, 0.)
    gradient = tl.load(GOut + base, valid_s, 0.)
    if FP64:
        x = x.to(tl.float64)
        weight = weight.to(tl.float64)
        gradient = gradient.to(tl.float64)
    mean, rstd = tl.load(Mean + row, row < N, 0.), tl.load(Rstd + row, row < N, 0.)
    norm = tl.where(ch[None, :] < S, (x - mean[:, None]) * rstd[:, None], 0.)
    gradient = gradient * weight[None, :]
    avg, projected = tl.sum(gradient, 1) / S, tl.sum(gradient * norm, 1) / S
    dx = (gradient - avg[:, None] - norm * projected[:, None]) * rstd[:, None]
    tl.store(DX + row[:, None] * S + ch[None, :], dx, valid_s)
    ux, uy, uz = _load_vector(VU, row, ch, N, T, VUS0, VUS1, VUS2)
    vx, vy, vz = _load_vector(VV, row, ch, N, T, VVS0, VVS1, VVS2)
    u0, u1, u2, u3, u4 = _load_tensor(TU, row, ch, N, T, TUS0, TUS1, TUS2)
    t0, t1, t2, t3, t4 = _load_tensor(TV, row, ch, N, T, TVS0, TVS1, TVS2)
    vn = tl.load(Features + base + S, valid_t, 1.)
    tn = tl.load(Features + base + S + T, valid_t, 1.)
    gv = tl.load(GOut + base + S, valid_t, 0.) / vn
    gt = tl.load(GOut + base + S + T, valid_t, 0.) / tn
    dot_base = row[:, None] * 2 * T + ch[None, :]
    gdv, gdt = tl.load(GDots + dot_base, valid_t, 0.), tl.load(GDots + dot_base + T, valid_t, 0.)
    dvx, dvy, dvz = gv * vx + gdv * ux, gv * vy + gdv * uy, gv * vz + gdv * uz
    dt0, dt1, dt2, dt3, dt4 = (gt * t0 + gdt * u0, gt * t1 + gdt * u1, gt * t2 + gdt * u2, gt * t3 + gdt * u3, gt * t4 + gdt * u4)
    if HIGHER:
        vr = tl.rsqrt(1. + vx * vx + vy * vy + vz * vz)
        trstd = tl.rsqrt(1. + t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3 + t4 * t4)
        x, y, z = vx * vr, vy * vr, vz * vr
        a, b, c, d, e = t0 * trstd, t1 * trstd, t2 * trstd, t3 * trstd, t4 * trstd
        tx, ty, tz = _matvec(a, b, c, d, e, x, y, z)
        ttx, tty, ttz = _matvec(a, b, c, d, e, tx, ty, tz)
        q0, q1, q2, q3, q4 = _square_order2(a, b, c, d, e)
        vv0, vv1, vv2, vv3, vv4 = _q(x, y, z)
        tv0, tv1, tv2, tv3, tv4 = _stv(tx, ty, tz, x, y, z)
        g3 = tl.load(GOut + base + S + 2 * T, valid_t, 0.)
        gvt = tl.load(GOut + base + S + 3 * T, valid_t, 0.)
        gtt = tl.load(GOut + base + S + 4 * T, valid_t, 0.)
        gx, gy, gz = 2 * (gvt * tx + gtt * ttx), 2 * (gvt * ty + gtt * tty), 2 * (gvt * tz + gtt * ttz)
        ga, gb, gc, gd, ge = (3 * g3 * q0 + gvt * vv0 + 2 * gtt * tv0, 3 * g3 * q1 + gvt * vv1 + 2 * gtt * tv1,
                              3 * g3 * q2 + gvt * vv2 + 2 * gtt * tv2, 3 * g3 * q3 + gvt * vv3 + 2 * gtt * tv3,
                              3 * g3 * q4 + gvt * vv4 + 2 * gtt * tv4)
        vp, tp = gx * x + gy * y + gz * z, ga * a + gb * b + gc * c + gd * d + ge * e
        dvx, dvy, dvz = dvx + (gx - vp * x) * vr, dvy + (gy - vp * y) * vr, dvz + (gz - vp * z) * vr
        dt0, dt1, dt2, dt3, dt4 = (dt0 + (ga - tp * a) * trstd, dt1 + (gb - tp * b) * trstd, dt2 + (gc - tp * c) * trstd,
                                   dt3 + (gd - tp * d) * trstd, dt4 + (ge - tp * e) * trstd)
    vbase, tbase = row[:, None] * 3 * T + ch[None, :], row[:, None] * 5 * T + ch[None, :]
    tl.store(DVU + vbase, gdv * vx, valid_t)
    tl.store(DVU + vbase + T, gdv * vy, valid_t)
    tl.store(DVU + vbase + 2 * T, gdv * vz, valid_t)
    tl.store(DVV + vbase, dvx, valid_t)
    tl.store(DVV + vbase + T, dvy, valid_t)
    tl.store(DVV + vbase + 2 * T, dvz, valid_t)
    tl.store(DTU + tbase, gdt * t0, valid_t)
    tl.store(DTU + tbase + T, gdt * t1, valid_t)
    tl.store(DTU + tbase + 2 * T, gdt * t2, valid_t)
    tl.store(DTU + tbase + 3 * T, gdt * t3, valid_t)
    tl.store(DTU + tbase + 4 * T, gdt * t4, valid_t)
    tl.store(DTV + tbase, dt0, valid_t)
    tl.store(DTV + tbase + T, dt1, valid_t)
    tl.store(DTV + tbase + 2 * T, dt2, valid_t)
    tl.store(DTV + tbase + 3 * T, dt3, valid_t)
    tl.store(DTV + tbase + 4 * T, dt4, valid_t)


class _LocalFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scalar, vu, vv, tu, tv, weight, bias, epsilon, higher, atom_tile, fp64):
        x = scalar.contiguous()
        n, s = x.shape
        t = vu.shape[-1]
        features = x.new_empty(n, s + (5 if higher else 2) * t)
        dots = x.new_empty(n, 2 * t)
        statistics = torch.float64 if fp64 else torch.float32
        mean, rstd = x.new_empty(n, dtype=statistics), x.new_empty(n, dtype=statistics)
        strides = tuple(stride for value in (vu, vv, tu, tv) for stride in value.stride())
        if n:
            _features_forward[(tr.cdiv(n, atom_tile),)](x, weight, bias, vu, vv, tu, tv, features, dots, mean, rstd, n, s, t, epsilon,
                                                        higher, fp64, *strides, atom_tile, tr.next_power_of_2(max(s, t)),
                                                        num_warps=4, enable_fp_fusion=True)
        ctx.save_for_backward(x, vu, vv, tu, tv, weight, features, mean, rstd)
        ctx.higher, ctx.atom_tile, ctx.strides, ctx.fp64 = higher, atom_tile, strides, fp64
        return features, dots

    @staticmethod
    def backward(ctx, gradient, dot_gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Force training needs differentiable reference invariants')
        x, vu, vv, tu, tv, weight, features, mean, rstd = ctx.saved_tensors
        outputs = tuple(value.new_empty(value.shape) for value in (x, vu, vv, tu, tv))
        n, s = x.shape
        t = vu.shape[-1]
        if n:
            _features_backward[(tr.cdiv(n, ctx.atom_tile),)](x, weight, vu, vv, tu, tv, features, mean, rstd, gradient.contiguous(),
                                                             dot_gradient.contiguous(), *outputs, n, s, t, ctx.higher, ctx.fp64, *ctx.strides,
                                                             ctx.atom_tile, tr.next_power_of_2(max(s, t)), num_warps=4, enable_fp_fusion=True)
        return (*outputs, None, None, None, None, None, None)


def local_features(scalar, vu, vv, tu, tv, norm, higher, atom_tile, fp64):
    if norm.weight.requires_grad or norm.bias.requires_grad:
        raise ValueError('Freeze affine normalization before selecting this backend')
    if atom_tile not in (1, 4, 8, 16, 32):
        raise ValueError('Unsupported atom tile')
    if any(value.device.type != 'cuda' or value.dtype != torch.float32 for value in (scalar, vu, vv, tu, tv, norm.weight, norm.bias)):
        raise ValueError('Fused local features require CUDA FP32 inputs')
    n, s = scalar.shape
    t = vu.shape[-1]
    if vu.shape != (n, 3, t) or vv.shape != vu.shape or tu.shape != (n, 5, t) or tv.shape != tu.shape or norm.weight.shape != (s,):
        raise ValueError('Incompatible local feature dimensions')
    return _LocalFeatures.apply(scalar, vu, vv, tu, tv, norm.weight, norm.bias, float(norm.eps), bool(higher), int(atom_tile), bool(fp64))


def _node_update(block, scalar, vector, tensor, projections=None):
    """`hyff.kernels.local_features_fp64._frozen_node_update` with the flagged features."""
    from hyff.models.l2_potential import _channel_tensor_vector, _channel_vector_tensor
    if projections is not None:
        vu, vv, tu, tv = projections
    else:
        if block.tensor_mixing_rank is not None:
            raise RuntimeError('Low-rank tensor mixing is not supported by this engine')
        vu, vv = block.joined_vector(vector).chunk(2, dim=-1)
        tu, tv = block.joined_tensor(tensor).chunk(2, dim=-1)
    features, dots = local_features(scalar, vu, vv, tu, tv, block.update_norm, block.higher_invariants,
                                    block._engine_feature_tile, block._engine_feature_fp64)
    update = block.update_mlp(features)
    vector_dot, tensor_dot = dots.chunk(2, dim=-1)
    if block.terminal:
        upd_s, upd_vdot, upd_tdot = update.split((block.scalar_hidden, block.tensor_hidden, block.tensor_hidden), dim=-1)
        return block._scalar_update(scalar, upd_s, upd_vdot * vector_dot, upd_tdot * tensor_dot), vector, tensor
    upd_s, upd_vdot, upd_v, upd_tdot, upd_t = update.split((block.scalar_hidden,) + (block.tensor_hidden,) * 4, dim=-1)
    scalar = block._scalar_update(scalar, upd_s, upd_vdot * vector_dot, upd_tdot * tensor_dot)
    if block.local_cross:
        vector_cross = _channel_tensor_vector(tu, vv)
        tensor_cross = _channel_vector_tensor(vu, vv)
        vector = vector + upd_v[:, None, :] * (vu + torch.tanh(block.vector_tensor_gain)[None, None, :] * vector_cross)
        tensor = tensor + upd_t[:, None, :] * (tu + torch.tanh(block.vector_vector_gain)[None, None, :] * tensor_cross)
    else:
        vector = vector + upd_v[:, None, :] * vu
        tensor = tensor + upd_t[:, None, :] * tu
    return scalar, vector, tensor


def enable_local_features(model, atom_tile=16, statistics='fp64'):
    if statistics not in STATISTICS:
        raise ValueError(f'statistics must be one of {STATISTICS}')
    if any(p.requires_grad for p in model.parameters()) or getattr(model, 'compiled_nodes', False):
        raise ValueError('Freeze the model and select local feature fusion before compiling atom updates')
    for block in model.blocks:
        if not block.narrow_tensors:
            raise RuntimeError('The node update split assumes narrow tensors (tensor_hidden != hidden)')
        block._engine_feature_tile = int(atom_tile)
        block._engine_feature_fp64 = statistics == 'fp64'
        block._node_update = MethodType(_node_update, block)
    model.frozen_local_features = True
    model.frozen_local_feature_atom_tile = int(atom_tile)
    model.frozen_local_feature_statistics = statistics
    return model


__all__ = ['FusedHead', 'DualHead', 'local_features', 'enable_local_features', 'STATISTICS']
