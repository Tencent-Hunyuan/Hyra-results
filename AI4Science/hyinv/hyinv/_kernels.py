"""Triton kernels for accelerated inference (imported only in fast mode). One decoding round = 6 launches:
table sum -> node update (with the next layer's projection) -> edge messages -> node update (with the next
layer's projection) -> edge messages (sampling positions only) -> node update (output head).

Edge messages: sum_k mask_ik * gelu( W2 gelu( R_i + [bw_ik ? Z_{j} : (E+G)_ik] ) + b2 ), where bw (whether
neighbor j is visible to i and precedes it) is computed in the kernel from the state and each edge's static
flags. Z already contains the sender token term, and (E+G) / E come from the two halves of EEG.
Node update: dh = (W3 a + b3 cnt)/scale; x = LN1(h + dh); y = LN2(x + FFN(x)), multiplied by the node mask,
then the next layer's [R|Z] projection or the output head.
state is int32 [B,L] (20 = unknown); flags is uint8 [L,K] (bit0: edge valid and neighbor has geometry,
bit1: neighbor comes earlier in decoding order).
prec: "bf16" (BF16 operands, FP32 accumulation) or "tf32x3" (FP32-level); gelu: 0 = erf, 1 = tanh form
(hardware tanh).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MASK = 20


@triton.jit
def _act(x, GELU: tl.constexpr):
    if GELU == 0:
        return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    else:
        u = x * (0.7978845608028654 + 0.035677408136300125 * x * x)
        t = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [u], dtype=tl.float32, is_pure=True, pack=1)
        return 0.5 * x * (1.0 + t)


@triton.jit
def _dot(x, w, PREC: tl.constexpr, LOWP: tl.constexpr):
    if LOWP:
        return tl.dot(x.to(w.dtype), w, input_precision="ieee")
    else:
        return tl.dot(x, w, input_precision=PREC)


@triton.jit
def _visible_backward(state_ptr, nbr_ptr, flags_ptr, geom_ptr, pb, node, srow, valid, L):
    vis_i = (tl.load(state_ptr + pb * L + node) != 20) & (tl.load(geom_ptr + node) != 0)
    nbr = tl.load(nbr_ptr + srow, mask=valid, other=0)
    sj = tl.load(state_ptr + pb * L + nbr, mask=valid, other=20)
    fl = tl.load(flags_ptr + srow, mask=valid, other=0)
    bw = ((fl & 1) != 0) & (sj != 20) & (((fl & 2) != 0) | (~vis_i)) & valid
    return nbr, sj, bw


@triton.jit
def _edge_kernel(R_ptr, Z_ptr, EEG_ptr, state_ptr, rows_ptr, nbr_ptr, flags_ptr, geom_ptr, mask_ptr, W2_ptr, b2_ptr, out_ptr,
                 n, L, LK, bR, bZ, rs,
                 K: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr, LOWP: tl.constexpr, GELU: tl.constexpr,
                 PER_ROW: tl.constexpr):
    pb = tl.program_id(0)
    i = tl.program_id(1)
    if PER_ROW:
        node = tl.load(rows_ptr + pb * rs + i)
    else:
        node = i
    k = tl.arange(0, BK)
    d = tl.arange(0, D)
    valid = k < K
    srow = node * K + k
    nbr, sj, bw = _visible_backward(state_ptr, nbr_ptr, flags_ptr, geom_ptr, pb, node, srow, valid, L)
    e = tl.load(EEG_ptr + (srow + tl.where(bw, 0, LK))[:, None] * D + d[None, :], mask=valid[:, None], other=0.0)
    zg = tl.load(Z_ptr + pb * bZ + nbr[:, None] * (2 * D) + d[None, :], mask=bw[:, None], other=0.0)
    rv = tl.load(R_ptr + pb * bR + node * (2 * D) + d)
    a = _act(rv.to(tl.float32)[None, :] + e.to(tl.float32) + zg.to(tl.float32), GELU)
    w2t = tl.load(W2_ptr + d[None, :] * D + d[:, None])
    acc = _dot(a, w2t, PREC, LOWP) + tl.load(b2_ptr + d)[None, :]
    a2 = _act(acc, GELU)
    m = tl.load(mask_ptr + srow, mask=valid, other=0.0)
    tl.store(out_ptr + (pb * n + i) * D + d, tl.sum(a2 * m[:, None], axis=0))


@triton.jit
def _table_kernel(T_ptr, state_ptr, rows_ptr, nbr_ptr, flags_ptr, geom_ptr, out_ptr, n, L, rs,
                  K: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, PER_ROW: tl.constexpr):
    pb = tl.program_id(0)
    i = tl.program_id(1)
    if PER_ROW:
        node = tl.load(rows_ptr + pb * rs + i)
    else:
        node = i
    k = tl.arange(0, BK)
    d = tl.arange(0, D)
    valid = k < K
    srow = node * K + k
    _, sj, bw = _visible_backward(state_ptr, nbr_ptr, flags_ptr, geom_ptr, pb, node, srow, valid, L)
    tok = tl.where(bw, sj, 20)
    rows = tl.load(T_ptr + (srow * 21 + tok)[:, None].to(tl.int64) * D + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    tl.store(out_ptr + (pb * n + i) * D + d, tl.sum(rows, axis=0))


@triton.jit
def _node_kernel(asum_ptr, hin_ptr, rows_ptr, cnt_ptr, geom_ptr, W3_ptr, b3_ptr, g1_ptr, be1_ptr, Wi_ptr, bi_ptr, Wo_ptr, bo_ptr, g2_ptr, be2_ptr,
                 Wp_ptr, bp_ptr, ts_ptr, state_ptr, Wout_ptr, bout_ptr, h_ptr, rz_ptr, logit_ptr, M, L, bH, rs, scale, eps,
                 D: tl.constexpr, FF: tl.constexpr, BM: tl.constexpr, PREC: tl.constexpr, LOWP: tl.constexpr, GELU: tl.constexpr,
                 PER_ROW: tl.constexpr, PROJ: tl.constexpr, RZ_BF16: tl.constexpr, NPAD: tl.constexpr):
    pb = tl.program_id(0)
    r = tl.program_id(1) * BM + tl.arange(0, BM)
    rv = r < M
    d = tl.arange(0, D)
    if PER_ROW:
        node = tl.load(rows_ptr + pb * rs + r, mask=rv, other=0)
    else:
        node = r
    a = tl.load(asum_ptr + (pb * M + r)[:, None] * D + d[None, :], mask=rv[:, None], other=0.0)
    w3 = tl.load(W3_ptr + d[None, :] * D + d[:, None])                                   # [in, out]
    cnt = tl.load(cnt_ptr + node, mask=rv, other=0.0)
    dh = (_dot(a, w3, PREC, LOWP) + tl.load(b3_ptr + d)[None, :] * cnt[:, None]) / scale
    x = tl.load(hin_ptr + (pb * bH + node)[:, None] * D + d[None, :], mask=rv[:, None], other=0.0) + dh
    xc = x - (tl.sum(x, axis=1) / D)[:, None]
    x = xc * (1.0 / tl.sqrt(tl.sum(xc * xc, axis=1) / D + eps))[:, None] * tl.load(g1_ptr + d)[None, :] + tl.load(be1_ptr + d)[None, :]
    acc = tl.zeros((BM, D), dtype=tl.float32)
    for c in range(FF // D):
        wi = tl.load(Wi_ptr + (c * D + d[None, :]) * D + d[:, None])                     # [in, hidden chunk]
        hid = _act(_dot(x, wi, PREC, LOWP) + tl.load(bi_ptr + c * D + d)[None, :], GELU)
        acc += _dot(hid, tl.load(Wo_ptr + d[None, :] * FF + (c * D + d[:, None])), PREC, LOWP)
    y = x + acc + tl.load(bo_ptr + d)[None, :]
    yc = y - (tl.sum(y, axis=1) / D)[:, None]
    y = yc * (1.0 / tl.sqrt(tl.sum(yc * yc, axis=1) / D + eps))[:, None] * tl.load(g2_ptr + d)[None, :] + tl.load(be2_ptr + d)[None, :]
    y = y * tl.load(geom_ptr + node, mask=rv, other=0).to(tl.float32)[:, None]
    if PROJ:      # store h and compute the next layer's [R | Z] projection; Z gets this node's own sender token term
        tl.store(h_ptr + (pb * L + node)[:, None] * D + d[None, :], y, mask=rv[:, None])
        for half in tl.static_range(2):
            z = _dot(y, tl.load(Wp_ptr + (half * D + d[None, :]) * D + d[:, None]), PREC, LOWP) + tl.load(bp_ptr + half * D + d)[None, :]
            if half == 1:
                tok = tl.load(state_ptr + pb * L + node, mask=rv, other=20)
                z += tl.load(ts_ptr + tok[:, None] * D + d[None, :])
            if RZ_BF16:
                z = z.to(tl.bfloat16)
            tl.store(rz_ptr + (pb * L + node)[:, None] * (2 * D) + half * D + d[None, :], z, mask=rv[:, None])
    else:         # output head: the first 20 dims are the logits
        o = tl.arange(0, NPAD)
        wq = tl.load(Wout_ptr + o[None, :] * D + d[:, None], mask=(o < 21)[None, :], other=0.0)
        lg = _dot(y, wq, PREC, LOWP) + tl.load(bout_ptr + o, mask=o < 21, other=0.0)[None, :]
        tl.store(logit_ptr + (pb * M + r)[:, None] * 20 + o[None, :], lg, mask=rv[:, None] & (o < 20)[None, :])


def _prec(prec):
    return ("ieee" if prec == "bf16" else prec), prec == "bf16"


def _bs(t):
    """Batch stride in elements: with a single row, treat it as shared by all candidates (stride 0)."""
    return 0 if t.shape[0] == 1 else t[0].numel()


def table_sum(T, state, rows, nbr, flags, geom):
    """Table-sum over the layer-1 token table -> [B,n,D] fp32; rows [B,n] int64 holds the nodes to compute for each candidate (None = all nodes)."""
    B, L = state.shape
    K, D = nbr.shape[1], T.shape[1]
    n = L if rows is None else rows.shape[1]
    out = torch.empty(B, n, D, device=T.device, dtype=torch.float32)
    _table_kernel[(B, n)](T, state, state if rows is None else rows, nbr, flags, geom, out, n, L, 0 if rows is None else rows.stride(0),
                          K=K, D=D, BK=max(16, triton.next_power_of_2(K)), PER_ROW=rows is not None, num_warps=4)
    return out


def edge_sum(rz, EEG, state, rows, nbr, flags, geom, mask, W2, b2, n, *, prec, gelu):
    """rz [B1,L,2D] (shared by all candidates when B1 = 1); rows [B,n] int64 holds the nodes for each candidate (None = all nodes). -> [B,n,D] fp32"""
    B, L = state.shape
    K, D = nbr.shape[1], EEG.shape[1]
    out = torch.empty(B, n, D, device=state.device, dtype=torch.float32)
    p, lowp = _prec(prec)
    _edge_kernel[(B, n)](rz, rz[..., D:], EEG, state, state if rows is None else rows, nbr, flags, geom, mask, W2, b2, out,
                         n, L, L * K, _bs(rz), _bs(rz), 0 if rows is None else rows.stride(0),
                         K=K, D=D, BK=max(16, triton.next_power_of_2(K)), PREC=p, LOWP=lowp, GELU=gelu, PER_ROW=rows is not None,
                         num_warps=4, num_stages=2)
    return out


def node_update(asum, hin, rows, cnt, geom, nw, state, *, scale, prec, gelu, BM, warps, proj=None, ts=None, head=None, rz_dtype=torch.float32):
    """proj = (Wp [2D,D], bp [2D]) and ts [21,D]: returns (h, rz), both [B,L,*] stored by node index;
    head = (Wout, bout): returns logits [B,M,20].
    asum [B,M,D]; hin: [B,L,D], or [L,D] shared by all candidates; rows [B,M] int64 (None = all nodes)."""
    B, M, D = asum.shape
    L = geom.shape[0]
    dev = asum.device
    p, lowp = _prec(prec)
    if proj is not None:
        h = torch.empty(B, L, D, device=dev, dtype=torch.float32)       # stored by node index; rows outside rows are never written or read
        rz = torch.empty(B, L, 2 * D, device=dev, dtype=rz_dtype)
        logit = asum
    else:
        h = rz = None
        logit = torch.empty(B, M, 20, device=dev, dtype=torch.float32)
    Wp, bp = proj if proj is not None else (nw["W3"], nw["b3"])
    Wo, bo = head if head is not None else (nw["W3"], nw["b3"])
    _node_kernel[(B, triton.cdiv(M, BM))](
        asum, hin, state if rows is None else rows, cnt, geom, nw["W3"], nw["b3"], nw["g1"], nw["be1"], nw["Wi"], nw["bi"], nw["Wo"], nw["bo"],
        nw["g2"], nw["be2"], Wp, bp, nw["b3"] if ts is None else ts, state, Wo, bo,
        asum if h is None else h, asum if rz is None else rz, logit, M, L, 0 if hin.dim() == 2 or hin.shape[0] == 1 else L,
        0 if rows is None else rows.stride(0), float(scale), float(nw["eps"]),
        D=D, FF=nw["Wi"].shape[0], BM=BM, PREC=p, LOWP=lowp, GELU=gelu, PER_ROW=rows is not None, PROJ=proj is not None,
        RZ_BF16=rz_dtype == torch.bfloat16, NPAD=32, num_warps=warps, num_stages=1)
    return (h, rz) if proj is not None else logit
