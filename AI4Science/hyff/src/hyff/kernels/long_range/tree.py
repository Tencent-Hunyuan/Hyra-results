"""GPU-built frozen-plan treecode for the isolated latent-charge pair energy (sub-quadratic).

Plan (built eagerly on the GPU, host syncs allowed, never during capture):
  * balanced k-d tree, fixed leaves of B atom slots (median split on the longest box axis, left
    child filled first, so only the last real leaf is partial); implicit heap: root 1, leaf l = L+l.
  * frozen node centers (box centers at build) and radii (max member distance + delta).
  * dual-tree traversal from (root, root): node pair far iff r_A+r_B <= theta*d and
    d-r_A-r_B >= rcut (rcut = rcut_sigma*sigma, so erf==1 to <3e-9 for every far atom pair while
    every atom stays within delta of its build position). Leaf pairs failing it are near (exact erf).
  * directed CSR lists, deterministic owner-computes kernels.
Energy (captured each step): P2M (all levels) -> M2L (two-sided degree-p, symmetric) -> L2P (all
ancestors) + near tiles. E_plan = 0.5 sum q_i phi_i; returned (phi, g) are the EXACT partial
derivatives of E_plan for the frozen plan (forces conservative between rebuilds).
Validity: max_i |x_i - x_i(build)| <= delta (checked outside capture; rebuild into same buffers).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from hyff.kernels.long_range import codegen as lq_codegen
from hyff.kernels.long_range.dense import radial_terms


@triton.jit
def _near_kernel(XS, QS, SIGMA, ROWPTR, COLS, PHI, G, B: tl.constexpr):
    """Accumulate exact near terms into PHI/G (which already hold the far part)."""
    leaf = tl.program_id(0)
    s = tl.load(SIGMA).to(tl.float32)
    inv_2s2 = 0.5 / (s * s)
    a = 0.7978845608028654 / s
    a_over_s2 = a / (s * s)
    i = leaf * B + tl.arange(0, B)
    xi = tl.load(XS + 3 * i)
    yi = tl.load(XS + 3 * i + 1)
    zi = tl.load(XS + 3 * i + 2)
    phi = tl.zeros((B,), tl.float32)
    gx = tl.zeros((B,), tl.float32)
    gy = tl.zeros((B,), tl.float32)
    gz = tl.zeros((B,), tl.float32)
    start = tl.load(ROWPTR + leaf)
    end = tl.load(ROWPTR + leaf + 1)
    for e in range(start, end):
        src = tl.load(COLS + e)
        j = src * B + tl.arange(0, B)
        xj = tl.load(XS + 3 * j)
        yj = tl.load(XS + 3 * j + 1)
        zj = tl.load(XS + 3 * j + 2)
        qj = tl.load(QS + j)
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
    tl.store(PHI + i, tl.load(PHI + i) + phi)
    tl.store(G + 3 * i, tl.load(G + 3 * i) + gx)
    tl.store(G + 3 * i + 1, tl.load(G + 3 * i + 1) + gy)
    tl.store(G + 3 * i + 2, tl.load(G + 3 * i + 2) + gz)


def _csr(targets, sources, n_rows):
    order = torch.argsort(targets, stable=True)
    cols = sources[order].to(torch.int32)
    counts = torch.bincount(targets, minlength=n_rows)
    rowptr = torch.zeros(n_rows + 1, dtype=torch.int32, device=targets.device)
    rowptr[1:] = torch.cumsum(counts, 0).to(torch.int32)
    return rowptr, cols


class TreePlan:
    """All plan tensors for one build. Shapes depend only on (N, B) except the CSR columns."""

    def __init__(self, positions, sigma, leaf=32, theta=0.5, delta=1.5, rcut_sigma=6.0):
        x = positions.detach().to(torch.float32)
        dev = x.device
        n = x.shape[0]
        self.n, self.B, self.theta, self.delta = n, leaf, float(theta), float(delta)
        self.sigma, self.rcut = float(sigma), float(rcut_sigma) * float(sigma)
        n_leaves = max(1, -(-n // leaf))
        D = max(1, math.ceil(math.log2(n_leaves)))
        L = 1 << D
        self.n_leaves, self.D, self.L = n_leaves, D, L
        # ---- balanced k-d sort (level synchronous)
        order = torch.arange(n, device=dev)
        k = torch.zeros(n, dtype=torch.long, device=dev)        # node index within current depth
        for depth in range(D):
            nk = 1 << depth
            p = x[order]
            lo = torch.full((nk, 3), float('inf'), device=dev).scatter_reduce(
                0, k[:, None].expand(-1, 3), p, 'amin', include_self=True)
            hi = torch.full((nk, 3), -float('inf'), device=dev).scatter_reduce(
                0, k[:, None].expand(-1, 3), p, 'amax', include_self=True)
            axis = (hi - lo).argmax(1)
            key = p.gather(1, axis[k][:, None]).squeeze(1)
            o1 = torch.argsort(key)
            o2 = torch.argsort(k[o1], stable=True)
            perm = o1[o2]
            order, k = order[perm], k[perm]
            counts = torch.bincount(k, minlength=nk)
            starts = torch.cumsum(counts, 0) - counts
            rank = torch.arange(n, device=dev) - starts[k]
            half = (L * leaf) >> (depth + 1)
            k = 2 * k + (rank >= half).long()
        # k is now the leaf index; atoms of each leaf contiguous in `order`.
        counts = torch.bincount(k, minlength=L)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(n, device=dev) - starts[k]
        slot = k * leaf + rank
        ns = n_leaves * leaf
        self.ns = ns
        if int(slot.max()) >= ns or int(counts[n_leaves:].sum()) != 0:
            raise RuntimeError("k-d packing invariant violated")
        slot_atom = torch.empty(ns, dtype=torch.long, device=dev)
        filled = torch.zeros(ns, dtype=torch.bool, device=dev)
        slot_atom[slot] = order
        filled[slot] = True
        last_first = order[starts[n_leaves - 1]]
        slot_atom[~filled] = last_first                         # padding duplicates a real atom
        self.slot_atom = slot_atom
        self.slot_mask = filled.to(torch.float32)
        self.atom_slot = torch.empty(n, dtype=torch.long, device=dev)
        self.atom_slot[order] = slot
        self.ref_positions = x.clone()
        # ---- frozen node geometry (heap layout, 2L entries, index 0 unused)
        xs = x[slot_atom]
        big = torch.full((L * leaf - ns, 3), float('nan'), device=dev)
        xpad = torch.cat([xs, big]) if L * leaf > ns else xs
        real = torch.cat([filled, torch.zeros(L * leaf - ns, dtype=torch.bool, device=dev)])
        centers = torch.zeros(2 * L, 3, device=dev)
        radii = torch.full((2 * L,), -1.0, device=dev)
        count = torch.zeros(2 * L, dtype=torch.long, device=dev)
        for lev in range(D + 1):
            m = L >> lev
            v = xpad.view(m, leaf << lev, 3)
            rv = real.view(m, leaf << lev)
            lo = torch.where(rv[..., None], v, torch.full_like(v, float('inf'))).amin(1)
            hi = torch.where(rv[..., None], v, torch.full_like(v, -float('inf'))).amax(1)
            c = 0.5 * (lo + hi)
            cnt = rv.sum(1)
            c = torch.where(cnt[:, None] > 0, c, torch.zeros_like(c))
            dist = torch.where(rv, (v - c[:, None, :]).norm(dim=-1), torch.zeros_like(rv, dtype=v.dtype))
            centers[m:2 * m] = c
            radii[m:2 * m] = torch.where(cnt > 0, dist.amax(1) + self.delta, torch.full_like(c[:, 0], -1.0))
            count[m:2 * m] = cnt
        self.centers, self.radii, self.count = centers.contiguous(), radii, count
        # ---- dual-tree traversal
        far, near = [], []
        pairs = torch.tensor([[1, 1]], dtype=torch.long, device=dev)
        theta, rcut = self.theta, self.rcut
        while pairs.numel():
            A, Bn = pairs[:, 0], pairs[:, 1]
            selfp = A == Bn
            leafA, leafB = A >= L, Bn >= L
            d = (centers[A] - centers[Bn]).norm(dim=-1)
            rs = radii[A] + radii[Bn]
            acc = (~selfp) & (rs <= theta * d) & (d - rs >= rcut)
            far.append(pairs[acc])
            nr = (~acc) & leafA & leafB
            near.append(pairs[nr])
            rest = (~acc) & (~nr)
            P, S = pairs[rest], selfp[rest]
            Ps = P[S, 0]
            ch_self = torch.cat([torch.stack([2 * Ps, 2 * Ps], 1), torch.stack([2 * Ps + 1, 2 * Ps + 1], 1),
                                 torch.stack([2 * Ps, 2 * Ps + 1], 1)])
            Q = P[~S]
            qa, qb = Q[:, 0], Q[:, 1]
            split_a = (qa < L) & ((qb >= L) | (radii[qa] >= radii[qb]))
            Qa, Qb = Q[split_a], Q[~split_a]
            ch = torch.cat([ch_self,
                            torch.stack([2 * Qa[:, 0], Qa[:, 1]], 1), torch.stack([2 * Qa[:, 0] + 1, Qa[:, 1]], 1),
                            torch.stack([Qb[:, 0], 2 * Qb[:, 1]], 1), torch.stack([Qb[:, 0], 2 * Qb[:, 1] + 1], 1)])
            keep = (count[ch[:, 0]] > 0) & (count[ch[:, 1]] > 0)
            pairs = ch[keep]
        far = torch.cat(far) if far else torch.zeros(0, 2, dtype=torch.long, device=dev)
        near = torch.cat(near)
        self.n_far_pairs, self.n_near_leaf_pairs = far.shape[0], near.shape[0]
        ft = torch.cat([far[:, 0], far[:, 1]])
        fs = torch.cat([far[:, 1], far[:, 0]])
        self.far_rowptr, self.far_cols = _csr(ft, fs, 2 * L)
        off = near[:, 0] != near[:, 1]
        nt = torch.cat([near[:, 0], near[off, 1]]) - L
        nsrc = torch.cat([near[:, 1], near[off, 0]]) - L
        self.near_rowptr, self.near_cols = _csr(nt, nsrc, n_leaves)
        # exact atom-pair counts (directed, i != j), for diagnostics
        cnt_leaf = count[L:L + n_leaves]
        self.near_atom_pairs = int((cnt_leaf[nt] * cnt_leaf[nsrc]).sum() - cnt_leaf.sum())

    def max_displacement(self, positions):
        return (positions.detach().to(torch.float32) - self.ref_positions).norm(dim=-1).max()

    def stats(self):
        return dict(atoms=self.n, leaf=self.B, leaves=self.n_leaves, depth=self.D, theta=self.theta,
                    delta=self.delta, sigma=self.sigma, rcut=self.rcut, far_node_pairs=self.n_far_pairs,
                    near_leaf_pairs=self.n_near_leaf_pairs, near_directed_atom_pairs=self.near_atom_pairs,
                    dense_directed_atom_pairs=self.n * (self.n - 1),
                    far_csr=int(self.far_cols.numel()), near_csr=int(self.near_cols.numel()))


class TreeBackend:
    """Static-buffer evaluator; rebuild() refreshes the plan IN PLACE when capacities allow."""
    name = "tree_kd_cartesian_triton_fp32"

    def __init__(self, order=6, leaf=32, theta=0.4, delta=1.5, rcut_sigma=6.0, headroom=1.25,
                 m2l_block=16):
        self.p, self.leaf, self.theta, self.delta = order, leaf, theta, delta
        self.rcut_sigma, self.headroom, self.m2l_block = rcut_sigma, headroom, m2l_block
        self.gen = lq_codegen.load(order)
        self.nm = self.gen.NM
        self.plan = None
        self.buffers = None
        self.rebuilds = 0
        self.reallocations = 0
        self.last_build_ms = None

    # -- plan management (eager only)
    def _alloc(self, plan):
        dev = plan.centers.device
        cap_far = max(16, int(plan.far_cols.numel() * self.headroom))
        cap_near = max(16, int(plan.near_cols.numel() * self.headroom))
        L, D, ns = plan.L, plan.D, plan.ns
        b = dict(
            slot_atom=torch.zeros(ns, dtype=torch.long, device=dev),
            slot_mask=torch.zeros(ns, device=dev),
            atom_slot=torch.zeros(plan.n, dtype=torch.long, device=dev),
            centers=torch.zeros(2 * L, 3, device=dev),
            far_rowptr=torch.zeros(2 * L + 1, dtype=torch.int32, device=dev),
            far_cols=torch.zeros(cap_far, dtype=torch.int32, device=dev),
            near_rowptr=torch.zeros(plan.n_leaves + 1, dtype=torch.int32, device=dev),
            near_cols=torch.zeros(cap_near, dtype=torch.int32, device=dev),
            part=torch.zeros(D + 1, L, self.nm, device=dev),
            mom=torch.zeros(2 * L, self.nm, device=dev),
            loc=torch.zeros(2 * L, self.nm, device=dev),
            xs=torch.zeros(ns, 3, device=dev), qs=torch.zeros(ns, device=dev),
            phi_s=torch.zeros(ns, device=dev), g_s=torch.zeros(ns, 3, device=dev))
        self.buffers = b
        self.shape_key = (plan.n, L, D, ns, plan.n_leaves)
        self.reallocations += 1

    def _load(self, plan):
        b = self.buffers
        for k in ('slot_atom', 'slot_mask', 'atom_slot', 'centers', 'far_rowptr', 'near_rowptr'):
            b[k].copy_(getattr(plan, k))
        b['far_cols'][:plan.far_cols.numel()].copy_(plan.far_cols)
        b['near_cols'][:plan.near_cols.numel()].copy_(plan.near_cols)
        self.plan = plan

    def rebuild(self, positions, sigma):
        """Build a plan for `positions`; returns True iff static buffers had to be reallocated."""
        t0 = torch.cuda.Event(True)
        t1 = torch.cuda.Event(True)
        t0.record()
        plan = TreePlan(positions, float(sigma), self.leaf, self.theta, self.delta, self.rcut_sigma)
        b = self.buffers
        realloc = (b is None or self.shape_key != (plan.n, plan.L, plan.D, plan.ns, plan.n_leaves)
                   or plan.far_cols.numel() > b['far_cols'].numel()
                   or plan.near_cols.numel() > b['near_cols'].numel())
        if realloc:
            self._alloc(plan)
        self._load(plan)
        t1.record()
        t1.synchronize()
        self.last_build_ms = t0.elapsed_time(t1)
        self.rebuilds += 1
        return realloc

    def needs_rebuild(self, positions, sigma=None):
        if self.plan is None:
            return True
        if sigma is not None and abs(float(sigma) - self.plan.sigma) > 1e-6 * self.plan.sigma:
            return True
        if positions.shape[0] != self.plan.n:
            return True
        return bool(self.plan.max_displacement(positions) > self.delta)

    # -- evaluation (capturable)
    def evaluate(self, positions, charges, sigma):
        b, plan, g = self.buffers, self.plan, self.gen
        if plan is None:
            raise RuntimeError("tree plan not built (call rebuild outside capture)")
        x = positions.to(torch.float32)
        torch.index_select(x, 0, b['slot_atom'], out=b['xs'])
        torch.mul(charges.to(torch.float32).index_select(0, b['slot_atom']), b['slot_mask'], out=b['qs'])
        L, D, nl, B = plan.L, plan.D, plan.n_leaves, plan.B
        g.p2m_kernel[(nl,)](b['xs'], b['qs'], b['centers'], b['part'], L, D, B=B, num_warps=1)
        mom, part = b['mom'], b['part']
        for lev in range(D + 1):
            m = L >> lev
            torch.sum(part[lev].view(m, 1 << lev, self.nm), 1, out=mom[m:2 * m])
        g.m2l_kernel[(2 * L,)](b['centers'], mom, b['far_rowptr'], b['far_cols'], b['loc'],
                               BLOCK=self.m2l_block, num_warps=1)
        g.l2p_kernel[(nl,)](b['xs'], b['centers'], b['loc'], b['phi_s'], b['g_s'], L, D, B=B, num_warps=1)
        s = torch.as_tensor(sigma, device=x.device).reshape(1).to(torch.float32)
        _near_kernel[(nl,)](b['xs'], b['qs'], s, b['near_rowptr'], b['near_cols'], b['phi_s'], b['g_s'],
                            B=B, num_warps=4)
        return b['phi_s'].index_select(0, b['atom_slot']), b['g_s'].index_select(0, b['atom_slot'])

    def describe(self):
        d = dict(backend=self.name, order=self.p, leaf=self.leaf, theta=self.theta, delta=self.delta,
                 rcut_sigma=self.rcut_sigma, headroom=self.headroom, rebuilds=self.rebuilds,
                 reallocations=self.reallocations, last_build_ms=self.last_build_ms)
        if self.plan is not None:
            d['plan'] = self.plan.stats()
        return d


def plan_reference(plan, positions, charges, sigma, p):
    """FP64 torch evaluation of the SAME plan energy (validates kernels; small systems only)."""
    from hyff.kernels.long_range.dense import radial_reference
    dev = positions.device
    x = positions.double()
    q = charges.double()
    mi = lq_codegen.multi_indices(p)
    E = torch.tensor(mi, device=dev)                                # (nm,3)
    L, D, B, nl = plan.L, plan.D, plan.B, plan.n_leaves
    xs = x[plan.slot_atom]
    qs = q[plan.slot_atom] * plan.slot_mask.double()
    c = plan.centers.double()
    leaf_of_slot = torch.arange(plan.ns, device=dev) // B

    def mono(a):
        return (a[:, None, :] ** E[None]).prod(-1)                      # (n,nm)
    mom = torch.zeros(2 * L, len(mi), dtype=torch.float64, device=dev)
    for lev in range(D + 1):
        anc = (L >> lev) + (leaf_of_slot >> lev)
        mom.index_add_(0, anc, qs[:, None] * mono(xs - c[anc]))
    terms = lq_codegen.m2l_terms(p)
    idx = {a: i for i, a in enumerate(mi)}
    ta = torch.tensor([idx[t[0]] for t in terms], device=dev)
    tb = torch.tensor([idx[t[1]] for t in terms], device=dev)
    tg = torch.tensor([idx[t[2]] for t in terms], device=dev)
    tc = torch.tensor([float(t[3]) for t in terms], dtype=torch.float64, device=dev)
    rowptr = plan.far_rowptr.long()
    tgt = torch.repeat_interleave(torch.arange(2 * L, device=dev), rowptr[1:] - rowptr[:-1])
    src = plan.far_cols.long()
    loc = torch.zeros_like(mom)
    if src.numel():
        bco = _taylor_reference(c[tgt] - c[src], mi)
        contrib = tc[None] * bco[:, tg] * mom[src][:, tb]
        lt = torch.zeros(src.numel(), len(mi), dtype=torch.float64, device=dev).index_add_(1, ta, contrib)
        loc.index_add_(0, tgt, lt)
    phi = torch.zeros(plan.ns, dtype=torch.float64, device=dev)
    gr = torch.zeros(plan.ns, 3, dtype=torch.float64, device=dev)
    for lev in range(D + 1):
        anc = (L >> lev) + (leaf_of_slot >> lev)
        a = (xs - c[anc]).requires_grad_(True)
        val = (loc[anc] * mono(a)).sum(-1)
        ga, = torch.autograd.grad(val.sum(), a)
        phi += val.detach()
        gr += ga
    nrow = plan.near_rowptr.long()
    nt = torch.repeat_interleave(torch.arange(nl, device=dev), nrow[1:] - nrow[:-1])
    nsrc = plan.near_cols.long()
    off = torch.arange(B, device=dev)
    for c0 in range(0, nt.numel(), 4096):
        ti = (nt[c0:c0 + 4096, None] * B + off)                       # (P,B)
        sj = (nsrc[c0:c0 + 4096, None] * B + off)
        d = xs[ti][:, :, None, :] - xs[sj][:, None, :, :]
        k, h = radial_reference(d.square().sum(-1), sigma)
        w = qs[sj][:, None, :].expand_as(k) * (ti[:, :, None] != sj[:, None, :])
        phi.index_add_(0, ti.reshape(-1), (w * k).sum(-1).reshape(-1))
        gr.index_add_(0, ti.reshape(-1), ((w * h)[..., None] * d).sum(-2).reshape(-1, 3))
    return phi[plan.atom_slot], gr[plan.atom_slot]


def _taylor_reference(R, mi):
    """b_g(R) = D^g(1/|R|)/g! by the recursion, float64 (P,nm)."""
    idx = {a: i for i, a in enumerate(mi)}
    inv_r2 = 1.0 / R.square().sum(-1)
    out = [None] * len(mi)
    out[0] = inv_r2.sqrt()
    for gi, gm in enumerate(mi[1:], 1):
        n = sum(gm)
        acc = torch.zeros_like(inv_r2)
        for i in range(3):
            if gm[i] >= 1:
                lo = list(gm)
                lo[i] -= 1
                acc = acc + (2 * n - 1) * R[:, i] * out[idx[tuple(lo)]]
            if gm[i] >= 2 and n > 1:
                lo = list(gm)
                lo[i] -= 2
                acc = acc + (n - 1) * out[idx[tuple(lo)]]
        out[gi] = -acc * inv_r2 / n
    return torch.stack(out, -1)
