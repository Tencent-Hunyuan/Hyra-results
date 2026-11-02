"""Accelerated inference (CUDA only): the same sampling procedure (8 rounds of blind drafting + 8 rounds of
rewriting, per-design position groups, fixed positions) with the decoder replaced by Triton kernels, and all
16 rounds captured in a single CUDA graph. Two modes:

* "exact": FP32-level (TF32x3 accumulation, erf GELU, FP32 storage). Sequences are almost always identical to
  the reference path under the same arguments (TF32x3 accumulates in a different order than FP32, so when the
  sampling probabilities are nearly tied a few positions may flip).
* "bf16": BF16 operands + FP32 accumulation, tanh GELU, BF16 storage. Numerically different: sequences are not
  identical to the reference path.

Backbone-dependent quantities (encoding, edge terms, the layer-1 token table) live in one "prep graph" per
length; sampling uses one "sampling graph" per (length, batch size, temperatures, number of fixed positions).
Only one prepared backbone is kept per length at a time, with at most 4 lengths cached (least recently used
evicted). The first call compiles the Triton kernels and captures the graphs; later calls only replay them.
"""
from __future__ import annotations

import collections

import numpy as np
import torch
import torch.nn.functional as F

from .model import MASK, Backbone
from .sampler import DRAFT_SEED_OFFSET, group_permutation

MODES = {"exact": dict(prec="tf32x3", store=torch.float32, gelu=0, BM=32, warps=8),
         "bf16": dict(prec="bf16", store=torch.bfloat16, gelu=1, BM=64, warps=4)}
_FIELDS = ("coords", "atom_mask", "residue_index", "chain_id", "slot_index")
_MAX_LENGTHS = 4


def _kernels():
    try:
        from . import _kernels as K
    except ImportError as e:
        raise ImportError("accelerated inference needs triton (and a CUDA build of PyTorch): pip install 'hyinv[fast]'") from e
    return K


def _capture(fn, warmup=2):
    """Warm up fn on a side stream (triggering Triton compilation), then capture it as a CUDA graph; returns (graph, fn's output)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    return graph, out


class _Decoder:
    """Decoder for one backbone: backbone-only quantities (edge terms, layer-1 token table) plus low-precision weight copies."""

    @torch.no_grad()
    def __init__(self, model, enc, cfg):
        D = self.D = model.dim
        h0, e, nbr, ev = enc["h"][0], enc["edge"][0], enc["nbr"][0], enc["edge_valid"][0]
        geom, prio = enc["valid"][0], enc["prio"][0]
        self.cfg, self.L, self.h0, self.scale = cfg, h0.shape[0], h0, model.decoder_layers[0].scale
        opd = torch.bfloat16 if cfg["prec"] == "bf16" else torch.float32
        self.nbr = nbr.to(torch.int32).contiguous()
        self.flags = ((ev & geom[nbr]).to(torch.uint8) | ((prio[nbr] < prio[:, None]).to(torch.uint8) << 1)).contiguous()
        self.geom = geom.to(torch.uint8).contiguous()
        self.cnt = ev.float().sum(-1)
        self.mask = ev.float().contiguous()
        self.eeg, self.w2, self.b2, self.ts, self.proj, self.nw = [], [], [], [], [], []
        emb = model.W_s.weight
        for layer in model.decoder_layers:
            Wr, We, Ws, Wh = (w.detach() for w in layer.W1.weight.split(D, dim=1))     # input layout [h_i | e_ij | s_j | h_j]
            b1 = layer.W1.bias.detach()
            E, G, TS = F.linear(e, We), F.linear(h0, Wh)[nbr], F.linear(emb, Ws)
            if not self.eeg:                                                           # layer 1: precompute messages keyed by neighbor token
                self.table = self._layer1_table(layer, F.linear(h0, Wr, b1)[:, None, :] + E + G, TS, cfg["store"])
            # E (backward edges) and E+G (all other edges) are stacked together; the kernel picks one half per edge
            self.eeg.append(torch.cat([E.to(cfg["store"]).reshape(-1, D), (E + G).to(cfg["store"]).reshape(-1, D)]).contiguous())
            self.w2.append(layer.W2.weight.detach().to(opd).contiguous())
            self.b2.append(layer.W2.bias.detach().float().contiguous())
            self.ts.append(TS.float().contiguous())
            self.proj.append((torch.cat([Wr, Wh]).to(opd).contiguous(), torch.cat([b1, torch.zeros_like(b1)]).float().contiguous()))
            f32 = lambda t: t.detach().float().contiguous()
            self.nw.append(dict(W3=layer.W3.weight.detach().to(opd).contiguous(), b3=f32(layer.W3.bias),
                                g1=f32(layer.norm1.weight), be1=f32(layer.norm1.bias),
                                Wi=layer.dense.W_in.weight.detach().to(opd).contiguous(), bi=f32(layer.dense.W_in.bias),
                                Wo=layer.dense.W_out.weight.detach().to(opd).contiguous(), bo=f32(layer.dense.W_out.bias),
                                g2=f32(layer.norm2.weight), be2=f32(layer.norm2.bias), eps=layer.norm1.eps))
        self.head = (model.W_out.weight.detach().to(opd).contiguous(), model.W_out.bias.detach().float().contiguous())

    def _layer1_table(self, layer, base, TS, store):
        """T[i,k,a]: the layer-1 message (before W3); a = neighbor token, where a = 20 means the neighbor is not a backward edge."""
        T = torch.empty(*base.shape[:2], 21, self.D, device=base.device)
        for a in range(21):
            hidden = base if a == 20 else base + TS[a]
            T[:, :, a] = layer.act(F.linear(layer.act(hidden), layer.W2.weight, layer.W2.bias)) * self.mask[..., None]
        return T.view(-1, self.D).to(store).contiguous()

    @torch.no_grad()
    def decode(self, state, act, rows=None, share=False):
        """state [B,L] long (20 = unknown), act [B,m]: each candidate's own sampling positions -> logits [B,m,20] fp32.
        rows [B,n]: the only nodes layers 1 and 2 need (draft stage: known positions plus this round's positions; None = all);
        share: all candidates have the same state (round 0), so layers 1 and 2 are computed once."""
        K, c, L = _kernels(), self.cfg, self.L
        st = state.to(torch.int32).contiguous()
        s1 = st[:1] if share else st
        kw = dict(prec=c["prec"], gelu=c["gelu"])
        nk = dict(scale=self.scale, BM=c["BM"], warps=c["warps"], rz_dtype=c["store"], **kw)
        n = L if rows is None else rows.shape[1]
        asum = K.table_sum(self.table, s1, rows, self.nbr, self.flags, self.geom)
        h, rz = K.node_update(asum, self.h0, rows, self.cnt, self.geom, self.nw[0], s1, proj=self.proj[1], ts=self.ts[1], **nk)
        asum = K.edge_sum(rz, self.eeg[1], s1, rows, self.nbr, self.flags, self.geom, self.mask, self.w2[1], self.b2[1], n, **kw)
        h, rz = K.node_update(asum, h, rows, self.cnt, self.geom, self.nw[1], s1, proj=self.proj[2], ts=self.ts[2], **nk)
        asum = K.edge_sum(rz, self.eeg[2], st, act, self.nbr, self.flags, self.geom, self.mask, self.w2[2], self.b2[2], act.shape[1], **kw)
        return K.node_update(asum, h, act, self.cnt, self.geom, self.nw[2], st, head=self.head, **{**nk, "BM": 16})


class _Prep:
    """Prep graph for one length: encoding + structure terms + layer-1 table. refresh copies a new backbone into the static buffers and replays."""

    def __init__(self, model, cfg, bb):
        self.static = Backbone(*(getattr(bb, f)[:1].clone() for f in _FIELDS))
        self.graph, self.dec = _capture(lambda: _Decoder(model, model.encode(self.static), cfg))
        self.rounds = {}
        self.graph.replay()

    def refresh(self, bb):
        for f in _FIELDS:
            getattr(self.static, f).copy_(getattr(bb, f)[:1])
        self.graph.replay()


class _Rounds:
    """All 16 sampling rounds (everything except the noise) for one (length, batch size, temperatures,
    number of fixed positions), captured as a single CUDA graph.
    Each design has its own position grouping: order[b] sorts the positions by (group, position), group r
    occupies [off_r, off_r+m_r), and every row has groups of the same sizes.
    In draft round r only the known positions plus this round's positions are read later, so layers 1 and 2
    compute only those nodes: the fixed positions pn (padded to p_pad) plus the first off_{r+1} entries of order."""

    def __init__(self, dec, B, draft_temperature, rewrite_temperature, rounds, p_pad):
        L, dev = dec.L, dec.h0.device
        self.dec, self.B, self.rounds, self.T = dec, B, rounds, (draft_temperature, rewrite_temperature)
        self.m = np.bincount(np.arange(L) * rounds // L, minlength=rounds)
        self.off = np.concatenate([[0], np.cumsum(self.m)])
        self.order = torch.zeros(B, L, dtype=torch.long, device=dev)
        self.pn = torch.zeros(1, p_pad, dtype=torch.long, device=dev)
        self.pm = torch.zeros(1, L, dtype=torch.bool, device=dev)
        self.pt = torch.zeros(1, L, dtype=torch.long, device=dev)
        self.state = torch.zeros(B, L, dtype=torch.long, device=dev)
        self.noise = [[torch.ones(B * int(m), 20, device=dev) for m in self.m] for _ in range(2)]
        self.graph, _ = _capture(self.run)

    def _pinned(self, act, logits):                     # fixed positions: logits become one-hot (every redraw yields the same residue)
        pa, ta = self.pm.expand(self.B, -1).gather(1, act), self.pt.expand(self.B, -1).gather(1, act)
        return torch.where(pa[..., None], torch.where(torch.arange(20, device=act.device) == ta[..., None], 0.0, -1e4), logits)

    def run(self):
        B, L, state = self.B, self.dec.L, self.state
        state.copy_(torch.where(self.pm, self.pt, MASK).expand(B, -1))
        for stage, T in enumerate(self.T):
            for r in range(self.rounds):
                if self.m[r] == 0:
                    continue
                act = self.order[:, self.off[r]:self.off[r + 1]]
                if stage == 0:
                    n = self.pn.shape[1] + self.off[r + 1]
                    rows = None if r == 0 or n >= L else torch.cat([self.pn.expand(B, -1), self.order[:, :self.off[r + 1]]], 1)
                    logits = self.dec.decode(state, act, rows, share=(r == 0))
                    probs = (self._pinned(act, logits) / T).softmax(-1)
                else:
                    masked = torch.where(self.pm, self.pt, state.scatter(1, act, MASK))
                    probs = torch.log_softmax(self._pinned(act, self.dec.decode(masked, act)) / T, -1).exp()
                # same as torch.multinomial(probs, 1, generator): argmax(probs / Exp(1) noise)
                state.scatter_(1, act, (probs.reshape(-1, 20) / self.noise[stage][r]).argmax(-1).view(B, -1))
        return state

    def sample(self, order, pn, pm, pt, seed):
        self.order.copy_(order)
        self.pn.copy_(pn)
        self.pm.copy_(pm)
        self.pt.copy_(pt)
        for stage, s in enumerate(((seed + DRAFT_SEED_OFFSET) % (2**63 - 1), int(seed))):    # two independent random streams, draft / rewrite
            gen = torch.Generator(device=self.order.device).manual_seed(s)
            for r in range(self.rounds):
                if self.m[r]:
                    self.noise[stage][r].exponential_(1, generator=gen)
        self.graph.replay()
        return self.state.clone()


class _Engine:
    def __init__(self, model, mode):
        self.model, self.cfg = model, MODES[mode]
        self.preps = collections.OrderedDict()

    @torch.no_grad()
    def sample(self, bb, ids, *, seed, draft_temperature, rewrite_temperature, rounds, pin_mask, pin_tokens, refresh):
        L, B, dev = bb.coords.shape[1], len(ids), bb.coords.device
        prep = self.preps.pop(L, None)
        if prep is None:
            prep, refresh = _Prep(self.model, self.cfg, bb), False       # a fresh prep has already replayed bb
            while len(self.preps) >= _MAX_LENGTHS:
                self.preps.popitem(last=False)
        self.preps[L] = prep
        if refresh:
            prep.refresh(bb)
        pm = torch.zeros(1, L, dtype=torch.bool) if pin_mask is None else pin_mask[:1].cpu()
        pt = torch.zeros(1, L, dtype=torch.long) if pin_tokens is None else pin_tokens[:1].cpu().clamp(0, 19)
        pn = pm[0].nonzero().squeeze(-1)
        pn = torch.cat([pn, pn.new_zeros(-len(pn) % 8)])[None]           # pad the number of fixed positions to a multiple of 8 (the padded nodes are arbitrary and harmless)
        key = (B, draft_temperature, rewrite_temperature, rounds, pn.shape[1])
        if key not in prep.rounds:
            prep.rounds[key] = _Rounds(prep.dec, B, draft_temperature, rewrite_temperature, rounds, pn.shape[1])
        perm = torch.stack([group_permutation(i, L, seed) for i in ids])
        off = prep.rounds[key].off
        order = torch.cat([perm[:, off[r]:off[r + 1]].sort(dim=1).values for r in range(rounds)], 1)   # = stable sort by (group, position)
        return prep.rounds[key].sample(order.to(dev), pn.to(dev), pm.to(dev), pt.to(dev), int(seed))


def fast_sample(model, mode, bb, ids, *, seed=0, draft_temperature=0.6, rewrite_temperature=0.34, rounds=8,
                pin_mask=None, pin_tokens=None, refresh=True):
    """Same interface and return value as sampler.sample ([B,L] amino-acid indices, returned asynchronously);
    every row of bb must be the same backbone (which is how design calls it).
    refresh=False: bb is the same backbone as the previous call of this length, so skip the prep-graph replay
    (design does this for later batches of the same structure)."""
    if mode not in MODES:
        raise ValueError(f"fast mode must be one of {sorted(MODES)}, got {mode!r}")
    if not bb.coords.is_cuda:
        raise ValueError("accelerated inference (exact / bf16) requires CUDA")
    _kernels()
    engines = model.__dict__.setdefault("_fast_engines", {})
    engine = engines.get(mode) or engines.setdefault(mode, _Engine(model, mode))
    return engine.sample(bb, ids, seed=seed, draft_temperature=draft_temperature, rewrite_temperature=rewrite_temperature,
                         rounds=rounds, pin_mask=pin_mask, pin_tokens=pin_tokens, refresh=refresh)
