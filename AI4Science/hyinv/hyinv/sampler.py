"""Deployment sampler: encode once -> 8 rounds of blind drafting (draft temperature) -> 8 rounds of
rewriting by fixed groups (rewrite temperature).

Positions are randomly split into 8 groups (determined by the seed, the design id and the length). The draft
stage starts from an all-unknown state and, in each round, predicts and samples the positions of one group
from the current state; the rewrite stage masks one group per round and resamples it from the remaining
positions (including the existing draft). The two stages use independent random streams.
Fixed positions: visible as the native residue from the first round on, and each resampling can only draw
that same residue.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import Tensor

from .model import AA, MASK, Backbone, HyInv
from .pdb import Structure, parse_fixed

DRAFT_SEED_OFFSET = 1_000_003


def group_permutation(sample_id: str, length: int, seed: int) -> Tensor:
    key = int.from_bytes(hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()[:8], "little")
    return torch.randperm(length, generator=torch.Generator().manual_seed(key % (2**63 - 1)))


def fixed_groups(sample_id: str, length: int, rounds: int, seed: int) -> Tensor:
    order = group_permutation(sample_id, length, seed)
    groups = torch.empty(length, dtype=torch.long)
    groups[order] = torch.arange(length) * rounds // length
    return groups


@torch.no_grad()
def sample(model: HyInv, bb: Backbone, ids, *, seed=0, draft_temperature=0.6, rewrite_temperature=0.34,
           rounds=8, pin_mask: Tensor | None = None, pin_tokens: Tensor | None = None) -> Tensor:
    """One design per row of bb, ids holds each row's design id; returns amino-acid indices [B,L]."""
    device = bb.coords.device
    enc = model.encode(bb)
    pinned = pin_mask is not None and bool(pin_mask.any())
    if pinned:
        pm, pt = pin_mask.to(device), pin_tokens.to(device).clamp(0, 19)
        onehot = torch.full((*pt.shape, 20), -1e4, device=device).scatter_(-1, pt[..., None], 0.0)

    def logits_of(state):
        if pinned:   # fixed positions always stay visible as the native residue; their logits become one-hot (every redraw yields the same residue)
            return torch.where(pm[..., None], onehot, model.decode(enc, torch.where(pm, pt, state)))
        return model.decode(enc, state)

    length = bb.coords.shape[1]
    groups = torch.stack([fixed_groups(i, length, rounds, seed) for i in ids]).to(device)
    state = torch.full(groups.shape, MASK, dtype=torch.long, device=device)

    # draft: each round samples the group's positions at the draft temperature (independent random stream)
    gen = torch.Generator(device=device).manual_seed((int(seed) + DRAFT_SEED_OFFSET) % (2**63 - 1))
    for r in range(rounds):
        active = groups.eq(r)
        if active.any():
            probs = (logits_of(state).float() / draft_temperature).softmax(-1)
            state = state.clone()
            state[active] = torch.multinomial(probs[active], 1, generator=gen).squeeze(-1)

    # rewrite: each round masks one group and resamples it at the rewrite temperature
    gen = torch.Generator(device=device).manual_seed(int(seed))
    for r in range(rounds):
        active = groups.eq(r)
        if active.any():
            log_probs = torch.log_softmax(logits_of(torch.where(active, MASK, state)).float() / rewrite_temperature, -1)
            state = state.clone()
            state[active] = torch.multinomial(log_probs[active].exp(), 1, generator=gen).squeeze(-1)
    return state


def design(model: HyInv, st: Structure, n=8, *, seed=0, fixed: str | None = None, draft_temperature=0.6,
           rewrite_temperature=0.34, rounds=8, batch_size=64, fast="off") -> list[str]:
    """Generate n sequences for structure st (multi-chain sequences are joined with "/"). Results are
    reproducible for a given (seed, batch_size, device); batch k uses seed + k, and the batch size is
    automatically tightened by length (at most about 16,384 residues per batch).
    fast: accelerated inference, "off" (reference implementation), "exact" (FP32-level, sequences almost
    always identical to the reference under the same arguments), "bf16" (faster, numerically different,
    CUDA only); batch splitting, seeds and each design's position groups match the reference path."""
    device = next(model.parameters()).device
    if fast != "off":
        from .fast import fast_sample
        run = lambda *a, **kw: fast_sample(model, fast, *a, **kw)
    else:
        run = lambda *a, **kw: sample(model, *a, **kw)
    length = len(st)
    pm = torch.from_numpy(parse_fixed(fixed, st))
    pin_tokens = torch.tensor([AA.find(c) for c in st.native])
    per_batch = max(1, min(batch_size, 16384 // length))
    rep = lambda a, dtype: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)[None].expand(min(per_batch, n), *np.shape(a)).contiguous()
    full = Backbone(rep(st.coords, torch.float32), rep(st.atom_mask, torch.bool), rep(st.residue_index, torch.long),
                    rep(st.chain, torch.long), rep(np.arange(length), torch.long))
    states = []
    for k, start in enumerate(range(0, n, per_batch)):
        ids = [f"{st.name}:{i}" for i in range(start, min(n, start + per_batch))]
        b = len(ids)
        bb = Backbone(**{f: getattr(full, f)[:b] for f in full.__dataclass_fields__})
        kw = dict(refresh=k == 0) if fast != "off" else {}
        states.append(run(bb, ids, seed=seed + k, draft_temperature=draft_temperature, rewrite_temperature=rewrite_temperature,
                          rounds=rounds, pin_mask=pm[None].expand(b, -1) if pm.any() else None,
                          pin_tokens=pin_tokens[None].expand(b, -1), **kw))     # batches are queued asynchronously and fetched at the end
    out = []
    for state in states:
        for row in state.cpu().tolist():
            seq = "".join(AA[x] for x in row)
            out.append("/".join("".join(seq[i] for i in np.flatnonzero(st.chain == c)) for c in range(len(st.chain_names))))
    return out
