"""
Single-file, single-GPU language model pretraining script.

Architecture: 9-layer, 768-dim transformer with ReLU^2 MLP, sliding-window
attention (short/long window pattern), Muon + AdamW optimizer with per-component
learning rates, QK-norm, RoPE, value embeddings, and softcap.

Feature stack: n-gram hash features (bigram + avalanche-hashed sparse trigram with
frequency-aware per-row learning rate), n-gram value-path injection, intra-document
position buckets, a causal induction head, and a learned multi-layer readout with
per-head attention temperature. Trained within a fixed wall-clock budget.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_sdpa_fallback():
    def _sdpa(q, k, v, causal=True, window_size=(-1, -1)):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return out.transpose(1, 2).contiguous()
    return _sdpa

try:
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 10:
        from flash_attn.cute import flash_attn_func as _fa4_raw
        from flash_attn.cute.interface import _flash_attn_bwd as _fa4_bwd_raw

        @torch.library.custom_op("fa4::fa4_causal", mutates_args=())
        def _fa4_causal_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           window_left: int) -> tuple[torch.Tensor, torch.Tensor]:
            ws = (window_left, 0) if window_left > 0 else (None, None)
            out, lse = _fa4_raw(q, k, v, causal=True, window_size=ws, return_lse=True)
            return out, lse

        @_fa4_causal_op.register_fake
        def _fa4_causal_fake(q, k, v, window_left):
            B, T, H, D = q.shape
            return torch.empty_like(q), torch.empty(B, H, T, device=q.device, dtype=torch.float32)

        def _fa4_setup_context(ctx, inputs, output):
            q, k, v, window_left = inputs
            out, lse = output
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.window_left = window_left

        @torch.library.custom_op("fa4::fa4_bwd", mutates_args=())
        def _fa4_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        out: torch.Tensor, grad_output: torch.Tensor, lse: torch.Tensor,
                        window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            wl = window_left if window_left > 0 else None
            dq, dk, dv = _fa4_bwd_raw(
                q, k, v, out, grad_output, lse,
                causal=True, window_size_left=wl, window_size_right=0,
            )
            return dq, dk, dv

        @_fa4_bwd_op.register_fake
        def _fa4_bwd_fake(q, k, v, out, grad_output, lse, window_left):
            return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        def _fa4_backward(ctx, grad_output, grad_lse):
            q, k, v, out, lse = ctx.saved_tensors
            dq, dk, dv = torch.ops.fa4.fa4_bwd(q, k, v, out, grad_output, lse, ctx.window_left)
            return dq, dk, dv, None

        _fa4_causal_op.register_autograd(_fa4_backward, setup_context=_fa4_setup_context)

        def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
            wl = window_size[0] if isinstance(window_size, tuple) else window_size
            if wl is None or wl <= 0 or wl >= q.shape[1]:
                wl = -1
            out, _lse = torch.ops.fa4.fa4_causal(q, k, v, wl)
            return out

        print(f"Using flash-attn-4 as custom op (GPU capability {cap})")
    else:
        from kernels import get_kernel
        repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
        flash_attn_func = get_kernel(repo).flash_attn_interface.flash_attn_func
        print(f"Using flash-attn-3 from {repo} (GPU capability {cap})")
except Exception as e:
    print(f"flash_attn unavailable ({e}), falling back to PyTorch SDPA")
    flash_attn_func = _make_sdpa_fallback()

from lib import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

TRIGRAM_TABLE_SIZE = int(os.environ.get("TRIGRAM_TABLE_SIZE", 27_000_000))
BIGRAM_TABLE_SIZE_ENV = int(os.environ.get("BIGRAM_TABLE_SIZE", 16777216))                            
BIGRAM_FACTORED = os.environ.get("BIGRAM_FACTORED", "1") == "1"                                             
BIGRAM_PRIME     = 31337
TRIGRAM_PRIME    = 1009

USE_AVALANCHE_TRIGRAM = os.environ.get("USE_AVALANCHE_TRIGRAM", "1") == "1"

USE_AVALANCHE_BIGRAM = os.environ.get("USE_AVALANCHE_BIGRAM", "0") == "1"

USE_INTRADOC_POS = os.environ.get("USE_INTRADOC_POS", "0") == "1"

USE_BIGRAM_INDUCTION = os.environ.get("USE_BIGRAM_INDUCTION", "0") == "1"
BIGRAM_INDUCTION_LR  = float(os.environ.get("BIGRAM_INDUCTION_LR", 0.3))                                  
INDUCTION_DOCMASK = os.environ.get("INDUCTION_DOCMASK", "0") == "1"                                                 
INDUCTION_ORDER = os.environ.get("INDUCTION_ORDER", "backoff").strip().lower()
USE_UNIGRAM_INDUCTION = os.environ.get("USE_UNIGRAM_INDUCTION", "0") == "1"
UNIGRAM_INDUCTION_LR = float(os.environ.get("UNIGRAM_INDUCTION_LR", 0.3))
UNIGRAM_INDUCTION_DOCMASK = os.environ.get("UNIGRAM_INDUCTION_DOCMASK", "1") == "1"
USE_INDUCTION_DIST = os.environ.get("USE_INDUCTION_DIST", "0") == "1"
NUM_INDUCTION_DIST_BUCKETS = int(os.environ.get("NUM_INDUCTION_DIST_BUCKETS", 32))
INDUCTION_DIST_LR = float(os.environ.get("INDUCTION_DIST_LR", 0.15))                                            
COMPILE_LOOKUPS = os.environ.get("COMPILE_LOOKUPS", "1") == "1"

USE_BIGRAM_RECALL = os.environ.get("USE_BIGRAM_RECALL", "0") == "1"
BIGRAM_RECALL_LR = float(os.environ.get("BIGRAM_RECALL_LR", 0.4))                                           

USE_LM_HEAD_BIAS   = os.environ.get("USE_LM_HEAD_BIAS", "0") == "1"
LM_HEAD_BIAS_INIT  = os.environ.get("LM_HEAD_BIAS_INIT", "1") == "1"
LM_HEAD_BIAS_LR    = float(os.environ.get("LM_HEAD_BIAS_LR", "0.03"))

USE_ATTN_SCALE = os.environ.get("USE_ATTN_SCALE", "0") == "1"
ATTN_SCALE_LR  = float(os.environ.get("ATTN_SCALE_LR", "0.02"))

USE_LAYER_READOUT = os.environ.get("USE_LAYER_READOUT", "0") == "1"
LAYER_READOUT_LR  = float(os.environ.get("LAYER_READOUT_LR", "0.01"))

USE_READOUT_NORM  = os.environ.get("USE_READOUT_NORM", "0") == "1"

USE_DENSE_DWA = os.environ.get("USE_DENSE_DWA", "0") == "1"
DENSE_DWA_LR  = float(os.environ.get("DENSE_DWA_LR", "0.01"))
DENSE_DWA_K   = int(os.environ.get("DENSE_DWA_K", "8"))                                      

BIGRAM_VE_LAYERS = tuple(
    int(x) for x in os.environ.get("BIGRAM_VE_LAYERS", "").replace(",", " ").split()
)

LONG_LAYERS = tuple(
    int(x) for x in os.environ.get("LONG_LAYERS", "").replace(",", " ").split()
)

NUM_POS_BUCKETS  = 32                                             

NGRAM_VE_LAYERS  = (6, 7, 8)                                                                             

def _fmix32(h):
    """MurmurHash3 fmix32 avalanche finalizer on an int64 tensor holding a 32-bit
    value. Every input bit is diffused across all 32 output bits. int64 mul wraps
    mod 2^64; masking to 32 bits after each mul extracts the correct low bits of
    the (possibly overflowing signed-int64) 32x32 product - verified 0 mismatches
    vs a pure-Python reference. Deterministic, near-zero FLOP."""
    h = h & 0xFFFFFFFF
    h = h ^ (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h

def compute_trigram_keys(idx, table_size):
    """Causal trigram hash key (prev2, prev1, cur) -> [0, table_size).
    USE_AVALANCHE_TRIGRAM=True: pack (prev1,cur) into 26 bits, fmix32-diffuse,
    fold prev2 (scaled by the golden-ratio constant 0x9e3779b1) and fmix32 again,
    so distinct trigrams spread ~uniformly across the table (a free effective-
    capacity increase). Else the linear structured form.
    BOTH are CAUSAL - use only prev tokens {j-2,j-1,j} <= j, never a future token -
    and run in the no_grad lookup path. Deterministic; int64 mul wraps mod 2^64 but
    each pass masks to 32 bits so the index is valid in [0, table_size). Handles
    power-of-2 (bit-mask) and non-power-of-2 (modulo) table sizes identically."""
    B, T = idx.shape
    prev1 = torch.cat([idx.new_zeros(B, 1), idx[:, :-1]], dim=1).long()
    prev2 = torch.cat([idx.new_zeros(B, 2), idx[:, :-2]], dim=1).long()
    if USE_AVALANCHE_TRIGRAM:
        cur = idx.long()
        h = _fmix32(((prev1 << 13) | cur) & 0xFFFFFFFF)                                          
        h = _fmix32(h ^ ((prev2 * 0x9e3779b1) & 0xFFFFFFFF))                                
    else:
        h2 = prev1 * BIGRAM_PRIME + idx.long()
        h = prev2 * TRIGRAM_PRIME + h2

    if (table_size & (table_size - 1)) == 0:
        return h & (table_size - 1)
    return h % table_size

def compute_bigram_keys(idx, table_size):
    """Causal bigram hash key (prev1, cur) -> [0, table_size).
    USE_AVALANCHE_BIGRAM=True: pack (prev1,cur) into 26 bits then fmix32-diffuse,
    mirroring the trigram avalanche, so distinct bigrams spread ~uniformly
    across the table (fewer structured collisions -> more effective capacity).
    Else the linear form (prev1*BIGRAM_PRIME + cur) - byte-identical to
    the original when the flag is off. BOTH are CAUSAL (use only {prev1, cur} <= t)
    and run in the no_grad lookup path. h >= 0, so power-of-2 mask and modulo agree."""
    B, T = idx.shape
    prev1 = torch.cat([idx.new_zeros(B, 1), idx[:, :-1]], dim=1).long()
    cur = idx.long()
    if USE_AVALANCHE_BIGRAM:
        h = _fmix32(((prev1 << 13) | cur) & 0xFFFFFFFF)                                     
    else:
        h = prev1 * BIGRAM_PRIME + cur
    if (table_size & (table_size - 1)) == 0:
        return h & (table_size - 1)
    return h % table_size

def compute_intradoc_pos_buckets(idx, num_buckets, seq_len, bos_id):
    """Causal intra-document position bucket: for each t, the distance since the
    most recent BOS at or before t, coarsened into num_buckets over seq_len.
    Row position 0 is always BOS (BOS-aligned dataloader), so last_bos is always
    defined. Uses cummax over j<=t -> depends ONLY on tokens {<=t}, never a future
    token. Computed OUTSIDE the compiled model (no_grad), so no compile/graph risk."""
    B, T = idx.shape
    arange = torch.arange(T, device=idx.device, dtype=torch.long)               
    is_bos = (idx == bos_id)                                                      
    bos_pos = torch.where(is_bos, arange.unsqueeze(0).expand(B, T),
                          torch.zeros_like(idx))                                                  
    last_bos = torch.cummax(bos_pos, dim=1).values                                                    
    intra = arange.unsqueeze(0) - last_bos                                             
    buckets = (intra * num_buckets) // seq_len
    return buckets.clamp_(0, num_buckets - 1)

def compute_last_bos(idx, bos_id):
    """[B,T] long: for each t, the position of the most recent BOS at or before t
    (cummax over j<=t, strictly causal). Row pos 0 is always BOS."""
    B, T = idx.shape
    arange = torch.arange(T, device=idx.device, dtype=torch.long)
    is_bos = (idx == bos_id)
    bos_pos = torch.where(is_bos, arange.unsqueeze(0).expand(B, T), torch.zeros_like(idx))
    return torch.cummax(bos_pos, dim=1).values

def compute_bigram_induction_next(idx, vocab_size, last_bos=None):
    """Causal SECOND-ORDER induction. For each (b,t), find the most recent previous
    position j<t whose BIGRAM CONTEXT (idx[j-1], idx[j]) equals the current context
    (idx[t-1], idx[t]), and return idx[j+1] - "the token that followed the last time
    this exact 2-token context appeared". A strictly higher-precision copy signal than
    single-token induction. First occurrences of a bigram context return `vocab_size`
    (a dedicated 'novel' row). Strictly causal: j<t so j+1<=t (never a future token),
    and the match key at t reads only idx[t-1],idx[t] (both <= t). Vectorized per row
    via ONE stable argsort on the context id (positions sorted by (prev1,cur) id, ties
    ascending by position -> consecutive same-context sorted neighbours are consecutive
    occurrences), then scatter_ back. Runs in the no_grad lookup path, OUTSIDE the
    compiled graph. Returns long [B,T] in [0, vocab_size].
    If last_bos is given (doc-mask), a match is accepted only when j >= last_bos[t]
    (the prior occurrence lies within the current document) - copies never cross
    packed-doc boundaries; still strictly causal (last_bos is a cummax over <=t)."""
    B, T = idx.shape
    idx = idx.long()
    prev1 = torch.cat([idx.new_zeros(B, 1), idx[:, :-1]], dim=1)                            
    ctx = prev1 * vocab_size + idx                                                             
    order = torch.argsort(ctx, dim=1, stable=True)                                                 
    sorted_ctx = torch.gather(ctx, 1, order)
    same = torch.zeros_like(idx, dtype=torch.bool)
    same[:, 1:] = sorted_ctx[:, 1:] == sorted_ctx[:, :-1]                                                 
    prev_pos_sorted = torch.zeros_like(idx)
    prev_pos_sorted[:, 1:] = order[:, :-1]                                                                    
    neg = torch.full_like(idx, -1)
    prev_occ = torch.full_like(idx, -1)
    prev_occ.scatter_(1, order, torch.where(same, prev_pos_sorted, neg))                              
    has_prev = prev_occ >= 0
    if last_bos is not None:
        has_prev = has_prev & (prev_occ >= last_bos)                                          
    next_idx = (prev_occ + 1).clamp_(0, T - 1)                                         
    nxt = torch.gather(idx, 1, next_idx)
    return torch.where(has_prev, nxt, torch.full_like(idx, vocab_size))

def _prev_occ_of_ctx(idx, ctx, last_bos=None):
    """[B,T] long: for each position t, the largest j<t whose context id ctx[j]==ctx[t]
    (the most-recent prior occurrence of the SAME context), else -1. One stable argsort:
    positions with equal ctx sort into ascending-position runs, so a run element's left
    neighbour in sorted order is its most-recent earlier occurrence. Strictly uses only
    ordering by position (j<t). If last_bos is given, occurrences before the current
    document (j<last_bos[t]) are rejected to -1 (so a later backoff can still fire)."""
    B, T = idx.shape
    order = torch.argsort(ctx, dim=1, stable=True)
    sorted_ctx = torch.gather(ctx, 1, order)
    same = torch.zeros_like(idx, dtype=torch.bool)
    same[:, 1:] = sorted_ctx[:, 1:] == sorted_ctx[:, :-1]
    prev_pos_sorted = torch.zeros_like(idx)
    prev_pos_sorted[:, 1:] = order[:, :-1]
    neg = torch.full_like(idx, -1)
    prev_occ = torch.full_like(idx, -1)
    prev_occ.scatter_(1, order, torch.where(same, prev_pos_sorted, neg))
    if last_bos is not None:
        prev_occ = torch.where(prev_occ >= last_bos, prev_occ, neg)                         
    return prev_occ

def compute_induction_next(idx, vocab_size, last_bos=None, order="backoff", return_prev_occ=False, return_bigram_next=False):
    """Causal induction with selectable match ORDER (see INDUCTION_ORDER note).
    order == 'trigram': match the 3-token context (idx[t-2],idx[t-1],idx[t]).
    order == 'backoff': prefer the 3-token match; where it is absent, fall back to the
                        2-token match (idx[t-1],idx[t]) - same recall as 'bigram', higher
                        precision. Returns idx[j+1] of the chosen match, else 'novel'
                        (index=vocab_size). Strictly causal: j<t => j+1<=t; the trigram
                        key reads only idx[t-2..t] (all <=t). Padding at t<2 mirrors the
                        prev1/prev2 zero-pad, so t=0,1 can only match same-pad
                        positions (a no-op in practice). Runs in the no_grad lookup path.
    return_prev_occ=True also returns the matched position j (or -1 where 'novel'),
    the byproduct the induction-match-distance feature buckets - costs NO extra argsort."""
    B, T = idx.shape
    idx = idx.long()
    if order == "unigram":

        prev_occ = _prev_occ_of_ctx(idx, idx, last_bos)
        has_prev = prev_occ >= 0
        next_idx = (prev_occ + 1).clamp_(0, T - 1)
        nxt = torch.gather(idx, 1, next_idx)
        result = torch.where(has_prev, nxt, torch.full_like(idx, vocab_size))
        if return_prev_occ:
            return result, torch.where(has_prev, prev_occ, torch.full_like(idx, -1))
        return result
    prev1 = torch.cat([idx.new_zeros(B, 1), idx[:, :-1]], dim=1)             
    prev2 = torch.cat([idx.new_zeros(B, 2), idx[:, :-2]], dim=1)             
    ctx3 = (prev2 * vocab_size + prev1) * vocab_size + idx                                           
    prev_occ = _prev_occ_of_ctx(idx, ctx3, last_bos)
    po2 = None
    if order == "backoff":
        ctx2 = prev1 * vocab_size + idx                                              
        po2 = _prev_occ_of_ctx(idx, ctx2, last_bos)
        prev_occ = torch.where(prev_occ >= 0, prev_occ, po2)                                      
    has_prev = prev_occ >= 0
    next_idx = (prev_occ + 1).clamp_(0, T - 1)                                         
    nxt = torch.gather(idx, 1, next_idx)
    result = torch.where(has_prev, nxt, torch.full_like(idx, vocab_size))
    outs = [result]
    if return_prev_occ:
        outs.append(torch.where(has_prev, prev_occ, torch.full_like(idx, -1)))
    if return_bigram_next:

        if po2 is None:
            ctx2 = prev1 * vocab_size + idx
            po2 = _prev_occ_of_ctx(idx, ctx2, last_bos)
        has_b = po2 >= 0
        b_next = torch.gather(idx, 1, (po2 + 1).clamp_(0, T - 1))
        outs.append(torch.where(has_b, b_next, torch.full_like(idx, vocab_size)))
    return tuple(outs) if len(outs) > 1 else outs[0]

def compute_induction_dist_buckets(prev_occ, num_buckets):
    """Log-bucket the induction MATCH DISTANCE d = t - prev_occ into [0, num_buckets-1];
    positions with no in-doc match (prev_occ < 0) map to the dedicated row `num_buckets`.
    prev_occ is the induction head's ALREADY-COMPUTED (docmasked, causal) match position,
    so this adds NO argsort. Strictly causal (prev_occ < t => d >= 1). Returns long [B,T]
    in [0, num_buckets]. Runs in the no_grad lookup path (inside the compiled graph)."""
    B, T = prev_occ.shape
    has_prev = prev_occ >= 0
    arange = torch.arange(T, device=prev_occ.device, dtype=torch.long)
    dist = (arange.unsqueeze(0) - prev_occ).clamp_(min=1)                                    
    logd = torch.log(dist.float()) / math.log(float(T))                         
    buckets = (logd * num_buckets).long().clamp_(0, num_buckets - 1)
    return torch.where(has_prev, buckets, torch.full_like(prev_occ, num_buckets))

def _lookups_impl(idx):
    """The exact per-step no_grad lookup block, in one function so it can
    be compiled as a single graph. Byte-identical to the eager per-function path
    (MAX_SEQ_LEN == config.sequence_len)."""
    bigram_keys = compute_bigram_keys(idx, BIGRAM_TABLE_SIZE)
    trigram_keys = compute_trigram_keys(idx, TRIGRAM_TABLE_SIZE)
    pos_bucket_ids = (compute_intradoc_pos_buckets(idx, NUM_POS_BUCKETS, MAX_SEQ_LEN, BOS_TOKEN_ID)
                      if USE_INTRADOC_POS else None)
    induction_dist_ids = None
    recall_next_tok = None
    if USE_BIGRAM_INDUCTION:
        lb = compute_last_bos(idx, BOS_TOKEN_ID) if INDUCTION_DOCMASK else None
        if INDUCTION_ORDER == "bigram":
            bind_next_tok = compute_bigram_induction_next(idx, vocab_size, lb)
        elif USE_INDUCTION_DIST:
            if USE_BIGRAM_RECALL:
                bind_next_tok, bind_prev_occ, recall_next_tok = compute_induction_next(
                    idx, vocab_size, lb, INDUCTION_ORDER, return_prev_occ=True, return_bigram_next=True)
            else:
                bind_next_tok, bind_prev_occ = compute_induction_next(
                    idx, vocab_size, lb, INDUCTION_ORDER, return_prev_occ=True)
            induction_dist_ids = compute_induction_dist_buckets(bind_prev_occ, NUM_INDUCTION_DIST_BUCKETS)
        else:
            if USE_BIGRAM_RECALL:
                bind_next_tok, recall_next_tok = compute_induction_next(
                    idx, vocab_size, lb, INDUCTION_ORDER, return_bigram_next=True)
            else:
                bind_next_tok = compute_induction_next(idx, vocab_size, lb, INDUCTION_ORDER)
    else:
        bind_next_tok = None
    if USE_UNIGRAM_INDUCTION:
        lb_uni = compute_last_bos(idx, BOS_TOKEN_ID) if UNIGRAM_INDUCTION_DOCMASK else None
        uni_next_tok = compute_induction_next(idx, vocab_size, lb_uni, "unigram")
    else:
        uni_next_tok = None
    return bigram_keys, trigram_keys, pos_bucket_ids, bind_next_tok, uni_next_tok, induction_dist_ids, recall_next_tok

_lookups_compiled = torch.compile(_lookups_impl, dynamic=False)
_lookups_enabled = COMPILE_LOOKUPS

def get_lookups(idx):
    """Fused per-step lookups: the compiled graph when enabled, with a permanent
    eager fallback on any error so a compile failure can never crash the run."""
    global _lookups_enabled
    if _lookups_enabled:
        try:
            return _lookups_compiled(idx)
        except Exception as e:                    
            print(f"[fusion] compiled lookups failed ({type(e).__name__}: {e}); "
                  f"falling back to eager for the rest of the run", flush=True)
            _lookups_enabled = False
    return _lookups_impl(idx)

def compute_unigram_logfreq(targets, vocab_size, floor=1e-6):
    """Centered log-unigram-frequency prior [vocab] (fp32) from target tokens.
    Counts token occurrences in `targets` (ignoring ignore_index=-1), converts to
    a smoothed log-probability, and centers it (softmax is shift-invariant, so
    centering leaves the induced distribution unchanged but keeps the bias values
    modest & symmetric). Used ONLY to initialize the lm_head output bias from the
    first TRAINING batch - a better starting point than zeros, not a val leak."""
    flat = targets.reshape(-1)
    flat = flat[flat >= 0]                                                          
    counts = torch.bincount(flat, minlength=vocab_size).float()
    total = counts.sum().clamp(min=1.0)
    logf = (counts / total + floor).log()
    return logf - logf.mean()

class GPTWithBigram(nn.Module):
    """Evaluation wrapper: combines compiled GPT + sparse bigram + sparse trigram
    for evaluate_bpb."""
    def __init__(self, compiled_gpt, bigram_embed, bigram_prime, bigram_table_size,
                 trigram_embed, trigram_table_size):
        super().__init__()
        self._gpt = compiled_gpt
        self._bigram_embed = bigram_embed
        self._bigram_prime = bigram_prime
        self._bigram_table_size = bigram_table_size
        self._trigram_embed = trigram_embed
        self._trigram_table_size = trigram_table_size

    def forward(self, idx, targets=None, reduction='mean'):
        keys = compute_bigram_keys(idx, self._bigram_table_size)
        bigram_out = self._bigram_embed(keys)             
        trigram_keys = compute_trigram_keys(idx, self._trigram_table_size)
        trigram_out = self._trigram_embed(trigram_keys)             
        pos_bucket_ids = (compute_intradoc_pos_buckets(idx, NUM_POS_BUCKETS, MAX_SEQ_LEN, BOS_TOKEN_ID)
                          if USE_INTRADOC_POS else None)

        induction_dist_ids = None
        recall_next_tok = None
        if USE_BIGRAM_INDUCTION:
            lb = compute_last_bos(idx, BOS_TOKEN_ID) if INDUCTION_DOCMASK else None
            if INDUCTION_ORDER == "bigram":
                bind_next_tok = compute_bigram_induction_next(idx, vocab_size, lb)
            elif USE_INDUCTION_DIST:
                if USE_BIGRAM_RECALL:
                    bind_next_tok, bind_prev_occ, recall_next_tok = compute_induction_next(
                        idx, vocab_size, lb, INDUCTION_ORDER, return_prev_occ=True, return_bigram_next=True)
                else:
                    bind_next_tok, bind_prev_occ = compute_induction_next(
                        idx, vocab_size, lb, INDUCTION_ORDER, return_prev_occ=True)
                induction_dist_ids = compute_induction_dist_buckets(bind_prev_occ, NUM_INDUCTION_DIST_BUCKETS)
            else:
                if USE_BIGRAM_RECALL:
                    bind_next_tok, recall_next_tok = compute_induction_next(
                        idx, vocab_size, lb, INDUCTION_ORDER, return_bigram_next=True)
                else:
                    bind_next_tok = compute_induction_next(idx, vocab_size, lb, INDUCTION_ORDER)
        else:
            bind_next_tok = None
        if USE_UNIGRAM_INDUCTION:
            lb_uni = compute_last_bos(idx, BOS_TOKEN_ID) if UNIGRAM_INDUCTION_DOCMASK else None
            uni_next_tok = compute_induction_next(idx, vocab_size, lb_uni, "unigram")
        else:
            uni_next_tok = None
        return self._gpt(idx, targets=targets, reduction=reduction,
                         bigram_offset=bigram_out, trigram_offset=trigram_out,
                         pos_bucket_ids=pos_bucket_ids, bind_next_tok=bind_next_tok,
                         uni_next_tok=uni_next_tok, induction_dist_ids=induction_dist_ids,
                         recall_next_tok=recall_next_tok)

class SparseBigramAdamW:
    """
    Sparse AdamW for a large hash embedding table.
    Only updates the ~131K accessed rows per step, not all TABLE_SIZE rows.

    factored_v: if True, the second moment (exp_avg_sq) is stored as a per-ROW
    scalar [n,1] fp32 (Adafactor-style row-wise RMS) instead of per-coordinate
    [n,d]. This roughly HALVES the optimizer-state VRAM (only the [n,d] momentum
    stays large), funding a bigger table WITHOUT the zero-quantization / denom-
    explosion hazard of int8 states. Adam math is still done in fp32; only the
    between-step STORAGE of the second moment is factored. Momentum stays
    per-coordinate (state_dtype). Used for the grown trigram; the bigram keeps
    the full per-coordinate optimizer (byte-identical to the original).
    """
    def __init__(self, weight, initial_lr, betas=(0.8, 0.98), eps=1e-10, weight_decay=0.003,
                 state_dtype=torch.float32, name="bigram", factored_v=False,
                 freq_lr=False, freq_lr_K=8.0, freq_lr_floor=0.0):
        self.weight = weight
        self.lr = initial_lr
        self.initial_lr = initial_lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state_dtype = state_dtype
        self.factored_v = factored_v

        self.freq_lr = freq_lr
        self.freq_lr_K = float(freq_lr_K)
        self.freq_lr_floor = float(freq_lr_floor)
        self.base_lr = initial_lr
        self.lrm = 1.0
        n, d = weight.shape
        device = weight.device
        if freq_lr:
            self.cnt = torch.zeros(n, dtype=torch.int32, device=device)
            print(f"[SparseBigramAdamW:{name}] freq_lr ON: per-row LR, K={self.freq_lr_K}, "
                  f"rare_row_floor={self.freq_lr_floor}, cnt tensor {n*4/1e9:.2f} GB int32")
        bytes_per = torch.finfo(state_dtype).bits // 8
        self.exp_avg = torch.zeros(n, d, dtype=state_dtype, device=device)
        if factored_v:
            self.exp_avg_sq = torch.zeros(n, 1, dtype=torch.float32, device=device)
            m_gb = n * d * bytes_per / 1e9
            v_gb = n * 4 / 1e9
            print(f"[SparseBigramAdamW:{name}] Allocating {n:,}×{d} exp_avg ({m_gb:.1f} GB {state_dtype}) "
                  f"+ {n:,}×1 factored exp_avg_sq ({v_gb:.2f} GB fp32) on {device}")
        else:
            self.exp_avg_sq = torch.zeros(n, d, dtype=state_dtype, device=device)
            gb = 2 * n * d * bytes_per / 1e9
            print(f"[SparseBigramAdamW:{name}] Allocating {n:,}×{d} optimizer states ({gb:.1f} GB {state_dtype}) on {device}")
        self.step_count = 0
        print(f"[SparseBigramAdamW:{name}] Done.")

    @torch.no_grad()
    def step(self):
        weight = self.weight
        grad = weight.grad
        if grad is None:
            return

        self.step_count += 1
        step = self.step_count
        lr = self.lr
        beta1, beta2 = self.betas
        eps = self.eps
        wd = self.weight_decay

        if grad.is_sparse:
            grad_c = grad.coalesce()
            indices = grad_c.indices()[0]
            grad_vals = grad_c.values().float()
        else:
            nz = grad.abs().any(dim=-1)
            indices = nz.nonzero(as_tuple=False).view(-1)
            if indices.numel() == 0:
                weight.grad = None
                return
            grad_vals = grad[indices].float()

        if indices.numel() == 0:
            weight.grad = None
            return

        m = self.exp_avg[indices]
        v = self.exp_avg_sq[indices]
        p = weight.data[indices].float()

        if self.freq_lr:
            c = self.cnt[indices].float()                                                      
            conf = c / (c + self.freq_lr_K)                                                
            row_floor = 1.0 - conf * (1.0 - self.freq_lr_floor)                                  
            row_mult = row_floor.clamp_(min=self.lrm)                                     
            lr_vec = (self.base_lr * row_mult).unsqueeze(1)                 
            self.cnt[indices] += 1
        else:
            lr_vec = None

        if lr_vec is None:
            p.mul_(1.0 - lr * wd)
        else:
            p.mul_(1.0 - lr_vec * wd)                                                         
        m_new = beta1 * m + (1.0 - beta1) * grad_vals
        if self.factored_v:
                                                                                  
            gsq = grad_vals.square().mean(dim=-1, keepdim=True)
            v_new = beta2 * v + (1.0 - beta2) * gsq
        else:
            v_new = beta2 * v + (1.0 - beta2) * grad_vals.square()
        bias1 = 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step
        denom = (v_new / bias2).sqrt_().add_(eps)                                       
        if lr_vec is None:
            p.addcdiv_(m_new, denom, value=-(lr / bias1))
        elif self.factored_v:

            coef = lr_vec / denom                                           
            p.addcmul_(m_new, coef, value=-(1.0 / bias1))
        else:
            p.addcdiv_(lr_vec * m_new, denom, value=-(1.0 / bias1))                             

        self.exp_avg[indices]    = m_new.to(self.exp_avg.dtype)
        self.exp_avg_sq[indices] = v_new.to(self.exp_avg_sq.dtype)
        weight.data[indices] = p.to(weight.dtype)
        weight.grad = None

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

        self.inject_ngram_v = layer_idx in NGRAM_VE_LAYERS
        self.ng_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if self.inject_ngram_v else None

        self.inject_bigram_v = layer_idx in BIGRAM_VE_LAYERS
        self.bg_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if self.inject_bigram_v else None

        self.attn_scale = nn.Parameter(torch.ones(self.n_head)) if USE_ATTN_SCALE else None

    def forward(self, x, ve, ng_v, bg_v, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        if ng_v is not None and self.ng_gate is not None:
                                                                                   
            ng_v = ng_v.view(B, T, self.n_kv_head, self.head_dim)
            ng_gate = 2 * torch.sigmoid(self.ng_gate(x[..., :self.ve_gate_channels]))
            v = v + ng_gate.unsqueeze(-1) * ng_v
        if bg_v is not None and self.bg_gate is not None:

            bg_v = bg_v.view(B, T, self.n_kv_head, self.head_dim)
            bg_gate = 2 * torch.sigmoid(self.bg_gate(x[..., :self.ve_gate_channels]))
            v = v + bg_gate.unsqueeze(-1) * bg_v
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        if self.attn_scale is not None:
            q = q * self.attn_scale.to(q.dtype).view(1, 1, self.n_head, 1)
        y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc   = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, ng_v, bg_v, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, ng_v, bg_v, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    """GPT with:
    - Dense trigram table inside (receives bigram_offset as kwarg)
    - Position bucket embedding (32 buckets, zero init, LR=0.15)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head_bias = nn.Parameter(torch.zeros(config.vocab_size)) if USE_LM_HEAD_BIAS else None
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
                                                                                    
        self.layer_readout_w = nn.Parameter(torch.zeros(config.n_layer)) if USE_LAYER_READOUT else None

        self.dense_A = nn.Parameter(torch.zeros(config.n_layer, config.n_layer)) if USE_DENSE_DWA else None
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

        self.pos_bucket_embed = nn.Embedding(NUM_POS_BUCKETS, config.n_embd)
        self.num_pos_buckets = NUM_POS_BUCKETS

        self.use_bigram_induction = USE_BIGRAM_INDUCTION
        if self.use_bigram_induction:
            self.bigram_induction_embed = nn.Embedding(config.vocab_size + 1, config.n_embd)
        self.use_unigram_induction = USE_UNIGRAM_INDUCTION
        if self.use_unigram_induction:
            self.unigram_induction_embed = nn.Embedding(config.vocab_size + 1, config.n_embd)

        self.use_induction_dist = USE_INDUCTION_DIST
        if self.use_induction_dist:
            self.induction_dist_embed = nn.Embedding(NUM_INDUCTION_DIST_BUCKETS + 1, config.n_embd)
            self.num_induction_dist_buckets = NUM_INDUCTION_DIST_BUCKETS

        self.use_bigram_recall = USE_BIGRAM_RECALL
        if self.use_bigram_recall:
            self.bigram_recall_embed = nn.Embedding(config.vocab_size + 1, config.n_embd)

        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 0.68 * 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.05)

        if self.layer_readout_w is not None:
            self.layer_readout_w.zero_()
            self.layer_readout_w[-1] = 1.0

        if self.dense_A is not None:
            self.dense_A.zero_()
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.ng_gate is not None:
                torch.nn.init.zeros_(block.attn.ng_gate.weight)
                                                    
        torch.nn.init.zeros_(self.pos_bucket_embed.weight)
                                                                                            
        if self.use_bigram_induction:
            torch.nn.init.zeros_(self.bigram_induction_embed.weight)
        if self.use_bigram_recall:
            torch.nn.init.zeros_(self.bigram_recall_embed.weight)
        if self.use_unigram_induction:
            torch.nn.init.zeros_(self.unigram_induction_embed.weight)
        if self.use_induction_dist:
            torch.nn.init.zeros_(self.induction_dist_embed.weight)

        if self.lm_head_bias is not None:
            torch.nn.init.zeros_(self.lm_head_bias)

        for block in self.transformer.h:
            if block.attn.attn_scale is not None:
                block.attn.attn_scale.fill_(1.0)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
        self.pos_bucket_embed.to(dtype=torch.bfloat16)
        if self.use_bigram_induction:
            self.bigram_induction_embed.to(dtype=torch.bfloat16)
        if self.use_bigram_recall:
            self.bigram_recall_embed.to(dtype=torch.bfloat16)
        if self.use_unigram_induction:
            self.unigram_induction_embed.to(dtype=torch.bfloat16)
        if self.use_induction_dist:
            self.induction_dist_embed.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=200000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        long_window = config.sequence_len // 2
        short_window = config.sequence_len // 8
        if LONG_LAYERS:

            window_sizes = [
                (long_window, 0) if i in LONG_LAYERS else (short_window, 0)
                for i in range(config.n_layer)
            ]
            print(f"[window_sizes] LONG_LAYERS override -> {[w[0] for w in window_sizes]}")
            return window_sizes
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        print(f"[window_sizes] pattern '{pattern}' -> {[w[0] for w in window_sizes]}")
        return window_sizes

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        bigram_induction_numel = (self.bigram_induction_embed.weight.numel()
                                  if self.use_bigram_induction else 0)
        unigram_induction_numel = (self.unigram_induction_embed.weight.numel()
                                   if self.use_unigram_induction else 0)
        induction_dist_numel = (self.induction_dist_embed.weight.numel()
                                if self.use_induction_dist else 0)
        bigram_recall_numel = (self.bigram_recall_embed.weight.numel()
                               if self.use_bigram_recall else 0)
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.pos_bucket_embed.weight.numel() + bigram_induction_numel +
                          unigram_induction_numel + induction_dist_numel + bigram_recall_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, value_embeds_lr=None,
                        matrix_lr=0.02, weight_decay=0.0,
                        adam_betas=(0.8, 0.95), scalar_lr=0.5, trigram_lr=0.15,
                        pos_bucket_lr=None, mlp_fc_lr=None, bigram_induction_lr=None,
                        unigram_induction_lr=None, induction_dist_lr=None, bigram_recall_lr=None):
        if value_embeds_lr is None:
            value_embeds_lr = embedding_lr
        if pos_bucket_lr is None:
            pos_bucket_lr = embedding_lr
        if mlp_fc_lr is None:
            mlp_fc_lr = matrix_lr
        if bigram_induction_lr is None:
            bigram_induction_lr = embedding_lr
        if unigram_induction_lr is None:
            unigram_induction_lr = embedding_lr
        if induction_dist_lr is None:
            induction_dist_lr = embedding_lr
        if bigram_recall_lr is None:
            bigram_recall_lr = embedding_lr
        model_dim = self.config.n_embd

        attn_scale_params = [b.attn.attn_scale for b in self.transformer.h if b.attn.attn_scale is not None]
        attn_scale_ids = {id(p) for p in attn_scale_params}
        matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in attn_scale_ids]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        pos_bucket_params = list(self.pos_bucket_embed.parameters())
        dmodel_lr_scale = (model_dim / 768) ** -0.5
                                                                          
        mlp_fc_shape = torch.Size([4 * model_dim, model_dim])
        print(f"Scaling LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.4f}")
        print(f"pos_bucket_lr={pos_bucket_lr:.4f} (trigram optimized externally by sparse AdamW)")
        print(f"matrix_lr={matrix_lr:.4f}, mlp_fc_lr={mlp_fc_lr:.4f} (effective c_fc: {mlp_fc_lr*2.0:.4f} with 2x aspect ratio)")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=value_embeds_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.003),
                                                                    
            dict(kind='adamw', params=pos_bucket_params, lr=pos_bucket_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        if self.lm_head_bias is not None:
            print(f"lm_head_bias group: lr={LM_HEAD_BIAS_LR * dmodel_lr_scale:.4f} (wd=0, {self.lm_head_bias.numel()} params)")
            param_groups.append(dict(kind='adamw', params=[self.lm_head_bias],
                lr=LM_HEAD_BIAS_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        if attn_scale_params:
            print(f"attn_scale group: lr={ATTN_SCALE_LR * dmodel_lr_scale:.4f} (wd=0, {len(attn_scale_params)} tensors x {attn_scale_params[0].numel()} heads)")
            param_groups.append(dict(kind='adamw', params=attn_scale_params,
                lr=ATTN_SCALE_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        if self.layer_readout_w is not None:
            print(f"layer_readout group: lr={LAYER_READOUT_LR * dmodel_lr_scale:.4f} (wd=0, {self.layer_readout_w.numel()} weights, init last-only)")
            param_groups.append(dict(kind='adamw', params=[self.layer_readout_w],
                lr=LAYER_READOUT_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        if self.dense_A is not None:
            print(f"dense_dwa group: lr={DENSE_DWA_LR * dmodel_lr_scale:.4f} (wd=0, {self.dense_A.numel()} matrix entries, K={DENSE_DWA_K}, init 0)")
            param_groups.append(dict(kind='adamw', params=[self.dense_A],
                lr=DENSE_DWA_LR * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        if self.use_bigram_induction:
            param_groups.append(dict(
                kind='adamw', params=list(self.bigram_induction_embed.parameters()),
                lr=bigram_induction_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10,
                weight_decay=0.003))
            print(f"bigram_induction_lr={bigram_induction_lr:.4f}, "
                  f"bigram_induction_vocab={self.config.vocab_size + 1}")

        if self.use_bigram_recall:
            param_groups.append(dict(
                kind='adamw', params=list(self.bigram_recall_embed.parameters()),
                lr=bigram_recall_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10,
                weight_decay=0.003))
            print(f"bigram_recall_lr={bigram_recall_lr:.4f}, "
                  f"bigram_recall_vocab={self.config.vocab_size + 1}")
        if self.use_unigram_induction:
            param_groups.append(dict(
                kind='adamw', params=list(self.unigram_induction_embed.parameters()),
                lr=unigram_induction_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10,
                weight_decay=0.003))
            print(f"unigram_induction_lr={unigram_induction_lr:.4f}, "
                  f"unigram_induction_vocab={self.config.vocab_size + 1}")

        if self.use_induction_dist:
            param_groups.append(dict(
                kind='adamw', params=list(self.induction_dist_embed.parameters()),
                lr=induction_dist_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10,
                weight_decay=0.001))
            print(f"induction_dist_lr={induction_dist_lr:.4f}, "
                  f"induction_dist_buckets={NUM_INDUCTION_DIST_BUCKETS + 1}")
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]

            group_lr = mlp_fc_lr if shape == mlp_fc_shape else matrix_lr
            param_groups.append(dict(
                kind='muon', params=group_params, lr=group_lr,
                momentum=0.95, ns_steps=5, beta2=0.85, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean', bigram_offset=None, trigram_offset=None,
                pos_bucket_ids=None, bind_next_tok=None, uni_next_tok=None, induction_dist_ids=None,
                recall_next_tok=None):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        if pos_bucket_ids is not None:
            pos_emb = self.pos_bucket_embed(pos_bucket_ids)                         
        else:
            pos = torch.arange(T, device=idx.device, dtype=torch.long)
            pos_buckets = (pos * self.num_pos_buckets) // self.config.sequence_len
            pos_buckets = pos_buckets.clamp(0, self.num_pos_buckets - 1)
            pos_emb = self.pos_bucket_embed(pos_buckets).unsqueeze(0)                  

        x = self.transformer.wte(idx)
        if bigram_offset is not None:
            x = x + bigram_offset
        if trigram_offset is not None:
            x = x + trigram_offset
        x = x + pos_emb

        if self.use_bigram_induction and bind_next_tok is not None:
            x = x + self.bigram_induction_embed(bind_next_tok)                     
                                                                                          
        if self.use_bigram_recall and recall_next_tok is not None:
            x = x + self.bigram_recall_embed(recall_next_tok)                      
        if self.use_unigram_induction and uni_next_tok is not None:
            x = x + self.unigram_induction_embed(uni_next_tok)                     
                                                                                       
        if self.use_induction_dist and induction_dist_ids is not None:
            x = x + self.induction_dist_embed(induction_dist_ids)                  
        x = norm(x)
        x0 = x
        readout = self.layer_readout_w is not None
        dense = self.dense_A is not None
        layer_outs = []
                                                                                       
        dense_states = [x0] if dense else None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

            if dense and i >= 2:
                j_lo = max(1, i - DENSE_DWA_K)
                for j in range(j_lo, i):
                    x = x + self.dense_A[i, j] * dense_states[j]
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
                                                                                      
            ng_v = trigram_offset if (i in NGRAM_VE_LAYERS and trigram_offset is not None) else None
                                                                                         
            bg_v = bigram_offset if (i in BIGRAM_VE_LAYERS and bigram_offset is not None) else None
            x = block(x, ve, ng_v, bg_v, cos_sin, self.window_sizes[i])
            if readout:
                layer_outs.append(x)
            if dense:
                dense_states.append(x)
        if readout:

            if USE_READOUT_NORM:
                x = self.layer_readout_w[0] * norm(layer_outs[0])
                for i in range(1, self.config.n_layer):
                    x = x + self.layer_readout_w[i] * norm(layer_outs[i])
            else:
                x = self.layer_readout_w[0] * layer_outs[0]
                for i in range(1, self.config.n_layer):
                    x = x + self.layer_readout_w[i] * layer_outs[i]
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if self.lm_head_bias is not None:
            logits = logits + self.lm_head_bias

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t     = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

ASPECT_RATIO = 85
HEAD_DIM = 128
WINDOW_PATTERN = "SSL"

BIGRAM_TABLE_SIZE = BIGRAM_TABLE_SIZE_ENV                                   
BIGRAM_LR    = float(os.environ.get("BIGRAM_LR", 0.65))                                           

TRIGRAM_LR   = float(os.environ.get("TRIGRAM_LR", 0.45))
TRIGRAM_LR_FLOOR = float(os.environ.get("TRIGRAM_LR_FLOOR", "0.0"))
BIGRAM_LR_FLOOR = float(os.environ.get("BIGRAM_LR_FLOOR", "0.0"))

USE_TRIGRAM_FREQ_LR = os.environ.get("USE_TRIGRAM_FREQ_LR", "0") == "1"
TRIGRAM_FREQ_LR_K   = float(os.environ.get("TRIGRAM_FREQ_LR_K", "8.0"))

USE_BIGRAM_FREQ_LR = os.environ.get("USE_BIGRAM_FREQ_LR", "0") == "1"
BIGRAM_FREQ_LR_K   = float(os.environ.get("BIGRAM_FREQ_LR_K", "16.0"))
BIGRAM_FREQ_LR_FLOOR = float(os.environ.get("BIGRAM_FREQ_LR_FLOOR", "0.0"))

POS_BUCKET_LR = float(os.environ.get("POS_BUCKET_LR", 0.15))

TOTAL_BATCH_SIZE = 2**17
EMBEDDING_LR    = 0.7                                                                                
VALUE_EMBEDS_LR = 0.6                                                      
UNEMBEDDING_LR  = 0.008                                             
MATRIX_LR       = 0.020                                               
MLP_FC_LR       = 0.018                                                        

SCALAR_LR       = 0.5
WEIGHT_DECAY    = 0.2
ADAM_BETAS      = (0.8, 0.98)
WARMUP_RATIO    = 0.0
WARMDOWN_RATIO  = 0.90
FINAL_LR_FRAC   = 0.02
MOMENTUM_WARMDOWN_DELAY = 0.45
MOMENTUM_END = 0.90
DEPTH = 9
DEVICE_BATCH_SIZE = 64

t_start = time.time()
_SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(_SEED)
torch.cuda.manual_seed(_SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
B200_BF16_PEAK_FLOPS = 2.25e15

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
BOS_TOKEN_ID = tokenizer.get_bos_token_id()
print(f"Vocab size: {vocab_size:,}")
print(f"BOS_TOKEN_ID: {BOS_TOKEN_ID}  USE_INTRADOC_POS: {USE_INTRADOC_POS}")
print(f"USE_BIGRAM_INDUCTION: {USE_BIGRAM_INDUCTION}  BIGRAM_INDUCTION_LR: {BIGRAM_INDUCTION_LR}  "
      f"INDUCTION_ORDER: {INDUCTION_ORDER}  INDUCTION_DOCMASK: {INDUCTION_DOCMASK}")
print(f"USE_INDUCTION_DIST: {USE_INDUCTION_DIST}  INDUCTION_DIST_LR: {INDUCTION_DIST_LR}  "
      f"NUM_INDUCTION_DIST_BUCKETS: {NUM_INDUCTION_DIST_BUCKETS}  COMPILE_LOOKUPS: {COMPILE_LOOKUPS}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")
print(f"BIGRAM_TABLE_SIZE: {BIGRAM_TABLE_SIZE:,} ({BIGRAM_TABLE_SIZE/1e6:.0f}M entries)")
print(f"TRIGRAM_TABLE_SIZE: {TRIGRAM_TABLE_SIZE:,} ({TRIGRAM_TABLE_SIZE/1e6:.1f}M entries, sparse)")
print(f"NUM_POS_BUCKETS: {NUM_POS_BUCKETS}")
print(f"NGRAM_VE_LAYERS (trigram->value path): {NGRAM_VE_LAYERS}")

if BIGRAM_FACTORED:
    bigram_opt_gb = (BIGRAM_TABLE_SIZE * 768 * 2 + BIGRAM_TABLE_SIZE * 4) / 1e9
else:
    bigram_opt_gb = 2 * BIGRAM_TABLE_SIZE * 768 * 2 / 1e9                               
                                                                                                
trigram_opt_gb = (TRIGRAM_TABLE_SIZE * 768 * 2 + TRIGRAM_TABLE_SIZE * 4) / 1e9
bigram_weight_gb_est = BIGRAM_TABLE_SIZE * 768 * 2 / 1e9
trigram_weight_gb_est = TRIGRAM_TABLE_SIZE * 768 * 2 / 1e9
print(f"Bigram optimizer memory: {bigram_opt_gb:.1f} GB (factored={BIGRAM_FACTORED})")
print(f"Trigram optimizer memory: {trigram_opt_gb:.2f} GB (bf16 momentum + fp32 factored v)")
print(f"[VRAM estimate] bigram(weight+opt)={bigram_weight_gb_est+bigram_opt_gb:.1f}GB  "
      f"trigram(weight+opt)={trigram_weight_gb_est+trigram_opt_gb:.1f}GB  "
      f"ngram-total={bigram_weight_gb_est+bigram_opt_gb+trigram_weight_gb_est+trigram_opt_gb:.1f}GB "
      f"(+~34GB rest => predicted peak ~{bigram_weight_gb_est+bigram_opt_gb+trigram_weight_gb_est+trigram_opt_gb+34:.0f}GB)")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

num_params_gpt = sum(p.numel() for p in model.parameters())
print(f"GPT params (transformer + pos_bucket, excl. sparse n-gram tables): {num_params_gpt/1e6:.1f}M")
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs/token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
print(f"Gradient accumulation steps: {grad_accum_steps}")

print(f"Building bigram embedding table ({BIGRAM_TABLE_SIZE:,} × {config.n_embd})...")

bigram_embed = nn.Embedding(BIGRAM_TABLE_SIZE, config.n_embd, sparse=True,
                            dtype=torch.bfloat16, device=device)
bigram_embed.weight.data.zero_()
bigram_weight_gb = bigram_embed.weight.data.element_size() * bigram_embed.weight.data.numel() / 1e9
print(f"Bigram weight: {bigram_weight_gb:.1f} GB bf16")

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR, embedding_lr=EMBEDDING_LR,
    value_embeds_lr=VALUE_EMBEDS_LR, scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS, matrix_lr=MATRIX_LR, weight_decay=WEIGHT_DECAY,
    trigram_lr=TRIGRAM_LR, pos_bucket_lr=POS_BUCKET_LR,
    mlp_fc_lr=MLP_FC_LR, bigram_induction_lr=BIGRAM_INDUCTION_LR,
    unigram_induction_lr=UNIGRAM_INDUCTION_LR, induction_dist_lr=INDUCTION_DIST_LR,
    bigram_recall_lr=BIGRAM_RECALL_LR,
)

initial_bigram_lr = BIGRAM_LR
bigram_sparse_opt = SparseBigramAdamW(
    weight=bigram_embed.weight,
    initial_lr=initial_bigram_lr,
    betas=ADAM_BETAS,
    eps=1e-10,
    weight_decay=0.003,
    state_dtype=torch.bfloat16,
    factored_v=BIGRAM_FACTORED,                                                                       
    name="bigram",
    freq_lr=USE_BIGRAM_FREQ_LR,                                                                       
    freq_lr_K=BIGRAM_FREQ_LR_K,
    freq_lr_floor=BIGRAM_FREQ_LR_FLOOR,                                                       
)

print(f"Building trigram embedding table ({TRIGRAM_TABLE_SIZE:,} × {config.n_embd})...")

trigram_embed = nn.Embedding(TRIGRAM_TABLE_SIZE, config.n_embd, sparse=True,
                             dtype=torch.bfloat16, device=device)
trigram_embed.weight.data.zero_()
trigram_weight_gb = trigram_embed.weight.data.element_size() * trigram_embed.weight.data.numel() / 1e9
print(f"Trigram weight: {trigram_weight_gb:.1f} GB bf16")
initial_trigram_lr = TRIGRAM_LR
trigram_sparse_opt = SparseBigramAdamW(
    weight=trigram_embed.weight,
    initial_lr=initial_trigram_lr,
    betas=ADAM_BETAS,
    eps=1e-10,
    weight_decay=0.003,
    state_dtype=torch.bfloat16,                                            
    factored_v=True,                                                                              
    name="trigram",
    freq_lr=USE_TRIGRAM_FREQ_LR,                                                      
    freq_lr_K=TRIGRAM_FREQ_LR_K,
    freq_lr_floor=TRIGRAM_LR_FLOOR,                                                        
)

lm_head_bias_param = model.lm_head_bias if USE_LM_HEAD_BIAS else None

model = torch.compile(model, dynamic=False, mode="max-autotune")

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)

if lm_head_bias_param is not None and LM_HEAD_BIAS_INIT:
    with torch.no_grad():
        logfreq0 = compute_unigram_logfreq(y, vocab_size).to(lm_head_bias_param.dtype)
        lm_head_bias_param.data.copy_(logfreq0)
    print(f"lm_head_bias smart-init from first batch: min={logfreq0.min().item():.3f} "
          f"max={logfreq0.max().item():.3f} mean={logfreq0.mean().item():.3f}")

print(f"Time budget: {TIME_BUDGET}s")
print(f"BIGRAM_LR: {BIGRAM_LR}, BIGRAM_TABLE_SIZE: {BIGRAM_TABLE_SIZE:,}")
print(f"BIGRAM_FREQ_LR: {USE_BIGRAM_FREQ_LR}, K: {BIGRAM_FREQ_LR_K}, rare_row_floor: {BIGRAM_FREQ_LR_FLOOR}")
print(f"TRIGRAM_LR: {TRIGRAM_LR}, TRIGRAM_TABLE_SIZE: {TRIGRAM_TABLE_SIZE:,}")
print(f"POS_BUCKET_LR: {POS_BUCKET_LR}, NUM_POS_BUCKETS: {NUM_POS_BUCKETS}")
print(f"MLP_FC_LR: {MLP_FC_LR} (effective {MLP_FC_LR*2:.4f} with 2x aspect ratio), MATRIX_LR: {MATRIX_LR}")
print(f"USE_LM_HEAD_BIAS: {USE_LM_HEAD_BIAS} (off), LM_HEAD_BIAS_LR: {LM_HEAD_BIAS_LR}")
print(f"USE_ATTN_SCALE: {USE_ATTN_SCALE}, ATTN_SCALE_LR: {ATTN_SCALE_LR}")
print(f"USE_LAYER_READOUT: {USE_LAYER_READOUT}, LAYER_READOUT_LR: {LAYER_READOUT_LR}, USE_READOUT_NORM: {USE_READOUT_NORM}")
print(f"USE_DENSE_DWA: {USE_DENSE_DWA}, DENSE_DWA_LR: {DENSE_DWA_LR}, DENSE_DWA_K: {DENSE_DWA_K}")

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step, progress):
    ramp_frac = min(step / 200, 1)
    base_momentum = (1 - ramp_frac) * 0.85 + ramp_frac * 0.95
    WARMDOWN_START = 1.0 - WARMDOWN_RATIO
    if progress >= WARMDOWN_START:
        warmdown_frac = (progress - WARMDOWN_START) / WARMDOWN_RATIO
        delayed_frac = max(0.0, (warmdown_frac - MOMENTUM_WARMDOWN_DELAY) / (1.0 - MOMENTUM_WARMDOWN_DELAY))
        base_momentum = 0.95 - delayed_frac * (0.95 - MOMENTUM_END)
    return base_momentum

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        (bigram_keys, trigram_keys, pos_bucket_ids,
         bind_next_tok, uni_next_tok, induction_dist_ids, recall_next_tok) = get_lookups(x)

    bigram_out = bigram_embed(bigram_keys)                
    trigram_out = trigram_embed(trigram_keys)             

    for micro_step in range(grad_accum_steps):
        with autocast_ctx:

            loss_tok = model(x, y, reduction='none',
                             bigram_offset=bigram_out, trigram_offset=trigram_out,
                             pos_bucket_ids=pos_bucket_ids, bind_next_tok=bind_next_tok,
                             uni_next_tok=uni_next_tok, induction_dist_ids=induction_dist_ids,
                             recall_next_tok=recall_next_tok)
            n_valid = (y.view(-1) != -1).sum().clamp(min=1)
            loss = loss_tok.sum() / n_valid
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step, progress)
    muon_weight_decay = get_weight_decay(progress)

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()

    if bigram_sparse_opt.freq_lr:

        bigram_sparse_opt.base_lr = initial_bigram_lr
        bigram_sparse_opt.lrm = lrm
    else:
        bigram_sparse_opt.lr = initial_bigram_lr * max(lrm, BIGRAM_LR_FLOOR)
    bigram_sparse_opt.step()

    if trigram_sparse_opt.freq_lr:

        trigram_sparse_opt.base_lr = initial_trigram_lr
        trigram_sparse_opt.lrm = lrm
    else:
        trigram_sparse_opt.lr = initial_trigram_lr * max(lrm, TRIGRAM_LR_FLOOR)
    trigram_sparse_opt.step()

    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()
    if train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / B200_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | mom: {muon_momentum:.4f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * TOTAL_BATCH_SIZE

model.eval()
bigram_embed.eval()
trigram_embed.eval()
eval_model = GPTWithBigram(model, bigram_embed, BIGRAM_PRIME, BIGRAM_TABLE_SIZE,
                           trigram_embed, TRIGRAM_TABLE_SIZE)
with autocast_ctx:
    val_bpb = evaluate_bpb(eval_model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()
steady_state_mfu = (100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) /
                    total_training_time / B200_BF16_PEAK_FLOPS) if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params_gpt / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"bigram_table_size: {BIGRAM_TABLE_SIZE:,}")
print(f"bigram_lr:        {BIGRAM_LR}")
print(f"trigram_table_size: {TRIGRAM_TABLE_SIZE:,}")
print(f"trigram_lr:       {TRIGRAM_LR}")
print(f"pos_bucket_lr:    {POS_BUCKET_LR}")
print(f"num_pos_buckets:  {NUM_POS_BUCKETS}")
print(f"momentum_delay:   {MOMENTUM_WARMDOWN_DELAY}")
print(f"momentum_end:     {MOMENTUM_END}")
print(f"mlp_fc_lr:        {MLP_FC_LR}")
print(f"matrix_lr:        {MATRIX_LR}")
