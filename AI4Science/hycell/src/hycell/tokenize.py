"""Turn one cell's counts into HyCell tokens.

Values are log1p of counts per 10,000 in-vocabulary counts. A cell with more than 818 expressed genes keeps
the 818 with the highest value (exact ties are broken at random). Measured zeros, a quarter as many as the
expressed genes, are drawn from the core genes that the dataset measures and the cell does not express. A
[CLS] token leads the set, so a cell has at most 1,024 tokens.

The random draws (tie breaks and zero genes) come from a generator seeded with (seed, row), where row is the
cell's position in the input, so an embedding does not depend on the other cells or on batching.

For fine-tuning, ``augment_counts`` applies the two pretraining augmentations, and ``sample=True`` draws the
kept genes of a large cell with probability increasing in their value instead of taking the highest.
"""
from __future__ import annotations

import numpy as np

MAX_TOKENS = 1024
ZERO_RATIO = 0.25


def cell_rng(seed: int, row: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(row)])


def augment_counts(genes, counts, rng, p_panel=0.3, panel_min=0.15, p_thin=0.5, thin_min=0.1):
    """Pretraining augmentations: with probability p_panel keep a random U[panel_min, 1] fraction of the genes, and
    with probability p_thin thin the counts binomially to a log-uniform [thin_min, 1] fraction of their depth.
    A cell left without counts keeps one count of its most expressed gene."""
    g, c = genes, counts
    if rng.random() < p_panel:
        keep = rng.random(len(g)) < rng.uniform(panel_min, 1.0)
        g, c = g[keep], c[keep]
    if rng.random() < p_thin:
        f = np.exp(rng.uniform(np.log(thin_min), 0.0))
        c = rng.binomial(np.rint(c).astype(np.int64), f).astype(np.float64)
        keep = c > 0
        g, c = g[keep], c[keep]
    if len(g) == 0:
        j = int(np.argmax(counts))
        g, c = genes[j:j + 1], np.ones(1)
    return g, c


def tokenize_cell(genes, counts, zero_pool, rng, cls_id, max_tokens=MAX_TOKENS, zero_ratio=ZERO_RATIO,
                  expressed=None, sample=False):
    """genes: ascending vocabulary ids of the expressed genes (int32); counts: their counts (float64, > 0);
    zero_pool: ascending vocabulary ids that may enter as measured zeros. expressed: ascending ids that may not
    enter as zeros (default genes; after augmentation, the cell's original genes). sample: draw the kept genes of
    a cell over the cap with probability increasing in value (training) instead of the highest (inference).
    Returns (tokens int32, values float32)."""
    if len(genes) == 0:
        return np.array([cls_id], np.int32), np.zeros(1, np.float32)
    expressed = genes if expressed is None else expressed
    x = np.log1p(counts / counts.sum() * 1e4).astype(np.float32)
    g = genes.astype(np.int32)
    n_expressed_max = int((max_tokens - 1) / (1 + zero_ratio))
    if len(g) > n_expressed_max:
        if sample:
            keys = np.log(x) - np.log(-np.log(rng.random(len(g))))
            top = np.argpartition(-keys, n_expressed_max - 1)[:n_expressed_max]
        else:
            top = np.lexsort((rng.random(len(g)), -x))[:n_expressed_max]
        g, x = g[top], x[top]
    n_zero = min(int(round(zero_ratio * len(g))), max_tokens - 1 - len(g))
    if n_zero > 0 and len(zero_pool) > 0:
        cand = zero_pool[rng.integers(0, len(zero_pool), size=n_zero + n_zero // 2 + 8)]
        pos = np.searchsorted(expressed, cand)
        hit = (pos < len(expressed)) & (expressed[np.minimum(pos, len(expressed) - 1)] == cand)
        cand = np.unique(cand[~hit])
        cand = cand[rng.permutation(len(cand))[:n_zero]]
        g = np.concatenate([g, cand.astype(np.int32)])
        x = np.concatenate([x, np.zeros(len(cand), np.float32)])
    tokens = np.concatenate([[cls_id], g]).astype(np.int32)
    values = np.concatenate([[0.0], x]).astype(np.float32)
    return tokens, values
