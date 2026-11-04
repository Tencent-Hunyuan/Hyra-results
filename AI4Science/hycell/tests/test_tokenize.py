import numpy as np

from hycell.tokenize import cell_rng, tokenize_cell

CLS = 36495


def cell(n, seed=0, n_vocab=36495):
    rng = np.random.default_rng(seed)
    g = np.sort(rng.choice(n_vocab, n, replace=False)).astype(np.int32)
    return g, rng.integers(1, 50, n).astype(np.float64)


def test_small_cell_keeps_all_genes_and_adds_zeros():
    g, x = cell(100)
    pool = np.arange(0, 36495, 3, dtype=np.int32)
    tok, val = tokenize_cell(g, x, pool, cell_rng(0, 7), CLS)
    assert tok[0] == CLS and val[0] == 0
    assert np.array_equal(tok[1:101], g)
    np.testing.assert_allclose(val[1:101], np.log1p(x / x.sum() * 1e4).astype(np.float32))
    zeros = tok[101:]
    assert len(zeros) == 25 and (val[101:] == 0).all()
    assert np.isin(zeros, pool).all() and not np.isin(zeros, g).any() and len(set(zeros)) == len(zeros)


def test_large_cell_is_capped_at_1024_tokens_by_value():
    g, x = cell(3000)
    tok, val = tokenize_cell(g, x, np.arange(36495, dtype=np.int32), cell_rng(0, 1), CLS)
    assert len(tok) <= 1024 and (val[1:819] > 0).all() and (val[819:] == 0).all()
    kept = tok[1:819]
    lv = np.log1p(x / x.sum() * 1e4).astype(np.float32)
    assert lv[np.isin(g, kept)].min() >= lv[~np.isin(g, kept)].max()


def test_deterministic_per_seed_and_row():
    g, x = cell(2000)
    pool = np.arange(36495, dtype=np.int32)
    a = tokenize_cell(g, x, pool, cell_rng(0, 5), CLS)
    b = tokenize_cell(g, x, pool, cell_rng(0, 5), CLS)
    c = tokenize_cell(g, x, pool, cell_rng(0, 6), CLS)
    assert np.array_equal(a[0], b[0]) and not np.array_equal(a[0], c[0])


def test_empty_cell_is_cls_only():
    tok, val = tokenize_cell(np.zeros(0, np.int32), np.zeros(0), np.arange(10, dtype=np.int32), cell_rng(0, 0), CLS)
    assert tok.tolist() == [CLS] and val.tolist() == [0.0]
