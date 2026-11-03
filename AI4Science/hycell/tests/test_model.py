"""The released model against the reference embeddings of the lung fixture."""
import numpy as np
import pytest
import torch

import hycell
from hycell.model import flash_available


def cosine(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def test_parameter_count(weights):
    m = hycell.load(weights, device='cpu')
    assert m.model.n_parameters() == 16_108_766 and m.embedding_dim == 512


def test_cpu_matches_reference(weights, lung, expected):
    m = hycell.load(weights, device='cpu')
    z = m.embed(lung)
    assert z.shape == (512, 512) and z.dtype == np.float32
    assert cosine(z, expected['embedding']).min() > 0.999


@pytest.mark.skipif(not flash_available('cuda'), reason='needs CUDA and flash-attn')
def test_flash_reproduces_reference_exactly(weights, lung, expected):
    z = hycell.load(weights, device='cuda', attention='flash').embed(lung)
    assert np.array_equal(z, expected['embedding'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_sdpa_on_gpu_matches_reference(weights, lung, expected):
    z = hycell.load(weights, device='cuda', attention='sdpa').embed(lung)
    assert cosine(z, expected['embedding']).min() > 0.999


def test_embedding_is_independent_of_batching(weights, lung):
    m = hycell.load(weights, device='cpu')
    a = m.embed(lung[:64].copy(), batch_tokens=4096)
    b = m.embed(lung[:64].copy(), batch_tokens=65536)
    assert cosine(a, b).min() > 0.9999


def test_decode_matches_reference(weights, lung, expected):
    m = hycell.load(weights, device='cpu')
    measured = m.measured_genes(lung)
    full = m.decode(expected['embedding'][:16], genes=measured)
    ens = m.genes.ensembl_id.values[m.gene_ids(expected['decode_genes'])]
    cols = [int(np.flatnonzero(measured == g)[0]) for g in ens]
    np.testing.assert_allclose(full[:, cols], expected['decode_cp10k'], rtol=1e-4)
    np.testing.assert_allclose(full.sum(1), 1e4, rtol=1e-4)


def test_rows_reproduce_full_file_embeddings(weights, lung):
    m = hycell.load(weights, device='cpu')
    sel = np.arange(100, 164)
    full = m.embed(lung[:200].copy())
    sub = m.embed(lung[sel].copy(), rows=sel)
    assert cosine(full[sel], sub).min() > 0.99999
