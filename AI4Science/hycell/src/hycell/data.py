"""Read raw counts from an AnnData object or an .h5ad file, one cell at a time, in vocabulary space."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .genes import GeneMap, map_genes

CHUNK = 4096


@dataclass
class CountSource:
    """Counts of an AnnData (in memory or backed) restricted to vocabulary genes."""
    matrix: object             # cells x genes: scipy sparse, numpy array or backed h5ad dataset
    obs_names: np.ndarray
    gene_map: GeneMap
    zero_pool: np.ndarray      # ascending vocabulary ids of measured core genes
    duplicated: bool           # several input genes map to one vocabulary gene (their counts are summed)
    adata: object = None

    @property
    def n_cells(self) -> int:
        return self.matrix.shape[0]

    def rows(self, start, stop):
        """CSR block of rows [start, stop)."""
        block = self.matrix[start:stop]
        if hasattr(block, 'to_memory'):
            block = block.to_memory()
        return block.tocsr() if sp.issparse(block) else sp.csr_matrix(np.asarray(block))

    def cells(self, start=0, stop=None):
        """Yield (row, ascending vocabulary ids, float64 counts) for every cell; counts are > 0."""
        stop = self.n_cells if stop is None else stop
        vid = self.gene_map.vocab_id
        for a in range(start, stop, CHUNK):
            b = min(a + CHUNK, stop)
            m = self.rows(a, b)
            ip, ind, dat = m.indptr, m.indices, m.data
            for i in range(b - a):
                cols, x = ind[ip[i]:ip[i + 1]], dat[ip[i]:ip[i + 1]].astype(np.float64)
                v = vid[cols]
                keep = (v >= 0) & (x > 0)
                v, x = v[keep], x[keep]
                if self.duplicated:
                    v, inv = np.unique(v, return_inverse=True)
                    x = np.bincount(inv, weights=x)
                else:
                    o = np.argsort(v)
                    v, x = v[o], x[o]
                yield a + i, v.astype(np.int32), x


def _check_counts(matrix):
    block = matrix[: min(matrix.shape[0], 500)]
    if hasattr(block, 'to_memory'):
        block = block.to_memory()
    x = block.data if sp.issparse(block) else np.asarray(block).ravel()
    if x.size == 0:
        return
    if (x < 0).any():
        raise ValueError('HyCell needs raw counts; the input has negative values')
    if np.abs(x - np.round(x)).max() > 1e-3:
        warnings.warn('the input values are not integers; HyCell expects raw counts (pass layer= or use_raw=True '
                      'if the counts are stored elsewhere)', stacklevel=3)


def open_counts(data, genes: pd.DataFrame, layer: str | None = None, use_raw: bool = False,
                gene_key: str | None = None) -> CountSource:
    """data: AnnData or path to an .h5ad file (read in backed mode). Counts come from ``layer``, from
    ``adata.raw`` (use_raw) or from X. Genes outside the vocabulary are dropped."""
    import anndata as ad
    adata = ad.read_h5ad(data, backed='r') if isinstance(data, (str, Path)) else data
    if use_raw:
        if adata.raw is None:
            raise ValueError('use_raw=True but adata.raw is empty')
        matrix, var = adata.raw.X, adata.raw.var
    else:
        matrix = adata.layers[layer] if layer is not None else adata.X
        var = adata.var
    if matrix is None:
        raise ValueError('the input has no count matrix')
    gm = map_genes(var, genes, gene_key)
    if gm.n_mapped == 0:
        raise ValueError(f'no input gene matches the HyCell vocabulary (tried {gm.key!r} as {gm.by})')
    if gm.n_mapped < 0.5 * len(gm.vocab_id):
        warnings.warn(f'only {gm.n_mapped} of {len(gm.vocab_id)} input genes map to the HyCell vocabulary '
                      f'({gm.key!r} as {gm.by})', stacklevel=2)
    _check_counts(matrix)
    ok = gm.vocab_id >= 0
    measured = np.unique(gm.vocab_id[ok]).astype(np.int32)
    core = genes.core.values
    return CountSource(matrix=matrix, obs_names=np.asarray(adata.obs_names, dtype=object), gene_map=gm,
                       zero_pool=measured[core[measured]], duplicated=len(measured) < int(ok.sum()), adata=adata)
