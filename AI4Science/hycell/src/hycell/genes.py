"""The HyCell gene vocabulary and the mapping of a dataset's genes onto it.

The vocabulary has 36,495 human genes: every protein-coding gene plus every gene detected in at least 0.1% of
the pretraining cells. Genes are identified by Ensembl gene id; symbols are accepted as a fallback. The
``core`` flag marks the 10,131 genes measured in at least 90% of the pretraining datasets, from which measured
zeros are drawn.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from importlib import resources

import numpy as np
import pandas as pd

ENSEMBL = re.compile(r'^ENS[A-Z]*G\d{11}(\.\d+)?$')
ENSEMBL_COLUMNS = ('ensembl_id', 'gene_ids', 'gene_id', 'feature_id', 'ensembl', 'ensembl_gene_id', 'gene_ensembl')
SYMBOL_COLUMNS = ('feature_name', 'gene_symbol', 'gene_symbols', 'gene_name', 'symbol')


def load_genes() -> pd.DataFrame:
    """The vocabulary table: ensembl_id, symbol, gene_type, core (bool), in model order."""
    with resources.files('hycell').joinpath('assets/genes.tsv.gz').open('rb') as f:
        t = pd.read_csv(gzip.open(f), sep='\t', dtype={'ensembl_id': str, 'symbol': str, 'gene_type': str})
    t['core'] = t['core'].astype(bool)
    return t


@dataclass
class GeneMap:
    """Vocabulary id of every input gene (-1 when outside the vocabulary)."""
    vocab_id: np.ndarray        # (n_input_genes,) int32
    key: str                    # how the input genes were identified
    by: str                     # 'ensembl' or 'symbol'

    @property
    def n_mapped(self) -> int:
        return int((self.vocab_id >= 0).sum())

    def summary(self) -> dict:
        ok = self.vocab_id >= 0
        return {'key': self.key, 'by': self.by, 'n_genes': int(len(ok)), 'n_mapped': int(ok.sum()),
                'n_vocab_genes': int(len(np.unique(self.vocab_id[ok])))}


def _is_ensembl(values) -> float:
    v = [str(x) for x in values[:2000] if x is not None and str(x) not in ('', 'nan', 'None')]
    return sum(bool(ENSEMBL.match(x)) for x in v) / max(len(v), 1)


def map_ensembl(ids, genes: pd.DataFrame) -> np.ndarray:
    lut = {e: i for i, e in enumerate(genes.ensembl_id.values)}
    out = np.full(len(ids), -1, np.int32)
    for j, e in enumerate(ids):
        if e is None or (isinstance(e, float) and np.isnan(e)):
            continue
        e = str(e)
        if e:
            out[j] = lut.get(e.split('.')[0], -1)
    return out


def map_symbols(symbols, genes: pd.DataFrame) -> np.ndarray:
    """Symbols shared by several vocabulary genes resolve to the protein-coding one, then to the first."""
    order = np.lexsort((np.arange(len(genes)), (genes.gene_type.values != 'protein_coding')))
    lut, lut_upper = {}, {}
    for i in order:
        s = genes.symbol.values[i]
        lut.setdefault(s, i)
        lut_upper.setdefault(s.upper(), i)
    out = np.full(len(symbols), -1, np.int32)
    for j, s in enumerate(symbols):
        if s is None or (isinstance(s, float) and np.isnan(s)):
            continue
        s = str(s)
        out[j] = lut.get(s, lut_upper.get(s.upper(), -1))
    return out


def map_genes(var: pd.DataFrame, genes: pd.DataFrame, gene_key: str | None = None) -> GeneMap:
    """Map the rows of an AnnData ``var`` table onto the vocabulary.

    gene_key: a ``var`` column (or 'index' for var_names) holding Ensembl ids or symbols. By default the first
    of ensembl_id, gene_ids, gene_id, feature_id, ... that holds Ensembl ids is used, then var_names if they are
    Ensembl ids, and otherwise gene symbols from var_names or a symbol column."""
    def column(k):
        return np.asarray(var.index if k == 'index' else var[k].values, dtype=object)

    if gene_key is not None:
        if gene_key != 'index' and gene_key not in var.columns:
            raise KeyError(f'gene_key {gene_key!r} is not a var column')
        vals = column(gene_key)
        if _is_ensembl(vals) >= 0.5:
            return GeneMap(map_ensembl(vals, genes), gene_key, 'ensembl')
        return GeneMap(map_symbols(vals, genes), gene_key, 'symbol')
    for k in (*[c for c in ENSEMBL_COLUMNS if c in var.columns], 'index'):
        vals = column(k)
        if _is_ensembl(vals) >= 0.5:
            return GeneMap(map_ensembl(vals, genes), k, 'ensembl')
    best = None
    for k in ('index', *[c for c in SYMBOL_COLUMNS if c in var.columns]):
        m = GeneMap(map_symbols(column(k), genes), k, 'symbol')
        if best is None or m.n_mapped > best.n_mapped:
            best = m
    return best
