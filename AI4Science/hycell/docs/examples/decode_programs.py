"""Gene programs from the HyCell decoder: decoded fold change between two groups of cells.

    python decode_programs.py cells.h5ad --groupby cell_type --a "Naive B cells" --b "Plasma cells" --top 15

Each cell is embedded and decoded back to expression over the genes the dataset measures, with the decoder's
code for an unknown dataset and donor. The decoded counts per 10,000 are averaged within each group, and genes
are ranked by log2((B + 1) / (A + 1)). The fold change from the counts themselves is printed alongside.
"""
import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import hycell
from hycell.data import open_counts


def group_mean_decoded(model, z, genes, chunk=512):
    total = np.zeros(len(genes))
    for i in range(0, len(z), chunk):
        total += model.decode(z[i:i + chunk], genes=genes).sum(0)
    return total / len(z)


def group_mean_counts(adata, mask, cols, layer=None):
    x = adata.layers[layer] if layer else adata.X
    x = x[mask][:, cols]
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    return (x / x.sum(1, keepdims=True).clip(min=1) * 1e4).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--groupby', required=True)
    ap.add_argument('--a', required=True, help='reference group')
    ap.add_argument('--b', required=True, help='target group')
    ap.add_argument('--layer')
    ap.add_argument('--top', type=int, default=15)
    a = ap.parse_args()

    adata = ad.read_h5ad(a.input)
    model = hycell.load()
    groups = adata.obs[a.groupby].astype(str).to_numpy()
    ma, mb = groups == a.a, groups == a.b
    if not ma.any() or not mb.any():
        raise SystemExit(f'groups {a.a!r} and {a.b!r} must both be present in obs[{a.groupby!r}]')
    sel = np.flatnonzero(ma | mb)
    z = model.embed(adata[sel].copy(), layer=a.layer, rows=sel)   # rows: the embeddings of the full file
    za, zb = z[ma[sel]], z[mb[sel]]
    genes = model.measured_genes(adata, layer=a.layer)
    dec = np.log2(group_mean_decoded(model, zb, genes) + 1) - np.log2(group_mean_decoded(model, za, genes) + 1)

    table = model.genes.set_index('ensembl_id').loc[genes, ['symbol', 'gene_type']].reset_index()
    table['decoded_log2fc'] = dec
    src = open_counts(adata, model.genes, layer=a.layer)
    col_of = {}
    for j, v in enumerate(src.gene_map.vocab_id):
        if v >= 0:
            col_of.setdefault(model.genes.ensembl_id.values[v], j)
    cols = [col_of[g] for g in genes]
    table['counts_log2fc'] = (np.log2(group_mean_counts(adata, mb, cols, a.layer) + 1)
                              - np.log2(group_mean_counts(adata, ma, cols, a.layer) + 1))
    table = table[table.gene_type == 'protein_coding'].sort_values('decoded_log2fc')
    pd.set_option('display.width', 120)
    print(f'{a.b} vs {a.a}: {mb.sum()} and {ma.sum()} cells, {len(genes)} measured genes')
    print('\nhighest in', a.b)
    print(table.tail(a.top)[::-1].to_string(index=False, float_format='%.2f'))
    print('\nhighest in', a.a)
    print(table.head(a.top).to_string(index=False, float_format='%.2f'))
    print(f'\nSpearman correlation of decoded and counted fold change: '
          f'{table.decoded_log2fc.corr(table.counts_log2fc, method="spearman"):.3f}')


if __name__ == '__main__':
    main()
