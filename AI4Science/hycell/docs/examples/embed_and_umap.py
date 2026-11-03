"""Embed an .h5ad file with HyCell and draw a UMAP of the embedding.

    python embed_and_umap.py cells.h5ad --color cell_type batch --out umap.png

The input needs raw counts (X, or --layer / --use-raw). Writes the UMAP figure and, with --write, the input
with the embedding in obsm['X_hycell'] and the UMAP in obsm['X_umap']. Needs scanpy and matplotlib.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc

import hycell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--color', nargs='+', default=[], help='obs columns to colour by')
    ap.add_argument('--layer')
    ap.add_argument('--use-raw', action='store_true')
    ap.add_argument('--out', default='hycell_umap.png')
    ap.add_argument('--write', help='write the AnnData with the embedding to this .h5ad')
    a = ap.parse_args()

    adata = sc.read_h5ad(a.input)
    model = hycell.load()
    adata.obsm['X_hycell'] = model.embed(adata, layer=a.layer, use_raw=a.use_raw, progress=True)
    sc.pp.neighbors(adata, use_rep='X_hycell', n_neighbors=15)
    sc.tl.umap(adata, random_state=0)
    colors = a.color or [None]
    fig, axes = plt.subplots(1, len(colors), figsize=(5 * len(colors), 4.6), squeeze=False)
    for ax, c in zip(axes[0], colors):
        sc.pl.umap(adata, color=c, ax=ax, show=False, frameon=False, title=c or 'HyCell')
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print('wrote', a.out)
    if a.write:
        adata.write_h5ad(a.write)


if __name__ == '__main__':
    main()
