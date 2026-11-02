"""Transfer cell-type labels between batches with a k-nearest-neighbour classifier on HyCell embeddings.

    python label_transfer.py cells.h5ad --label cell_type --batch batch            # every batch held out in turn
    python label_transfer.py cells.h5ad --label cell_type --batch batch --query 2  # one query batch

Each held-out batch is annotated from the cells of the other batches (15 neighbours), and the macro-F1 and
accuracy against its own labels are reported. This is the kNN annotation-transfer protocol of the HyCell
benchmark. Needs scikit-learn.
"""
import argparse

import anndata as ad
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

import hycell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--label', required=True, help='obs column with cell types')
    ap.add_argument('--batch', required=True, help='obs column with batches')
    ap.add_argument('--query', help='batch to annotate (default: each batch in turn)')
    ap.add_argument('--layer')
    ap.add_argument('-k', type=int, default=15)
    a = ap.parse_args()

    adata = ad.read_h5ad(a.input)
    z = hycell.load().embed(adata, layer=a.layer)
    labels = adata.obs[a.label].astype(str).to_numpy()
    batches = adata.obs[a.batch].astype(str).to_numpy()
    queries = [a.query] if a.query else sorted(set(batches))
    scores = []
    for q in queries:
        test = batches == q
        knn = KNeighborsClassifier(n_neighbors=a.k).fit(z[~test], labels[~test])
        pred = knn.predict(z[test])
        f1, acc = f1_score(labels[test], pred, average='macro'), accuracy_score(labels[test], pred)
        scores.append(f1)
        print(f'batch {q}: {test.sum()} cells, macro-F1 {f1:.3f}, accuracy {acc:.3f}')
    if len(scores) > 1:
        print(f'mean macro-F1 over {len(scores)} held-out batches: {np.mean(scores):.3f}')


if __name__ == '__main__':
    main()
