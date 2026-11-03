"""Fine-tune HyCell as a cell-type annotator on reference batches and annotate a query batch.

    python finetune_annotation.py cells.h5ad --label cell_type --batch batch --query B4

The cells of the query batch are annotated by a HyCell model fine-tuned on the other batches, and, for
comparison, by a 15-nearest-neighbour classifier on frozen HyCell embeddings. Prints the macro-F1 of both.
Needs scikit-learn.
"""
import argparse

import anndata as ad
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

import hycell


def macro_f1(y, pred):
    return f1_score(y, pred, labels=sorted(set(y)), average='macro', zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--label', required=True, help='obs column with cell types')
    ap.add_argument('--batch', required=True, help='obs column with batches')
    ap.add_argument('--query', required=True, help='batch to annotate')
    ap.add_argument('--save', help='directory to save the fine-tuned annotator')
    a = ap.parse_args()

    adata = ad.read_h5ad(a.input)
    batches = adata.obs[a.batch].astype(str).to_numpy()
    labels = adata.obs[a.label].astype(str).to_numpy()
    query = np.flatnonzero(batches == a.query)
    ref = np.flatnonzero(batches != a.query)
    if not len(query):
        raise SystemExit(f'no cells with obs[{a.batch!r}] == {a.query!r}')

    model = hycell.load()
    z = model.embed(adata)
    knn = KNeighborsClassifier(n_neighbors=15).fit(z[ref], labels[ref]).predict(z[query])

    tuned = model.finetune(adata[ref].copy(), a.label)
    pred = tuned.predict(adata[query].copy(), rows=query)
    print(f'query batch {a.query}: {len(query)} cells, reference: {len(ref)} cells')
    print(f'macro-F1, kNN on frozen embeddings: {macro_f1(labels[query], knn):.3f}')
    print(f'macro-F1, fine-tuned annotator:     {macro_f1(labels[query], pred):.3f}')
    if a.save:
        tuned.save(a.save)
        print('saved', a.save)


if __name__ == '__main__':
    main()
