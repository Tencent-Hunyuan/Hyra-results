import numpy as np
import pandas as pd
import pytest

from hycell.data import open_counts
from hycell.genes import load_genes, map_genes


@pytest.fixture(scope='module')
def genes():
    return load_genes()


def test_vocabulary(genes):
    assert len(genes) == 36495 and genes.core.sum() == 10131
    assert genes.ensembl_id.str.startswith('ENSG').all() and not genes.ensembl_id.duplicated().any()


def test_map_by_ensembl_column_with_versions(genes):
    var = pd.DataFrame({'ensembl_id': ['ENSG00000168484.12', 'ENSG00000149021', 'not_a_gene']},
                       index=['SFTPC', 'SCGB1A1', 'X'])
    m = map_genes(var, genes)
    assert m.by == 'ensembl' and m.key == 'ensembl_id'
    assert genes.symbol.values[m.vocab_id[:2]].tolist() == ['SFTPC', 'SCGB1A1'] and m.vocab_id[2] == -1


def test_map_by_symbol(genes):
    var = pd.DataFrame(index=['SFTPC', 'sftpc', 'PTPRC', 'NOT_A_GENE'])
    m = map_genes(var, genes)
    assert m.by == 'symbol' and m.vocab_id[0] == m.vocab_id[1] >= 0 and m.vocab_id[3] == -1


def test_duplicate_genes_are_summed(genes):
    import anndata as ad
    import scipy.sparse as sp
    var = pd.DataFrame({'ensembl_id': ['ENSG00000168484', 'ENSG00000168484', 'ENSG00000149021']},
                       index=['a', 'b', 'c'])
    a = ad.AnnData(sp.csr_matrix(np.array([[2, 3, 0], [0, 1, 4]], np.float32)), var=var)
    src = open_counts(a, genes)
    cells = list(src.cells())
    sftpc = int(genes.index[genes.symbol == 'SFTPC'][0])
    assert cells[0][1].tolist() == [sftpc] and cells[0][2].tolist() == [5.0]
    assert len(cells[1][1]) == 2


def test_lung_fixture_uses_ensembl_column(lung, genes):
    src = open_counts(lung, genes)
    assert src.gene_map.key == 'ensembl_id' and src.gene_map.n_mapped > 13000
    assert len(src.zero_pool) > 0 and genes.core.values[src.zero_pool].all()
