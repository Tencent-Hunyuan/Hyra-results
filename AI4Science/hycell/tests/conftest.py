import os
from pathlib import Path

import pytest

DATA = Path(__file__).parent / 'data'


@pytest.fixture(scope='session')
def weights():
    """$HYCELL_TEST_WEIGHTS or the weights that ship with the package."""
    from hycell.weights import packaged
    return Path(os.environ.get('HYCELL_TEST_WEIGHTS') or packaged())


@pytest.fixture(scope='session')
def lung():
    import anndata as ad
    return ad.read_h5ad(DATA / 'lung_512.h5ad')


@pytest.fixture(scope='session')
def expected():
    import numpy as np
    return dict(np.load(DATA / 'lung_512_expected.npz'))
