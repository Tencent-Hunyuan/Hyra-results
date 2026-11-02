"""HyCell: a compact embedding model for single-cell transcriptomes.

>>> import hycell
>>> model = hycell.load()                 # the released weights ship with the package
>>> adata.obsm['X_hycell'] = model.embed(adata)
>>> tuned = model.finetune(reference, 'cell_type')   # optional: an annotator for new data
>>> query.obs['predicted'] = tuned.predict(query)
"""
from .api import HyCell, embed, load
from .finetune import finetune
from .genes import load_genes

__version__ = '0.1.0'
__all__ = ['HyCell', 'load', 'embed', 'finetune', 'load_genes', '__version__']
