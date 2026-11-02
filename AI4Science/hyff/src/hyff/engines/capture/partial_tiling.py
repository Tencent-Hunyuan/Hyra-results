"""Use the full-chain tiled EFM term (first VJP only) for selected blocks.

The tiled term is the one of tiled_recompute.py. Blocks outside the selection
keep their checked native callables/backends. Install only before preparing or
capturing the model.
"""
from hyff.engines.capture.tiled_recompute import make_tiled_term


def install_tiled_terms(state, tile_edges=32768, compile_tiles=True,
                        tiled_blocks=(1, 2)):
    if getattr(state, '_tiled_efm_original_terms', None) is not None:
        raise ValueError('Tiled terms already installed')
    if any(getattr(state, name, None) is not None
           for name in ('batch', 'csr', 'capacity')):
        raise ValueError('Partial tiling must precede prepare/evaluate/capture')
    selected = tuple(tiled_blocks)
    if not selected or any(type(index) is not int for index in selected):
        raise ValueError('tiled_blocks must contain integer block indices')
    if len(selected) != len(set(selected)):
        raise ValueError('Duplicate tiled block index')
    if not set(selected).issubset(state.efms):
        raise ValueError('Unknown EFM block selected')
    selected = tuple(sorted(selected))
    native = set(state.efms) - set(selected)
    original_terms = dict(state.compiled_term)
    original_backends = dict(state.term_backend)
    replacements = {index: make_tiled_term(state.efms[index], tile_edges,
                                            compile_tiles)
                    for index in selected}
    state._tiled_efm_original_terms = original_terms
    state._tiled_efm_original_backends = original_backends
    state.compiled_term.update(replacements)
    for index in selected:
        state.term_backend[index] = 'EXPERIMENTAL_full_chain_tiled_recompute'
    assert all(state.compiled_term[index] is original_terms[index]
               and state.term_backend[index] == original_backends[index]
               for index in native)
    return dict(tile_edges=tile_edges, compile_tiles=bool(compile_tiles),
                blocks=list(selected), native_blocks=sorted(native),
                native_callables_preserved=True,
                block_execution={index: function.tiled_efm_metadata
                                 for index, function in replacements.items()})
