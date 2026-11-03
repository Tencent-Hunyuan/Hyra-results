"""Edge-buffer growth and optional EFM tiling for one andante engine instance.

Installed through the captured constructor's post-install, pre-preparation
callback. The tiled recipe (edge tiles of 131072, compiled blocks 1 and 2,
native blocks 0 and 3) bounds EFM memory for very large systems; route
selection belongs to the engine dispatcher (capacity_policy).
"""
from __future__ import annotations

from copy import deepcopy
import math

from hyff.engines.capture.edge_growth import install_growth_policy
from hyff.engines.capture.partial_tiling import install_tiled_terms


TILE_EDGES = 131072
TILED_BLOCKS = (1, 2)
NATIVE_BLOCKS = (0, 3)
EDGE_RESERVE = 1.5
DEFAULT_EDGE_RESERVE = 1.05
_EXPECTED_DIMENSIONS = {
    0: dict(channels=64, input_is_scalar=True, m_blocks=[], depth=2),
    1: dict(channels=64, input_is_scalar=False, m_blocks=[1, 2], depth=2),
    2: dict(channels=64, input_is_scalar=False, m_blocks=[1, 2], depth=2),
    3: dict(channels=32, input_is_scalar=False, m_blocks=[1, 2], depth=0),
}


def _dimensions(state):
    return {index: dict(channels=int(efm.channels),
                        input_is_scalar=bool(efm.input_is_scalar),
                        m_blocks=list(efm.m_blocks),
                        depth=len(getattr(efm, 'layers', ())))
            for index, efm in sorted(state.efms.items())}


def _require_native_identity(state):
    for index in NATIVE_BLOCKS:
        if (state.compiled_term[index] is not state._tiled_efm_original_terms[index]
                or state.term_backend[index] != state._tiled_efm_original_backends[index]):
            raise RuntimeError('Unselected EFM block callable or backend changed')


def _reserve(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('EFM edge reserve must be finite and at least 1')
    value = float(value)
    if not math.isfinite(value) or value < 1.:
        raise ValueError('EFM edge reserve must be finite and at least 1')
    return value


def make_post_install_efm(*, edge_reserve=EDGE_RESERVE):
    """Return a callback for the captured constructor's post-install, pre-prepare hook.

    The recipe is fixed: edge tiles of 131072, compiled blocks 1/2, native
    blocks 0/3. The dispatcher passes its resolved edge reserve (the standalone
    default is 1.5).
    """
    edge_reserve = _reserve(edge_reserve)

    def install(state):
        if set(state.efms) != set(TILED_BLOCKS + NATIVE_BLOCKS):
            raise ValueError('The tiled recipe requires EFM blocks 0, 1, 2, 3')
        dimensions = _dimensions(state)
        if dimensions != _EXPECTED_DIMENSIONS:
            raise ValueError('The tiled recipe does not match the EFM architecture')
        if state.efm_precision != 'fp32' or state.reserve != edge_reserve:
            raise ValueError('The tiled recipe requires fp32 EFM and the resolved edge reserve')
        report = install_tiled_terms(state, tile_edges=TILE_EDGES,
                                     compile_tiles=True, tiled_blocks=TILED_BLOCKS)
        _require_native_identity(state)
        report.update(compile_tiles_requested=True,
                      route_policy='native_unselected_blocks_exact_guarded_selector',
                      actual_efm_keys=sorted(state.efms),
                      requested_tiled_blocks=list(TILED_BLOCKS),
                      efm_dimensions=dimensions,
                      actual_backends=dict(state.term_backend),
                      efm_edge_reserve=float(state.reserve),
                      installed_before_prepare=True,
                      installed_before_first_force=True,
                      installation='instance_local_post_install_efm_callback')
        state._owned_tiled_selection_report = report

    return install


def make_post_install(*, edge_reserve=DEFAULT_EDGE_RESERVE, tiled=False):
    """Bind growth first, then optional tiling, once per fresh EFM state.

    Binding the growth policy does not size buffers or perform a growth event.
    Later overflow grows the same state and recaptures without this callback.
    """
    edge_reserve = _reserve(edge_reserve)
    if type(tiled) is not bool:
        raise ValueError('The tiled selection must be a bool')
    install_tiling = make_post_install_efm(edge_reserve=edge_reserve) if tiled else None

    def install(state):
        if (getattr(state, '_hyff_efm_runtime_policy', None) is not None
                or getattr(state, '_efm_growth_policy', None) is not None
                or getattr(state, '_tiled_efm_original_terms', None) is not None
                or any(getattr(state, name, None) is not None for name in ('batch', 'csr', 'capacity'))):
            raise ValueError('A fresh unprepared EFM state is required')
        keys = set(state.efms)
        if (state.efm_precision != 'fp32' or state.reserve != edge_reserve or not keys
                or keys != set(state.compiled_term) or keys != set(state.term_backend)):
            raise ValueError('The EFM policy does not match the fresh state')
        original_check, original_set = state.check_overflow, state.set_capacity
        growth = install_growth_policy(state, reserve=edge_reserve)
        if state.check_overflow != original_check or state.set_capacity != original_set:
            raise RuntimeError('The EFM overflow check or initial sizing changed')
        if install_tiling is not None:
            install_tiling(state)
        state._hyff_efm_runtime_policy = dict(
            installation='instance_local_post_install_efm_callback', installations=1,
            installed_before_prepare=True, installed_before_first_force=True,
            efm_edge_reserve=float(state.reserve), initial_slots=int(state.slots),
            growth=deepcopy(growth), selected_tiling=tiled,
            callback_replayed_after_growth=False)

    return install


def runtime_metadata(state):
    """Return construction policy metadata, not a runtime growth trace."""
    report = getattr(state, '_hyff_efm_runtime_policy', None)
    if report is None or state.reserve != report['efm_edge_reserve']:
        raise ValueError('The EFM runtime policy has not been installed or its reserve changed')
    return deepcopy(report)


def tiled_metadata(state):
    """Describe an installed tiled recipe after the captured engine's first force."""
    report = getattr(state, '_owned_tiled_selection_report', None)
    if report is None or state.reserve != report['efm_edge_reserve']:
        raise ValueError('The tiled recipe has not been installed or its reserve changed')
    if state.capacity is None:
        raise ValueError('Tiled metadata requires the prepared edge capacity')
    _require_native_identity(state)
    result = deepcopy(report)
    result.update(expert_alias_applied=False, precision='fp32', statistics='fp64',
                  efm_edge_reserve=float(state.reserve), padded_edge_capacity=int(state.capacity))
    return result
