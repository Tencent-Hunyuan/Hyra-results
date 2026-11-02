"""Construction-time opt-in EFM capacity routing; no OOM or mid-run fallback."""
from __future__ import annotations


_THRESHOLDS = {'largo': 20388, 'andante': 50000}


def resolve_capacity_policy(kind, requested, n_atoms):
    """Return the resolved capacity policy, or None to keep the default engine path.

    The atom-count thresholds were set from measured runs. They are heuristics and
    cannot guarantee fit for arbitrary geometries, devices or neighbor counts.
    """
    if requested is None:
        return None
    if not isinstance(requested, str) or requested not in ('native', 'partial', 'tiled', 'auto'):
        raise ValueError('capacity_policy must be native, partial, tiled, auto, or None')
    if kind not in _THRESHOLDS:
        raise ValueError('capacity_policy is supported only by largo and andante')
    if requested == 'partial' and kind != 'largo':
        raise ValueError('partial capacity_policy is supported only by largo')
    if type(n_atoms) is not int or n_atoms < 1:
        raise ValueError('capacity routing requires a positive atom count')
    threshold = _THRESHOLDS[kind]
    selected = requested
    if requested == 'auto':
        selected = ('native' if n_atoms <= threshold else
                    'partial' if kind == 'largo' and n_atoms <= 51720 else 'tiled')
    report = dict(requested=requested, selected=selected, n_atoms=n_atoms,
                  native_through_atoms=threshold, threshold_is_fit_guarantee=False,
                  selection_phase='construction_before_preparation',
                  mid_run_switch=False, oom_recovery=False)
    if kind == 'largo':
        report.update(partial_through_atoms=51720, partial_measured_atoms=[51720],
                      intermediate_atom_counts_are_unmeasured=True)
    return report


def install_largo_capacity(model, policy, edge_reserve):
    """Install edge-buffer growth and, for the partial or tiled route, EFM tiling.

    Called after native install_efm, before latent-charge preparation and capture.
    Overflow rejection and the engine's rebuild/recapture are kept.
    """
    state = model.__dict__.get('_efm_state')
    if state is None or getattr(state, '_hyff_largo_capacity', None) is not None:
        raise ValueError('A fresh native largo EFM state is required')
    if any(getattr(state, name, None) is not None for name in ('batch', 'csr', 'capacity')):
        raise ValueError('Install capacity policy before EFM preparation')
    keys = set(state.efms)
    if not keys or keys != set(state.compiled_term) or keys != set(state.term_backend):
        raise ValueError('Native EFM blocks and callables must match')
    if policy['selected'] not in ('native', 'partial', 'tiled'):
        raise ValueError('Unsupported resolved largo capacity route')
    if getattr(state, '_tiled_efm_original_terms', None) is not None:
        raise ValueError('Tiled terms already installed')
    from hyff.engines.capture.edge_growth import install_growth_policy
    from hyff.engines.capture.tiled_recompute import install_tiled_terms
    growth = install_growth_policy(state, reserve=edge_reserve)
    tiled = None
    if policy['selected'] == 'tiled':
        tiled = install_tiled_terms(state, 32768, True)
    elif policy['selected'] == 'partial':
        from hyff.engines.capture.partial_tiling import install_tiled_terms as install_partial
        tiled = install_partial(state, 32768, True, tiled_blocks=(1, 2))
    result = dict(policy, growth=growth, tiling=tiled)
    state._hyff_largo_capacity = result
    return result
