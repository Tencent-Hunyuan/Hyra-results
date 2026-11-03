"""Long-range evaluation of the largo latent-charge term: native, dense, tree or SPME.

``auto`` selects the periodic SPME sum for a system with a cell (the only route that is invariant to
lattice translations of molecules) and the isolated sum otherwise: dense up to ``LQ_DENSE_MAX`` atoms
(default 4096), tree above. All settings are checked here, before any model preparation or kernel
compilation.

Environment settings (read when an engine is built):

| variable | default | meaning |
|---|---|---|
| LQ_DENSE_MAX | 4096 | auto: dense through this many atoms, tree above (isolated systems) |
| LQ_ORDER, LQ_LEAF, LQ_THETA, LQ_DELTA, LQ_RCUT_SIGMA, LQ_HEADROOM | 6, 32, 0.4, 1.5, 6.0, 1.25 | tree expansion order, leaf size, opening angle, plan displacement (A), near-field radius (sigma), buffer headroom |
| LQ_SPME_H_SIGMA | 0.5 | SPME mesh spacing in units of sigma (an upper bound; the mesh has at least LQ_SPME_ORDER points) |
| LQ_SPME_ORDER | 6 | B-spline order |
| LQ_SPME_DTYPE | fp32 | mesh arithmetic: fp32 or fp64 |
| LQ_SPME_MAX_H_SIGMA | = LQ_SPME_H_SIGMA | guard: largest accepted spacing/sigma |
"""
from __future__ import annotations

import math
import os

import numpy as np

DEFAULT_DENSE_MAX = 4096


def _value(name, cast, default, check, requirement):
    raw = os.environ.get(name)
    if raw in (None, ''):
        return default
    try:
        value = cast(raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name}={raw!r}: {requirement}') from None
    if not check(value):
        raise ValueError(f'{name}={raw!r}: {requirement}')
    return value


def _positive(value):
    return math.isfinite(value) and value > 0


def settings():
    """The long-range settings, from the environment or the defaults; ValueError for an invalid value."""
    values = dict(
        LQ_DENSE_MAX=_value('LQ_DENSE_MAX', int, DEFAULT_DENSE_MAX, lambda v: v >= 1, 'a positive integer'),
        LQ_ORDER=_value('LQ_ORDER', int, 6, lambda v: 1 <= v <= 12, 'an integer from 1 to 12'),
        LQ_LEAF=_value('LQ_LEAF', int, 32, lambda v: v >= 1, 'a positive integer'),
        LQ_THETA=_value('LQ_THETA', float, .4, lambda v: math.isfinite(v) and 0 < v < 1, 'a number in (0, 1)'),
        LQ_DELTA=_value('LQ_DELTA', float, 1.5, _positive, 'a positive number of Angstrom'),
        LQ_RCUT_SIGMA=_value('LQ_RCUT_SIGMA', float, 6., _positive, 'a positive number'),
        LQ_HEADROOM=_value('LQ_HEADROOM', float, 1.25, lambda v: math.isfinite(v) and v >= 1, 'a number >= 1'),
        LQ_SPME_H_SIGMA=_value('LQ_SPME_H_SIGMA', float, .5, _positive, 'a positive number'),
        LQ_SPME_ORDER=_value('LQ_SPME_ORDER', int, 6, lambda v: 3 <= v <= 16, 'an integer from 3 to 16'),
        LQ_SPME_DTYPE=_value('LQ_SPME_DTYPE', str, 'fp32', lambda v: v in ('fp32', 'fp64'), 'fp32 or fp64'))
    values['LQ_SPME_MAX_H_SIGMA'] = _value('LQ_SPME_MAX_H_SIGMA', float, values['LQ_SPME_H_SIGMA'],
                                           lambda v: _positive(v) and v >= values['LQ_SPME_H_SIGMA'],
                                           'a positive number >= LQ_SPME_H_SIGMA')
    return values


def check_cell(cell):
    """A periodic cell for SPME: (3, 3), finite, nonzero volume."""
    if cell is None:
        raise ValueError('long_range=spme needs a periodic cell (box); for an isolated system use auto, dense or tree')
    value = np.asarray(cell, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all() or abs(float(np.linalg.det(value))) <= 0:
        raise ValueError('long_range=spme needs a finite (3, 3) cell with nonzero volume')
    return value


def resolve_long_range(kind, requested, n_atoms, periodic=False):
    """Selected route for a request: auto -> spme with a cell, dense/tree without one."""
    if requested is None:
        return None
    if requested not in ('native', 'dense', 'tree', 'auto', 'spme'):
        raise ValueError('long_range must be native, dense, tree, auto, spme, or None')
    if kind != 'largo':
        raise ValueError('long_range is supported only by largo')
    if type(n_atoms) is not int or n_atoms < 1:
        raise ValueError('long_range requires a positive atom count')
    if requested == 'spme' and not periodic:
        raise ValueError('long_range=spme needs a periodic cell (box); for an isolated system use auto, dense or tree')
    values = settings()
    dense_max = values['LQ_DENSE_MAX']
    if requested == 'auto':
        selected = 'spme' if periodic else ('dense' if n_atoms <= dense_max else 'tree')
    else:
        selected = requested
    return dict(requested=requested, selected=selected, n_atoms=n_atoms, periodic=bool(periodic),
                dense_through_atoms=dense_max, selection_phase='construction_before_preparation',
                isolated_nonperiodic_sum=selected != 'spme',
                periodic_image_sum=selected == 'spme',
                extrapolation=selected == 'spme')


def install_long_range(model, physical, policy):
    """Install once after the latent-charge context, before prepare/force/capture.

    Every route keeps native charge conservation and its autograd chain.
    Only SPME changes the isolated sum to a periodic image sum.
    """
    if policy['selected'] == 'native':
        return dict(policy)
    if policy['selected'] not in ('dense', 'tree', 'spme'):
        raise ValueError('Unsupported resolved long_range route')
    term = getattr(model, 'latent_charge_energy', None)
    if term is None or getattr(model, 'training', True) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Long range requires a frozen evaluation largo model')
    if ('_long_range_original_forward' in term.__dict__ or
            '_long_range_original_prepare_energy' in model.__dict__):
        raise ValueError('Long range is already installed')
    state = model.__dict__.get('_efm_state')
    if state is None or any(getattr(state, name, None) is not None for name in ('batch', 'csr', 'capacity')):
        raise ValueError('Install long range before EFM preparation')
    metadata = model.__dict__.get('_latent_charge_prepared_metadata')
    if not isinstance(metadata, tuple) or len(metadata) != 2:
        raise ValueError('Install the latent-charge prepared context before long range')
    atom_batch, total_charge = metadata
    if atom_batch.numel() != policy['n_atoms'] or total_charge.numel() != 1 or bool((atom_batch != 0).any()):
        raise ValueError('Long range supports one graph with zero atom_batch only')
    from hyff.kernels.long_range.backends import install_dense, install_tree, install_spme
    settings()
    installer = {'dense': install_dense, 'tree': install_tree, 'spme': install_spme}[policy['selected']]
    report = installer(model, physical, 'cuda')
    # Keep the installer's extrapolation description next to the policy boolean.
    extrapolation_detail = report.pop('extrapolation', None)
    return dict(report, **policy,
                extrapolation_detail=extrapolation_detail,
                full_model_step_validated=False)
