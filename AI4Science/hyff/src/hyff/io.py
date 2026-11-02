"""Prepared system files: loading and validation of system and restart NPZ files.

A system NPZ holds ``positions_A`` (N, 3), ``atomic_numbers`` (N,), and for
MD also ``velocities_A_ps``, ``masses``, ``box_A`` (3, 3), ``constraint_pairs``,
``constraint_distances_A``, ``total_charge`` and optionally ``multiplicity``,
``water_idx`` and ``bond_pairs``. See docs/inputs.md.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np


def load_system(name):
    """Load an explicit prepared .npz and optional sibling .json metadata."""
    path = Path(name).expanduser().resolve()
    if path.suffix != '.npz':
        raise ValueError('system must be an explicit prepared .npz path')
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as prepared:
        data = {key: prepared[key].copy() for key in prepared.files}
    meta = path.with_suffix('.json')
    data['meta'] = json.loads(meta.read_text()) if meta.is_file() else {}
    return data


MD_KEYS = ('positions_A', 'atomic_numbers', 'velocities_A_ps', 'masses', 'box_A',
           'constraint_pairs', 'constraint_distances_A', 'total_charge')


def check_md_system(system, name='system'):
    """Validate a prepared MD system; raises ValueError with the problem and the key concerned."""
    missing = [key for key in MD_KEYS if key not in system]
    if missing:
        hint = (' (this looks like a single-structure file; MD also needs velocities, masses, a box and '
                'constraints, see docs/inputs.md)' if {'velocities_A_ps', 'masses'} <= set(missing) else
                ' (see docs/inputs.md)')
        raise ValueError(f'{name} is missing MD keys: {", ".join(missing)}{hint}')

    def require(condition, message):
        if not condition:
            raise ValueError(f'{name}: {message}')

    x, z = np.asarray(system['positions_A']), np.asarray(system['atomic_numbers'])
    require(x.ndim == 2 and x.shape[1:] == (3,) and len(x) > 0 and x.dtype.kind in 'fiu'
            and np.isfinite(x).all(), 'positions_A must be a finite (N, 3) array')
    n = len(x)
    require(z.shape == (n,) and z.dtype.kind in 'iu', f'atomic_numbers must be N = {n} integers')
    bad = sorted(set(z[(z < 1) | (z > 83)].tolist()))
    require(not bad, f'atomic_numbers outside 1-83 (H to Bi): {bad[:5]}')
    v = np.asarray(system['velocities_A_ps'])
    require(v.shape == (n, 3) and v.dtype.kind in 'fiu' and np.isfinite(v).all(),
            'velocities_A_ps must be a finite (N, 3) array in A/ps')
    m = np.asarray(system['masses'])
    require(m.shape == (n,) and m.dtype.kind in 'fiu' and np.isfinite(m).all() and (m > 0).all(),
            'masses must be N finite positive values in amu')
    box = np.asarray(system['box_A'])
    require(box.shape == (3, 3) and box.dtype.kind in 'fiu' and np.isfinite(box).all(),
            'box_A must be a finite (3, 3) cell (rows are lattice vectors)')
    require(float(np.linalg.det(box.astype(np.float64))) > 0,
            'box_A must have positive volume (right-handed lattice vectors)')
    pairs, distances = np.asarray(system['constraint_pairs']), np.asarray(system['constraint_distances_A'])
    if pairs.size == 0:
        pairs = pairs.reshape(0, 2)
    require(pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.dtype.kind in 'iu',
            'constraint_pairs must be a (C, 2) integer array; use np.zeros((0, 2), np.int64) for none')
    require(((pairs >= 0) & (pairs < n)).all() and (pairs[:, 0] != pairs[:, 1]).all(),
            'constraint_pairs must hold distinct atom indices 0..N-1')
    require(distances.shape == (len(pairs),) or (distances.size == 0 and not len(pairs)),
            'constraint_distances_A must hold one distance per constraint pair')
    require(distances.dtype.kind in 'fiu' and np.isfinite(distances).all() and (distances > 0).all(),
            'constraint_distances_A must be finite and positive')
    charge = np.asarray(system['total_charge'])
    require(charge.shape == () and charge.dtype.kind in 'fiu' and np.isfinite(charge),
            'total_charge must be a finite scalar')
    for key in ('water_idx', 'bond_pairs'):
        if key in system:
            value = np.asarray(system[key])
            require(value.dtype.kind in 'iu' and ((value >= 0) & (value < n)).all(),
                    f'{key} must hold integer atom indices 0..N-1')
    if 'bond_pairs' in system:
        require(np.asarray(system['bond_pairs']).reshape(-1, 2).shape[0] * 2 == np.asarray(system['bond_pairs']).size,
                'bond_pairs must be a (B, 2) integer array')
    system_multiplicity(system)
    return system


def system_multiplicity(system):
    """Spin multiplicity 2S+1 of a system; a system without one has M=1."""
    value = np.asarray(system.get('multiplicity', 1))
    if value.shape != () or value.dtype.kind not in 'iu' or int(value) < 1:
        raise ValueError(f'system multiplicity must be a positive integer scalar 2S+1 (e.g. np.int64(2)), '
                         f'got {value!r}')
    return int(value)


def atomic_numbers_sha256(numbers):
    """Ordered species identity, independent of the input integer width."""
    value = np.asarray(numbers)
    if value.ndim != 1 or value.dtype.kind not in 'iu' or not len(value):
        raise ValueError('atomic_numbers must be a nonempty integer vector')
    value = np.ascontiguousarray(value, dtype='<i8')
    header = json.dumps([value.dtype.str, list(value.shape)], separators=(',', ':')).encode('ascii')
    return hashlib.sha256(len(header).to_bytes(8, 'big') + header + value.tobytes()).hexdigest()


class RestartState(dict):
    """Loaded arrays plus the path and sha256 of the exact source bytes deserialized."""


def load_restart_state(name):
    path = Path(name).expanduser().resolve()
    payload = path.read_bytes()
    fields = ('positions_A', 'velocities_A_fs', 'total_charge', 'multiplicity',
              'bounded_positions_A', 'molecule_images', 'molecule_ids', 'periodic_cell_A',
              'n_atoms', 'atomic_numbers_sha256', 'time_ps', 'committed_steps', 'box_A')
    with np.load(io.BytesIO(payload), allow_pickle=False) as selected:
        state = RestartState({key: selected[key].copy() for key in fields if key in selected})
    state.source_path = str(path)
    state.source_sha256 = hashlib.sha256(payload).hexdigest()
    return state


def restart_source(initial):
    """Source of a restart; metadata missing from older restart files stays None, segment counters still reset."""
    return dict(source_path=getattr(initial, 'source_path', None),
                source_sha256=getattr(initial, 'source_sha256', None),
                time_ps=float(initial['time_ps']) if 'time_ps' in initial else None,
                committed_steps=int(initial['committed_steps']) if 'committed_steps' in initial else None,
                scope='SOURCE_CHECKPOINT_FOR_NEW_SEGMENT')


def validate_restart_state(system, initial):
    """Check the identity/metadata fields present in a restart; fields absent from older files are skipped."""
    if 'total_charge' in initial:
        charge = np.asarray(initial['total_charge'])
        if (charge.shape != () or charge.dtype.kind not in 'fiu' or
                not np.isfinite(charge) or float(charge) != float(system['total_charge'])):
            raise ValueError('restart total_charge must be a finite scalar matching the system')
    if 'multiplicity' in initial and system_multiplicity(initial) != system_multiplicity(system):
        raise ValueError('restart multiplicity must match the system')
    if 'n_atoms' in initial:
        count = np.asarray(initial['n_atoms'])
        if (count.shape != () or count.dtype.kind not in 'iu' or
                int(count) != len(system['atomic_numbers'])):
            raise ValueError('restart n_atoms must be an integer scalar matching the system')
    if 'atomic_numbers_sha256' in initial:
        digest = np.asarray(initial['atomic_numbers_sha256'])
        if (digest.shape != () or digest.dtype.kind != 'U' or
                str(digest) != atomic_numbers_sha256(system['atomic_numbers'])):
            raise ValueError('restart atomic_numbers_sha256 must match the ordered system species')
    if 'time_ps' in initial:
        stamp = np.asarray(initial['time_ps'])
        if (stamp.shape != () or stamp.dtype.kind not in 'fiu' or
                not np.isfinite(stamp) or float(stamp) < 0):
            raise ValueError('restart time_ps must be a finite nonnegative scalar')
    if 'committed_steps' in initial:
        steps = np.asarray(initial['committed_steps'])
        if steps.shape != () or steps.dtype.kind not in 'iu' or int(steps) < 0:
            raise ValueError('restart committed_steps must be a nonnegative integer scalar')
    if 'box_A' in initial:
        box = np.asarray(initial['box_A'])
        if (box.shape != (3, 3) or box.dtype.kind not in 'fiu' or not np.isfinite(box).all() or
                float(np.linalg.det(box.astype(np.float64))) <= 0):
            raise ValueError('restart box_A must be a finite (3, 3) cell with positive volume')
