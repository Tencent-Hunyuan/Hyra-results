"""Hydrogen mass repartitioning (HMR) for constrained MD."""
from __future__ import annotations

import math

import numpy as np


def water_indices(system):
    """Water atoms: ``water_idx`` when present, else every constraint cluster made of exactly one O and two H."""
    if 'water_idx' in system:
        return np.asarray(system['water_idx']).astype(np.int64).ravel()
    numbers = np.asarray(system['atomic_numbers'])
    pairs = np.asarray(system['constraint_pairs']).reshape(-1, 2)
    n = len(numbers)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in pairs:
        parent[root(int(i))] = root(int(j))
    clusters = {}
    for atom in np.unique(pairs):
        clusters.setdefault(root(int(atom)), []).append(int(atom))
    water = [sorted(atoms, key=lambda a: -numbers[a]) for atoms in clusters.values()
             if len(atoms) == 3 and sorted(numbers[atoms].tolist()) == [1, 1, 8]]
    return np.asarray(water, dtype=np.int64).reshape(-1)


def _require_hydrogen_constraints(system):
    numbers = np.asarray(system['atomic_numbers'])
    pairs = np.asarray(system['constraint_pairs']).reshape(-1, 2)
    if not len(pairs) or not (numbers[pairs] == 1).any():
        raise ValueError('Hydrogen mass repartitioning needs the X-H bonds in constraint_pairs '
                         '(they identify each hydrogen\'s heavy partner); this system has none')


def repartition_hydrogen_mass(system, hydrogen_mass):
    """OpenMM-style HMR: non-water H gets `hydrogen_mass`, its bonded heavy atom keeps the total."""
    try:
        hydrogen_mass = float(hydrogen_mass)
    except (TypeError, ValueError):
        raise ValueError('hmr_mass must be a finite positive mass in amu') from None
    if not (math.isfinite(hydrogen_mass) and hydrogen_mass > 0):
        raise ValueError('hmr_mass must be a finite positive mass in amu')
    _require_hydrogen_constraints(system)
    masses = system['masses'].astype(np.float64).copy()
    numbers = system['atomic_numbers']
    water = np.zeros(len(masses), dtype=bool)
    water[water_indices(system)] = True
    for i, j in np.asarray(system['constraint_pairs']).reshape(-1, 2):
        if water[i] or water[j]:
            continue
        h, heavy = (i, j) if numbers[i] == 1 else (j, i)
        if numbers[h] != 1 or numbers[heavy] == 1:
            continue
        delta = hydrogen_mass - masses[h]
        masses[h] += delta
        masses[heavy] -= delta
    if (masses <= 0).any():
        atom = int(np.argmin(masses))
        raise ValueError(f'Hydrogen mass repartitioning with hmr_mass={hydrogen_mass} leaves atom {atom} '
                         f'(Z={int(numbers[atom])}) with a nonpositive mass')
    return masses


water_indices_of = water_indices


def repartition_hydrogen_mass_inclusive(system):
    """Set every H, including rigid-water H, to 3 amu using its heavy donor.

    Starts from the supplied masses and moves mass from each donor in
    constraint-row order. Heavy masses must remain finite and at least 3 amu;
    this bound is a precaution, not a validated stability threshold.
    repartition_hydrogen_mass (which skips water) is not affected.
    """
    target = 3.0
    _require_hydrogen_constraints(system)
    original = np.asarray(system['masses'])
    if original.dtype.kind not in 'fiu':
        raise ValueError('Inclusive HMR requires real numeric input masses')
    old = original.astype(np.float64).copy()
    numbers = np.asarray(system['atomic_numbers'])
    pairs = np.asarray(system['constraint_pairs']).reshape(-1, 2)
    distances = np.asarray(system['constraint_distances_A']).reshape(-1)
    water_indices = water_indices_of(system)
    if (old.ndim != 1 or not len(old) or numbers.shape != old.shape
            or numbers.dtype.kind not in 'iu' or (numbers <= 0).any()):
        raise ValueError('Inclusive HMR requires one mass and positive integer element number per atom')
    if not np.isfinite(old).all() or (old <= 0).any():
        raise ValueError('Inclusive HMR requires finite positive input masses')
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu'
            or (pairs < 0).any() or (pairs >= len(old)).any()):
        raise ValueError('Inclusive HMR requires integer constraint pairs within the atom range')
    if (distances.shape != (len(pairs),) or distances.dtype.kind not in 'fiu'
            or not np.isfinite(distances).all() or (distances <= 0).any()):
        raise ValueError('Inclusive HMR requires one finite positive distance per constraint pair')
    if (water_indices.dtype.kind not in 'iu' or (water_indices < 0).any()
            or (water_indices >= len(old)).any()):
        raise ValueError('Inclusive HMR requires integer water indices within the atom range')
    water_indices = water_indices.ravel()
    if len(np.unique(water_indices)) != len(water_indices):
        raise ValueError('Inclusive HMR requires distinct water indices')
    water = np.zeros(len(old), dtype=bool)
    water[water_indices] = True
    if not np.all(np.isin(numbers[water], (1, 8))):
        raise ValueError('Inclusive HMR water indices must contain only O and H')

    donors = {}
    for i, j in pairs:
        h, heavy = (int(i), int(j)) if numbers[i] == 1 else (int(j), int(i))
        if numbers[h] != 1 or numbers[heavy] <= 1:
            continue
        if h in donors and donors[h] != heavy:
            raise ValueError('Inclusive HMR requires one unique heavy donor per H')
        if water[h] != water[heavy]:
            raise ValueError('Inclusive HMR donor crosses the water boundary')
        donors[h] = heavy
    hydrogen = np.flatnonzero(numbers == 1)
    if set(donors) != set(hydrogen.tolist()):
        raise ValueError('Inclusive HMR requires a constrained heavy donor for every H')

    # Dictionary first-insertion order fixes the transfer order, so the masses are
    # reproducible bit for bit. Do not group transfers or replace these sequential
    # binary64 subtractions.
    new = old.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        for h, heavy in donors.items():
            delta = target - old[h]
            new[h] = target
            new[heavy] -= delta
    heavy_indices = np.flatnonzero(numbers > 1)
    if len(heavy_indices):
        minimum_index = int(heavy_indices[np.argmin(new[heavy_indices])])
        minimum = float(new[minimum_index])
        if not np.isfinite(new[heavy_indices]).all() or minimum < target:
            raise ValueError(
                'Inclusive HMR requires finite heavy masses >= 3 amu; '
                f'minimum heavy mass {minimum:.17g} amu at atom index {minimum_index} '
                f'(element Z={int(numbers[minimum_index])}). '
                'This bound is not a validated stability threshold.')

    groups = []
    for oxygen in np.flatnonzero(water & (numbers == 8)):
        attached = sorted(h for h, donor in donors.items() if donor == oxygen and water[h])
        if len(attached) != 2:
            raise ValueError('Inclusive HMR requires exactly two water H donors per O')
        groups.append([int(oxygen), *attached])
    groups = np.asarray(groups, dtype=np.int64).reshape(-1, 3)
    if (set(groups.ravel().tolist()) != set(water_indices.tolist())
            or len(groups) * 3 != len(water_indices)):
        raise ValueError('Inclusive HMR requires complete three-atom water groups')
    with np.errstate(over='ignore', invalid='ignore'):
        total_before, total_after = float(old.sum()), float(new.sum())
        water_error = float(np.max(np.abs(new[groups].sum(axis=1) - old[groups].sum(axis=1)), initial=0.))
    if (not np.isfinite(new).all() or not (new > 0).all()
            or not np.all(new[hydrogen] == target)
            or not math.isfinite(total_before) or not math.isfinite(total_after)
            or not np.isclose(total_after, total_before, rtol=0, atol=1e-9)
            or not math.isfinite(water_error) or water_error > 1e-12):
        raise ValueError('Inclusive HMR failed finite positive mass or mass-conservation invariants')
    return new
