"""Minimal fine-tuning example: write a labelled extended-XYZ file, fine-tune, use the result.

The reference energies and forces here come from another HyFF model (default vivace) so that the example
runs without external data; in practice they come from your electronic-structure calculations.
Needs ase and a CUDA GPU (or --device cpu, slower).
Usage: python finetune_example.py --output DIR [--model presto] [--reference vivace] [--structures 400]
"""
import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

import hyff
from hyff.calculator import HyFFCalculator

MOLECULES = ('CH4', 'NH3', 'CH3OH', 'CH3CHO', 'HCOOH', 'C2H6', 'CH3CN', 'C6H6', 'CH3COOH', 'H2CO')


def water_cluster(count, rng):
    water = molecule('H2O')
    cluster = Atoms()
    while len(cluster) < 3 * count:
        trial = water.copy()
        trial.rotate(rng.uniform(0, 360), rng.normal(size=3), center='COP')
        trial.translate(rng.uniform(-1.6, 1.6, 3) * count ** (1 / 3) * 1.4)
        if len(cluster) == 0 or np.min(np.linalg.norm(cluster.positions[:, None] - trial.positions[None], axis=2)) > 1.7:
            cluster += trial
    return cluster


def structures(count, rng):
    frames = []
    for k in range(count):
        atoms = water_cluster(int(rng.integers(2, 7)), rng) if k % 2 else molecule(MOLECULES[k % len(MOLECULES)])
        atoms.positions += rng.normal(scale=0.08, size=atoms.positions.shape)   # thermal-like distortions
        atoms.info.update(charge=0, multiplicity=1)
        frames.append(atoms)
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='presto')
    parser.add_argument('--reference', default='vivace')
    parser.add_argument('--structures', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    # 1. Labelled structures in extended XYZ (energy in eV, forces in eV/A).
    frames = structures(args.structures, np.random.default_rng(0))
    reference = HyFFCalculator(args.reference, engine='eager', device=args.device)
    for atoms in frames:
        atoms.calc = reference
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    write(root / 'train.extxyz', frames)

    # 2. Fine-tune (5% of the structures are held out for validation).
    summary = hyff.finetune(args.model, train=root / 'train.extxyz', output=root / 'run', epochs=args.epochs,
                            device=args.device)
    start, best = summary['history'][0]['valid'], summary['history'][summary['best_epoch']]['valid']
    print(f'validation force MAE {start["force_mae_meV_A"]:.1f} -> {best["force_mae_meV_A"]:.1f} meV/A, '
          f'energy MAE {start["energy_mae_meV_atom"]:.2f} -> {best["energy_mae_meV_atom"]:.2f} meV/atom')

    # 3. Use the fine-tuned weights like any model.
    atoms = molecule('CH3OH')
    atoms.calc = HyFFCalculator(weights=summary['best'], engine='eager', device=args.device)
    print('fine-tuned energy of methanol:', atoms.get_potential_energy(), 'eV')


if __name__ == '__main__':
    main()
