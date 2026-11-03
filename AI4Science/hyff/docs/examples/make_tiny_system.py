"""Write tiny example inputs (one water molecule): inference.npz, md.npz and collection.npz.

Usage: python make_tiny_system.py --output DIR
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    x = np.array([[0., 0., 0.], [.9572, 0., 0.], [-.2399872, .927297, 0.]], dtype=np.float64)
    z = np.array([8, 1, 1], dtype=np.int64)
    pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    distances = np.linalg.norm(x[pairs[:, 0]] - x[pairs[:, 1]], axis=1)
    np.savez(root / 'inference.npz', positions_A=x.astype(np.float32), atomic_numbers=z)
    np.savez(root / 'collection.npz', positions=x.astype(np.float32), numbers=z,
             offsets=np.array([0, 3], dtype=np.int64), total_charge=np.array([0.]),
             multiplicity=np.array([1], dtype=np.int64))
    np.savez(root / 'md.npz', positions_A=x, atomic_numbers=z, velocities_A_ps=np.zeros_like(x),
             masses=np.array([15.9994, 1.007947, 1.007947]), box_A=np.eye(3) * 30,
             constraint_pairs=pairs, constraint_distances_A=distances,
             water_idx=np.array([[0, 1, 2]], dtype=np.int64), total_charge=np.array(0.),
             multiplicity=np.array(1))
    print('Wrote inference.npz, md.npz and collection.npz to', root)


if __name__ == '__main__':
    main()
