"""Explicit checkpoint family for the experimental symmetric radial potential."""
import torch

from hyff.models.pair_radials import PairRadialConfig, PairRadialPotential


ARCHITECTURE = 'smooth_l2_pair_conditioning'


def load_checkpoint(path, device='cpu'):
    raise ValueError('Checkpoint loading is not available here; use hyff.weights.load_model')
