"""Bussi-Donadio-Parrinello stochastic velocity rescaling."""
from __future__ import annotations

import math


def bussi_factor(kinetic, target, ndof, dt_over_tau, rng):
    """Bussi-Donadio-Parrinello CSVR scaling factor for one application."""
    c = math.exp(-dt_over_tau)
    r1 = rng.standard_normal()
    sr = rng.chisquare(ndof - 1) if ndof > 1 else 0.
    ratio = target / (ndof * kinetic)
    alpha2 = c + (1 - c) * ratio * (r1 * r1 + sr) + 2 * r1 * math.sqrt(c * (1 - c) * ratio)
    return math.sqrt(max(alpha2, 0.))
