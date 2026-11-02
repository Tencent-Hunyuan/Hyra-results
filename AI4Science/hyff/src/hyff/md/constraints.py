"""SHAKE/RATTLE for disjoint small constraint clusters on unwrapped coordinates.

This is standard constrained integration, not a learned force modification.
Every requested constraint must be represented. Clusters larger than four
atoms or three constraints are rejected explicitly. Independent clusters allow
linear cost and no atomics. FP64 coordinates/velocities avoid losing constraint
accuracy when a molecule moves far from the origin. A system without
constraints (flexible MD) has no clusters; every projection is then a no-op.
"""
import hashlib

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _solve3(a00, a01, a02, a10, a11, a12, a20, a21, a22, b0, b1, b2):
    c00 = a11*a22-a12*a21
    c01 = a02*a21-a01*a22
    c02 = a01*a12-a02*a11
    c10 = a12*a20-a10*a22
    c11 = a00*a22-a02*a20
    c12 = a02*a10-a00*a12
    c20 = a10*a21-a11*a20
    c21 = a01*a20-a00*a21
    c22 = a00*a11-a01*a10
    det = a00*c00+a01*c10+a02*c20
    return ((c00*b0+c01*b1+c02*b2)/det,
            (c10*b0+c11*b1+c12*b2)/det,
            (c20*b0+c21*b1+c22*b2)/det)


@triton.jit
def _project_coupled(Previous, Position, Velocity, InverseMass, Atoms, Left, Right,
                     Distance2, Error, N: tl.constexpr, DT: tl.constexpr,
                     POSITION: tl.constexpr, TOL: tl.constexpr, MAXITER: tl.constexpr,
                     B: tl.constexpr):
    group = tl.program_id(0)*B+tl.arange(0, B)
    slot = tl.arange(0, 4)
    axis = tl.arange(0, 4)
    atom = tl.load(Atoms+group[:, None]*4+slot[None, :], group[:, None] < N, -1)
    valid = (group[:, None, None] < N) & (atom[:, :, None] >= 0) & (axis[None, None, :] < 3)
    address = atom[:, :, None]*3+axis[None, None, :]
    q = tl.load(Position+address, valid, 0.).to(tl.float64)
    before = q
    v = tl.load(Velocity+address, valid, 0.).to(tl.float64)
    old = tl.load(Previous+address, valid, 0.).to(tl.float64)
    w = tl.load(InverseMass+atom, (group[:, None] < N) & (atom >= 0), 0.).to(tl.float64)
    i0 = tl.load(Left+group*3, group < N, -1); j0 = tl.load(Right+group*3, group < N, -1)
    i1 = tl.load(Left+group*3+1, group < N, -1); j1 = tl.load(Right+group*3+1, group < N, -1)
    i2 = tl.load(Left+group*3+2, group < N, -1); j2 = tl.load(Right+group*3+2, group < N, -1)
    s0 = (slot[None, :] == i0[:, None]).to(tl.float64)-(slot[None, :] == j0[:, None]).to(tl.float64)
    s1 = (slot[None, :] == i1[:, None]).to(tl.float64)-(slot[None, :] == j1[:, None]).to(tl.float64)
    s2 = (slot[None, :] == i2[:, None]).to(tl.float64)-(slot[None, :] == j2[:, None]).to(tl.float64)
    d0 = tl.load(Distance2+group*3, group < N, 1.)
    d1 = tl.load(Distance2+group*3+1, group < N, 1.)
    d2 = tl.load(Distance2+group*3+2, group < N, 1.)
    m00 = tl.sum(w*s0*s0, 1); m01 = tl.sum(w*s0*s1, 1); m02 = tl.sum(w*s0*s2, 1)
    m11 = tl.sum(w*s1*s1, 1); m12 = tl.sum(w*s1*s2, 1); m22 = tl.sum(w*s2*s2, 1)
    # The position SHAKE correction remains in the OLD constraint-gradient
    # subspace. Replacing it with the current gradient would change RATTLE.
    if POSITION:
        g0 = tl.sum(old*s0[:, :, None], 1)
        g1 = tl.sum(old*s1[:, :, None], 1)
        g2 = tl.sum(old*s2[:, :, None], 1)
    else:
        g0 = tl.sum(q*s0[:, :, None], 1)
        g1 = tl.sum(q*s1[:, :, None], 1)
        g2 = tl.sum(q*s2[:, :, None], 1)
    error = tl.full((B,), 1., tl.float64)
    iteration = 0
    while (tl.max(error, 0) > TOL) & (iteration < MAXITER):
        r0 = tl.sum(q*s0[:, :, None], 1)
        r1 = tl.sum(q*s1[:, :, None], 1)
        r2 = tl.sum(q*s2[:, :, None], 1)
        if POSITION:
            b0 = tl.where(i0 >= 0, .5*(d0-tl.sum(r0*r0, 1)), 0.)
            b1 = tl.where(i1 >= 0, .5*(d1-tl.sum(r1*r1, 1)), 0.)
            b2 = tl.where(i2 >= 0, .5*(d2-tl.sum(r2*r2, 1)), 0.)
        else:
            b0 = -tl.sum(r0*tl.sum(v*s0[:, :, None], 1), 1)
            b1 = -tl.sum(r1*tl.sum(v*s1[:, :, None], 1), 1)
            b2 = -tl.sum(r2*tl.sum(v*s2[:, :, None], 1), 1)
        a00 = tl.where(i0 >= 0, m00*tl.sum(r0*g0, 1), 1.)
        a11 = tl.where(i1 >= 0, m11*tl.sum(r1*g1, 1), 1.)
        a22 = tl.where(i2 >= 0, m22*tl.sum(r2*g2, 1), 1.)
        a01 = m01*tl.sum(r0*g1, 1); a02 = m02*tl.sum(r0*g2, 1)
        a10 = m01*tl.sum(r1*g0, 1); a12 = m12*tl.sum(r1*g2, 1)
        a20 = m02*tl.sum(r2*g0, 1); a21 = m12*tl.sum(r2*g1, 1)
        l0, l1, l2 = _solve3(a00, a01, a02, a10, a11, a12, a20, a21, a22, b0, b1, b2)
        correction = w[:, :, None]*(s0[:, :, None]*l0[:, None, None]*g0[:, None, :]+
                     s1[:, :, None]*l1[:, None, None]*g1[:, None, :]+
                     s2[:, :, None]*l2[:, None, None]*g2[:, None, :])
        if POSITION:q += correction
        else:v += correction
        r0 = tl.sum(q*s0[:, :, None], 1)
        r1 = tl.sum(q*s1[:, :, None], 1)
        r2 = tl.sum(q*s2[:, :, None], 1)
        if POSITION:
            e0 = tl.abs(tl.sum(r0*r0, 1)/d0-1.)
            e1 = tl.abs(tl.sum(r1*r1, 1)/d1-1.)
            e2 = tl.abs(tl.sum(r2*r2, 1)/d2-1.)
        else:
            e0 = tl.abs(tl.sum(r0*tl.sum(v*s0[:, :, None], 1), 1)/d0)
            e1 = tl.abs(tl.sum(r1*tl.sum(v*s1[:, :, None], 1), 1)/d1)
            e2 = tl.abs(tl.sum(r2*tl.sum(v*s2[:, :, None], 1), 1)/d2)
        error = tl.maximum(tl.where(i0 >= 0, e0, 0.), tl.maximum(
                           tl.where(i1 >= 0, e1, 0.), tl.where(i2 >= 0, e2, 0.)))
        iteration += 1
    if POSITION:
        v += (q-before)/DT
        tl.store(Position+address, q, valid)
    tl.store(Velocity+address, v, valid)
    tl.store(Error+group, error, group < N)


@triton.jit
def _project(Previous, Position, Velocity, InverseMass, Atoms, Left, Right,
             Distance2, Error, N: tl.constexpr, DT: tl.constexpr,
             POSITION: tl.constexpr, TOL: tl.constexpr, MAXITER: tl.constexpr,
             B: tl.constexpr):
    group = tl.program_id(0) * B + tl.arange(0, B)
    slot = tl.arange(0, 4)
    axis = tl.arange(0, 4)
    atom = tl.load(Atoms + group[:, None] * 4 + slot[None, :], group[:, None] < N, -1)
    valid = (group[:, None, None] < N) & (atom[:, :, None] >= 0) & (axis[None, None, :] < 3)
    address = atom[:, :, None] * 3 + axis[None, None, :]
    q = tl.load(Position + address, valid, 0.).to(tl.float64)
    before = q
    v = tl.load(Velocity + address, valid, 0.).to(tl.float64)
    old = tl.load(Previous + address, valid, 0.).to(tl.float64)
    weight = tl.load(InverseMass + atom, (group[:, None] < N) & (atom >= 0), 0.).to(tl.float64)
    error = tl.full((B,), 1., tl.float64)
    iteration = 0
    while (tl.max(error, 0) > TOL) & (iteration < MAXITER):
        for c in tl.static_range(3):
            left = tl.load(Left + group * 3 + c, group < N, -1)
            right = tl.load(Right + group * 3 + c, group < N, -1)
            target = tl.load(Distance2 + group * 3 + c, group < N, 1.)
            active = (group < N) & (left >= 0)
            sign = ((slot[None, :] == left[:, None]).to(tl.float64) -
                    (slot[None, :] == right[:, None]).to(tl.float64))
            r = tl.sum(q * sign[:, :, None], 1)
            mass_sum = tl.sum(weight * tl.abs(sign), 1)
            if POSITION:
                direction = tl.sum(old * sign[:, :, None], 1)
                residual = tl.sum(r * r, 1) - target
                denominator = 2. * mass_sum * tl.sum(r * direction, 1)
                multiplier = tl.where(active, -residual / denominator, 0.)
                q += multiplier[:, None, None] * weight[:, :, None] * sign[:, :, None] * direction[:, None, :]
            else:
                relative_v = tl.sum(v * sign[:, :, None], 1)
                residual = tl.sum(r * relative_v, 1)
                denominator = mass_sum * tl.sum(r * r, 1)
                multiplier = tl.where(active, -residual / denominator, 0.)
                v += multiplier[:, None, None] * weight[:, :, None] * sign[:, :, None] * r[:, None, :]
        error = tl.full((B,), 0., tl.float64)
        for c in tl.static_range(3):
            left = tl.load(Left + group * 3 + c, group < N, -1)
            right = tl.load(Right + group * 3 + c, group < N, -1)
            target = tl.load(Distance2 + group * 3 + c, group < N, 1.)
            active = (group < N) & (left >= 0)
            sign = ((slot[None, :] == left[:, None]).to(tl.float64) -
                    (slot[None, :] == right[:, None]).to(tl.float64))
            r = tl.sum(q * sign[:, :, None], 1)
            if POSITION:
                relative_error = tl.abs(tl.sum(r * r, 1) / target - 1.)
            else:
                relative_v = tl.sum(v * sign[:, :, None], 1)
                relative_error = tl.abs(tl.sum(r * relative_v, 1) / target)
            error = tl.maximum(error, tl.where(active, relative_error, 0.))
        iteration += 1
    if POSITION:
        v += (q - before) / DT
        tl.store(Position + address, q, valid)
    tl.store(Velocity + address, v, valid)
    tl.store(Error + group, error, group < N)


class RattleConstraints:
    def __init__(self, mass, pairs, distances, tolerance=1.e-10, max_iterations=100,
                 source=None, check=True, solver='sweeps'):
        if not mass.is_cuda:
            raise ValueError('RattleConstraints requires a CUDA device')
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        n = len(mass)
        if (distances.shape != (len(pairs),) or
            not np.isfinite(distances).all() or (distances <= 0).any() or
            (pairs < 0).any() or (pairs >= n).any() or (pairs[:, 0] == pairs[:, 1]).any()):
            raise ValueError('Invalid bond constraints')
        if len(pairs) and len(np.unique(np.sort(pairs, axis=1), axis=0)) != len(pairs):
            raise ValueError('Duplicate constraint')
        if not 0 < tolerance < 1.e-3 or max_iterations < 1:
            raise ValueError('Invalid constraint solver settings')
        if solver not in ('sweeps', 'coupled'):
            raise ValueError('Unknown constraint solver')
        parent = np.arange(n)
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return int(i)
        for i, j in pairs:
            parent[root(i)] = root(j)
        groups = {}
        for c, (i, _) in enumerate(pairs):
            groups.setdefault(root(i), []).append(c)
        atoms = np.full((len(groups), 4), -1, dtype=np.int32)
        left = np.full((len(groups), 3), -1, dtype=np.int32)
        right = left.copy()
        d2 = np.ones((len(groups), 3), dtype=np.float64)
        for row, constraints in enumerate(groups.values()):
            nodes = np.unique(pairs[constraints])
            if len(nodes) > 4 or len(constraints) > 3:
                raise ValueError('Unsupported constraint cluster; no constraints were dropped')
            atoms[row, :len(nodes)] = nodes
            left[row, :len(constraints)] = np.searchsorted(nodes, pairs[constraints, 0])
            right[row, :len(constraints)] = np.searchsorted(nodes, pairs[constraints, 1])
            d2[row, :len(constraints)] = distances[constraints] ** 2
        self.inverse_mass = mass.double().reciprocal().contiguous()
        if not bool(torch.isfinite(self.inverse_mass).all()) or not bool((mass > 0).all()):
            raise ValueError('Masses must be finite and positive')
        self.atoms, self.left, self.right = [torch.as_tensor(a, device=mass.device) for a in (atoms, left, right)]
        self.distance2 = torch.as_tensor(d2, device=mass.device)
        # One zero residual slot without constraints keeps the device checks' buffers nonempty.
        self.error = (torch.empty(len(groups), dtype=torch.float64, device=mass.device) if groups else
                      torch.zeros(1, dtype=torch.float64, device=mass.device))
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.check = check
        self.solver = solver
        self.n_atoms = n
        self.description = {
            'algorithm': 'SHAKE/RATTLE on disjoint clusters',
            'solver': solver,
            'constraints': len(pairs), 'clusters': len(groups),
            'coordinate_precision': 'FP64 unwrapped positions and velocities',
            'position_squared_distance_relative_tolerance': tolerance,
            'velocity_dot_over_distance_squared_tolerance_per_fs': tolerance,
            'max_iterations': max_iterations, 'checked_every_projection': check,
            'pairs_sha256': hashlib.sha256(pairs.tobytes()).hexdigest(),
            'distances_sha256': hashlib.sha256(distances.tobytes()).hexdigest(),
            'source': source,
        }

    @property
    def empty(self):
        """True for flexible MD: no constraint clusters, every projection is a no-op."""
        return len(self.atoms) == 0

    def _run(self, previous, position, velocity, dt, project_positions):
        for value in (previous, position, velocity):
            if (value.shape != (self.n_atoms, 3) or value.dtype != torch.float64 or
                value.device != self.inverse_mass.device or not value.is_contiguous()):
                raise ValueError('Constraint state must be contiguous FP64 on the constraint device')
        if dt <= 0:
            raise ValueError('Positive timestep required')
        if self.empty:
            return
        kernel = _project_coupled if self.solver == 'coupled' else _project
        kernel[(triton.cdiv(len(self.atoms), 32),)](
            previous, position, velocity, self.inverse_mass, self.atoms, self.left,
            self.right, self.distance2, self.error, len(self.atoms), float(dt),
            project_positions, self.tolerance, self.max_iterations, 32,
            num_warps=4, enable_fp_fusion=False)
        if self.check:
            maximum = float(self.error.max())
            if not np.isfinite(maximum) or maximum > self.tolerance:
                raise FloatingPointError(f'Constraint projection failed: residual={maximum}')

    def project_positions(self, previous, position, velocity, dt):
        self._run(previous, position, velocity, dt, True)

    def project_velocities(self, position, velocity):
        self._run(position, position, velocity, 1., False)
