"""Gaussian smooth-particle-mesh Ewald for the latent-charge term (experimental periodic option).

Periodic definition (tin-foil boundary, neutralizing background through the same kernel):

    E = C*scale * [ 1/2 sum_{i,j} sum'_n q_i q_j K(|r_i - r_j + n|)  minus the k=0 term ]
    K(r) = erf(r/(sqrt2 sigma))/r,   K^(k) = 4 pi exp(-sigma^2 k^2/2)/k^2,   C = 14.399645

' drops only i = j at n = 0, so intramolecular pairs enter exactly as in the native isolated sum
(which is sum_{i<j} q_i q_j K(r_ij), all pairs, no exclusions). Charges and sigma are the native
module's (conserved_charges, softplus sigma). Dropping the whole k=0 term of K^ is the same as a
uniform background -Q/V interacting through K; a background through bare 1/r would differ by the
position-independent constant pi Q^2 sigma^2 / V.

Ewald split at any Gaussian width s > 0 (split=None ties s to sigma: no real-space part at all):

    real   1/2 sum' q_i q_j [erfc(r/sqrt2 s) - erfc(r/sqrt2 sigma)]/r
    recip  (2 pi/V) sum_{k!=0} exp(-s^2 k^2/2)/k^2 |S(k)|^2      SPME: B-splines order p, FFT
    self   -sum_i q_i^2 / (s sqrt(2 pi))
    bg     -pi Q^2 (s^2 - sigma^2) / V

Isolated limit (cubic box L, one molecule, xi = 2.837297):
    E_periodic = E_isolated - xi Q^2/(2L) + 2pi/(3V) (Q sum q_i r_i^2 - |M|^2) + pi Q^2 sigma^2/V + O(L^-5)

Everything is plain torch (autograd forces, any dtype/device); the mesh part has static shapes and
no plan, so it is CUDA-graph capturable and invariant to lattice translations of any atom.
The real-space part (any fixed split, even one equal to sigma, since it carries dE/dsigma) is a
brute-force image sum for CPU reference work only.

Mesh size: K_i = smallest 2^a3^b5^c >= max(order, |a_i|/spacing) points along lattice vector a_i, so
the B-spline support always fits the mesh and the spacing never exceeds the requested one.

Reciprocal underflow: when every nonzero-k Gaussian factor exp(-2 pi^2 s^2 m^2) is below 1e-45 (for
the largo checkpoint, sigma = 73 A, in every cell shorter than about 32 A along its shortest reciprocal
direction), the mesh sum is exactly zero in FP32 and is not computed; the energy is then the self term
(+ background), which is the exact periodic value to a relative accuracy of 1e-45.
"""
from __future__ import annotations

import itertools
import math

import torch

COUPLING = 14.399645
MADELUNG_SC = 2.837297479480620
SQRT2PI = math.sqrt(2 * math.pi)


def fft_size(n):
    """Smallest 2^a 3^b 5^c >= n."""
    n = max(int(n), 1)
    while True:
        m = n
        for f in (2, 3, 5):
            while m % f == 0:
                m //= f
        if m == 1:
            return n
        n += 1


def screened(r2, width):
    """erf(r/(sqrt2 w))/r with the native module's small-r series branch (any dtype)."""
    x = r2 / (2 * width * width)
    series = (math.sqrt(2 / math.pi) / width) * (1 - x / 3 + x * x / 10 - x.pow(3) / 42)
    u = x.clamp_min(1e-3).sqrt()
    regular = torch.erf(u) / (math.sqrt(2) * width * u)
    return torch.where(x < 1e-3, series, regular)


def split_real(r2, sigma, s):
    """[erfc(r/sqrt2 s) - erfc(r/sqrt2 sigma)]/r, i.e. K_sigma(r) - K_s(r); any s, sigma > 0."""
    sigma, s = (torch.as_tensor(v, dtype=r2.dtype, device=r2.device) for v in (sigma, s))
    w = torch.minimum(sigma, s)                                             # series valid for both terms
    xg, xs, xw = r2 / (2 * sigma * sigma), r2 / (2 * s * s), r2 / (2 * w * w)
    poly = lambda x: 1 - x / 3 + x * x / 10 - x.pow(3) / 42  # noqa: E731
    series = math.sqrt(2 / math.pi) * (poly(xg) / sigma - poly(xs) / s)
    r = torch.maximum(r2, 2e-3 * w * w).sqrt()
    regular = (torch.erfc(r / (math.sqrt(2) * s)) - torch.erfc(r / (math.sqrt(2) * sigma))) / r
    return torch.where(xw < 1e-3, series, regular)


def bspline(t, order):
    """Cardinal B-spline M_p(t + j), j = 0..p-1, for t in [0, 1); shape t.shape + (p,)."""
    j = torch.arange(order, dtype=t.dtype, device=t.device)
    x = t[..., None] + j
    w = (j == 0).to(t.dtype).expand(x.shape)
    for n in range(2, order + 1):
        shifted = torch.cat([torch.zeros_like(w[..., :1]), w[..., :-1]], dim=-1)
        w = (x * w + (n - x) * shifted) / (n - 1)
    return w


def bspline_moduli(size, order, dtype=torch.float64):
    """|b(m)|^2 of Essmann et al. (1995) eq. 4.4 for m = 0..K-1 (zeros at Nyquist interpolated)."""
    c = bspline(torch.zeros((), dtype=torch.float64), order)[1:]            # M_p(k+1), k = 0..p-2
    m = torch.arange(size, dtype=torch.float64)
    arg = 2 * math.pi * m[:, None] * torch.arange(order - 1, dtype=torch.float64) / size
    d2 = (c * arg.cos()).sum(-1).square() + (c * arg.sin()).sum(-1).square()
    b2 = 1 / d2.clamp_min(1e-300)
    bad = d2 < 1e-7
    if bad.any():
        idx = bad.nonzero().flatten()
        b2[idx] = 0.5 * (b2[(idx - 1) % size] + b2[(idx + 1) % size])
    return b2.to(dtype)


class GaussianSPME:
    """Mesh (reciprocal) part for one cell (``set_cell`` changes it); energies in e^2/A (multiply by C*scale).

    cell: rows are lattice vectors. split: Gaussian width s of the reciprocal kernel (None: s =
    sigma, read per call so dE/dsigma is exact). spacing: target mesh spacing (A) along each
    lattice vector; mesh = smallest 2^a3^b5^c with length/K <= spacing. order: B-spline order p.
    """

    def __init__(self, cell, spacing, order=6, split=None, dtype=torch.float64, device="cpu", mesh=None):
        self.dtype, self.device, self.order, self.split = dtype, torch.device(device), int(order), split
        cell64 = torch.as_tensor(cell, dtype=torch.float64).reshape(3, 3).cpu()
        self.cell64 = cell64
        self.volume = float(torch.linalg.det(cell64).abs())
        self.frac64 = torch.linalg.inv(cell64)                                 # r @ frac = fractional
        lengths = cell64.norm(dim=1)
        if self.order < 2:
            raise ValueError("SPME_ORDER_AT_LEAST_2")
        if mesh is None and not (math.isfinite(spacing) and spacing > 0):
            raise ValueError("SPME_POSITIVE_SPACING_REQUIRED")
        self.mesh = tuple(int(k) for k in mesh) if mesh is not None else tuple(
            fft_size(max(self.order, math.ceil(float(a) / spacing))) for a in lengths)
        if min(self.mesh) < self.order:
            raise ValueError("SPME_MESH_SMALLER_THAN_ORDER")
        self.skip_reciprocal = False
        m2, pref = self._tables()
        self.m2_min = float(m2[m2 > 0].min())                                  # shortest nonzero k (A^-2)
        self.m2 = m2.to(dtype=dtype, device=self.device)
        self.pref = pref.to(dtype=dtype, device=self.device)
        self.frac = self.frac64.to(device=self.device)
        self.offsets = torch.arange(order, device=self.device)
        self.k_tensor = torch.tensor(self.mesh, device=self.device)

    def _tables(self):
        k1, k2, k3 = self.mesh
        f1 = torch.fft.fftfreq(k1, 1 / k1, dtype=torch.float64)
        f2 = torch.fft.fftfreq(k2, 1 / k2, dtype=torch.float64)
        f3 = torch.fft.rfftfreq(k3, 1 / k3, dtype=torch.float64)
        grid = torch.stack(torch.meshgrid(f1, f2, f3, indexing="ij"), -1)
        mvec = grid @ self.frac64.T                                             # m1 b1 + m2 b2 + m3 b3
        m2 = mvec.square().sum(-1)
        b = (bspline_moduli(k1, self.order)[:, None, None] * bspline_moduli(k2, self.order)[None, :, None]
             * bspline_moduli(k3, self.order)[None, None, :k3 // 2 + 1])
        half = torch.full((k3 // 2 + 1,), 2.0, dtype=torch.float64)
        half[0] = 1.0
        if k3 % 2 == 0:
            half[-1] = 1.0
        # recip = 1/2 sum_m pref(m) exp(-2 pi^2 s^2 m^2) |F(Q)(m)|^2, pref(0) = 0
        pref = torch.where(m2 > 0, b * half / (math.pi * self.volume * m2.clamp_min(1e-300)), torch.zeros_like(m2))
        return m2, pref

    SKIP_KEEP_EXPONENT = 30 * math.log(10)   # an omitted mesh sum stays omitted while every factor is < 1e-30

    def set_cell(self, cell, sigma):
        """Same mesh, new cell (constant-pressure MD): the reciprocal tables are rewritten in place.

        A captured evaluation keeps its decision to omit or compute the mesh sum. An omitted sum stays
        omitted while every nonzero-k factor is below 1e-30; otherwise this raises.
        """
        if self.split is not None:
            raise ValueError("SPME_FIXED_SPLIT_KEEPS_ITS_CELL")   # its background term holds the volume
        cell64 = torch.as_tensor(cell, dtype=torch.float64).reshape(3, 3).cpu()
        if not bool(torch.isfinite(cell64).all()) or float(torch.linalg.det(cell64).abs()) <= 0:
            raise ValueError("SPME_NONSINGULAR_CELL_REQUIRED")
        self.cell64, self.volume = cell64, float(torch.linalg.det(cell64).abs())
        self.frac64 = torch.linalg.inv(cell64)
        m2, pref = self._tables()
        self.m2_min = float(m2[m2 > 0].min())
        s = float(sigma) if self.split is None else float(self.split)
        if self.skip_reciprocal and not 2 * math.pi ** 2 * s * s * self.m2_min > self.SKIP_KEEP_EXPONENT:
            raise ValueError("SPME_RECIPROCAL_REGIME_CHANGED: the cell grew until the omitted mesh sum matters; "
                             "build a new engine")
        self.m2.copy_(m2)
        self.pref.copy_(pref)
        self.frac.copy_(self.frac64)

    def width(self, sigma):
        return sigma if self.split is None else torch.as_tensor(self.split, dtype=sigma.dtype, device=sigma.device)

    UNDERFLOW_EXPONENT = 45 * math.log(10)   # exp(-x) < 1e-45: zero in FP32

    def reciprocal_negligible(self, sigma):
        """True when every nonzero-k Gaussian factor exp(-2 pi^2 s^2 m^2) is below 1e-45 (host value of sigma)."""
        s = float(sigma) if self.split is None else float(self.split)
        return 2 * math.pi ** 2 * s * s * self.m2_min > self.UNDERFLOW_EXPONENT

    def set_sigma(self, sigma):
        """Decide (host side, outside capture) whether the mesh sum is computed for this sigma."""
        self.skip_reciprocal = self.reciprocal_negligible(sigma)
        return self.skip_reciprocal

    def spread(self, q, positions):
        """Charge mesh Q (flat) from FP64 fractional coordinates; differentiable in q and positions."""
        p = self.order
        f = positions.to(torch.float64) @ self.frac                            # [N, 3], FP64 always
        u = (f - torch.floor(f)) * self.k_tensor.to(torch.float64)
        base = torch.floor(u)
        t = (u - base).to(self.dtype)
        w = bspline(t, p)                                                       # [N, 3, p]
        idx = (base.long()[..., None] - self.offsets) % self.k_tensor[:, None]  # [N, 3, p]
        k1, k2, k3 = self.mesh
        flat = (idx[:, 0, :, None, None] * k2 + idx[:, 1, None, :, None]) * k3 + idx[:, 2, None, None, :]
        val = q.to(self.dtype)[:, None, None, None] * (
            w[:, 0, :, None, None] * w[:, 1, None, :, None] * w[:, 2, None, None, :])
        mesh = torch.zeros(k1 * k2 * k3, dtype=self.dtype, device=self.device)
        return mesh.index_add(0, flat.reshape(-1), val.reshape(-1))

    def reciprocal(self, q, positions, sigma):
        if self.skip_reciprocal:
            return torch.zeros((), dtype=self.dtype, device=q.device)
        s = self.width(sigma).to(self.dtype)
        mesh = self.spread(q, positions).view(self.mesh)
        F = torch.fft.rfftn(mesh)
        green = self.pref * torch.exp(-2 * math.pi ** 2 * s * s * self.m2)
        return 0.5 * (green * (F.real.square() + F.imag.square())).sum()

    def energy(self, q, positions, sigma, real_space=None):
        """Reduced periodic energy (e^2/A) and its parts.

        real_space(q, x, sigma, s) is required whenever split is set -- also at split == sigma, where
        the real part is zero but its sigma derivative is not -- and must be differentiable in q, x
        and sigma (real_space_images is).
        """
        q = q.to(self.dtype)
        sigma = sigma.to(self.dtype) if torch.is_tensor(sigma) else torch.tensor(sigma, dtype=self.dtype, device=q.device)
        s = self.width(sigma)
        parts = dict(reciprocal=self.reciprocal(q, positions, sigma),
                     self=-(q.square().sum()) / (s * SQRT2PI),
                     background=-math.pi * q.sum().square() * (s * s - sigma * sigma) / self.volume)
        if self.split is not None:
            if real_space is None:
                raise ValueError("SPME_REAL_SPACE_REQUIRED_FOR_FIXED_SPLIT")
            parts["real"] = real_space(q, positions, sigma, s)
        return sum(parts.values()), parts

    def describe(self):
        return dict(method="Gaussian SPME (tin-foil, kernel-consistent neutralizing background)",
                    mesh=list(self.mesh), order=self.order,
                    reciprocal=("omitted: every nonzero-k Gaussian factor < 1e-45 (zero in FP32)"
                                if self.skip_reciprocal else "B-spline mesh and FFT"),
                    split="sigma (reciprocal only)" if self.split is None else float(self.split),
                    dtype=str(self.dtype), volume_A3=self.volume)


# ---------------------------------------------------------------- FP64 references (CPU, brute force)

def wrap(positions, cell):
    cell = torch.as_tensor(cell, dtype=positions.dtype)
    f = positions @ torch.linalg.inv(cell)
    return (f - torch.floor(f)) @ cell


class _Attach(torch.autograd.Function):
    """An off-tape value and its saved first derivatives as a graph-connected scalar. First order only:
    backward under create_graph raises (force training would otherwise silently drop this term's
    second derivatives)."""

    @staticmethod
    def forward(ctx, value, grads, *inputs):
        ctx.grads, ctx.dtypes = grads, [t.dtype for t in inputs]
        return value.clone()

    @staticmethod
    def backward(ctx, grad_out):
        if torch.is_grad_enabled():
            raise RuntimeError("SPME_REFERENCE_FIRST_ORDER_ONLY")
        return (None, None) + tuple((grad_out * g).to(d) for g, d in zip(ctx.grads, ctx.dtypes))


class _Accumulate:
    """Chunked energy with exact first derivatives, so FP64 references never hold a full tape.

    x, q, sigma: the caller's inputs (sigma may be a float); value_x: the coordinates actually used
    (wrapped, d value_x/dx = 1). Each chunk energy is differentiated at once w.r.t. detached leaves and
    the gradients summed. result() attaches the energy to the caller's graph when x, q or sigma
    requires grad (forces, dE/dq, dE/dsigma exact; create_graph raises), and with grad=True also
    returns dE/dx.
    """

    def __init__(self, x, q, sigma=None, grad=False, value_x=None):
        self.inputs = [x, q, sigma]
        on = torch.is_grad_enabled()
        self.track = [torch.is_tensor(t) and t.requires_grad and on for t in self.inputs]
        self.track[0] = self.track[0] or grad
        values = [(x if value_x is None else value_x), q, sigma]
        self.x, self.q, self.sigma = (v.detach().requires_grad_(k) if torch.is_tensor(v) else v
                                      for v, k in zip(values, self.track))
        self.leaves = [self.x, self.q, self.sigma]
        self.grad = grad
        self.e = values[0].new_zeros(())
        self.g = [torch.zeros_like(v) if k else None for v, k in zip(self.leaves, self.track)]

    def add(self, e):
        idx = [i for i, k in enumerate(self.track) if k]
        if idx:
            for i, gi in zip(idx, torch.autograd.grad(e, [self.leaves[i] for i in idx], allow_unused=True)):
                if gi is not None:
                    self.g[i] += gi
        self.e += e.detach()

    def result(self):
        on = [i for i, t in enumerate(self.inputs) if self.track[i] and t.requires_grad]
        e = _Attach.apply(self.e, [self.g[i] for i in on], *[self.inputs[i] for i in on]) if on else self.e
        return (e, self.g[0]) if self.grad else e


def isolated(q, positions, sigma, block=512, grad=False):
    """Native definition: sum_{i<j} q_i q_j K_sigma(r_ij) (no cell)."""
    acc = _Accumulate(positions, q, sigma, grad)
    x, q, sigma, n = acc.x, acc.q, acc.sigma, len(q)
    cols = torch.arange(n)
    for b in range(0, n, block):
        rows = cols[b:b + block]
        r2 = (x[rows, None] - x[None]).square().sum(-1)
        mask = rows[:, None] < cols[None]
        acc.add((q[rows, None] * q[None] * screened(r2, sigma) * mask).sum())
    return acc.result()


def lattice_images(cell, reach):
    """Integer image vectors n with |n . a| possibly < reach for wrapped coordinates."""
    cell = torch.as_tensor(cell, dtype=torch.float64)
    v = abs(float(torch.linalg.det(cell)))
    heights = [v / float(torch.linalg.cross(cell[(a + 1) % 3], cell[(a + 2) % 3]).norm()) for a in range(3)]
    nmax = [math.ceil(reach / h) + 1 for h in heights]
    ints = torch.tensor(list(itertools.product(*[range(-k, k + 1) for k in nmax])), dtype=torch.float64)
    return ints @ cell, ints


def real_space_images(q, positions, cell, sigma, s, tol=1e-17, block=16, grad=False):
    """1/2 sum_{i,j} sum'_n q_i q_j [K_sigma - K_s](|r_ij + n|), cutoff where both erfc < tol."""
    rcut = math.sqrt(2) * max(float(s), float(sigma)) * _erfcinv(tol)
    acc = _Accumulate(positions, q, sigma, grad, value_x=wrap(positions.detach(), cell))
    x, q, sigma = acc.x, acc.q, acc.sigma
    shifts, ints = lattice_images(cell, rcut)
    shifts = shifts.to(x.dtype)
    zero = (ints.abs().sum(-1) == 0)
    n = len(q)
    cols = torch.arange(n)
    for b in range(0, n, block):
        rows = cols[b:b + block]
        d = x[rows, None, None] - x[None, :, None] + shifts[None, None]           # [B, N, images, 3]
        r2 = d.square().sum(-1)
        keep = (r2 < rcut * rcut) & ~((rows[:, None, None] == cols[None, :, None]) & zero[None, None])
        g = split_real(torch.where(keep, r2, torch.ones_like(r2)), sigma, s) * keep
        acc.add(0.5 * (q[rows, None] * q[None] * g.sum(-1)).sum())
    return acc.result()


def ewald_reciprocal_direct(q, positions, cell, s, tol=1e-18, chunk=2048, grad=False):
    """(2 pi/V) sum_{k!=0} exp(-s^2 k^2/2)/k^2 |S(k)|^2 by explicit structure factors (half space)."""
    cell = torch.as_tensor(cell, dtype=torch.float64)
    frac = torch.linalg.inv(cell)
    volume = abs(float(torch.linalg.det(cell)))
    mcut = math.sqrt(math.log(1 / tol) / (2 * math.pi ** 2)) / float(s)
    nmax = [math.ceil(mcut * float(a)) for a in cell.norm(dim=1)]
    r = [torch.arange(-k, k + 1, dtype=torch.float64) for k in nmax]
    ints = torch.stack(torch.meshgrid(*r, indexing="ij"), -1).reshape(-1, 3)
    upper = (ints[:, 0] > 0) | ((ints[:, 0] == 0) & (ints[:, 1] > 0)) | (
        (ints[:, 0] == 0) & (ints[:, 1] == 0) & (ints[:, 2] > 0))
    m = ints[upper] @ frac.T
    m2 = m.square().sum(-1)
    m, m2 = m[m2 <= mcut * mcut], m2[m2 <= mcut * mcut]
    weight = torch.exp(-2 * math.pi ** 2 * float(s) ** 2 * m2) / (math.pi * volume * m2)
    acc = _Accumulate(positions, q, None, grad, value_x=wrap(positions.detach(), cell))
    x, q = acc.x, acc.q
    for b in range(0, len(m), chunk):
        phase = 2 * math.pi * (x @ m[b:b + chunk].T)                              # [N, chunk]
        c, sn = q @ torch.cos(phase), q @ torch.sin(phase)
        acc.add((weight[b:b + chunk] * (c * c + sn * sn)).sum())
    return acc.result(), dict(k_vectors_half=len(m), m_cut=mcut)


def ewald_direct(q, positions, cell, sigma, s, tol=1e-18, grad=False):
    """Full FP64 reference with explicit k-sum (and image real space when s != sigma).

    Returns (energy, parts, info) and, with grad=True, dE/dx as a fourth item.
    """
    cell = torch.as_tensor(cell, dtype=torch.float64)
    volume = abs(float(torch.linalg.det(cell)))
    out, info = ewald_reciprocal_direct(q, positions, cell, s, tol, grad=grad)
    rec, g = out if grad else (out, None)
    parts = dict(reciprocal=rec, self=-(q.square().sum()) / (s * SQRT2PI),
                 background=-math.pi * q.sum().square() * (s * s - sigma * sigma) / volume)
    if float(s) != float(sigma):
        out = real_space_images(q, positions, cell, sigma, s, grad=grad)
        parts["real"], g = (out[0], g + out[1]) if grad else (out, None)
    energy = sum(parts.values())
    return (energy, parts, info, g) if grad else (energy, parts, info)


def isolated_limit_prediction(q, positions, sigma, length):
    """Cubic box: E_iso - xi Q^2/(2L) + 2pi/(3V)(Q sum q r^2 - M^2) + pi Q^2 sigma^2/V (error O(L^-5))."""
    v = length ** 3
    qt = q.sum()
    mvec = (q[:, None] * positions).sum(0)
    second = (q * positions.square().sum(-1)).sum()
    e_iso = isolated(q, positions, sigma)
    return dict(isolated=e_iso, madelung=-MADELUNG_SC * qt * qt / (2 * length),
                moment=2 * math.pi / (3 * v) * (qt * second - mvec.square().sum()),
                gaussian_background=math.pi * qt * qt * sigma * sigma / v)


def _erfcinv(y):
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if math.erfc(mid) > y else (lo, mid)
    return 0.5 * (lo + hi)


def periodic_forward(term, mesh, scalars, positions, atom_batch, total_charge, keep_dtype=False):
    """Drop-in for LatentChargeEnergy.forward on one graph: native charges/sigma/scale, SPME energy.

    Returns the positions' dtype like the native module (FP32), or the mesh dtype with keep_dtype.
    """
    if total_charge.numel() != 1:
        raise ValueError("SPME_SINGLE_GRAPH_ONLY")
    with torch.autocast(device_type=positions.device.type, enabled=False):
        q = term.conserved_charges(scalars, atom_batch.long(), total_charge)
        energy, _ = mesh.energy(q, positions, term.sigma)
        energy = term.coupling_ev_angstrom * term.scale.to(energy.dtype) * energy
        return (energy if keep_dtype else energy.to(positions.dtype)).reshape(1)
