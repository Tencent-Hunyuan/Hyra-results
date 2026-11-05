"""Building blocks of HyPDE: field vocabulary, boundary handling, positions, patch embedding, processor block, head
and refiner.

Conventions: spatial tensors are channels-first (B, C, X, Y) at native resolution; token grids are channels-last
(B, nx, ny, d). Every field (each vector or tensor component separately) and every constant input has an index in
the field vocabulary `NAMES`; the model gathers per-field weights by these indices, so the order is part of the
checkpoint. Index 0 is the validity channel (1 inside the domain, 0 in non-periodic padding) that the stem and the
refiner prepend. Names follow the field names of The Well as Walrus renames them (`velocity_x`, `D_xy`, ...).
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

WALL, OPEN, PERIODIC = 0, 1, 2          # boundary-condition codes of The Well, per axis and side
P0 = 4                                  # base patch size

NAMES = [
    "__valid__",
    "pressure", "density", "energy", "internal_energy", "temperature", "entropy", "electron_fraction",
    "concentration", "buoyancy", "tracer", "height", "A", "B", "c_zz", "speed_of_sound", "mask",
    "zeros_like_density", "log10_density", "log10_temperature", "pressure_re", "pressure_im",
    "velocity_x", "velocity_y", "velocity_z", "velocity_r", "velocity_theta", "velocity_phi", "velocity_log_r",
    "momentum_x", "momentum_y", "momentum_z",
    "magnetic_field_x", "magnetic_field_y", "magnetic_field_z",
    "magnetic_field_log_r", "magnetic_field_theta", "magnetic_field_phi",
    *[f"{t}_{a}{b}" for t in ("D", "E", "C") for a in "xyz" for b in "xyz"],
    *[f"__spare_{i}__" for i in range(8)],
]
VOCAB_SIZE = len(NAMES)
_INDEX = {n: i for i, n in enumerate(NAMES)}


def field_ids(names, device=None) -> torch.Tensor:
    """Vocabulary indices of field names; unknown names raise KeyError."""
    return torch.tensor([_INDEX[n] for n in names], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------------------------------- numerics
@contextlib.contextmanager
def tf32(enabled: bool):
    """Set both TF32 switches (cuDNN and cuBLAS) for the body of the block and restore them afterwards."""
    prev = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev


def fp32(device_type: str):
    """Autocast off: the stem, conditioning, output projection, head and refiner always run in fp32."""
    return torch.autocast(device_type=device_type, enabled=False)


# ---------------------------------------------------------------------------------------------------- geometry
def pad_bc(x: torch.Tensor, pad: int, periodic) -> torch.Tensor:
    """Pad the spatial axes of x (B, C, X, Y) by `pad` on both sides: circularly on periodic axes, with zeros
    elsewhere."""
    for i, per in enumerate(periodic):
        if not pad:
            continue
        ax = 2 + i
        n = x.shape[ax]
        if per:
            if pad > n:
                raise ValueError(f"circular pad {pad} larger than axis size {n}")
            x = torch.cat([x.narrow(ax, n - pad, pad), x, x.narrow(ax, 0, pad)], dim=ax)
        else:
            shape = list(x.shape)
            shape[ax] = pad
            z = x.new_zeros(shape)
            x = torch.cat([z, x, z], dim=ax)
    return x


@dataclass(frozen=True)
class Geometry:
    """Grid, boundary conditions and token plan of one batch (one dataset per batch).

    merges: per-axis number of 2x token merges above the base patch, so the patch on axis i is 4 * 2**merges[i].
    """
    grid: tuple
    bcs: tuple                               # ((lo, hi), (lo, hi)) codes
    merges: tuple
    periodic: tuple = field(init=False)
    tok_grid: tuple = field(init=False)      # tokens seen by the processor

    def __post_init__(self):
        if len(self.grid) != 2 or len(self.bcs) != 2 or len(self.merges) != 2:
            raise ValueError(f"HyPDE is a 2D model: grid {self.grid}, bcs {self.bcs}, merges {self.merges}")
        per = []
        for lo, hi in self.bcs:
            if (lo == PERIODIC) != (hi == PERIODIC):
                raise ValueError(f"inconsistent periodic boundary condition on one axis: {lo}, {hi}")
            per.append(lo == PERIODIC)
        for n, m in zip(self.grid, self.merges):
            if n % (P0 * 2 ** m):
                raise ValueError(f"axis size {n} not divisible by the patch {P0 * 2 ** m}")
        object.__setattr__(self, "periodic", tuple(per))
        object.__setattr__(self, "tok_grid", tuple(n // (P0 * 2 ** m) for n, m in zip(self.grid, self.merges)))


def patch_merges(grid) -> tuple:
    """Token plan: the least anisotropic patch (4, 8 or 16 per axis) giving 1024-2048 tokens, then fewer tokens;
    if none does, the one closest to 1024 tokens."""
    import itertools
    cands = [p for p in itertools.product((4, 8, 16), repeat=2) if all(g % q == 0 for g, q in zip(grid, p))]
    tok = lambda p: (grid[0] // p[0]) * (grid[1] // p[1])
    aniso = lambda p: max(p) / min(p)
    ok = [p for p in cands if 1024 <= tok(p) <= 2048]
    best = min(ok, key=lambda p: (aniso(p), tok(p))) if ok else \
        min(cands, key=lambda p: (abs(math.log(tok(p) / 1024)), aniso(p)))
    return tuple(int(round(math.log2(p // P0))) for p in best)


# ---------------------------------------------------------------------------------------------------- positions
ROPE_FRAC = 0.75     # fraction of the rotation pairs of a head that carry positions; the rest are content-only


class AxialRoPE:
    """Partial axial rotary embedding over prefix + token grid, exactly periodic on periodic axes.

    The first n_rot pairs of each head rotate with the token position, half of them per axis ("half" layout:
    channel j pairs with j + head_dim/2). Frequencies are geometric from pi/2 to pi/256 radians per token; on a
    periodic axis of n tokens each is snapped to 2*pi*m/n, so a circular shift of the grid permutes the
    attention outputs exactly. Prefix rows have their rotated pairs zeroed: conditioning tokens interact with
    the grid through the content pairs only, which keeps translation equivariance exact.
    """

    def __init__(self, head_dim: int, frac: float = ROPE_FRAC):
        self.half = head_dim // 2
        self.n_rot = int(self.half * frac) // 2 * 2
        self._cache: dict = {}

    def _freqs(self, n: int, periodic: bool, f_ax: int) -> torch.Tensor:
        w_max, w_min = math.pi / 2, math.pi / 256
        geo = torch.tensor([w_max * (w_min / w_max) ** (k / max(f_ax - 1, 1)) for k in range(f_ax)],
                           dtype=torch.float64)
        if not periodic:
            return geo
        m = torch.clamp(torch.round(geo * n / (2 * math.pi)), 1, max(n // 2, 1))
        return 2 * math.pi * m / n

    def tables(self, tok_grid, periodic, n_prefix: int, device, dtype=torch.float32):
        key = (tuple(tok_grid), tuple(periodic), n_prefix, str(device), dtype)
        if key in self._cache:
            return self._cache[key]
        f_ax = self.n_rot // 2
        pos = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in tok_grid],
                                         indexing="ij"), -1).reshape(-1, 2)
        ang = torch.zeros(pos.shape[0], self.n_rot, dtype=torch.float64)
        for ax in range(2):
            ang[:, ax * f_ax:(ax + 1) * f_ax] = pos[:, ax:ax + 1] * self._freqs(tok_grid[ax], periodic[ax], f_ax)[None]
        ang = torch.remainder(ang, 2 * math.pi)                      # exact periodicity before the cast
        n = pos.shape[0]
        cos = torch.ones(n_prefix + n, self.half, device=device, dtype=dtype)
        sin = torch.zeros(n_prefix + n, self.half, device=device, dtype=dtype)
        cos[n_prefix:, :self.n_rot] = torch.cos(ang).to(device=device, dtype=dtype)
        sin[n_prefix:, :self.n_rot] = torch.sin(ang).to(device=device, dtype=dtype)
        cos[:n_prefix, :self.n_rot] = 0.0
        self._cache[key] = (cos, sin)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dt = x.dtype
    xf = x.to(cos.dtype)
    h = xf.shape[-1] // 2
    x1, x2 = xf[..., :h], xf[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(dt)


class BoundaryEmbedding(nn.Module):
    """Learned (axis, side, boundary type, distance) vector added to the tokens within 4 tokens of every
    non-periodic boundary. Interior tokens and periodic axes get none, so there is no absolute position."""

    def __init__(self, dim: int, n_buckets: int = 4):
        super().__init__()
        self.n_buckets = n_buckets
        self.emb = nn.Parameter(torch.zeros(3, 2, 3, n_buckets, dim))     # (axis, side, type, distance, dim)

    def forward(self, tok: torch.Tensor, bcs) -> torch.Tensor:
        out = tok
        for ax, (lo, hi) in enumerate(bcs):
            if lo == PERIODIC:
                continue
            n = tok.shape[1 + ax]
            nb = min(self.n_buckets, n)
            add = tok.new_zeros(n, tok.shape[-1])
            add[:nb] = add[:nb] + self.emb[ax, 0, lo, :nb].to(tok.dtype)
            add[n - nb:] = add[n - nb:] + self.emb[ax, 1, hi, :nb].flip(0).to(tok.dtype)
            shape = [1] * tok.dim()
            shape[1 + ax], shape[-1] = n, tok.shape[-1]
            out = out + add.view(shape)
        return out


# ---------------------------------------------------------------------------------------------------- stem / head
_SUBSETS = ((0,), (1,), (0, 1))


def _key(level: int, axes) -> str:
    return f"l{level}_" + "".join(str(a) for a in axes)


def merge_tokens(t: torch.Tensor, axes) -> torch.Tensor:
    """(B, n0, n1, c) -> (B, n0', n1', 2^k c): concatenate the 2 neighbours along every axis in `axes`."""
    B, n0, n1, c = t.shape
    if axes == (0, 1):
        t = t.reshape(B, n0 // 2, 2, n1 // 2, 2, c).permute(0, 1, 3, 2, 4, 5)
        return t.reshape(B, n0 // 2, n1 // 2, 4 * c)
    if axes == (0,):
        return t.reshape(B, n0 // 2, 2, n1, c).permute(0, 1, 3, 2, 4).reshape(B, n0 // 2, n1, 2 * c)
    return t.reshape(B, n0, n1 // 2, 2 * c)


def unmerge_tokens(t: torch.Tensor, axes, c: int) -> torch.Tensor:
    """Inverse of merge_tokens."""
    B, n0, n1, _ = t.shape
    if axes == (0, 1):
        t = t.reshape(B, n0, n1, 2, 2, c).permute(0, 1, 3, 2, 4, 5)
        return t.reshape(B, 2 * n0, 2 * n1, c)
    if axes == (0,):
        return t.reshape(B, n0, n1, 2, c).permute(0, 1, 3, 2, 4).reshape(B, 2 * n0, n1, c)
    return t.reshape(B, n0, 2 * n1, c)


class PatchEmbed(nn.Module):
    """Per-field-type patch embedding: x (B, C, X, Y) -> validity channel prepended -> conv with kernel 8,
    stride 4, weights gathered from a (vocabulary, c, 8*8) table, BC-aware padding of 2 -> LayerNorm.
    Runs with TF32 off: the model input is not centred (see `hypde.api`), and TF32's 10-bit mantissa
    would quantise its mean and lose the fluctuation about it."""

    def __init__(self, c: int, overlap: int = 2):
        super().__init__()
        self.c, self.overlap, self.k = c, overlap, P0 + 2 * overlap
        self.weight = nn.Parameter(torch.zeros(VOCAB_SIZE, c, self.k ** 2))
        self.norm = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor, ids: torch.Tensor, geom: Geometry) -> torch.Tensor:
        B = x.shape[0]
        x = torch.cat([x.new_ones(B, 1, *x.shape[2:]), x], 1)
        ids = torch.cat([ids.new_zeros(1), ids])
        x = pad_bc(x, self.overlap, geom.periodic)
        w = self.weight[ids].view(len(ids), self.c, self.k, self.k).transpose(0, 1)
        with tf32(False):
            t = F.conv2d(x, w.to(x.dtype), stride=P0)
        return self.norm(t.movedim(1, -1))


class Stem(nn.Module):
    """Patch embedding, hierarchical patch merging and boundary embedding: x (B, C, X, Y) -> PatchEmbed
    -> 0, 1 or 2 merge levels along the axes that need them (concat neighbours -> LN -> Linear)
    -> + merge-count embedding -> LN -> Linear(c, d) -> + boundary embedding -> (B, nx, ny, d)."""

    def __init__(self, c: int, d: int, max_merge: int = 2, overlap: int = 2):
        super().__init__()
        self.embed = PatchEmbed(c, overlap)
        self.merge = nn.ModuleDict({_key(l, s): nn.Sequential(nn.LayerNorm(2 ** len(s) * c),
                                                              nn.Linear(2 ** len(s) * c, c))
                                    for l in range(max_merge) for s in _SUBSETS})
        self.merge_emb = nn.Parameter(torch.zeros(2, max_merge + 1, c))
        self.out_norm = nn.LayerNorm(c)
        self.out = nn.Linear(c, d)
        self.bnd = BoundaryEmbedding(d)

    def forward(self, x: torch.Tensor, ids: torch.Tensor, geom: Geometry) -> torch.Tensor:
        t = self.embed(x, ids, geom)
        for l in range(max(geom.merges)):
            axes = tuple(i for i, m in enumerate(geom.merges) if m > l)
            t = self.merge[_key(l, axes)](merge_tokens(t, axes))
        t = self.out(self.out_norm(t + sum(self.merge_emb[i, m] for i, m in enumerate(geom.merges))))
        return self.bnd(t, geom.bcs)


class Head(nn.Module):
    """Processor tokens -> per-field coarse increment at native resolution: LN -> Linear(d, c) -> unmerge levels
    (Linear -> unmerge -> LN -> GELU) -> per-field transposed conv (kernel 8, stride 4, BC-aware) + per-field bias."""

    def __init__(self, c: int, d: int, max_merge: int = 2, overlap: int = 2):
        super().__init__()
        self.c, self.overlap, self.k = c, overlap, P0 + 2 * overlap
        self.norm = nn.LayerNorm(d)
        self.inp = nn.Linear(d, c)
        self.unmerge = nn.ModuleDict({_key(l, s): nn.Linear(c, 2 ** len(s) * c)
                                      for l in range(max_merge) for s in _SUBSETS})
        self.unorm = nn.ModuleDict({_key(l, s): nn.LayerNorm(c) for l in range(max_merge) for s in _SUBSETS})
        self.weight = nn.Parameter(torch.zeros(VOCAB_SIZE, c, self.k ** 2))
        self.bias = nn.Parameter(torch.zeros(VOCAB_SIZE))

    def forward(self, t: torch.Tensor, ids_out: torch.Tensor, geom: Geometry) -> torch.Tensor:
        t = self.inp(self.norm(t))
        for l in reversed(range(max(geom.merges))):
            axes = tuple(i for i, m in enumerate(geom.merges) if m > l)
            k = _key(l, axes)
            t = F.gelu(self.unorm[k](unmerge_tokens(self.unmerge[k](t), axes, self.c)))
        t = pad_bc(t.movedim(-1, 1), 1, geom.periodic)          # one token of padding per side
        w = self.weight[ids_out].view(len(ids_out), self.c, self.k, self.k).transpose(0, 1)
        y = F.conv_transpose2d(t, w.to(t.dtype), self.bias[ids_out].to(t.dtype), stride=P0)
        s = P0 + self.overlap
        return y[:, :, s:s + geom.grid[0], s:s + geom.grid[1]]


class Refiner(nn.Module):
    """Full-resolution refiner: lifts the coarse increment and the current frame (fields, constants, validity)
    with per-field 1x1 embeddings to 32 channels, applies two residual blocks [BC-aware depthwise 5x5 conv
    -> 1x1 -> GELU -> 1x1] and adds a per-field readout to the coarse increment. All of it runs with TF32 off."""

    def __init__(self, hidden: int = 32, n_blocks: int = 2, kernel: int = 5):
        super().__init__()
        self.h, self.kernel = hidden, kernel
        self.lift_y = nn.Parameter(torch.zeros(VOCAB_SIZE, hidden))
        self.lift_x = nn.Parameter(torch.zeros(VOCAB_SIZE, hidden))
        self.dw = nn.ParameterList([nn.Parameter(torch.zeros(hidden, 1, kernel, kernel)) for _ in range(n_blocks)])
        self.dw_b = nn.ParameterList([nn.Parameter(torch.zeros(hidden)) for _ in range(n_blocks)])
        self.pw1 = nn.ModuleList([nn.Linear(hidden, 2 * hidden) for _ in range(n_blocks)])
        self.pw2 = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(n_blocks)])
        self.readout = nn.Parameter(torch.zeros(VOCAB_SIZE, hidden))

    def forward(self, y, ids_out, x, ids_in, geom: Geometry):
        """y: coarse increment (B, C_out, X, Y); x: current frame with constants (B, C_in, X, Y)."""
        B, r = y.shape[0], self.kernel // 2
        ids_v = torch.cat([ids_in.new_zeros(1), ids_in])
        xv = torch.cat([x.new_ones(B, 1, *x.shape[2:]), x], 1)
        with tf32(False):
            h = torch.einsum("bc...,ch->bh...", y, self.lift_y[ids_out].to(y.dtype)) \
                + torch.einsum("bc...,ch->bh...", xv, self.lift_x[ids_v].to(y.dtype))
            h = h / math.sqrt(y.shape[1] + xv.shape[1])
            for i in range(len(self.dw)):
                z = F.conv2d(pad_bc(h, r, geom.periodic), self.dw[i].to(h.dtype), self.dw_b[i].to(h.dtype),
                             groups=self.h).movedim(1, -1)
                z = self.pw2[i](F.gelu(self.pw1[i](z)))
                h = h + z.movedim(-1, 1)
            return y + torch.einsum("bh...,ch->bc...", h, self.readout[ids_out].to(h.dtype))


class ConditioningTokens(nn.Module):
    """Prefix tokens that join the spatial attention: dataset id, one per boundary (axis, side), one per input
    field (field embedding + MLP of three scale statistics, zero for HyPDE's normalisation), the time stride, the
    grid and patch plan, and 4 learned registers."""

    def __init__(self, d: int, n_datasets: int = 32, n_registers: int = 4, max_stride: int = 8):
        super().__init__()
        self.n_datasets = n_datasets
        self.ds = nn.Parameter(torch.zeros(n_datasets + 1, d))
        self.bc_type = nn.Parameter(torch.zeros(3, d))
        self.bc_pos = nn.Parameter(torch.zeros(3, 2, d))
        self.field = nn.Parameter(torch.zeros(VOCAB_SIZE, d))
        self.stats = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.stride = nn.Parameter(torch.zeros(max_stride + 1, d))
        self.grid = nn.Sequential(nn.Linear(9, d), nn.GELU(), nn.Linear(d, d))
        self.reg = nn.Parameter(torch.zeros(n_registers, d))

    def forward(self, B: int, geom: Geometry, ids_in: torch.Tensor, dataset_id: int) -> torch.Tensor:
        dev, dt = self.ds.device, self.ds.dtype
        did = torch.full((B,), int(dataset_id), dtype=torch.long, device=dev)
        toks = [self.ds[did][:, None]]
        bc = torch.stack([self.bc_type[t] + self.bc_pos[a, s] for a, row in enumerate(geom.bcs)
                          for s, t in enumerate(row)])
        toks.append(bc[None].expand(B, -1, -1))
        st = torch.zeros(B, ids_in.shape[0], 3, device=dev, dtype=dt)
        toks.append(self.field[ids_in][None] + self.stats(st))
        toks.append(self.stride[torch.ones(B, dtype=torch.long, device=dev)][:, None])
        g = torch.zeros(9, device=dev, dtype=dt)
        for i, (n, m) in enumerate(zip(geom.grid, geom.merges)):
            g[3 * i:3 * i + 3] = torch.tensor([math.log2(n) / 8, float(m), 1.0], dtype=dt)
        toks.append(self.grid(g)[None, None].expand(B, 1, -1))
        toks.append(self.reg[None].expand(B, -1, -1))
        return torch.cat(toks, 1)


# ---------------------------------------------------------------------------------------------------- processor
def ln_fp32(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast(x.device.type, enabled=False):
        return norm(x.float())


class Attention(nn.Module):
    """Full softmax attention over prefix + grid tokens with QK-LayerNorm and the partial axial RoPE."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm, self.k_norm = nn.LayerNorm(self.dh), nn.LayerNorm(self.dh)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        M, L, d = x.shape
        q, k, v = self.qkv(x).view(M, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q = apply_rope(ln_fp32(self.q_norm, q), cos, sin).to(v.dtype)
        k = apply_rope(ln_fp32(self.k_norm, k), cos, sin).to(v.dtype)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(M, L, d))


class MLP(nn.Module):
    """Depthwise 3x3 conv on the token grid (circular on periodic axes; prefix tokens pass through), then SwiGLU."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.dw = nn.Parameter(torch.zeros(d, 1, 3, 3))
        self.fc1 = nn.Linear(d, 2 * hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x, n_prefix, tok_grid, periodic):
        M = x.shape[0]
        sp = x[:, n_prefix:].reshape(M, *tok_grid, -1).movedim(-1, 1)
        sp = sp + F.conv2d(pad_bc(sp.contiguous(), 1, periodic), self.dw.to(sp.dtype), groups=sp.shape[1])
        x = torch.cat([x[:, :n_prefix], sp.movedim(1, -1).reshape(M, -1, x.shape[-1])], 1)
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)


class Block(nn.Module):
    """Pre-LN transformer block with an fp32 residual stream: x += Attn(LN x); x += MLP(LN x)."""

    def __init__(self, d: int, heads: int, hidden: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.n2 = nn.LayerNorm(d)
        self.mlp = MLP(d, hidden)

    def forward(self, h, cos, sin, n_prefix, tok_grid, periodic):
        h = h + self.attn(ln_fp32(self.n1, h), cos, sin).to(h.dtype)
        return h + self.mlp(ln_fp32(self.n2, h), n_prefix, tok_grid, periodic).to(h.dtype)
