"""The HyPDE network: a fused-history transformer forecaster for 2D PDE fields.

One call maps the last H = 6 normalised frames of a trajectory to the next K = 4 frames, as cumulative increments
over the last input frame:

    per-frame stem (shared over frames; fields + constant inputs, BC-aware padding, validity channel)
    -> concatenate the 6 frame tokens of each patch -> Linear(6 * 384, 1024)                       (fusion)
    -> prefix conditioning tokens (dataset, boundary conditions, fields, stride, grid, 4 registers)
    -> 24 x [full attention (QK-LayerNorm, partial axial RoPE, exactly periodic on periodic axes)
             + depthwise 3x3 token conv + SwiGLU]                                                   (processor)
    -> LayerNorm -> Linear(1024, 4 * 384) -> per-frame head -> full-resolution refiner -> * s_k

Stem, fusion, conditioning, output projection, head and refiner run in fp32 (autocast off); the processor runs under
the caller's autocast with an fp32 residual stream.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AxialRoPE, Block, ConditioningTokens, Geometry, Head, Refiner, Stem, fp32


@dataclass(frozen=True)
class HyPDEConfig:
    d: int = 1024                 # processor width
    depth: int = 24               # processor blocks
    heads: int = 16
    mlp_hidden: int = 2816        # SwiGLU hidden width (2.75 d)
    history: int = 6              # input frames per call
    k_max: int = 4                # output frames per call
    c_stem: int = 384             # stem / head width
    d_frame: int = 384            # per-frame token width before fusion
    max_merge: int = 2            # patches of 4, 8 or 16 per axis
    refiner_hidden: int = 32
    refiner_blocks: int = 2
    refiner_kernel: int = 5
    n_registers: int = 4
    n_datasets: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


class HyPDE(nn.Module):
    def __init__(self, cfg: HyPDEConfig = HyPDEConfig()):
        super().__init__()
        self.cfg = c = cfg
        self.stem = Stem(c.c_stem, c.d_frame, c.max_merge)
        self.head = Head(c.c_stem, c.d_frame, c.max_merge)
        self.refiner = Refiner(c.refiner_hidden, c.refiner_blocks, c.refiner_kernel)
        self.fuse = nn.Linear(c.history * c.d_frame, c.d)
        self.pad_emb = nn.Parameter(torch.zeros(c.d_frame))     # marks left-padded frames when fewer are given
        self.cond = ConditioningTokens(c.d, c.n_datasets, c.n_registers)
        self.blocks = nn.ModuleList([Block(c.d, c.heads, c.mlp_hidden) for _ in range(c.depth)])
        self.out_norm = nn.LayerNorm(c.d)
        self.out_proj = nn.Linear(c.d, c.k_max * c.d_frame)
        self.out_scale = nn.Parameter(torch.ones(c.k_max))     # learned scale of the k-th cumulative increment
        self._rope = AxialRoPE(c.d // c.heads)

    def num_params(self, part: str = "all") -> int:
        mod = self.blocks if part == "processor" else self
        return sum(p.numel() for p in mod.parameters())

    def forward(self, z: torch.Tensor, ids: torch.Tensor, geom: Geometry, dataset_id: int,
                const: torch.Tensor = None, const_ids: torch.Tensor = None, k: int = None) -> torch.Tensor:
        """z (B, H_in, C, X, Y): normalised history, H_in <= 6 (fewer frames are left-padded with the oldest one and
        flagged). ids (C,): vocabulary indices of the fields; const (B, Cc, X, Y) normalised constant inputs with
        const_ids (Cc,). Returns (B, k, C, X, Y): the cumulative increments y_1..y_k of the next k frames over the
        last input frame, in units of the window's increment scale (see `hypde.api`)."""
        c = self.cfg
        B, H_in, C = z.shape[:3]
        grid = tuple(z.shape[3:])
        H, dev = c.history, z.device
        k = c.k_max if k is None else int(k)
        if not (1 <= k <= c.k_max and 1 <= H_in <= H):
            raise ValueError(f"k={k} (at most {c.k_max}), history {H_in} (at most {H})")
        has_const = const is not None and const.shape[1] > 0
        ids_in = torch.cat([ids, const_ids]) if has_const else ids

        with fp32(dev.type):
            x = z.float()
            if H_in < H:
                x = torch.cat([x[:, :1].expand(B, H - H_in, *x.shape[2:]), x], dim=1)
            last = x[:, -1]
            if has_const:
                cst = const.float()
                last = torch.cat([last, cst], dim=1)
                x = torch.cat([x, cst[:, None].expand(B, H, *cst.shape[1:])], dim=2)
            tok = self.stem(x.reshape(B * H, x.shape[2], *grid), ids_in, geom)       # (B*H, nx, ny, d_frame)
            tg = tok.shape[1:-1]
            N = math.prod(tg)
            tok = tok.reshape(B, H, N, c.d_frame)
            if H_in < H:
                flag = torch.zeros(H, device=dev, dtype=tok.dtype)
                flag[:H - H_in] = 1.0
                tok = tok + flag.view(1, H, 1, 1) * self.pad_emb
            h = self.fuse(tok.permute(0, 2, 1, 3).reshape(B, N, H * c.d_frame))       # (B, N, d)
            prefix = self.cond(B, geom, ids_in, dataset_id)
            n_pre = prefix.shape[1]
            h = torch.cat([prefix, h], dim=1).float()
            cos, sin = self._rope.tables(geom.tok_grid, geom.periodic, n_pre, dev)

        for blk in self.blocks:
            h = blk(h, cos, sin, n_pre, tuple(tg), geom.periodic)

        with fp32(dev.type):
            s = h[:, n_pre:].float()
            w = self.out_proj
            o = F.linear(self.out_norm(s), w.weight[:k * c.d_frame], w.bias[:k * c.d_frame])
            o = o.reshape(B, N, k, c.d_frame).transpose(1, 2).reshape(B * k, *tg, c.d_frame)
            y = self.head(o, ids, geom)                                                 # (B*k, C, X, Y)
            lastk = last[:, None].expand(B, k, *last.shape[1:]).reshape(B * k, *last.shape[1:])
            y = self.refiner(y, ids, lastk, ids_in, geom)
            return y.reshape(B, k, C, *grid) * self.out_scale[:k].view(1, k, 1, 1, 1)
