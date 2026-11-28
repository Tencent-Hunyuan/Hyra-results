"""Pure-PyTorch montage-agnostic spatio-temporal Transformer backbone for HyEEG."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HyEEGConfig:
    """Architectural specification of a HyEEG encoder."""

    embed_dim: int = 512
    depth: int = 8
    heads: int = 8
    head_dim: int = 64
    mlp_dim_ratio: float = 2.66
    use_geglu: bool = True
    freqs: int = 4
    patch_size: int = 200
    patch_overlap: int = 20
    sampling_rate_hz: int = 200
    noise_ratio: float = 0.0025
    norm_eps: float = 1e-6
    model_name: str = "HyEEG"

    @classmethod
    def from_json(cls, path: str | Path) -> "HyEEGConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, geglu: bool = True, eps: float = 1e-6):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim, eps=eps),
            nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False),
            GEGLU() if geglu else nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    """Pre-norm bidirectional multi-head self-attention via PyTorch SDPA."""

    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64, eps: float = 1e-6):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.head_dim = head_dim
        self.norm = RMSNorm(dim, eps=eps)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        return self.to_out(out)


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        head_dim: int,
        mlp_dim: int,
        geglu: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SelfAttention(dim, heads=heads, head_dim=head_dim, eps=eps),
                        FeedForward(dim, mlp_dim, geglu=geglu, eps=eps),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, return_out_layers: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        out_layers = [x] if return_out_layers else None
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            if out_layers is not None:
                out_layers.append(x)
        return out_layers if out_layers is not None else x


class FourierEmb4D(nn.Module):
    """Continuous 4D sinusoidal Fourier embedding for (x, y, z, t) token coordinates."""

    def __init__(
        self,
        dimension: int,
        freqs: int = 4,
        increment_time: float = 0.1,
        margin: float = 0.4,
    ):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin

    def forward(self, positions_: torch.Tensor) -> torch.Tensor:
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        batch_dims = list(positions.shape[:-1])

        freqs_w = torch.arange(self.freqs, device=positions.device, dtype=positions.dtype)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1.0 + 2.0 * self.margin
        positions = positions + self.margin
        p_x = 2.0 * math.pi * freqs_x / width
        p_y = 2.0 * math.pi * freqs_y / width
        p_z = 2.0 * math.pi * freqs_z / width
        p_w = 2.0 * math.pi * freqs_w / width

        positions = positions[..., None, None, None, None, :]
        loc = (
            positions[..., 0] * p_x
            + positions[..., 1] * p_y
            + positions[..., 2] * p_z
            + positions[..., 3] * p_w
        )
        loc = loc.reshape(*batch_dims, -1)

        half_dim = self.dimension // 2
        current_dim = loc.shape[-1]
        if current_dim != half_dim:
            if current_dim > half_dim:
                loc = loc[..., :half_dim]
            else:
                raise ValueError(
                    f"Fourier frequency grid ({current_dim}) is smaller than embed_dim // 2 ({half_dim})."
                )
        return torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)

    @classmethod
    def add_time_patch(cls, pos: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Expand (B, C, 3) 3D electrode coordinates to (B, C * num_patches, 4) channel-major tokens."""
        batch, nchans, _ = pos.shape
        pos_repeated = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)
        time_values = torch.arange(0, num_patches, 1, device=pos.device, dtype=pos.dtype)
        time_values = time_values.view(1, 1, num_patches, 1).expand(batch, nchans, num_patches, 1)
        pos_with_time = torch.cat((pos_repeated, time_values), dim=-1)
        return pos_with_time.reshape(batch, nchans * num_patches, 4)


class HyEEGModel(nn.Module):
    """Montage-agnostic HyEEG encoder with continuous 4D positional encoding and query pooling."""

    def __init__(self, config: HyEEGConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.freqs = config.freqs
        self.patch_size = config.patch_size
        self.overlap_size = config.patch_overlap
        self.noise_ratio = config.noise_ratio

        mlp_dim = int(config.embed_dim * config.mlp_dim_ratio)
        self.transformer = TransformerBackbone(
            dim=config.embed_dim,
            depth=config.depth,
            heads=config.heads,
            head_dim=config.head_dim,
            mlp_dim=mlp_dim,
            geglu=config.use_geglu,
            eps=config.norm_eps,
        )
        self.to_patch_embedding = nn.Sequential(nn.Linear(self.patch_size, self.embed_dim))
        self.fourier4d = FourierEmb4D(self.embed_dim, freqs=self.freqs)
        self.mlp4d = nn.Sequential(
            nn.Linear(4, self.embed_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )
        self.ln = nn.LayerNorm(self.embed_dim)
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def forward(
        self,
        eeg: torch.Tensor,
        pos: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Encode (B, C, T) EEG segments with (B, C, 3) 3D coordinates into (B, C, P, E) tokens.

        If ``return_all_layers`` is True, returns a list of length ``depth + 1`` where index 0 is
        the patch+position embedding and index ``k`` is the output of block ``k``, each shaped
        ``(B, C, P, E)``.
        """
        if eeg.ndim != 3:
            raise ValueError(f"Expected eeg of shape (B, C, T), got {tuple(eeg.shape)}")
        if pos.ndim != 3 or pos.shape[0] != eeg.shape[0] or pos.shape[1] != eeg.shape[1] or pos.shape[2] != 3:
            raise ValueError(
                f"Expected pos of shape ({eeg.shape[0]}, {eeg.shape[1]}, 3), got {tuple(pos.shape)}"
            )
        if eeg.shape[2] < self.patch_size:
            raise ValueError(
                f"Input length T={eeg.shape[2]} is shorter than patch_size={self.patch_size} "
                f"(1.0 s at {self.config.sampling_rate_hz} Hz)."
            )

        param_dtype = self.cls_query_token.dtype
        eeg = eeg.to(dtype=param_dtype)
        pos = pos.to(device=eeg.device, dtype=param_dtype)

        step = self.patch_size - self.overlap_size
        patches = eeg.unfold(dimension=2, size=self.patch_size, step=step)
        b, c, num_patches, _ = patches.shape

        pos4d = FourierEmb4D.add_time_patch(pos, num_patches)
        pos_embed = self.ln(self.fourier4d(pos4d) + self.mlp4d(pos4d))

        x = self.to_patch_embedding(patches).reshape(b, c * num_patches, self.embed_dim) + pos_embed
        out = self.transformer(x, return_out_layers=return_all_layers)

        if return_all_layers:
            assert isinstance(out, list)
            return [h.reshape(b, c, num_patches, self.embed_dim) for h in out]
        assert isinstance(out, torch.Tensor)
        return out.reshape(b, c, num_patches, self.embed_dim)

    def attention_pooling(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool (B, C, P, E) spatio-temporal tokens via the pretrained cross-attention query token."""
        b, c, p, e = tokens.shape
        flat = tokens.reshape(b, c * p, e)
        query = self.cls_query_token.expand(b, -1, -1)
        scores = torch.matmul(query, flat.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, flat).squeeze(1)
