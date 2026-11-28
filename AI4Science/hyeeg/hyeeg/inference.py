"""High-level frozen-encoder inference API for HyEEG-Small, HyEEG-Base, and HyEEG-Large."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from safetensors.torch import load_file
import torch

from .model import HyEEGConfig, HyEEGModel
from .positions import resolve_positions

SCALES = ("small", "base", "large")


def window_normalize(
    eeg: torch.Tensor | np.ndarray,
    rel_floor: float = 1e-3,
    clip: float = 15.0,
) -> torch.Tensor:
    """Apply HyEEG's per-window, per-channel z-normalization with relative std floor and clipping.

    Matches the evaluation normalization protocol: computes per-channel mean and population std
    in float64 over each trial window, floors channel std at ``rel_floor * median_channel_std``
    (protecting near-flat leads), and clips standardized values to ``[-clip, clip]``.

    Args:
        eeg: Array or tensor of shape ``(C, T)`` or ``(B, C, T)``.
        rel_floor: Floor relative to the window's median channel standard deviation.
        clip: Symmetric amplitude clip threshold after standardization.

    Returns:
        Float32 CPU tensor with the same shape as ``eeg``.
    """
    arr = eeg.detach().cpu().numpy() if isinstance(eeg, torch.Tensor) else np.asarray(eeg)
    single = arr.ndim == 2
    if single:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected EEG array of shape (C, T) or (B, C, T), got {arr.shape}")

    out = np.empty(arr.shape, dtype=np.float32)
    for i in range(arr.shape[0]):
        e = arr[i].astype(np.float64, copy=False)  # (C, T)
        mean = e.mean(axis=1, keepdims=True)
        std = e.std(axis=1, keepdims=True)
        med = float(np.median(std))
        std = np.maximum(std, rel_floor * med) if med > 0.0 else np.ones_like(std)
        std[std == 0.0] = 1.0
        z = ((e - mean) / std).astype(np.float32)
        if clip is not None:
            np.clip(z, -clip, clip, out=z)
        out[i] = z

    tensor = torch.from_numpy(out)
    return tensor[0] if single else tensor


def _resolve_checkpoint_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if path.is_dir():
        return path
    name = str(checkpoint).lower()
    if name in SCALES:
        candidate = Path(__file__).resolve().parent.parent / "weights" / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Checkpoint directory not found: {checkpoint!r}. "
        "Pass a directory containing config.json and model.safetensors (e.g. 'weights/large'), "
        "or run `bash weights/download.sh` first."
    )


class HyEEGEncoder:
    """Frozen-feature interface for HyEEG-Small, HyEEG-Base, and HyEEG-Large.

    Inputs are multi-channel EEG segments resampled to 200 Hz (shape ``[C, T]`` or ``[B, C, T]``)
    together with either standard channel names (``channels=['Fp1', 'C3', ...]``) or explicit 3D
    scalp coordinates in meters (``positions`` of shape ``[C, 3]`` or ``[B, C, 3]``).
    """

    def __init__(self, model: HyEEGModel, device: torch.device, dtype: torch.dtype):
        self.model = model
        self.config = model.config
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "weights/large",
        device: str | torch.device = "auto",
        precision: str = "fp32",
    ) -> "HyEEGEncoder":
        """Load a HyEEG checkpoint directory containing ``config.json`` and ``model.safetensors``.

        Args:
            checkpoint: Path to a checkpoint directory (e.g. ``"weights/small"``,
                ``"weights/base"``, ``"weights/large"``) or shorthand scale name
                (``"small"``, ``"base"``, ``"large"``).
            device: ``"auto"`` (uses CUDA when available, else CPU), ``"cuda"``, ``"cuda:N"``,
                or ``"cpu"``.
            precision: ``"fp32"`` (default, exact evaluation parity) or ``"bf16"``.
        """
        if precision not in {"fp32", "bf16"}:
            raise ValueError(f"precision must be 'fp32' or 'bf16', got {precision!r}")
        if device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(device)

        dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        ckpt_dir = _resolve_checkpoint_dir(checkpoint)
        config_path = ckpt_dir / "config.json"
        weights_path = ckpt_dir / "model.safetensors"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config.json in {ckpt_dir}")
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Missing model.safetensors in {ckpt_dir}. "
                "Download weights with `bash weights/download.sh`."
            )

        config = HyEEGConfig.from_json(config_path)
        model = HyEEGModel(config)
        state = load_file(str(weights_path), device="cpu")
        model.load_state_dict(state, strict=True)
        model.to(device=torch_device, dtype=dtype)
        model.eval()
        return cls(model=model, device=torch_device, dtype=dtype)

    @torch.inference_mode()
    def encode(
        self,
        eeg: torch.Tensor | np.ndarray,
        *,
        channels: Sequence[str] | None = None,
        positions: torch.Tensor | np.ndarray | None = None,
        pooling: str = "attention",
        layer: int | None = None,
        normalize: bool = True,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Extract EEG embeddings as a CPU FP32 tensor.

        Args:
            eeg: Multi-channel EEG sampled at 200 Hz, shape ``(C, T)`` or ``(B, C, T)``.
            channels: Sequence of ``C`` standard scalp electrode names (e.g. ``['Fp1', 'Cz', ...]``).
                Ignored if ``positions`` is provided explicitly.
            positions: Explicit 3D electrode coordinates in meters, shape ``(C, 3)`` or ``(B, C, 3)``.
            pooling:
                - ``"attention"`` (default): pretrained query-token readout, shape ``(B, 512)``
                  (or ``(512,)`` for a single ``(C, T)`` trial).
                - ``"mean"``: mean over all channel-time patches, shape ``(B, 512)``.
                - ``"tokens"``: unpooled spatio-temporal tokens, shape ``(B, C, P, 512)``.
            layer: Optional intermediate layer index in ``0..depth`` (0 is patch+position embedding,
                ``depth`` is the final Transformer block).
            normalize: If True (default), applies per-window, per-channel z-normalization with
                relative std floor and +/-15 clipping before encoding.
            batch_size: Mini-batch size for batched forward passes.

        Returns:
            CPU float32 tensor of trial embeddings or spatio-temporal tokens.
        """
        if pooling not in {"attention", "mean", "tokens"}:
            raise ValueError("pooling must be 'attention', 'mean', or 'tokens'")
        if layer is not None and not (0 <= layer <= self.config.depth):
            raise ValueError(f"layer must be between 0 and {self.config.depth}, got {layer}")
        if channels is None and positions is None:
            raise ValueError("Provide either `channels` (list of electrode names) or `positions` ((C, 3) coords)")

        x = torch.as_tensor(eeg, dtype=torch.float32)
        single = x.ndim == 2
        if single:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected eeg of shape (C, T) or (B, C, T), got {tuple(x.shape)}")
        b, c, _ = x.shape

        if positions is not None:
            pos = torch.as_tensor(positions, dtype=torch.float32)
            if pos.ndim == 2:
                pos = pos.unsqueeze(0).expand(b, -1, -1)
            if pos.shape != (b, c, 3):
                raise ValueError(f"Expected positions of shape ({b}, {c}, 3), got {tuple(pos.shape)}")
        else:
            assert channels is not None
            if len(channels) != c:
                raise ValueError(f"len(channels)={len(channels)} does not match C={c}")
            pos = resolve_positions(channels).unsqueeze(0).expand(b, -1, -1)

        if normalize:
            x = window_normalize(x)

        outputs: list[torch.Tensor] = []
        for start in range(0, b, batch_size):
            xb = x[start : start + batch_size].to(device=self.device, dtype=self.dtype)
            pb = pos[start : start + batch_size].to(device=self.device, dtype=self.dtype)

            if layer is None:
                tokens = self.model(xb, pb, return_all_layers=False)
                assert isinstance(tokens, torch.Tensor)
            else:
                all_layers = self.model(xb, pb, return_all_layers=True)
                assert isinstance(all_layers, list)
                tokens = all_layers[layer]

            if pooling == "attention":
                out = self.model.attention_pooling(tokens)
            elif pooling == "mean":
                out = tokens.mean(dim=(1, 2))
            else:
                out = tokens

            outputs.append(out.float().cpu())

        result = torch.cat(outputs, dim=0)
        return result[0] if single else result
