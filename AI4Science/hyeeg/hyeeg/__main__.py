"""Usage: python -m hyeeg eeg.npy --channels Fp1 Fp2 C3 Cz C4 O1 O2 --checkpoint weights/large"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .inference import HyEEGEncoder


def _load_array(path: Path) -> np.ndarray | torch.Tensor:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(obj, torch.Tensor):
            raise ValueError(f"Expected a Tensor in {path}, got {type(obj).__name__}")
        return obj
    raise ValueError(f"Unsupported input format {path.suffix!r}; use .npy or .pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract HyEEG frozen EEG embeddings")
    parser.add_argument("input", type=Path, help="Input .npy or .pt EEG array of shape (C, T) or (B, C, T) at 200 Hz")
    parser.add_argument("--channels", nargs="+", default=None, help="Standard electrode names (e.g. Fp1 Fp2 C3 Cz C4)")
    parser.add_argument("--positions", type=Path, default=None, help="Optional .npy/.pt file of (C, 3) 3D scalp coordinates in meters")
    parser.add_argument("--checkpoint", default="weights/large", help="Checkpoint directory or scale (small, base, large)")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--pooling", choices=["attention", "mean", "tokens"], default="attention")
    parser.add_argument("--layer", type=int, default=None, help="Optional intermediate layer index (0..depth)")
    parser.add_argument("--no-normalize", action="store_true", help="Disable per-window z-normalization")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="embeddings.pt", help="Output .pt file path")
    args = parser.parse_args()

    eeg = _load_array(args.input)
    positions = _load_array(args.positions) if args.positions is not None else None

    encoder = HyEEGEncoder.from_pretrained(args.checkpoint, device=args.device, precision=args.precision)
    vectors = encoder.encode(
        eeg,
        channels=args.channels,
        positions=positions,
        pooling=args.pooling,
        layer=args.layer,
        normalize=not args.no_normalize,
        batch_size=args.batch_size,
    )
    torch.save(vectors, args.output)
    print(f"Saved tensor of shape {tuple(vectors.shape)} to {args.output}")


if __name__ == "__main__":
    main()
