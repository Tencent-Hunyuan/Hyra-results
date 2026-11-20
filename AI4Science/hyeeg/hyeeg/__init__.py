"""HyEEG: montage-agnostic self-supervised EEG foundation models."""

from .inference import HyEEGEncoder, window_normalize
from .model import HyEEGConfig, HyEEGModel
from .positions import list_standard_channels, resolve_positions

__all__ = [
    "HyEEGConfig",
    "HyEEGEncoder",
    "HyEEGModel",
    "list_standard_channels",
    "resolve_positions",
    "window_normalize",
]

__version__ = "1.0.0"
