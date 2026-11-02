"""Raw-charge bound of the largo latent-charge term, selected by the weights' model_config.

The bound c (the charge-bound entry of model_config) is applied as c*tanh(raw/c)
after the charge head and before FP32 or FP64 centering, so replacing or
compiling the head cannot discard it. It is a fixed model constant, not a
parameter or checkpoint buffer; state-dict names are unchanged.
"""
from __future__ import annotations

import math
import struct
from types import MethodType

import torch
from torch import nn


_BOUND_ATTRIBUTE = "_hyff_latent_charge_bound"


def normalize_charge_bound(value):
    """Return the effective FP32 bound (positive, finite) of a plain model_config number."""
    if type(value) not in (int, float):
        raise ValueError("The charge bound must be a finite positive number")
    try:
        requested = float(value)
    except (OverflowError, ValueError):
        raise ValueError("The charge bound must be a finite positive number") from None
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("The charge bound must be a finite positive number")
    try:
        effective = struct.unpack("!f", struct.pack("!f", requested))[0]
    except (OverflowError, struct.error):
        raise ValueError("The charge bound must remain finite and positive in FP32") from None
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError("The charge bound must remain finite and positive in FP32")
    return effective


def apply_charge_bound(raw, bound):
    """Apply c*tanh(raw/c) in FP32, as in training, with an already normalized bound."""
    if not torch.is_tensor(raw) or raw.dtype != torch.float32:
        raise ValueError("The charge bound requires FP32 raw charges")
    return bound * torch.tanh(raw / bound)


def charge_bound_value(model_or_term):
    """Return the latent-charge term's bound, or None when the model has none."""
    term = getattr(model_or_term, "latent_charge_energy", model_or_term)
    return term.__dict__.get(_BOUND_ATTRIBUTE)


def raw_charges(term, scalars):
    """Call the current head, then apply the model_config bound before centering.

    FP64 conservation and multi-GPU conservation use this same function.
    The absent-bound path keeps the native head/float/squeeze operation order.
    """
    raw = term.charge_head(scalars.float())
    bound = charge_bound_value(term)
    if bound is not None:
        if not torch.is_tensor(raw) or raw.ndim != 2 or raw.shape[1] != 1:
            raise ValueError("The charge bound requires an [N, 1] charge-head output")
        raw = apply_charge_bound(raw, bound)
    return raw.squeeze(-1)


def _conserved_charges_with_bound(self, scalars, atom_batch, total_charge):
    # The LatentChargeEnergy.conserved_charges centering sequence; only the raw
    # charge acquisition uses the shared transform after the installed head.
    raw = raw_charges(self, scalars)
    groups = total_charge.numel()
    sums = raw.new_zeros(groups).index_add(0, atom_batch, raw)
    counts = raw.new_zeros(groups).index_add(0, atom_batch, torch.ones_like(raw))
    shifts = (total_charge.float().reshape(-1) - sums) / counts.clamp_min(1)
    return raw + shifts[atom_batch]


def is_native_bound_conservation(term):
    """Identify only this build-time native override, for later route installers."""
    return (charge_bound_value(term) is not None and
            getattr(term.conserved_charges, "__func__", None) is _conserved_charges_with_bound)


def install_charge_bound(model, value):
    """Install once during model construction, before load/prepare/compile."""
    bound = normalize_charge_bound(value)
    term = getattr(model, "latent_charge_energy", None)
    head = getattr(term, "charge_head", None)
    if _BOUND_ATTRIBUTE in getattr(term, "__dict__", {}):
        raise ValueError("The charge bound is already installed")
    if term is None:
        raise ValueError("The charge bound requires a latent-charge (largo) model")
    if (type(head) is not nn.Linear or head.out_features != 1 or
            head.bias is not None or head.weight.dtype != torch.float32):
        raise ValueError("The charge bound requires the FP32 charge head")
    if head._forward_hooks or head._forward_pre_hooks:
        raise ValueError("Install the charge bound before charge-head observers")
    if "conserved_charges" in term.__dict__:
        raise ValueError("Install the charge bound before conservation overrides")
    term.__dict__[_BOUND_ATTRIBUTE] = bound
    term.conserved_charges = MethodType(_conserved_charges_with_bound, term)
    # Plain metadata only: no extra parameter or persistent buffer is added.
    if isinstance(getattr(model, "latent_charge_spec", None), dict):
        model.latent_charge_spec = dict(
            model.latent_charge_spec,
            charge_bound=bound,
            raw_charge_transform="c*tanh(raw/c), before native charge centering",
        )
    return model


__all__ = [
    "normalize_charge_bound", "apply_charge_bound", "raw_charges",
    "charge_bound_value", "install_charge_bound", "is_native_bound_conservation",
]
