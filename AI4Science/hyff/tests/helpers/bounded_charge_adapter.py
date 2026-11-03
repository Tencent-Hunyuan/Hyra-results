"""Independent reference implementation of the latent-charge raw-charge bound, as a forward hook.

It applies c*tanh(raw/c) to the charge-head output before the unchanged FP32
charge centering. The package installs the same bound inside the model; the
tests compare both bitwise. No learned parameters or buffers are added.
"""
from __future__ import annotations

import math
import weakref
from typing import Callable

import torch
from torch import nn


_SENTINEL = "_test_bounded_charge_adapter"


def _effective_c(c: float) -> float:
    if isinstance(c, bool) or not isinstance(c, (int, float)):
        raise ValueError("CHARGE_BOUND_FINITE_POSITIVE_SCALAR_REQUIRED")
    try:
        requested = float(c)
    except (OverflowError, ValueError):
        raise ValueError("CHARGE_BOUND_FINITE_POSITIVE_SCALAR_REQUIRED") from None
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("CHARGE_BOUND_FINITE_POSITIVE_SCALAR_REQUIRED")
    effective = float(torch.tensor(requested, dtype=torch.float32).item())
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError("CHARGE_BOUND_POSITIVE_FINITE_FP32_REQUIRED")
    return effective


def bound_raw_charges(raw: torch.Tensor, *, c: float) -> torch.Tensor:
    """Differentiable FP32 c*tanh(r/c), with no charge centering here."""
    effective = _effective_c(c)
    if not torch.is_tensor(raw) or raw.dtype != torch.float32:
        raise ValueError("CHARGE_BOUND_FP32_RAW_REQUIRED")
    return effective * torch.tanh(raw / effective)


class ChargeBoundHandle:
    """Close removes only this adapter. It never retains a feature graph."""

    def __init__(self, head: nn.Linear, c: float, observer: Callable | None):
        self.requested_c = float(c)
        self.c = _effective_c(c)
        self.closed = False
        self._head = weakref.ref(head)
        self._observer = observer
        self._hook = head.register_forward_hook(self._apply)
        head.__dict__[_SENTINEL] = self

    def _apply(self, head, _inputs, raw):
        if self.closed or head.__dict__.get(_SENTINEL) is not self:
            raise RuntimeError("CHARGE_BOUND_HANDLE_NOT_ACTIVE")
        # Observers appended later must be read-only and return None. A hook
        # prepended before this transform, or any head pre-hook, fails closed.
        if (not head._forward_hooks or
                next(iter(head._forward_hooks)) != self._hook.id or
                head._forward_pre_hooks):
            raise RuntimeError("CHARGE_BOUND_HOOK_ORDER_CHANGED")
        if (not torch.is_tensor(raw) or raw.dtype != torch.float32 or
                raw.ndim != 2 or raw.shape[1] != 1):
            raise ValueError("CHARGE_BOUND_NATIVE_HEAD_OUTPUT_REQUIRED")
        bounded = self.c * torch.tanh(raw / self.c)
        if self._observer is not None:
            raw_version, bound_version = raw._version, bounded._version
            try:
                returned = self._observer(raw, bounded)
            except Exception:
                raise RuntimeError("CHARGE_BOUND_OBSERVER_FAILED") from None
            if (returned is not None or raw._version != raw_version or
                    bounded._version != bound_version):
                raise RuntimeError("CHARGE_BOUND_OBSERVER_MUST_BE_READ_ONLY")
        return bounded

    def close(self) -> None:
        if self.closed:
            return
        self._hook.remove()
        head = self._head()
        if head is not None and head.__dict__.get(_SENTINEL) is self:
            del head.__dict__[_SENTINEL]
        self._observer = None
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError("CHARGE_BOUND_HANDLE_CLOSED")
        return self

    def __exit__(self, *_args):
        self.close()


def install_charge_bound_hook(model: nn.Module, *, c: float,
                                 observer: Callable | None = None) -> ChargeBoundHandle:
    """Bound model.latent_charge_energy.charge_head output before centering.

    Caller admits the native model source/checkpoint separately. Existing
    forward transforms are rejected. Later diagnostic hooks must be reviewed
    read-only hooks, installed after this adapter and closed before close().
    """
    _effective_c(c)
    if observer is not None and not callable(observer):
        raise ValueError("CHARGE_BOUND_CALLABLE_OBSERVER_REQUIRED")
    term = getattr(model, "latent_charge_energy", None)
    head = getattr(term, "charge_head", None)
    if (type(head) is not nn.Linear or head.out_features != 1 or
            head.bias is not None or head.weight.dtype != torch.float32):
        raise ValueError("CHARGE_BOUND_NATIVE_FP32_HEAD_REQUIRED")
    if _SENTINEL in head.__dict__:
        raise ValueError("CHARGE_BOUND_ALREADY_INSTALLED")
    if head._forward_hooks or head._forward_pre_hooks:
        raise ValueError("CHARGE_BOUND_INSTALL_BEFORE_HEAD_OBSERVERS")
    return ChargeBoundHandle(head, c, observer)
