"""Stable fixed-shape edge selection; no data-dependent output allocation.

Inactive slots point to the sentinel at ``mask.numel()``. Consumers must append
one inert record before gathering. ``overflow`` stays on-device and MUST be
checked before a force is returned; a truncated graph is never a valid result.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Selection:
    indices: torch.Tensor
    valid: torch.Tensor
    count: torch.Tensor
    overflow: torch.Tensor


def compact(mask: torch.Tensor, capacity: int) -> Selection:
    """Select true indices in input order into exactly ``capacity`` slots.

    All scatter destinations are distinct, including for inactive inputs. This
    avoids a many-writer sink and gives a deterministic stable partition. Only
    tensor metadata controls Python branches, so the same operations can be
    captured and replayed when membership changes.
    """
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError('mask must be a one-dimensional bool tensor')
    if type(capacity) is not int or capacity < 0:
        raise ValueError('capacity must be a nonnegative integer')
    size = mask.numel()
    prefix = mask.to(torch.int64).cumsum(0)
    count = mask.sum(dtype=torch.int64)
    original = torch.arange(size, device=mask.device, dtype=torch.int64)
    destinations = torch.where(mask, prefix - 1, count + original - prefix)
    order = torch.full((max(size, capacity),), size, device=mask.device, dtype=torch.int64)
    order.scatter_(0, destinations, original)
    valid = torch.arange(capacity, device=mask.device) < count
    indices = torch.where(valid, order[:capacity], size)
    return Selection(indices, valid, count, count > capacity)


def gather_with_sentinel(values: torch.Tensor, selection: Selection) -> torch.Tensor:
    """Gather selected rows and zero-filled inactive slots with safe gradients."""
    sentinel = values.new_zeros((1, *values.shape[1:]))
    return torch.cat((values, sentinel), dim=0).index_select(0, selection.indices)
