"""Install reserve-based edge growth on one EFM state instance.

The caller must keep the native overflow check and rebuild/recapture loop.
An overflowing force remains rejected. This module imports no model code,
allocates no tensors, and does not change any class or initial-capacity policy.
"""
from __future__ import annotations

import math
import operator
from types import MethodType


POLICY_ID = "efm_observed_edges_reserve_minimum_one_tile"


def _integer(value, name, *, minimum):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer at least {minimum}") from error
    if result < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return result


def install_growth_policy(state, reserve=1.05):
    """Bind a replacement ``grow`` to ``state`` and return a description of the policy.

    Native geometry records ``edge_excess = selected.count - capacity``.
    ``selected.count`` counts live entries in the current slot window; when
    slots also overflow, the next checked rebuild may require another growth.
    The new edge capacity is::

        ceil(max((capacity + edge_excess) * reserve,
                 capacity + tile) / tile) * tile

    Tiles are 1024 edges on CUDA and 16 on CPU, as in the native EFM state.
    Assigning the capacity directly avoids applying ``state.reserve`` a second
    time through the native ``set_capacity``. Initial sizing and its statistics
    are not changed. Slot growth, flag clearing and the growth counter behave
    as in the native ``grow``.
    ``check_overflow`` is untouched. Invalid overflow records fail before any
    capacity, slot, flag, or counter mutation.
    """
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)):
        raise ValueError("reserve must be finite and at least one")
    reserve = float(reserve)
    if not math.isfinite(reserve) or reserve < 1:
        raise ValueError("reserve must be finite and at least one")
    device_type = state.device.type
    if device_type not in ("cpu", "cuda"):
        raise ValueError("only CPU and CUDA capacity tiles are supported")
    tile = 1024 if device_type == "cuda" else 16
    _integer(state.slots, "slots", minimum=1)
    if state.capacity is not None:
        _integer(state.capacity, "capacity", minimum=1)
    _integer(state.stats["buffer_growths"], "buffer_growths", minimum=0)
    if not callable(state.check_overflow) or not callable(state.flags.zero_):
        raise ValueError("native overflow check and clearable flags are required")

    def grow(this, error):
        edge_excess = _integer(error.edge_excess, "edge_excess", minimum=0)
        slot_excess = _integer(error.slot_excess, "slot_excess", minimum=0)
        if not (edge_excess or slot_excess):
            raise ValueError("a positive edge or slot overflow is required")
        capacity = this.capacity
        slots = _integer(this.slots, "slots", minimum=1)
        growths = _integer(this.stats["buffer_growths"], "buffer_growths", minimum=0)
        if edge_excess:
            capacity = _integer(capacity, "capacity", minimum=1)
            requested = max((capacity + edge_excess) * reserve, capacity + tile)
            if not math.isfinite(requested):
                raise ValueError("requested edge capacity must be finite")
            capacity = math.ceil(requested / tile) * tile
        if slot_excess:
            slots = max(slots * 2, slots + slot_excess)
        if edge_excess:
            this.capacity = capacity
        if slot_excess:
            this.slots = slots
        this.flags.zero_()
        this.stats["buffer_growths"] = growths + 1

    policy = dict(
        policy_id=POLICY_ID,
        scope="one_state_instance_grow_only",
        edge_growth_reserve=reserve,
        edge_capacity_tile=tile,
        edge_units="directed_edges",
        edge_count="capacity + edge_excess = live entries in current slot window",
        edge_capacity_formula="ceil(max((capacity+edge_excess)*reserve,capacity+tile)/tile)*tile",
        reserve_applications_per_edge_growth=1,
        slot_growth_formula="max(slots*2,slots+slot_excess)",
        native_overflow_check_preserved=True,
        flags_cleared_after_growth=True,
        buffer_growths_increment_per_call=1,
        initial_capacity_policy_preserved=True,
        initial_edge_statistic_preserved=True,
        caller_must_rebuild_and_recapture=True,
    )
    state.grow = MethodType(grow, state)
    state._efm_growth_policy = dict(policy)
    return policy
