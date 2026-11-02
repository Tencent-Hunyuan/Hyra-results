"""Pure capacity arithmetic; no torch import, allocation, or device queries.

The baseline peak is allocated bytes, never allocated plus reserved. The free
baseline sample is physical device free *after* the completed replay baseline
and its peak. At a later preflight, reserved-minus-allocated is cache already
owned by this allocator. Credit it only against the baseline's remaining
transient demand, and cap availability by the baseline physical-free sample.
Neither empty retention nor reserve shortfall rejects an established replay.
"""
from __future__ import annotations

POLICY_TOKEN = "capacity_retention_v1_baseline_observed_donation_off"
ALL_SUBSETS = tuple(tuple(i for i in range(4) if mask & (1 << i)) for mask in range(16))


def _require(ok, code):
    if not ok:
        raise ValueError(code)


def _integers(values):
    return all(type(value) is int and value >= 0 for value in values)


def memory_budget(*, baseline_peak_allocated_bytes, baseline_free_bytes,
                  allocated_bytes, reserved_bytes, physical_free_bytes,
                  device_total_bytes, growth_reserve_bytes, workspace_reserve_bytes):
    values = locals().copy()
    _require(_integers(values.values()) and device_total_bytes > 0 and
             allocated_bytes <= reserved_bytes <= device_total_bytes and
             physical_free_bytes + reserved_bytes <= device_total_bytes and
             baseline_free_bytes <= device_total_bytes and
             baseline_peak_allocated_bytes <= device_total_bytes,
             "EFM_CAPACITY_MEMORY_OBSERVATION_INVALID")
    cache = reserved_bytes - allocated_bytes
    transient = max(0, baseline_peak_allocated_bytes - allocated_bytes)
    available = max(0, min(baseline_free_bytes, physical_free_bytes + cache - transient))
    budget = max(0, available - growth_reserve_bytes - workspace_reserve_bytes)
    return dict(values, schema="EFM_CAPACITY_MEMORY_BUDGET",
                reclaimable_allocator_cache_bytes=cache,
                baseline_transient_bytes=transient,
                available_after_baseline_bytes=available,
                retention_budget_bytes=budget, calibration_allowed=budget > 0,
                budget_reason="HEADROOM_AVAILABLE" if budget else "REPLAY_RESERVE_SHORTFALL",
                baseline_peak_kind="ALLOCATED", fixed_retention_cap_bytes=None,
                fixed_allocator_ceiling_bytes=None)


def choose_capacity_plan(rank_observations, *, policy_token=POLICY_TOKEN):
    """Evaluate every subset and maximize declared replay-work units.

    Each rank supplies four capacities, logical_rows, tile_edges,
    capacity_generations, retained_bytes_per_tile (None means unobserved), and
    replay_work_units_per_tile. Capacity is the reserved row envelope, not the
    current logical count. The same subset must fit every rank. Ties prefer
    fewer retained bytes and then a deterministic block mask. No measurements
    are synthesized for an unobserved nonempty rank/block.
    """
    _require(isinstance(rank_observations, (list, tuple)) and rank_observations and
             isinstance(policy_token, str) and bool(policy_token), "EFM_CAPACITY_RANKS_INVALID")
    ranks, tokens = [], []
    for rank, observation in enumerate(rank_observations):
        required = ("capacities", "logical_rows", "tile_edges", "capacity_generations",
                    "retained_bytes_per_tile", "replay_work_units_per_tile")
        _require(all(isinstance(observation.get(key), (list, tuple)) and
                     len(observation[key]) == 4 for key in required), "EFM_CAPACITY_BLOCKS_INVALID")
        capacities, logical, tiles, generations, saved, work = (list(observation[key]) for key in required)
        _require(_integers((*capacities, *logical, *tiles, *generations, *work,
                           observation.get("epoch"), observation.get("generation"))) and
                 all(tile > 0 and cap % tile == 0 and rows <= cap and rows % tile == 0
                     for cap, rows, tile in zip(capacities, logical, tiles)) and
                 all(value is None or type(value) is int and value > 0 for value in saved),
                 "EFM_CAPACITY_OBSERVATION_INVALID")
        memory = observation["memory"]
        keys = ("baseline_peak_allocated_bytes", "baseline_free_bytes", "allocated_bytes", "reserved_bytes",
                "physical_free_bytes", "device_total_bytes", "growth_reserve_bytes", "workspace_reserve_bytes")
        checked_memory = memory_budget(**{key: memory[key] for key in keys})
        _require(memory == checked_memory, "EFM_CAPACITY_BUDGET_OBSERVATION_CHANGED")
        budget = memory["retention_budget_bytes"]
        _require(_integers((budget,)), "EFM_CAPACITY_BUDGET_INVALID")
        estimates = [0 if cap == 0 else None if count is None else
                     (cap // tile) * count for cap, tile, count in zip(capacities, tiles, saved)]
        scores = [(rows // tile) * cost for rows, tile, cost in zip(logical, tiles, work)]
        ranks.append(dict(rank=rank, estimates_by_block=estimates, scores_by_block=scores,
                          memory=dict(observation["memory"]), capacities=capacities,
                          logical_rows=logical, retained_bytes_per_tile=saved,
                          tile_edges=tiles, capacity_generations=generations,
                          replay_work_units_per_tile=work))
        tokens.append(dict(epoch=observation["epoch"], generation=observation["generation"],
                           capacities=capacities, capacity_generations=generations,
                           policy_token=policy_token))
    _require(len({(token["epoch"], token["generation"]) for token in tokens}) == 1,
             "EFM_CAPACITY_EPOCH_DISAGREEMENT")
    attempts = []
    for mask, blocks in enumerate(ALL_SUBSETS):
        totals, reasons = [], []
        for rank in ranks:
            estimates = [rank["estimates_by_block"][i] for i in blocks]
            total = None if any(value is None for value in estimates) else sum(estimates)
            totals.append(total)
            reasons.append("OBSERVATION_ABSENT" if total is None else
                           "INSUFFICIENT_HEADROOM" if total > rank["memory"]["retention_budget_bytes"]
                           else "FIT")
        attempts.append(dict(mask=mask, blocks=list(blocks), retained_bytes_by_rank=totals,
                             reasons_by_rank=reasons, fits_budget=all(reason == "FIT" for reason in reasons),
                             replay_work_score=sum(sum(rank["scores_by_block"][i] for i in blocks)
                                                   for rank in ranks)))
    feasible = [attempt for attempt in attempts if attempt["fits_budget"]]
    selected = max(feasible, key=lambda attempt: (attempt["replay_work_score"],
                   -max(attempt["retained_bytes_by_rank"]), -len(attempt["blocks"]), -attempt["mask"]))
    return dict(schema="EFM_CAPACITY_RETENTION_PLAN", blocks=selected["blocks"],
                mask=selected["mask"], attempts=attempts, ranks=ranks, tokens=tokens,
                policy_token=policy_token, replay_work_score=selected["replay_work_score"],
                selection_objective="MAX_SUM_LOGICAL_TILES_X_DECLARED_EDGE_WORK_UNITS",
                selected_reason="RETAINED_SUBSET" if selected["blocks"] else "REPLAY_NO_RETAINED_BENEFIT_FITS",
                fixed_retention_cap_bytes=None, fixed_allocator_ceiling_bytes=None,
                all_ranks_agreed=False, planning_only=True, midforce_fallback=False, fit_guaranteed=False)


def validate_plan_token(token, *, epoch, generation, capacities, logical_rows,
                        policy_token=POLICY_TOKEN, capacity_generations=None):
    """Logical growth inside the bound envelope is valid; envelope/epoch change is not."""
    _require(_integers((epoch, generation)) and isinstance(token, dict) and token.get("epoch") == epoch and
             token.get("generation") == generation and token.get("policy_token") == policy_token and
             list(capacities) == token.get("capacities") and
             (capacity_generations is None or list(capacity_generations) == token.get("capacity_generations")) and
             len(logical_rows) == len(capacities) == 4 and _integers((*capacities, *logical_rows)) and
             all(rows <= cap for rows, cap in zip(logical_rows, capacities)),
             "EFM_CAPACITY_PLAN_STALE")
    return True
