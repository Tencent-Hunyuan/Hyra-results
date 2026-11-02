"""Setup-only retained-tile storage observations and fixed plan selection.

Counts come from the actual autograd saved tensors; they do not guarantee an
allocator fit. The caller runs the one replay setup force, agrees the result
across ranks and handles fatal errors.
"""
from __future__ import annotations

import math
import torch

from hyff.multigpu.projected_efm import _masked, _require

GIB = 1024 ** 3
RETENTION_SETS = ((0, 1, 2, 3), (1, 2, 3), (1, 2), (1,))


def _storage(value):
    if not value.numel():
        return None, 0
    storage = value.untyped_storage()
    return (value.device.type, value.device.index, storage.data_ptr()), storage.nbytes()


def calibration_headroom(device, *, calibration_headroom_bytes,
                         allocator_ceiling_bytes=64 * GIB, reserve_bytes=16 * GIB):
    _require(all(type(x) is int and x >= 0 for x in
                 (calibration_headroom_bytes, allocator_ceiling_bytes, reserve_bytes)) and
             calibration_headroom_bytes > 0 and allocator_ceiling_bytes > reserve_bytes,
             "EFM_RETENTION_CALIBRATION_BUDGET_INVALID")
    if device.type != "cuda":
        return {"device_type": device.type, "allocator_observed": False}
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free, total = torch.cuda.mem_get_info(device)
    _require(reserve_bytes >= max(16 * GIB, math.ceil(total * 0.2)),
             "EFM_RETENTION_PHYSICAL_RESERVE_INSUFFICIENT")
    _require(max(allocated, reserved) + calibration_headroom_bytes <= allocator_ceiling_bytes and
             free >= calibration_headroom_bytes + reserve_bytes,
             "EFM_RETENTION_CALIBRATION_HEADROOM_INSUFFICIENT")
    return {"device_type": "cuda", "allocator_observed": True,
            "allocated_before_bytes": allocated, "reserved_before_bytes": reserved,
            "physical_free_before_bytes": free, "device_total_bytes": total}


def calibrate_retained_tile(efm, edge_function, native_edges, local, mask, *,
                            check_parity, compiled):
    """Observe one actual full tile. The returned record contains no tensors.

    saved_tensors_hooks sees the actual callable's autograd saves. Shared frozen
    parameters/buffers are excluded by storage identity; tensor views count the
    entire storage once. Pointers are local bookkeeping and never reported.
    """
    excluded = {_storage(value)[0] for value in (*efm.parameters(), *efm.buffers())}
    storage_sizes = {}
    hook_keys = set()
    hook_calls = 0

    def observe(value, *, hook=False):
        key, size = _storage(value)
        if key is not None and key not in excluded:
            storage_sizes[key] = size
            if hook:
                hook_keys.add(key)

    def pack(value):
        nonlocal hook_calls
        hook_calls += 1
        observe(value, hook=True)
        # A packed object must not keep its originating autograd graph alive.
        return value.detach()

    with torch.enable_grad():
        leaves = tuple(value.detach().requires_grad_(True) for value in local)
        for value in leaves:
            observe(value)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            outputs = tuple(_masked(edge, mask) for edge in edge_function(*leaves))
        for value in outputs:
            observe(value)
        _require(hook_calls > 0 and bool(hook_keys), "EFM_RETENTION_CALIBRATION_NO_SAVED_TENSORS")
        channels = (torch.arange(efm.channels, device=leaves[0].device) % 17 + 1).to(leaves[0].dtype) / 17
        cotangents = tuple((channels * (index + 1)).expand(mask.numel(), *tail, efm.channels).contiguous()
                          for index, tail in enumerate(((), (3,), (5,))))
        gradients = torch.autograd.grad(outputs, leaves, cotangents, allow_unused=True,
                                        create_graph=False, retain_graph=True)
        reference_leaves = tuple(value.detach().requires_grad_(True) for value in local)
        reference = tuple(_masked(edge, mask) for edge in native_edges(*reference_leaves))
        forward_rel = check_parity(outputs, reference)
        reference_gradients = torch.autograd.grad(reference, reference_leaves, cotangents,
                                                  allow_unused=True, create_graph=False)
        backward_rel = check_parity(gradients, reference_gradients)
        # Exercise a second VJP on the actual retained graph and a zero exit.
        zeros = tuple(torch.zeros_like(value) for value in cotangents)
        zero_gradients = torch.autograd.grad(outputs, leaves, zeros, allow_unused=True,
                                             create_graph=False, retain_graph=False)
        _require(all(value is None or not bool(value.ne(0).any()) for value in zero_gradients),
                 "EFM_RETENTION_CALIBRATION_ZERO_VJP_FAILED")
    return {"schema": "OWNED_EFM_RETAINED_TILE_CALIBRATION",
            "rows": mask.numel(), "compiled": bool(compiled),
            "dtype": str(local[0].dtype), "device_type": local[0].device.type,
            "input_shapes": [list(value.shape) for value in local],
            "requires_grad_pattern": [True] * 7,
            "unique_retained_bytes": sum(storage_sizes.values()),
            "unique_retained_storages": len(storage_sizes), "saved_hook_calls": hook_calls,
            "unique_hook_storages": len(hook_keys),
            "shared_parameters_buffers_excluded": True,
            "leaves_outputs_included": True, "full_storage_views_deduplicated": True,
            "shared_metadata_conservatively_included": True,
            "forward_relative_error": forward_rel, "backward_relative_error": backward_rel,
            "zero_vjp_passed": True, "graph_returned": False,
            "coverage": "ACTUAL_AUTOGRAD_SAVED_TENSORS_PLUS_LEAVES_OUTPUTS",
            "workspace_included_in_storage_count": False,
            "compiler_rematerialization_eliminated": False}


def choose_retained_plan(calibrations, rows_by_block, *, baseline_peak_bytes,
                         reserve_bytes, workspace_bytes,
                         retention_cap_bytes=16 * GIB, allocator_ceiling_bytes=64 * GIB,
                         physical_free_bytes=None, device_total_bytes=None):
    """Select once in declared order; reject if even block 1 has no budget.

    Baseline peak and workspace/reserve are supplied observations/budgets. The
    formula is deliberately conservative about overlap; it does not guarantee a fit.
    CUDA execution must still enforce the allocator ceiling and fail closed.
    """
    _require(all(type(value) is int and value >= 0 for value in
                 (baseline_peak_bytes, reserve_bytes, workspace_bytes, retention_cap_bytes,
                  allocator_ceiling_bytes)), "EFM_RETENTION_PLAN_BUDGET_INVALID")
    estimates = {}
    for index in (0, 1, 2, 3):
        report, rows = calibrations[index], rows_by_block[index]
        tile = report["rows"]
        _require(type(rows) is int and rows >= 0 and type(tile) is int and tile > 0 and
                 report["saved_hook_calls"] > 0 and report["unique_hook_storages"] > 0 and
                 report["unique_retained_bytes"] > 0 and not report["graph_returned"] and
                 report["leaves_outputs_included"] and report["zero_vjp_passed"],
                 "EFM_RETENTION_PLAN_CALIBRATION_INVALID")
        estimates[index] = math.ceil(rows / tile) * report["unique_retained_bytes"]
    attempts = []
    for blocks in RETENTION_SETS:
        retained_bytes = sum(estimates[index] for index in blocks)
        total = baseline_peak_bytes + retained_bytes + workspace_bytes
        physical_fit = (physical_free_bytes is None or
                        retained_bytes + workspace_bytes + reserve_bytes <= physical_free_bytes)
        if device_total_bytes is not None:
            _require(reserve_bytes >= max(16 * GIB, math.ceil(device_total_bytes * 0.2)),
                     "EFM_RETENTION_PHYSICAL_RESERVE_INSUFFICIENT")
        fits = (retained_bytes <= retention_cap_bytes and total <= allocator_ceiling_bytes and physical_fit)
        attempts.append({"blocks": list(blocks), "retained_estimate_bytes": retained_bytes,
                         "allocator_plan_bytes": total, "fits_budget": fits})
        if fits:
            return {"schema": "OWNED_EFM_RETENTION_PLAN", "blocks": list(blocks),
                    "max_rows_by_block": {str(index): rows_by_block[index] for index in blocks},
                    "retained_estimate_bytes": retained_bytes, "allocator_plan_bytes": total,
                    "retention_cap_bytes": retention_cap_bytes,
                    "allocator_ceiling_bytes": allocator_ceiling_bytes,
                    "baseline_peak_bytes": baseline_peak_bytes, "workspace_bytes": workspace_bytes,
                    "reserve_bytes": reserve_bytes, "attempts": attempts,
                    "physical_free_bytes": physical_free_bytes,
                    "device_total_bytes": device_total_bytes,
                    "selected_once": True, "midcall_fallback": False,
                    "same_shapes_and_compiler_path_required": True,
                    "fit_guaranteed": False}
    raise RuntimeError("EFM_RETENTION_NO_SET_FITS_BUDGET")
