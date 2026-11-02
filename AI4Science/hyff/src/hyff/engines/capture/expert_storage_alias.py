"""Inference-only option: share expert parameter and packed-cache storage.

Call immediately after bind_base has installed FrozenElementExpertMLP, before
EFM/node compilation, autograd evaluation or CUDA graph capture. No arithmetic
or kernel changes are made. The existing Parameter objects become contiguous views
of their already-identical packed execution buffers, including both biases.

This is unsuitable for training or optimizers. Parameter and packed-buffer
version counters do not become one mutation counter; all weights must remain
frozen. Run the whole-model force/capture checks after installation.
"""

from __future__ import annotations

import torch
from torch import nn


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr()


def _storage_summary(tensors):
    storages = {}
    for tensor in tensors:
        if tensor.device.type == "meta":
            raise ValueError("EXPERT_ALIAS_REQUIRES_MATERIALIZED_MODEL")
        storage = tensor.untyped_storage()
        if storage.nbytes():
            storages[_storage_key(tensor)] = storage.nbytes()
    return dict(unique_storages=len(storages), unique_storage_bytes=sum(storages.values()))


def alias_grouped_expert_storage(model, *, dry_run=False):
    """Point the expert parameters at their packed slices, keeping each Parameter object, shape, stride and dtype.

    The returned counts describe model-visible unique storage. CUDA allocator
    reservation need not fall, and references outside the model can delay
    actual allocation release. ``dry_run=True`` validates without mutation.
    """
    from hyff.kernels.grouped_experts import FrozenElementExpertMLP

    if model.training:
        raise ValueError("EXPERT_ALIAS_REQUIRES_EVAL_MODEL")
    if getattr(model, "compiled_nodes", False) or getattr(model, "_efm_state", None) is not None:
        raise ValueError("EXPERT_ALIAS_MUST_PRECEDE_EFM_AND_CAPTURE")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("EXPERT_ALIAS_REQUIRES_FROZEN_PARAMETERS_WITHOUT_GRADIENTS")

    original_parameters = tuple(model.named_parameters())
    original_keys = tuple(model.state_dict())
    parameters = [parameter for _, parameter in original_parameters]
    buffers = list(model.buffers())
    before = _storage_summary([*parameters, *buffers])
    plans, seen_parameters = [], {}
    grouped_modules = []
    for module_name, module in model.named_modules():
        if not isinstance(module, FrozenElementExpertMLP):
            continue
        grouped_modules.append(module)
        if len(module.experts) != module.count:
            raise ValueError("EXPERT_ALIAS_GROUP_COUNT_MISMATCH")
        if tuple(parameter._version for parameter in module.parameters()) != module._parameter_versions:
            raise ValueError("EXPERT_ALIAS_PACKED_CACHE_VERSION_MISMATCH")
        for stage, position in ((1, 0), (2, -1)):
            weight = getattr(module, f"weight{stage}")
            bias = getattr(module, f"bias{stage}")
            if (weight.ndim != 3 or bias.ndim != 2 or weight.shape[0] != module.count
                    or bias.shape[:2] != weight.shape[:2]):
                raise ValueError("EXPERT_ALIAS_PACKED_CACHE_SHAPE_MISMATCH")
            if weight.requires_grad or bias.requires_grad:
                raise ValueError("EXPERT_ALIAS_REQUIRES_FROZEN_PACKED_BUFFERS")
            for expert_index, expert in enumerate(module.experts):
                layer = expert[position]
                for attribute, packed in (("weight", weight), ("bias", bias)):
                    parameter = getattr(layer, attribute)
                    view = packed[expert_index]
                    if not isinstance(parameter, nn.Parameter):
                        raise ValueError("EXPERT_ALIAS_ORIGINAL_PARAMETER_REQUIRED")
                    if (parameter.shape != view.shape or parameter.stride() != view.stride()
                            or parameter.dtype != view.dtype or parameter.device != view.device
                            or parameter.dtype != torch.float32 or not view.is_contiguous()):
                        raise ValueError("EXPERT_ALIAS_EXACT_CONTIGUOUS_FP32_LAYOUT_REQUIRED")
                    if not torch.equal(parameter.detach(), view):
                        raise ValueError("EXPERT_ALIAS_PACKED_VALUE_MISMATCH")
                    destination = (_storage_key(view), view.storage_offset())
                    if id(parameter) in seen_parameters:
                        if seen_parameters[id(parameter)] != destination:
                            raise ValueError("EXPERT_ALIAS_TIED_PARAMETER_HAS_MULTIPLE_PACKED_SLICES")
                        continue
                    seen_parameters[id(parameter)] = destination
                    plans.append(dict(
                        parameter=parameter, view=view, version=parameter._version,
                        shape=parameter.shape, stride=parameter.stride(),
                        dtype=parameter.dtype, requires_grad=parameter.requires_grad,
                    ))

    if not grouped_modules:
        raise ValueError("EXPERT_ALIAS_REQUIRES_NATIVE_GROUPED_EXPERTS")
    replacement = {id(plan["parameter"]): plan["view"] for plan in plans}
    projected = _storage_summary([
        *(replacement.get(id(parameter), parameter) for parameter in parameters), *buffers,
    ])
    report = dict(
        schema="grouped_expert_storage_alias",
        dry_run=bool(dry_run), grouped_modules=len(grouped_modules),
        expert_parameters=len(plans),
        unique_storages_before=before["unique_storages"],
        unique_storages_after=projected["unique_storages"],
        unique_storage_bytes_before=before["unique_storage_bytes"],
        unique_storage_bytes_after=projected["unique_storage_bytes"],
        model_storage_savings_bytes=before["unique_storage_bytes"] - projected["unique_storage_bytes"],
        savings_scope="unique_model_visible_storage_not_allocator_reserved_memory",
        original_parameter_objects_preserved=True,
        state_dict_schema_and_values_preserved=True,
        kernels_and_arithmetic_changed=False,
        must_remain_frozen=True,
    )
    if dry_run:
        return report

    # Keep rollback views only until the transaction succeeds. These views own
    # the old storage, so allocation release happens after this function exits.
    originals = [(plan["parameter"], plan["parameter"].detach()) for plan in plans]
    try:
        with torch.no_grad():
            for plan in plans:
                plan["parameter"].data = plan["view"]
        actual_parameters = tuple(model.named_parameters())
        if (len(actual_parameters) != len(original_parameters)
                or any(name != old_name or parameter is not old_parameter
                       for (name, parameter), (old_name, old_parameter)
                       in zip(actual_parameters, original_parameters))):
            raise RuntimeError("EXPERT_ALIAS_PARAMETER_NAMESPACE_OR_IDENTITY_CHANGED")
        if tuple(model.state_dict()) != original_keys:
            raise RuntimeError("EXPERT_ALIAS_STATE_DICT_NAMESPACE_CHANGED")
        for plan in plans:
            parameter, view = plan["parameter"], plan["view"]
            if (parameter.shape != plan["shape"] or parameter.stride() != plan["stride"]
                    or parameter.dtype != plan["dtype"]
                    or parameter.requires_grad != plan["requires_grad"]
                    or parameter._version != plan["version"]
                    or parameter.data_ptr() != view.data_ptr()
                    or _storage_key(parameter) != _storage_key(view)):
                raise RuntimeError("EXPERT_ALIAS_PARAMETER_ABI_CHANGED")
        actual = _storage_summary([*model.parameters(), *model.buffers()])
        if actual != projected:
            raise RuntimeError("EXPERT_ALIAS_STORAGE_ACCOUNTING_MISMATCH")
    except BaseException:
        with torch.no_grad():
            for parameter, original in originals:
                parameter.data = original
        raise
    return report
