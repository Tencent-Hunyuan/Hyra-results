"""Process-wide torch.compile policy for the largo engine: dynamic shapes, static parameter shapes."""
from __future__ import annotations

from functools import wraps
import torch

_CALLS = []


def _pin_tensor_tails(value, seen):
    if isinstance(value, torch.Tensor):
        if id(value) in seen:
            return
        seen.add(id(value))
        axes = range(value.ndim) if isinstance(value, torch.nn.Parameter) else range(1, value.ndim)
        for axis in axes:
            torch._dynamo.mark_static(value, axis)
    elif isinstance(value, dict):
        for item in value.values():
            _pin_tensor_tails(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _pin_tensor_tails(item, seen)


def install():
    """Preserve callsite fullgraph/backend/mode/options; override dynamic only."""
    from torch._dynamo import config
    config.force_nn_module_property_static_shapes = False
    config.force_parameter_static_shapes = True
    config.allow_unspec_int_on_nn_module = True
    if getattr(torch, "_hyff_all_dynamic_policy_installed", False):
        return
    original_compile = torch.compile

    def compile_dynamic(model=None, *args, **options):
        options["dynamic"] = True
        if model is None:
            return lambda function: compile_dynamic(function, *args, **options)
        _CALLS.append(dict(
            function_name=getattr(model, "__qualname__", type(model).__name__),
            fullgraph=bool(options.get("fullgraph", False)),
            disable=bool(options.get("disable", False))))
        compiled = original_compile(model, *args, **options)
        # Preserve the OptimizedModule interface and namespace. Model parameters
        # and buffers keep the explicit static parameter-shape policy.
        if isinstance(model, torch.nn.Module) or options.get("disable", False):
            return compiled

        @wraps(model)
        def dynamic_function(*inputs, **keyword_inputs):
            seen = set()
            _pin_tensor_tails(inputs, seen)
            _pin_tensor_tails(keyword_inputs, seen)
            return compiled(*inputs, **keyword_inputs)

        dynamic_function.hyff_compile_dynamic = True
        return dynamic_function

    torch.compile = compile_dynamic
    torch._hyff_all_dynamic_policy_installed = True
    torch._hyff_all_dynamic_original_compile = original_compile


def report():
    return dict(
        dynamic_scope="all_process_local_torch_compile_calls",
        torch_compile_dynamic=True,
        callsite_fullgraph_choices_preserved=True,
        callsite_backend_mode_options_preserved=True,
        force_nn_module_property_static_shapes=False,
        force_parameter_static_shapes=True,
        allow_unspec_int_on_nn_module=True,
        function_tensor_axes_after_axis_0_static=True,
        parameter_tensor_shapes_static=True,
        compiler_calls=len(_CALLS),
        fullgraph_true_calls=sum(call["fullgraph"] for call in _CALLS),
        fullgraph_false_calls=sum(not call["fullgraph"] for call in _CALLS),
        padding_added=False,
        physics_context_changed=False)
