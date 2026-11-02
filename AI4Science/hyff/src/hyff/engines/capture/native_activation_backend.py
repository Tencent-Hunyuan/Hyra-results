"""Native EFM activations with compiled FP32 rounding barriers (largo engine option).

Each EFM term built here passes the ordinary capture self-check before use.
The private compiler interfaces used here are those of PyTorch 2.7.1. No global
lowering registration, process precision flag, or accuracy guard is modified.
"""
from __future__ import annotations

from collections import Counter
import functools

import torch
from torch._decomp import remove_decompositions
from torch._inductor import lowering
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.ir import IRNode
from torch._inductor.virtualized import ops
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND

from hyff.engines.capture import efm_engine


def _handler(original, arithmetic=None):
    def scalar(left, right):
        return ops.inline_asm_elementwise(
            left, right,
            asm=f"{arithmetic}.rn.f32 $0, $1, $2;",
            constraints="=f,f,f", dtype=torch.float32, is_pure=True, pack=1,
        )

    def call(*args, **kwargs):
        if not any(isinstance(value, IRNode) for value in (*args, *kwargs.values())):
            return original(*args, **kwargs)
        tensors = [value for value in args if isinstance(value, IRNode)]
        cuda_fp32 = bool(tensors) and all(
            value.get_dtype() == torch.float32 and value.get_device().type == "cuda"
            for value in tensors
        )
        unit_alpha = not kwargs or (set(kwargs) == {"alpha"} and kwargs["alpha"] == 1)
        if arithmetic is None or not cuda_fp32 or not unit_alpha:
            return lowering.fallback_handler(
                original, add_to_fallback_set=False
            )(*args, **kwargs)
        promoted, _ = lowering.transform_args(
            list(args), {}, True, ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, False
        )
        return lowering.make_pointwise(scalar, override_return_dtype=torch.float32)(*promoted)

    name = "_efm_native_" + str(original).replace(".", "_")
    call.__name__ = call.__qualname__ = name
    call.__module__ = __name__
    call._inductor_lowering_function = True
    globals()[name] = call
    return call


def make_backend():
    """Build a backend limited to graphs explicitly passed by its caller."""
    activations = {
        getattr(torch.ops.aten, name).default: name
        for name in ("silu", "silu_backward", "sigmoid", "sigmoid_backward")
    }
    arithmetic = {}
    for name in ("add", "sub", "mul"):
        for overload in ("Tensor", "Scalar"):
            arithmetic[getattr(getattr(torch.ops.aten, name), overload)] = name
        arithmetic[getattr(torch.ops.prims, name).default] = name
    handlers = {target: _handler(target) for target in activations}
    handlers.update({target: _handler(target, name) for target, name in arithmetic.items()})
    stats = {"kind": "precise_inductor_native_activations", "compilations": []}

    def inner(gm, example_inputs, **kwargs):
        counts = Counter()
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            if target in activations:
                node.target = handlers[target]
                counts[activations[target]] += 1
            elif target in arithmetic:
                value = node.meta.get("val")
                if (
                    isinstance(value, torch.Tensor)
                    and value.dtype == torch.float32
                    and value.device.type == "cuda"
                ):
                    node.target = handlers[target]
                    counts[arithmetic[target]] += 1
        gm.graph.lint()
        gm.recompile()
        phase = (
            "backward" if kwargs.get("is_backward", False)
            else "inference" if kwargs.get("is_inference", False)
            else "forward"
        )
        stats["compilations"].append({"phase": phase, "rewritten": dict(counts)})
        return compile_fx_inner(gm, example_inputs, **kwargs)

    def backend(gm, example_inputs):
        decompositions = dict(select_decomp_table())
        remove_decompositions(
            decompositions,
            [torch.ops.aten.silu, torch.ops.aten.silu_backward,
             torch.ops.aten.sigmoid, torch.ops.aten.sigmoid_backward],
        )
        return compile_fx(
            gm, example_inputs, inner_compile=inner, decompositions=decompositions,
            config_patches={"triton.cudagraphs": False, "force_disable_caches": True},
        )

    return backend, stats


def make_term(efm):
    """Return one compiled EFM term; the caller's self-check and fallback choices still apply."""
    backend, stats = make_backend()
    term = torch.compile(
        functools.partial(efm_engine._efm_term, efm),
        backend=backend, fullgraph=True, dynamic=False,
    )
    term.backend_stats = stats
    return term
