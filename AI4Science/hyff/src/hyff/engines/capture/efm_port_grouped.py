"""EFM installation for largo: compiled without fullgraph, with graph breaks at the (grouped) element experts."""
from __future__ import annotations
from types import MethodType
import torch
from hyff.engines.capture.efm_port_dynamic import (
    EFMState as StaticEFMState, EFMOverflow, describe_efms, sibling,
)
from hyff.models.element_model import ElementExpertMLP
from hyff.kernels.grouped_experts import FrozenElementExpertMLP


class EFMState(StaticEFMState):
    def compile(self, function):
        if not self.compile_nodes:
            return function
        if function is sibling._geometry:
            return super().compile(function)
        options = dict(sibling.COMPILE_OPTIONS)
        options["fullgraph"] = False
        return torch.compile(function, **options)

    def report(self):
        record = super().report()
        record.update(
            segmented_expert_forward=True,
            segmented_expert_modules=int(getattr(self.model, "_xl_segmented_grouped_expert_modules", 0)),
            finish_fullgraph=False,
            term_fullgraph="callsite_choice",
            geometry_fullgraph="callsite_choice",
            expert_forward_arithmetic="element_expert_or_grouped_expert_forward",
            acceptance_changed=False,
            weights_changed=False)
        return record

def install_efm(model, *, slots=64, edge_reserve=1.5, compile_nodes=True,
                _state_factory=None):
    """Complete the staged base port in place, preserving attached EFM weights."""
    if getattr(model, 'compiled_nodes', False) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Uncompiled model with frozen parameters required')
    if any(len(group) for group in getattr(model, 'refinements', [])):
        raise ValueError('Atomic refinements have not been ported')
    efms, info = describe_efms(model)
    for index, radius in info['radii_A'].items():
        if radius > model.layer_cutoffs[index] or radius >= model.cutoff or radius not in model.layer_cutoffs:
            raise ValueError('EFM radius must be an existing layer radius below the model cutoff')
    native_count = 0
    for module in model.modules():
        if isinstance(module, (ElementExpertMLP, FrozenElementExpertMLP)):
            if not getattr(module, "_xl_grouped_expert_segmented", False):
                module.forward = torch.compiler.disable(module.forward, recursive=True)
                module._xl_grouped_expert_segmented = True
            native_count += 1
    model._xl_segmented_grouped_expert_modules = native_count
    state_factory = EFMState if _state_factory is None else _state_factory
    state = state_factory(model, efms, info, slots=slots, edge_reserve=edge_reserve, compile_nodes=compile_nodes)
    model.__dict__['_efm_state'] = state
    for index, block in enumerate(model.blocks):
        if index in efms:
            block.__dict__['_efm_state'], block.__dict__['_efm_index'] = state, index
            state.original_forward[index] = MethodType(type(block).forward, block)
            state.base_finish[index] = state.compile(block._finish_messages)

            def finish(scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t, ds, dv, dt, block=block):
                return sibling._finish_with_efm(block, scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t, ds, dv, dt)

            state.efm_finish[index] = state.compile(finish)
            block.forward = MethodType(sibling._block_forward_with_efm, block)
            block._finish_messages = MethodType(sibling._finish_dispatch, block)
        else:
            block.uncompiled_finish_messages = block._finish_messages
            block._finish_messages = state.compile(block._finish_messages)
        block.compiled_nodes = bool(compile_nodes)
    model.compiled_nodes = bool(compile_nodes)
    state.inner_forward = model.forward
    model.forward = MethodType(sibling._model_forward_with_efm, model)
    return state
