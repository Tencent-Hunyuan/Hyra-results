"""Owned native node stages with explicitly ordered global electronic-embedding softmax.

The caller owns graph geometry, projected-source packets, base/EFM aggregates,
pair/LR energy and the final energy/control anchor. This module only projects
owned rows and finishes native blocks. Prepare after binding the frozen base
execution policy. No single-GPU EFM wrapper or model/latent-charge forward is
installed or called.

The first base packet contains only the constant prepared scalar messages
(252 values per atom for largo). Inert first V/T caches belong to the caller's
base-kernel adapter and are never projection packet fields here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import hashlib
import math
from pathlib import Path
import weakref

import torch
from torch.nn import functional as F

from hyff.multigpu.electronic_state import OwnedPreparedState, _check_prepared
from hyff.multigpu.ordered_collectives import prepare_ordered_exchange, ordered_global_softmax
from hyff.models.l2_potential import L2Interaction
from hyff.engines.capture.efm_engine import _finish_with_efm
from hyff.models.latent_charge_model import _CONTEXT as _LATENT_CONTEXT, _capture_final_scalar


HERE = Path(__file__).resolve().parent
_ACTIVE = "_owned_node_sequence_active"


class OwnedNodeError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise OwnedNodeError(code)



@dataclass(frozen=True)
class EFMChannelAggregates:
    """Pre-exit channel aggregates, as returned by preparedEFM.aggregate.

    Do not construct this from preparedEFM.evaluate: that method has already
    applied the exits. Raw tuples are deliberately rejected by finish_block.
    """
    scalar: torch.Tensor
    vector: torch.Tensor
    tensor: torch.Tensor


@dataclass(frozen=True)
class ProjectedOwnedSources:
    normalized: torch.Tensor
    base_message: torch.Tensor
    base_vector: torch.Tensor | None
    base_tensor: torch.Tensor | None
    efm_scalar: torch.Tensor | None
    efm_vector: torch.Tensor | None
    efm_tensor: torch.Tensor | None

    @property
    def base_fields(self):
        return ((self.base_message,) if self.base_vector is None else
                (self.base_message, self.base_vector, self.base_tensor))

    @property
    def efm_fields(self):
        if self.efm_scalar is None:
            return ()
        return ((self.efm_scalar,) if self.efm_vector is None else
                (self.efm_scalar, self.efm_vector, self.efm_tensor))


@dataclass(frozen=True)
class OwnedNodeBlockResult:
    scalar: torch.Tensor
    vector: torch.Tensor
    tensor: torch.Tensor
    control: torch.Tensor


@dataclass(frozen=True)
class OwnedNodeResult:
    atomic_energy: torch.Tensor
    final_scalar: torch.Tensor
    control: torch.Tensor


def _method_identity(value):
    return id(getattr(value, "__func__", value))


def _execution_signature(model):
    # Version/identity checks contain no tensor-value reads. The module list
    # includes optimized linears and frozen expert inners after bind_base.
    return (
        tuple((id(module), _method_identity(module.forward)) for module in model.modules()),
        tuple((id(block), _method_identity(block._finish_messages),
               _method_identity(block._node_update)) for block in model.blocks),
        tuple((id(value), value._version, value.requires_grad) for value in model.parameters()),
    )


def _contexts_valid(model, capture_context=None):
    _require("_efm_state" not in model.__dict__ and
             all("_efm_state" not in block.__dict__ for block in model.blocks),
             "NODE_SINGLE_GPU_EFM_WRAPPER_INSTALLED")
    _require(model.__dict__.get(_LATENT_CONTEXT) is capture_context,
             "NODE_LATENT_CHARGE_CONTEXT_ACTIVE")


def _native_capture_hook(model):
    """Find the native final-scalar capture hook without removing any registered hook."""
    found = []
    for index, norm in enumerate(model.readout_norms):
        for hook in norm._forward_pre_hooks.values():
            if isinstance(hook, partial) and hook.func is _capture_final_scalar:
                _require(index == len(model.readout_norms) - 1 and len(hook.args) == 1 and
                         not hook.keywords and isinstance(hook.args[0], weakref.ReferenceType) and
                         hook.args[0]() is model, "NODE_NATIVE_CAPTURE_HOOK_INVALID")
                found.append(hook)
    _require(len(found) <= 1, "NODE_NATIVE_CAPTURE_HOOK_DUPLICATE")
    return found[0] if found else None


@dataclass(frozen=True)
class PreparedOwnedNodePipeline:
    model: object
    state: OwnedPreparedState
    aee_exchanges: tuple
    expert_modules: tuple
    execution_signature: tuple
    readout_scale: float
    native_capture_hook: object

    def metadata(self):
        return {
            "schema": "LARGO_OWNED_NODE_PIPELINE",
            "global_atoms": self.state.runtime.n_global,
            "owned_atoms": self.state.runtime.n_local,
            "block_count": len(self.model.blocks),
            "first_base_values": self.state.first_messages_owned.shape[1],
            "first_base_fields": ["M"],
            "first_messages_constant": True,
            "efm_input": "PRE_EXIT_CHANNEL_AGGREGATES",
            "node_finish": "NATIVE_METHODS",
            "placement": "NODE_UPDATE_AEE_REFINEMENTS_READOUT",
            "aee_physical_gradients": [value.physical_gradient for value in self.aee_exchanges],
            "expert_partition_scope": "OWNED_ROWS_RESTORED_ON_EXIT",
            "final_scalar": "AFTER_REFINEMENTS_BEFORE_FINAL_READOUT_NORM",
            "native_latent_charge_capture_hook": self.native_capture_hook is not None,
            "latent_charge_context": "SCOPED_CAPTURE_ONLY_RESTORED_ON_EXIT",
            "local_latent_charge_forward_called": False,
            "element_offset_count": 1,
            "includes_pair_or_long_range": False,
            "gpu_validated": False,
            "capture_validated": False,
        }


def prepare_owned_node_pipeline(model, state, *, aee_physical_gradients, trace=None, reuse=None):
    """Eager fixed-epoch preparation; all ranks agree explicit AEE grad flags.

    ``state`` must be prepared from this same model after bind_base (if used).
    This pipeline supports the largo scalar-first/cross-order native layout and
    rejects neighbor-count normalization, which needs a separate degree input.
    """
    _check_prepared(model, state)
    _contexts_valid(model)
    _require(_ACTIVE not in model.__dict__, "NODE_SEQUENCE_ALREADY_ACTIVE")
    if state.runtime.device.type == "cuda":
        _require(not torch.cuda.is_current_stream_capturing(), "NODE_PREPARATION_OUTSIDE_CAPTURE")
    count = len(model.blocks)
    flags = tuple(aee_physical_gradients)
    _require(len(flags) == count and all(type(value) is bool for value in flags),
             "NODE_EXPLICIT_AEE_GRADIENT_FLAGS_REQUIRED")
    _require(len(model.readout_norms) == count and len(model.readouts) == count and
             len(model.refinements) <= count, "NODE_READOUT_LAYOUT_INVALID")
    _require(all(not value.requires_grad and value.dtype == state.initial_scalar.dtype and
                 value.device == state.runtime.device for value in model.parameters()),
             "NODE_FROZEN_PARAMETER_TYPE_REQUIRED")
    for index, block in enumerate(model.blocks):
        _require(isinstance(block, L2Interaction) and block.input_is_scalar == (index == 0) and
                 block.cross_order == (index > 0) and
                 block.scalar_hidden == model.config.hidden and block.tensor_hidden == model.tensor_hidden,
                 "NODE_NATIVE_BLOCK_LAYOUT_REQUIRED")
        _require(getattr(block, "stab_aggregate_normalization", None) is None,
                 "NODE_AGGREGATE_NORMALIZATION_UNSUPPORTED")
    _require(state.first_messages_owned.shape ==
             (state.runtime.n_local, 3 * model.blocks[0].message_hidden) and
             not state.first_messages_owned.requires_grad and not state.initial_scalar.requires_grad,
             "NODE_CONSTANT_FIRST_MESSAGE_REQUIRED")
    expert_indices = tuple(model.config.expert_blocks)
    _require(len(set(expert_indices)) == len(expert_indices) and
             all(type(index) is int and 0 <= index < count for index in expert_indices),
             "NODE_EXPERT_BLOCKS_INVALID")
    experts = tuple(model.blocks[index].update_mlp for index in expert_indices)
    _require(all(hasattr(module, "partition") for module in experts),
             "NODE_NATIVE_EXPERT_PARTITION_REQUIRED")
    capture_hook = _native_capture_hook(model)
    if reuse is None:
        exchanges = tuple(prepare_ordered_exchange(state.softmax_spec,
                          physical_gradient=flag, label="node_aee_" + str(index), trace=trace)
                          for index, flag in enumerate(flags))
    else:
        _require(isinstance(reuse, PreparedOwnedNodePipeline) and reuse.model is model and
                 reuse.execution_signature == _execution_signature(model) and
                 reuse.expert_modules == experts and reuse.native_capture_hook is capture_hook and
                 reuse.state.runtime.metadata.counts == state.runtime.metadata.counts and
                 tuple(value.physical_gradient for value in reuse.aee_exchanges) == flags and
                 all(value.trace is trace for value in reuse.aee_exchanges),
                 "NODE_REUSE_POLICY_CHANGED")
        exchanges = tuple(replace(value, spec=state.softmax_spec) for value in reuse.aee_exchanges)
    return PreparedOwnedNodePipeline(model, state, exchanges, experts,
                                     _execution_signature(model), 1.0 / math.sqrt(count), capture_hook)


def _tensor(value, state, tail, code):
    _require(isinstance(value, torch.Tensor) and
             value.shape == (state.runtime.n_local, *tail) and
             value.dtype == state.initial_scalar.dtype and value.device == state.runtime.device, code)


def _features(values, prepared):
    _require(isinstance(values, tuple) and len(values) == 3, "NODE_FEATURE_TRIPLE_REQUIRED")
    scalar, vector, tensor = values
    _tensor(scalar, prepared.state, (prepared.model.config.hidden,), "NODE_SCALAR_METADATA_INVALID")
    _tensor(vector, prepared.state, (3, prepared.model.tensor_hidden), "NODE_VECTOR_METADATA_INVALID")
    _tensor(tensor, prepared.state, (5, prepared.model.tensor_hidden), "NODE_TENSOR_METADATA_INVALID")


class OwnedNodeSequence:
    """One ordered node loop, with a scoped owned expert partition.

    Project once, then finish once for each block. The caller threads control
    through its packet/base boundaries and supplies the resulting token to
    finish_block. Finish returns the last AEE token for the next boundary.
    No token is numerically added to learned features or energy.
    """
    def __init__(self, prepared):
        _require(isinstance(prepared, PreparedOwnedNodePipeline), "NODE_PREPARED_PIPELINE_REQUIRED")
        self.prepared = prepared
        self.entered = self.closed = self.finished = False
        self.next_block = 0
        self.pending = self.expected = None
        self.atomic_energy = self.final_scalar = self.control = None
        self.saved_partitions = []
        self.capture_context = None
        self.context_installed = False
        self.had_latent_context = False
        self.previous_latent_context = None

    def __enter__(self):
        _require(not self.entered and not self.closed, "NODE_SEQUENCE_ALREADY_ENTERED")
        prepared = self.prepared
        model, state = prepared.model, prepared.state
        _check_prepared(model, state)
        _contexts_valid(model)
        _require(_ACTIVE not in model.__dict__, "NODE_SEQUENCE_ALREADY_ACTIVE")
        _require(_execution_signature(model) == prepared.execution_signature, "NODE_MODEL_CHANGED_REPREPARE")
        _require(_native_capture_hook(model) is prepared.native_capture_hook,
                 "NODE_NATIVE_CAPTURE_HOOK_CHANGED_REPREPARE")
        _require(torch.is_grad_enabled(), "NODE_FORCE_GRAD_MODE_REQUIRED")
        model.__dict__[_ACTIVE] = self
        try:
            # The model's readout pre-hook captures only the scalar. It does
            # not call the latent-charge energy, charge conservation, or model forward.
            self.had_latent_context = _LATENT_CONTEXT in model.__dict__
            self.previous_latent_context = model.__dict__.get(_LATENT_CONTEXT)
            self.capture_context = {"scalar": None}
            model.__dict__[_LATENT_CONTEXT] = self.capture_context
            self.context_installed = True
            for module in prepared.expert_modules:
                self.saved_partitions.append((module, module.partition))
                module.partition = state.partition
            self.atomic_energy = state.initial_scalar.new_zeros(state.runtime.n_local)
            self.entered = True
            return self
        except BaseException:
            self._restore()
            self.closed = True
            raise

    def _active(self, index):
        _require(self.entered and not self.closed and not self.finished, "NODE_SEQUENCE_NOT_ACTIVE")
        _require(type(index) is int and index == self.next_block and index < len(self.prepared.model.blocks),
                 "NODE_BLOCK_ORDER_OR_DUPLICATE")
        _check_prepared(self.prepared.model, self.prepared.state)
        _contexts_valid(self.prepared.model, self.capture_context)
        _require(_native_capture_hook(self.prepared.model) is self.prepared.native_capture_hook,
                 "NODE_NATIVE_CAPTURE_HOOK_CHANGED_REPREPARE")
        _require(self.prepared.model.__dict__.get(_ACTIVE) is self and
                 all(module.partition is self.prepared.state.partition for module in self.prepared.expert_modules),
                 "NODE_EXPERT_PARTITION_SCOPE_CHANGED")

    def project_owned_sources(self, index, scalar, vector, tensor):
        self._active(index)
        _require(self.pending is None, "NODE_SOURCES_ALREADY_PROJECTED")
        prepared, values = self.prepared, (scalar, vector, tensor)
        _features(values, prepared)
        if index == 0:
            _require(scalar is prepared.state.initial_scalar, "NODE_FIRST_SCALAR_CACHE_MISMATCH")
        else:
            _require(all(value is previous for value, previous in zip(values, self.expected)),
                     "NODE_PREVIOUS_BLOCK_OUTPUT_REQUIRED")
        block = prepared.model.blocks[index]
        normalized = block.scalar_norm(scalar)
        message = (prepared.state.first_messages_owned if index == 0 else
                   block.scalar_message(normalized))
        base_vector = base_tensor = None
        if not block.input_is_scalar:
            base_vector = block.edge_vector_in(vector) if block.tensor_bottleneck else vector
            base_tensor = block.edge_tensor_in(tensor) if block.tensor_bottleneck else tensor
        efm = getattr(block, "efm", None)
        efm_scalar = efm_vector = efm_tensor = None
        if efm is not None:
            efm_scalar = efm.node_scalar(normalized)
            if efm.m_blocks:
                efm_vector, efm_tensor = efm.node_vector(vector), efm.node_tensor(tensor)
        self.pending = tuple((value, value._version) for value in values)
        return ProjectedOwnedSources(normalized, message, base_vector, base_tensor,
                                     efm_scalar, efm_vector, efm_tensor)

    def finish_block(self, index, values, *, base_aggregates, efm_aggregates=None, control):
        self._active(index)
        _require(self.pending is not None, "NODE_SOURCES_MUST_BE_PROJECTED")
        prepared = self.prepared
        _features(values, prepared)
        _require(all(value is previous and value._version == version
                     for value, (previous, version) in zip(values, self.pending)),
                 "NODE_PROJECTED_INPUT_CHANGED")
        model, state = prepared.model, prepared.state
        block = model.blocks[index]
        _require(isinstance(base_aggregates, tuple) and len(base_aggregates) == 3,
                 "NODE_BASE_AGGREGATE_TRIPLE_REQUIRED")
        for value, tail in zip(base_aggregates, ((block.message_hidden,),
                                                (3, block.message_hidden), (5, block.message_hidden))):
            _tensor(value, state, tail, "NODE_BASE_AGGREGATE_METADATA_INVALID")
        efm = getattr(block, "efm", None)
        if efm is None:
            _require(efm_aggregates is None, "NODE_UNEXPECTED_EFM_AGGREGATES")
            output = block._finish_messages(*values, *base_aggregates)
        else:
            _require(type(efm_aggregates) is EFMChannelAggregates, "NODE_EFM_PRE_EXIT_CHANNELS_REQUIRED")
            channels = (efm_aggregates.scalar, efm_aggregates.vector, efm_aggregates.tensor)
            for value, tail in zip(channels, ((efm.channels,), (3, efm.channels), (5, efm.channels))):
                _tensor(value, state, tail, "NODE_EFM_CHANNEL_METADATA_INVALID")
            ds = efm.exit_scalar(channels[0])
            dv = efm.exit_vector(channels[1])
            dt = efm.exit_tensor(channels[2])
            output = _finish_with_efm(block, *values, *base_aggregates, ds, dv, dt)
        # The direct native finish bypasses block.forward and its AEE hook.
        # Keep the native score/embedding arithmetic; only the softmax is global.
        scalar, vector, tensor = output
        head = block.electronic_atom_attention
        normalized = F.layer_norm(scalar, (model.config.hidden,))
        logits = head.score_projection(F.silu(head.score_hidden(
            torch.cat((normalized, state.node_state), dim=-1))))
        weights, next_control = ordered_global_softmax(logits, control, prepared.aee_exchanges[index])
        allocation = weights * state.totals.index_select(0, state.graph_index)
        scalar = scalar + head.electronic_embedding(allocation)
        if index < len(model.refinements):
            for refinement in model.refinements[index]:
                scalar, vector, tensor = refinement(scalar, vector, tensor)
        self.atomic_energy = self.atomic_energy + prepared.readout_scale * model.readouts[index](
            model.readout_norms[index](scalar)).squeeze(-1)
        if index == len(model.blocks) - 1:
            expected_capture = scalar if prepared.native_capture_hook is not None else None
            _require(self.capture_context.get("scalar") is expected_capture,
                     "NODE_NATIVE_FINAL_SCALAR_CAPTURE_MISMATCH")
        self.expected = (scalar, vector, tensor)
        self.final_scalar, self.control = scalar, next_control
        self.pending = None
        self.next_block += 1
        return OwnedNodeBlockResult(scalar, vector, tensor, next_control)

    def finish(self):
        _require(self.entered and not self.closed and not self.finished, "NODE_SEQUENCE_NOT_ACTIVE")
        _require(self.next_block == len(self.prepared.model.blocks) and self.pending is None,
                 "NODE_BLOCK_APPLICATION_MISSING")
        model, state = self.prepared.model, self.prepared.state
        _contexts_valid(model, self.capture_context)
        expected_capture = self.final_scalar if self.prepared.native_capture_hook is not None else None
        _require(self.capture_context.get("scalar") is expected_capture,
                 "NODE_NATIVE_FINAL_SCALAR_CAPTURE_MISMATCH")
        atomic_energy = self.atomic_energy + model.element_offset(state.owned_atomic_numbers).squeeze(-1)
        result = OwnedNodeResult(atomic_energy, self.final_scalar, self.control)
        self.finished = True
        return result

    def _restore(self):
        try:
            for module, previous in reversed(self.saved_partitions):
                module.partition = previous
        finally:
            self.saved_partitions.clear()
            model = self.prepared.model
            if model.__dict__.get(_ACTIVE) is self:
                model.__dict__.pop(_ACTIVE)
            if self.context_installed:
                if self.had_latent_context:
                    model.__dict__[_LATENT_CONTEXT] = self.previous_latent_context
                else:
                    model.__dict__.pop(_LATENT_CONTEXT, None)
                self.capture_context.clear()
                self.context_installed = False
                self.capture_context = None
            self.pending = self.expected = None
            self.atomic_energy = self.final_scalar = self.control = None

    def __exit__(self, exception_type, exception, traceback):
        try:
            if exception_type is None:
                _require(self.next_block == len(self.prepared.model.blocks) and self.pending is None,
                         "NODE_BLOCK_APPLICATION_MISSING")
                _require(self.finished, "NODE_RESULT_NOT_FINALIZED")
        finally:
            self.closed = True
            self._restore()
        return False


__all__ = ["OwnedNodeError", "EFMChannelAggregates", "ProjectedOwnedSources",
           "OwnedNodeBlockResult", "OwnedNodeResult", "PreparedOwnedNodePipeline",
           "prepare_owned_node_pipeline", "OwnedNodeSequence"]
