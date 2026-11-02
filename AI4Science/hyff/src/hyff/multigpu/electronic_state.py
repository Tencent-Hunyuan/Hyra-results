"""Owned state preparation and native AEE algebra for one physical graph.

Preparation is eager and uses global N/Q/M independently of owned species.
Only owned scalar rows are stored; the native atomic-number first-message
table and the native species-ordered expert schedule remain setup products.
No model hook, forward method, expert partition, or parameter is replaced.

The caller finishes each native block directly (bypassing its AEE hook), then
calls ``sequence.after_block`` before refinements/readout. The sequence checks
ordered exactly-once application. It is not used during EFM tile replay.
All ranks, including empty owners, must follow the same forward/backward
collective order. Only first derivatives are supported; CUDA graph capture is
not validated.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from hyff.multigpu.collectives import (
    CollectiveRuntime, GLOBAL_SOFTMAX, _agreement, global_two_channel_softmax, prepare_exchange,
)
from hyff.kernels.grouped_experts import bind_partition
from hyff.models.base_model import forward_fp64


HERE = Path(__file__).resolve().parent


class OwnedAEEStateError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise OwnedAEEStateError(code)



def _modules(model):
    first = model.blocks[0]
    return (model.embedding, model.state_projection, first.scalar_norm, first.scalar_message,
            *(block.electronic_atom_attention for block in model.blocks))


def _parameter_signature(model):
    return tuple((id(parameter), parameter._version, parameter.requires_grad)
                 for module in _modules(model) for parameter in module.parameters())


def _native_context_absent(model):
    _require(not model.__dict__.get("_attention_electronic_active", False) and
             all("_attention_electronic_context" not in block.__dict__ for block in model.blocks),
             "AEE_NATIVE_CONTEXT_ACTIVE")


@dataclass(frozen=True)
class OwnedPreparedState:
    runtime: CollectiveRuntime
    owned_atomic_numbers: torch.Tensor
    species_order: tuple[int, ...]
    partition: object
    state_features: torch.Tensor
    initial_scalar: torch.Tensor
    first_message_table: torch.Tensor
    first_messages_owned: torch.Tensor
    graph_index: torch.Tensor
    totals: torch.Tensor
    score_state: torch.Tensor
    node_state: torch.Tensor
    softmax_spec: object
    block_ids: tuple[int, ...]
    parameter_signature: tuple
    tensor_versions: tuple
    initial_scalar_table: torch.Tensor

    def metadata(self):
        return {
            "schema": "LARGO_OWNED_AEE_STATE", "global_atoms": self.runtime.n_global,
            "owned_atoms": self.runtime.n_local, "species_count": len(self.species_order),
            "species_order": list(self.species_order), "block_count": len(self.block_ids),
            "state_features": ["Q/N", "(M-1)/N"] if self.state_features.shape[1] == 2 else ["Q/N"],
            "aee_score_features": ["Q", "M-1", "log1p(N)"],
            "first_message_table_index": "ATOMIC_NUMBER", "expert_indices": "OWNED_LOCAL_ROWS",
            "softmax": "GLOBAL_MAX_SUM_AND_GLOBAL_VJP_CONTRACTION",
            "placement": "AFTER_NATIVE_BLOCK_BEFORE_REFINEMENTS_AND_READOUT",
            "constant_first_sources_require_grad": False, "model_mutated": False,
            "capture_validated": False, "gpu_validated": False,
        }


@torch.no_grad()
def prepare_owned_state(model, owned_atomic_numbers, runtime, *, total_charge, multiplicity):
    """Prepare a fixed species/ownership/physical-state epoch outside capture.

``runtime`` already binds canonical global IDs and ownership. The caller must
    pass species in its canonical owned-row order and assign ``state.partition``
    to the native grouped experts when binding that epoch. All 83 native species
    entries, including zero-count groups, keep the model's species order.
    No local atom count is substituted into physical state features.
    """
    _require(isinstance(runtime, CollectiveRuntime), "AEE_RUNTIME_REQUIRED")
    _require(runtime.n_global > 0, "AEE_POSITIVE_GLOBAL_COUNT_REQUIRED")
    if runtime.device.type == "cuda":
        _require(not torch.cuda.is_current_stream_capturing(), "AEE_PREPARATION_OUTSIDE_CAPTURE")
    _require(type(total_charge) in (int, float) and math.isfinite(total_charge), "AEE_CHARGE_INVALID")
    _require(type(multiplicity) in (int, float) and math.isfinite(multiplicity) and
             multiplicity >= 1 and int(multiplicity) == multiplicity, "AEE_MULTIPLICITY_INVALID")
    _require(len(model.blocks) > 0, "AEE_NATIVE_BLOCKS_REQUIRED")
    _native_context_absent(model)
    reference = model.embedding.weight
    _require(reference.dtype in (torch.float32, torch.float64) and reference.device == runtime.device,
             "AEE_REFERENCE_TYPE_INVALID")
    _require(isinstance(owned_atomic_numbers, torch.Tensor) and owned_atomic_numbers.dtype == torch.int64 and
             owned_atomic_numbers.device == runtime.device and owned_atomic_numbers.shape == (runtime.n_local,),
             "AEE_OWNED_SPECIES_METADATA_INVALID")
    species = tuple(model.atomic_numbers)
    _require(bool(species) and all(type(z) is int and 1 <= z < reference.shape[0] for z in species) and
             len(species) == len(set(species)), "AEE_SPECIES_ORDER_INVALID")
    _require(all(parameter.dtype == reference.dtype and parameter.device == reference.device and
                 not parameter.requires_grad for module in _modules(model) for parameter in module.parameters()),
             "AEE_FROZEN_PARAMETER_TYPE_REQUIRED")
    width = int(model.config.hidden)
    _require(reference.shape[1] == width and model.state_projection.weight.shape ==
             (width, 2 if model.condition_spin else 1), "AEE_STATE_PROJECTION_INVALID")
    _require(model.condition_spin or multiplicity == 1, "AEE_NONSINGLET_REQUIRES_SPIN")
    for block in model.blocks:
        head = block.electronic_atom_attention
        _require(head.__class__.__name__ == "ElectronicAtomAttention" and
                 head.__class__.__module__ == "hyff.models.electronic_embedding" and
                 head.score_hidden.weight.shape[1] == width + 3 and
                 head.score_projection.weight.shape[0] == 2 and
                 head.electronic_embedding.weight.shape == (width, 2) and
                 head.electronic_embedding.bias is None, "AEE_NATIVE_WIDTH_HEAD_REQUIRED")
    # Agree physical inputs as well as ownership, with the eager signature check
    # of collectives._agreement; its host checks never enter tensor evaluation.
    _agreement(runtime, {"component": "owned_aee_state", "n_global": runtime.n_global,
                         "total_charge": float(total_charge), "multiplicity": int(multiplicity),
                         "dtype": str(reference.dtype), "width": width, "species": species,
                         "condition_spin": bool(model.condition_spin), "blocks": len(model.blocks)})
    partition = bind_partition(owned_atomic_numbers, species)
    count = reference.new_full((1,), runtime.n_global)
    charge = reference.new_full((1,), total_charge)
    spin = reference.new_full((1,), multiplicity) - 1.0
    _require(bool(torch.isfinite(count).all() & torch.isfinite(charge).all() & torch.isfinite(spin).all()),
             "AEE_STATE_OUTSIDE_DTYPE_RANGE")
    intensive = [charge / count]
    if model.condition_spin:
        intensive.append(spin / count)
    features = torch.stack(intensive, dim=-1)
    state = F.linear(features, model.state_projection.weight, model.state_projection.bias)
    initial_table = reference + state
    first = model.blocks[0]
    normalized = forward_fp64(first.scalar_norm, initial_table)
    source = first.scalar_message
    table = F.linear(F.silu(F.linear(normalized, source[0].weight, source[0].bias)),
                     source[2].weight, source[2].bias)
    initial = initial_table.index_select(0, owned_atomic_numbers).detach()
    messages = table.index_select(0, owned_atomic_numbers).detach()
    graph_index = torch.zeros(runtime.n_local, dtype=torch.int64, device=runtime.device)
    totals = torch.stack((charge, spin), dim=-1)
    score_state = torch.cat((totals, torch.log1p(count).unsqueeze(-1)), dim=-1)
    node_state = score_state.index_select(0, graph_index)
    spec = prepare_exchange(runtime, (2,), reference.dtype, GLOBAL_SOFTMAX)
    numbers = owned_atomic_numbers.detach().clone()
    tensors = (numbers, features, initial, initial_table, table, messages, graph_index, totals, score_state, node_state,
               runtime.local_indices, partition.order, partition.starts, partition.tiles)
    return OwnedPreparedState(runtime, numbers, species, partition, features, initial, table, messages,
                              graph_index, totals, score_state, node_state, spec,
                              tuple(id(block) for block in model.blocks), _parameter_signature(model),
                              tuple((value, value._version) for value in tensors), initial_table)


@torch.no_grad()
def rebind_owned_state(model, previous, owned_atomic_numbers, runtime):
    """Reindex invariant native tables without repeating model arithmetic.

    This is an internal MD epoch boundary, not a new physical-state factory.
    N, Q, M, model and species order stay fixed by the caller's reuse state.
    Matching row counts alone never allow reuse of owned values; the canonical
    owner map must be unchanged. A changed map gets a fresh partition,
    local table selections and current exchange/runtime, including empty ranks.
    """
    _check_prepared(model, previous)
    _require(isinstance(runtime, CollectiveRuntime) and
             runtime.metadata.global_ids == previous.runtime.metadata.global_ids and
             runtime.world_size == previous.runtime.world_size and
             runtime.rank == previous.runtime.rank and runtime.group is previous.runtime.group and
             runtime.device == previous.runtime.device and runtime.backend == previous.runtime.backend,
             "AEE_REBIND_RUNTIME_IDENTITY_CHANGED")
    _require(isinstance(owned_atomic_numbers, torch.Tensor) and owned_atomic_numbers.dtype == torch.int64 and
             owned_atomic_numbers.device == runtime.device and owned_atomic_numbers.shape == (runtime.n_local,),
             "AEE_OWNED_SPECIES_METADATA_INVALID")
    same_owners = previous.runtime.metadata.owners == runtime.metadata.owners
    if same_owners:
        _require(torch.equal(owned_atomic_numbers, previous.owned_atomic_numbers),
                 "AEE_REBIND_SPECIES_CHANGED")
        # The maps may be shared by a new runtime whose edge-load diagnostics
        # differ. All exchange consumers must see that new runtime metadata.
        spec = replace(previous.softmax_spec, runtime=runtime)
        return replace(previous, runtime=runtime, softmax_spec=spec,
                       tensor_versions=tuple((value, version) for value, version in previous.tensor_versions
                                             if value is not previous.runtime.local_indices) +
                           ((runtime.local_indices, runtime.local_indices._version),))
    partition = bind_partition(owned_atomic_numbers, previous.species_order)
    initial = previous.initial_scalar_table.index_select(0, owned_atomic_numbers).detach()
    messages = previous.first_message_table.index_select(0, owned_atomic_numbers).detach()
    graph_index = torch.zeros(runtime.n_local, dtype=torch.int64, device=runtime.device)
    node_state = previous.score_state.index_select(0, graph_index)
    numbers = owned_atomic_numbers.detach().clone()
    spec = (replace(previous.softmax_spec, runtime=runtime)
            if previous.runtime.metadata.counts == runtime.metadata.counts else
            prepare_exchange(runtime, (2,), previous.initial_scalar.dtype, GLOBAL_SOFTMAX))
    tensors = (numbers, previous.state_features, initial, previous.initial_scalar_table,
               previous.first_message_table, messages, graph_index, previous.totals,
               previous.score_state, node_state, runtime.local_indices,
               partition.order, partition.starts, partition.tiles)
    return replace(previous, runtime=runtime, owned_atomic_numbers=numbers, partition=partition,
                   initial_scalar=initial, first_messages_owned=messages, graph_index=graph_index,
                   node_state=node_state, softmax_spec=spec,
                   tensor_versions=tuple((value, value._version) for value in tensors))


def _check_prepared(model, state):
    _require(isinstance(state, OwnedPreparedState), "AEE_PREPARED_STATE_REQUIRED")
    current = (state.owned_atomic_numbers, state.state_features, state.initial_scalar,
               state.initial_scalar_table, state.first_message_table, state.first_messages_owned,
               state.graph_index, state.totals, state.score_state, state.node_state,
               state.runtime.local_indices, state.partition.order, state.partition.starts, state.partition.tiles)
    # The unchanged-owner rebind puts its current runtime map last. Compare
    # identity multisets, including duplicates, without inspecting values.
    _require(sorted(map(id, current)) == sorted(id(value) for value, _version in state.tensor_versions),
             "AEE_PREPARED_TENSOR_REPLACED")
    _require(tuple(id(block) for block in model.blocks) == state.block_ids and
             _parameter_signature(model) == state.parameter_signature,
             "AEE_MODEL_CHANGED_REPREPARE")
    _require(all(value._version == version for value, version in state.tensor_versions),
             "AEE_PREPARED_TENSOR_CHANGED")
    _native_context_absent(model)


def owned_aee_allocations(head, scalar, state):
    """AEE score/embedding step with the global softmax and its exact transpose.

This mathematical primitive performs one collective softmax. Use the block
    sequence for normal model placement. The caller orders it relative to all
    other forward/reverse collectives, also on zero-row ranks.
    """
    _require(isinstance(state, OwnedPreparedState), "AEE_PREPARED_STATE_REQUIRED")
    width = state.initial_scalar.shape[1]
    _require(isinstance(scalar, torch.Tensor) and scalar.shape == (state.runtime.n_local, width) and
             scalar.dtype == state.initial_scalar.dtype and scalar.device == state.runtime.device,
             "AEE_OWNED_SCALAR_METADATA_INVALID")
    _require(scalar.requires_grad, "AEE_OWNED_GRADIENT_DEPENDENCY_REQUIRED")
    normalized = F.layer_norm(scalar, (width,))
    logits = head.score_projection(F.silu(head.score_hidden(torch.cat((normalized, state.node_state), dim=-1))))
    weights = global_two_channel_softmax(logits, state.softmax_spec)
    return weights * state.totals.index_select(0, state.graph_index)


class OwnedAEEBlockSequence:
    """One explicit AEE application per native block in one eager force call.

Use as a context manager around the caller's ordered block/refinement/readout
    loop. No output tensor or autograd graph is retained on this object/model.
    Calling a hooked block normally is outside this direct-finish contract.
    """
    def __init__(self, model, state):
        _check_prepared(model, state)
        self.model, self.state = model, state
        self.next_block = 0
        self.entered = False
        self.closed = False

    def __enter__(self):
        _require(not self.entered and not self.closed, "AEE_SEQUENCE_ALREADY_ENTERED")
        self.entered = True
        return self

    def after_block(self, index, output):
        _require(self.entered and not self.closed, "AEE_SEQUENCE_NOT_ACTIVE")
        _require(type(index) is int and index == self.next_block and index < len(self.state.block_ids),
                 "AEE_BLOCK_ORDER_OR_DUPLICATE")
        _check_prepared(self.model, self.state)
        _require(isinstance(output, tuple) and len(output) == 3, "AEE_NATIVE_OUTPUT_TUPLE_REQUIRED")
        scalar, vector, tensor = output
        for value, components in ((vector, 3), (tensor, 5)):
            _require(isinstance(value, torch.Tensor) and value.ndim == 3 and
                     value.shape[:2] == (self.state.runtime.n_local, components) and
                     value.shape[2] == self.model.tensor_hidden and
                     value.device == self.state.runtime.device and value.dtype == self.state.initial_scalar.dtype,
                     "AEE_NATIVE_TENSOR_METADATA_INVALID")
        head = self.model.blocks[index].electronic_atom_attention
        allocation = owned_aee_allocations(head, scalar, self.state)
        result = scalar + head.electronic_embedding(allocation)
        self.next_block += 1
        return result, vector, tensor

    def __exit__(self, exception_type, exception, traceback):
        self.closed = True
        if exception_type is None:
            _require(self.next_block == len(self.state.block_ids), "AEE_BLOCK_APPLICATION_MISSING")
        return False


__all__ = ["OwnedAEEStateError", "OwnedPreparedState", "prepare_owned_state", "rebind_owned_state",
           "owned_aee_allocations", "OwnedAEEBlockSequence"]
