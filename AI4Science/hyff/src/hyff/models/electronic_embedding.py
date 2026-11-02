"""Atom-attention electronic embedding of the global charge and spin.

Each block receives an extra invariant electronic embedding. Charge and
multiplicity-minus-one are allocated by two graph-wide atom softmaxes. The
allocations are recomputed from each block's scalar features; there is no
persistent electronic density, equilibration solver or extra model pass.
All forces remain derivatives of the complete molecular energy.
"""
from __future__ import annotations

from types import MethodType

import torch
from torch import nn


ATTENTION_HIDDEN = 8
_CONTEXT = "_attention_electronic_context"
_ACTIVE = "_attention_electronic_active"
_PREPARED_CONTEXT = "_electronic_prepared_context"
_NATIVE_ENERGY = "_electronic_native_energy"


def prepare_electronic_context(batch, reference):
    """Keep the undiluted graph Q, U=M-1, and graph size for atom attention."""
    graph_index = batch["batch"].to(device=reference.device, dtype=torch.long)
    charge = batch["total_charge"].to(reference).reshape(-1)
    multiplicity = batch["multiplicity"].to(reference).reshape(-1)
    if charge.shape != multiplicity.shape or graph_index.ndim != 1:
        raise ValueError("Native batch must have atom indices and one charge/multiplicity per graph")
    atom_counts = torch.bincount(graph_index, minlength=charge.numel()).to(reference)
    if atom_counts.shape != charge.shape:
        raise ValueError("Atom indices must correspond to the native graph state rows")
    totals = torch.stack((charge, multiplicity - 1.0), dim=-1)
    score_state = torch.cat((totals, torch.log1p(atom_counts).unsqueeze(-1)), dim=-1)
    return graph_index, totals, score_state.index_select(0, graph_index)


def graph_atom_softmax(logits, graph_index, graph_count):
    """Stable atom softmax separately for every graph and electronic channel.

    The detached shift is mathematically neutral to both first and second
    derivatives of softmax and avoids differentiating a max tie.
    """
    index = graph_index.unsqueeze(-1).expand_as(logits)
    maxima = logits.new_full((graph_count, logits.shape[-1]), float("-inf"))
    maxima.scatter_reduce_(0, index, logits.detach(), reduce="amax", include_self=True)
    numerators = torch.exp(logits - maxima.index_select(0, graph_index))
    denominators = logits.new_zeros((graph_count, logits.shape[-1]))
    denominators.index_add_(0, graph_index, numerators)
    return numerators / denominators.index_select(0, graph_index)


class ElectronicAtomAttention(nn.Module):
    """Separate charge/spin atom weights and zero-init scalar embeddings."""

    def __init__(self, width):
        super().__init__()
        self.width = int(width)
        self.score_hidden = nn.Linear(self.width + 3, ATTENTION_HIDDEN)
        self.score_projection = nn.Linear(ATTENTION_HIDDEN, 2)
        self.electronic_embedding = nn.Linear(2, self.width, bias=False)
        nn.init.zeros_(self.electronic_embedding.weight)

    def allocations(self, scalar, context):
        graph_index, totals, node_state = context
        normalized_scalar = torch.nn.functional.layer_norm(scalar, (self.width,))
        inputs = torch.cat((normalized_scalar, node_state), dim=-1)
        logits = self.score_projection(torch.nn.functional.silu(self.score_hidden(inputs)))
        weights = graph_atom_softmax(logits, graph_index, totals.shape[0])
        return weights * totals.index_select(0, graph_index)

    def forward(self, scalar, context):
        allocation = self.allocations(scalar, context)
        return scalar + self.electronic_embedding(allocation)


def _block_hook(block, _inputs, output):
    if not isinstance(output, tuple) or len(output) != 3:
        raise TypeError("Electronic attention requires the native scalar/vector/tensor tuple")
    scalar, vector, tensor = output
    if scalar.ndim != 2 or scalar.shape[-1] != block.electronic_atom_attention.width:
        raise ValueError("Electronic attention requires scalar features of the model width")
    context = block.__dict__.get(_CONTEXT)
    if context is None:
        raise RuntimeError("Native initial_scalar must precede attention-conditioned blocks")
    return block.electronic_atom_attention(scalar, context), vector, tensor


def _initial_scalar(self, batch):
    scalar = self.__dict__["_attention_electronic_native_initial_scalar"](batch)
    if self.__dict__.get(_ACTIVE, False):
        context = prepare_electronic_context(batch, scalar)
        for block in self.blocks:
            block.__dict__[_CONTEXT] = context
    return scalar


def _forward(self, batch, *args, **kwargs):
    if self.__dict__.get(_ACTIVE, False):
        raise RuntimeError("Electronic attention does not support reentrant model forwards")
    self.__dict__[_ACTIVE] = True
    try:
        return self.__dict__["_attention_electronic_native_forward"](batch, *args, **kwargs)
    finally:
        self.__dict__.pop(_ACTIVE, None)
        for block in self.blocks:
            block.__dict__.pop(_CONTEXT, None)


def install_attention(model):
    """Add one zero-initialized attention head per block; parameters of the base model are unchanged."""
    if not hasattr(model, "initial_scalar") or not hasattr(model, "blocks"):
        raise TypeError("Electronic attention requires state/block interfaces")
    if not len(model.blocks):
        raise ValueError("Electronic attention requires existing blocks")
    if any(hasattr(block, "electronic_atom_attention") for block in model.blocks):
        raise ValueError("Electronic attention is already installed")
    width = int(model.config.hidden)
    with torch.random.fork_rng(devices=[]):
        for block in model.blocks:
            reference = next(block.parameters())
            head = ElectronicAtomAttention(width).to(device=reference.device, dtype=reference.dtype)
            block.add_module("electronic_atom_attention", head)
            block.register_forward_hook(_block_hook)
    model.__dict__["_attention_electronic_native_forward"] = model.forward
    model.__dict__["_attention_electronic_native_initial_scalar"] = model.initial_scalar
    model.initial_scalar = MethodType(_initial_scalar, model)
    model.forward = MethodType(_forward, model)
    model.attention_electronic_spec = {
        "block_count": len(model.blocks),
        "scalar_width": width,
        "hidden": ATTENTION_HIDDEN,
        "state": "undiluted Q, multiplicity-1, log(1+atom_count)",
        "allocation": "two atom softmaxes per graph, sums Q and multiplicity-1",
        "injection": "zero-init Linear(2,width), scalar only, after each block",
    }
    return model


def build_model(config):
    """EFM model with an electronic attention head (no scalar expert capacity)."""
    from hyff.models.efm_model import build_model as efm_model

    if "scalar_expert_capacity" in config:
        raise ValueError("This constructor does not accept scalar_expert_capacity")
    return install_attention(efm_model(config))


def build_capacity_model(config):
    """EFM model with widened scalar experts and an electronic attention head."""
    from hyff.models.scalar_expert_capacity import build_model as capacity_model

    return install_attention(capacity_model(config))


def _prepared_energy(self, batch, *args, **kwargs):
    # Prepared CSR calls energy directly after the base port replaced forward and
    # initial_scalar. The context is fixed physical metadata prepared once outside
    # capture; the scalar-dependent softmax still executes in each block and call.
    context = self.__dict__[_PREPARED_CONTEXT]
    if len(batch['positions']) != len(context[0]):
        raise ValueError('Prepared electronic context has a different atom count')
    if any(_CONTEXT in b.__dict__ for b in self.blocks):
        raise RuntimeError('Prepared electronic context is already active')
    for block in self.blocks:
        block.__dict__[_CONTEXT] = context
    try:
        return self.__dict__[_NATIVE_ENERGY](batch, *args, **kwargs)
    finally:
        for block in self.blocks:
            block.__dict__.pop(_CONTEXT, None)


def install_prepared_context(model, physical, device):
    """Bind the fixed single-system electronic context for prepared (captured) execution."""
    if not hasattr(model, 'attention_electronic_spec'):
        return False
    if _NATIVE_ENERGY in model.__dict__:
        raise ValueError('Prepared electronic context is already installed')
    n = len(physical['positions'])
    batch = dict(batch=torch.zeros(n, device=device, dtype=torch.long),
                 total_charge=torch.tensor([float(physical.get('total_charge', 0.))], device=device),
                 multiplicity=torch.tensor([float(physical.get('multiplicity', 1))], device=device))
    reference = torch.empty((), device=device, dtype=torch.float32)
    model.__dict__[_PREPARED_CONTEXT] = prepare_electronic_context(batch, reference)
    model.__dict__[_NATIVE_ENERGY] = model.energy
    model.energy = MethodType(_prepared_energy, model)
    return True
