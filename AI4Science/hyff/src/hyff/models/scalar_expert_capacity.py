"""Widen native atomwise scalar experts without widening edge or tensor states.

The added hidden units have zero output columns at initialization.  The expert
function is therefore unchanged mathematically, including its coordinate
derivatives.  Finite-precision GEMM reduction order can change when width changes.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


DEFAULT_SETTINGS = {
    "species": "all",
    "blocks": (0, 1, 2, 3, 4),
    "multiplier": 2,
    "seed": 202_610_141_553,
}


def _get_expander():
    # Native dependencies are loaded only when installing the capacity change.
    from hyff.models.radial_capacity import expand_network
    return expand_network


def _validated_settings(settings):
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise TypeError("scalar_expert_capacity must be a mapping")
    unknown = set(settings).difference(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError("Unknown scalar expert capacity setting")
    values = {**DEFAULT_SETTINGS, **dict(settings)}
    if values["species"] != "all":
        raise ValueError("Scalar expert capacity widens all species")
    blocks = values["blocks"]
    if not isinstance(blocks, (list, tuple)) or not blocks:
        raise ValueError("blocks must be a nonempty list or tuple")
    if any(type(i) is not int or i < 0 for i in blocks):
        raise ValueError("block indices must be nonnegative integers")
    if len(set(blocks)) != len(blocks):
        raise ValueError("Repeated capacity block")
    if type(values["multiplier"]) is not int or values["multiplier"] < 2:
        raise ValueError("multiplier must be an integer of at least two")
    if type(values["seed"]) is not int or values["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    values["blocks"] = tuple(blocks)
    return values


def _network_dimensions(network):
    if not isinstance(network, nn.Sequential) or len(network) != 3:
        raise ValueError("Expected a native affine/SiLU/affine expert")
    first, activation, last = network
    if not isinstance(first, nn.Linear) or not isinstance(last, nn.Linear):
        raise ValueError("Native expert endpoints must be affine layers")
    if not isinstance(activation, nn.SiLU) or activation.inplace:
        raise ValueError("Expected the smooth native SiLU activation")
    if first.bias is None or last.bias is None:
        raise ValueError("Native expert expansion expects both affine biases")
    if first.out_features != last.in_features:
        raise ValueError("Native expert has inconsistent hidden width")
    parameters = tuple(network.parameters())
    if any(p.device.type != "cpu" for p in parameters):
        raise ValueError("Capacity expansion must precede moving the model")
    if any(p.dtype != first.weight.dtype for p in parameters):
        raise ValueError("Native expert parameters must use one floating dtype")
    if not first.weight.is_floating_point():
        raise ValueError("Native expert parameters must be floating point")
    if not all(p.requires_grad for p in parameters):
        raise ValueError("Capacity expansion is for fresh trainable experts")
    return first.in_features, first.out_features, last.out_features


def install_scalar_expert_capacity(model, settings=None):
    """Expand the existing expert entries; keep their native routing and callers."""
    if hasattr(model, "scalar_expert_capacity_metadata"):
        raise ValueError("Scalar expert capacity already installed")
    values = _validated_settings(settings)
    species = tuple(model.atomic_numbers)
    if not species or len(set(species)) != len(species):
        raise ValueError("Native species must be nonempty and distinct")
    targets = []
    for block_index in values["blocks"]:
        if block_index >= len(model.blocks):
            raise ValueError("Capacity block is outside the native model")
        update = model.blocks[block_index].update_mlp
        experts = getattr(update, "experts", None)
        if not isinstance(experts, nn.ModuleList) or len(experts) != len(species):
            raise ValueError("Expected the native per-species expert ModuleList")
        for species_index, atomic_number in enumerate(species):
            network = experts[species_index]
            inputs, old_hidden, outputs = _network_dimensions(network)
            targets.append((experts, species_index, network, {
                "block": block_index,
                "atomic_number": int(atomic_number),
                "input_features": inputs,
                "output_features": outputs,
                "old_hidden": old_hidden,
                "new_hidden": old_hidden * values["multiplier"],
            }))

    generator = torch.Generator(device="cpu").manual_seed(values["seed"])
    expand_network = _get_expander()
    # Stage every replacement before mutating the model.
    replacements = []
    rows = []
    for experts, index, network, row in targets:
        expanded = expand_network(network, row["new_hidden"], generator)
        expanded.train(network.training)
        extra = row["new_hidden"] - row["old_hidden"]
        row["added_parameters"] = extra * (
            row["input_features"] + row["output_features"] + 1)
        row["added_macs_per_selected_atom"] = extra * (
            row["input_features"] + row["output_features"])
        replacements.append((experts, index, expanded))
        rows.append(row)
    for experts, index, expanded in replacements:
        experts[index] = expanded

    model.scalar_expert_capacity_metadata = {
        "mechanism": "all_species_native_scalar_expert_hidden_capacity",
        "species": "all",
        "blocks": list(values["blocks"]),
        "multiplier": values["multiplier"],
        "seed": values["seed"],
        "experts": rows,
        "added_parameters": sum(row["added_parameters"] for row in rows),
        "initial_function_preserved_mathematically": True,
        "native_routing_preserved": True,
        "scalar_state_width_preserved": True,
        "tensor_state_width_preserved": True,
        "edge_kernels_preserved": True,
        "native_readouts_preserved": True,
        "force_mode": "negative coordinate gradient of full native energy",
        "new_hidden_output_columns_initially_zero": True,
    }
    return model


def build_model(mapping):
    """Native factory protocol; only the capacity setting is removed."""
    if not isinstance(mapping, Mapping):
        raise TypeError("Model settings must be a mapping")
    native_settings = dict(mapping)
    capacity = _validated_settings(
        native_settings.pop("scalar_expert_capacity", None))
    from hyff.models.efm_model import build_model as native_build
    model = native_build(native_settings)
    return install_scalar_expert_capacity(model, capacity)
