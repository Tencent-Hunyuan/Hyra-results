"""Construct HyFF models from explicit, data-only configurations (weights.json).

This module never reads a checkpoint or resolves a user-supplied Python import.
Configuration describes the model; weights are loaded separately.
"""
from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
import math
from types import MappingProxyType


FORMAT = "hyff-weights"
LAYERNORM_POLICY = "layernorm_centered_fp64"
# Spellings written by earlier exports of the same formats.
LEGACY_FORMATS = ("hyff-weights-v1",)
LEGACY_LAYERNORM_POLICIES = ("layernorm_centered_fp64_internal_same_epsilon_v1",)
# model_config key of the latent-charge bound.
CHARGE_BOUND_KEY = "charge_bound"
# Spelling of the same key in weight JSON files written by earlier exports; read as charge_bound.
LEGACY_CHARGE_BOUND_KEYS = ("latentq_charge_bound",)

BASE_CONFIG_FIELDS = frozenset({
    "hidden", "n_rbf", "interactions", "cutoff", "switch_width",
    "edge_hidden", "tensor_hidden", "atomic_refinements", "layer_cutoffs",
    "mlp_hidden", "scalar_source_blocks", "higher_invariants",
    "tensor_mixing_rank", "backbone", "atomic_numbers", "expert_routing",
    "expert_count", "expert_hidden", "expert_blocks", "radial_conditioning",
    "radial_hidden", "condition_spin",
})
EFM_CONFIG_FIELDS = BASE_CONFIG_FIELDS | frozenset({
    "stab_invariant_transform", "stab_output_gates", "stab_gate_init",
    "stab_residual_scale", "stab_aggregate_normalization", "efm_blocks",
    "efm_channels", "efm_mmax", "efm_layers", "efm_aggregation",
    "efm_complex", "efm_frame", "angular_channels", "angular_degrees",
    "angular_init_scale", "angular_blocks", "receiver_filter", "efm_receiver",
    "edge_mlp", "edge_mlp_hidden", "edge_mlp_radial", "efm_edge_cutoff",
})
MODEL_CONFIG_FIELDS = EFM_CONFIG_FIELDS | {
    "scalar_expert_capacity", CHARGE_BOUND_KEY, "trunk_dot_soft_cap",
}

# Model identifiers in weights.json name a constructor in this package.
_FACTORIES = {
    "hyff-l2": "hyff.models.base_model:build_model",
    "hyff-efm": "hyff.models.efm_model:build_model",
    "hyff-efm-capacity": "hyff.models.scalar_expert_capacity:build_model",
    "hyff-efm-electronic": "hyff.models.electronic_embedding:build_model",
    "hyff-efm-capacity-electronic": "hyff.models.electronic_embedding:build_capacity_model",
    "hyff-efm-charges-w672c1": "hyff.models.latent_charge_model:build_w672c1",
    "hyff-efm-charges-w640c3": "hyff.models.latent_charge_model:build_w640c3",
    "hyff-efm-charges-w256c1": "hyff.models.latent_charge_model:build_w256c1",
    "hyff-efm-charges-w704c1": "hyff.models.latent_charge_model:build_w704c1",
    "hyff-efm-charges-w576c4": "hyff.models.latent_charge_model:build_w576c4",
}
# Identifiers written by earlier exports of the same architectures.
_LEGACY_MODEL_IDS = {
    "state": "hyff-l2",
    "gen4": "hyff-efm",
    "gen4-capacity": "hyff-efm-capacity",
    "gen4-aee": "hyff-efm-electronic",
    "gen4-capacity-aee": "hyff-efm-capacity-electronic",
    "main-w672c1-latentq": "hyff-efm-charges-w672c1",
    "main-w640c3-latentq": "hyff-efm-charges-w640c3",
    "main-w256c1-latentq": "hyff-efm-charges-w256c1",
    "main-w704c1-latentq": "hyff-efm-charges-w704c1",
    "main-w576c4-latentq": "hyff-efm-charges-w576c4",
}

_ALIASES = dict(_FACTORIES)
_ALIASES.update({old: _FACTORIES[new] for old, new in _LEGACY_MODEL_IDS.items()})
for _factory in _FACTORIES.values():
    _ALIASES[_factory] = _factory
MODEL_FACTORIES = MappingProxyType(_ALIASES)
SUPPORTED_MODEL_IDS = tuple(_FACTORIES)
del _factory

CHARGE_FACTORIES = frozenset(
    factory for name, factory in _FACTORIES.items() if name.startswith("hyff-efm-charges-")
)
_CAPACITY_FACTORIES = frozenset(_FACTORIES[name] for name in (
    "hyff-efm-capacity", "hyff-efm-capacity-electronic", "hyff-efm-charges-w640c3",
    "hyff-efm-charges-w576c4",
))

_CAPACITY_FIELDS = frozenset({"species", "blocks", "multiplier", "seed"})
_STRING_VALUES = frozenset({
    "m26", "l4a0", "custom", "element", "shared", "asinh", "layernorm",
    "sqrt", "mean", "inv_sqrt_blocks", "projector", "explicit", "sum",
    "degree", "hidden", "rbf", "all",
})
_NUMERICAL_FIELDS = frozenset({
    "policy", "layernorm_modules", "parameter_changes",
    "mathematical_model_changes",
})


def factory_id(model_id):
    """Return the constructor identifier ("module:function") for a model_id."""
    if type(model_id) is not str or model_id not in MODEL_FACTORIES:
        raise ValueError("Unsupported model_id; use one of the HyFF model identifiers")
    return MODEL_FACTORIES[model_id]


def _copy_value(value, *, capacity=False):
    """Validate and copy constructor data without accepting executable objects."""
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Model configuration numbers must be finite")
        return value
    if type(value) is str:
        if value not in _STRING_VALUES:
            raise ValueError("Model configuration contains an unsupported string value")
        return value
    if type(value) in (list, tuple):
        copied = [_copy_value(item) for item in value]
        return tuple(copied) if type(value) is tuple else copied
    if isinstance(value, Mapping):
        copied = {}
        for key, item in value.items():
            if capacity:
                if type(key) is not str or key not in _CAPACITY_FIELDS:
                    raise ValueError("Unknown scalar expert capacity field")
            elif not ((type(key) is int and key >= 0) or
                      (type(key) is str and key.isascii() and key.isdecimal())):
                raise ValueError("Per-block model settings require nonnegative integer keys")
            copied[key] = _copy_value(item)
        return copied
    raise TypeError("Model configuration accepts only plain data values")


def charge_bound_setting(model_config):
    """Return the latent-charge bound entry of a model_config (any accepted spelling), or None."""
    present = [key for key in (CHARGE_BOUND_KEY,) + LEGACY_CHARGE_BOUND_KEYS if key in model_config]
    if len(present) > 1:
        raise ValueError("model_config lists the charge bound more than once")
    return model_config[present[0]] if present else None


def validate_model_config(model_id, model_config):
    """Return a validated copy; reject runtime paths and unrelated metadata."""
    factory = factory_id(model_id)
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping")
    if any(key in model_config for key in LEGACY_CHARGE_BOUND_KEYS):
        bound = charge_bound_setting(model_config)
        model_config = {CHARGE_BOUND_KEY if key in LEGACY_CHARGE_BOUND_KEYS else key: value
                        for key, value in model_config.items()}
        model_config[CHARGE_BOUND_KEY] = bound
    if "trunk_dot_soft_cap" in model_config and not (
            type(model_config["trunk_dot_soft_cap"]) is str and
            model_config["trunk_dot_soft_cap"] == "ABSENT"):
        raise ValueError("trunk_dot_soft_cap is not supported in this release")
    if CHARGE_BOUND_KEY in model_config and factory not in CHARGE_FACTORIES:
        raise ValueError("A charge bound is supported only by latent-charge constructors")
    fields = BASE_CONFIG_FIELDS if factory == _FACTORIES["hyff-l2"] else EFM_CONFIG_FIELDS
    fields = fields | {"trunk_dot_soft_cap"}
    if factory in CHARGE_FACTORIES:
        fields = fields | {CHARGE_BOUND_KEY}
    if factory in _CAPACITY_FACTORIES:
        fields = fields | {"scalar_expert_capacity"}
    if any(type(key) is not str or key not in fields for key in model_config):
        raise ValueError("model_config contains fields outside the constructor schema")
    values = {}
    for key, value in model_config.items():
        if key == CHARGE_BOUND_KEY:
            from hyff.models.charge_bound import normalize_charge_bound
            values[key] = normalize_charge_bound(value)
        elif key == "trunk_dot_soft_cap":
            values[key] = value
        else:
            values[key] = _copy_value(value, capacity=key == "scalar_expert_capacity")
    return values


def _check_numerics(model, requested):
    actual = getattr(model, "numerical_execution", None)
    if not isinstance(actual, Mapping) or actual.get("policy") != LAYERNORM_POLICY:
        raise ValueError("Constructed model lacks the FP64 LayerNorm policy")
    if requested is None:
        return
    if not isinstance(requested, Mapping) or any(
            type(key) is not str or key not in _NUMERICAL_FIELDS for key in requested):
        raise ValueError("Unknown numerical_execution fields")
    if requested.get("policy") not in (LAYERNORM_POLICY,) + LEGACY_LAYERNORM_POLICIES:
        raise ValueError("Unsupported numerical execution policy")
    for key in ("parameter_changes", "mathematical_model_changes"):
        if key in requested and requested[key] is not False:
            raise ValueError("The numerical policy preserves model parameters and equations")
    if "layernorm_modules" in requested:
        names = requested["layernorm_modules"]
        if type(names) not in (list, tuple) or any(type(name) is not str for name in names):
            raise TypeError("layernorm_modules must be a sequence of module names")
        if list(names) != list(actual.get("layernorm_modules", ())):
            raise ValueError("Constructed LayerNorm module set differs from numerical_execution")


def build_model(config):
    """Build a model from ``{model_id, model_config, numerical_execution?}``.

    ``format``, when present, must be ``hyff-weights``. The returned model
    keeps its constructor initialization and training/evaluation mode. Loading
    weights, choosing a device, and selecting an inference engine are separate.
    """
    if not isinstance(config, Mapping):
        raise TypeError("Model specification must be a mapping")
    allowed = {"format", "model_id", "model_config", "numerical_execution"}
    if any(type(key) is not str or key not in allowed for key in config):
        raise ValueError("Model specification contains unknown fields")
    if "format" in config and config["format"] not in (FORMAT,) + LEGACY_FORMATS:
        raise ValueError("Unsupported model specification format")
    if "model_id" not in config or "model_config" not in config:
        raise ValueError("Model specification requires model_id and model_config")
    factory = factory_id(config["model_id"])
    values = validate_model_config(config["model_id"], config["model_config"])
    bound = values.pop(CHARGE_BOUND_KEY, None)
    # trunk_dot_soft_cap="ABSENT" may appear in weight JSON files; it is never a constructor option.
    values.pop("trunk_dot_soft_cap", None)
    module, name = factory.split(":")
    model = getattr(import_module(module), name)(values)
    if bound is not None:
        from hyff.models.charge_bound import install_charge_bound
        install_charge_bound(model, bound)
    _check_numerics(model, config.get("numerical_execution"))
    return model


__all__ = [
    "build_model", "factory_id", "validate_model_config", "MODEL_FACTORIES",
    "SUPPORTED_MODEL_IDS", "MODEL_CONFIG_FIELDS", "BASE_CONFIG_FIELDS",
    "EFM_CONFIG_FIELDS", "FORMAT", "LAYERNORM_POLICY",
]
