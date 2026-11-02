"""largo model constructors: EFM model + electronic embedding + latent-charge long-range term.

The latent-charge energy is added through the model forward. Prepared (captured)
engines must call install_prepared_context AFTER their base binder has replaced
forward. No block recompute or energy prefixes.
"""
from __future__ import annotations

from functools import partial
from types import MethodType
import weakref

import torch

from hyff.models.latent_charges import LatentChargeEnergy


_CONTEXT = '_latent_charge_call_context'
_FORWARD = '_latent_charge_original_forward'
_ENERGY = '_latent_charge_prepared_original_energy'


def _capture_final_scalar(owner_ref, _norm, inputs):
    owner = owner_ref()
    if owner is None:
        raise RuntimeError('LATENT_CHARGE_OWNER_RELEASED')
    context = owner.__dict__.get(_CONTEXT)
    if context is None:
        # Native prepared binding may call forward only inside wrapped energy.
        raise RuntimeError('LATENT_CHARGE_FORWARD_OR_PREPARED_ENERGY_WRAPPER_REQUIRED')
    if context.get('scalar') is not None:
        raise RuntimeError('LATENT_CHARGE_FINAL_SCALAR_CAPTURE_REPEATED')
    if len(inputs) != 1 or inputs[0].ndim != 2:
        raise ValueError('LATENT_CHARGE_FINAL_SCALAR_MATRIX_REQUIRED')
    context['scalar'] = inputs[0]


def _evaluate(self, batch, original, args, kwargs, prepared=None):
    if self.__dict__.get(_CONTEXT) is not None:
        raise RuntimeError('LATENT_CHARGE_REENTRANT_OR_RECOMPUTED_FORWARD_UNSUPPORTED')
    if kwargs.get('prefix_depth') is not None or kwargs.get('return_prefix_depth') is not None:
        raise ValueError('LATENT_CHARGE_ENERGY_PREFIX_UNSUPPORTED')
    context = dict(scalar=None)
    self.__dict__[_CONTEXT] = context
    try:
        energy = original(batch, *args, **kwargs)
        scalar = context['scalar']
        if scalar is None or scalar.shape[-1] != int(self.config.hidden):
            raise RuntimeError('LATENT_CHARGE_FINAL_NATIVE_SCALAR_NOT_CAPTURED')
        if not torch.is_tensor(energy) or energy.ndim != 1:
            raise ValueError('LATENT_CHARGE_NATIVE_GRAPH_ENERGY_VECTOR_REQUIRED')
        positions = batch['positions']
        if prepared is None:
            atom_batch, total_charge = batch['batch'], batch['total_charge']
        else:
            atom_batch, total_charge = prepared
            if positions.shape[0] != atom_batch.numel():
                raise ValueError('LATENT_CHARGE_PREPARED_ATOM_COUNT_MISMATCH')
        if scalar.shape[0] != positions.shape[0] or energy.numel() != total_charge.numel():
            raise ValueError('LATENT_CHARGE_NATIVE_GRAPH_SHAPE_MISMATCH')
        term = self.latent_charge_energy(scalar, positions, atom_batch, total_charge)
        return energy + term.to(dtype=energy.dtype)
    finally:
        # Never keep the feature/coordinate autograd graph on the module.
        self.__dict__.pop(_CONTEXT, None)
        context.clear()


def _forward(self, batch, *args, **kwargs):
    if (len(args) > 2 and args[2] is not None) or (len(args) > 3 and args[3] is not None):
        raise ValueError('LATENT_CHARGE_ENERGY_PREFIX_UNSUPPORTED')
    return _evaluate(self, batch, self.__dict__[_FORWARD], args, kwargs)


def _prepared_energy(self, batch, *args, **kwargs):
    if len(args) > 3 and args[3] is not None:
        raise ValueError('LATENT_CHARGE_ENERGY_PREFIX_UNSUPPORTED')
    return _evaluate(self, batch, self.__dict__[_ENERGY], args, kwargs,
                     prepared=self.__dict__['_latent_charge_prepared_metadata'])


def install_latent_charges(model):
    if hasattr(model, 'latent_charge_energy') or _FORWARD in model.__dict__:
        raise ValueError('LATENT_CHARGE_ALREADY_INSTALLED')
    if not hasattr(model, 'attention_electronic_spec'):
        raise ValueError('LATENT_CHARGE_REQUIRES_ELECTRONIC_EMBEDDING')
    if not len(model.readout_norms) or len(model.readout_norms) != len(model.blocks):
        raise ValueError('LATENT_CHARGE_NATIVE_FINAL_READOUT_REQUIRED')
    reference = next(model.parameters())
    if reference.dtype != torch.float32:
        raise ValueError('LATENT_CHARGE_FP32_NATIVE_PARAMETERS_REQUIRED')
    width = int(model.config.hidden)
    original_ids = tuple(id(p) for p in model.parameters())
    # Keep the base/electronic-embedding RNG stream and every existing parameter object.
    with torch.random.fork_rng(devices=[]):
        term = LatentChargeEnergy(width).to(device=reference.device)
    model.add_module('latent_charge_energy', term)
    if tuple(id(p) for p in model.parameters())[:len(original_ids)] != original_ids:
        raise ValueError('LATENT_CHARGE_ORIGINAL_PARAMETER_IDENTITY_CHANGED')
    model.readout_norms[-1].register_forward_pre_hook(
        partial(_capture_final_scalar, weakref.ref(model)))
    model.__dict__[_FORWARD] = model.forward
    model.forward = MethodType(_forward, model)
    model.latent_charge_spec = dict(
        feature_width=width, added_parameters=width + 2,
        zero_init_energy_scale=True,
        final_scalars='last_native_readout_norm_input_after_block_refinements',
        training_forces='autograd_of_native_energy_plus_latent_energy',
        dense_limit=term.dense_limit, block_rows=term.block_rows,
        prefix_or_block_recompute_supported=False,
        prepared_MD_requires_composite_installer_after_native_bind=True,
    )
    return model


def _build(mapping, width, capacity):
    if capacity == 1:
        from hyff.models.electronic_embedding import build_model as base
        if 'scalar_expert_capacity' in mapping:
            raise ValueError('LATENT_CHARGE_NATIVE_CAPACITY1_MAPPING_REQUIRED')
    else:
        from hyff.models.electronic_embedding import build_capacity_model as base
        settings = mapping.get('scalar_expert_capacity')
        if not isinstance(settings, dict) or settings.get('multiplier') != capacity:
            raise ValueError('LATENT_CHARGE_EXACT_CAPACITY_MAPPING_REQUIRED')
    model = base(mapping)
    if int(model.config.hidden) != width:
        raise ValueError('LATENT_CHARGE_EXACT_WIDTH_MAPPING_REQUIRED')
    return install_latent_charges(model)


def build_w672c1(mapping):
    return _build(mapping, 672, 1)


def build_w640c3(mapping):
    return _build(mapping, 640, 3)


def build_w256c1(mapping):
    return _build(mapping, 256, 1)


def build_w704c1(mapping):
    return _build(mapping, 704, 1)


def build_w576c4(mapping):
    return _build(mapping, 576, 4)


def install_prepared_context(model, physical, device):
    """Composite installer, called after native prepared forward replacement."""
    from hyff.models.electronic_embedding import install_prepared_context as install_electronic
    if not hasattr(model, 'latent_charge_energy'):
        raise ValueError('LATENT_CHARGE_FACTORY_REQUIRED')
    if _ENERGY in model.__dict__:
        raise ValueError('LATENT_CHARGE_PREPARED_CONTEXT_ALREADY_INSTALLED')
    if getattr(model.forward, '__func__', None) is _forward:
        raise ValueError('LATENT_CHARGE_NATIVE_MD_BIND_MUST_PRECEDE_PREPARED_INSTALL')
    n = len(physical['positions'])
    if n < 1:
        raise ValueError('LATENT_CHARGE_POSITIVE_PHYSICAL_ATOM_COUNT_REQUIRED')
    install_electronic(model, physical, device)
    metadata = (
        torch.zeros(n, device=device, dtype=torch.long),
        torch.tensor([float(physical.get('total_charge', 0.))], device=device,
                     dtype=torch.float32),
    )
    model.__dict__['_latent_charge_prepared_metadata'] = metadata
    model.__dict__[_ENERGY] = model.energy
    model.energy = MethodType(_prepared_energy, model)
    return True
