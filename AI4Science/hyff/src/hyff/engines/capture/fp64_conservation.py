"""Optional FP64 charge conservation of the largo latent-charge term (default: native FP32)."""
from __future__ import annotations

from types import MethodType

import torch
from hyff.models.charge_bound import raw_charges, is_native_bound_conservation


_REPORT = '_hyff_fp64_conservation_report'


def resolve_conservation(kind, requested):
    if requested is None:
        return None
    if requested not in ('native', 'fp64'):
        raise ValueError('conservation must be native, fp64, or None')
    if kind != 'largo':
        raise ValueError('conservation is supported only by largo')
    return dict(requested=requested, selected=requested,
                variant='V64' if requested == 'fp64' else 'native_fp32',
                selection_phase='construction_before_preparation')


def _v64(self, scalars, atom_batch, total_charge):
    # LatentChargeEnergy.conserved_charges in FP64, then one cast: q stays FP32 downstream.
    raw = raw_charges(self, scalars)
    raw64, groups = raw.double(), total_charge.numel()
    sums = raw64.new_zeros(groups).index_add(0, atom_batch, raw64)
    counts = raw64.new_zeros(groups).index_add(0, atom_batch, torch.ones_like(raw64))
    shifts = (total_charge.double().reshape(-1) - sums) / counts.clamp_min(1)
    return (raw64 + shifts[atom_batch]).float()


def install_fp64_conservation(model, policy):
    """Install once after the latent-charge context, before long range and prepare/force/capture.

    One instance override reaches every ``self.conserved_charges`` call on
    this term: the native forward and the dense/tree/SPME long-range forwards.
    The multi-GPU route takes conserved charges from its caller instead.
    """
    if policy['selected'] == 'native':
        return dict(policy)
    if policy['selected'] != 'fp64':
        raise ValueError('Unsupported resolved conservation')
    term = getattr(model, 'latent_charge_energy', None)
    if term is None or getattr(model, 'training', True) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('FP64 conservation requires a frozen evaluation largo model')
    if (_REPORT in term.__dict__ or
            ('conserved_charges' in term.__dict__ and not is_native_bound_conservation(term))):
        raise ValueError('FP64 conservation is already installed')
    if ('_long_range_original_forward' in term.__dict__ or
            '_long_range_original_prepare_energy' in model.__dict__):
        raise ValueError('Install FP64 conservation before long range')
    state = model.__dict__.get('_efm_state')
    if state is None or any(getattr(state, name, None) is not None for name in ('batch', 'csr', 'capacity')):
        raise ValueError('Install FP64 conservation before EFM preparation')
    metadata = model.__dict__.get('_latent_charge_prepared_metadata')
    if not isinstance(metadata, tuple) or len(metadata) != 2:
        raise ValueError('Install the latent-charge prepared context before FP64 conservation')
    from hyff.models.latent_charges import LatentChargeEnergy
    if type(term) is not LatentChargeEnergy:
        raise ValueError('FP64 conservation requires the LatentChargeEnergy module')
    report = dict(policy,
                  dtypes=dict(sum='float64', shift='float64', add='float64', q='float32'),
                  native_site='LatentChargeEnergy.conserved_charges',
                  charge_head='native head then checkpoint raw-charge bound, before FP64 centering',
                  reduction='same index_add over atom_batch as native, in FP64',
                  graphs=int(metadata[1].numel()),
                  callers='self.conserved_charges on this term: native forward and long range',
                  autograd_chain_kept=True, bitwise_deterministic=False)
    term.__dict__[_REPORT] = report
    term.conserved_charges = MethodType(_v64, term)
    return dict(report)
