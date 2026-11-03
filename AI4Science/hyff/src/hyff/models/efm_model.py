"""EFM model constructor: hyff.models.efm_backbone.build_model plus an optional per-block EFM edge cutoff.

  efm_edge_cutoff  absent (= the backbone model), or dict {block: radius} / list of length n_blocks with null on blocks that keep the
                   block cutoff. For an EFM block with edge radius r_e <= its layer cutoff, the EFM term of that block
                   receives the envelope smooth_switch(d, r_e, switch_width) * valid in place of the block envelope, so
                   it selects (envelope != 0) and weights exactly the edges a block at cutoff r_e would; the block's own
                   message passing, its radial features and every other block are unchanged.

Why: the MD cost of EFM is ~ edges x layers x C^2. With r_e = 3.5 A on a block with a 4.5 A layer cutoff the
stacked EFM layers run on about 13 edges/atom instead of 31, the same edge set the 3.5 A blocks already use.

Parameters, construction order and initialization are exactly those of the backbone build of the same config without
efm_edge_cutoff; EFM exits are zero, so the initial function equals the base model.

Implementation: the model forward is wrapped on the instance (MethodType) to compute the per-radius envelopes from the
batch exactly as L2Potential.forward computes the block envelopes (same displacement / valid / distance expressions, so
forces carry the envelope derivative), to put them on the EFM modules for the duration of the call and to remove them
afterwards; each EFM with an edge radius has its forward wrapped to read that envelope. Eager reference geometry only
(csr None, no reference chunking), as the backbone. The captured engines pass the r_e edge set to each EFM.
"""
from __future__ import annotations

from types import MethodType

import torch

from hyff.models.l2_potential import smooth_switch
from hyff.models import efm_backbone as BACKBONE

VARIANT = 'efm_edge_cutoff'
STASH = '_efm_edge_envelope'


def resolve_edge_cutoff(value, efm_blocks, layer_cutoffs):
    """{block: radius} for the EFM blocks whose EFM edge radius is given."""
    n = len(layer_cutoffs)
    if isinstance(value, dict):
        items = [(int(k), v) for k, v in value.items()]
    elif isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f'efm_edge_cutoff list must have length {n}')
        items = [(i, v) for i, v in enumerate(value) if v is not None]
    else:
        raise ValueError('efm_edge_cutoff must be a dict {block: radius} or a per-block list')
    radii = {}
    for index, radius in items:
        if index not in efm_blocks:
            raise ValueError(f'efm_edge_cutoff block {index} has no EFM')
        radius = float(radius)
        if not 0. < radius <= float(layer_cutoffs[index]):
            raise ValueError(f'block {index}: efm_edge_cutoff must be in (0, {layer_cutoffs[index]}]')
        radii[index] = radius
    if not radii:
        raise ValueError('efm_edge_cutoff given without any block')
    return radii


def _efm_forward(self, normalized_scalar, vector, tensor, radial, direction, quadrupole, source, destination, envelope):
    edge_envelope = self.__dict__.get(STASH)
    if edge_envelope is None or edge_envelope.shape != envelope.shape:
        raise RuntimeError('an EFM with an edge cutoff needs the efm model forward (eager reference geometry)')
    return type(self).forward(self, normalized_scalar, vector, tensor, radial, direction, quadrupole, source, destination,
                              edge_envelope)


def edge_envelopes(model, batch, radii):
    """smooth_switch(d, r, switch_width) * valid for each r, with the expressions of L2Potential.forward (csr None)."""
    positions = batch['positions']
    source, destination = batch['edge_index'][0], batch['edge_index'][1]
    displacement = positions[source] - positions[destination]
    if 'shifts' in batch:
        displacement = displacement + batch['shifts']
    valid = source != destination
    if 'edge_valid' in batch:
        valid = batch['edge_valid']
    elif 'shifts' in batch:
        valid = valid | (batch['shifts'] != 0).any(-1)
    displacement = torch.where(valid[:, None], displacement, model.padding_displacement)
    distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-9)
    return {r: smooth_switch(distance, r, model.config.switch_width) * valid for r in radii}


def _model_forward(self, batch, *args, **kwargs):
    csr = args[0] if args else kwargs.get('csr')
    if csr is not None or self.reference_chunk_size:
        raise ValueError('efm_edge_cutoff supports the eager reference geometry only (no CSR, no reference chunking)')
    targets = self.__dict__['_efm_efm_targets']
    envelopes = edge_envelopes(self, batch, sorted(set(r for _, r in targets)))
    for efm, radius in targets:
        efm.__dict__[STASH] = envelopes[radius]
    try:
        return self.__dict__['_efm_inner_forward'](batch, *args, **kwargs)
    finally:
        for efm, _ in targets:
            efm.__dict__.pop(STASH, None)


def build_model(config):
    config = dict(config)
    edge_cutoff = config.pop('efm_edge_cutoff', None)
    model = BACKBONE.build_model(config)
    if edge_cutoff is None:
        return model
    efm_blocks = [i for i, block in enumerate(model.blocks) if getattr(block, 'efm', None) is not None]
    radii = resolve_edge_cutoff(edge_cutoff, efm_blocks, model.layer_cutoffs)
    targets = []
    for index, radius in radii.items():
        efm = model.blocks[index].efm
        if not isinstance(efm, BACKBONE.GeneralEdgeFrameMixing) and not isinstance(efm, BACKBONE.EFM.EdgeFrameMixing):
            raise ValueError(f'block {index}: unknown EFM module {type(efm).__name__}')
        efm.forward = MethodType(_efm_forward, efm)
        efm.__dict__['edge_cutoff'] = radius
        targets.append((efm, radius))
        model.efm_backbone_spec['blocks'][index]['efm']['edge_cutoff'] = radius
    model.__dict__['_efm_efm_targets'] = targets
    model.__dict__['_efm_inner_forward'] = model.forward
    model.forward = MethodType(_model_forward, model)
    model.efm_edge_cutoff_spec = dict(variant=VARIANT, efm_edge_cutoff={int(k): v for k, v in radii.items()},
                         layer_cutoffs=[float(r) for r in model.layer_cutoffs],
                         implementation='backbone build unchanged; per-radius EFM envelopes from the batch in a wrapped model '
                                        'forward; eager reference geometry only')
    return model
