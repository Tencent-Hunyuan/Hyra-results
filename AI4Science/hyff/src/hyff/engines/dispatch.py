"""Construct one energy/force engine for one physical system.

Engines:
  presto, vivace, moderato  native fused-kernel engines of the three smaller models;
  andante, largo            captured engines with edge-frame-mixing kernels;
  reference                 the plain PyTorch model (eager);
  reference_cuda_graph      the plain PyTorch model captured in a CUDA graph.

The returned dictionary holds ``model``, ``batch``, ``engine`` and ``metadata``.
``engine(batch)`` returns the energy and the forces for ``batch['positions']``.
"""
from __future__ import annotations

import math

import torch

from hyff.registry import CAPTURE_ENGINES, NATIVE_ENGINES, REFERENCE_ENGINES
from hyff.weights import load_model


def create_engine(kind, checkpoint, positions32, numbers, cell=None, total_charge=0.,
                  multiplicity=1, skin=1.5, reserve=1.5, precision=None,
                  statistics='fp64', *, edge_reserve=None, expert_storage_alias=None,
                  capacity_policy=None, long_range=None, capture_branch_policy='first',
                  conservation=None):
    if capture_branch_policy not in ('off', 'first', 'every'):
        raise ValueError('capture_branch_policy must be off, first or every')
    if kind not in NATIVE_ENGINES + CAPTURE_ENGINES + REFERENCE_ENGINES:
        raise ValueError(f'Unknown engine {kind!r}')
    from hyff.engines.capture.capacity import resolve_capacity_policy
    from hyff.engines.capture.long_range import resolve_long_range
    from hyff.engines.capture.fp64_conservation import resolve_conservation
    capacity = resolve_capacity_policy(kind, capacity_policy, len(numbers))
    lr_policy = resolve_long_range(kind, long_range, len(numbers), periodic=cell is not None)
    conservation_policy = resolve_conservation(kind, conservation)
    precision = precision or ('fp32' if kind in ('andante',) + REFERENCE_ENGINES else 'tf32x3')
    allowed_statistics = ('fp64', 'fp32') if kind in NATIVE_ENGINES else ('fp64',)
    if statistics not in allowed_statistics:
        raise ValueError('statistics must be fp64, or fp32 for the presto, vivace and moderato engines')
    if kind in ('vivace', 'moderato') and statistics == 'fp32' and precision != 'tf32x3':
        raise ValueError('fp32 statistics require tf32x3 products')
    if expert_storage_alias is None:
        # The largo engine's weights are frozen: its expert parameters share storage with the packed
        # kernel buffers (same values; 4.7 GB less peak memory, same speed and forces).
        expert_storage_alias = kind == 'largo'
    if type(expert_storage_alias) is not bool:
        raise ValueError('expert_storage_alias must be a bool or None')
    if expert_storage_alias and kind != 'largo':
        raise ValueError('expert_storage_alias is supported only by the largo engine')
    if edge_reserve is not None:
        if kind not in CAPTURE_ENGINES:
            raise ValueError('edge_reserve is supported only by the andante and largo engines')
        try:
            edge_reserve = float(edge_reserve)
        except (TypeError, ValueError, OverflowError):
            raise ValueError('edge_reserve must be finite and at least 1') from None
        if not math.isfinite(edge_reserve) or edge_reserve < 1.:
            raise ValueError('edge_reserve must be finite and at least 1')
    if kind == 'largo' and precision != 'tf32x3':
        raise ValueError('The largo engine requires tf32x3 products')
    if kind == 'andante' and precision != 'fp32':
        raise ValueError('The andante engine requires fp32')
    if kind in NATIVE_ENGINES + CAPTURE_ENGINES:
        from hyff.engines.compiler_budget import reserve_compiled_engine_attempt
        reserve_compiled_engine_attempt(kind)
    if kind == 'largo':
        return _largo(checkpoint, positions32, numbers, cell, total_charge, multiplicity, skin, reserve,
                      precision, edge_reserve, expert_storage_alias, capacity, lr_policy, conservation_policy)
    if kind == 'andante':
        return _andante(checkpoint, positions32, numbers, cell, total_charge, multiplicity, skin, reserve,
                        precision, edge_reserve, capacity)
    if kind == 'presto':
        from hyff.engines.native import presto
        return presto.create_engine(positions32, numbers, cell, checkpoint, 'cuda',
                                    total_charge, multiplicity, skin=skin,
                                    reserve=reserve, precision=precision,
                                    layernorm_statistics=statistics, joined_fast=(precision != 'fp32'),
                                    capture_branch_policy=capture_branch_policy)
    if kind == 'moderato':
        from hyff.engines.native import moderato
        return moderato.create_full_engine(positions32, numbers, cell, checkpoint, 'cuda',
                                           total_charge, multiplicity, skin=skin,
                                           reserve=reserve, precision=precision, statistics=statistics,
                                           capture_branch_policy=capture_branch_policy)
    if kind == 'vivace':
        from hyff.engines.native import vivace
        return vivace.create_full_engine(positions32, numbers, cell, checkpoint, 'cuda',
                                         total_charge, multiplicity, skin=skin,
                                         reserve=reserve, precision=precision, statistics=statistics,
                                         capture_branch_policy=capture_branch_policy)
    return _reference(kind, checkpoint, positions32, numbers, cell, total_charge, multiplicity, skin,
                      capture_branch_policy)


def _largo(checkpoint, positions32, numbers, cell, total_charge, multiplicity, skin, reserve,
           precision, edge_reserve, expert_storage_alias, capacity, lr_policy, conservation_policy):
    from hyff.engines.capture.compile_policy import configure_largo_compiler
    from hyff.engines.capture.largo import create_engine as captured
    from hyff.engines.capture.native_activation_state import (
        NativeActivationEFMState, native_activation_enabled,
    )
    from hyff.models.charge_bound import charge_bound_value
    from hyff.models.latent_charge_model import install_prepared_context
    configure_largo_compiler()
    precise_efm_enabled = native_activation_enabled()
    model, _ = load_model(checkpoint, 'cpu')
    if lr_policy is not None and lr_policy['selected'] == 'spme':
        # Build the SPME mesh once on the CPU now: any cell or setting problem surfaces before compilation.
        from hyff.kernels.long_range.backends import validate_spme
        validate_spme(model.latent_charge_energy, cell)
    artifact_charge_bound = charge_bound_value(model)
    physical = dict(positions=positions32, numbers=numbers, cell=cell,
                    total_charge=total_charge, multiplicity=multiplicity)
    capacity_report, lr_report, conservation_report, alias_report = {}, {}, {}, {}
    effective_edge_reserve = (edge_reserve if edge_reserve is not None else
                              (1.05 if capacity is not None else 1.5))

    def install_context(current):
        if charge_bound_value(current) != artifact_charge_bound:
            raise ValueError('The latent charge bound changed during captured node binding')
        if capacity is not None:
            from hyff.engines.capture.capacity import install_largo_capacity
            capacity_report.update(install_largo_capacity(current, capacity, effective_edge_reserve))
        install_prepared_context(current, physical, 'cuda')
        if conservation_policy is not None:
            from hyff.engines.capture.fp64_conservation import install_fp64_conservation
            conservation_report.update(install_fp64_conservation(current, conservation_policy))
        if lr_policy is not None:
            from hyff.engines.capture.long_range import install_long_range
            lr_report.update(install_long_range(current, physical, lr_policy))

    capture_options = ({} if edge_reserve is None and capacity is None else
                       {'edge_reserve': effective_edge_reserve})
    if expert_storage_alias:
        from hyff.engines.capture.expert_storage_alias import alias_grouped_expert_storage

        def install_alias(current):
            alias_report.update(alias_grouped_expert_storage(current))

        capture_options['after_bind_base'] = install_alias
    made = captured(positions32, numbers, cell, None, 'cuda', total_charge,
                    multiplicity, skin, reserve, precision, model=model,
                    grouped_experts=True, joined_projections=False, fold_first=False,
                    prepared_context=install_context,
                    _state_factory=NativeActivationEFMState if precise_efm_enabled else None,
                    **capture_options)
    metadata = made['metadata']
    metadata.update(edge_reserve=float(made['state'].reserve), precise_efm=precise_efm_enabled)
    if expert_storage_alias:
        metadata['expert_storage_alias'] = alias_report
    if capacity is not None:
        metadata['capacity_policy'] = capacity_report
    if lr_policy is not None:
        metadata['long_range'] = lr_report
    if conservation_policy is not None:
        metadata['conservation'] = conservation_report
    if artifact_charge_bound is not None:
        metadata['charge_bound'] = artifact_charge_bound
    metadata['efm_backends'] = dict(getattr(made['state'], 'term_backend', {}))
    return made


def _andante(checkpoint, positions32, numbers, cell, total_charge, multiplicity, skin, reserve,
             precision, edge_reserve, capacity):
    from hyff.engines.capture.andante import create_engine as captured
    from hyff.engines.capture.efm_tiling import (
        DEFAULT_EDGE_RESERVE, make_post_install, runtime_metadata, tiled_metadata,
    )
    model, _ = load_model(checkpoint, 'cpu')
    effective_edge_reserve = DEFAULT_EDGE_RESERVE if edge_reserve is None else edge_reserve
    capture_options = dict(edge_reserve=effective_edge_reserve,
                           post_install_efm=make_post_install(
                               edge_reserve=effective_edge_reserve,
                               tiled=capacity is not None and capacity['selected'] == 'tiled'))
    made = captured(positions32, numbers, cell, None, 'cuda', total_charge,
                    multiplicity, skin, reserve, precision, model=model, **capture_options)
    try:
        made['metadata'].update(base_reserve=float(reserve),
                                edge_reserve=float(made['state'].reserve),
                                efm_policy=runtime_metadata(made['state']))
        if capacity is not None:
            report = dict(capacity)
            if capacity['selected'] == 'tiled':
                report['tiling'] = tiled_metadata(made['state'])
            made['metadata']['capacity_policy'] = report
    except BaseException:
        # Do not keep a constructed CUDA engine when a post-construction guard rejects it.
        try:
            made['engine'].close()
        except Exception:
            pass
        raise
    return made


def _reference(kind, checkpoint, positions32, numbers, cell, total_charge, multiplicity, skin,
               capture_branch_policy):
    from hyff.engines import reference
    from hyff.engines.force_engine import ForceEngine
    model, _ = load_model(checkpoint, 'cuda')
    batch = reference.system_batch(positions32, numbers, cell, total_charge, multiplicity, 'cuda')
    context = reference.bind_global_state(model, len(numbers), total_charge, multiplicity)
    backend = 'cuda_graph' if kind == 'reference_cuda_graph' else 'eager'
    engine = ForceEngine(model, batch, backend, skin=skin,
                         capture_branch_policy=capture_branch_policy)
    energy, force = engine(batch)
    if not bool(torch.isfinite(energy).all() & torch.isfinite(force).all()):
        engine.close()
        raise FloatingPointError('Nonfinite energy or force')
    return dict(model=model, batch=batch, engine=engine,
                metadata=dict(backend=backend, global_context=context,
                              skin_A=float(skin), precision='fp32'))


class Potential:
    """Eager energy/force evaluation on CPU or CUDA with Verlet-skin neighbour reuse.

    Structures of any size may follow each other (a new atom count starts a new neighbour list).
    For a latent-charge model (largo) with a periodic cell, ``long_range`` 'auto' or 'spme' evaluates
    the latent-charge term as the periodic SPME sum, like the largo engine; 'native', 'dense' and 'tree'
    keep the isolated sum (identical in eager evaluation).
    """

    def __init__(self, weights, device='cpu', skin=1.5, long_range='auto'):
        from hyff.graph import NeighborCache
        if long_range not in (None, 'auto', 'spme', 'native', 'dense', 'tree'):
            raise ValueError('long_range must be auto, spme, native, dense or tree')
        self.model, self.config = load_model(weights, device)
        self.device = device
        self.skin = skin
        self.long_range = long_range
        self.cache = NeighborCache(self.model.cutoff, skin)
        self._periodic = None   # (cell, SPME backend) of the last periodic call

    def _spme(self, batch):
        from hyff.kernels.long_range.backends import periodic_term, spme_backend
        term = getattr(self.model, 'latent_charge_energy', None)
        if term is None or 'cell' not in batch or self.long_range not in ('auto', 'spme'):
            if self.long_range == 'spme' and term is not None:
                raise ValueError('long_range=spme needs a periodic cell')
            return None
        cell = batch['cell'][0]
        if self._periodic is None or not torch.equal(self._periodic[0], cell):
            self._periodic = (cell.clone(), spme_backend(term, cell, self.device))
        return periodic_term(term, self._periodic[1])

    def __call__(self, positions, atomic_numbers, cell=None, total_charge=0., multiplicity=1):
        from contextlib import nullcontext
        from hyff.engines.reference import system_batch
        from hyff.graph import NeighborCache, energy_forces
        batch = system_batch(positions, atomic_numbers, cell, total_charge, multiplicity, self.device)
        if self.cache.signature is not None and self.cache.signature[0] != tuple(batch['positions'].shape):
            # A structure with another atom count: start a new neighbour list.
            self.cache = NeighborCache(self.model.cutoff, self.skin)
        edges = self.cache.get(batch)
        with self._spme(batch) or nullcontext():
            energy, force = energy_forces(self.model, batch, edges)
        return energy.detach(), force.detach()
