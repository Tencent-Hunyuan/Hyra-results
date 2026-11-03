"""Public flat-collection prediction with bounded, shared-model native buckets.

Input charges and multiplicities are explicit per molecule. Native batching
uses each model's default precision and statistics; reference means IEEE FP32
products. largo evaluates its latent-charge term over packed within-molecule
pairs. A failed native call never falls back to another route. Importing this module requires only NumPy.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import re
import sys
import threading
import time

import numpy as np

TIERS = ('presto', 'vivace', 'moderato', 'andante', 'largo')


class CapacityClassLimitError(RuntimeError):
    """A predictor holds at most four capacity specs over its lifetime."""


class CollectionCapacityOverflowError(RuntimeError):
    """A known fixed-buffer overflow, attributed to its packed input batch."""
    def __init__(self, *, tier, molecule_indices, capacities, overflow_kind):
        self.tier = tier
        self.packed_molecule_indices = tuple(int(i) for i in molecule_indices)
        self.molecule_indices = tuple(sorted(self.packed_molecule_indices))
        self.overflow_kind = overflow_kind
        for name in ('edge_capacity', 'neighbor_capacity', 'atom_capacity', 'graph_capacity'):
            setattr(self, name, int(capacities[name]))
        self.expert_capacities = dict(capacities.get('expert_capacities', {}))
        advice = ('Rerun with larger edge_capacity or larger neighbor_capacity; '
                  'neighbor_capacity must be a positive multiple of 8. '
                  'automatic planning reserves are 10% for edges and 5% for degree.')
        if overflow_kind == 'expert_rows':
            advice = ('Replan the full intended collection with a fresh predictor to recompute '
                      'per-element expert rows; edge/neighbor overrides do not enlarge expert rows.')
        super().__init__(
            f'{tier} {overflow_kind} capacity overflow in packed batch with original input molecule '
            f'indices {list(self.molecule_indices)}; a unique culprit molecule is not identified. '
            f'Current edge_capacity={self.edge_capacity}, neighbor_capacity={self.neighbor_capacity}, '
            f'atom_capacity={self.atom_capacity}, graph_capacity={self.graph_capacity}. {advice}')


def _capacity_overflow_kind(error):
    """Classify only the known capacity-overflow errors of the batch modules; never classify OOM."""
    if type(error) is ValueError and len(error.args) == 1 and type(error.args[0]) is str:
        trace = error.__traceback__
        while trace is not None and trace.tb_next is not None:
            trace = trace.tb_next
        if trace is None:
            return None
        origin = (trace.tb_frame.f_globals.get('__name__'), trace.tb_frame.f_code.co_name)
        message = error.args[0]
        if (origin == ('hyff.batch.base', '_update_edges') and
                re.fullmatch(r'Edge capacity exceeded: required [0-9]+, capacity [0-9]+', message)):
            return 'global_edges'
        if (origin == ('hyff.batch.native_presto', '_update') and
                message == 'Fixed per-atom neighbor capacity exceeded'):
            return 'neighbor_slots'
        if ((origin == ('hyff.batch.native', '__call__') and
                message == 'andante fixed per-element expert row capacity exceeded') or
                (origin == ('hyff.batch.native_efm', '_stage_andante_partition') and
                 message == 'andante per-element expert row capacity exceeded')):
            return 'expert_rows'
    modules = ('hyff.engines.capture.efm_port', 'hyff.engines.capture.efm_port_dynamic')
    for name in modules:
        expected = getattr(sys.modules.get(name), 'EFMOverflow', None)
        if (expected is not None and type(error) is expected and len(error.args) == 1 and
                type(error.args[0]) is str and error.args[0] == 'EFM fixed buffer overflow; force rejected'):
            return 'efm_buffers'
    return None


def _integers(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'iu':
        raise ValueError(name + ' must contain integers')
    if array.size and array.dtype.kind == 'u' and int(np.max(array)) > int(np.iinfo(np.int64).max):
        raise ValueError(name + ' exceeds int64 range')
    return array.astype(np.int64, copy=True)


def _floats(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(name + ' must contain finite real numbers')
    with np.errstate(over='ignore', invalid='ignore'):
        result = array.astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(name + ' exceeds finite FP32 range')
    return result


def validate_collection(positions, numbers, offsets, total_charge, multiplicity):
    xyz, z, ptr = _floats(positions, 'positions'), _integers(numbers, 'numbers'), _integers(offsets, 'offsets')
    q, m = _floats(total_charge, 'total_charge'), _integers(multiplicity, 'multiplicity')
    if ptr.ndim != 1 or len(ptr) < 2 or ptr[0] != 0 or not np.all(ptr[1:] > ptr[:-1]):
        raise ValueError('offsets must start at zero and strictly increase')
    n, graphs = int(ptr[-1]), len(ptr) - 1
    if xyz.shape != (n, 3) or z.shape != (n,) or q.shape != (graphs,) or m.shape != (graphs,):
        raise ValueError('Flat geometry/state shapes do not match offsets')
    if np.any(z < 1) or np.any(z > 118) or np.any(m < 1):
        raise ValueError('Atomic numbers must be 1..118 and multiplicities at least 1')
    return dict(positions=xyz, numbers=z, offsets=ptr, total_charge=q, multiplicity=m)


def load_collection_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return validate_collection(*(data[key] for key in
            ('positions', 'numbers', 'offsets', 'total_charge', 'multiplicity')))


def plan_collection(offsets, *, atom_capacity=8192, graph_capacity=512, bucket_policy='size_bands'):
    ptr = _integers(offsets, 'offsets')
    if ptr.ndim != 1 or len(ptr) < 2 or ptr[0] != 0 or not np.all(ptr[1:] > ptr[:-1]):
        raise ValueError('Invalid collection offsets')
    if (type(atom_capacity) is not int or type(graph_capacity) is not int or
            graph_capacity < 1 or atom_capacity < graph_capacity + 1):
        raise ValueError('Positive capacities must reserve nonempty dummy graphs')
    if bucket_policy not in {'size_bands', 'single'}:
        raise ValueError('bucket_policy must be size_bands or single')
    sizes = ptr[1:] - ptr[:-1]
    budget = atom_capacity - graph_capacity - 1
    if np.any(sizes - 1 > budget):
        raise ValueError('A molecule exceeds the atom budget including reserved dummy graphs')
    bands = np.searchsorted([50, 100, 200], sizes, side='right') if bucket_policy == 'size_bands' else np.zeros(len(sizes), np.int64)
    schedule, specs = [], []
    for band in sorted(set(bands.tolist())):
        indices = np.flatnonzero(bands == band)
        indices = indices[np.argsort(sizes[indices], kind='stable')]
        batches, pending, cost = [], [], 0
        for index in indices:
            next_cost = int(sizes[index]) - 1
            if pending and (len(pending) == graph_capacity or cost + next_cost > budget):
                batches.append(pending)
                pending, cost = [], 0
            pending.append(int(index))
            cost += next_cost
        if pending:
            batches.append(pending)
        actual_graphs = max(map(len, batches))
        needed = max(sum(int(sizes[i]) for i in batch) + actual_graphs - len(batch) + 1 for batch in batches)
        atoms = min(atom_capacity, ((needed + 31) // 32) * 32)
        bucket = len(specs)
        specs.append(dict(atom_capacity=atoms, graph_capacity=actual_graphs, size_band=int(band)))
        schedule.extend(dict(bucket=bucket, indices=batch) for batch in batches)
    return dict(buckets=specs, schedule=schedule, requested_atom_capacity=atom_capacity,
        requested_graph_capacity=graph_capacity, bucket_policy=bucket_policy,
        dummy_constraint='sum(N_i-1)<=atom_capacity-graph_capacity-1')


def geometry_capacities(values, plan, cutoff, *, edge_capacity=0, neighbor_capacity=0, device='cpu'):
    """Dispatch capacity counting explicitly; CUDA errors propagate unchanged."""
    options = dict(edge_capacity=edge_capacity, neighbor_capacity=neighbor_capacity)
    if str(device).startswith('cuda'):
        from hyff.batch.geometry_gpu import geometry_capacities as count
        return count(values, plan, cutoff, device=device, **options)
    from hyff.batch.geometry_axiswise import geometry_capacities as count
    return count(values, plan, cutoff, **options)


def _host(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    return np.asarray(value)


class CollectionPotential:
    """Own a fresh raw model; predict many nonperiodic molecules in input order.

    Use from_weights to include model loading in first-call cold metadata.
    Native defaults: presto/vivace/moderato TF32x3 products with FP32 norms;
    largo TF32x3/FP64; andante FP32/FP64. largo uses packed within-molecule
    latent-charge pairs. Parameters are frozen.
    Model/capture storage is reused on later calls. Calls are serialized.
    One bucket is the default for every tier; size_bands is an explicit option.
    Auto requires allow_reference_numerics=True because it can select IEEE
    reference products. Explicit reference routing is already an opt-in.
    CUDA capacity planning uses GPU counts and propagates every error; CPU
    reference planning uses the axiswise NumPy counter.
    """
    def __init__(self, model, *, tier, device='cuda', route='native', backend='cuda_graph',
                 mode='energy_forces', atom_capacity=0, graph_capacity=0,
                 edge_capacity=0, neighbor_capacity=0, bucket_policy=None,
                 precision=None, statistics=None, compiled_nodes=False, auto_policy=None,
                 allow_reference_numerics=False):
        if tier not in TIERS or route not in {'native', 'reference', 'auto'}:
            raise ValueError('Unknown tier or route')
        if backend not in {'eager', 'cuda_graph'} or mode not in {'energy', 'energy_forces'}:
            raise ValueError('Invalid backend or mode')
        for name, value in (('atom_capacity', atom_capacity), ('graph_capacity', graph_capacity),
                            ('edge_capacity', edge_capacity), ('neighbor_capacity', neighbor_capacity)):
            if type(value) is not int or value < 0:
                raise ValueError(name + ' must be a nonnegative integer')
        if neighbor_capacity % 8:
            raise ValueError('neighbor_capacity must be zero or a positive multiple of eight')
        if bucket_policy not in {None, 'size_bands', 'single'} or type(compiled_nodes) is not bool:
            raise ValueError('Invalid bucket policy or compiled_nodes flag')
        if compiled_nodes:
            raise ValueError('compiled_nodes=True is unsupported for collection inference; use compiled_nodes=False')
        if bucket_policy is None:
            bucket_policy = 'single'
        if route == 'native' and not str(device).startswith('cuda'):
            raise ValueError('Native batching requires CUDA')
        if route == 'reference' and backend == 'cuda_graph' and not str(device).startswith('cuda'):
            raise ValueError('Reference CUDA graph capture requires CUDA')
        if route == 'reference' and (precision is not None or statistics is not None):
            raise ValueError('Reference uses IEEE FP32 products and model norms; omit overrides')
        if route != 'reference':
            expected_p, expected_s = ('fp32' if tier == 'andante' else 'tf32x3'), ('fp32' if tier in {'presto', 'vivace', 'moderato'} else 'fp64')
            allowed_statistics = {None, expected_s}
            if tier in {'vivace', 'moderato'}:
                allowed_statistics.add('fp64')
            if precision not in {None, expected_p} or statistics not in allowed_statistics:
                raise ValueError('Precision/statistics differ from the declared tier policy')
        if type(allow_reference_numerics) is not bool:
            raise ValueError('allow_reference_numerics must be a bool')
        if route == 'auto':
            if not allow_reference_numerics:
                raise ValueError('Auto includes IEEE reference numerics; set allow_reference_numerics=True to opt in')
            if (not isinstance(auto_policy, dict) or set(auto_policy) != {'reference_max_molecules', 'capture_min_molecules'} or
                    any(type(value) is not int for value in auto_policy.values()) or
                    not 0 < auto_policy['reference_max_molecules'] < auto_policy['capture_min_molecules']):
                raise ValueError('Auto requires explicit positive reference/capture crossover thresholds')
            auto_policy = dict(auto_policy)
        elif auto_policy is not None:
            raise ValueError('auto_policy is only valid for route auto')
        self.model, self.tier, self.device = model, tier, device
        self.route, self.backend, self.mode = route, backend, mode
        self.atom_capacity = atom_capacity or (2048 if tier == 'largo' else 8192)
        self.graph_capacity = graph_capacity or (64 if tier == 'largo' else 512)
        if self.atom_capacity < self.graph_capacity + 1:
            raise ValueError('Atom capacity must reserve nonempty dummy graphs')
        self.edge_capacity, self.neighbor_capacity, self.bucket_policy = edge_capacity, neighbor_capacity, bucket_policy
        self.precision, self.statistics = precision, statistics
        self.compiled_nodes, self.auto_policy = compiled_nodes, auto_policy
        self.allow_reference_numerics = allow_reference_numerics
        self._pool, self._runners, self._selected = None, {}, None
        self._lock, self._load_seconds, self._load_pending = threading.RLock(), None, False

    @classmethod
    def from_weights(cls, weights, **options):
        result = cls(None, **options)
        result._load_weights(weights)
        return result

    def _load_weights(self, weights):
        from hyff.weights import load_model
        started = time.perf_counter()
        self.model, _ = load_model(weights, self.device)
        self._load_seconds, self._load_pending = time.perf_counter() - started, True

    def _execution(self, molecules):
        if self.route != 'auto':
            return self.route, self.backend
        if molecules <= self.auto_policy['reference_max_molecules']:
            if not self.allow_reference_numerics:
                raise ValueError('Auto reference numerics require allow_reference_numerics=True')
            return 'reference', 'eager'
        return 'native', ('cuda_graph' if molecules >= self.auto_policy['capture_min_molecules'] else 'eager')

    def _preflight(self, values):
        plan = plan_collection(values['offsets'], atom_capacity=self.atom_capacity,
            graph_capacity=self.graph_capacity, bucket_policy=self.bucket_policy)
        selected = self._execution(len(values['total_charge']))
        if self._selected is not None and selected != self._selected:
            raise ValueError('Auto selection changed; use a fresh raw model for the new numerical route')
        if selected[0] == 'reference' and (self.precision is not None or self.statistics is not None):
            raise ValueError('Selected reference route requires omitted precision/statistics overrides')
        if selected[0] == 'native' and not str(self.device).startswith('cuda'):
            raise ValueError('Selected native route requires CUDA')
        return plan, selected

    def predict(self, positions, atomic_numbers, offsets, total_charge, multiplicity):
        started = time.perf_counter()
        values = validate_collection(positions, atomic_numbers, offsets, total_charge, multiplicity)
        with self._lock:
            plan, selected = self._preflight(values)
            specs = geometry_capacities(values, plan, self.model.cutoff,
                edge_capacity=self.edge_capacity, neighbor_capacity=self.neighbor_capacity, device=self.device)
            requested = []
            for spec in specs:
                options = {key: spec[key] for key in ('atom_capacity', 'edge_capacity', 'graph_capacity')}
                if selected[0] == 'native':
                    options['neighbor_capacity'] = spec['neighbor_capacity']
                    if self.tier == 'andante':
                        options['expert_capacities'] = spec['expert_capacities']
                    if self.tier == 'largo':
                        options['pair_capacity'] = spec['pair_capacity']
                key = json.dumps(options, sort_keys=True)
                requested.append((key, options))
            if len(set(self._runners).union(key for key, _ in requested)) > 4:
                raise CapacityClassLimitError(
                    'CollectionPotential supports at most four lifetime capacity specs; create a fresh instance')
            self._selected = selected
            runners = []
            for key, options in requested:
                if key not in self._runners:
                    if selected[0] == 'reference':
                        from hyff.batch.reference import ReferenceBatchedPotential
                        runner = ReferenceBatchedPotential(self.model, **options, device=self.device,
                            backend=selected[1], mode=self.mode, max_cached_signatures=1)
                    elif self._pool is None:
                        if self.tier == 'largo':
                            from hyff.batch.latent_charge_pool import NativeBucketPool
                        else:
                            from hyff.batch.pool import NativeBucketPool
                        self._pool = NativeBucketPool(self.model, [options], tier=self.tier,
                            device=self.device, backend=selected[1], mode=self.mode,
                            precision=self.precision or ('fp32' if self.tier == 'andante' else 'tf32x3'),
                            statistics=self.statistics or ('fp32' if self.tier in {'presto', 'vivace', 'moderato'} else 'fp64'),
                            compiled_nodes=self.compiled_nodes)
                        runner = self._pool.buckets[0]
                    else:
                        runner = self._pool.add_bucket(**options)
                    self._runners[key] = runner
                runners.append(self._runners[key])
            ptr = values['offsets']
            energies = np.empty(len(ptr) - 1, np.float64)
            forces = np.empty_like(values['positions']) if self.mode == 'energy_forces' else None
            calls = []
            for entry in plan['schedule']:
                indices = entry['indices']
                xyz = np.concatenate([values['positions'][ptr[i]:ptr[i + 1]] for i in indices])
                z = np.concatenate([values['numbers'][ptr[i]:ptr[i + 1]] for i in indices])
                local = np.r_[np.int64(0), np.cumsum([int(ptr[i + 1] - ptr[i]) for i in indices], dtype=np.int64)]
                runner = runners[entry['bucket']]
                try:
                    output = runner(xyz, z, local, values['total_charge'][indices], values['multiplicity'][indices])
                except Exception as error:
                    overflow_kind = _capacity_overflow_kind(error)
                    if overflow_kind is None:
                        raise
                    raise CollectionCapacityOverflowError(tier=self.tier, molecule_indices=indices,
                        capacities=specs[entry['bucket']], overflow_kind=overflow_kind) from error
                energy = _host(output['energies'])
                if energy.shape != (len(indices),) or not np.isfinite(energy).all():
                    raise FloatingPointError('Invalid or nonfinite batch energies')
                energies[indices] = energy
                if forces is not None:
                    force, padding = _host(output['forces']), _host(output['padding_forces'])
                    expected_padding = specs[entry['bucket']]['atom_capacity'] - len(xyz)
                    if (force.shape != xyz.shape or padding.shape != (expected_padding, 3) or
                            not np.isfinite(force).all() or not np.isfinite(padding).all() or np.count_nonzero(padding)):
                        raise FloatingPointError('Nonfinite forces or nonzero padding forces')
                    for local_index, original_index in enumerate(indices):
                        forces[ptr[original_index]:ptr[original_index + 1]] = force[local[local_index]:local[local_index + 1]]
                elif output.get('forces') is not None or output.get('padding_forces') is not None:
                    raise ValueError('Energy-only route unexpectedly evaluated forces')
                calls.append(dict(bucket=entry['bucket'], indices=indices, **runner.last_call))
                if selected[0] == 'reference':
                    runner.clear()
            load = self._load_seconds if self._load_pending else 0.
            self._load_pending = False
            metadata = dict(schema='hyff-collection-prediction', tier=self.tier, requested_route=self.route,
                route=selected[0], backend=selected[1], mode=self.mode, molecules=len(energies), atoms=len(forces) if forces is not None else len(values['positions']),
                auto_policy=self.auto_policy, allow_reference_numerics=self.allow_reference_numerics,
                geometry_capacity_backend='gpu' if str(self.device).startswith('cuda') else 'cpu_axiswise',
                bucket_policy=self.bucket_policy, bucket_specs=specs,
                calls=calls, execution=[runner.metadata for runner in runners],
                model_load_host_seconds=load, model_load_measured=self._load_seconds is not None,
                prediction_host_seconds=time.perf_counter() - started,
                includes_geometry_planning_and_every_first_bucket_capture=True,
                latent_charge_pairs='packed within-molecule pairs' if self.tier == 'largo' and selected[0] == 'native' else None,
                reference_signature_cache='cleared between scheduled calls' if selected[0] == 'reference' else None)
            metadata['end_to_end_host_seconds'] = metadata['prediction_host_seconds'] + (load or 0.)
            return dict(energies=energies, forces=forces, offsets=ptr.copy(), metadata=metadata)

    def clear(self):
        with self._lock:
            for runner in self._runners.values():
                runner.clear()


def predict_collection(weights_or_model, positions, atomic_numbers, offsets, total_charge, multiplicity, **options):
    """Predict a flat collection from a weight path or a freshly loaded model, which the predictor then owns."""
    started = time.perf_counter()
    predictor = CollectionPotential(None, **options)
    values = validate_collection(positions, atomic_numbers, offsets, total_charge, multiplicity)
    predictor._preflight(values)
    if isinstance(weights_or_model, (str, Path)):
        predictor._load_weights(weights_or_model)
    else:
        predictor.model = weights_or_model
    try:
        result = predictor.predict(values['positions'], values['numbers'], values['offsets'],
            values['total_charge'], values['multiplicity'])
    finally:
        predictor.clear()
    result['metadata']['end_to_end_host_seconds'] = time.perf_counter() - started
    result['metadata']['end_to_end_scope'] = 'public API validation, load, geometry planning, prediction, and clear'
    return result
