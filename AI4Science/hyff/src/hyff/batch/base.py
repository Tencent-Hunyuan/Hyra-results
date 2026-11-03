"""Nonperiodic multi-molecule reference inference with explicit fixed capacities.

This path evaluates the plain PyTorch model with IEEE FP32 products and the
model's own LayerNorm; the native TF32x3 route is hyff.batch.native. CUDA
capture is cached by exact discrete input signature; preparation/capture cost
must be included or amortized explicitly when reporting dataset throughput.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import threading
import time
from types import MethodType

import numpy as np
import torch


@dataclass(frozen=True)
class BatchBucket:
    """Fixed tensor capacities; graph_capacity counts real molecules only.

    At least one atom is reserved for a separate dummy graph. Each unused real
    graph slot also receives one dummy atom so state denominators stay positive.
    """
    atom_capacity: int
    edge_capacity: int
    graph_capacity: int

    def __post_init__(self):
        for name in ('atom_capacity', 'edge_capacity', 'graph_capacity'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if self.atom_capacity < self.graph_capacity + 1:
            raise ValueError('Atom capacity must reserve a nonempty separate dummy graph')


def _array(value, dtype, shape, name):
    if torch.is_tensor(value):
        if value.device.type != 'cpu':
            raise ValueError(name + ' must be a CPU input for pinned asynchronous staging')
        value = value.detach().numpy()
    value = np.asarray(value)
    if value.dtype != np.dtype(dtype) or (shape is not None and value.shape != shape):
        raise ValueError(name + ' has an invalid dtype or shape')
    if value.dtype.kind == 'f' and not np.isfinite(value).all():
        raise ValueError(name + ' must be finite')
    return np.ascontiguousarray(value)


def _inputs(bucket, positions, numbers, offsets, charge, multiplicity):
    positions = _array(positions, np.float32, None, 'positions')
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError('positions must have shape (N,3)')
    n = len(positions)
    numbers = _array(numbers, np.int64, (n,), 'atomic_numbers')
    offsets = _array(offsets, np.int64, None, 'offsets')
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != n:
        raise ValueError('offsets must start at zero and end at the real atom count')
    counts = np.diff(offsets)
    graphs = len(counts)
    if (counts <= 0).any() or graphs > bucket.graph_capacity:
        raise ValueError('Molecules must be nonempty and fit graph capacity')
    if n + bucket.graph_capacity - graphs + 1 > bucket.atom_capacity:
        raise ValueError('Real atoms and nonempty dummy graph slots exceed atom capacity')
    if ((numbers < 1) | (numbers > 118)).any():
        raise ValueError('Atomic numbers must lie in 1..118')
    charge = _array(np.zeros(graphs, np.float32) if charge is None else charge,
                    np.float32, (graphs,), 'total_charge')
    multiplicity = _array(np.ones(graphs, np.int64) if multiplicity is None else multiplicity,
                          np.int64, (graphs,), 'multiplicity')
    if (multiplicity < 1).any():
        raise ValueError('Multiplicity M=2S+1 must be a positive integer')
    dummy_counts = np.ones(bucket.graph_capacity - graphs + 1, dtype=np.int64)
    dummy_counts[-1] = bucket.atom_capacity - n - (len(dummy_counts) - 1)
    padded_counts = np.concatenate((counts, dummy_counts))
    ptr = np.concatenate((np.zeros(1, np.int64), np.cumsum(padded_counts)))
    z = np.ones(bucket.atom_capacity, np.int64)
    z[:n] = numbers
    q = np.zeros(bucket.graph_capacity + 1, np.float32)
    m = np.ones(bucket.graph_capacity + 1, np.int64)
    q[:graphs], m[:graphs] = charge, multiplicity
    digest = hashlib.sha256()
    for value in (z, ptr, q, m, np.asarray([graphs], np.int64)):
        digest.update(value.tobytes(order='C'))
    return dict(positions=positions, numbers=z, ptr=ptr, charge=q, multiplicity=m,
                counts=padded_counts, real_atoms=n, real_graphs=graphs, signature=digest.hexdigest())


@contextmanager
def _reference_products():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


@contextmanager
def _prepared_initial_scalar(model):
    # Species and global-state preparation belongs to the exact signature,
    # outside capture. The cached scalar is coordinate independent.
    original = getattr(model, 'initial_scalar', None)
    if original is None:
        yield
        return
    def initial_scalar(self, batch):
        return batch['_batch_reference_initial_scalar']
    model.initial_scalar = MethodType(initial_scalar, model)
    try:
        yield
    finally:
        model.initial_scalar = original


class BatchedPotential:
    """Reference E or E+F on disconnected nonperiodic molecules.

    CPU inputs use FP32 positions/charges and int64 numbers/offsets/multiplicity.
    Returned tensors are independent copies on ``device``. ``metadata`` exposes
    cache misses and preparation cost; callers synchronize for end-to-end timing.
    CUDA capture signatures include atomic numbers, offsets, Q/M and real graph
    count, so arbitrary new dataset compositions can require new captures.
    """
    def __init__(self, model, atom_capacity, edge_capacity, graph_capacity, *,
                 device='cuda', backend='cuda_graph', mode='energy_forces',
                 max_cached_signatures=16):
        if backend not in {'eager', 'cuda_graph'} or mode not in {'energy', 'energy_forces'}:
            raise ValueError('backend must be eager/cuda_graph and mode energy/energy_forces')
        if type(max_cached_signatures) is not int or max_cached_signatures < 1:
            raise ValueError('max_cached_signatures must be a positive integer')
        self.bucket = BatchBucket(atom_capacity, edge_capacity, graph_capacity)
        self.device = torch.device(device)
        if backend == 'cuda_graph' and self.device.type != 'cuda':
            raise ValueError('CUDA graph capture requires a CUDA device')
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.backend, self.mode = backend, mode
        self.max_cached_signatures = max_cached_signatures
        self._entries = {}
        self._lock = threading.Lock()
        self.last_call = {}

    @classmethod
    def from_weights(cls, weights, *args, device='cuda', **kwargs):
        from hyff.weights import load_model
        model, _ = load_model(weights, device)
        return cls(model, *args, device=device, **kwargs)

    @property
    def metadata(self):
        return dict(execution_route='raw_reference_ieee_fp32', native_default_route=False,
            model_statistics_policy='model LayerNorm; no replacement',
            mode=self.mode, backend=self.backend, nonperiodic_only=True,
            atom_capacity=self.bucket.atom_capacity, edge_capacity=self.bucket.edge_capacity,
            graph_capacity=self.bucket.graph_capacity, cached_signatures=len(self._entries),
            capture_scope='model E or E+F only; GPU neighbor build and pinned host staging outside capture',
            signature_scope='exact Z/offsets/Q/M/real graph count; charge/spin/species changes may recapture',
            padding='separate nonempty dummy graphs; edges built for real atoms only; only real E differentiated',
            timing='caller must include staging/neighbors/signature preparation and synchronize for end-to-end throughput',
            last_call=dict(self.last_call))

    def _stage(self, entry, values):
        with torch.no_grad():
            entry['host_positions'].zero_()
            entry['host_positions'][:values['real_atoms']].copy_(torch.from_numpy(values['positions']))
            entry['batch']['positions'].copy_(entry['host_positions'], non_blocking=self.device.type == 'cuda')
        self._update_edges(entry)

    def _update_edges(self, entry):
        from hyff.graph import cell_list_edges, dense_reference_edges
        batch, n, graphs = entry['batch'], entry['real_atoms'], entry['real_graphs']
        physical = dict(positions=batch['positions'][:n], atomic_numbers=batch['atomic_numbers'][:n],
                        batch=batch['batch'][:n], ptr=batch['ptr'][:graphs + 1])
        edges = (cell_list_edges(physical, self.model.cutoff) if self.device.type == 'cuda'
                 else dense_reference_edges(physical, self.model.cutoff))
        count = edges.shape[1]
        if count > self.bucket.edge_capacity:
            raise ValueError('Edge capacity exceeded: required ' + str(count) + ', capacity ' + str(self.bucket.edge_capacity))
        with torch.no_grad():
            entry['edges'].zero_()
            entry['edges'][:, :count].copy_(edges)
        entry['edge_count'] = count

    def _compute(self, entry):
        positions = entry['batch']['positions']
        with torch.set_grad_enabled(self.mode == 'energy_forces'):
            all_energy = self.model.energy(entry['batch'], entry['edges'])
            energy = all_energy[:entry['real_graphs']]
            force = (-torch.autograd.grad(energy.sum(), positions)[0]
                     if self.mode == 'energy_forces' else None)
        return energy.detach(), force.detach() if force is not None else None

    def _make_entry(self, values):
        from hyff.models.element_model import prepare_batch
        start = time.perf_counter()
        a = self.bucket.atom_capacity
        positions = torch.zeros((a, 3), dtype=torch.float32, device=self.device,
                                requires_grad=self.mode == 'energy_forces')
        counts = torch.as_tensor(values['counts'], device=self.device)
        batch = dict(positions=positions,
            atomic_numbers=torch.as_tensor(values['numbers'], device=self.device),
            ptr=torch.as_tensor(values['ptr'], device=self.device),
            batch=torch.repeat_interleave(torch.arange(len(counts), device=self.device), counts),
            n_atoms=counts.to(torch.float32),
            total_charge=torch.as_tensor(values['charge'], device=self.device),
            multiplicity=torch.as_tensor(values['multiplicity'], device=self.device))
        batch = prepare_batch(batch)
        if hasattr(self.model, 'initial_scalar'):
            with torch.no_grad():
                batch['_batch_reference_initial_scalar'] = self.model.initial_scalar(batch).detach()
        entry = dict(batch=batch, edges=torch.zeros((2, self.bucket.edge_capacity), dtype=torch.int64, device=self.device),
            host_positions=torch.empty((a, 3), dtype=torch.float32, pin_memory=self.device.type == 'cuda'),
            real_atoms=values['real_atoms'], real_graphs=values['real_graphs'], graph=None)
        self._stage(entry, values)
        capture_start = time.perf_counter()
        with _reference_products(), _prepared_initial_scalar(self.model):
            if self.backend == 'cuda_graph':
                current = torch.cuda.current_stream(self.device)
                warmup = torch.cuda.Stream(device=self.device)
                warmup.wait_stream(current)
                with torch.cuda.stream(warmup):
                    for _ in range(2):
                        self._compute(entry)
                current.wait_stream(warmup)
                torch.cuda.synchronize(self.device)
                entry['graph'] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(entry['graph']):
                    entry['energy'], entry['force'] = self._compute(entry)
            else:
                entry['energy'], entry['force'] = self._compute(entry)
        entry['preparation_host_seconds'] = time.perf_counter() - start
        entry['capture_host_seconds'] = time.perf_counter() - capture_start if entry['graph'] is not None else 0.
        return entry

    def __call__(self, positions, atomic_numbers, offsets, total_charge=None, multiplicity=None):
        values = _inputs(self.bucket, positions, atomic_numbers, offsets, total_charge, multiplicity)
        started = time.perf_counter()
        with self._lock:
            entry = self._entries.get(values['signature'])
            miss = entry is None
            if miss:
                if len(self._entries) >= self.max_cached_signatures:
                    raise ValueError('Signature cache full; use another bucket or clear() and report new preparation cost')
                entry = self._make_entry(values)
                self._entries[values['signature']] = entry
            else:
                self._stage(entry, values)
            if entry['graph'] is not None:
                entry['graph'].replay()
            else:
                with _reference_products(), _prepared_initial_scalar(self.model):
                    entry['energy'], entry['force'] = self._compute(entry)
            energy = entry['energy'].clone()
            force = entry['force']
            result = dict(energies=energy,
                forces=force[:values['real_atoms']].clone() if force is not None else None,
                padding_forces=force[values['real_atoms']:].clone() if force is not None else None)
            # Protect pinned staging storage from a subsequent host overwrite.
            # End-to-end calls are deliberately synchronous; copies are pinned/nonblocking.
            if self.device.type == 'cuda':
                torch.cuda.current_stream(self.device).synchronize()
            self.last_call = dict(signature_cache_miss=miss, signature=values['signature'],
                signature_preparation_host_seconds=entry['preparation_host_seconds'] if miss else 0.,
                signature_capture_host_seconds=entry['capture_host_seconds'] if miss else 0.,
                call_host_seconds=time.perf_counter() - started,
                real_atoms=values['real_atoms'], real_graphs=values['real_graphs'],
                active_edges=entry['edge_count'], padding_atoms=self.bucket.atom_capacity - values['real_atoms'])
            result['metadata'] = self.metadata
            return result

    def clear(self):
        with self._lock:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self._entries.clear()
