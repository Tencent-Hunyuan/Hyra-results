"""Fixed-bucket native presto batching.

Uses the single-system presto C8 CSR message kernels, TF32x3 node products, selected
LayerNorm statistics and setup-only FP32 state/first-message arithmetic.
Only final energy reduction is changed from one global sum to graph sums.
"""
from __future__ import annotations
import math
import time
from types import MethodType
import numpy as np
import torch
from torch.nn import functional as F

from hyff.batch.base import BatchedPotential, _inputs, _reference_products
from hyff.engines.native import presto as fast
from hyff.engines.native.base_adapter import PreparedGeneralNeighbors


def presto_batched_forward(self, batch, csr=None, stable_sum=False,
                          prefix_depth=None, return_prefix_depth=None):
    """presto CSR forward with a per-graph final energy reduction."""
    if csr is None or prefix_depth is not None or return_prefix_depth is not None:
        raise ValueError('Native batching requires prepared CSR and full energy')
    partition = batch['_engine_partition']
    for block in self.blocks:
        block.update_mlp.partition = partition
    positions = batch['positions']
    scalar = batch['_engine_initial_scalar']
    n = scalar.shape[0]
    blocks = list(self.blocks)
    vector = scalar.new_zeros((n, 3, self.tensor_hidden))
    tensor = scalar.new_zeros((n, 5, self.tensor_hidden))
    readout_scale = 1.0 / math.sqrt(len(blocks))
    heads = self._engine_heads
    layouts = self._engine_layouts
    if self._engine_select_two:
        fast.refresh_two(csr, positions, len(blocks), self._engine_select_tile)
    else:
        csr.refresh_subgraphs(positions, len(blocks))
    message_vector = message_tensor = node_message = None
    atomic_energy = None
    for index, block in enumerate(blocks):
        view = csr.radius_view(block.cutoff)
        table = self.radial_tables[index].coefficients
        if index == 0:
            aggregates = fast.FirstMessages8.apply(positions, view.cell, view.neighbors, view.shifts, view.counts,
                batch['_engine_first_message_atoms'], table, float(block.cutoff), float(block.switch_width),
                layouts['first_forward'], layouts['first_backward'])
        else:
            if node_message is None:
                node_message = heads[f'message{index}'](scalar)
            aggregates = fast.L2Messages8.apply(positions, view.cell, view.neighbors, view.shifts, view.counts,
                node_message, message_vector, message_tensor, table, float(block.cutoff), float(block.switch_width),
                layouts['l2_forward'], layouts['l2_backward'])
        outputs = self._engine_closures[index](scalar, vector, tensor, *aggregates)
        scalar, vector, tensor = outputs[:3]
        if len(outputs) == 5:
            message_vector, message_tensor = outputs[3:]
        dual = heads.get(f'dual{index}')
        if dual is not None:
            readout, node_message = dual(scalar)
        else:
            readout, node_message = heads[f'readout{index}'](scalar), None
        contribution = readout_scale * readout.squeeze(-1)
        atomic_energy = contribution if atomic_energy is None else atomic_energy + contribution
    atomic_energy = atomic_energy + batch['_engine_offset']
    if stable_sum:
        atomic_energy = atomic_energy.double()
    atomic_energy = atomic_energy + fast.pair_energy(positions, csr.radius_view(self.cutoff),
        batch['_engine_elements'], self.radial_tables[-1])
    return atomic_energy.new_zeros(batch['ptr'].numel() - 1).index_add(0, batch['batch'], atomic_energy)


def prepare_presto(model, precision='tf32x3', statistics='fp32', *, compiled_nodes=True,
                  expert_block=64, neighbor_reserve=1.5):
    """Apply the native presto transformations to an already loaded plain model.

    No checkpoint/config loading or learned-weight precision conversion occurs.
    The model is owned by this API and must not already have native transforms.
    """
    if precision != 'tf32x3' or statistics not in {'fp32', 'fp64'}:
        raise ValueError('Native presto batching requires tf32x3 products and fp32/fp64 statistics')
    model.eval().requires_grad_(False)
    fast.check_blocks(model)
    if model.radial_tables is not None or model.fused_messages or model.reference_chunk_size:
        raise ValueError('A fresh raw model is required; native transformations cannot be repeated')
    fast.compile_tables(model, 200)
    model.fused_message_settings = fast.SETTINGS
    model.filter_skin_neighbors = True
    model.fused_messages = model.fused = True
    fast.configure_neighbor_reserve(model, neighbor_reserve)
    model._engine_global_context = None
    fast.prepare_nodes(model, precision=precision, statistics=statistics,
        expert_block=expert_block, joined_fast=True)
    model._engine_layouts = fast.check_layouts(None)
    model._engine_select_two = True
    model._engine_select_tile = 4
    model.forward = MethodType(presto_batched_forward, model)
    fast.install_closures(model, compiled=compiled_nodes, inductor_gemm='autotune')
    return model


class _FixedPreparedNeighbors(PreparedGeneralNeighbors):
    """Reuse a fixed CSR allocation; overflow raises instead of dropping an edge."""
    def __init__(self, model, batch, edges, capacity):
        self.fixed_capacity = capacity
        super().__init__(model, batch, edges)
    @torch.no_grad()
    def _update(self, check_symmetry=False):
        if self.capacity is None:
            self.capacity = self.fixed_capacity
            device = self.edges.device
            self.neighbors = torch.full((self.n, self.capacity), -1, dtype=torch.int64, device=device)
            self.shifts = torch.zeros((self.n, self.capacity, 3), dtype=torch.int32, device=device)
            self.counts = torch.zeros(self.n, dtype=torch.int32, device=device)
        success = super()._update(check_symmetry=check_symmetry)
        if not success:
            raise ValueError('Fixed per-atom neighbor capacity exceeded')
        return success


class NativeBatchedPotential(BatchedPotential):
    """One native capture per capacity bucket with mutable Z/Q/M/routing storage.

    Currently supports presto only. Calls accept the same flat CPU arrays and
    return the same tensors as BatchedPotential. Staging, per-graph state setup
    and GPU neighbor construction are included in call cost, outside capture.
    """
    def __init__(self, model, atom_capacity, edge_capacity, graph_capacity, *, device='cuda',
                 backend='cuda_graph', mode='energy_forces', tier='presto', precision='tf32x3',
                 statistics='fp32', neighbor_capacity=128, compiled_nodes=True):
        if tier != 'presto':
            raise NotImplementedError('This native batching class supports presto only; no reference fallback')
        if torch.device(device).type != 'cuda':
            raise ValueError('The native prepared kernels require CUDA')
        if type(neighbor_capacity) is not int or neighbor_capacity < 8 or neighbor_capacity % 8:
            raise ValueError('neighbor_capacity must be a positive multiple of eight, at least eight')
        super().__init__(model, atom_capacity, edge_capacity, graph_capacity,
            device=device, backend=backend, mode=mode, max_cached_signatures=1)
        with _reference_products():
            prepare_presto(self.model, precision, statistics, compiled_nodes=compiled_nodes)
        self.tier, self.precision, self.statistics = tier, precision, statistics
        self.neighbor_capacity = neighbor_capacity
        self._fixed_entry = None

    @property
    def metadata(self):
        value = super().metadata
        value.update(execution_route='presto_native_batched', native_default_route=True,
            tier=self.tier, precision=self.precision, statistics=self.statistics,
            signature_scope='one fixed capacity bucket; Z/ptr/Q/M/species routing updated in captured storage',
            model_statistics_policy='native ' + self.statistics + ' force-time LayerNorm statistics',
            neighbor_capacity=self.neighbor_capacity,
            native_math='C8 CSR messages/TF32x3 nodes; final atom energy reduction segmented by graph',
            setup_math='per-graph FP32 Q/N,(M-1)/N, FP32 state/first-message products, unmodified first-block scalar_norm')
        return value

    def _update_edges(self, entry):
        super()._update_edges(entry)
        if entry.get('prepared') is not None:
            entry['prepared'].update_edges()

    def _stage_native(self, entry, values):
        batch, host = entry['batch'], entry['host']
        entry['real_atoms'], entry['real_graphs'] = values['real_atoms'], values['real_graphs']
        arrays = dict(atomic_numbers=values['numbers'], ptr=values['ptr'],
            batch=np.repeat(np.arange(len(values['counts']), dtype=np.int64), values['counts']),
            total_charge=values['charge'], multiplicity=values['multiplicity'])
        mask = np.zeros(self.bucket.graph_capacity + 1, np.float32)
        mask[:values['real_graphs']] = 1.
        arrays['_batch_real_graph_mask'] = mask
        with torch.no_grad():
            for name, array in arrays.items():
                host[name].copy_(torch.from_numpy(array))
                batch[name].copy_(host[name], non_blocking=True)
            schedule = fast.schedule(torch.from_numpy(values['numbers']), self.model.atomic_numbers,
                                     self.model._engine_expert_block)
            bound = batch['_engine_partition']
            for name in ('order', 'starts', 'tiles'):
                getattr(bound, name).copy_(getattr(schedule, name), non_blocking=False)
            count = (batch['ptr'][1:] - batch['ptr'][:-1]).to(self.model.embedding.weight)
            features = [batch['total_charge'].to(count) / count]
            if self.model.condition_spin:
                features.append((batch['multiplicity'].to(count) - 1.) / count)
            elif np.any(values['multiplicity'] != 1):
                raise ValueError('A charge-only model has no trained nonsinglet response')
            state = F.linear(torch.stack(features, dim=-1), self.model.state_projection.weight,
                             self.model.state_projection.bias)
            table = self.model.embedding.weight[None, :, :] + state[:, None, :]
            first = self.model.blocks[0]
            normalized = first.scalar_norm(table)
            source = first.scalar_message
            messages = F.linear(F.silu(F.linear(normalized, source[0].weight, source[0].bias)),
                                source[2].weight, source[2].bias)
            ids, z = batch['batch'], batch['atomic_numbers']
            batch['_engine_initial_scalar'].copy_(table[ids, z])
            batch['_engine_first_message_atoms'].copy_(messages[ids, z])
            elements = self.model.element_lut[z]
            batch['_engine_elements'].copy_(elements)
            batch['_engine_offset'].copy_(self.model.element_offset(z).squeeze(-1))
            if entry.get('prepared') is not None:
                entry['prepared'].elements.copy_(elements)
        self._stage(entry, values)

    def _compute(self, entry):
        with torch.set_grad_enabled(self.mode == 'energy_forces'):
            energy = entry['prepared'](entry['batch']['positions'])
            objective = (energy * entry['batch']['_batch_real_graph_mask']).sum()
            force = (-torch.autograd.grad(objective, entry['batch']['positions'])[0]
                     if self.mode == 'energy_forces' else None)
        return energy[:self.bucket.graph_capacity].detach(), force.detach() if force is not None else None

    def _make_native_entry(self, values):
        started = time.perf_counter()
        a, g = self.bucket.atom_capacity, self.bucket.graph_capacity + 1
        positions = torch.zeros((a, 3), dtype=torch.float32, device=self.device,
                                requires_grad=self.mode == 'energy_forces')
        batch = dict(positions=positions,
            atomic_numbers=torch.empty(a, dtype=torch.int64, device=self.device),
            ptr=torch.empty(g + 1, dtype=torch.int64, device=self.device),
            batch=torch.empty(a, dtype=torch.int64, device=self.device),
            total_charge=torch.empty(g, dtype=torch.float32, device=self.device),
            multiplicity=torch.empty(g, dtype=torch.int64, device=self.device),
            _batch_real_graph_mask=torch.empty(g, dtype=torch.float32, device=self.device),
            _engine_initial_scalar=torch.empty((a, self.model.config.hidden), dtype=torch.float32, device=self.device),
            _engine_first_message_atoms=torch.empty((a, self.model.blocks[0].scalar_message[2].weight.shape[0]), dtype=torch.float32, device=self.device),
            _engine_elements=torch.empty(a, dtype=torch.int64, device=self.device),
            _engine_offset=torch.empty(a, dtype=torch.float32, device=self.device))
        batch['_engine_partition'] = fast.schedule(torch.from_numpy(values['numbers']).to(self.device),
            self.model.atomic_numbers, self.model._engine_expert_block)
        host = {name: torch.empty_like(batch[name], device='cpu', pin_memory=True)
                for name in ('atomic_numbers', 'ptr', 'batch', 'total_charge', 'multiplicity', '_batch_real_graph_mask')}
        entry = dict(batch=batch, host=host, host_positions=torch.empty((a, 3), dtype=torch.float32, pin_memory=True),
            edges=torch.zeros((2, self.bucket.edge_capacity), dtype=torch.int64, device=self.device),
            real_atoms=values['real_atoms'], real_graphs=values['real_graphs'], prepared=None, graph=None)
        with _reference_products():
            self._stage_native(entry, values)
            entry['prepared'] = _FixedPreparedNeighbors(self.model, batch, entry['edges'], self.neighbor_capacity)
            capture_started = time.perf_counter()
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
        entry['preparation_host_seconds'] = time.perf_counter() - started
        entry['capture_host_seconds'] = time.perf_counter() - capture_started if entry['graph'] is not None else 0.
        return entry

    def __call__(self, positions, atomic_numbers, offsets, total_charge=None, multiplicity=None):
        values = _inputs(self.bucket, positions, atomic_numbers, offsets, total_charge, multiplicity)
        started = time.perf_counter()
        with self._lock, _reference_products():
            miss = self._fixed_entry is None
            if miss:
                self._fixed_entry = self._make_native_entry(values)
                self._entries['fixed_bucket'] = self._fixed_entry
            else:
                self._stage_native(self._fixed_entry, values)
            entry = self._fixed_entry
            if entry['graph'] is not None:
                entry['graph'].replay()
            else:
                entry['energy'], entry['force'] = self._compute(entry)
            force = entry['force']
            result = dict(energies=entry['energy'][:values['real_graphs']].clone(),
                forces=force[:values['real_atoms']].clone() if force is not None else None,
                padding_forces=force[values['real_atoms']:].clone() if force is not None else None)
            torch.cuda.current_stream(self.device).synchronize()
            self.last_call = dict(signature_cache_miss=miss, fixed_bucket_reused=not miss,
                signature_preparation_host_seconds=entry['preparation_host_seconds'] if miss else 0.,
                signature_capture_host_seconds=entry['capture_host_seconds'] if miss else 0.,
                call_host_seconds=time.perf_counter() - started,
                real_atoms=values['real_atoms'], real_graphs=values['real_graphs'],
                active_edges=entry['edge_count'], padding_atoms=self.bucket.atom_capacity - values['real_atoms'])
            result['metadata'] = self.metadata
            return result

    def clear(self):
        super().clear()
        self._fixed_entry = None
