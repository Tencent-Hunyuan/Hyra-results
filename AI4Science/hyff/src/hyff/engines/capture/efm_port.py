"""Instance-local EFM installation; reuses the efm_engine algebra (imported as ``sibling``) and reads no files.

The CPU mode is for parity testing; the GPU path is captured in a CUDA graph.
"""
from __future__ import annotations

import copy
import math
from types import MethodType

import torch

from hyff.engines.capture import efm_engine as sibling

from hyff.engines.capture.compaction import compact, gather_with_sentinel


class EFMOverflow(RuntimeError):
    def __init__(self, edge_excess, slot_excess):
        self.edge_excess, self.slot_excess = int(edge_excess), int(slot_excess)
        super().__init__('EFM fixed buffer overflow; force rejected')


def comparison_error(actual, expected):
    """Finite relative error, including unused-gradient agreement."""
    if actual is None or expected is None:
        return 0.0 if actual is expected else math.inf
    if actual.shape != expected.shape:
        return math.inf
    if actual.numel() == 0:
        return 0.0
    if not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
        return math.inf
    return float((actual - expected).abs().max()) / max(float(expected.abs().max()), 1e-30)


def describe_efms(model):
    efms, radii = {}, {}
    for index, block in enumerate(model.blocks):
        efm = getattr(block, 'efm', None)
        if efm is None:
            continue
        if (efm.aggregation != 'sum' or efm.frame != 'projector' or efm.complex_structure
                or getattr(efm, 'receiver_in', None) is not None):
            raise ValueError('Only sum-aggregated projector EFM without receiver input is supported')
        efms[index] = efm
        radii[index] = float(getattr(efm, 'edge_cutoff', model.layer_cutoffs[index]))
    if not efms:
        raise ValueError('Model has no EFM modules')
    return efms, dict(blocks=sorted(efms), radii_A=radii,
        message_scale={i: float(e.message_scale) for i, e in efms.items()},
        switch_width=float(model.config.switch_width),
        channels={i: e.channels for i, e in efms.items()},
        m_blocks={i: list(e.m_blocks) for i, e in efms.items()},
        layers={i: int(getattr(e, 'n_layers', 1)) for i, e in efms.items()},
        parameters=sum(p.numel() for e in efms.values() for p in e.parameters()))


class EFMState(sibling.EFMState):
    """Sibling rotated algebra with checked compaction and finite self-checks."""
    def __init__(self, model, efms, info, *, slots=64, edge_reserve=1.5, compile_nodes=True):
        if type(slots) is not int or slots < 1:
            raise ValueError('slots must be a positive integer')
        if not math.isfinite(edge_reserve) or edge_reserve < 1:
            raise ValueError('edge_reserve must be finite and at least one')
        self.model, self.efms, self.info = model, efms, info
        self.device = next(model.parameters()).device
        self.efm_precision = 'fp32'
        self.slots, self.reserve = slots, float(edge_reserve)
        self.compile_nodes = bool(compile_nodes)
        self.enabled = True
        self.batch = self.csr = None
        self.geometry, self.term, self.pad_cache = {}, {}, {}
        self.capacity = None
        self.flags = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.centers, self.gamma = model.rbf.centers.detach(), model.rbf.gamma.detach()
        self.rbf_cutoff = float(model.rbf.cutoff)
        self.width, self.padding = float(info['switch_width']), model.padding_displacement.detach()
        self.stats = dict(csr_geometry_calls=0, eager_geometry_calls=0, buckets=[],
                         initial_edges=None, initial_max_count=None, buffer_growths=0)
        self.compiled_geometry = self.compile(sibling._geometry)
        self.compiled_term, self.term_backend = {}, {}
        for index, efm in efms.items():
            if self.compile_nodes:
                self.compiled_term[index], self.term_backend[index] = self._checked_compile(index, efm)
            else:
                self.compiled_term[index] = self._plain_term(efm)
                self.term_backend[index] = 'eager_blocked_cpu_validation'
        self.efm_finish, self.base_finish, self.original_forward = {}, {}, {}

    def compile(self, function):
        return torch.compile(function, **sibling.COMPILE_OPTIONS) if self.compile_nodes else function

    @staticmethod
    def _plain_term(efm):
        def term(normalized, vector, tensor, rbf, R, D, envelope, src, dst):
            return sibling._efm_term(efm, normalized, vector, tensor, rbf, R, D, envelope, src, dst)
        return term

    def _checked_compile(self, index, efm):
        # A zero-exit random-init module cannot test a backward implementation.
        # Check a private nonzero-exit copy AND the actual weights; the module is not modified.
        probe = copy.deepcopy(efm)
        generator = torch.Generator(device=self.device).manual_seed(20100 + index)
        with torch.no_grad():
            for layer in (probe.exit_scalar, probe.exit_vector, probe.exit_tensor):
                layer.weight.normal_(0., .05, generator=generator)
        actual_choices, reference_fn = self._term_choices(efm)
        probe_choices, probe_reference_fn = self._term_choices(probe)
        problem = self._check_problem(efm)
        ref_out, ref_grad, weights = self._run_check(reference_fn, problem, None, False)
        probe_out, probe_grad, _ = self._run_check(probe_reference_fn, problem, weights, False)
        if any(x is not None and not bool(torch.isfinite(x).all())
               for x in [*ref_out, *ref_grad, *probe_out, *probe_grad]):
            raise FloatingPointError('Nonfinite EFM compilation reference')
        report = {}
        for (name, factory), (probe_name, probe_factory) in zip(actual_choices, probe_choices):
            if name != probe_name:
                raise RuntimeError('EFM compile choice schedules differ')
            # Known torch 2.7.1 Inductor backward defect of the scalar-source block (see efm_engine).
            if name == 'inductor' and efm.input_is_scalar:
                report[name] = dict(status='excluded_known_scalar_source_backward_defect')
                continue
            try:
                probe_term = probe_factory(False)
                out, grad, _ = self._run_check(probe_term, problem, weights, False)
                probe_error = max(comparison_error(a, b)
                                  for a, b in zip([*out, *grad], [*probe_out, *probe_grad]))
                term = factory(False)
                out, grad, _ = self._run_check(term, problem, weights, False)
                actual_error = max(comparison_error(a, b)
                                   for a, b in zip([*out, *grad], [*ref_out, *ref_grad]))
            except (RuntimeError, AssertionError) as error:
                report[name] = dict(status='compile_check_failed', exception_type=type(error).__name__)
                continue
            passed = math.isfinite(probe_error) and math.isfinite(actual_error) and max(probe_error, actual_error) < 1e-3
            report[name] = dict(status='passed' if passed else 'rejected',
                nonzero_probe_relative_error=probe_error if math.isfinite(probe_error) else None,
                actual_relative_error=actual_error if math.isfinite(actual_error) else None)
            if passed:
                self.stats.setdefault('term_self_check', {})[index] = dict(
                    selected=name, tolerance=1e-3, nonzero_probe=True, choices=report)
                return term, name
        raise RuntimeError('No finite, verified EFM compiled implementation')

    def set_capacity(self, count):
        bucket = 1024 if self.device.type == 'cuda' else 16
        self.capacity = max(bucket, math.ceil(count * self.reserve / bucket) * bucket)
        self.stats['initial_edges'] = int(count)

    def _csr_geometry(self, radius, positions):
        view = self.csr.radius_view(radius)
        n, columns = view.neighbors.shape
        K = min(self.slots, columns)
        if self.capacity is None or self.stats['initial_max_count'] is None:
            # Setup only: every runtime replay below has fixed shapes and no scalar reads.
            max_count, count = int(view.counts.max()), int(view.counts.sum())
            self.slots = max(self.slots, max_count)
            K = min(self.slots, columns)
            if self.capacity is None:
                self.set_capacity(count)
            self.stats['initial_max_count'] = max_count
        neighbors, shifts = view.neighbors[:, :K], view.shifts[:, :K]
        selected = compact((neighbors >= 0).reshape(-1), self.capacity)
        src = gather_with_sentinel(neighbors.reshape(-1).clamp_min(0), selected)
        dst_all = torch.arange(n, device=self.device)[:, None].expand(n, K).reshape(-1)
        dst = gather_with_sentinel(dst_all, selected)
        shift = gather_with_sentinel(shifts.reshape(-1, 3), selected)
        pad = self.pad_nodes(self.capacity, n)
        src, dst = torch.where(selected.valid, src, pad), torch.where(selected.valid, dst, pad)
        self.flags[0].copy_(torch.maximum(self.flags[0], (selected.count - self.capacity).to(torch.int32)))
        self.flags[1].copy_(torch.maximum(self.flags[1], view.counts.max() - K))
        translation = shift.to(positions.dtype) @ self.csr.cell.to(positions.dtype)
        self.stats['csr_geometry_calls'] += 1
        return self._finish_geometry(positions, src, dst, translation, radius)

    def check_overflow(self):
        edges, slots = self.flags.tolist()
        if edges > 0 or slots > 0:
            raise EFMOverflow(max(edges, 0), max(slots, 0))
        return [edges, slots]

    def grow(self, error):
        if error.edge_excess:
            self.set_capacity(max(self.capacity * 2, self.capacity + error.edge_excess))
        if error.slot_excess:
            self.slots = max(self.slots * 2, self.slots + error.slot_excess)
        self.flags.zero_()
        self.stats['buffer_growths'] += 1

    def report(self):
        return dict(efm_precision='fp32', capacity=self.capacity, slots=self.slots,
                    edge_reserve=self.reserve, term_backend=dict(self.term_backend),
                    geometry_selection='stable fixed-capacity CSR compaction, refreshed every force call',
                    overflow_policy='checked before force return; grow and recapture', **self.stats)


def install_efm(model, *, slots=64, edge_reserve=1.5, compile_nodes=True):
    """Complete the staged base port in place, preserving attached EFM weights."""
    if getattr(model, 'compiled_nodes', False) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Uncompiled model with frozen parameters required')
    if any(len(group) for group in getattr(model, 'refinements', [])):
        raise ValueError('Atomic refinements have not been ported')
    efms, info = describe_efms(model)
    for index, radius in info['radii_A'].items():
        if radius > model.layer_cutoffs[index] or radius >= model.cutoff or radius not in model.layer_cutoffs:
            raise ValueError('EFM radius must be an existing layer radius below the model cutoff')
    state = EFMState(model, efms, info, slots=slots, edge_reserve=edge_reserve, compile_nodes=compile_nodes)
    model.__dict__['_efm_state'] = state
    for index, block in enumerate(model.blocks):
        if index in efms:
            block.__dict__['_efm_state'], block.__dict__['_efm_index'] = state, index
            state.original_forward[index] = MethodType(type(block).forward, block)
            state.base_finish[index] = state.compile(block._finish_messages)

            def finish(scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t, ds, dv, dt, block=block):
                return sibling._finish_with_efm(block, scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t, ds, dv, dt)

            state.efm_finish[index] = state.compile(finish)
            block.forward = MethodType(sibling._block_forward_with_efm, block)
            block._finish_messages = MethodType(sibling._finish_dispatch, block)
        else:
            block.uncompiled_finish_messages = block._finish_messages
            block._finish_messages = state.compile(block._finish_messages)
        block.compiled_nodes = bool(compile_nodes)
    model.compiled_nodes = bool(compile_nodes)
    state.inner_forward = model.forward
    model.forward = MethodType(sibling._model_forward_with_efm, model)
    return state
