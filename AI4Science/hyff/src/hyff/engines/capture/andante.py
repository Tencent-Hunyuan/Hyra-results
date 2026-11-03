"""Captured force engine of the andante model (EFM architecture).

The base nodes run on the fused CSR path and the edge-frame-mixing blocks on
compiled fixed-capacity edge buffers, all inside one CUDA graph. The capture
self-check compares the prepared CSR evaluator outside and inside the graph;
it does not run the plain PyTorch model, which is available separately as
the reference engine.
"""
from __future__ import annotations

import math
import time

import torch

from hyff.engines.reference import system_batch, state_features
from hyff.models.efm_model import build_model
from hyff.data import physical
from hyff.engines.force_engine import ForceEngine

from hyff.engines.capture.base_port import bind_base
from hyff.engines.capture.efm_port import EFMOverflow, install_efm


class CaptureValidationError(AssertionError):
    pass


def require_efm_fp32():
    """Reject ambient TF32 instead of silently changing the caller's policy.

    EFM always uses FP32, including when explicit Triton base nodes use another
    supported precision. Its native linears inherit this process-global flag.
    """
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('EFM fp32 requires CUDA matmul TF32 to be disabled')


def build_proxy(seed=201):
    raise ValueError('Supply an explicit checkpoint or an already-built model')



def load_weights_into_proxy(checkpoint, seed=201):
    from hyff.weights import load_model
    return load_model(checkpoint, 'cpu')[0]



class CapturedEFMForceEngine(ForceEngine):
    """Graph-captured prepared evaluator with per-call fail-closed checks."""
    def __init__(self, model, batch, *, skin=1.5, max_buffer_retries=3):
        if type(max_buffer_retries) is not int or max_buffer_retries < 0:
            raise ValueError('max_buffer_retries must be a nonnegative integer')
        self.efm_state = model.__dict__['_efm_state']
        self.max_buffer_retries = max_buffer_retries
        self.capture_checks = []
        super().__init__(model, batch, 'cuda_graph', skin=skin, capture_force_atol=1e-4)

    def _release_capture(self):
        self.graph = None
        for name in ('static_energy', 'static_force'):
            if hasattr(self, name):
                setattr(self, name, None)
        previous = getattr(self, '_prepared_energy', None)
        if hasattr(previous, 'close'):
            previous.close()
        self._prepared_energy = None
        self.static_batch = self.static_edges = None

    def _capture(self, batch, edges, minimum_solver_budget=0):
        require_efm_fp32()
        if minimum_solver_budget:
            raise ValueError('Field solvers are not supported by the captured EFM engine')
        started = time.perf_counter()
        self._release_capture()
        self.efm_state.flags.zero_()
        self.static_edges, self.padding_nodes = self._edge_buffer(batch, edges)
        self._copy_edges(edges)
        self.static_batch = {key: value.detach().clone() for key, value in physical(batch).items()}
        position = self.static_batch['positions'].requires_grad_(True)
        stream = torch.cuda.Stream(device=position.device)
        stream.wait_stream(torch.cuda.current_stream(position.device))
        with torch.cuda.stream(stream):
            self._prepared_energy = self.model.prepare_energy(self.static_batch, self.static_edges)
        evaluate = self._prepared_energy

        def compute():
            energy = evaluate(position)
            force = -torch.autograd.grad(energy.sum(), position)[0]
            return energy, force

        # The bounded-memory prepared evaluator is the capture reference.
        # Agreement with the plain PyTorch model is checked separately (reference engine).
        with torch.cuda.stream(stream), torch.enable_grad():
            for _ in range(3):
                energy, force = compute()
            expected_e, expected_f = energy.detach().clone(), force.detach().clone()
        torch.cuda.current_stream(position.device).wait_stream(stream)
        torch.cuda.synchronize(position.device)
        del energy, force
        self.efm_state.check_overflow()
        if not bool(torch.isfinite(expected_e).all() & torch.isfinite(expected_f).all()):
            raise FloatingPointError('Nonfinite prepared reference evaluation')
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), torch.enable_grad():
            self.static_energy, self.static_force = compute()
        torch.cuda.synchronize(position.device)
        self.graph = graph
        graph.replay()
        self.efm_state.check_overflow()
        finite = bool(torch.isfinite(self.static_energy).all() & torch.isfinite(self.static_force).all())
        e_ok = finite and torch.allclose(self.static_energy, expected_e, rtol=5e-6, atol=1e-5 * len(position))
        f_ok = finite and torch.allclose(self.static_force, expected_f, rtol=1e-4, atol=self.capture_force_atol)
        delta = (self.static_force.detach() - expected_f).double()
        self.capture_checks.append(dict(reference='prepared_CSR_outside_graph',
            n_atoms=len(position), accepted=bool(e_ok and f_ok),
            force_max_error_eV_A=float(delta.abs().max()) if finite else None,
            force_rms_error_eV_A=float(delta.square().mean().sqrt()) if finite else None,
            energy_error_eV=float((self.static_energy - expected_e).abs().max()) if finite else None))
        if not (e_ok and f_ok):
            raise CaptureValidationError('Captured graph differs from the prepared evaluator; force rejected')
        self.captures += 1
        self.capture_seconds += time.perf_counter() - started

    def __call__(self, batch):
        for attempt in range(self.max_buffer_retries + 1):
            try:
                require_efm_fp32()
                energy, force = super().__call__(batch)
                self.efm_state.check_overflow()
                if not bool(torch.isfinite(energy).all() & torch.isfinite(force).all()):
                    raise FloatingPointError('Nonfinite energy/force; force rejected')
                return energy, force
            except EFMOverflow as error:
                if attempt == self.max_buffer_retries:
                    self._release_capture()
                    raise
                self.efm_state.grow(error)
                self._release_capture()
            except BaseException:
                # A rejected capture must not become reusable on the next call.
                self._release_capture()
                raise
        raise AssertionError('Unreachable buffer retry state')


def create_engine(positions, atomic_numbers, cell, checkpoint=None, device='cuda',
                  total_charge=0., multiplicity=1, skin=1.5, reserve=1.5,
                  precision='fp32', *, model=None, slots=64, edge_reserve=1.5,
                  max_buffer_retries=3, post_install_efm=None, **base_options):
    """Return the engine dictionary (model, batch, engine, state, metadata).

    A provided model is modified in place; use a separate copy for comparisons
    with the plain PyTorch model.
    An optional instance-local EFM callback runs after installation and before
    batch preparation, the first full-system force, or CUDA capture.
    """
    if post_install_efm is not None and not callable(post_install_efm):
        raise TypeError('post_install_efm must be callable or None')
    if torch.device(device).type != 'cuda':
        raise ValueError('Captured engine requires CUDA; use the reference engine on CPU')
    require_efm_fp32()
    if not math.isfinite(skin) or skin <= 0:
        raise ValueError('skin must be finite and positive')
    if model is not None and checkpoint is not None:
        raise ValueError('Supply either model or checkpoint, not both')
    if model is None:
        model = load_weights_into_proxy(checkpoint) if checkpoint is not None else build_proxy()
    model = model.to(device).eval().requires_grad_(False)
    n = len(atomic_numbers)
    context = dict(n_atoms=n, total_charge=float(total_charge), multiplicity=int(multiplicity),
                   state_features=state_features(n, total_charge, multiplicity))
    bind_base(model, global_context=context, reserve=reserve, precision=precision, **base_options)
    state = install_efm(model, slots=slots, edge_reserve=edge_reserve, compile_nodes=True)
    if post_install_efm is not None:
        post_install_efm(state)
    batch = system_batch(positions, atomic_numbers, cell, total_charge, multiplicity, device)
    engine = CapturedEFMForceEngine(model, batch, skin=skin, max_buffer_retries=max_buffer_retries)
    try:
        engine(batch)
    except BaseException:
        engine.close()
        raise
    metadata = dict(backend='captured_efm',
        precision=precision, efm_precision='fp32', skin_A=float(skin), reserve=float(reserve),
        cuda_matmul_allow_tf32=False,
        global_context=context, capture_reference='prepared_CSR_outside_graph',
        original_eager_parity='requires_separate_guard', weights_changed=False,
        efm=state.report())
    return dict(model=model, batch=batch, engine=engine, state=state, metadata=metadata)
