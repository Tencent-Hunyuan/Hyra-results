"""CPU checks that the latent-charge bound survives every engine route.

Real node conversion, latent-charge energy/forces, FP64 centering and the
prepared wrappers run on CPU. The long-range kernels and multi-GPU collectives
are replaced by CPU doubles at their explicit seams: these checks cover
dispatch and autograd, not CUDA or NCCL execution.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyff.engines.capture import base_port
from hyff.kernels.fused_linear import FastLinear, enable_fast_nodes
from hyff.models import latent_charge_model as prepared
from hyff.models.charge_bound import charge_bound_value, install_charge_bound, raw_charges
from hyff.models.factory import CHARGE_BOUND_KEY
from hyff.models.latent_charges import LatentChargeEnergy


BOUND = 6.824962139129639


def make_holder(bounded=True, width=4):
    holder = nn.Module()
    holder.latent_charge_energy = LatentChargeEnergy(width)
    with torch.no_grad():
        holder.latent_charge_energy.charge_head.weight.copy_(
            torch.linspace(-2.5, 2.0, width).reshape(1, width))
        holder.latent_charge_energy.scale.fill_(1.)
    if bounded:
        install_charge_bound(holder, BOUND)
    return holder.eval().requires_grad_(False)


def frame():
    return torch.tensor([[0., 0., 0.], [1.2, .2, -.1], [-.4, 1.1, .3],
                         [.3, -.6, 1.4]], dtype=torch.float32, requires_grad=True)


def scalar_field(position):
    return torch.cat((position * 9., (position.square().sum(-1, keepdim=True) + 1.) * 4.), -1)


def centering(raw, atom_batch, totals, dtype=torch.float32):
    promoted = raw.to(dtype)
    sums = promoted.new_zeros(totals.numel()).index_add(0, atom_batch, promoted)
    counts = promoted.new_zeros(totals.numel()).index_add(0, atom_batch, torch.ones_like(promoted))
    return (promoted + ((totals.to(dtype) - sums) / counts.clamp_min(1))[atom_batch]).float()


def energy_force(term, position):
    atom_batch = torch.zeros(len(position), dtype=torch.long)
    totals = torch.tensor([1.], dtype=torch.float32)
    energy = term(scalar_field(position), position, atom_batch, totals)
    force = -torch.autograd.grad(energy.sum(), position)[0]
    return energy, force


class ConversionRouteTests(unittest.TestCase):
    def test_actual_capture_conversion_preserves_bound_and_parameters(self):
        for precision in ('fp32', 'tf32x3', 'fp16x3'):
            for width in (4, 672):
                with self.subTest(precision=precision, width=width):
                    holder = make_holder(width=width)
                    original = holder.latent_charge_energy.charge_head
                    values = torch.linspace(-3., 2., 4 * width).reshape(4, width)
                    expected = raw_charges(holder.latent_charge_energy, values)
                    state_keys = tuple(holder.state_dict())
                    base_port._convert_nodes(holder, precision=precision,
                        frozen_layernorm=False, layernorm_atom_tile=8, excluded=set())
                    actual = holder.latent_charge_energy.charge_head
                    self.assertEqual(charge_bound_value(holder), BOUND)
                    self.assertIs(actual.weight, original.weight)
                    self.assertEqual(tuple(holder.state_dict()), state_keys)
                    self.assertTrue(torch.equal(raw_charges(holder.latent_charge_energy, values), expected))
                    self.assertIsInstance(actual, nn.Linear if precision == 'fp32' else FastLinear)

    def test_capture_conversion_keeps_bound_energy_and_force_bitwise(self):
        holder = make_holder()
        expected = energy_force(holder.latent_charge_energy, frame())
        base_port._convert_nodes(holder, precision='tf32x3', frozen_layernorm=False,
                                 layernorm_atom_tile=8, excluded=set())
        actual = energy_force(holder.latent_charge_energy, frame())
        for measured, reference in zip(actual, expected):
            self.assertTrue(torch.equal(measured, reference))

    def test_generic_fast_node_conversion_preserves_bound(self):
        for precision in ('tf32x3', 'fp16x3'):
            with self.subTest(precision=precision):
                holder = make_holder()
                expected = energy_force(holder.latent_charge_energy, frame())
                enable_fast_nodes(holder, fuse_activations=True, strided_inputs=True,
                                  precision=precision)
                actual = energy_force(holder.latent_charge_energy, frame())
                self.assertIsInstance(holder.latent_charge_energy.charge_head, FastLinear)
                self.assertEqual(charge_bound_value(holder), BOUND)
                for measured, reference in zip(actual, expected):
                    self.assertTrue(torch.equal(measured, reference))

    def test_unbounded_conversion_is_original_fastlinear(self):
        holder = make_holder(False)
        expected = energy_force(holder.latent_charge_energy, frame())
        enable_fast_nodes(holder, precision='tf32x3')
        self.assertIs(type(holder.latent_charge_energy.charge_head), FastLinear)
        self.assertIsNone(charge_bound_value(holder))
        for measured, reference in zip(energy_force(holder.latent_charge_energy, frame()), expected):
            self.assertTrue(torch.equal(measured, reference))

    def test_compiled_head_cpu_graph_keeps_bound_and_input_gradient(self):
        # backend=eager checks Dynamo's full graph, not an Inductor/CUDA kernel.
        holder = make_holder()
        enable_fast_nodes(holder)
        head = lambda value: raw_charges(holder.latent_charge_energy, value)
        compiled = torch.compile(head, backend='eager', fullgraph=True)
        value = scalar_field(frame()).detach().requires_grad_(True)
        other = value.detach().clone().requires_grad_(True)
        direct, traced = head(value), compiled(other)
        self.assertTrue(torch.equal(direct, traced))
        self.assertTrue(torch.equal(torch.autograd.grad(direct.sum(), value)[0],
                                    torch.autograd.grad(traced.sum(), other)[0]))

    def test_replacing_charge_head_does_not_remove_term_bound(self):
        holder = make_holder()
        term = holder.latent_charge_energy
        replacement = nn.Linear(4, 1, bias=False, dtype=torch.float32).requires_grad_(False)
        with torch.no_grad():
            replacement.weight.copy_(term.charge_head.weight * 2.)
        term.charge_head = replacement
        scalars = scalar_field(frame())
        expected = BOUND * torch.tanh(replacement(scalars).squeeze(-1) / BOUND)
        self.assertEqual(charge_bound_value(holder), BOUND)
        self.assertTrue(torch.equal(raw_charges(term, scalars), expected))

    def test_actual_bind_base_preserves_term_bound(self):
        from hyff.models.factory import build_model
        config = dict(hidden=672, n_rbf=8, interactions=2, cutoff=4.5,
            switch_width=1., edge_hidden=8, tensor_hidden=16,
            atomic_refinements=0, layer_cutoffs=[3.5, 4.5], mlp_hidden=32,
            scalar_source_blocks=[], higher_invariants=True, tensor_mixing_rank=None,
            backbone='l4a0', atomic_numbers=[1, 6], expert_routing='element',
            expert_count=2, expert_hidden=32, expert_blocks=[0, 1],
            radial_conditioning='shared', radial_hidden=32, condition_spin=True,
            stab_invariant_transform='asinh', stab_output_gates=True,
            stab_gate_init=1., stab_residual_scale=None, stab_aggregate_normalization=None,
            efm_blocks=[0], efm_channels=[4, None], efm_layers=[1, None],
            efm_mmax=2, efm_edge_cutoff={'0': 3.})
        config[CHARGE_BOUND_KEY] = BOUND
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1018)
            holder = build_model({'model_id': 'hyff-efm-charges-w672c1', 'model_config': config})
        holder.eval().requires_grad_(False)
        term = holder.latent_charge_energy
        with torch.no_grad():
            term.charge_head.weight.copy_(torch.linspace(-2., 2.5, 672).reshape(1, 672))
            term.scale.fill_(1.)
        scalars = torch.linspace(-3., 2., 4 * 672).reshape(4, 672)
        b, total = torch.zeros(4, dtype=torch.long), torch.tensor([1.])
        original_head = term.charge_head
        expected = term.conserved_charges(scalars, b, total)
        base_port.bind_base(holder, global_context={'n_atoms': 4, 'total_charge': 1., 'multiplicity': 1},
                            precision='tf32x3', radial_intervals=2)
        self.assertIsNot(term.charge_head, original_head)
        self.assertIsInstance(term.charge_head, FastLinear)
        self.assertIs(term.charge_head.weight, original_head.weight)
        self.assertEqual(charge_bound_value(holder), BOUND)
        self.assertTrue(torch.equal(term.conserved_charges(scalars, b, total), expected))


class ConservationAndPreparedRouteTests(unittest.TestCase):
    def test_fp64_installer_centers_bounded_head_and_keeps_force_chain(self):
        from hyff.engines.capture.fp64_conservation import install_fp64_conservation
        holder = make_holder()
        holder._efm_state = SimpleNamespace(batch=None, csr=None, capacity=None)
        atom_batch = torch.zeros(4, dtype=torch.long)
        totals = torch.tensor([1.])
        holder._latent_charge_prepared_metadata = (atom_batch, totals)
        report = install_fp64_conservation(holder, {'requested': 'fp64', 'selected': 'fp64'})
        position = frame()
        scalars = scalar_field(position)
        raw = raw_charges(holder.latent_charge_energy, scalars)
        expected = centering(raw, atom_batch, totals, torch.float64)
        actual = holder.latent_charge_energy.conserved_charges(scalars, atom_batch, totals)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(report['dtypes']['sum'], 'float64')
        e, f = energy_force(holder.latent_charge_energy, frame())
        self.assertTrue(bool(torch.isfinite(e).all() & torch.isfinite(f).all()))
        self.assertGreater(float(f.abs().max()), 0.)

    def test_bound_precedes_native_centering_for_multiple_graphs(self):
        holder = make_holder()
        p = frame()
        s = scalar_field(p)
        batches, totals = torch.tensor([0, 0, 1, 1]), torch.tensor([1., -2.])
        head = holder.latent_charge_energy.charge_head
        linear = torch.nn.functional.linear(s, head.weight)
        expected = centering((BOUND * torch.tanh(linear / BOUND)).squeeze(-1), batches, totals)
        q = holder.latent_charge_energy.conserved_charges(s, batches, totals)
        self.assertTrue(torch.equal(q, expected))
        wrong = BOUND * torch.tanh(centering(linear.squeeze(-1), batches, totals) / BOUND)
        self.assertFalse(torch.equal(q, wrong))

    def test_prepared_latent_charge_wrapper_uses_bounded_head(self):
        holder = make_holder()
        holder.config = SimpleNamespace(hidden=4)
        batches, totals = torch.zeros(4, dtype=torch.long), torch.tensor([1.])
        holder._latent_charge_prepared_metadata = (batches, totals)
        def original(batch):
            holder.__dict__[prepared._CONTEXT]['scalar'] = scalar_field(batch['positions'])
            return batch['positions'].square().sum().reshape(1)
        holder.__dict__[prepared._ENERGY] = original
        reference_position, actual_position = frame(), frame()
        # Match the wrapper's graph construction order: original() produces
        # the final scalar before its base energy, then _evaluate adds the latent-charge term.
        # Reversing those branches changes FP32 gradient accumulation order.
        reference_scalar = scalar_field(reference_position)
        reference_base = reference_position.square().sum().reshape(1)
        expected = reference_base + holder.latent_charge_energy(
            reference_scalar, reference_position, batches, totals)
        actual = prepared._prepared_energy(holder, {'positions': actual_position})
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(torch.autograd.grad(actual.sum(), actual_position)[0],
                                    torch.autograd.grad(expected.sum(), reference_position)[0]))
        self.assertNotIn(prepared._CONTEXT, holder.__dict__)

    def test_dense_tree_dispatch_gets_bounded_centered_q(self):
        from hyff.kernels.long_range import backends as lr
        for with_plan in (False, True):
            with self.subTest(with_plan=with_plan):
                holder = make_holder()
                term = holder.latent_charge_energy
                backend = SimpleNamespace()
                if with_plan:
                    backend.rebuild = lambda *_: None
                    backend.needs_rebuild = lambda *_: False
                term.__dict__[lr._BACKEND] = backend
                seen = []
                def pair(x, q, coupling, sigma, received):
                    self.assertIs(received, backend)
                    seen.append(q)
                    return (x.square().sum() + q.square().sum()).reshape(1)
                p = frame()
                s = scalar_field(p)
                b, q = torch.zeros(4, dtype=torch.long), torch.tensor([1.])
                with patch.object(lr.LatentChargePair, 'apply', side_effect=pair), \
                     patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
                    energy = lr._lr_forward(term, s, p, b, q)
                self.assertTrue(torch.equal(seen[0], term.conserved_charges(s, b, q)))
                self.assertTrue(bool(torch.isfinite(torch.autograd.grad(energy.sum(), p)[0]).all()))

    def test_spme_dispatch_gets_bounded_centered_q(self):
        from hyff.kernels.long_range import backends as lr
        holder = make_holder()
        term = holder.latent_charge_energy
        seen = []
        def mesh_energy(q, positions, sigma):
            seen.append(q)
            return q.square().sum() + positions.square().sum(), None
        term.__dict__[lr._BACKEND] = SimpleNamespace(
            mesh=SimpleNamespace(energy=mesh_energy), check_sigma=lambda _: None)
        p = frame()
        s, b, totals = scalar_field(p), torch.zeros(4, dtype=torch.long), torch.tensor([1.])
        with patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
            e = lr._spme_forward(term, s, p, b, totals)
        self.assertTrue(torch.equal(seen[0], term.conserved_charges(s, b, totals)))
        self.assertTrue(bool(torch.isfinite(torch.autograd.grad(e.sum(), p)[0]).all()))

    def test_prepared_plan_proxy_retains_callable_and_term(self):
        from hyff.kernels.long_range.backends import _PreparedWithPlan
        holder = make_holder()
        term = holder.latent_charge_energy
        inner = lambda p: term(scalar_field(p), p, torch.zeros(4, dtype=torch.long), torch.tensor([1.]))
        proxy = _PreparedWithPlan(inner, SimpleNamespace(), term)
        p = frame()
        self.assertIs(proxy._term, term)
        self.assertEqual(charge_bound_value(proxy._term), BOUND)
        self.assertTrue(torch.equal(proxy(p), inner(p)))


class OwnedAndEntryRouteTests(unittest.TestCase):
    def test_owned_evaluate_passes_bound_to_charge_collective(self):
        # Execute the real evaluator body; replace only unrelated GPU stages and
        # collectives with CPU doubles. No NCCL/cross-rank parity is claimed.
        from hyff.multigpu import model as module
        holder, p = make_holder(), frame()
        evaluator = object.__new__(module.PreparedOwnedModel)
        evaluator.model = holder
        evaluator.runtime = SimpleNamespace(n_local=4, rank=0, device=torch.device('cpu'))
        evaluator.state = SimpleNamespace(initial_scalar=torch.zeros(4, 4))
        evaluator._initial_vector = evaluator._initial_tensor = torch.empty(0)
        evaluator._model_radial_stamp = 'unchanged'
        evaluator._charge_head_stamp = (id(holder.latent_charge_energy.charge_head), BOUND)
        evaluator._prepared_buffer_stamps = ()
        evaluator._active_base_pair_enabled = False
        evaluator._active_efm_geometries = ()
        evaluator._share_efm_geometry = False
        evaluator._shared_efm_geometry = None
        evaluator.layers, evaluator.trace = (), None
        evaluator.nodes = evaluator.positions = evaluator.pair_operator = evaluator.pair = None
        evaluator.charge = evaluator.charge_gather = None
        evaluator.root_long_range = SimpleNamespace(evaluate=lambda x, q: q.square().sum().reshape(1))
        class Sequence:
            def __init__(self, ignored): pass
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def finish(self):
                return SimpleNamespace(final_scalar=scalar_field(p), atomic_energy=p.square().sum(-1), control=None)
        seen = []
        def global_charge(raw, control, spec):
            seen.append(raw)
            return raw + (raw.new_tensor(1.) - raw.sum()) / len(raw), control
        with patch.object(module, '_model_radial_stamp', return_value='unchanged'), \
             patch.object(module, 'OwnedNodeSequence', Sequence), \
             patch.object(module.ordered, 'seed_control', return_value=None), \
             patch.object(module.boundary, 'exchange_positions', side_effect=lambda x, c, _: (SimpleNamespace(global_positions=x), c)), \
             patch.object(module.boundary, 'pair_snapshot', return_value=None), \
             patch.object(module.boundary, 'ordered_pair_energy', side_effect=lambda _, c, __: (p.new_zeros(4), c)), \
             patch.object(module.ordered, 'ordered_global_charge', side_effect=global_charge), \
             patch.object(module.ordered, 'ordered_partial_gather', side_effect=lambda q, c, _: (q, c)), \
             patch.object(module.ordered, 'ordered_energy_anchor', side_effect=lambda e, c, **kw: e):
            result, force = evaluator.energy_and_force(p)
            self.assertTrue(torch.equal(seen[0], raw_charges(holder.latent_charge_energy, scalar_field(p))))
            self.assertTrue(torch.equal(result.local_raw_charge, seen[0]))
            self.assertTrue(bool(torch.isfinite(force).all()))
            holder.latent_charge_energy._hyff_latent_charge_bound *= 2
            with self.assertRaisesRegex(ValueError, 'MODEL_COMPONENT_MUTATED_REPREPARE'):
                evaluator.evaluate(frame())

    def test_multigpu_rank_agreement_includes_bound(self):
        from hyff.multigpu.prepare_model import prepare_owned_model
        source = inspect.getsource(prepare_owned_model)
        self.assertIn('"charge_bound": charge_bound_value(model)', source)

    def test_supported_legacy_reconstruction_uses_native_factory(self):
        from hyff.engines import reference as adapter
        self.assertIn('from hyff.models.factory import build_model', inspect.getsource(adapter.build))
        self.assertIn("model_config=saved['model_config']", inspect.getsource(adapter.build))

    def test_hmr_shares_checkpoint_and_engine_route(self):
        # Source contract only: actual HMR/RATTLE execution remains CUDA-only.
        from hyff.md import simulation as md
        source = inspect.getsource(md)
        self.assertIn('if cfg.inclusive_hmr:', source)
        self.assertIn('repartition_hydrogen_mass_inclusive(s)', source)
        self.assertIn('create_engine(self.kind, self.checkpoint,', source)
        self.assertNotIn('charge_head', source)


if __name__ == '__main__':
    unittest.main()
