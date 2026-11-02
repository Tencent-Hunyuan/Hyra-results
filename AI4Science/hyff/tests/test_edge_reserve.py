"""CPU checks of the andante engine: edge-reserve dispatch, edge-buffer growth and the retry lifecycle.

No checkpoint is loaded, nothing is compiled and no CUDA engine is built.
The small blocked/tiled first-VJP comparisons are CPU algebra checks.
"""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from hyff.engines import dispatch as inference
from hyff.engines.capture import andante as captured_engine
from hyff.engines.capture import efm_tiling
from hyff.engines.capture import partial_tiling as partial
from hyff.engines.capture.efm_port import EFMOverflow, EFMState
from hyff.engines.capture.tiled_recompute import make_tiled_term


class TinyLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.m0 = nn.Linear(3 * channels, 3 * channels)
        self.gate = nn.Linear(channels, 5 * channels)
        self.m1 = nn.Linear(2 * channels, 2 * channels)
        self.m2 = nn.Linear(channels, channels)


class TinyEFM(nn.Module):
    """Complete frozen CPU algebra with the andante selector's architecture fields."""
    def __init__(self, channels, scalar, depth):
        super().__init__()
        self.channels, self.input_is_scalar = channels, scalar
        self.m_blocks = [] if scalar else [1, 2]
        self.aggregation, self.frame = 'sum', 'projector'
        self.complex_structure, self.receiver_in = False, None
        self.message_scale = .25
        self.node_scalar = nn.Linear(3, channels)
        self.node_vector = nn.Linear(3, channels, bias=False)
        self.node_tensor = nn.Linear(3, channels, bias=False)
        self.radial = nn.Linear(2, channels if scalar else 4 * channels)
        self.m0 = nn.Linear((2 if scalar else 4) * channels, 3 * channels)
        self.gate = nn.Linear(channels, (2 if scalar else 5) * channels)
        self.m1_a = nn.Linear(2 * channels, 2 * channels)
        self.m2_a = nn.Linear(channels, channels)
        self.layers = nn.ModuleList(TinyLayer(channels) for _ in range(depth))
        self.exit_scalar = nn.Linear(channels, 3)
        self.exit_vector = nn.Linear(channels, 3, bias=False)
        self.exit_tensor = nn.Linear(channels, 3, bias=False)
        self.eval().requires_grad_(False)


def make_state(reserve=1.05):
    # Restore the caller's RNG; all tensors and parameters are explicitly CPU.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2018)
        model = nn.Module()
        model.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        model.rbf = nn.Module()
        model.rbf.register_buffer('centers', torch.tensor([.5, 1.5]))
        model.rbf.register_buffer('gamma', torch.tensor(1.))
        model.rbf.cutoff = 5.
        model.register_buffer('padding_displacement', torch.tensor([1., 0., 0.]))
        efms = {0: TinyEFM(64, True, 2), 1: TinyEFM(64, False, 2),
                2: TinyEFM(64, False, 2), 3: TinyEFM(32, False, 0)}
        model.efms = nn.ModuleDict({str(index): efm for index, efm in efms.items()})
        state = EFMState(model, efms, {'switch_width': .5},
                         edge_reserve=reserve, compile_nodes=False)
    # These maps are normally bound by install_efm. Distinct sentinels make
    # preservation observable without invoking model/geometry evaluation.
    for name in ('base_finish', 'efm_finish', 'original_forward'):
        getattr(state, name).update({index: object() for index in efms})
    return state


def install_partial_without_compile(state, **options):
    if options['compile_tiles'] is not True:
        raise AssertionError('The production callback must request compilation')
    return partial.install_tiled_terms(state, tile_edges=options['tile_edges'],
                                      compile_tiles=False,
                                      tiled_blocks=options['tiled_blocks'])


def preserved_maps(state):
    return {name: dict(getattr(state, name)) for name in
            ('compiled_term', 'term_backend', 'base_finish', 'efm_finish', 'original_forward')}


class CPUOnlyCase(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(torch, 'compile', side_effect=AssertionError('compilation forbidden')).start()
        patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')).start()

    def assert_maps_preserved(self, state, before):
        for name, entries in before.items():
            self.assertEqual(set(getattr(state, name)), set(entries))
            for index, previous in entries.items():
                if name == 'term_backend':
                    self.assertEqual(getattr(state, name)[index], previous)
                else:
                    self.assertIs(getattr(state, name)[index], previous)


class AndanteDispatchTest(CPUOnlyCase):
    def setUp(self):
        super().setUp()
        # Each dispatch reserves one compiled-engine attempt; the per-process budget is tested separately.
        patch('hyff.engines.compiler_budget.reserve_compiled_engine_attempt', return_value=1).start()

    def dispatch(self, *, n_atoms=7, edge_reserve=None, capacity_policy=None,
                 metadata_error=None):
        calls, events = [], []
        made = {}

        def captured(*args, **kwargs):
            calls.append((args, kwargs))
            state = make_state(kwargs['edge_reserve'])
            events.append('install')
            kwargs['post_install_efm'](state)
            self.assertIsNone(state.capacity)
            self.assertEqual(state.stats['buffer_growths'], 0)
            events.append('prepare')
            state.set_capacity(17)
            events.append('first_force')
            made.update(state=state, engine=Mock(), metadata={})
            return made

        runtime = efm_tiling.runtime_metadata
        with patch.object(inference, 'load_model', return_value=(object(), {})) as load, \
             patch.object(captured_engine, 'create_engine', side_effect=captured), \
             patch.object(efm_tiling, 'install_tiled_terms', side_effect=install_partial_without_compile), \
             patch.object(efm_tiling, 'runtime_metadata',
                          side_effect=metadata_error if metadata_error else runtime):
            if metadata_error:
                with self.assertRaises(type(metadata_error)) as caught:
                    inference.create_engine('andante', 'unused', None, range(n_atoms),
                                            edge_reserve=edge_reserve,
                                            capacity_policy=capacity_policy)
                self.assertIs(caught.exception, metadata_error)
            else:
                result = inference.create_engine('andante', 'unused', None, range(n_atoms),
                                                 edge_reserve=edge_reserve,
                                                 capacity_policy=capacity_policy)
                self.assertIs(result, made)
            load.assert_called_once_with('unused', 'cpu')
        return made, calls, events

    def test_default_native_and_auto_routes_keep_base_and_edge_reserves_separate(self):
        for requested, n_atoms, selected in [(None, 7, None), ('native', 7, 'native'),
                                              ('auto', 50000, 'native'),
                                              ('tiled', 7, 'tiled'),
                                              ('auto', 50001, 'tiled')]:
            with self.subTest(requested=requested, n_atoms=n_atoms):
                made, calls, events = self.dispatch(n_atoms=n_atoms, capacity_policy=requested)
                args, options = calls[0]
                self.assertEqual(args[7:10], (1.5, 1.5, 'fp32'))
                self.assertEqual(options['edge_reserve'], 1.05)
                self.assertEqual(events, ['install', 'prepare', 'first_force'])
                meta = made['metadata']
                self.assertEqual(meta['base_reserve'], 1.5)
                self.assertEqual(meta['edge_reserve'], 1.05)
                policy = meta['efm_policy']
                self.assertEqual(policy['installations'], 1)
                self.assertEqual(policy['initial_slots'], 64)
                self.assertEqual(policy['selected_tiling'], selected == 'tiled')
                if requested is None:
                    self.assertNotIn('capacity_policy', meta)
                else:
                    self.assertEqual(meta['capacity_policy']['selected'], selected)
                if selected == 'tiled':
                    tiled = meta['capacity_policy']['tiling']
                    self.assertEqual(tiled['efm_edge_reserve'], 1.05)
                    self.assertEqual(tiled['requested_tiled_blocks'], [1, 2])
                else:
                    self.assertFalse(hasattr(made['state'], '_tiled_efm_original_terms'))

    def test_explicit_override_is_forwarded_and_reported_on_both_routes(self):
        for reserve in (1., 1.5, 1.2, '1.25', True):
            for route in (None, 'tiled'):
                with self.subTest(reserve=reserve, route=route):
                    made, calls, _ = self.dispatch(edge_reserve=reserve, capacity_policy=route)
                    self.assertEqual(calls[0][1]['edge_reserve'], float(reserve))
                    self.assertEqual(made['metadata']['edge_reserve'], float(reserve))
                    self.assertEqual(made['metadata']['efm_policy']['growth']['edge_growth_reserve'],
                                     float(reserve))
                    self.assertEqual(made['metadata']['base_reserve'], 1.5)

    def test_invalid_overrides_fail_before_load_or_construction(self):
        for reserve in (0., .999, -1., math.nan, math.inf, -math.inf, 'bad', object()):
            with self.subTest(reserve=reserve), \
                 patch.object(inference, 'load_model') as load, \
                 patch.object(captured_engine, 'create_engine') as captured:
                with self.assertRaises(ValueError):
                    inference.create_engine('andante', 'unused', None, range(7),
                                            edge_reserve=reserve)
                load.assert_not_called()
                captured.assert_not_called()

    def test_other_tiers_still_reject_edge_override_before_load(self):
        for kind in ('presto', 'vivace', 'moderato', 'reference'):
            with self.subTest(kind=kind), patch.object(inference, 'load_model') as load:
                with self.assertRaisesRegex(ValueError, 'supported only'):
                    inference.create_engine(kind, 'unused', None, range(7), edge_reserve=1.05)
                load.assert_not_called()

    def test_metadata_failure_closes_constructed_engine_and_preserves_error(self):
        error = RuntimeError('metadata guard')
        made, _, _ = self.dispatch(metadata_error=error)
        made['engine'].close.assert_called_once_with()


class EdgeGrowthTest(CPUOnlyCase):
    def test_growth_then_optional_tiling_precedes_preparation(self):
        original_growth = efm_tiling.install_growth_policy
        for tiled in (False, True):
            with self.subTest(tiled=tiled):
                state, events = make_state(), []
                original_maps = preserved_maps(state)
                original_check, original_set = state.check_overflow, state.set_capacity

                def growth(current, **options):
                    events.append('growth')
                    return original_growth(current, **options)

                def tiling(current, **options):
                    self.assertTrue(hasattr(current, '_efm_growth_policy'))
                    events.append('tiling')
                    return install_partial_without_compile(current, **options)

                with patch.object(efm_tiling, 'install_growth_policy', side_effect=growth), \
                     patch.object(efm_tiling, 'install_tiled_terms', side_effect=tiling):
                    efm_tiling.make_post_install(tiled=tiled)(state)
                self.assertEqual(events, ['growth', 'tiling'] if tiled else ['growth'])
                self.assertEqual(state.check_overflow, original_check)
                self.assertEqual(state.set_capacity, original_set)
                self.assertIsNone(state.capacity)
                self.assertIsNone(state.stats['initial_edges'])
                self.assertEqual(state.stats['buffer_growths'], 0)
                self.assertEqual(state.flags.tolist(), [0, 0])
                if not tiled:
                    self.assert_maps_preserved(state, original_maps)
                else:
                    for index in (0, 3):
                        self.assertIs(state.compiled_term[index], original_maps['compiled_term'][index])
                    for index in (1, 2):
                        self.assertIsNot(state.compiled_term[index], original_maps['compiled_term'][index])

    def test_one_callback_reuses_no_state_and_installs_once_per_fresh_state(self):
        callback = efm_tiling.make_post_install()
        first, second = make_state(), make_state()
        class_grow = EFMState.grow
        callback(first)
        callback(second)
        self.assertIs(first.grow.__self__, first)
        self.assertIs(second.grow.__self__, second)
        self.assertIsNot(first.grow, second.grow)
        self.assertIsNot(first._hyff_efm_runtime_policy, second._hyff_efm_runtime_policy)
        self.assertIs(EFMState.grow, class_grow)
        with self.assertRaisesRegex(ValueError, 'fresh'):
            callback(first)
        self.assertEqual(first._hyff_efm_runtime_policy['installations'], 1)

    def test_prepared_duplicate_precision_and_key_mismatch_fail_before_install(self):
        for mutation in ('batch', 'csr', 'capacity', 'growth', 'tiling', 'precision', 'keys', 'reserve'):
            with self.subTest(mutation=mutation):
                state = make_state()
                if mutation in ('batch', 'csr', 'capacity'):
                    setattr(state, mutation, object())
                elif mutation == 'growth':
                    state._efm_growth_policy = {}
                elif mutation == 'tiling':
                    state._tiled_efm_original_terms = {}
                elif mutation == 'precision':
                    state.efm_precision = 'tf32'
                elif mutation == 'keys':
                    state.compiled_term.pop(3)
                else:
                    state.reserve = 1.5
                with patch.object(efm_tiling, 'install_growth_policy') as growth, \
                     patch.object(efm_tiling, 'install_tiled_terms') as tiling:
                    with self.assertRaises(ValueError):
                        efm_tiling.make_post_install(tiled=True)(state)
                    growth.assert_not_called()
                    tiling.assert_not_called()

    def test_growth_preserves_every_callable_and_initial_sizing_statistics(self):
        for tiled in (False, True):
            with self.subTest(tiled=tiled), \
                 patch.object(efm_tiling, 'install_tiled_terms', side_effect=install_partial_without_compile) as installer:
                state = make_state()
                efm_tiling.make_post_install(tiled=tiled)(state)
                before, geometry = preserved_maps(state), state.compiled_geometry
                original_check, original_set = state.check_overflow, state.set_capacity
                state.set_capacity(17)
                self.assertEqual(state.capacity, 32)
                for edge_excess, slot_excess in ((1, 0), (0, 5), (41, 200)):
                    capacity, slots, growths = state.capacity, state.slots, state.stats['buffer_growths']
                    state.flags.copy_(torch.tensor([edge_excess, slot_excess], dtype=torch.int32))
                    with self.assertRaises(EFMOverflow) as caught:
                        state.check_overflow()
                    state.grow(caught.exception)
                    expected = (math.ceil(max((capacity + edge_excess) * 1.05,
                                              capacity + 16) / 16) * 16
                                if edge_excess else capacity)
                    self.assertEqual(state.capacity, expected)
                    self.assertGreaterEqual(state.capacity, capacity)
                    self.assertEqual(state.slots, max(2 * slots, slots + slot_excess)
                                     if slot_excess else slots)
                    self.assertEqual(state.flags.tolist(), [0, 0])
                    self.assertEqual(state.stats['buffer_growths'], growths + 1)
                    self.assertEqual(state.stats['initial_edges'], 17)
                    state.begin({'cpu_fixture': True}, object())
                    state.geometry['transient'] = object()
                    state.term['transient'] = object()
                    state.end()
                    self.assertEqual(state.geometry, {})
                    self.assertEqual(state.term, {})
                    self.assert_maps_preserved(state, before)
                    self.assertIs(state.compiled_geometry, geometry)
                    self.assertEqual(state.check_overflow, original_check)
                    self.assertEqual(state.set_capacity, original_set)
                self.assertEqual(installer.call_count, int(tiled))
                self.assertEqual(state._hyff_efm_runtime_policy['installations'], 1)
                self.assertFalse(state._hyff_efm_runtime_policy['callback_replayed_after_growth'])
                if tiled:
                    with self.assertRaises(ValueError):
                        partial.install_tiled_terms(state, compile_tiles=False)

    def test_invalid_growth_records_leave_state_unchanged(self):
        state = make_state()
        efm_tiling.make_post_install()(state)
        state.set_capacity(17)
        for edges, slots in ((0, 0), (-1, 1), (1, -1), (1.5, 0), (True, 0), (0, False)):
            with self.subTest(edges=edges, slots=slots):
                state.flags.fill_(9)
                before = (state.capacity, state.slots, deepcopy(state.stats), state.flags.clone())
                with self.assertRaises(ValueError):
                    state.grow(SimpleNamespace(edge_excess=edges, slot_excess=slots))
                self.assertEqual((state.capacity, state.slots, state.stats), before[:3])
                torch.testing.assert_close(state.flags, before[3], rtol=0, atol=0)

    def test_metadata_is_independent_and_rejects_changed_reserve(self):
        state = make_state()
        with patch.object(efm_tiling, 'install_tiled_terms', side_effect=install_partial_without_compile):
            efm_tiling.make_post_install(tiled=True)(state)
        state.set_capacity(17)
        runtime, tiled = efm_tiling.runtime_metadata(state), efm_tiling.tiled_metadata(state)
        runtime['growth']['edge_growth_reserve'] = 99.
        tiled['efm_dimensions'][1]['channels'] = 99
        self.assertEqual(efm_tiling.runtime_metadata(state)['growth']['edge_growth_reserve'], 1.05)
        self.assertEqual(efm_tiling.tiled_metadata(state)['efm_dimensions'][1]['channels'], 64)
        state.reserve = 1.5
        for metadata in (efm_tiling.runtime_metadata, efm_tiling.tiled_metadata):
            with self.assertRaisesRegex(ValueError, 'reserve changed'):
                metadata(state)

    def test_direct_callback_validates_reserve_and_tiled_types(self):
        for value in (True, False, '1.05', .99, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                efm_tiling.make_post_install(edge_reserve=value)
        with self.assertRaises(ValueError):
            efm_tiling.make_post_install(tiled=1)


class TiledFirstVJPTest(CPUOnlyCase):
    def test_same_selected_callable_accepts_new_capacity_and_matches_blocked_cpu_vjp(self):
        state = make_state()
        native = state.compiled_term[1]
        parameters = list(state.efms[1].parameters())
        saved = [(parameter, parameter.detach().clone()) for parameter in parameters]
        with patch.object(efm_tiling, 'install_tiled_terms', side_effect=install_partial_without_compile):
            efm_tiling.make_post_install(tiled=True)(state)
        selected = state.compiled_term[1]
        # A separate small-tile term exercises multiple replay tiles on CPU.
        small_tile = make_tiled_term(state.efms[1], tile_edges=13, compile_tiles=False)
        state.set_capacity(17)
        generator = torch.Generator(device='cpu').manual_seed(1018)

        for grow in (False, True):
            if grow:
                state.grow(EFMOverflow(41, 0))
            length, nodes = state.capacity, 7
            self.assertEqual(length, 80 if grow else 32)
            self.assertIs(state.compiled_term[1], selected)
            values = [torch.randn(shape, generator=generator) for shape in
                      ((nodes, 3), (nodes, 3, 3), (nodes, 5, 3), (length, 2),
                       (length, 3, 3), (length, 5, 5), (length,))]
            values[-1][-5:] = 0.
            src = torch.arange(length) % nodes
            dst = (torch.arange(length) * 3 + 1) % nodes
            weights = [torch.randn(shape, generator=generator) for shape in
                       ((nodes, 3), (nodes, 3, 3), (nodes, 5, 3))]

            def evaluate(function):
                leaves = [value.detach().clone().requires_grad_(True) for value in values]
                outputs = function(*leaves, src, dst)
                gradients = torch.autograd.grad(outputs, leaves, weights, allow_unused=True)
                return outputs, gradients

            expected_outputs, expected_gradients = evaluate(native)
            self.assertTrue(any(bool(value.abs().max() > 0) for value in expected_gradients
                                if value is not None))
            for function in (selected, small_tile):
                with self.subTest(length=length, multiple_tiles=function is small_tile):
                    outputs, gradients = evaluate(function)
                    for actual, expected in zip((*outputs, *gradients),
                                                (*expected_outputs, *expected_gradients)):
                        if expected is None:
                            self.assertIsNone(actual)
                        else:
                            self.assertIsNotNone(actual)
                            self.assertTrue(bool(torch.isfinite(actual).all()))
                            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        for actual, (previous, value) in zip(state.efms[1].parameters(), saved):
            self.assertIs(actual, previous)
            self.assertFalse(actual.requires_grad)
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        self.assertEqual(efm_tiling.runtime_metadata(state)['installations'], 1)


class RetryLifecycleTest(CPUOnlyCase):
    def make_engine(self, events, retries=3):
        state = make_state()
        efm_tiling.make_post_install()(state)
        state.set_capacity(17)
        engine = object.__new__(captured_engine.CapturedEFMForceEngine)
        engine.efm_state, engine.max_buffer_retries = state, retries
        engine.graph = object()
        engine.static_energy, engine.static_force = object(), object()
        engine.static_batch, engine.static_edges = object(), object()
        engine._prepared_energy = SimpleNamespace(close=lambda: events.append('close'))
        real_grow = state.grow

        def grow(error):
            events.append('grow')
            real_grow(error)

        state.grow = grow
        return engine

    def assert_released(self, engine):
        for name in ('graph', 'static_energy', 'static_force', '_prepared_energy',
                     'static_batch', 'static_edges'):
            self.assertIsNone(getattr(engine, name))

    def test_overflow_rejects_result_then_grows_releases_and_retries_same_state(self):
        for source in ('parent', 'post_call_check'):
            with self.subTest(source=source):
                events = []
                engine = self.make_engine(events)
                state, before = engine.efm_state, preserved_maps(engine.efm_state)
                rejected = (torch.tensor([99.]), torch.full((1, 3), 99.))
                accepted = (torch.tensor([1.]), torch.ones(1, 3))
                calls = []

                def parent(current, batch):
                    self.assertIs(current.efm_state, state)
                    calls.append(batch)
                    events.append('parent')
                    if len(calls) == 1:
                        if source == 'parent':
                            raise EFMOverflow(41, 0)
                        state.flags[0] = 41
                        return rejected
                    self.assert_released(current)
                    self.assertEqual(state.capacity, 80)
                    return accepted

                with patch.object(captured_engine, 'require_efm_fp32', side_effect=lambda: events.append('precision')), \
                     patch.object(captured_engine.ForceEngine, '__call__', new=parent):
                    result = engine({'fixture': True})
                self.assertIs(result[0], accepted[0])
                self.assertIs(result[1], accepted[1])
                self.assertEqual(events, ['precision', 'parent', 'grow', 'close', 'precision', 'parent'])
                self.assertEqual(state.stats['buffer_growths'], 1)
                self.assertIs(engine.efm_state, state)
                self.assert_maps_preserved(state, before)

    def test_exhaustion_is_bounded_and_releases_without_final_growth(self):
        for retries in (0, 3):
            with self.subTest(retries=retries):
                events = []
                engine = self.make_engine(events, retries=retries)
                error = EFMOverflow(1, 0)

                def parent(current, batch):
                    events.append('parent')
                    raise error

                with patch.object(captured_engine, 'require_efm_fp32'), \
                     patch.object(captured_engine.ForceEngine, '__call__', new=parent):
                    with self.assertRaises(EFMOverflow) as caught:
                        engine({})
                self.assertIs(caught.exception, error)
                self.assertEqual(events.count('parent'), retries + 1)
                self.assertEqual(events.count('grow'), retries)
                self.assertEqual(events.count('close'), 1)
                self.assertEqual(engine.efm_state.stats['buffer_growths'], retries)
                self.assert_released(engine)

    def test_nonfinite_output_and_generic_errors_release_without_growth(self):
        for failure in ('energy', 'force', 'generic', 'precision'):
            with self.subTest(failure=failure):
                events = []
                engine = self.make_engine(events)
                sentinel = RuntimeError('rejected parent or precision')

                def parent(current, batch):
                    events.append('parent')
                    if failure == 'generic':
                        raise sentinel
                    energy = torch.tensor([math.nan if failure == 'energy' else 1.])
                    force = torch.full((1, 3), math.inf if failure == 'force' else 1.)
                    return energy, force

                with patch.object(captured_engine, 'require_efm_fp32',
                                  side_effect=sentinel if failure == 'precision' else None), \
                     patch.object(captured_engine.ForceEngine, '__call__', new=parent):
                    expected = RuntimeError if failure in ('generic', 'precision') else FloatingPointError
                    with self.assertRaises(expected) as caught:
                        engine({})
                if failure in ('generic', 'precision'):
                    self.assertIs(caught.exception, sentinel)
                self.assertEqual(events.count('parent'), int(failure != 'precision'))
                self.assertEqual(events.count('close'), 1)
                self.assertNotIn('grow', events)
                self.assertEqual(engine.efm_state.stats['buffer_growths'], 0)
                self.assert_released(engine)


if __name__ == '__main__':
    unittest.main()
