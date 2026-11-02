"""Experimental multi-GPU largo MD launcher (single node, 1/2/4/8 GPUs).

Selected by ``hyff.Simulation(..., gpus=N)`` / ``hyff md --gpus N`` (automatic
card policy) or an explicit ``route='multigpu'`` with ``ranks``. Without
torchrun, run_md returns status ``launch_required`` and the torchrun ``argv``;
``hyff.Simulation`` then starts that command. Gloo coordinates acceptance
before NCCL exists; unused ranks return without allocating CUDA. Failures
propagate to torchrun; there is no fallback or retry after a collective
failure. Workers wait for root commands under a finite collective timeout
(180 s by default).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import sys


RANKS = (1, 2, 4, 8)
logger = logging.getLogger('hyff')
# Process-wide multi-GPU service options, read from the environment at startup (all ranks must agree).
# Native single-GPU runs ignore them.
OPTION_ENV = dict(
    native_nodes='HYFF_MULTIGPU_NATIVE_NODES',            # tf32x3 node products and grouped experts (unsupported; default off)
    compiled_efm='HYFF_MULTIGPU_COMPILED_EFM',            # compiled fixed-shape edge-frame-mixing tiles (default on)
    compact_efm_edges='HYFF_MULTIGPU_COMPACT_EFM_EDGES',  # skip inactive edge-frame-mixing edges (default on; needs compiled_efm)
    compact_base_pair='HYFF_MULTIGPU_COMPACT_BASE_PAIR',  # skip inactive base/pair rows (default: same as compact_efm_edges)
)
OPTION_DEFAULTS = dict(native_nodes='0', compiled_efm='1')  # the compaction defaults follow compiled_efm
EFM_TILE_EDGES_ENV = 'HYFF_MULTIGPU_EFM_TILE_EDGES'
EFM_TILE_EDGES = ('8192', '16384', '32768')
# Node options of the native-nodes variant: the single-GPU largo base options. EFM stays FP32 nn.Linear.
NATIVE_NODE_BASE_OPTIONS = dict(precision='tf32x3', joined_projections=False, fold_first=False,
                                frozen_layernorm=False, fused_features=False, grouped_experts=True)


class DeviceBindingUnavailable(RuntimeError):
    """The allocated device or its software versions could not be queried."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def multigpu_options(environ=None):
    """Resolve the multi-GPU service options from the environment."""
    environ = os.environ if environ is None else environ
    values = {}
    for key, default in OPTION_DEFAULTS.items():
        value = environ.get(OPTION_ENV[key], default)
        _require(value in ('0', '1'), 'MD_MULTIGPU_OPTION_INVALID')
        values[key] = value == '1'
    edge_value = environ.get(OPTION_ENV['compact_efm_edges'], '1' if values['compiled_efm'] else '0')
    _require(edge_value in ('0', '1'), 'MD_MULTIGPU_OPTION_INVALID')
    values['compact_efm_edges'] = edge_value == '1'
    pair_value = environ.get(OPTION_ENV['compact_base_pair'], '1' if values['compact_efm_edges'] else '0')
    _require(pair_value in ('0', '1'), 'MD_MULTIGPU_COMPACT_BASE_PAIR_INVALID')
    values['compact_base_pair'] = pair_value == '1'
    _require(not values['compact_base_pair'] or values['compact_efm_edges'],
             'MD_MULTIGPU_COMPACT_BASE_PAIR_REQUIRES_COMPACT_EFM_EDGES')
    tile_edges = environ.get(EFM_TILE_EDGES_ENV, '32768')
    _require(tile_edges in EFM_TILE_EDGES, 'MD_MULTIGPU_EFM_TILE_EDGES_INVALID')
    _require(not values['compact_efm_edges'] or values['compiled_efm'],
             'MD_MULTIGPU_COMPACT_EFM_EDGES_REQUIRES_COMPILED_EFM')
    _require(tile_edges == '32768' or values['compiled_efm'], 'MD_MULTIGPU_EFM_TILE_EDGES_REQUIRES_COMPILED_EFM')
    values['efm_tile_edges'] = int(tile_edges)
    return values


def service_precision(environ=None):
    return 'tf32x3_nodes_fp32_efm' if multigpu_options(environ)['native_nodes'] else 'fp32'


def _service_policy(options=None, *, capacity_retention=None):
    from hyff.multigpu.service_policy import service_policy
    options = multigpu_options() if options is None else options
    policy = service_policy(native_nodes=options['native_nodes'], compiled_efm=options['compiled_efm'],
                            compact_efm_edges=options['compact_efm_edges'],
                            compact_base_pair=options['compact_base_pair'],
                            efm_tile_edges=options['efm_tile_edges'])
    default = '1' if policy.get('active_efm_compaction') else '0'
    enabled = os.environ.get('HYFF_CAPACITY_RETENTION', default)
    _require(enabled in ('0', '1'), 'MD_CAPACITY_RETENTION_SWITCH_INVALID')
    if capacity_retention is not None:
        _require(type(capacity_retention) is bool, 'MD_CAPACITY_RETENTION_OVERRIDE_INVALID')
        enabled = '1' if capacity_retention else '0'
    if enabled == '1':
        _require(policy.get('active_efm_compaction') and policy.get('compile_efm_tiles'),
                 'MD_CAPACITY_RETENTION_REQUIRES_COMPILED_COMPACT_EFM')
        policy.update(capacity_retention=True, share_efm_geometry=True)
    return policy


def _agree_service_policy(expected, *, requested_service_policy=None):
    """Check that this rank's service policy equals the root's before any model load or CUDA bind."""
    requested = _service_policy()
    if requested_service_policy is None:
        _require(requested == expected, 'MD_MULTIGPU_RANK_POLICY_MISMATCH')
    else:
        _require(requested == requested_service_policy and
                 requested.get('capacity_retention') is True and
                 expected == _service_policy(capacity_retention=False),
                 'MD_MULTIGPU_RANK_POLICY_MISMATCH')
    return expected


def _auto_timing_service_policy(requested):
    """The service policy the measured card-count timings were taken with (no EFM reuse)."""
    if requested.get('capacity_retention') is True:
        return _service_policy(capacity_retention=False)
    return requested


def _auto_policy_decision(decision, requested, timing_policy):
    result = dict(decision, requested_service_policy=requested,
                  selected_service_policy=requested if decision.get('mode') == 'multigpu' else None)
    if requested != timing_policy:
        count = decision.get('ranks') or decision.get('launch_required_ranks')
        measured_scope = count == 4 and decision.get('requested_atoms') == 103440
        result.update(capacity_timing_acceptance=('MEASURED_4GPU_EFM_REUSE_TIMING'
                          if measured_scope else 'TRANSFERRED_EFM_REUSE_TIMING'),
                      capacity_policy_transfer_label=('measured 4-GPU 103,440-atom timing'
                          if measured_scope else 'timing transferred from the 4-GPU 103,440-atom measurement'),
                      timing_service_policy=timing_policy,
                      timing_policy_scope='CARD_COUNT_SELECTION_ONLY',
                      proposed_service_policy=requested if decision.get('launch_required_ranks') else None,
                      capacity_policy_selection=('AUTO_USE_REQUESTED_POLICY_WITH_EFM_REUSE'
                          if decision.get('mode') == 'multigpu' else
                          'AUTO_EFM_REUSE_POLICY_LAUNCH_HINT' if decision.get('launch_required_ranks') else
                          'AUTO_NATIVE_SCOPE_OR_AVAILABILITY'))
        if decision.get('mode') == 'multigpu':
            # Timings taken without EFM reuse do not validate the requested policy.
            result.update(timing_policy_validation=decision.get('validation'),
                          validation='NOT_VALIDATED',
                          scientific_validation=False, nve_validation=False,
                          measurement_matches_request=False, nve_scope=None,
                          cross_rank_nve_support=False)
    return result


def _execution_policy_arguments(decision, request_route):
    """Check an auto-selected service policy; an explicit request is never changed."""
    if decision.get('capacity_policy_selection') == 'AUTO_USE_REQUESTED_POLICY_WITH_EFM_REUSE':
        requested = decision.get('requested_service_policy')
        _require(request_route == 'auto' and decision.get('mode') == 'multigpu' and
                 requested == _service_policy() and requested.get('capacity_retention') is True and
                 decision.get('selected_service_policy') == requested and
                 decision.get('timing_service_policy') == _service_policy(capacity_retention=False),
                 'MD_CAPACITY_AUTO_POLICY_SELECTION_INVALID')
        return {}
    if decision.get('capacity_policy_selection') != 'AUTO_USE_TIMED_POLICY_WITHOUT_EFM_REUSE':
        return {}
    requested = decision.get('requested_service_policy')
    selected = decision.get('selected_service_policy')
    _require(request_route == 'auto' and decision.get('mode') == 'multigpu' and
             decision.get('capacity_timing_acceptance') == 'NO_EFM_REUSE_TIMINGS' and
             requested == _service_policy() and requested.get('capacity_retention') is True and
             selected == _service_policy(capacity_retention=False),
             'MD_CAPACITY_AUTO_POLICY_SELECTION_INVALID')
    return dict(requested_service_policy=requested)


def _error_code(error):
    """Expose only a bounded symbolic code, never arbitrary exception text."""
    value = error.args[0] if error.args else None
    return value if type(value) is str and re.fullmatch(r'[A-Z][A-Z0-9_]{0,95}', value) else 'MD_UNCLASSIFIED_ERROR'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


@dataclass(frozen=True)
class LaunchContext:
    world_size: int
    rank: int
    local_rank: int
    torchrun: bool

    @classmethod
    def from_environ(cls, environ):
        names = ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE')
        present = [name in environ for name in names]
        if not any(present):
            return cls(1, 0, 0, False)
        _require(all(present), 'MD_TORCHRUN_CONTEXT_INCOMPLETE')
        values = []
        for name in names:
            value = environ[name]
            _require(isinstance(value, str) and value.isascii() and value.isdecimal(),
                     'MD_TORCHRUN_CONTEXT_INVALID')
            values.append(int(value))
        world, rank, local, local_world = values
        _require(world in RANKS and local_world == world and rank == local and 0 <= rank < world,
                 'MD_SINGLE_NODE_TORCHRUN_1_2_4_8_REQUIRED')
        return cls(world, rank, local, True)


def _actual_device_binding(context):
    """Runtime-only root query on the already allocated torchrun device.

    No device UUID, address, host identifier or unallocated-card state is read.
    NVML supplies the global driver version; it is not a memory-fit assertion.
    """
    _require(context.torchrun and context.rank == 0, 'MD_ACTUAL_DEVICE_QUERY_REQUIRES_TORCHRUN_ROOT')
    try:
        import ctypes
        import torch
        import triton
        torch.cuda.set_device(context.local_rank)
        properties = torch.cuda.get_device_properties(context.local_rank)
        nvml = ctypes.CDLL('libnvidia-ml.so.1')
        _require(nvml.nvmlInit_v2() == 0, 'MD_DRIVER_VERSION_QUERY_FAILED')
        try:
            output = ctypes.create_string_buffer(128)
            status = nvml.nvmlSystemGetDriverVersion(output, ctypes.c_uint(len(output)))
            _require(status == 0, 'MD_DRIVER_VERSION_QUERY_FAILED')
            driver = output.value.decode('ascii')
        finally:
            nvml.nvmlShutdown()
        return dict(device_class=str(properties.name), capability=[int(properties.major), int(properties.minor)],
            versions=dict(torch=str(torch.__version__), cuda=str(torch.version.cuda),
                          triton=str(triton.__version__), driver=driver))
    except Exception as error:
        raise DeviceBindingUnavailable('MD_ACTUAL_DEVICE_BINDING_UNAVAILABLE') from error


def available_ranks(context, cap=None):
    _require(cap is None or (type(cap) is int and cap in RANKS), 'MD_RANK_CAP_INVALID')
    return tuple(value for value in RANKS if value <= context.world_size and
                 (cap is None or value <= cap))


def _validate_distributed_options(cfg):
    _require(cfg.model == 'largo', 'MD_MULTIGPU_LARGO_ONLY')
    _require(cfg.engine in (None, 'largo'), 'MD_MULTIGPU_ENGINE_UNSUPPORTED')
    _require(cfg.precision in (None, 'fp32'), 'MD_MULTIGPU_FP32_REQUIRED')
    _require(cfg.statistics in (None, 'fp64'), 'MD_MULTIGPU_STATISTICS_UNSUPPORTED')
    _require(cfg.capacity_policy is None and cfg.edge_reserve is None and
             cfg.expert_storage_alias in (None, False), 'MD_MULTIGPU_FIXED_NATIVE_OWNED_TILING_REQUIRED')
    # The multi-GPU service evaluates the long-range term densely; None means the default.
    _require(cfg.long_range in (None, 'dense'), 'MD_MULTIGPU_DENSE_LR_REQUIRED')
    _require(cfg.conservation in (None, 'native'), 'MD_MULTIGPU_CONSERVATION_UNSUPPORTED')
    _require(cfg.recenter_interval_ps is None and cfg.molecule_system_xml is None,
             'MD_MULTIGPU_RECENTER_UNSUPPORTED')
    _require(cfg.capture_retries == 0 and cfg.capture_branch_policy == 'first',
             'MD_MULTIGPU_CAPTURE_OPTIONS_UNSUPPORTED')
    _require(type(cfg.skin) in (int, float) and math.isfinite(cfg.skin) and cfg.skin > 0,
             'MD_MULTIGPU_SKIN_INVALID')
    _require(type(cfg.reserve) in (int, float) and math.isfinite(cfg.reserve) and cfg.reserve >= 1,
             'MD_MULTIGPU_RESERVE_INVALID')


def _auto_config_error(cfg, physical=None):
    try:
        _validate_distributed_options(cfg)
    except ValueError as error:
        return _error_code(error)
    if physical is not None and physical.get('native_engine') != 'largo':
        return 'MD_AUTO_NATIVE_ENGINE_NOT_LARGO'
    return None


def _card_auto_config_error(cfg):
    """None when the automatic card-count rule applies to this configuration."""
    try:
        _validate_distributed_options(cfg)
        options = multigpu_options()
    except ValueError as error:
        return _error_code(error)
    # The measured timings used compiled, compacted edge-frame mixing with default tiles and fp32 nodes.
    if (options['native_nodes'] or not options['compiled_efm'] or not options['compact_efm_edges']
            or options['efm_tile_edges'] != 32768):
        return 'MD_AUTO_CARD_SERVICE_POLICY_UNSUPPORTED'
    return None


def _auto_service_fp32(cfg, native_engine=None):
    """Only largo service requests need a separate native precision."""
    return (cfg.route == 'auto' and cfg.model == 'largo' and cfg.precision == 'fp32'
            and cfg.engine in (None, 'largo')
            and native_engine in (None, 'largo'))


def _visible_card_ranks(context, cap=None):
    """Count only visible devices; do not allocate or parse CUDA visibility text."""
    try:
        import torch
        visible = torch.cuda.device_count()
    except Exception:
        visible = 0
    if type(visible) is not int or visible < 0:
        visible = 0
    limit = min(visible, context.world_size if context.torchrun else 8,
                cap if cap is not None else 8)
    # One GPU (the native route) is always listed, even when no device is visible;
    # the native setup reports its own CUDA errors.
    return tuple(value for value in RANKS if value == 1 or value <= limit)


def _card_auto_route(cfg, context, physical):
    """Apply the automatic card-count rule (see card_guidance.select_card_count)."""
    from hyff.multigpu.card_guidance import request_scope_error, select_card_count
    requested_policy = _service_policy()
    policy = _auto_timing_service_policy(requested_policy)
    error = request_scope_error(cfg, physical, service_policy=policy)
    if error is not None:
        return dict(_route('native', 1, 'AUTO_CARD_SCOPE_INELIGIBLE'),
                    config_ineligible_code=error)
    available = _visible_card_ranks(context, cfg.ranks)
    scope = 'MIN_TORCHRUN_WORLD_VISIBLE_DEVICES_AND_CAP' if context.torchrun else 'VISIBLE_DEVICES_AND_CAP_LAUNCH_HINT_ONLY'
    count = 8 if 8 in available else 4 if 4 in available else 1
    if count == 1:
        return dict(_route('native', 1, 'AUTO_CARD_FEWER_THAN_FOUR_AVAILABLE'),
                    availability_scope=scope)
    if not context.torchrun:
        # Device class/version acceptance happens in the actual torchrun root.
        # This is only an executable launch hint, never a selected GPU route.
        return _auto_policy_decision(dict(mode=None, ranks=None, reason='AUTO_CARD_SELECTION_DEFERRED_TO_TORCHRUN',
                    validation='NOT_SELECTED', launch_required_ranks=count,
                    availability_scope=scope, timing_guaranteed=False,
                    device_binding_verified=False,
                    measured_atoms=103440, requested_atoms=physical['atoms'],
                    size_scope='AT_OR_ABOVE_MEASURED_KNOT',
                    threshold_applied=False,
                    threshold_label='at or above the measured 103,440-atom point (forecast)',
                    extrapolation_applied=False,
                    forecast_requested=physical['atoms'] > 103440,
                    timing_basis=('UPWARD_EXTRAPOLATION' if physical['atoms'] > 103440
                                  else 'MEASURED_POINT_TRANSFER')), requested_policy, policy)
    try:
        device = _actual_device_binding(context)
    except DeviceBindingUnavailable:
        return dict(_route('native', 1, 'ACTUAL_DEVICE_BINDING_UNAVAILABLE'),
                    availability_scope=scope)
    decision = select_card_count(cfg, physical, available, device, service_policy=policy)
    return _auto_policy_decision({**_route(decision['mode'], decision['ranks'], decision['reason']), **decision,
            'availability_scope': scope, 'requested_native_precision': physical['native_precision'],
            'selected_precision': (service_precision() if decision['mode'] == 'multigpu'
                                   else physical['native_precision'])}, requested_policy, policy)


def _card_explicit_warnings(cfg, physical=None):
    """Explicit service requests are never silently changed by speed advice."""
    from hyff.multigpu.card_guidance import explicit_warning
    warning = explicit_warning(cfg, physical, service_policy=_service_policy())
    return [] if warning is None else [warning]


def _route(mode, ranks, reason):
    return dict(mode=mode, ranks=ranks, reason=reason, validation='NOT_VALIDATED',
                mean_ms=None, median_ms=None, timing_guaranteed=False, threshold_applied=False,
                memory_acceptance='PENDING_COLLECTIVE_SETUP' if mode == 'multigpu' else 'NOT_APPLICABLE',
                trajectory_capacity_guaranteed=False)


def _decision_context(decision, cfg, context, table_sha256=None):
    result = dict(decision, request_route=cfg.route, request_ranks_cap=cfg.ranks,
                  world_size=context.world_size,
                  table_sha256=decision.get('table_sha256', table_sha256))
    if cfg.route == 'auto':
        result.update(requested_cli_precision=cfg.precision,
                      native_precision_normalized=decision.get('native_precision_normalized',
                                                                False))
    return result


def native_cli_decision(cfg):
    """Ordinary native CLI deliberately ignores inherited torchrun variables."""
    return _decision_context(_route('native', 1, 'EXPLICIT_NATIVE'), cfg,
                             LaunchContext(1, 0, 0, False))


def _log_route(decision, *, rank=0):
    """One short line for the route decision; details at debug level."""
    if rank != 0:
        return
    mode, ranks = decision.get('mode'), decision.get('ranks')
    if mode == 'multigpu':
        logger.info('hyff md: running on %d GPUs', ranks)
    elif decision.get('launch_required_ranks'):
        logger.info('hyff md: %d GPUs selected; starting torchrun', decision['launch_required_ranks'])
    else:
        logger.info('hyff md: running on one GPU (%s)', str(decision.get('reason', '')).lower().replace('_', ' '))
    logger.debug('route decision: %s', json.dumps(decision, sort_keys=True, default=str))
    for warning in decision.get('performance_warnings', ()) or ():
        message = warning.get('message') if isinstance(warning, dict) else str(warning)
        logger.warning('hyff md: %s', message)


def launch_command(cfg, ranks):
    """Return argv and printable shell spelling; no shell or child is started."""
    _require(type(ranks) is int and ranks in RANKS, 'MD_LAUNCH_RANKS_INVALID')
    payload = _canonical(asdict(cfg)).decode()
    argv = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc-per-node', str(ranks), '--module', 'hyff.multigpu.launcher', '--config-json', payload]
    return argv, shlex.join(argv)


def _launch_required(cfg, ranks, reason, context, table_sha256=None, *, recommendation=None):
    argv, command = launch_command(cfg, ranks)
    result = dict(status='launch_required', reason=reason, gpus=ranks, command=command, argv=argv)
    if cfg.debug:
        decision = _decision_context(dict(mode=None, ranks=None, reason=reason, validation='NOT_SELECTED'),
                                     cfg, context, table_sha256)
        if recommendation is not None:
            decision.update({key: value for key, value in recommendation.items()
                             if key not in ('mode', 'ranks', 'launch_required_ranks')})
            decision.update(mode=None, ranks=None, validation='NOT_SELECTED')
        result['debug'] = dict(route_decision=decision)
    logger.debug('torchrun command: %s', command)
    return result


class _TorchControl:
    """CPU control plane; owns only process groups it constructs."""
    def __init__(self, context, timeout_seconds):
        self.context, self.groups = context, []
        self.timeout = timedelta(seconds=timeout_seconds)
        self.dist = None
        if context.torchrun:
            import torch.distributed as dist
            _require(dist.is_available() and not dist.is_initialized(), 'MD_FRESH_TORCHRUN_GROUP_REQUIRED')
            dist.init_process_group('gloo', init_method='env://', timeout=self.timeout)
            self.dist = dist

    def broadcast(self, value):
        if self.dist is None:
            return value
        packet = [value if self.context.rank == 0 else None]
        self.dist.broadcast_object_list(packet, src=0)
        return packet[0]

    def group(self, ranks, backend):
        if self.dist is None:
            _require(ranks == 1, 'MD_MULTI_RANK_REQUIRES_TORCHRUN')
            return None
        group = self.dist.new_group(ranks=list(range(ranks)), backend=backend, timeout=self.timeout)
        if self.context.rank < ranks:
            self.groups.append(group)
        return group

    def active_gather(self, value, ranks, group):
        if self.dist is None:
            return [value]
        output = [None] * ranks
        self.dist.all_gather_object(output, value, group=group)
        return output

    def stage(self, name, ranks, group, function):
        """All world ranks join status, even unused ranks awaiting NCCL creation."""
        output, statuses = None, None
        if self.context.rank < ranks:
            try:
                output = function()
                status = dict(ok=True, error_type=None, error_code=None)
            except Exception as error:
                status = dict(ok=False, error_type=type(error).__name__, error_code=_error_code(error))
            statuses = self.active_gather(status, ranks, group)
        failure = None
        if self.context.rank == 0:
            bad = [index for index, status in enumerate(statuses) if not status['ok']]
            failure = dict(stage=name, failed_ranks=bad,
                           error_types=[statuses[index]['error_type'] for index in bad],
                           error_codes=[statuses[index].get('error_code', 'MD_UNCLASSIFIED_ERROR') for index in bad])
        failure = self.broadcast(failure)
        if failure['failed_ranks']:
            raise RuntimeError('MD_COHERENT_PREBUILD_FAILURE:' + json.dumps(failure, sort_keys=True))
        return output

    def close(self):
        if self.dist is not None:
            for group in reversed(self.groups):
                self.dist.destroy_process_group(group)
            self.dist.destroy_process_group()


def _array_digest(values):
    import numpy as np
    digest = hashlib.sha256()
    for key in sorted(values):
        value = np.ascontiguousarray(values[key])
        _require(value.dtype.kind in 'biuf', 'MD_PHYSICAL_ARRAY_TYPE_INVALID')
        header = _canonical([key, str(value.dtype), list(value.shape)])
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_bootstrap(cfg):
    """Load the system, optional restart state and checkpoint identity on the root rank.

    No physical arrays are written to the logs.
    """
    import numpy as np
    from hyff.io import load_restart_state, load_system, system_multiplicity, validate_restart_state
    from hyff.models.factory import charge_bound_setting
    from hyff.weights import config_path, read_config
    system = load_system(cfg.system)
    multiplicity = system_multiplicity(system)
    numbers = np.asarray(system['atomic_numbers'])
    positions = np.asarray(system['positions_A'])
    cell = np.asarray(system['box_A'])
    charge = np.asarray(system['total_charge'])
    _require(numbers.ndim == 1 and len(numbers) > 0 and numbers.dtype.kind in 'iu' and
             bool((numbers > 0).all()), 'MD_ATOMIC_NUMBERS_INVALID')
    atoms = len(numbers)
    _require(positions.shape == (atoms, 3) and positions.dtype.kind in 'fiu' and
             bool(np.isfinite(positions).all()), 'MD_INITIAL_POSITIONS_INVALID')
    _require(cell.shape == (3, 3) and cell.dtype.kind in 'fiu' and bool(np.isfinite(cell).all()),
             'MD_CELL_INVALID')
    _require(charge.shape == () and charge.dtype.kind in 'fiu' and math.isfinite(float(charge)),
             'MD_TOTAL_CHARGE_INVALID')
    required = ('positions_A', 'velocities_A_ps', 'atomic_numbers', 'box_A', 'masses',
                'constraint_pairs', 'constraint_distances_A', 'total_charge')
    _require('water_idx' in system or not (cfg.hmr_mass or cfg.inclusive_hmr),
             'MD_HMR_WATER_IDX_REQUIRED')
    if 'water_idx' in system:
        required += ('water_idx',)
    physical_digest = _array_digest({key: system[key] for key in required} | {'multiplicity': multiplicity})
    initial = None
    if cfg.initial_positions:
        initial = load_restart_state(cfg.initial_positions)
        _require(all(value.shape == (atoms, 3) and value.dtype.kind in 'fiu' and
                     bool(np.isfinite(value).all()) for value in
                     (initial['positions_A'], initial['velocities_A_fs'])), 'MD_RESTART_STATE_INVALID')
        validate_restart_state(system, initial)
        positions = initial['positions_A']
    # The initial-state digest covers numeric arrays only; restart metadata is
    # not part of it.
    initial_digest = _array_digest({key: initial[key] for key in
        ('positions_A', 'velocities_A_fs', 'total_charge', 'multiplicity') if key in initial}
        if initial is not None else
        {key: system[key] for key in ('positions_A', 'velocities_A_ps')})
    # The fp32 precision of an explicit multi-GPU request belongs to the service; the
    # single-GPU comparison keeps the model's own engine options.
    try:
        selection = replace(cfg, precision=None if cfg.precision == 'fp32' else cfg.precision,
                            route='native', ranks=None, gpus=1).resolved()
    except ValueError as error:
        raise ValueError('MD_NATIVE_OPTIONS_INVALID') from error
    native_engine = selection.engine
    native_precision = selection.options['precision']
    native_options = selection.options
    resolved = dict(local=selection.weights, sha256=selection.sha256)
    _path, artifact = read_config(resolved['local'])
    artifact_path = config_path(resolved['local'])
    physical = dict(positions=np.ascontiguousarray(positions, dtype=np.float32),
        numbers=np.ascontiguousarray(numbers, dtype=np.int64),
        cell=np.ascontiguousarray(cell, dtype=np.float32), atoms=atoms,
        total_charge=float(charge), multiplicity=multiplicity,
        checkpoint=resolved['local'], checkpoint_sha256=resolved['sha256'],
        artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        model_id=artifact['model_id'],
        native_engine=native_engine, native_precision=native_precision,
        native_statistics=native_options['statistics'],
        charge_bound=charge_bound_setting(artifact['model_config']),
        workload_sha256=physical_digest, initial_state_sha256=initial_digest)
    _require(bool(np.isfinite(physical['positions']).all() & np.isfinite(physical['cell']).all()),
             'MD_FP32_PHYSICAL_OVERFLOW')
    return dict(physical=physical, system=system, initial_state=initial)


def _choose_route(cfg, context, physical):
    if cfg.route == 'native':
        return _route('native', 1, 'EXPLICIT_NATIVE')
    options = multigpu_options()
    policy = _service_policy(options)
    if cfg.route == 'multigpu':
        _require(type(cfg.ranks) is int and cfg.ranks in available_ranks(context),
                 'MD_EXPLICIT_RANKS_UNAVAILABLE')
        return dict(_route('multigpu', cfg.ranks, 'EXPLICIT_MULTIGPU'),
                    requested_native_precision=physical['native_precision'],
                    selected_precision=service_precision(), service_options=options,
                    requested_service_policy=policy, selected_service_policy=policy)
    ineligible = _auto_config_error(cfg, physical)
    return dict(_route('native', 1, 'AUTO_CONFIG_NOT_MULTIGPU_ELIGIBLE' if ineligible else 'AUTO_NO_MEASURED_TIMING'),
                config_ineligible_code=ineligible, service_options=options,
                requested_service_policy=policy, selected_service_policy=None)


def _load_cpu_model(physical, *, service_policy=None, requested_service_policy=None):
    policy = _agree_service_policy(_service_policy() if service_policy is None else service_policy,
                                   requested_service_policy=requested_service_policy)
    if policy.get('capacity_retention', False):
        from hyff.multigpu.efm_capacity_runtime import install_retention_process_policy
        install_retention_process_policy()
    from hyff.engines.capture.compile_policy import configure_largo_compiler
    configure_largo_compiler()  # before binding or any lazy native-neighbor import
    from hyff.weights import config_path, load_model, sha256
    artifact_path = config_path(physical['checkpoint'])
    _require(sha256(physical['checkpoint']) == physical['checkpoint_sha256'] and
        hashlib.sha256(artifact_path.read_bytes()).hexdigest() == physical['artifact_sha256'],
        'MD_ARTIFACT_CHANGED_BEFORE_MODEL_LOAD')
    model, _config = load_model(physical['checkpoint'], 'cpu')
    from hyff.models.charge_bound import charge_bound_value
    _require(charge_bound_value(model) == physical['charge_bound'], 'MD_NATIVE_CHARGE_BOUND_MISMATCH')
    return model


def _bind_cuda_model(model, physical, cfg, context, service_policy=None, *, requested_service_policy=None):
    service_policy = _agree_service_policy(_service_policy() if service_policy is None else service_policy,
                                           requested_service_policy=requested_service_policy)
    import torch
    torch.cuda.set_device(context.local_rank)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda', context.local_rank)
    model = model.to(device).eval().requires_grad_(False)
    from hyff.engines.capture.base_port import bind_base
    options = (NATIVE_NODE_BASE_OPTIONS if service_policy['grouped_experts'] else
               dict(precision='fp32', joined_projections=False, fold_first=False,
                    frozen_layernorm=False, fused_features=False, grouped_experts=False))
    bind_base(model, global_context=dict(n_atoms=physical['atoms'],
        total_charge=physical['total_charge'], multiplicity=physical['multiplicity']),
        reserve=cfg.reserve, **options)
    from hyff.models.charge_bound import charge_bound_value
    _require(charge_bound_value(model) == physical['charge_bound'], 'MD_BOUND_CHANGED_DURING_BIND')
    _precision_report(model, service_policy=service_policy, requested_service_policy=requested_service_policy)
    return model


def _node_modules(model, *, native_nodes=False):
    """Count node-path replacements; EFM descendants must keep FP32 nn.Linear."""
    from torch import nn
    from hyff.kernels.fused_linear import FastLinear
    from hyff.kernels.grouped_experts import FrozenElementExpertMLP
    efm_ids = set()
    for block in model.blocks:
        efm = getattr(block, 'efm', None)
        if efm is not None:
            efm_ids.update(id(module) for module in efm.modules())
    fast = [module for module in model.modules() if isinstance(module, FastLinear)]
    grouped = [module for module in model.modules() if isinstance(module, FrozenElementExpertMLP)]
    efm_linears = [module for module in model.modules() if id(module) in efm_ids and isinstance(module, nn.Linear)]
    nodes = dict(fast_linear=len(fast), fast_linear_precisions=sorted({module.precision for module in fast}),
        fast_linear_inside_efm=sum(id(module) in efm_ids for module in fast),
        grouped_experts=len(grouped), grouped_precisions=sorted({module.precision for module in grouped}),
        efm_fp32_linears=len(efm_linears),
        expert_partition_exposed=all(hasattr(model.blocks[index].update_mlp, 'partition')
                                     for index in model.config.expert_blocks),
        expert_blocks=len(tuple(model.config.expert_blocks)))
    if native_nodes:
        import torch
        from hyff.models.l2_potential import _RowSelectedLinear
        nodes['ordinary_node_linears'] = sum(id(module) not in efm_ids and
            isinstance(module, (nn.Linear, _RowSelectedLinear)) for module in model.modules())
        # Grouped kernels read these packed buffers, not the expert Parameters,
        # so Parameter dtypes alone do not determine their arithmetic.
        nodes['grouped_cache_fp32'] = all(
            name in module._non_persistent_buffers_set and
            isinstance(module._buffers.get(name), torch.Tensor) and
            module._buffers[name].dtype == torch.float32
            for module in grouped for name in ('weight1', 'bias1', 'weight2', 'bias2'))
    return nodes


def _precision_report(model, *, service_policy=None, requested_service_policy=None):
    """Observe actual model/ambient arithmetic policy on this selected rank.

    The default service requires the all-FP32 node tuple; the native-nodes option requires exactly the
    single-GPU largo node tuple (tf32x3, grouped experts) with FP32 EFM; anything else fails.
    """
    import torch
    service_options = multigpu_options()
    service_policy = _agree_service_policy(_service_policy(service_options) if service_policy is None else service_policy,
                                           requested_service_policy=requested_service_policy)
    native_nodes = service_options['native_nodes']
    metadata = getattr(model, 'deployment_metadata', {})
    option_names = ('compiled_nodes', 'joined_projections', 'folded_first_projections',
                    'frozen_layernorm', 'fused_local_features', 'grouped_experts', 'learned_weights_changed')
    options = {key: metadata.get(key) for key in option_names}
    model_flags = {key: bool(getattr(model, key, False)) for key in
                   ('compiled_nodes', 'fast_nodes', 'joined_nodes', 'frozen_local_features')}
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    nodes = _node_modules(model, native_nodes=native_nodes)
    report = dict(status='OBSERVED', base_bound=getattr(model, '_xl_capture_base_bound', False),
        precision=metadata.get('precision'), parameter_dtypes=dtypes, base_options=options,
        model_flags=model_flags, float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), service_options=service_options, node_modules=nodes,
        service_policy=service_policy,
        sparse_expert_nodes=bool(getattr(model, 'sparse_expert_nodes', False)),
        fast_node_precision=getattr(model, 'fast_node_precision', None),
        grouped_expert_precision=getattr(model, 'grouped_expert_precision', None))
    common = (report['base_bound'] is True and dtypes == ['torch.float32'] and
        report['float32_matmul_precision'] == 'highest' and
        not report['cuda_matmul_allow_tf32'] and not report['cudnn_allow_tf32'] and
        not any(model_flags[key] for key in ('compiled_nodes', 'joined_nodes', 'frozen_local_features')) and
        nodes['fast_linear_inside_efm'] == 0 and nodes['efm_fp32_linears'] > 0 and
        nodes['expert_partition_exposed'] and nodes['expert_blocks'] > 0)
    if not native_nodes:
        _require(common and report['precision'] == 'fp32' and
            all(value is False for value in options.values()) and not any(model_flags.values()) and
            nodes['fast_linear'] == 0 and nodes['grouped_experts'] == 0 and
            not report['sparse_expert_nodes'], 'MD_ACTUAL_FP32_PRECISION_POLICY_MISMATCH')
    else:
        _require(common and report['precision'] == 'tf32x3' and options['grouped_experts'] is True and
            all(options[key] is False for key in option_names if key != 'grouped_experts') and
            model_flags['fast_nodes'] and report['sparse_expert_nodes'] and
            report['fast_node_precision'] == 'tf32x3' and report['grouped_expert_precision'] == 'tf32x3' and
            nodes['fast_linear'] > 0 and nodes['fast_linear_precisions'] == ['tf32x3'] and
            nodes['ordinary_node_linears'] == 0 and nodes['grouped_cache_fp32'] and
            nodes['grouped_experts'] == nodes['expert_blocks'] and nodes['grouped_precisions'] == ['tf32x3'],
            'MD_ACTUAL_NATIVE_NODES_PRECISION_POLICY_MISMATCH')
    return report


def _dependency_directory(explicit):
    if explicit is not None:
        return Path(explicit).expanduser().resolve(strict=True)
    spec = importlib.util.find_spec('nvalchemiops')
    _require(spec is not None and isinstance(spec.origin, str), 'MD_NEIGHBOR_DEPENDENCY_REQUIRED')
    return Path(spec.origin).resolve().parent.parent


def _prepare_service(model, physical, cfg, context, ranks, group, dependency_root, cache_directory,
                     service_policy=None, *, requested_service_policy=None):
    service_policy = _agree_service_policy(_service_policy() if service_policy is None else service_policy,
                                           requested_service_policy=requested_service_policy)
    import torch
    from hyff.multigpu.md_session import prepare_md_session
    from hyff.multigpu.force_service import OwnedForceService
    cache_root = (Path(cache_directory).expanduser().absolute() if cache_directory is not None else
                  Path(cfg.output).expanduser().absolute() / 'multigpu_cache')
    cache = cache_root / f'rank{context.rank}'
    cache.mkdir(parents=True, exist_ok=True)
    compaction = ({'active_efm_compaction': True}
                  if service_policy.get('active_efm_compaction', False) else {})
    compaction['active_base_pair_compaction'] = service_policy.get('active_base_pair_compaction', False)
    if service_policy.get('capacity_retention', False):
        compaction.update(share_efm_geometry=True, capacity_retention=True)
    session = prepare_md_session(model,
        torch.as_tensor(physical['numbers'], dtype=torch.int64, device='cpu').contiguous(),
        torch.as_tensor(physical['positions'], dtype=torch.float32, device='cpu').contiguous(),
        torch.as_tensor(physical['cell'], dtype=torch.float32, device='cpu').contiguous(),
        global_ids=torch.arange(physical['atoms'], dtype=torch.int64, device='cpu'),
        device=torch.device('cuda', context.local_rank), ranks=ranks,
        dependency_root=_dependency_directory(dependency_root), cache_directory=cache,
        root_lr_route='dense', experimental=True, group=group,
        total_charge=physical['total_charge'], multiplicity=physical['multiplicity'],
        skin=cfg.skin, tile_edges=service_policy.get('efm_tile_edges', 32768),
        compile_efm_tiles=service_policy['compile_efm_tiles'], **compaction)
    return OwnedForceService(session)


def _record_abort(run, service):
    attempt = dict(code='CALLER_ABORT')
    try:
        attempt['result'] = service.abort('CALLER_ABORT')
    except BaseException as error:
        attempt['error_type'] = type(error).__name__
    if run is not None:
        run.result.setdefault('distributed_abort_attempts', []).append(attempt)
    return attempt


def _run_root(run, root_runner, service=None):
    failed = False
    setup_complete = False
    try:
        run.setup()
        setup_complete = True
        if service is not None:
            # This follows collective service construction, initial force and
            # root RATTLE allocation. It observes only this configuration.
            acceptance = dict(status='OBSERVED_INITIAL_SETUP_FIT', measurement='measured',
                scope='MODEL_NEIGHBOR_INITIAL_FORCE_AND_INTEGRATOR_SETUP',
                requested_atoms=run.result.get('n_atoms'),
                trajectory_capacity_guaranteed=False, failure_action='TERMINATE_NO_RETRY_OR_FALLBACK')
            run.result['distributed_memory_acceptance'] = acceptance
            run.result['route_decision'] = dict(run.result.get('route_decision', {}),
                memory_acceptance=acceptance['status'], memory_fit_scope=acceptance['scope'],
                trajectory_capacity_guaranteed=False)
            run.save()
        return run.run() if root_runner is None else root_runner(run)
    except BaseException as error:
        failed = True
        if service is not None:
            _record_abort(run, service)
            if not setup_complete:
                run.result['distributed_memory_acceptance'] = dict(status='SETUP_FAILED',
                    observed_fit=False, error_type=type(error).__name__,
                    failure_action='TERMINATE_NO_RETRY_OR_FALLBACK')
            persist = getattr(run, '_persist_restored_failure', None)
            if callable(persist):
                persist(error)
        try:
            run.result.update(status='failed', error_type=type(error).__name__)
            run.save()
        except BaseException:
            pass
        raise
    finally:
        try:
            run.close()
        except BaseException:
            if not failed:
                raise


def _native_run(cfg, decision, root_runner):
    from hyff.md.simulation import MDRun
    # For auto, an explicit FP32 flag requests service arithmetic. A native
    # fallback keeps the registered native default rather than passing FP32 to
    # the TF32x3-only captured engine only after bootstrap established that
    # actual engine. Unknown native defaults and explicit native stay unchanged.
    normalize = _auto_service_fp32(cfg) and decision.get('native_precision_normalized', False)
    precision = None if normalize else cfg.precision
    run = MDRun(replace(cfg, route='native', ranks=1, gpus=1, precision=precision))
    run.result['route_decision'] = decision
    result = _run_root(run, root_runner)
    return run.public_result() if result is None else result


def _distributed_run(cfg, bootstrap, decision, service, root_runner):
    import numpy as np
    from hyff.multigpu.md_bridge import DistributedMDRun
    physical = bootstrap['physical']

    def factory(run, positions, numbers, cell, charge, multiplicity):
        _require(np.array_equal(numbers, physical['numbers']) and
            np.array_equal(np.asarray(cell, dtype=np.float32), physical['cell']) and
            charge == physical['total_charge'] and multiplicity == physical['multiplicity'],
            'MD_SETUP_PHYSICAL_BINDING_CHANGED')
        _require(run.checkpoint == physical['checkpoint'] and
                 run.checkpoint_sha256 == physical['checkpoint_sha256'], 'MD_SETUP_CHECKPOINT_CHANGED')
        return dict(model=service.model, batch={}, engine=service, metadata=service.metadata())

    run, failed = None, False
    try:
        selected = replace(cfg, ranks=decision['ranks'])
        run = DistributedMDRun(selected, engine_factory=factory,
            system=bootstrap['system'], initial_state=bootstrap['initial_state'],
            service_policy=decision['selected_service_policy'],
            **_execution_policy_arguments(decision, cfg.route))
        run.result['route_decision'] = decision
        result = _run_root(run, root_runner, service)
        return run.public_result() if result is None else result
    except BaseException:
        failed = True
        # Covers constructor/pre-adoption failures as well as normal run errors.
        if run is None or not run.result.get('distributed_abort_attempts'):
            _record_abort(run, service)
            if run is not None:
                try:
                    run.save()
                except BaseException:
                    pass
        raise
    finally:
        try:
            service.close()
        except BaseException:
            if not failed:
                raise


def _log_compiler_cache(context, compiler_cache):
    """Emit bounded startup policy fields before a potentially failing load."""
    variables = {name: {key: value[key] for key in ('source', 'effective')}
                 for name, value in compiler_cache['variables'].items()}
    logger.debug('compiler cache: %s', json.dumps(dict(rank=context.rank,
        local_rank=context.local_rank, variables=variables,
        **{name: compiler_cache[name] for name in ('policy_version', 'status', 'namespace_source',
            'override', 'environment_changed', 'existing_pools_retargeted',
            'caller_per_rank_assertion_unverified')},
        compiler_modules_already_imported_count=len(compiler_cache['compiler_modules_already_imported'])),
        sort_keys=True, allow_nan=False))


def _memory_report(context, service, model, compiler_cache=None):
    import torch
    device = torch.device('cuda', context.local_rank)
    return dict(rank=context.rank, service=service.metadata(), precision=_precision_report(model),
        compiler_cache=compiler_cache,
        setup_inclusive_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        setup_inclusive_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        timed_peak_allocated_bytes=None, timed_peak_reserved_bytes=None,
        timed_memory_measurement='NOT_OBSERVED_BY_LAUNCHER')


def _persist_rank_reports(cfg, reports, timeout_seconds):
    """With debug=True, merge the per-rank memory/precision reports into the saved result.json."""
    if not cfg.debug:
        return
    target = Path(cfg.output) / 'result.json'
    document = json.loads(target.read_text())
    _require(type(document) is dict, 'MD_SAVED_RESULT_INVALID')
    document.setdefault('debug', {}).update(rank_reports=reports, collective_timeout_seconds=timeout_seconds)
    temporary = target.with_name('result.rank_reports.pending.json')
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)


def run_md(cfg, *, _root_runner=None, _timeout_seconds=180,
           _dependency_root=None, _cache_directory=None):
    """Select the route and run MD, or return the torchrun command to launch it.

    The private keyword arguments never change the force service. _root_runner,
    if given, is called with the run after setup instead of run.run.
    A caller must supervise the whole torchrun job, including its wall deadline.
    The finite per-collective timeout is not a total-job timeout. A service worker
    waits under that timeout while root is idle: callbacks must return to the
    service within the bound.
    """
    _require(type(_timeout_seconds) in (int, float) and math.isfinite(_timeout_seconds) and
             1 <= _timeout_seconds <= 3600, 'MD_COLLECTIVE_TIMEOUT_INVALID')
    _require(_root_runner is None or callable(_root_runner), 'MD_ROOT_RUNNER_INVALID')
    context = LaunchContext.from_environ(os.environ)
    _require(cfg.route in ('native', 'auto', 'multigpu'), 'MD_ROUTE_INVALID')
    _require(cfg.ranks is None or (type(cfg.ranks) is int and cfg.ranks in RANKS), 'MD_RANK_CAP_INVALID')
    if cfg.route == 'multigpu' and context.torchrun:
        _require(cfg.ranks is not None and cfg.ranks <= context.world_size, 'MD_EXPLICIT_RANKS_UNAVAILABLE')
    card_auto = cfg.route == 'auto' and _card_auto_config_error(cfg) is None
    table_sha = None
    if cfg.route == 'multigpu':
        _validate_distributed_options(cfg)
        _require(type(cfg.ranks) is int and cfg.ranks in RANKS, 'MD_EXPLICIT_RANKS_REQUIRED')
        if not context.torchrun and cfg.ranks > 1:
            advice = dict(performance_warnings=_card_explicit_warnings(cfg))
            result = _launch_required(cfg, cfg.ranks, 'MULTI_RANK_TORCHRUN_REQUIRED', context,
                                      recommendation=advice)
            _log_route(dict(advice, launch_required_ranks=cfg.ranks))
            return result
    control = _TorchControl(context, _timeout_seconds)
    successful = False
    try:
        bootstrap, decision, packet = None, None, None
        if context.rank == 0:
            try:
                if cfg.route == 'multigpu' or card_auto:
                    bootstrap = _load_bootstrap(cfg)
                if card_auto:
                    decision = _card_auto_route(cfg, context, bootstrap['physical'])
                if decision is None:
                    decision = _choose_route(cfg, context, None if bootstrap is None else bootstrap['physical'])
                if cfg.route == 'multigpu':
                    decision['performance_warnings'] = _card_explicit_warnings(cfg, bootstrap['physical'])
                if cfg.route == 'auto' and bootstrap is not None:
                    decision['native_precision_normalized'] = _auto_service_fp32(
                        cfg, bootstrap['physical']['native_engine'])
                if decision['mode'] == 'multigpu':
                    _validate_distributed_options(cfg)
                if cfg.route != 'native':
                    # The root sets these fields in the broadcast decision packet,
                    # also for a native route chosen from an auto request.
                    options = multigpu_options()
                    decision['service_options'] = options
                    decision.setdefault('requested_service_policy', _service_policy(options))
                    if decision['mode'] == 'multigpu':
                        decision.setdefault('selected_service_policy', _service_policy(options))
                    else:
                        decision['selected_service_policy'] = None
                decision = _decision_context(decision, cfg, context, table_sha)
                packet = dict(ok=True, decision=decision)
            except Exception as error:
                packet = dict(ok=False, error_type=type(error).__name__, error_code=_error_code(error))
        packet = control.broadcast(packet)
        if not packet['ok']:
            raise RuntimeError('MD_ROOT_ACCEPTANCE_FAILED:' + packet.get('error_code', 'MD_UNCLASSIFIED_ERROR')
                               + ':' + packet['error_type'])
        decision = _decision_context(packet['decision'], cfg, context, table_sha)
        if context.rank == 0:
            _log_route(decision)
        if decision.get('launch_required_ranks') is not None:
            successful = True
            return _launch_required(cfg, decision['launch_required_ranks'], decision['reason'], context,
                                    table_sha, recommendation=decision)
        if decision['mode'] == 'native':
            if context.rank == 0:
                result = _native_run(cfg, decision, _root_runner)
            else:
                result = dict(status='unused_rank', rank=context.rank, route='native',
                              reason=decision['reason'], cuda_initialized=False, route_decision=decision)
                logger.debug('unused rank: %s', json.dumps(result, sort_keys=True, default=str))
            if isinstance(result, dict) and cfg.debug:
                result.setdefault('debug', {}).setdefault('route_decision', decision)
            successful = True
            return result
        ranks = decision['ranks']
        physical = control.broadcast(bootstrap['physical'] if context.rank == 0 else None)
        cpu_group = control.group(ranks, 'gloo')  # all world ranks participate
        execution_policy_arguments = control.stage('EXECUTION_POLICY_SELECTION', ranks, cpu_group,
            lambda: _execution_policy_arguments(decision, cfg.route))
        service_policy = control.stage('SERVICE_POLICY_AGREEMENT', ranks, cpu_group,
            lambda: _agree_service_policy(decision['selected_service_policy'], **execution_policy_arguments))
        from hyff.multigpu.compiler_cache import configure_compiler_cache
        compiler_cache = control.stage('COMPILER_CACHE_CONFIG', ranks, cpu_group,
                                       lambda: configure_compiler_cache(context.local_rank))
        control.stage('COMPILER_CACHE_TELEMETRY', ranks, cpu_group,
                      lambda: _log_compiler_cache(context, compiler_cache))
        model = control.stage('CPU_MODEL_LOAD', ranks, cpu_group,
            lambda: _load_cpu_model(physical, service_policy=service_policy, **execution_policy_arguments))
        model = control.stage('CUDA_MODEL_BIND', ranks, cpu_group,
            lambda: _bind_cuda_model(model, physical, cfg, context, service_policy, **execution_policy_arguments))
        gpu_group = control.group(ranks, 'nccl')  # same call order on unused ranks
        if context.rank >= ranks:
            result = dict(status='unused_rank', rank=context.rank, selected_ranks=ranks,
                          reason='OUTSIDE_SELECTED_SUBGROUP', cuda_initialized=False, route_decision=decision)
            logger.debug('unused rank: %s', json.dumps(result, sort_keys=True, default=str))
            successful = True
            return result
        service = _prepare_service(model, physical, cfg, context, ranks, gpu_group,
            _dependency_root, _cache_directory, service_policy, **execution_policy_arguments)
        if context.rank == 0:
            result = _distributed_run(cfg, bootstrap, decision, service, _root_runner)
        else:
            result = service.serve()
            if service.terminal_reason != 'STOP':
                # Root propagates its failure to torchrun. Do not enter a
                # normal-completion gather that root will never join.
                raise RuntimeError('MD_WORKER_TERMINATED_WITHOUT_NORMAL_STOP')
        # This gather is reached only after ordinary STOP/ABORT/rejection return.
        # A fatal exception skips it and group destruction entirely.
        reports = control.active_gather(_memory_report(context, service, model, compiler_cache), ranks, cpu_group)
        if context.rank == 0:
            _persist_rank_reports(cfg, reports, _timeout_seconds)
            if not isinstance(result, dict):
                result = dict(status='complete', root_runner_result=result)
            if cfg.debug:
                result.setdefault('debug', {}).update(rank_reports=reports, route_decision=decision,
                                                      collective_timeout_seconds=_timeout_seconds)
        successful = True
        return result
    finally:
        if successful:
            control.close()


def _main(argv=None):
    import argparse
    from hyff.md.simulation import RunConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-json', required=True)
    args = parser.parse_args(argv)
    data = json.loads(args.config_json)
    _require(type(data) is dict, 'MD_LAUNCH_CONFIG_INVALID')
    logging.basicConfig(format='%(message)s')
    logger.setLevel(logging.INFO)
    result = run_md(RunConfig(**data))
    return 2 if result.get('status') == 'launch_required' else 0


if __name__ == '__main__':
    sys.exit(_main())
