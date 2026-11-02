"""Automatic GPU-count rule for multi-GPU largo MD, from measured MD timings.

Measured on NVIDIA H20 (1 fs NVE, chunk 100, skin and reserve 1.5; wall time of the timed run divided by
its steps): at 103,440 atoms one GPU (single-GPU largo engine with its defaults) took 6,347 ms/step and the
multi-GPU service (fp32, dense long range) 2,207 ms/step on four GPUs and 1,173 ms/step on eight; at 2,586
atoms one GPU took 121 ms/step and two GPUs 301 ms/step. The rule therefore selects eight GPUs (else four)
at or above 103,440 atoms on H20-class devices and a single GPU otherwise. Two GPUs are never selected
automatically. Timings above the measured size are a forecast, not a guarantee.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math


MEASURED_ATOMS = 103440


@dataclass(frozen=True)
class MeasuredRow:
    atoms: int
    ranks: int
    route: str
    mean_ms: float
    median_ms: float


# 103,440 atoms: one timed run each (150 steps on one GPU, 300 on four and eight), so mean = median.
MEASURED_TABLE = (
    MeasuredRow(103440, 1, 'native', 6347.409625540798, 6347.409625540798),
    MeasuredRow(103440, 4, 'multigpu', 2206.9099989238507, 2206.9099989238507),
    MeasuredRow(103440, 8, 'multigpu', 1173.1695027758058, 1173.1695027758058),
    MeasuredRow(2586, 1, 'native', 120.96674870699644, 120.96674870699644),
    MeasuredRow(2586, 2, 'multigpu', 300.8367072045803, 155.8677051216364),
)
# Service options the timings were measured with; skipping inactive base/pair rows (on or off) is
# assumed to inherit the same timings.
_SERVICE_FIXED = dict(
    node_precision='fp32', efm_precision='fp32', parameter_dtype='float32', grouped_experts=False,
    compile_efm_tiles=True, active_efm_compaction=True, efm_tile_edges=32768,
    float32_matmul_precision='highest', cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
    compiled_nodes=False, joined_projections=False, folded_first_projections=False, frozen_layernorm=False,
    fused_local_features=False, learned_weights_changed=False)


def measured_table():
    return [asdict(row) for row in MEASURED_TABLE]


def _get(value, key, default=None):
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _number(value, expected):
    return type(value) in (int, float) and math.isfinite(value) and value == expected


def _policy_matches(value):
    if not isinstance(value, Mapping):
        return False
    actual = dict(value)
    if type(actual.pop('active_base_pair_compaction', None)) is not bool:
        return False
    return actual == _SERVICE_FIXED and all(type(actual[key]) is type(item) for key, item in _SERVICE_FIXED.items())


def _base_scope_error(cfg, physical, service_policy):
    if _get(cfg, 'model') != 'largo':
        return 'MD_CARD_GUIDANCE_MODEL_UNMEASURED'
    if not _policy_matches(service_policy):
        return 'MD_CARD_GUIDANCE_SERVICE_POLICY_UNMEASURED'
    if (_get(cfg, 'engine') not in (None, 'largo') or
            _get(cfg, 'precision') not in (None, 'fp32') or
            _get(cfg, 'statistics') not in (None, 'fp64')):
        return 'MD_CARD_GUIDANCE_OPTIONS_UNMEASURED'
    if (_get(cfg, 'long_range') not in (None, 'dense') or
            not _number(_get(cfg, 'timestep_fs'), 1.0) or
            type(_get(cfg, 'chunk')) is not int or _get(cfg, 'chunk') != 100 or
            not _number(_get(cfg, 'skin'), 1.5) or
            not _number(_get(cfg, 'reserve'), 1.5) or
            _get(cfg, 'hmr_mass') is not None or
            _get(cfg, 'inclusive_hmr', False) is not False or
            not _number(_get(cfg, 'tau_ps'), 0.0)):
        return 'MD_CARD_GUIDANCE_MD_SETTINGS_UNMEASURED'
    if not isinstance(physical, Mapping):
        return 'MD_CARD_GUIDANCE_SYSTEM_UNAVAILABLE'
    if (physical.get('native_engine') != 'largo' or
            physical.get('native_precision') != 'tf32x3' or
            physical.get('native_statistics') not in (None, 'fp64')):
        return 'MD_CARD_GUIDANCE_SINGLE_GPU_ROUTE_UNMEASURED'
    return None


def request_scope_error(cfg, physical, *, service_policy=None):
    """None when the automatic rule applies to this request, else a reason code."""
    error = _base_scope_error(cfg, physical, service_policy)
    if error is not None:
        return error
    if type(physical.get('atoms')) is not int or physical['atoms'] < MEASURED_ATOMS:
        return 'MD_CARD_GUIDANCE_SIZE_BELOW_MEASURED'
    return None


def _device_status(device):
    if device is None:
        return 'DEVICE_UNVERIFIED'
    if not isinstance(device, Mapping):
        return 'DEVICE_NOT_MEASURED'
    name = device.get('device_class')
    capability = device.get('capability')
    if (name in ('NVIDIA H20', 'H20') and isinstance(capability, (tuple, list)) and
            len(capability) == 2 and tuple(capability) == (9, 0)):
        return 'MEASURED_DEVICE_CLASS'
    return 'DEVICE_NOT_MEASURED'


def _decision(ranks, reason, device_status, *, requested_atoms=None):
    mode = 'native' if ranks == 1 else 'multigpu'
    in_range = type(requested_atoms) is int and requested_atoms >= MEASURED_ATOMS
    return dict(mode=mode, route=mode, ranks=ranks, reason=reason, timing_guaranteed=False,
                device_status=device_status, measured_atoms=MEASURED_ATOMS,
                requested_atoms=requested_atoms,
                timing_basis=('forecast above the measured size' if in_range and requested_atoms > MEASURED_ATOMS
                              else 'measured' if in_range else 'below the measured size'))


def select_card_count(cfg, physical, available_ranks, device=None, *, service_policy=None):
    """8 GPUs if available, else 4, at or above the measured size; otherwise one GPU."""
    status = _device_status(device)
    atoms = physical.get('atoms') if isinstance(physical, Mapping) else None
    atoms = atoms if type(atoms) is int else None
    error = request_scope_error(cfg, physical, service_policy=service_policy)
    if error is not None:
        return _decision(1, error, status, requested_atoms=atoms)
    if status != 'MEASURED_DEVICE_CLASS':
        return _decision(1, 'MD_CARD_GUIDANCE_' + status, status, requested_atoms=atoms)
    try:
        counts = tuple(available_ranks)
    except TypeError:
        return _decision(1, 'MD_CARD_GUIDANCE_AVAILABLE_RANKS_INVALID', status, requested_atoms=atoms)
    rows = tuple(row for row in MEASURED_TABLE if row.atoms == MEASURED_ATOMS)
    for count in (8, 4):
        if count not in counts:
            continue
        choice = next(row for row in rows if row.ranks == count)
        smaller = tuple(row for row in rows if row.ranks < count)
        if all(choice.mean_ms < row.mean_ms and choice.median_ms < row.median_ms for row in smaller):
            return _decision(count, 'MEASURED_FASTER_CARD_COUNT', status, requested_atoms=atoms)
    return _decision(1, 'NO_MEASURED_FASTER_AVAILABLE_CARD_COUNT', status, requested_atoms=atoms)


def explicit_warning(cfg, physical, device=None, *, service_policy=None):
    """Advice for an explicit GPU count; the request itself is never changed."""
    if _get(cfg, 'route') != 'multigpu':
        return None
    ranks = _get(cfg, 'ranks')
    error = _base_scope_error(cfg, physical, service_policy)
    atoms = physical.get('atoms') if isinstance(physical, Mapping) else None
    if error is None and atoms is not None and atoms < MEASURED_ATOMS and ranks == 2:
        message = ('At 2,586 atoms one GPU was faster than two GPUs (121 vs 301 ms/step); '
                   'multi-GPU MD pays off only for very large systems.')
        recommended = 1
    elif error is None and atoms == MEASURED_ATOMS and ranks in (4, 8):
        return None
    else:
        message = ('No measured timing shows this GPU count to be faster than fewer GPUs for this '
                   'system; more GPUs can be slower. Running as requested.')
        recommended = None
    return dict(code='MULTIGPU_SPEED_ADVICE', message=message, requested_ranks=ranks,
                recommended_ranks=recommended, request_unchanged=True)
