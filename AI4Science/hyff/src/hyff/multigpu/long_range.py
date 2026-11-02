"""Rank-zero long-range latent-charge term for multi-GPU MD.

Construct RankZeroLongRange on rank zero after the device is selected. No
model installer or charge projection is called here. The inputs to evaluate
are global continuous coordinates and globally conserved charges, in strictly
increasing stable-ID order, with each atom present once and no halo/image
duplicates.

Caller protocol (this class performs NO collectives):
  1. All ranks finish the prior replay. Rank zero calls prepare outside capture.
  2. Broadcast the typed decision and reach reject/recapture consensus. REJECT
     ends the step on every rank; RECAPTURE invalidates any older CUDA graph.
  3. Gather canonical x and conserved q using the SUM-transpose convention.
     Rank zero seeds evaluate(x, q) once. Other ranks add a zero dependency on
     BOTH gathered inputs; the adapter must order all autograd collectives.
  4. Call prepare on every subsequent eager evaluation or captured replay.
     A graph replay bypasses Python evaluate, so the caller must enforce its
     generation and sigma guards before replay. Warmups/capture may evaluate
     after a successful RECAPTURE decision while building the replacement graph.

prepare can synchronize/read tensors and rebuild the root tree. evaluate only
checks tensor metadata and the sigma parameter's identity/storage/version; it
does not inspect tensor values, synchronize, rebuild, or coordinate ranks.
Native/None requests fail explicitly; they are never changed to auto.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib
import math


class LRCode(str, Enum):
    OK = 'OK'
    UNSUPPORTED_NATIVE_ROUTE = 'UNSUPPORTED_NATIVE_ROUTE'
    REQUEST_INVALID = 'REQUEST_INVALID'
    GLOBAL_ATOM_COUNT_INVALID = 'GLOBAL_ATOM_COUNT_INVALID'
    RANK_ZERO_REQUIRED = 'RANK_ZERO_REQUIRED'
    OWNED_CODEGEN_CACHE_REQUIRED = 'OWNED_CODEGEN_CACHE_REQUIRED'
    POLICY_REJECTED = 'POLICY_REJECTED'
    FROZEN_NATIVE_CUDA_TERM_REQUIRED = 'FROZEN_NATIVE_CUDA_TERM_REQUIRED'
    CURRENT_DEVICE_MISMATCH = 'CURRENT_DEVICE_MISMATCH'
    PREPARATION_OUTSIDE_CAPTURE_REQUIRED = 'PREPARATION_OUTSIDE_CAPTURE_REQUIRED'
    BACKEND_CONSTRUCTION_FAILED = 'BACKEND_CONSTRUCTION_FAILED'
    POSITION_METADATA_MISMATCH = 'POSITION_METADATA_MISMATCH'
    CHARGE_METADATA_MISMATCH = 'CHARGE_METADATA_MISMATCH'
    GLOBAL_IDS_INVALID = 'GLOBAL_IDS_INVALID'
    GLOBAL_IDS_CHANGED = 'GLOBAL_IDS_CHANGED'
    NONFINITE_GLOBAL_POSITION = 'NONFINITE_GLOBAL_POSITION'
    NONFINITE_GLOBAL_CHARGE = 'NONFINITE_GLOBAL_CHARGE'
    GLOBAL_CHARGE_NOT_CONSERVED = 'GLOBAL_CHARGE_NOT_CONSERVED'
    CHARGE_TARGET_INVALID = 'CHARGE_TARGET_INVALID'
    SIGMA_INVALID = 'SIGMA_INVALID'
    SIGMA_CHANGED_SINCE_PREPARE = 'SIGMA_CHANGED_SINCE_PREPARE'
    SPME_CELL_REQUIRED = 'SPME_CELL_REQUIRED'
    SPME_CELL_INVALID = 'SPME_CELL_INVALID'
    SPME_FIXED_CELL_CHANGED = 'SPME_FIXED_CELL_CHANGED'
    SPME_SIGMA_GUARD_REJECTED = 'SPME_SIGMA_GUARD_REJECTED'
    TREE_PREPARE_FAILED = 'TREE_PREPARE_FAILED'
    PREPARE_FAILED = 'PREPARE_FAILED'
    PREPARE_REQUIRED = 'PREPARE_REQUIRED'
    CAPTURE_GENERATION_STALE = 'CAPTURE_GENERATION_STALE'
    RECENTER_REQUIRES_PERIODIC_LR = 'RECENTER_REQUIRES_PERIODIC_LR'


class OwnedLRError(RuntimeError):
    def __init__(self, code):
        self.code = code if isinstance(code, LRCode) else LRCode.PREPARE_FAILED
        super().__init__(self.code.value)


def _require(value, code):
    if not value:
        raise OwnedLRError(code)


class LRStatus(str, Enum):
    READY = 'READY'
    RECAPTURE = 'RECAPTURE'
    REJECT = 'REJECT'


@dataclass(frozen=True)
class LRPrepareDecision:
    status: LRStatus
    code: LRCode
    generation: int
    rebuilt: bool = False
    buffers_reallocated: bool = False

    @property
    def accepted(self):
        return self.status is not LRStatus.REJECT

    def public(self):
        return {'status': self.status.value, 'code': self.code.value,
                'accepted': self.accepted, 'generation': self.generation,
                'rebuilt': self.rebuilt, 'buffers_reallocated': self.buffers_reallocated,
                'caller_must_collectively_apply_decision': True,
                'helper_performed_collectives': False}


def _import(name):
    return importlib.import_module(name)


def _sigma_stamp(term):
    parameter = term.sigma_raw
    # Internal identity/storage/version keys are never emitted in metadata.
    return id(parameter), parameter.data_ptr(), parameter._version


def _require_owned_codegen_cache(directory):
    from hyff.multigpu.cache_scope import require_cache_directory
    try:
        require_cache_directory(directory)
    except (ValueError, OSError):
        raise OwnedLRError(LRCode.OWNED_CODEGEN_CACHE_REQUIRED) from None


class RankZeroLongRange:
    """Explicit opt-in root backend; construction may allocate GPU resources.

    The runner must set its owned codegen/cache paths before construction.
    The TreeBackend constructor invokes kernel code generation. This
    helper neither selects a device nor changes process environment/import paths.
    n_global is fixed for this instance; changing N requires reconstructing it.
    """

    def __init__(self, term, *, n_global, requested='auto', rank=0, cell=None):
        _require(requested not in (None, 'native'), LRCode.UNSUPPORTED_NATIVE_ROUTE)
        _require(requested in ('auto', 'dense', 'tree', 'spme'), LRCode.REQUEST_INVALID)
        _require(type(n_global) is int and n_global > 0, LRCode.GLOBAL_ATOM_COUNT_INVALID)
        _require(type(rank) is int and rank == 0, LRCode.RANK_ZERO_REQUIRED)
        import torch
        self.torch = torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _require(torch.distributed.get_rank() == 0, LRCode.RANK_ZERO_REQUIRED)
        native = _import('hyff.models.latent_charges')
        _require(type(term) is native.LatentChargeEnergy and not term.training and
                 all(p.device.type == 'cuda' and p.dtype == torch.float32 and not p.requires_grad
                     for p in term.parameters()), LRCode.FROZEN_NATIVE_CUDA_TERM_REQUIRED)
        self.term, self.n_global = term, n_global
        self.device = term.scale.device
        _require(all(p.device == self.device for p in term.parameters()),
                 LRCode.FROZEN_NATIVE_CUDA_TERM_REQUIRED)
        _require(self.device.index == torch.cuda.current_device(), LRCode.CURRENT_DEVICE_MISMATCH)
        self._outside_capture()
        policy_module = _import('hyff.engines.capture.long_range')
        try:
            policy = policy_module.resolve_long_range('largo', requested, n_global)
        except (ValueError, TypeError, OverflowError):
            raise OwnedLRError(LRCode.POLICY_REJECTED) from None
        self.policy = {key: policy[key] for key in (
            'requested', 'selected', 'n_atoms', 'dense_through_atoms', 'selection_phase',
            'isolated_nonperiodic_sum', 'periodic_image_sum', 'extrapolation')}
        self.route = self.policy['selected']
        pair_module = _import('hyff.kernels.long_range.pair_function')
        self.pair = pair_module.LatentChargePair
        _require(term.scale.numel() == term.sigma_raw.numel() == 1 and
                 bool(torch.isfinite(term.scale).all()) and
                 bool(torch.isfinite(term.sigma).all()) and float(term.sigma.detach()) > 0,
                 LRCode.FROZEN_NATIVE_CUDA_TERM_REQUIRED)
        self._ids = self._prepared_sigma = None
        self._prepared = False
        self._neighbor_epoch_invalidated = False
        self.neighbor_epoch_invalidations = 0
        self.generation = 0
        self.prepare_calls = 0
        self.last_decision = LRPrepareDecision(LRStatus.REJECT, LRCode.PREPARE_REQUIRED, 0)
        try:
            if self.route == 'tree':
                # Verify dependency origins before the constructor executes
                # kernel code generation into the runner's configured cache.
                codegen_module = _import('hyff.kernels.long_range.codegen')
                from hyff.multigpu.cache_scope import owned_codegen_cache
                with owned_codegen_cache(codegen_module):
                    _require_owned_codegen_cache(codegen_module.GEN_DIR)
                    _import('hyff.kernels.long_range.dense')
                    tree_module = _import('hyff.kernels.long_range.tree')
                    self.backend = tree_module.TreeBackend(order=6, leaf=32, theta=0.4,
                                                           delta=1.5, rcut_sigma=6.0, headroom=1.25)
            elif self.route == 'dense':
                _import('hyff.kernels.long_range.dense')
                self.backend = pair_module.DenseBackend()
            else:
                _import('hyff.kernels.long_range.spme')
                installer_module = _import('hyff.kernels.long_range.backends')
                normalized_cell = self._cell(cell)
                self.backend = installer_module.SPMEBackend(
                    term, normalized_cell, 0.5, 6, torch.float32, 0.5, self.device)
            _import('hyff.kernels.long_range')
        except OwnedLRError:
            raise
        except Exception:
            raise OwnedLRError(LRCode.BACKEND_CONSTRUCTION_FAILED) from None

    def _outside_capture(self):
        _require(not self.torch.cuda.is_current_stream_capturing(),
                 LRCode.PREPARATION_OUTSIDE_CAPTURE_REQUIRED)

    def invalidate_neighbor_epoch(self):
        """Keep the backend, generated code and buffers; require a fresh LR plan.

        The tree route's plan depends on geometry, so reusing the backend across
        neighbor epochs must not extend that plan's lifetime: drop only the plan
        and force the ordinary prepare path to rebuild it at the new snapshot.
        Dense/SPME backends are geometry-independent. Every route requires a
        fresh prepare, and any old capture generation is invalidated on success.
        """
        self._outside_capture()
        self._prepared = False
        self._neighbor_epoch_invalidated = True
        self.neighbor_epoch_invalidations += 1
        if self.route == 'tree':
            self.backend.plan = None
        self.last_decision = LRPrepareDecision(LRStatus.REJECT, LRCode.PREPARE_REQUIRED, self.generation)

    def _position_metadata(self, positions):
        torch = self.torch
        _require(torch.is_tensor(positions) and positions.device == self.device and
                 positions.dtype in (torch.float32, torch.float64) and
                 positions.ndim == 2 and tuple(positions.shape) == (self.n_global, 3),
                 LRCode.POSITION_METADATA_MISMATCH)

    def _charge_metadata(self, charges):
        torch = self.torch
        _require(torch.is_tensor(charges) and charges.device == self.device and
                 charges.dtype == torch.float32 and charges.ndim == 1 and
                 tuple(charges.shape) == (self.n_global,), LRCode.CHARGE_METADATA_MISMATCH)

    def _cell(self, cell):
        torch = self.torch
        _require(cell is not None, LRCode.SPME_CELL_REQUIRED)
        _require(torch.is_tensor(cell) and tuple(cell.shape) in ((3, 3), (1, 3, 3)) and
                 not cell.requires_grad, LRCode.SPME_CELL_INVALID)
        # Same convention as the single-GPU _engine_cell: FP32 engine batch cell,
        # promoted to CPU FP64.
        result = cell.detach().cpu().to(torch.float32).reshape(3, 3).double()
        _require(bool(torch.isfinite(result).all()) and
                 float(torch.linalg.det(result).abs()) > 0, LRCode.SPME_CELL_INVALID)
        return result

    def prepare(self, global_positions, *, global_ids, cell=None):
        """Root eager guard/rebuild. Return a fixed typed decision, no consensus.

        Global IDs are checked strictly increasing and fixed across prepares.
        Coordinates are checked finite here, but their continuous/unwrapped
        physical convention remains the caller's responsibility.
        """
        self._prepared = False
        self.prepare_calls += 1
        rebuilt = reallocated = False
        try:
            self._outside_capture()
            self._position_metadata(global_positions)
            torch = self.torch
            _require(bool(torch.isfinite(global_positions).all()), LRCode.NONFINITE_GLOBAL_POSITION)
            _require(torch.is_tensor(global_ids) and global_ids.dtype == torch.int64 and
                     global_ids.ndim == 1 and tuple(global_ids.shape) == (self.n_global,),
                     LRCode.GLOBAL_IDS_INVALID)
            ids = global_ids.detach().cpu()
            _require(bool((ids[1:] > ids[:-1]).all()), LRCode.GLOBAL_IDS_INVALID)
            _require(self._ids is None or torch.equal(ids, self._ids), LRCode.GLOBAL_IDS_CHANGED)
            sigma = self.term.sigma.detach()
            _require(bool(torch.isfinite(sigma).all()) and float(sigma) > 0, LRCode.SIGMA_INVALID)
            if self.route == 'tree':
                try:
                    positions32 = global_positions.detach().float()
                    if self.backend.needs_rebuild(positions32, sigma):
                        torch.cuda.synchronize(self.device)
                        reallocated = bool(self.backend.rebuild(positions32, sigma))
                        rebuilt = True
                except Exception:
                    raise OwnedLRError(LRCode.TREE_PREPARE_FAILED) from None
            elif self.route == 'spme':
                self._cell(cell)
                try:
                    self.backend.check_cell({'cell': cell})
                except Exception:
                    raise OwnedLRError(LRCode.SPME_FIXED_CELL_CHANGED) from None
                try:
                    self.backend.check_sigma(self.term)
                except Exception:
                    raise OwnedLRError(LRCode.SPME_SIGMA_GUARD_REJECTED) from None
            first = self.generation == 0 or self._neighbor_epoch_invalidated
            sigma_stamp = _sigma_stamp(self.term)
            sigma_storage_changed = (self._prepared_sigma is not None and
                                     sigma_stamp[:2] != self._prepared_sigma[:2])
            if first or reallocated or sigma_storage_changed:
                self.generation += 1
            if self._ids is None:
                self._ids = ids.clone()
            self._prepared_sigma = sigma_stamp
            self._prepared = True
            self._neighbor_epoch_invalidated = False
            status = LRStatus.RECAPTURE if first or reallocated or sigma_storage_changed else LRStatus.READY
            decision = LRPrepareDecision(status, LRCode.OK, self.generation, rebuilt, reallocated)
        except OwnedLRError as error:
            decision = LRPrepareDecision(LRStatus.REJECT, error.code, self.generation, rebuilt, reallocated)
        except Exception:
            decision = LRPrepareDecision(LRStatus.REJECT, LRCode.PREPARE_FAILED,
                                         self.generation, rebuilt, reallocated)
        self.last_decision = decision
        return decision

    def validate_global_charge(self, charges, total_charge, *, absolute_bar=1e-5):
        """Optional eager physical guard; never projects or detaches live q.

        The caller must collectively propagate any returned REJECT before
        entering evaluation/backward. The bar is a charge-conservation check,
        not a numerical force-parity or term-activity acceptance bar.
        """
        try:
            self._outside_capture()
            self._charge_metadata(charges)
            _require(type(total_charge) in (int, float) and math.isfinite(total_charge) and
                     type(absolute_bar) in (int, float) and math.isfinite(absolute_bar) and absolute_bar > 0,
                     LRCode.CHARGE_TARGET_INVALID)
            _require(bool(self.torch.isfinite(charges).all()), LRCode.NONFINITE_GLOBAL_CHARGE)
            error = abs(float(charges.detach().double().sum()) - total_charge)
            _require(error <= absolute_bar, LRCode.GLOBAL_CHARGE_NOT_CONSERVED)
            return LRPrepareDecision(LRStatus.READY, LRCode.OK, self.generation)
        except OwnedLRError as error:
            self._prepared = False
            return LRPrepareDecision(LRStatus.REJECT, error.code, self.generation)
        except Exception:
            self._prepared = False
            return LRPrepareDecision(LRStatus.REJECT, LRCode.PREPARE_FAILED, self.generation)

    def assert_replay_generation(self, generation):
        """Eager caller guard before a CUDA graph replay (no rank consensus)."""
        self._outside_capture()
        _require(self._prepared, LRCode.PREPARE_REQUIRED)
        _require(type(generation) is int and generation == self.generation,
                 LRCode.CAPTURE_GENERATION_STALE)
        _require(_sigma_stamp(self.term) == self._prepared_sigma, LRCode.SIGMA_CHANGED_SINCE_PREPARE)

    def evaluate(self, global_positions, conserved_global_charge):
        """One global LR energy on rank zero, preserving x and q autograd paths.

        Static metadata and sigma-version guards only. The caller supplies a
        fresh successful prepare decision before every eager call or replay.
        No values are read on the host and no collectives are performed here.
        """
        self._position_metadata(global_positions)
        self._charge_metadata(conserved_global_charge)
        _require(self._prepared, LRCode.PREPARE_REQUIRED)
        _require(_sigma_stamp(self.term) == self._prepared_sigma, LRCode.SIGMA_CHANGED_SINCE_PREPARE)
        torch, term = self.torch, self.term
        with torch.autocast(device_type=global_positions.device.type, enabled=False):
            charge = conserved_global_charge
            if self.route == 'spme':
                energy, _parts = self.backend.mesh.energy(charge, global_positions, term.sigma)
                energy = term.coupling_ev_angstrom * term.scale.to(energy.dtype) * energy
                return energy.to(torch.float32).reshape(1)
            return self.pair.apply(global_positions.float(), charge, term.coupling_ev_angstrom * term.scale,
                                   term.sigma, self.backend)

    def check_recenter(self):
        _require(self.route == 'spme', LRCode.RECENTER_REQUIRES_PERIODIC_LR)
        return 'spme'

    def metadata(self):
        """Allowlisted policy/backend projection, outside capture only."""
        self._outside_capture()
        backend = self.backend.describe()
        if self.route == 'dense':
            description = {key: backend[key] for key in ('backend', 'block_i', 'block_j', 'num_warps')}
        elif self.route == 'tree':
            description = {key: backend[key] for key in (
                'backend', 'order', 'leaf', 'theta', 'delta', 'rcut_sigma', 'headroom',
                'rebuilds', 'reallocations', 'last_build_ms')}
            if 'plan' in backend:
                description['plan'] = {key: backend['plan'][key] for key in (
                    'atoms', 'leaf', 'leaves', 'depth', 'theta', 'delta', 'sigma', 'rcut',
                    'far_node_pairs', 'near_leaf_pairs', 'near_directed_atom_pairs',
                    'dense_directed_atom_pairs', 'far_csr', 'near_csr')}
            description['last_build_ms_scope'] = 'root_tree_rebuild_cuda_events_only'
        else:
            description = {key: backend[key] for key in (
                'method', 'mesh', 'order', 'split', 'dtype', 'volume_A3', 'cell_A', 'spacing_A',
                'h_over_sigma', 'guard_max_h_over_sigma', 'sigma_at_install', 'sigma_checked',
                'sigma_checks', 'sigma_reads')}
        return {'schema': 'multigpu_long_range', **dict(self.policy),
                'atoms': self.n_global, 'backend': description,
                'rank_zero_only': True, 'global_canonical_input_required': True,
                'continuous_coordinates_required': self.route != 'spme',
                'global_conservation_done_by_caller': True,
                'helper_performs_collectives': False, 'model_forward_modified': False,
                'long_range_energy_enabled': True, 'long_range_force_enabled': True,
                'learned_charge_chain_kept': True, 'full_model_step_validated': False,
                'production_validated': False, 'recenter_allowed': self.route == 'spme',
                'precision': ('FP32 mesh; FP64 fractional coordinates; FP32 output' if self.route == 'spme'
                              else 'FP32 pair/tree arithmetic, FP64 energy reduction'),
                'generation': self.generation, 'prepare_calls': self.prepare_calls,
                'prepared_for_evaluation': self._prepared,
                'last_prepare_decision': self.last_decision.public(),
                'coordinator_timing_measured': False}


__all__ = ['RankZeroLongRange', 'LRPrepareDecision', 'LRStatus', 'LRCode',
           'OwnedLRError']
