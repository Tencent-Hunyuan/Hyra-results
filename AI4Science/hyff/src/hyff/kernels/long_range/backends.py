"""Long-range installers for the largo latent-charge term: install_dense, install_tree, install_spme.

install() picks dense up to LQ_DENSE_MAX atoms (4096) and the treecode above (isolated systems; the
engine policy in hyff.engines.capture.long_range selects SPME for periodic systems).

Call order: after install_prepared_context (latent-charge prepared metadata present), before
prepare/capture. Only latent_charge_energy.forward is replaced; the native charge head and
per-graph conservation (conserved_charges) stay on the autograd tape, so forces include the
learned-charge chain dE/dq * dq/dx through the scalars. Same energy as native (isolated,
no cell, erf(r/(sqrt2 sigma))/r, coupling 14.399645*scale), MD contract: no sigma/scale
parameter gradients are needed (dE/dscale is provided, dE/dsigma is not).

Tree only: model.prepare_energy is wrapped so the prepared evaluator gains
needs_recapture(batch): if any atom moved more than delta from its plan-build position the plan
is rebuilt on the GPU into the SAME static buffers (no recapture) and False is returned; True
(recapture) only when a capacity grows. The wrapped evaluator's needs_recapture/close and all
other attributes are chained/delegated. Single graph only.

SPME (install_spme): the PERIODIC energy, the trained pair term summed over all images in the fixed
MD cell (hyff.kernels.long_range.spme.GaussianSPME, split = sigma, tin-foil, kernel-consistent
background). The same native charges (conserved_charges) and q(R) chain. Requires physical['cell'].
The cell is taken as the engine's FP32 batch cell, and a cell change fails closed at prepare. The mesh
is built from sigma at install (at most h_sigma * sigma per lattice vector and at least the B-spline
order, 2^a3^b5^c sizes). Guards, checked at install and whenever the sigma parameter's storage or
in-place version changes (host side, never inside capture): max_i |a_i|/K_i <= LQ_SPME_MAX_H_SIGMA *
sigma, and an unchanged reciprocal regime (the mesh sum is omitted when it underflows; see spme.py).

Recentering (whole-molecule lattice shifts): check_recenter(model) raises
LR_RECENTER_REQUIRES_PERIODIC_LR unless the long range is SPME or absent; an isolated sum (native,
dense, tree) changes at every shift.

Settings: hyff.engines.capture.long_range.settings() (environment LQ_*, validated before use).
"""
from __future__ import annotations

import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from hyff.kernels.long_range.pair_function import DenseBackend, LatentChargePair

_ORIGINAL_FORWARD = "_long_range_original_forward"
_ORIGINAL_PREPARE = "_long_range_original_prepare_energy"
_BACKEND = "_long_range_backend"
_KIND = "_long_range_kind"
_PREPARED_ENERGY = "_latent_charge_prepared_original_energy"
_PREPARED_METADATA = "_latent_charge_prepared_metadata"
_CALL_CONTEXT = "_latent_charge_call_context"
DEFAULT_DENSE_MAX = 4096


def _lr_forward(self, scalars, positions, atom_batch, total_charge):
    backend = self.__dict__[_BACKEND]
    with torch.autocast(device_type=positions.device.type, enabled=False):
        q = self.conserved_charges(scalars, atom_batch.long(), total_charge)
        sigma = self.sigma
        x = positions.float()
        if hasattr(backend, "rebuild") and not torch.cuda.is_current_stream_capturing():
            # Eager calls (prepared warmups/reference) own plan validity; captured replays rely
            # on needs_recapture having refreshed the plan before positions were copied in.
            if backend.needs_rebuild(x, sigma.detach()):
                torch.cuda.synchronize(x.device)
                backend.rebuild(x, sigma.detach())
        return LatentChargePair.apply(x, q, self.coupling_ev_angstrom * self.scale, sigma, backend)


class _PreparedWithPlan:
    """Prepared evaluator proxy adding tree-plan validity to the engine's recapture hook."""

    def __init__(self, inner, backend, term):
        self.__dict__.update(_inner=inner, _backend=backend, _term=term, plan_rebuilds=0)

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def needs_recapture(self, batch):
        chained = getattr(self._inner, "needs_recapture", None)
        if chained is not None and chained(batch):
            return True
        positions = batch["positions"]
        sigma = self._term.sigma.detach()
        if not self._backend.needs_rebuild(positions, sigma):
            return False
        # The previous replay may still read the plan buffers on another stream.
        torch.cuda.synchronize(positions.device)
        self.__dict__["plan_rebuilds"] += 1
        return self._backend.rebuild(positions, sigma)

    def close(self):
        chained = getattr(self._inner, "close", None)
        if chained is not None:
            chained()


def _wrapped_prepare(self, *args, **kwargs):
    inner = self.__dict__[_ORIGINAL_PREPARE](*args, **kwargs)
    term = self.latent_charge_energy
    return _PreparedWithPlan(inner, term.__dict__[_BACKEND], term)


def _engine_cell(cell):
    """The cell exactly as the engine batch holds it (FP32, system_batch), promoted to FP64."""
    if cell is None:
        raise ValueError("LR_SPME_CELL_REQUIRED")
    cell = cell.detach().cpu() if torch.is_tensor(cell) else torch.as_tensor(np.asarray(cell))
    return cell.to(torch.float32).reshape(3, 3).double()


class SPMEBackend:
    """Gaussian SPME for one graph; mesh from sigma at install, cell changes via set_cell, fail-closed h/sigma guard."""

    def __init__(self, term, cell, h_sigma, order, dtype, max_h_sigma, device):
        from hyff.kernels.long_range.spme import GaussianSPME
        if not (math.isfinite(h_sigma) and math.isfinite(max_h_sigma) and 0 < h_sigma <= max_h_sigma):
            raise ValueError("LR_SPME_H_SIGMA_OUTSIDE_GUARD")
        if not bool(torch.isfinite(cell).all()) or float(torch.linalg.det(cell).abs()) <= 0:
            raise ValueError("LR_SPME_NONSINGULAR_CELL_REQUIRED")
        self.cell, self.h_sigma, self.max_h_sigma, self.dtype = cell, float(h_sigma), float(max_h_sigma), dtype
        self.sigma_install = float(term.sigma.detach())                  # host sync: install, outside capture
        self.mesh = GaussianSPME(cell, spacing=self.h_sigma * self.sigma_install, order=order, dtype=dtype,
                                 device=device)
        self.reciprocal_omitted = self.mesh.set_sigma(self.sigma_install)
        self.spacing = [float(a) / k for a, k in zip(cell.norm(dim=1), self.mesh.mesh)]
        self.checks = self.sigma_reads = 0
        self._stamp = None
        self.check_sigma(term)

    def check_sigma(self, term):
        """Sync-free unless sigma_raw was replaced or modified in place since the last read."""
        self.checks += 1
        p = term.sigma_raw
        stamp = (id(p), p.data_ptr(), p._version)
        if stamp == self._stamp:
            return
        sigma = float(term.sigma.detach())
        ratio = max(self.spacing) / sigma
        if not ratio <= self.max_h_sigma * (1 + 1e-12):
            raise ValueError(f"LR_SPME_MESH_TOO_COARSE_FOR_SIGMA: h/sigma {ratio:.6g} > {self.max_h_sigma:g} "
                             f"(mesh {list(self.mesh.mesh)} built for sigma {self.sigma_install:.6g}, now {sigma:.6g})")
        if self.mesh.reciprocal_negligible(sigma) != self.reciprocal_omitted:
            raise ValueError(f"LR_SPME_RECIPROCAL_REGIME_CHANGED: sigma {sigma:.6g} (installed for "
                             f"{self.sigma_install:.6g}); build a new engine")
        self.sigma_reads += 1
        self._stamp, self.sigma, self.h_over_sigma = stamp, sigma, ratio

    def set_cell(self, cell, term):
        """Constant-pressure MD: the same mesh in a new cell, with the h/sigma guard rechecked."""
        cell = _engine_cell(cell)
        sigma = float(term.sigma.detach())
        spacing = [float(a) / k for a, k in zip(cell.norm(dim=1), self.mesh.mesh)]
        if not max(spacing) / sigma <= self.max_h_sigma * (1 + 1e-12):
            raise ValueError(f"LR_SPME_MESH_TOO_COARSE_FOR_SIGMA: h/sigma {max(spacing) / sigma:.6g} > "
                             f"{self.max_h_sigma:g} after a cell change; build a new engine")
        self.mesh.set_cell(cell, sigma)
        self.cell, self.spacing, self.h_over_sigma = cell, spacing, max(spacing) / sigma

    def check_cell(self, batch):
        cell = batch.get("cell") if isinstance(batch, dict) else None
        if cell is None:
            raise ValueError("LR_SPME_CELL_REQUIRED")
        if not torch.equal(_engine_cell(cell), self.cell):
            raise ValueError("LR_SPME_FIXED_CELL_CHANGED")

    def describe(self):
        return dict(self.mesh.describe(), cell_A=self.cell.tolist(), spacing_A=self.spacing,
                    reciprocal_omitted=self.reciprocal_omitted,
                    h_over_sigma=self.h_over_sigma, guard_max_h_over_sigma=self.max_h_sigma,
                    sigma_at_install=self.sigma_install, sigma_checked=self.sigma, sigma_checks=self.checks,
                    sigma_reads=self.sigma_reads)


def _spme_energy(term, backend, scalars, positions, atom_batch, total_charge):
    with torch.autocast(device_type=positions.device.type, enabled=False):
        q = term.conserved_charges(scalars, atom_batch.long(), total_charge)
        if not torch.cuda.is_current_stream_capturing():
            backend.check_sigma(term)
        energy, _ = backend.mesh.energy(q, positions, term.sigma)
        energy = term.coupling_ev_angstrom * term.scale.to(energy.dtype) * energy
        return energy.to(torch.float32).reshape(1)                      # native output: FP32, one graph


def _spme_forward(self, scalars, positions, atom_batch, total_charge):
    return _spme_energy(self, self.__dict__[_BACKEND], scalars, positions, atom_batch, total_charge)


def spme_backend(term, cell, device):
    """An SPME backend with the configured settings for one cell (also a cheap pre-compilation check)."""
    from hyff.engines.capture.long_range import settings
    values = settings()
    dtypes = {"fp32": torch.float32, "fp64": torch.float64}
    return SPMEBackend(term, _engine_cell(cell), values["LQ_SPME_H_SIGMA"], values["LQ_SPME_ORDER"],
                       dtypes[values["LQ_SPME_DTYPE"]], values["LQ_SPME_MAX_H_SIGMA"], device)


def validate_spme(term, cell):
    """Build the SPME mesh on the CPU for a cell and return its description; raises before any compilation."""
    return spme_backend(term, cell, "cpu").describe()


class periodic_term:
    """Context manager: evaluate one latent-charge term with the SPME sum in a fixed cell (eager use)."""

    def __init__(self, term, backend):
        self.term, self.backend = term, backend

    def __enter__(self):
        term, backend = self.term, self.backend
        self.saved = term.__dict__.get("forward")
        term.forward = MethodType(lambda self_, s, x, a, q: _spme_energy(self_, backend, s, x, a, q), term)
        return self

    def __exit__(self, *exc):
        if self.saved is None:
            del self.term.forward
        else:
            self.term.forward = self.saved


class _PreparedSPME:
    """Prepared evaluator proxy: fixed-cell check at prepare, sigma guard on every recapture query."""

    def __init__(self, inner, backend, term):
        self.__dict__.update(_inner=inner, _backend=backend, _term=term)

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def needs_recapture(self, batch):
        chained = getattr(self._inner, "needs_recapture", None)
        recapture = chained is not None and chained(batch)
        self._backend.check_sigma(self._term)
        return bool(recapture)

    def close(self):
        chained = getattr(self._inner, "close", None)
        if chained is not None:
            chained()


def _wrapped_prepare_spme(self, batch, *args, **kwargs):
    term = self.latent_charge_energy
    backend = term.__dict__[_BACKEND]
    backend.check_cell(batch)
    backend.check_sigma(term)
    return _PreparedSPME(self.__dict__[_ORIGINAL_PREPARE](batch, *args, **kwargs), backend, term)


def _check(model, physical, device):
    if _PREPARED_ENERGY not in model.__dict__:
        raise ValueError("LR_REQUIRES_LATENT_CHARGE_PREPARED_CONTEXT_FIRST")
    metadata = model.__dict__.get(_PREPARED_METADATA)
    if not isinstance(metadata, tuple) or len(metadata) != 2:
        raise ValueError("LR_REQUIRES_LATENT_CHARGE_PREPARED_METADATA")
    if model.__dict__.get(_CALL_CONTEXT) is not None:
        raise ValueError("LR_INSTALL_DURING_EVALUATION_UNSUPPORTED")
    term = getattr(model, "latent_charge_energy", None)
    if not isinstance(term, torch.nn.Module) or not callable(getattr(term, "conserved_charges", None)):
        raise ValueError("LR_NATIVE_LATENT_CHARGE_MODULE_REQUIRED")
    if _ORIGINAL_FORWARD in term.__dict__:
        raise ValueError("LR_ALREADY_INSTALLED")
    n = len(physical["positions"])
    atom_batch, total_charge = metadata
    if n < 1 or atom_batch.numel() != n or total_charge.numel() != 1:
        raise ValueError("LR_SINGLE_GRAPH_PREPARED_METADATA_MISMATCH")
    if atom_batch.device.type != torch.device(device).type or atom_batch.device.type != "cuda":
        raise ValueError("LR_CUDA_PREPARED_METADATA_REQUIRED")
    return term, n


def _install(model, physical, device, kind):
    from hyff.engines.capture.long_range import settings
    values = settings()
    term, n = _check(model, physical, device)
    if kind == "spme":
        if _ORIGINAL_PREPARE in model.__dict__:
            raise ValueError("LR_PREPARE_WRAPPER_ALREADY_INSTALLED")
        name = values["LQ_SPME_DTYPE"]
        backend = spme_backend(term, physical.get("cell"), device)
    elif kind == "tree":
        from hyff.kernels.long_range.tree import TreeBackend
        backend = TreeBackend(order=values["LQ_ORDER"], leaf=values["LQ_LEAF"], theta=values["LQ_THETA"],
                              delta=values["LQ_DELTA"], rcut_sigma=values["LQ_RCUT_SIGMA"],
                              headroom=values["LQ_HEADROOM"])
    else:
        backend = DenseBackend()
    term.__dict__[_BACKEND] = backend
    term.__dict__[_KIND] = kind
    term.__dict__[_ORIGINAL_FORWARD] = term.forward
    term.forward = MethodType(_spme_forward if kind == "spme" else _lr_forward, term)
    if kind == "tree":
        if _ORIGINAL_PREPARE in model.__dict__:
            raise ValueError("LR_PREPARE_WRAPPER_ALREADY_INSTALLED")
        model.__dict__[_ORIGINAL_PREPARE] = model.prepare_energy
        model.prepare_energy = MethodType(_wrapped_prepare, model)
    if kind == "spme":
        model.__dict__[_ORIGINAL_PREPARE] = model.prepare_energy
        model.prepare_energy = MethodType(_wrapped_prepare_spme, model)
    report = {
        "kind": kind,
        "full_model_step_validated": True,
        "atoms": n,
        "backend": backend.describe(),
        "long_range_energy_enabled": True,
        "long_range_force_enabled": True,
        "charge_projection": "native conserved_charges on the autograd tape (learned-charge chain kept)",
        "energy": "isolated latent-charge sum: sum_{i<j} q_i q_j erf(r/(sqrt2 sigma))/r * 14.399645*scale",
        "precision": "FP32 pair/tree arithmetic, FP64 energy reduction",
        "parameter_gradients": "dE/dscale yes; dE/dsigma not provided (MD only)",
        "plan_validity": ("max displacement <= delta, GPU rebuild in place, recapture only on capacity growth"
                          if kind == "tree" else "none (plan-free)"),
        "recenter_allowed": kind == "spme",
    }
    if kind == "spme":
        report.update(
            energy=("periodic latent-charge sum (not the isolated sum): 1/2 sum_ij sum'_n q_i q_j erf(|r_ij+n|/(sqrt2 "
                    "sigma))/|r_ij+n| minus k=0, * 14.399645*scale; fixed cell, tin-foil, background -pi Q^2/(2 "
                    "alpha^2 V) + pi sigma^2 Q^2/V at alpha = 1/(sqrt2 sigma) (= 0 at split sigma)"),
            extrapolation=("trained on isolated molecules; the periodic image sum is outside the training "
                           "distribution"),
            precision=f"{name.upper()} mesh (B-spline weights, FFT); FP64 fractional coordinates; FP32 output",
            parameter_gradients="dE/dscale yes; dE/dsigma exact through the mesh (MD needs neither)",
            plan_validity=("fixed cell (prepare fails closed on a cell change); mesh fixed at install; h/sigma "
                           "guard on sigma parameter changes, sync-free otherwise"))
    return report


def install_dense(model, physical, device):
    return _install(model, physical, device, "dense")


def install_tree(model, physical, device):
    return _install(model, physical, device, "tree")


def install_spme(model, physical, device):
    """Periodic LR in the fixed MD cell (the engine policy's choice for periodic systems)."""
    return _install(model, physical, device, "spme")


def install(model, physical, device):
    from hyff.engines.capture.long_range import settings
    n = len(physical["positions"])
    kind = "dense" if n <= settings()["LQ_DENSE_MAX"] else "tree"
    return _install(model, physical, device, kind)


def lr_kind(model):
    """'spme' | 'dense' | 'tree' when installed here, 'native' for the module's own isolated sum, None without a latent-charge term."""
    term = getattr(model, "latent_charge_energy", None)
    if term is None:
        return None
    return term.__dict__.get(_KIND, "native")


def set_cell(model, cell):
    """Constant-pressure MD: move an installed SPME sum to a new cell; returns the long-range kind.

    No-op without a latent-charge term; an isolated sum (dense, tree, native) has no cell and is refused.
    """
    kind = lr_kind(model)
    if kind is None:
        return None
    if kind != "spme":
        raise ValueError(f"LR_CELL_CHANGE_REQUIRES_SPME: long range is {kind} (isolated sum)")
    term = model.latent_charge_energy
    term.__dict__[_BACKEND].set_cell(cell, term)
    return kind


def check_recenter(model):
    """Guard for whole-molecule recentering (lattice shifts; RunConfig.recenter_interval_ps).

    A lattice shift leaves the periodic SPME energy unchanged but makes an isolated sum jump (for example several
    eV over 2 ps of NVE, all of it at the shifts). Returns the kind when allowed (SPME or no latent-charge term).
    """
    kind = lr_kind(model)
    if kind not in ("spme", None):
        raise ValueError(f"LR_RECENTER_REQUIRES_PERIODIC_LR: long range is {kind} (isolated sum); "
                         "install_spme, or keep continuous coordinates (recenter off)")
    return kind


def plan_statistics(model):
    """Current backend description (rebuild counts, plan sizes), or None."""
    term = getattr(model, "latent_charge_energy", None)
    backend = None if term is None else term.__dict__.get(_BACKEND)
    return None if backend is None else backend.describe()
