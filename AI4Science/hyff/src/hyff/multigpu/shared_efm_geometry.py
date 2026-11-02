"""Prepared compatibility groups and one-evaluation differentiable EFM reuse.

The prepared object holds only epoch constants/operators. Force caches are
caller-local and are never stored on it. No geometry tensor is detached,
copied, overwritten, or carried into another force evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


class SharedEFMGeometryError(ValueError):
    pass


def _require(value, code):
    if not value:
        raise SharedEFMGeometryError(code)


def _stamp(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape),
            tuple(value.stride()), value.dtype, value.device, value.requires_grad)


def _policy(geometry):
    meta = geometry.operator.metadata
    return (id(geometry.active_selector), meta.epoch, meta.generation,
            meta.n_global, meta.n_owned, meta.tile_edges, meta.dtype, meta.device,
            geometry.radius, geometry.width, geometry.rbf_cutoff,
            geometry.operator.efm.radial[0].in_features, geometry.operator.efm.frame)


@dataclass(frozen=True)
class GeometryBinding:
    geometry: object
    group: int
    policy: tuple


@dataclass(frozen=True)
class PreparedSharedEFMGeometry:
    bindings: tuple
    group_count: int

    def new_force(self, positions):
        return ForceLocalSharedEFMGeometry(self, positions)

    def metadata(self):
        return {"schema": "PREPARED_SHARED_EFM_GEOMETRY",
                "groups": self.group_count, "blocks": len(self.bindings),
                "lifetime": "ONE_ENERGY_GRAPH_RETAINED_BY_AUTOGRAD_ONLY",
                "selection_required": True, "geometry_compiled": False,
                "edge_replay_policy_changed": False,
                "cross_force_cache": False, "gpu_validated": False}


def prepare_shared_efm_geometry(geometries):
    """Preparation-only value checks; equal cloned radial buffers can share."""
    _require(isinstance(geometries, (tuple, list)) and len(geometries) > 0,
             "SHARED_EFM_NONEMPTY_GEOMETRIES_REQUIRED")
    groups = {}
    bindings = []
    identities = set()
    for geometry in geometries:
        _require(id(geometry) not in identities, "SHARED_EFM_DUPLICATE_GEOMETRY")
        identities.add(id(geometry))
        operator = geometry.operator
        operator._assert_current()
        selector = geometry.active_selector
        _require(selector is not None and getattr(operator, "_active_selector", None) is selector
                 and selector.accepts(operator), "SHARED_EFM_ACTIVE_SELECTOR_REQUIRED")
        selector.assert_current()
        _require(tuple(_stamp(value) for value in (geometry.centers, geometry.gamma,
                 geometry.translation, geometry.padding)) == geometry.buffer_stamps,
                 "SHARED_EFM_GEOMETRY_CHANGED")
        policy = _policy(geometry)
        if policy not in groups:
            groups[policy] = (len(groups), geometry)
        group, reference = groups[policy]
        _require(all(torch.equal(a, b) for a, b in zip(
                 (geometry.centers, geometry.gamma, geometry.translation, geometry.padding),
                 (reference.centers, reference.gamma, reference.translation, reference.padding))),
                 "SHARED_EFM_GROUP_CONSTANTS_MISMATCH")
        bindings.append(GeometryBinding(geometry, group, policy))
    return PreparedSharedEFMGeometry(tuple(bindings), len(groups))


class ForceLocalSharedEFMGeometry:
    """Private cache for one exact global-position autograd graph."""
    def __init__(self, prepared, positions):
        _require(isinstance(prepared, PreparedSharedEFMGeometry), "SHARED_EFM_PREPARED_REQUIRED")
        self.prepared = prepared
        self.positions = positions
        self.position_stamp = _stamp(positions)
        self.grad_enabled = torch.is_grad_enabled()
        self.entries = {}
        self.hits = 0
        self.misses = 0

    def evaluate(self, geometry, positions, *, selection):
        _require(torch.is_grad_enabled() == self.grad_enabled,
                 "SHARED_EFM_GRAD_MODE_CHANGED")
        _require(positions is self.positions and _stamp(positions) == self.position_stamp,
                 "SHARED_EFM_FORCE_FRAME_CHANGED")
        binding = next((value for value in self.prepared.bindings if value.geometry is geometry), None)
        _require(binding is not None and _policy(geometry) == binding.policy,
                 "SHARED_EFM_PREPARED_GROUP_CHANGED")
        # Every hit executes exactly the normal operator, position, geometry-
        # buffer and finalized-selection guards. No tensor values reach host.
        geometry.validate_force_inputs(positions, selection=selection)
        key = (binding.group, id(selection), selection.position_stamp,
               selection.capacity_generation, selection.allocation_capacity,
               selection.logical_count, selection.source.numel())
        if key in self.entries:
            saved_selection, values, stamps = self.entries[key]
            _require(saved_selection is selection and tuple(_stamp(value) for value in values) == stamps,
                     "SHARED_EFM_CACHED_GEOMETRY_CHANGED")
            self.hits += 1
            return values
        values = geometry.evaluate(positions, selection=selection)
        _require(isinstance(values, tuple) and len(values) == 4,
                 "SHARED_EFM_GEOMETRY_TUPLE_REQUIRED")
        self.entries[key] = (selection, values, tuple(_stamp(value) for value in values))
        self.misses += 1
        return values

    def metadata(self):
        # Scalars only: the prepared model may keep this diagnostic projection,
        # never this cache or its tensors/selection/positions.
        return {"schema": "FORCE_LOCAL_SHARED_EFM_GEOMETRY",
                "groups": self.prepared.group_count, "cache_entries": len(self.entries),
                "hits": self.hits, "misses": self.misses,
                "geometry_forward_calls": self.misses,
                "cross_force_reuse": False, "shared_output_tensors_detached": False}


__all__ = ["SharedEFMGeometryError", "PreparedSharedEFMGeometry",
           "ForceLocalSharedEFMGeometry", "prepare_shared_efm_geometry"]
