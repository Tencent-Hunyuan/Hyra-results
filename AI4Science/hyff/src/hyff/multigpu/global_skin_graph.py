"""Full periodic image skin graph from the nvalchemiops cell-list query, run by the caller.

Uses the same nvalchemiops cell-list estimator, spatial buffers, neighbor_list,
and matrix-to-COO conversion as hyff.graph.cell_list_edges. This module adds
bounded capacity growth, signed-int32/resource checks, checked counts, exact
native integer shifts, and deterministic CPU ordering. It does not modify or
call the unbounded retry loop of cell_list_edges.

Supports fully periodic, positive-volume triclinic cells, including multiple
images and nonzero self images. Exact FP32 input values are promoted to FP64
for the native query; the query radius includes a small positive margin (see
preflight_global_skin_graph). Completeness relies on the native cell-list
contract and successful overflow checks. CUDA cutoff rounding at the boundary
is not validated here.

No module-level numerical imports, process groups, device selection, model
binding, force evaluation, capture, or remote execution occur here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


_INT32_SAFE = 2**29
_SPATIAL_FIELDS = ("cells_per_dimension", "neighbor_search_radius", "atom_periodic_shifts",
                   "atom_to_cell_mapping", "atoms_per_cell_count", "cell_atom_start_indices", "cell_atom_list")


class GlobalSkinGraphError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise GlobalSkinGraphError(code)


def _native_modules(dependency_root=None):
    from nvalchemiops.torch.neighbors import neighbor_list
    from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes
    from nvalchemiops.torch.neighbors.neighbor_utils import (
        allocate_cell_list, get_neighbor_list_from_neighbor_matrix,
    )
    return neighbor_list, estimate_cell_list_sizes, allocate_cell_list, get_neighbor_list_from_neighbor_matrix


@dataclass(frozen=True)
class GlobalSkinGraphInput:
    positions: object
    cell: object
    global_ids: object
    radii: tuple[float, ...]
    skin: float
    supplied_radius: float
    query_radius: float
    query_padding: float
    initial_capacity: int
    max_capacity: int
    max_attempts: int
    max_matrix_entries: int
    max_nbins: int
    maximum_possible_degree: int
    _versions: tuple

    def assert_integrity(self):
        _require(all(value._version == version for value, version in self._versions),
                 "GLOBAL_SKIN_INPUT_BUFFER_CHANGED")


@dataclass(frozen=True)
class GlobalSkinGraph:
    reference_global_positions: object
    cell: object
    global_ids: object
    edge_index: object
    integer_shifts: object
    supplied_radius: float
    query_radius: float
    radii: tuple[float, ...]
    skin: float
    attempts: tuple
    capacity: int
    maximum_degree: int
    spatial_bins: int
    _versions: tuple

    def assert_integrity(self):
        _require(all(value._version == version for value, version in self._versions),
                 "GLOBAL_SKIN_RESULT_BUFFER_CHANGED")

    def metadata(self):
        self.assert_integrity()
        return {"schema": "LARGO_GLOBAL_SKIN_GRAPH", "global_atoms": len(self.global_ids),
                "global_edges": self.edge_index.shape[1], "supplied_radius_A": self.supplied_radius,
                "native_query_radius_A": self.query_radius, "query_padding_A": self.query_radius - self.supplied_radius,
                "operator_radii_A": list(self.radii), "skin_A": self.skin,
                "native_backend": "PINNED_NVALCHEMIOPS_FULL_CELL_LIST", "native_query_dtype": "float64",
                "reference_geometry_dtype": "float32", "periodic_axes": [True, True, True],
                "cell_scope": "POSITIVE_VOLUME_TRICLINIC_CONDITION_AT_MOST_10000",
                "completeness_attestation": "PINNED_NATIVE_FULL_QUERY_CONTRACT_WITH_CHECKED_OVERFLOW",
                "native_integer_shifts_reused_exactly": True, "reciprocal_images_verified": True,
                "duplicate_images_rejected": True, "zero_shift_self_edges_rejected": True,
                "edge_order": "DESTINATION_SOURCE_INTEGER_SHIFT_X_Y_Z",
                "attempts": [{"capacity": capacity, "actual_maximum_degree": required}
                             for capacity, required in self.attempts],
                "final_capacity": self.capacity, "maximum_degree": self.maximum_degree,
                "spatial_bins": self.spatial_bins, "bounded_native_calls": True,
                "geometric_all_pairs_reference_run": False, "gpu_cutoff_boundary_rounding_validated": False,
                "gpu_validated": False, "capture_validated": False, "performance_measured": False}


def preflight_global_skin_graph(global_positions, cell, *, global_ids, radii, skin,
                                initial_capacity=None, max_capacity=16384, max_attempts=8,
                                max_matrix_entries=2**28):
    """CPU-only input checks; no native module import or CUDA operation.

    Native bin products, unwrapped bin coordinates, image shifts and actual
    row-count upper bounds must fit comfortably within signed int32. Extremely
    skew/ill-conditioned cells or coordinate scales are explicitly rejected.
    """
    import torch
    _require(isinstance(global_positions, torch.Tensor) and global_positions.device.type == "cpu" and
             global_positions.dtype == torch.float32 and global_positions.ndim == 2 and
             global_positions.shape[1] == 3 and not global_positions.requires_grad,
             "GLOBAL_SKIN_FP32_CPU_POSITIONS_REQUIRED")
    n = len(global_positions)
    _require(0 < n < _INT32_SAFE, "GLOBAL_SKIN_ATOM_COUNT_OUTSIDE_SUPPORTED_RANGE")
    _require(isinstance(cell, torch.Tensor) and cell.device.type == "cpu" and cell.dtype == torch.float32 and
             cell.shape == (3, 3) and not cell.requires_grad, "GLOBAL_SKIN_FP32_CPU_CELL_REQUIRED")
    _require(isinstance(global_ids, torch.Tensor) and global_ids.device.type == "cpu" and
             global_ids.dtype == torch.int64 and global_ids.shape == (n,), "GLOBAL_SKIN_CANONICAL_IDS_REQUIRED")
    _require(bool((global_ids[1:] > global_ids[:-1]).all()), "GLOBAL_SKIN_CANONICAL_IDS_REQUIRED")
    _require(bool(torch.isfinite(global_positions).all() & torch.isfinite(cell).all()), "GLOBAL_SKIN_NONFINITE_GEOMETRY")
    values = tuple(radii)
    _require(bool(values) and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values),
             "GLOBAL_SKIN_RADII_INVALID")
    _require(type(skin) in (int, float) and math.isfinite(skin) and skin >= 0, "GLOBAL_SKIN_SKIN_INVALID")
    supplied = max(values) + skin
    _require(math.isfinite(supplied), "GLOBAL_SKIN_RADIUS_OVERFLOW")
    _require(all(type(value) is int and value > 0 for value in (max_capacity, max_attempts, max_matrix_entries)) and
             max_capacity < _INT32_SAFE and max_attempts <= 32 and max_matrix_entries < _INT32_SAFE,
             "GLOBAL_SKIN_RESOURCE_LIMIT_INVALID")
    with torch.no_grad():
        positions = global_positions.detach().clone().contiguous()
        fixed_cell = cell.detach().clone().contiguous()
        ids = global_ids.detach().clone().contiguous()
        cell64 = fixed_cell.double()
        determinant = float(torch.linalg.det(cell64))
        _require(math.isfinite(determinant) and determinant > 0, "GLOBAL_SKIN_POSITIVE_VOLUME_REQUIRED")
        condition = float(torch.linalg.cond(cell64))
        _require(math.isfinite(condition) and condition <= 10000, "GLOBAL_SKIN_CELL_CONDITION_UNSUPPORTED")
        # Deliberately larger than FP64 arithmetic roundoff. This is a numerical
        # safety margin; it is not checked against an all-pairs reference.
        scale = max(1.0, supplied, float(positions.abs().max()), float(fixed_cell.abs().max()))
        padding = max(1.0e-6, 64.0 * 2.0**-23 * scale)
        query = math.nextafter(supplied + padding, math.inf)
        _require(math.isfinite(query) and query <= 1.25 * supplied,
                 "GLOBAL_SKIN_COORDINATE_SCALE_PADDING_EXCESSIVE")
        inverse = torch.linalg.inv(cell64)
        heights = 1.0 / inverse.norm(dim=0)
        fractional = positions.double() @ inverse
        max_fractional = fractional.abs().max(dim=0).values
        # Round bounds outward by whole integer cells, then keep a factor-four
        # margin from the library's signed-int32 limits before any native call.
        upper_bins = [max(1, math.ceil(float(height) / query) + 1) for height in heights]
        _require(math.prod(upper_bins) < _INT32_SAFE, "GLOBAL_SKIN_RAW_BIN_PRODUCT_INT32_RISK")
        image_span = [math.ceil(query / float(height)) + 2 for height in heights]
        _require(all((float(frac) + 2) * bins < _INT32_SAFE for frac, bins in zip(max_fractional, upper_bins)),
                 "GLOBAL_SKIN_UNWRAPPED_BIN_INDEX_INT32_RISK")
        _require(all(2 * (float(frac) + 2) + span < _INT32_SAFE for frac, span in zip(max_fractional, image_span)),
                 "GLOBAL_SKIN_IMAGE_SHIFT_INT32_RISK")
        maximum_degree = n * math.prod(2 * span + 1 for span in image_span)
        _require(maximum_degree < _INT32_SAFE, "GLOBAL_SKIN_ACTUAL_COUNT_INT32_RISK")
        capacity = (max(32, math.ceil(0.15 * 4 * math.pi / 3 * query**3 / 32) * 32)
                    if initial_capacity is None else initial_capacity)
        _require(type(capacity) is int and 0 < capacity <= max_capacity and n * capacity <= max_matrix_entries,
                 "GLOBAL_SKIN_INITIAL_CAPACITY_OUTSIDE_LIMIT")
    buffers = (positions, fixed_cell, ids)
    return GlobalSkinGraphInput(positions, fixed_cell, ids, tuple(map(float, values)), float(skin), float(supplied),
        query, query - supplied, capacity, max_capacity, max_attempts, max_matrix_entries,
        max(8192, n), maximum_degree, tuple((value, value._version) for value in buffers))


def _checked_query_with_growth(torch, prepared, query):
    """Bounded capacity-growth loop; query(capacity) runs the native neighbor query."""
    prepared.assert_integrity()
    capacity = prepared.initial_capacity
    history = []
    n = len(prepared.positions)
    for _ in range(prepared.max_attempts):
        _require(capacity <= prepared.max_capacity and n * capacity <= prepared.max_matrix_entries,
                 "GLOBAL_SKIN_GROWTH_RESOURCE_LIMIT")
        result = query(capacity)
        _require(isinstance(result, (tuple, list)) and len(result) == 3, "GLOBAL_SKIN_NATIVE_RESULT_INVALID")
        matrix, counts, shifts = result
        _require(all(isinstance(value, torch.Tensor) for value in result) and
                 matrix.dtype == counts.dtype == shifts.dtype == torch.int32 and
                 matrix.device == counts.device == shifts.device and matrix.shape == (n, capacity) and
                 counts.shape == (n,) and shifts.shape == (n, capacity, 3), "GLOBAL_SKIN_NATIVE_RESULT_INVALID")
        _require(bool((counts >= 0).all()), "GLOBAL_SKIN_NATIVE_COUNT_INVALID")
        required = int(counts.max())
        _require(required <= prepared.maximum_possible_degree, "GLOBAL_SKIN_NATIVE_COUNT_EXCEEDS_BOUND")
        history.append((capacity, required))
        if required <= capacity:
            _require(int(counts.sum(dtype=torch.int64)) < 2**31, "GLOBAL_SKIN_COO_PREFIX_INT32_RISK")
            mask = matrix != n
            _require(torch.equal(mask.sum(dim=1).to(torch.int32), counts), "GLOBAL_SKIN_MATRIX_COUNT_MISMATCH")
            _require(bool(((matrix[mask] >= 0) & (matrix[mask] < n)).all()), "GLOBAL_SKIN_NATIVE_ENDPOINT_INVALID")
            return matrix, counts, shifts, tuple(history)
        capacity = max(capacity * 2, ((required + 31) // 32) * 32)
        del matrix, counts, shifts, result
    raise GlobalSkinGraphError("GLOBAL_SKIN_RETRY_LIMIT_EXHAUSTED")


def _canonical_cpu_graph(torch, native_edges, native_shifts, counts, n_global):
    """Exact integer conversion and lexicographic identity checks; no geometry."""
    import numpy as np
    _require(isinstance(native_edges, torch.Tensor) and isinstance(native_shifts, torch.Tensor) and
             native_edges.dtype == native_shifts.dtype == torch.int32 and native_edges.ndim == 2 and
             native_edges.shape[0] == 2 and native_shifts.shape == (native_edges.shape[1], 3),
             "GLOBAL_SKIN_NATIVE_COO_INVALID")
    edges = native_edges.detach().flip(0).to(device="cpu", dtype=torch.int64).contiguous()
    shifts = native_shifts.detach().to(device="cpu", dtype=torch.int64).contiguous()
    counts = counts.detach().to(device="cpu", dtype=torch.int64)
    _require(counts.shape == (n_global,) and int(counts.sum()) == edges.shape[1] and
             bool(((edges >= 0) & (edges < n_global)).all()), "GLOBAL_SKIN_COO_COUNT_OR_ENDPOINT_INVALID")
    _require(torch.equal(torch.bincount(edges[1], minlength=n_global), counts), "GLOBAL_SKIN_COO_DEGREE_MISMATCH")
    _require(bool(((edges[0] != edges[1]) | (shifts != 0).any(dim=1)).all()), "GLOBAL_SKIN_ZERO_SHIFT_SELF_EDGE")
    src, dst, image = edges[0].numpy(), edges[1].numpy(), shifts.numpy()
    order = np.lexsort((image[:, 2], image[:, 1], image[:, 0], src, dst))
    edges = torch.from_numpy(edges.numpy()[:, order].copy()).contiguous()
    shifts = torch.from_numpy(image[order].copy()).contiguous()
    _require(not bool(((edges[:, 1:] == edges[:, :-1]).all(dim=0) &
                       (shifts[1:] == shifts[:-1]).all(dim=1)).any()), "GLOBAL_SKIN_DUPLICATE_IMAGE")
    # Sort exact reversed integer records independently; native writes both
    # orientations even for self images. Do not manufacture missing reverses.
    src, dst, image = edges[0].numpy(), edges[1].numpy(), shifts.numpy()
    reverse = np.lexsort((-image[:, 2], -image[:, 1], -image[:, 0], dst, src))
    chunk = 262144
    for start in range(0, len(reverse), chunk):
        stop = min(start + chunk, len(reverse))
        selected = reverse[start:stop]
        _require(np.array_equal(src[start:stop], dst[selected]) and
                 np.array_equal(dst[start:stop], src[selected]) and
                 np.array_equal(image[start:stop], -image[selected]), "GLOBAL_SKIN_RECIPROCAL_IMAGE_MISSING")
    return edges, shifts


def _query_global_skin_graph_native(prepared, *, device, dependency_root):
    """Bounded native query shared by both graph builders; E-sized results stay on the device.

    The CPU-reference builder and the device builder use the same native
    arguments, overflow/growth checks and COO conversion; only the handling of
    the result differs.
    """
    prepared.assert_integrity()
    import torch
    device = torch.device(device)
    _require(device.type == "cuda" and device.index is not None and device.index == torch.cuda.current_device() and
             not torch.cuda.is_current_stream_capturing(), "GLOBAL_SKIN_EAGER_CURRENT_CUDA_DEVICE_REQUIRED")
    neighbor_list, estimate, allocate, convert = _native_modules(dependency_root)
    with torch.no_grad():
        positions = prepared.positions.to(device=device, dtype=torch.float64)
        query_cell = prepared.cell.to(device=device, dtype=torch.float64).unsqueeze(0)
        pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
        total_cells, search_radius = estimate(query_cell, pbc.squeeze(0), prepared.query_radius,
                                              max_nbins=prepared.max_nbins)
        _require(type(total_cells) is int and 0 < total_cells <= prepared.max_nbins and
                 search_radius.shape == (3,) and search_radius.dtype == torch.int32 and
                 bool(((search_radius >= 0) & (search_radius < _INT32_SAFE)).all()),
                 "GLOBAL_SKIN_NATIVE_SPATIAL_ESTIMATE_INVALID")
        storage = allocate(len(positions), total_cells, search_radius, device)
        _require(len(storage) == len(_SPATIAL_FIELDS), "GLOBAL_SKIN_NATIVE_SPATIAL_STORAGE_INVALID")
        kwargs = dict(zip(_SPATIAL_FIELDS, storage))

        def query(capacity):
            return neighbor_list(positions, prepared.query_radius, cell=query_cell, pbc=pbc,
                method="cell_list", half_fill=False, fill_value=len(positions), return_neighbor_list=False,
                max_neighbors=capacity, **kwargs)

        matrix, counts, shifts, history = _checked_query_with_growth(torch, prepared, query)
        native_edges, _, native_shifts = convert(matrix, num_neighbors=counts,
                                                neighbor_shift_matrix=shifts, fill_value=len(positions))
        # Matrix/spatial scratch is not retained through canonical sorting or
        # local-view allocation; only the checked device COO and O(N) counts.
        del matrix, shifts, storage, kwargs
    return native_edges, native_shifts, counts, history, total_cells


def prepare_global_skin_graph(global_positions, cell, *, global_ids, radii, skin, device, dependency_root,
                              initial_capacity=None, max_capacity=16384, max_attempts=8,
                              max_matrix_entries=2**28):
    """Build the CPU-reference graph from the bounded native query."""
    prepared = preflight_global_skin_graph(global_positions, cell, global_ids=global_ids, radii=radii, skin=skin,
        initial_capacity=initial_capacity, max_capacity=max_capacity, max_attempts=max_attempts,
        max_matrix_entries=max_matrix_entries)
    native_edges, native_shifts, counts, history, total_cells = _query_global_skin_graph_native(
        prepared, device=device, dependency_root=dependency_root)
    import torch
    with torch.no_grad():
        edges, images = _canonical_cpu_graph(torch, native_edges, native_shifts, counts, len(prepared.positions))
    buffers = (prepared.positions, prepared.cell, prepared.global_ids, edges, images)
    return GlobalSkinGraph(*buffers, prepared.supplied_radius, prepared.query_radius, prepared.radii, prepared.skin,
        history, history[-1][0], history[-1][1], total_cells, tuple((value, value._version) for value in buffers))


@dataclass(frozen=True)
class DeviceGlobalSkinGraph:
    """Root-only resident topology plus the actual bounded native-query facts."""
    snapshot: object
    attempts: tuple
    capacity: int
    maximum_degree: int
    spatial_bins: int
    query_radius: float
    supplied_radius: float

    def assert_integrity(self):
        self.snapshot.assert_integrity()

    def metadata(self):
        self.assert_integrity()
        return {"schema": "LARGO_DEVICE_GLOBAL_SKIN_GRAPH",
            "global_edges": self.snapshot.header["global_edge_count"],
            "capacity": self.capacity, "maximum_degree": self.maximum_degree,
            "attempts": len(self.attempts), "growths": len(self.attempts) - 1,
            "capacity_history": [list(pair) for pair in self.attempts],
            "spatial_bins": self.spatial_bins, "native_query_radius_A": self.query_radius,
            "supplied_radius_A": self.supplied_radius,
            "native_backend": "PINNED_NVALCHEMIOPS_FULL_CELL_LIST",
            "native_query_dtype": "float64", "reference_geometry_dtype": "float32",
            "graph_residency": "ROOT_DEVICE_TOPOLOGY_LOCAL_DEVICE_VIEWS",
            "host_graph_payload": "NONE_COUNTS_AND_OWNERS_ONLY",
            "graph_certificate": dict(self.snapshot.header["certificate"]),
            "bounded_native_calls": True, "gpu_validated": False}


def prepare_global_skin_graph_device(global_positions, cell, *, global_ids, radii, skin,
                                     device, dependency_root, world_size, epoch=0, generation=1,
                                     initial_capacity=None, max_capacity=16384, max_attempts=8,
                                     max_matrix_entries=2**28, chunk_size=262144):
    """Device-resident variant: the same query, then device epoch preparation.

    No E-sized tensor is transferred to CPU. Only reference geometry and the
    O(R*N) counts/ownership header cross the host boundary. Membership uses
    explicit eager FP64 separate-operation squared distance in both the CPU
    reference and device path; there is no ambiguity fallback.
    """
    prepared = preflight_global_skin_graph(global_positions, cell, global_ids=global_ids,
        radii=radii, skin=skin, initial_capacity=initial_capacity, max_capacity=max_capacity,
        max_attempts=max_attempts, max_matrix_entries=max_matrix_entries)
    native_edges, native_shifts, counts, history, total_cells = _query_global_skin_graph_native(
        prepared, device=device, dependency_root=dependency_root)
    from hyff.multigpu.device_neighbor_epoch import prepare_device_neighbor_root
    snapshot = prepare_device_neighbor_root(native_edges, native_shifts, counts,
        prepared.positions, prepared.cell, global_ids=prepared.global_ids, radii=prepared.radii,
        skin=prepared.skin, supplied_radius=prepared.supplied_radius, world_size=world_size,
        epoch=epoch, generation=generation, padding_multiple=8, reserve_factor=1.2,
        chunk_size=chunk_size)
    return DeviceGlobalSkinGraph(snapshot, history, history[-1][0], history[-1][1], total_cells,
                                prepared.query_radius, prepared.supplied_radius)


__all__ = ["GlobalSkinGraphError", "GlobalSkinGraphInput", "GlobalSkinGraph", "preflight_global_skin_graph",
           "prepare_global_skin_graph", "DeviceGlobalSkinGraph",
           "prepare_global_skin_graph_device"]
