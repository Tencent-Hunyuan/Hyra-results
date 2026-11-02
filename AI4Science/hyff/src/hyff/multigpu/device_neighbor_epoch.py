"""Device graph preparation and local-only immutable neighbor epochs.

The root keeps canonical int32 graph records on its device. Only O(R*N)
counts/ownership metadata reach the CPU; graph records never do.
Each rank consumes the same ordered stream and allocates only its own views.
There are no collectives or implicit device selections in this module.

Numerical membership policy
---------------------------
The shared CPU/device helper evaluates one explicit eager FP64 operation tree:
shift=(s0*c0+s1*c1)+s2*c2; displacement=(p_src-p_dst)+shift;
squared=(dx*dx+dy*dy)+dz*dz; strict comparison to (radius+skin)**2.
There is no matmul, norm, compiled fusion, interval, or host boundary fallback.
Finite FP32 geometry/int32 shifts promote exactly and stay within FP64 exponent
bounds. Every nonzero algebraic term is a multiple of 2**-149 and all accepted
squares are below 2**326, so no FP64 overflow or underflow is required. Under
IEEE binary64 round-to-nearest, corresponding CPU/device scalar operations
therefore give the same membership, including exact-boundary cases. Bitwise
agreement on actual GPUs is not checked by this module.

This predicate differs from the single-GPU CPU matmul/norm distance expression.
measure_legacy_filter_flips counts the membership differences on a graph that
is already on the CPU; the device path never calls it and never copies the
full graph to the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from hyff.multigpu.neighbor_epoch import (
    NeighborEpochError, NeighborSkinDecision, OwnedRadiusView, _cpu_float, _loads, _require,
    explicit_fp64_squared_distance, explicit_fp64_cutoff_squared,
)


_SCHEMA = "LARGO_DEVICE_NEIGHBOR_HEADER"
_CERTIFICATE_SCHEMA = "LARGO_DEVICE_GRAPH_CERTIFICATE"
_CHUNK = 262144
_HEADER_KEYS = frozenset(("schema", "epoch", "generation", "global_ids", "world_size", "owners",
    "layer_radii", "skin", "supplied_radius", "rank_order", "padding_multiple", "reserve_factor",
    "global_edge_count", "supplied_incoming_counts", "weighted_incoming_counts", "radii", "certificate"))
_RADIUS_KEYS = frozenset(("radius", "multiplicity", "incoming_counts", "edge_count", "base_capacity",
    "per_rank_edge_counts", "per_rank_edge_capacities"))


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _copy_json(value):
    return json.loads(_json_bytes(value))


def _capacity(required, reserve):
    # Bucket the actual row capacity shared by allocation and header validation.
    k8 = max(8, ((int(required * reserve) + 1 + 7) // 8) * 8)
    return ((k8 + 31) // 32) * 32


def _padded(number, multiple):
    return ((number + multiple - 1) // multiple) * multiple


def _geometry(torch, positions, cell, atoms):
    _cpu_float(torch, positions, (atoms, 3), "NEIGHBOR_POSITION_METADATA_INVALID")
    _cpu_float(torch, cell, (3, 3), "NEIGHBOR_CELL_METADATA_INVALID")
    _require(positions.dtype == cell.dtype == torch.float32,
             "NEIGHBOR_FP32_RUNTIME_GEOMETRY_REQUIRED")
    _require(not positions.requires_grad and not cell.requires_grad,
             "DEVICE_NEIGHBOR_FIXED_GEOMETRY_REQUIRED")
    _require(float(torch.linalg.det(cell.double()).abs()) > 0.0, "NEIGHBOR_SINGULAR_CELL")


def _validate_options(torch, positions, cell, global_ids, radii, skin, supplied_radius,
                      world_size, epoch, generation, rank_order, padding_multiple, reserve_factor):
    _require(type(world_size) is int and world_size > 0, "NEIGHBOR_WORLD_SIZE_INVALID")
    _require(type(epoch) is int and epoch >= 0, "NEIGHBOR_EPOCH_INVALID")
    _require(type(generation) is int and generation > 0, "DEVICE_NEIGHBOR_GENERATION_INVALID")
    if isinstance(global_ids, torch.Tensor):
        _require(global_ids.device.type == "cpu" and global_ids.dtype == torch.int64 and global_ids.ndim == 1,
                 "NEIGHBOR_CANONICAL_IDS_REQUIRED")
        ids = tuple(global_ids.tolist())
    else:
        ids = tuple(global_ids)
    _require(all(type(value) is int for value in ids) and all(a < b for a, b in zip(ids, ids[1:])),
             "NEIGHBOR_CANONICAL_IDS_REQUIRED")
    _geometry(torch, positions, cell, len(ids))
    layer_radii = tuple(radii)
    _require(layer_radii and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in layer_radii),
             "NEIGHBOR_RADII_INVALID")
    _require(type(skin) in (int, float) and math.isfinite(skin) and skin >= 0, "NEIGHBOR_SKIN_INVALID")
    _require(type(supplied_radius) in (int, float) and math.isfinite(supplied_radius) and
             supplied_radius >= max(layer_radii) + skin, "NEIGHBOR_SUPPLIED_RADIUS_TOO_SMALL")
    _require(type(padding_multiple) is int and padding_multiple > 0 and
             type(reserve_factor) in (int, float) and math.isfinite(reserve_factor) and reserve_factor >= 1,
             "NEIGHBOR_CAPACITY_SETTINGS_INVALID")
    order = tuple(range(world_size)) if rank_order is None else tuple(rank_order)
    _require(len(order) == world_size and all(type(v) is int for v in order) and
             set(order) == set(range(world_size)), "NEIGHBOR_RANK_ORDER_INVALID")
    return ids, tuple(map(float, layer_radii)), order


def _lex_order(torch, edges, shifts, atoms, *, reverse=False):
    """Int64 composite-key sort when the key provably fits, else an exact labelled fallback."""
    if not edges.shape[1]:
        return torch.empty(0, dtype=torch.int64, device=edges.device), {
            "strategy": "EMPTY_GRAPH", "maximum_key": "0", "shift_minima": [0, 0, 0], "shift_spans": [1, 1, 1]}
    minima = [int(shifts[:, axis].min()) for axis in range(3)]
    maxima = [int(shifts[:, axis].max()) for axis in range(3)]
    if reverse:
        minima, maxima = [-v for v in maxima], [-v for v in minima]
    spans = [high-low+1 for low, high in zip(minima, maxima)]
    # Python integers cannot overflow. This proves every intermediate
    # nonnegative mixed-radix key fits before any device int64 arithmetic.
    maximum = atoms * atoms * math.prod(spans) - 1
    telemetry = {"strategy": "INT64_COMPOSITE_KEY", "maximum_key": str(maximum),
                 "shift_minima": minima, "shift_spans": spans}
    if maximum <= 2**63-1:
        key = edges[0 if reverse else 1].long()
        key.mul_(atoms).add_(edges[1 if reverse else 0].long())
        for axis, span in enumerate(spans):
            image = shifts[:, axis].long()
            if reverse:
                image.neg_()
            image.sub_(minima[axis])
            key.mul_(span).add_(image)
        return torch.argsort(key, stable=True), telemetry
    telemetry["strategy"] = "EXACT_STABLE_LEXICOGRAPHIC_FALLBACK"
    order = torch.arange(edges.shape[1], dtype=torch.int64, device=edges.device)
    # Stable least-significant-first sorting preserves all five integer keys.
    keys = (shifts[:, 2], shifts[:, 1], shifts[:, 0], edges[0], edges[1])
    if reverse:
        keys = (-shifts[:, 2], -shifts[:, 1], -shifts[:, 0], edges[1], edges[0])
    for key in keys:
        permutation = torch.argsort(key[order], stable=True)
        order = order[permutation]
    return order, telemetry


def _ordered_pairs(torch, left_edges, left_shifts, right_edges, right_shifts):
    """Return scalar tensors (any decreasing key, any complete duplicate)."""
    same = torch.ones(left_edges.shape[1], dtype=torch.bool, device=left_edges.device)
    bad = torch.zeros((), dtype=torch.bool, device=left_edges.device)
    for left, right in ((left_edges[1], right_edges[1]), (left_edges[0], right_edges[0]),
                        (left_shifts[:, 0], right_shifts[:, 0]),
                        (left_shifts[:, 1], right_shifts[:, 1]),
                        (left_shifts[:, 2], right_shifts[:, 2])):
        bad |= (same & (left > right)).any()
        same &= left == right
    return bad, same.any()


def _canonicalize(torch, native_edges, native_shifts, native_counts, atoms, chunk_size):
    _require(isinstance(native_edges, torch.Tensor) and isinstance(native_shifts, torch.Tensor) and
             isinstance(native_counts, torch.Tensor) and native_edges.device == native_shifts.device == native_counts.device and
             native_edges.device.type in ("cpu", "cuda") and
             native_edges.dtype == native_shifts.dtype == native_counts.dtype == torch.int32 and
             native_edges.ndim == 2 and native_edges.shape[0] == 2 and
             tuple(native_shifts.shape) == (native_edges.shape[1], 3) and
             tuple(native_counts.shape) == (atoms,), "DEVICE_NEIGHBOR_NATIVE_GRAPH_INVALID")
    _require(native_edges.shape[1] < 2**31 and bool((native_counts >= 0).all()) and
             int(native_counts.sum(dtype=torch.int64)) == native_edges.shape[1],
             "DEVICE_NEIGHBOR_NATIVE_COUNT_INVALID")
    edges = native_edges.detach().flip(0).contiguous()
    shifts = native_shifts.detach().clone().contiguous()
    _require(bool(((edges >= 0) & (edges < atoms)).all()), "NEIGHBOR_EDGE_OUTSIDE_GLOBAL_ROWS")
    _require(bool(((edges[0] != edges[1]) | (shifts != 0).any(dim=1)).all()),
             "NEIGHBOR_ZERO_SHIFT_SELF_EDGE")
    # -INT32_MIN cannot have an int32 reciprocal. Reject before negation.
    _require(not bool((shifts == -(2**31)).any()), "NEIGHBOR_RECIPROCAL_IMAGE_REQUIRED")
    supplied = torch.bincount(edges[1].long(), minlength=atoms)
    _require(torch.equal(supplied, native_counts.long()), "DEVICE_NEIGHBOR_NATIVE_DEGREE_MISMATCH")
    order, canonical_sort = _lex_order(torch, edges, shifts, atoms)
    edges, shifts = edges[:, order].contiguous(), shifts[order].contiguous()
    del order
    for start in range(1, edges.shape[1], chunk_size):
        stop = min(start + chunk_size, edges.shape[1])
        _, duplicate = _ordered_pairs(torch, edges[:, start-1:stop-1], shifts[start-1:stop-1],
                                     edges[:, start:stop], shifts[start:stop])
        _require(not bool(duplicate), "NEIGHBOR_DUPLICATE_IMAGE")
    reverse, reciprocal_sort = _lex_order(torch, edges, shifts, atoms, reverse=True)
    for start in range(0, edges.shape[1], chunk_size):
        stop = min(start + chunk_size, edges.shape[1])
        selected = reverse[start:stop]
        _require(torch.equal(edges[0, start:stop], edges[1, selected]) and
                 torch.equal(edges[1, start:stop], edges[0, selected]) and
                 torch.equal(shifts[start:stop], -shifts[selected]), "NEIGHBOR_RECIPROCAL_IMAGE_REQUIRED")
    return edges, shifts, supplied, reverse, {"canonical": canonical_sort, "reciprocal": reciprocal_sort}


def _require_reciprocal_membership(torch, membership, reverse, chunk_size):
    for start in range(0, membership.shape[1], chunk_size):
        stop = min(start+chunk_size, membership.shape[1])
        _require(torch.equal(membership[:, start:stop], membership[:, reverse[start:stop]]),
                 "DEVICE_NEIGHBOR_ASYMMETRIC_RADIUS_MEMBERSHIP")


def _membership(torch, positions, cell, edges, shifts, reverse, radii, skin, chunk_size):
    device_positions = positions.to(device=edges.device, dtype=torch.float64)
    device_cell = cell.to(device=edges.device, dtype=torch.float64)
    result = torch.empty((len(radii), edges.shape[1]), dtype=torch.bool, device=edges.device)
    counts = torch.zeros((len(radii), len(positions)), dtype=torch.int64, device=edges.device)
    thresholds = [explicit_fp64_cutoff_squared(radius, skin) for radius in radii]
    for start in range(0, edges.shape[1], chunk_size):
        stop = min(start + chunk_size, edges.shape[1])
        part, images = edges[:, start:stop], shifts[start:stop]
        squared = explicit_fp64_squared_distance(device_positions, device_cell, part, images)
        _require(bool(torch.isfinite(squared).all()), "DEVICE_NEIGHBOR_NONFINITE_SQUARED_DISTANCE")
        for i, threshold in enumerate(thresholds):
            result[i, start:stop] = squared < threshold
        for i in range(len(radii)):
            counts[i] += torch.bincount(part[1, result[i, start:stop]].long(), minlength=len(positions))
    _require_reciprocal_membership(torch, result, reverse, chunk_size)
    return result, counts.cpu().tolist()


def measure_legacy_filter_flips(global_positions, cell, edge_index, integer_shifts, *, radii, skin):
    """CPU check helper on an ALREADY CPU reference graph.

    Evaluate the single-GPU full-batch distance expression, not a chunked substitute, and
    report only per-radius integer flip counts. Device preparation never calls
    this helper and does not copy its global graph to CPU for this measurement.
    """
    import torch
    _require(all(isinstance(v, torch.Tensor) and v.device.type == "cpu"
                 for v in (global_positions, cell, edge_index, integer_shifts)),
             "DEVICE_NEIGHBOR_LEGACY_CPU_REFERENCE_REQUIRED")
    _require(edge_index.dtype == integer_shifts.dtype == torch.int64 and edge_index.ndim == 2 and
             edge_index.shape[0] == 2 and tuple(integer_shifts.shape) == (edge_index.shape[1], 3),
             "DEVICE_NEIGHBOR_LEGACY_INTEGER_GRAPH_REQUIRED")
    _geometry(torch, global_positions, cell, len(global_positions))
    with torch.no_grad():
        legacy = (global_positions.double()[edge_index[0]] - global_positions.double()[edge_index[1]] +
                  integer_shifts.double() @ cell.double()).norm(dim=-1)
        squared = explicit_fp64_squared_distance(global_positions, cell, edge_index, integer_shifts)
        rows = []
        for radius in sorted(set(radii)):
            old = legacy < radius + skin
            new = squared < explicit_fp64_cutoff_squared(radius, skin)
            rows.append({"radius": float(radius), "legacy_selected": int(old.sum()),
                "explicit_selected": int(new.sum()), "flips": int((old != new).sum()),
                "legacy_only": int((old & ~new).sum()), "explicit_only": int((new & ~old).sum())})
    return {"schema": "LARGO_EXPLICIT_FP64_LEGACY_FLIPS", "graph_edges": edge_index.shape[1],
        "reference_expression": "FULL_BATCH_CPU_MATMUL_NORM", "compared_expression": "EXPLICIT_FP64_SEPARATE_OPS_SQUARED_STRICT",
        "device_to_cpu_graph_transfer": False, "per_radius": rows, "gpu_validated": False}


@dataclass(frozen=True)
class DeviceNeighborChunk:
    offset: int
    edge_index: object
    integer_shifts: object
    membership: object


@dataclass(frozen=True)
class DeviceRadiusSummary:
    radius: float
    multiplicity: int
    incoming_counts: tuple[int, ...]
    edge_count: int
    base_capacity: int
    per_rank_edge_counts: tuple[int, ...]
    per_rank_edge_capacities: tuple[int, ...]


@dataclass(frozen=True)
class DeviceNeighborRootSnapshot:
    edge_index: object
    integer_shifts: object
    membership: object
    metadata: object
    global_positions: object
    cell: object
    chunk_size: int
    _header_bytes: bytes
    _versions: tuple

    @property
    def header(self):
        return json.loads(self._header_bytes)

    def assert_integrity(self):
        current = (self.edge_index, self.integer_shifts, self.membership, self.global_positions, self.cell)
        _require(all(value is original and value._version == version
                     for value, (original, version) in zip(current, self._versions)),
                 "DEVICE_NEIGHBOR_ROOT_BUFFER_CHANGED")

    def iter_chunks(self, chunk_size=None):
        self.assert_integrity()
        size = self.chunk_size if chunk_size is None else chunk_size
        _require(type(size) is int and size > 0, "DEVICE_NEIGHBOR_CHUNK_SIZE_INVALID")
        for start in range(0, self.edge_index.shape[1], size):
            stop = min(start + size, self.edge_index.shape[1])
            yield DeviceNeighborChunk(start, self.edge_index[:, start:stop],
                                      self.integer_shifts[start:stop], self.membership[:, start:stop])
        self.assert_integrity()


def prepare_device_neighbor_root(native_edge_index, native_integer_shifts, native_counts,
        global_positions, cell, *, global_ids, radii, skin, supplied_radius, world_size,
        epoch=0, generation=1, rank_order=None, padding_multiple=8, reserve_factor=1.2, chunk_size=_CHUNK):
    """Root-only full validation/filtering; CPU is supported for exact tests."""
    import torch
    from hyff.multigpu.collectives import edge_balanced_metadata, make_ownership_metadata
    ids, layer_radii, order = _validate_options(torch, global_positions, cell, global_ids, radii, skin,
        supplied_radius, world_size, epoch, generation, rank_order, padding_multiple, reserve_factor)
    _require(type(chunk_size) is int and chunk_size > 0, "DEVICE_NEIGHBOR_CHUNK_SIZE_INVALID")
    with torch.no_grad():
        positions, fixed_cell = global_positions.detach().clone().contiguous(), cell.detach().clone().contiguous()
        edges, shifts, supplied, reverse, sorting = _canonicalize(torch, native_edge_index, native_integer_shifts,
                                                                 native_counts, len(ids), chunk_size)
        unique_radii = tuple(sorted(set(layer_radii)))
        membership, radius_counts = _membership(torch, positions, fixed_cell, edges, shifts, reverse,
                                                unique_radii, float(skin), chunk_size)
        del reverse
        weighted = [sum(layer_radii.count(radius) * radius_counts[i][row]
                        for i, radius in enumerate(unique_radii)) for row in range(len(ids))]
        balanced = edge_balanced_metadata(ids, weighted, world_size=world_size)
        metadata = make_ownership_metadata(ids, tuple(order[rank] for rank in balanced.owners),
                                            world_size=world_size, edge_counts=weighted)
        descriptions = []
        for radius, incoming in zip(unique_radii, radius_counts):
            per_rank = [sum(incoming[row] for row in rows) for rows in metadata.indices_by_rank]
            descriptions.append({"radius": radius, "multiplicity": layer_radii.count(radius),
                "incoming_counts": incoming, "edge_count": sum(incoming),
                "base_capacity": _capacity(max(incoming, default=0), reserve_factor),
                "per_rank_edge_counts": per_rank,
                "per_rank_edge_capacities": [_padded(v, padding_multiple) for v in per_rank]})
        certificate = {"schema": _CERTIFICATE_SCHEMA, "integer_validation": "ROOT_FULL_EXACT",
            "canonical_order": ["destination", "source", "sx", "sy", "sz"],
            "reciprocal_images_verified": True, "per_radius_reciprocity_verified": True,
            "integer_sorting": sorting, "cutoff_policy": "EXPLICIT_FP64_SEPARATE_OPS_SQUARED_STRICT",
            "compared_cpu_device_predicate": "SHARED_EAGER_IEEE_FP64_OPERATION_TREE",
            "boundary_ambiguity_pairs": 0, "boundary_host_fallback_pairs": 0,
            "legacy_filter_flip_measurement": "SEPARATE_CPU_CHECK_NOT_RUN_HERE",
            "transport": "NCCL_BROADCAST_TRUSTED_OR_DIRECT_CPU_TEST_STREAM",
            "receiver_validation": "INDEPENDENT_LOCAL_COUNTS_AND_CANONICAL_STREAM",
            "graph_payload_sha256": None, "whole_global_graph_cpu_copy": False,
            "helper_performed_collectives": False, "gpu_validated": False}
        header = {"schema": _SCHEMA, "epoch": epoch, "generation": generation,
            "global_ids": list(ids), "world_size": world_size, "owners": list(metadata.owners),
            "layer_radii": list(layer_radii), "skin": float(skin), "supplied_radius": float(supplied_radius),
            "rank_order": list(order), "padding_multiple": padding_multiple, "reserve_factor": float(reserve_factor),
            "global_edge_count": edges.shape[1], "supplied_incoming_counts": supplied.cpu().tolist(),
            "weighted_incoming_counts": weighted, "radii": descriptions, "certificate": certificate}
        buffers = (edges, shifts, membership, positions, fixed_cell)
        return DeviceNeighborRootSnapshot(edges, shifts, membership, metadata, positions, fixed_cell,
            chunk_size, _json_bytes(header), tuple((value, value._version) for value in buffers))


def _header_metadata(torch, header, positions, cell):
    from hyff.multigpu.collectives import make_ownership_metadata
    _require(type(header) is dict and set(header) == _HEADER_KEYS and header["schema"] == _SCHEMA,
             "DEVICE_NEIGHBOR_HEADER_INVALID")
    ids, radii, order = _validate_options(torch, positions, cell, header["global_ids"], header["layer_radii"],
        header["skin"], header["supplied_radius"], header["world_size"], header["epoch"], header["generation"],
        header["rank_order"], header["padding_multiple"], header["reserve_factor"])
    atoms, ranks = len(ids), header["world_size"]
    def count_list(value, length):
        return type(value) is list and len(value) == length and all(type(v) is int and 0 <= v < 2**63 for v in value)
    _require(type(header["global_edge_count"]) is int and 0 <= header["global_edge_count"] < 2**31 and
             count_list(header["supplied_incoming_counts"], atoms) and
             sum(header["supplied_incoming_counts"]) == header["global_edge_count"] and
             count_list(header["weighted_incoming_counts"], atoms), "DEVICE_NEIGHBOR_HEADER_COUNTS_INVALID")
    _require(count_list(header["owners"], atoms) and all(v < ranks for v in header["owners"]),
             "DEVICE_NEIGHBOR_HEADER_OWNERS_INVALID")
    metadata = make_ownership_metadata(ids, header["owners"], world_size=ranks,
                                       edge_counts=header["weighted_incoming_counts"])
    descriptions = header["radii"]
    _require(type(descriptions) is list and len(descriptions) == len(set(radii)),
             "DEVICE_NEIGHBOR_HEADER_RADII_INVALID")
    summaries = []
    for radius, item in zip(sorted(set(radii)), descriptions):
        _require(type(item) is dict and set(item) == _RADIUS_KEYS and type(item["radius"]) in (int, float) and
                 item["radius"] == radius and type(item["multiplicity"]) is int and
                 item["multiplicity"] == radii.count(radius) and count_list(item["incoming_counts"], atoms) and
                 count_list(item["per_rank_edge_counts"], ranks) and count_list(item["per_rank_edge_capacities"], ranks),
                 "DEVICE_NEIGHBOR_HEADER_RADII_INVALID")
        incoming = item["incoming_counts"]
        _require(all(v <= n for v, n in zip(incoming, header["supplied_incoming_counts"])) and
                 type(item["edge_count"]) is int and item["edge_count"] == sum(incoming) and
                 type(item["base_capacity"]) is int and
                 item["base_capacity"] == _capacity(max(incoming, default=0), header["reserve_factor"]) and
                 item["per_rank_edge_counts"] == [sum(incoming[row] for row in rows) for rows in metadata.indices_by_rank] and
                 item["per_rank_edge_capacities"] == [_padded(v, header["padding_multiple"]) for v in item["per_rank_edge_counts"]],
                 "DEVICE_NEIGHBOR_HEADER_RADII_COUNTS_INVALID")
        summaries.append(DeviceRadiusSummary(radius, item["multiplicity"], tuple(incoming), item["edge_count"],
            item["base_capacity"], tuple(item["per_rank_edge_counts"]), tuple(item["per_rank_edge_capacities"])))
    _require(header["weighted_incoming_counts"] == [sum(item.multiplicity * item.incoming_counts[row]
             for item in summaries) for row in range(atoms)], "DEVICE_NEIGHBOR_HEADER_WEIGHTED_COUNTS_INVALID")
    certificate = header["certificate"]
    _require(type(certificate) is dict and certificate.get("schema") == _CERTIFICATE_SCHEMA and
             certificate.get("integer_validation") == "ROOT_FULL_EXACT" and
             certificate.get("reciprocal_images_verified") is True and
             certificate.get("cutoff_policy") == "EXPLICIT_FP64_SEPARATE_OPS_SQUARED_STRICT" and
             certificate.get("per_radius_reciprocity_verified") is True and
             certificate.get("graph_payload_sha256") is None and
             certificate.get("whole_global_graph_cpu_copy") is False,
             "DEVICE_NEIGHBOR_HEADER_CERTIFICATE_INVALID")
    return metadata, tuple(summaries)


@dataclass(frozen=True)
class DeviceOwnedNeighborEpoch:
    epoch: int
    generation: int
    metadata: object
    global_positions: object
    cell: object
    layer_radii: tuple[float, ...]
    skin: float
    supplied_radius: float
    supplied_incoming_counts: tuple[int, ...]
    weighted_incoming_counts: tuple[int, ...]
    rank_order: tuple[int, ...]
    padding_multiple: int
    reserve_factor: float
    rank: int
    radius_summaries: tuple[DeviceRadiusSummary, ...]
    local_views: tuple[OwnedRadiusView, ...]
    _certificate_bytes: bytes
    _versions: tuple

    @property
    def graph_certificate(self):
        """Rank-common certificate for agreement; metadata SHA is not graph SHA."""
        return json.loads(self._certificate_bytes)

    def assert_integrity(self):
        current = [self.global_positions, self.cell]
        for view in self.local_views:
            current.extend(view.buffers())
        _require(len(current) == len(self._versions) and all(value is original and value._version == version
                 for value, (original, version) in zip(current, self._versions)), "NEIGHBOR_EPOCH_BUFFER_CHANGED")

    def assert_generation(self, generation):
        self.assert_integrity()
        _require(type(generation) is int and generation == self.generation, "NEIGHBOR_EPOCH_GENERATION_STALE")

    def radius_summary(self, radius):
        self.assert_integrity()
        _require(type(radius) in (int, float) and math.isfinite(radius), "NEIGHBOR_RADIUS_UNKNOWN")
        for summary in self.radius_summaries:
            if summary.radius == radius:
                return summary
        raise NeighborEpochError("NEIGHBOR_RADIUS_UNKNOWN")

    def radius_view(self, radius, rank):
        _require(type(rank) is int and rank == self.rank, "DEVICE_NEIGHBOR_NONLOCAL_VIEW_FORBIDDEN")
        summary = self.radius_summary(radius)
        return self.local_views[self.radius_summaries.index(summary)]

    def assess_skin(self, global_positions, cell):
        self.assert_integrity()
        import torch
        _cpu_float(torch, global_positions, (self.metadata.n_global, 3), "NEIGHBOR_POSITION_METADATA_INVALID")
        _cpu_float(torch, cell, (3, 3), "NEIGHBOR_CELL_METADATA_INVALID")
        with torch.no_grad():
            displacement = global_positions.double() - self.global_positions.double()
            maximum = float(displacement.norm(dim=-1).max()) if self.metadata.n_global else 0.0
            reasons = []
            if not torch.equal(cell, self.cell):
                reasons.append("CELL_CHANGED")
            exhausted = (not torch.equal(global_positions, self.global_positions) if self.skin == 0 else
                         2.0 * maximum > self.skin)
            if exhausted:
                reasons.append("SKIN_EXHAUSTED")
        return NeighborSkinDecision("REBUILD_REQUIRED" if reasons else "READY", tuple(reasons),
                                    self.epoch, self.generation, maximum, self.skin)

    def public_metadata(self):
        self.assert_integrity()
        return {"schema": "LARGO_DEVICE_OWNED_NEIGHBOR_EPOCH", "epoch": self.epoch,
            "generation": self.generation, "global_atoms": self.metadata.n_global,
            "world_size": self.metadata.world_size, "local_rank": self.rank,
            "atom_counts": list(self.metadata.counts), "layer_radii_A": list(self.layer_radii),
            "skin_A": self.skin, "supplied_radius_A": self.supplied_radius,
            "rank_order": list(self.rank_order), "rank_local_views_only": True,
            "ownership_policy": "GREEDY_SUM_OF_REPEATED_LAYER_SKIN_DESTINATION_EDGE_COUNTS",
            "weighted_layer_load": _loads(self.weighted_incoming_counts, self.metadata),
            "full_supplied_skin_load": _loads(self.supplied_incoming_counts, self.metadata),
            "radii": [{"radius_A": summary.radius, "layer_multiplicity": summary.multiplicity,
                "global_skin_edges": summary.edge_count, **_loads(summary.incoming_counts, self.metadata),
                "base_capacity": summary.base_capacity, "per_rank_efm_edges": list(summary.per_rank_edge_counts),
                "per_rank_efm_capacities": list(summary.per_rank_edge_capacities)} for summary in self.radius_summaries],
            "graph_certificate": self.graph_certificate, "receiver_local_counts_verified": True,
            "complete_supplied_graph_asserted_by_caller": True, "geometric_completeness_verified": False,
            "coordinate_convention": "CONTINUOUS_UNWRAPPED_CARTESIAN",
            "neighbor_policy": "FIXED_SKIN_SUPERSET_PER_EPOCH",
            "helper_performed_collectives": False, "gpu_validated": False, "capture_validated": False}


class DeviceNeighborLocalBuilder:
    """One rank's bounded stream consumer. Any failure permanently poisons it."""
    def __init__(self, header, *, rank, device, global_positions, cell):
        self._initialize(header, rank=rank, device=device,
            global_positions=global_positions, cell=cell, copy_header=True)

    @classmethod
    def _from_transport_owned_header(cls, header, *, rank, device, global_positions, cell):
        """Consume a freshly decoded header that the transport hands over uncopied.

        This private entry is only for transport-created JSON with no external
        aliases; all builder validation and the final header hash still apply.
        The public constructor copies caller-supplied headers.
        """
        result = cls.__new__(cls)
        result._initialize(header, rank=rank, device=device,
            global_positions=global_positions, cell=cell, copy_header=False)
        return result

    def _initialize(self, header, *, rank, device, global_positions, cell, copy_header):
        import torch
        self._header = _copy_json(header) if copy_header else header
        self.metadata, self.summaries = _header_metadata(torch, self._header, global_positions, cell)
        _require(type(rank) is int and 0 <= rank < self.metadata.world_size, "NEIGHBOR_RANK_INVALID")
        self.rank, self.device = rank, torch.device(device)
        _require(self.device.type in ("cpu", "cuda"), "DEVICE_NEIGHBOR_DEVICE_INVALID")
        self.positions, self.cell = global_positions.detach().clone().contiguous(), cell.detach().clone().contiguous()
        self._next, self._failed, self._finished, self._last = 0, False, False, None
        self.views, self.filled, self.cursors = [], [], []
        local = torch.tensor(self.metadata.indices_by_rank[rank], dtype=torch.int64, device=self.device)
        inverse = torch.full((self.metadata.n_global,), -1, dtype=torch.int64, device=self.device)
        inverse[local] = torch.arange(len(local), dtype=torch.int64, device=self.device)
        self._supplied_expected = torch.tensor([self._header["supplied_incoming_counts"][row]
            for row in self.metadata.indices_by_rank[rank]], dtype=torch.int64, device=self.device)
        self._supplied_filled = torch.zeros(len(local), dtype=torch.int64, device=self.device)
        for summary in self.summaries:
            n_local, capacity = len(local), summary.base_capacity
            number, padded = summary.per_rank_edge_counts[rank], summary.per_rank_edge_capacities[rank]
            counts = torch.tensor([summary.incoming_counts[row] for row in self.metadata.indices_by_rank[rank]],
                                  dtype=torch.int32, device=self.device)
            self.views.append(OwnedRadiusView(summary.radius, rank, local, inverse,
                torch.full((n_local, capacity), -1, dtype=torch.int64, device=self.device),
                torch.zeros((n_local, capacity, 3), dtype=torch.int32, device=self.device), counts, capacity,
                torch.full((padded,), -1, dtype=torch.int64, device=self.device),
                torch.full((padded,), -1, dtype=torch.int64, device=self.device),
                torch.zeros(padded, dtype=torch.bool, device=self.device),
                torch.zeros((padded, 3), dtype=torch.int64, device=self.device), number, padded))
            self.filled.append(torch.zeros(n_local, dtype=torch.int64, device=self.device))
            self.cursors.append(0)

    def append(self, chunk):
        _require(not self._failed and not self._finished, "DEVICE_NEIGHBOR_BUILDER_CLOSED")
        try:
            self._append(chunk)
        except Exception:
            self._failed = True
            raise

    def _append(self, chunk):
        import torch
        _require(isinstance(chunk, DeviceNeighborChunk) and type(chunk.offset) is int and chunk.offset == self._next,
                 "DEVICE_NEIGHBOR_STREAM_OFFSET_INVALID")
        edges, shifts, member = chunk.edge_index, chunk.integer_shifts, chunk.membership
        _require(all(isinstance(v, torch.Tensor) for v in (edges, shifts, member)) and
                 edges.device == shifts.device == member.device == self.device and
                 edges.dtype == shifts.dtype == torch.int32 and member.dtype == torch.bool and
                 edges.ndim == 2 and edges.shape[0] == 2 and tuple(shifts.shape) == (edges.shape[1], 3) and
                 tuple(member.shape) == (len(self.views), edges.shape[1]), "DEVICE_NEIGHBOR_STREAM_SHAPE_INVALID")
        length = edges.shape[1]
        _require(length > 0 and self._next + length <= self._header["global_edge_count"],
                 "DEVICE_NEIGHBOR_STREAM_LENGTH_INVALID")
        _require(bool(((edges >= 0) & (edges < self.metadata.n_global)).all()), "NEIGHBOR_EDGE_OUTSIDE_GLOBAL_ROWS")
        _require(bool(((edges[0] != edges[1]) | (shifts != 0).any(dim=1)).all()) and
                 not bool((shifts == -(2**31)).any()), "DEVICE_NEIGHBOR_STREAM_INTEGER_INVALID")
        if self._last is not None:
            bad, duplicate = _ordered_pairs(torch, self._last[0], self._last[1], edges[:, :1], shifts[:1])
            _require(not bool(bad | duplicate), "DEVICE_NEIGHBOR_STREAM_ORDER_INVALID")
        if length > 1:
            bad, duplicate = _ordered_pairs(torch, edges[:, :-1], shifts[:-1], edges[:, 1:], shifts[1:])
            _require(not bool(bad | duplicate), "DEVICE_NEIGHBOR_STREAM_ORDER_INVALID")
        rows = self.views[0].global_to_local[edges[1].long()]
        supplied = self._supplied_filled + torch.bincount(rows[rows >= 0], minlength=len(self._supplied_filled))
        _require(bool((supplied <= self._supplied_expected).all()), "DEVICE_NEIGHBOR_SUPPLIED_COUNT_EXCEEDED")
        for i, view in enumerate(self.views):
            selected = torch.nonzero(member[i] & (rows >= 0), as_tuple=False).flatten()
            n = len(selected)
            local_rows = rows[selected]
            per_row = torch.bincount(local_rows, minlength=len(view.local_to_global))
            total = self.filled[i] + per_row
            _require(bool((total <= view.counts).all()) and self.cursors[i] + n <= view.edge_count,
                     "DEVICE_NEIGHBOR_LOCAL_COUNT_EXCEEDED")
            if n:
                starts = torch.cumsum(per_row, dim=0) - per_row
                columns = (torch.arange(n, dtype=torch.int64, device=self.device) -
                           torch.repeat_interleave(starts, per_row, output_size=n) + self.filled[i][local_rows])
                _require(bool((columns < view.base_capacity).all()), "NEIGHBOR_CAPACITY_INVALID")
                view.neighbors[local_rows, columns] = edges[0, selected].long()
                view.base_integer_shifts[local_rows, columns] = shifts[selected]
                start, stop = self.cursors[i], self.cursors[i] + n
                view.source[start:stop] = edges[0, selected].long()
                view.destination[start:stop] = local_rows
                view.integer_shifts[start:stop] = shifts[selected].long()
                view.valid[start:stop] = True
            self.filled[i] = total
            self.cursors[i] += n
        self._supplied_filled = supplied
        self._next += length
        self._last = (edges[:, -1:].clone(), shifts[-1:].clone())

    def finish(self):
        _require(not self._failed and not self._finished, "DEVICE_NEIGHBOR_BUILDER_CLOSED")
        try:
            result = self._finish()
        except Exception:
            self._failed = True
            raise
        self._finished = True
        return result

    def _finish(self):
        import torch
        _require(self._next == self._header["global_edge_count"], "DEVICE_NEIGHBOR_STREAM_INCOMPLETE")
        _require(torch.equal(self._supplied_filled, self._supplied_expected),
                 "DEVICE_NEIGHBOR_SUPPLIED_COUNT_MISMATCH")
        for view, filled, cursor in zip(self.views, self.filled, self.cursors):
            _require(cursor == view.edge_count and torch.equal(filled, view.counts.long()),
                     "DEVICE_NEIGHBOR_LOCAL_COUNT_MISMATCH")
        h = self._header
        certificate = _copy_json(h["certificate"])
        certificate["header_metadata_sha256"] = hashlib.sha256(_json_bytes(h)).hexdigest()
        buffers = [self.positions, self.cell]
        for view in self.views:
            buffers.extend(view.buffers())
        return DeviceOwnedNeighborEpoch(h["epoch"], h["generation"], self.metadata, self.positions, self.cell,
            tuple(h["layer_radii"]), h["skin"], h["supplied_radius"], tuple(h["supplied_incoming_counts"]),
            tuple(h["weighted_incoming_counts"]), tuple(h["rank_order"]), h["padding_multiple"], h["reserve_factor"],
            self.rank, self.summaries, tuple(self.views), _json_bytes(certificate),
            tuple((value, value._version) for value in buffers))


__all__ = ["DeviceNeighborChunk", "DeviceRadiusSummary", "DeviceNeighborRootSnapshot",
           "DeviceOwnedNeighborEpoch", "DeviceNeighborLocalBuilder", "prepare_device_neighbor_root",
           "measure_legacy_filter_flips"]
