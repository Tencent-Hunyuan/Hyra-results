"""CPU preparation of one immutable, caller-supplied complete skin epoch.

No neighbors are discovered here. The caller asserts completeness through
supplied_radius; reciprocal image identity and all row mappings are validated.
Repeated radii denote repeated layer work and weight edge-balanced ownership.
All real skin edges remain valid: evaluation applies an envelope exactly zero
at distance >= radius. Work on the fixed superset costs more than a compacted graph.

Positions and displacement checks use continuous, unwrapped Cartesian values.
CPU checks accept FP32/FP64 geometry; the CUDA bridge requires FP32 reference
positions and cell, so it cannot silently change geometry by downcasting.
The caller makes the collective decisions and handles epoch retirement, growth
and recapture.
Module import requires only stdlib; no group or device is selected here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


class NeighborEpochError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise NeighborEpochError(code)



@dataclass(frozen=True)
class OwnedRadiusView:
    radius: float
    rank: int
    local_to_global: object
    global_to_local: object
    neighbors: object
    base_integer_shifts: object
    counts: object
    base_capacity: int
    source: object
    destination: object
    valid: object
    integer_shifts: object
    edge_count: int
    edge_capacity: int

    def buffers(self):
        return (self.local_to_global, self.global_to_local, self.neighbors,
                self.base_integer_shifts, self.counts, self.source, self.destination,
                self.valid, self.integer_shifts)


@dataclass(frozen=True)
class RadiusSkinGraph:
    radius: float
    multiplicity: int
    edge_index: object
    integer_shifts: object
    incoming_counts: tuple[int, ...]
    owned: tuple[OwnedRadiusView, ...]


@dataclass(frozen=True)
class NeighborSkinDecision:
    status: str
    reasons: tuple[str, ...]
    epoch: int
    generation: int
    maximum_displacement: float
    skin: float

    @property
    def ready(self):
        return self.status == "READY"

    def public(self):
        return {"status": self.status, "reasons": list(self.reasons), "epoch": self.epoch,
                "generation": self.generation, "maximum_displacement_A": self.maximum_displacement,
                "skin_A": self.skin, "collective_decision_required": True,
                "helper_performed_collectives": False}


def _loads(counts, metadata):
    loads = [sum(counts[row] for row in rows) for rows in metadata.indices_by_rank]
    mean = sum(loads) / metadata.world_size
    return {"per_rank_edges": loads, "mean_edges": mean, "maximum_edges": max(loads),
            "max_mean_edges_ratio": max(loads) / mean if mean else 1.0}


@dataclass(frozen=True)
class OwnedNeighborEpoch:
    epoch: int
    generation: int
    metadata: object
    global_positions: object
    edge_index: object
    integer_shifts: object
    cell: object
    layer_radii: tuple[float, ...]
    skin: float
    supplied_radius: float
    supplied_incoming_counts: tuple[int, ...]
    weighted_incoming_counts: tuple[int, ...]
    rank_order: tuple[int, ...]
    padding_multiple: int
    reserve_factor: float
    radius_graphs: tuple[RadiusSkinGraph, ...]
    _versions: tuple

    def assert_integrity(self):
        current = [self.global_positions, self.edge_index, self.integer_shifts, self.cell]
        for graph in self.radius_graphs:
            current.extend((graph.edge_index, graph.integer_shifts))
            for view in graph.owned:
                current.extend(view.buffers())
        _require(len(current) == len(self._versions) and
                 all(value is original and value._version == version
                     for value, (original, version) in zip(current, self._versions)),
                 "NEIGHBOR_EPOCH_BUFFER_CHANGED")

    def assert_generation(self, generation):
        self.assert_integrity()
        _require(type(generation) is int and generation == self.generation,
                 "NEIGHBOR_EPOCH_GENERATION_STALE")

    def radius_graph(self, radius):
        self.assert_integrity()
        _require(type(radius) in (int, float) and math.isfinite(radius), "NEIGHBOR_RADIUS_UNKNOWN")
        for graph in self.radius_graphs:
            if graph.radius == radius:
                return graph
        raise NeighborEpochError("NEIGHBOR_RADIUS_UNKNOWN")

    def radius_view(self, radius, rank):
        _require(type(rank) is int and 0 <= rank < self.metadata.world_size, "NEIGHBOR_RANK_INVALID")
        return self.radius_graph(radius).owned[rank]

    def assess_skin(self, global_positions, cell):
        """Eager CPU decision only; no MPI/NCCL, rebuild or generation change.

        2*max displacement <= skin is safe because the reference superset is
        strict distance < radius+skin and the physical envelope vanishes at
        distance >= radius. Raw wrapping jumps are not reduced modulo cell.
        Skin zero accepts exactly unchanged Cartesian coordinates only.
        """
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
            exhausted = (not torch.equal(global_positions, self.global_positions) if self.skin == 0.0 else
                         2.0 * maximum > self.skin)
            if exhausted:
                reasons.append("SKIN_EXHAUSTED")
        return NeighborSkinDecision("REBUILD_REQUIRED" if reasons else "READY", tuple(reasons),
                                    self.epoch, self.generation, maximum, self.skin)

    def public_metadata(self):
        self.assert_integrity()
        return {"schema": "LARGO_OWNED_NEIGHBOR_EPOCH", "epoch": self.epoch,
            "generation": self.generation, "global_atoms": self.metadata.n_global,
            "world_size": self.metadata.world_size, "atom_counts": list(self.metadata.counts),
            "layer_radii_A": list(self.layer_radii), "skin_A": self.skin,
            "supplied_radius_A": self.supplied_radius, "rank_order": list(self.rank_order),
            "ownership_policy": "GREEDY_SUM_OF_REPEATED_LAYER_SKIN_DESTINATION_EDGE_COUNTS",
            "weighted_layer_load": _loads(self.weighted_incoming_counts, self.metadata),
            "full_supplied_skin_load": _loads(self.supplied_incoming_counts, self.metadata),
            "radii": [{"radius_A": graph.radius, "layer_multiplicity": graph.multiplicity,
                "global_skin_edges": len(graph.integer_shifts), **_loads(graph.incoming_counts, self.metadata),
                "base_capacity": graph.owned[0].base_capacity,
                "per_rank_efm_edges": [view.edge_count for view in graph.owned],
                "per_rank_efm_capacities": [view.edge_capacity for view in graph.owned]}
                for graph in self.radius_graphs],
            "complete_supplied_graph_asserted_by_caller": True,
            "geometric_completeness_verified": False, "reciprocal_images_verified": True,
            "coordinate_convention": "CONTINUOUS_UNWRAPPED_CARTESIAN",
            "cuda_bridge_reference_geometry_dtype": "FLOAT32_REQUIRED",
            "cutoff_policy": "EXPLICIT_FP64_SEPARATE_OPS_SQUARED_STRICT",
            "gpu_cutoff_boundary_rounding_validated": False,
            "neighbor_policy": "FIXED_SKIN_SUPERSET_PER_EPOCH",
            "efm_valid_means": "NONPADDING_SKIN_EDGE_RADIUS_ENVELOPE_APPLIED_AT_EVALUATION",
            "caller_owns_consensus_growth_and_retirement": True,
            "helper_performed_collectives": False, "gpu_validated": False,
            "capture_validated": False, "native_compaction_performance_equivalence": False}


def _cpu_float(torch, value, shape, code):
    _require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
             value.dtype in (torch.float32, torch.float64) and tuple(value.shape) == tuple(shape), code)
    _require(bool(torch.isfinite(value).all()), "NEIGHBOR_NONFINITE_INPUT")


@dataclass(frozen=True)
class _IncomingColumns:
    """Per-edge column within its destination, preserving input edge order."""
    counts: object
    columns: object


def _incoming_columns(destination, atoms):
    import numpy as np
    counts = np.bincount(destination, minlength=atoms)
    starts = np.cumsum(counts, dtype=np.int64) - counts
    if len(destination) < 2 or bool((destination[1:] >= destination[:-1]).all()):
        # Native canonical ordering survives every radius filter.  This is an
        # exact content check, not an inference from a shape or origin tag.
        columns = np.arange(len(destination), dtype=np.int64) - np.repeat(starts, counts)
        return _IncomingColumns(counts, columns)
    order = np.argsort(destination, kind="stable")
    columns = np.empty(len(destination), dtype=np.int64)
    columns[order] = np.arange(len(destination), dtype=np.int64) - np.repeat(starts, counts)
    return _IncomingColumns(counts, columns)


def _canonical_integer_order(src, dst, image):
    """Whether exact records are nondecreasing in (dst,src,sx,sy,sz)."""
    import numpy as np
    # Bound temporary comparison storage independently of the global edge
    # count.  Equal records remain allowed here: the duplicate guard below
    # rejects them with NEIGHBOR_DUPLICATE_IMAGE.
    for start in range(1, len(src), 262144):
        stop = min(start + 262144, len(src))
        unresolved = np.ones(stop - start, dtype=np.bool_)
        for key in (dst, src, image[:, 0], image[:, 1], image[:, 2]):
            left, right = key[start - 1:stop - 1], key[start:stop]
            if bool((unresolved & (left > right)).any()):
                return False
            unresolved &= left == right
            if not bool(unresolved.any()):
                break
    return True


def _reciprocal_order_canonical(src, dst, image):
    """Exact reciprocal-sort permutation of already canonical records.

    Stable src sorting preserves dst order and ascending image order within
    each endpoint pair.  Reversing each equal-pair run gives descending image
    order, exactly the lexicographic order of (-sx,-sy,-sz).  The arrays being
    validated are never reordered or mutated by this helper.
    """
    import numpy as np
    reverse = np.argsort(src, kind="stable")
    if len(reverse) < 2:
        return reverse
    same = np.empty(len(reverse) - 1, dtype=np.bool_)
    for start in range(0, len(same), 262144):
        stop = min(start + 262144, len(same))
        left, right = reverse[start:stop], reverse[start + 1:stop + 1]
        same[start:stop] = (src[left] == src[right]) & (dst[left] == dst[right])
    if not bool(same.any()):
        return reverse
    # Only repeated endpoint pairs need reversal.  A true run [a,b) in
    # adjacent-pair equality describes the edge interval [a,b+1).
    transitions = np.flatnonzero(same[1:] != same[:-1]) + 1
    starts = transitions[same[transitions]]
    stops = transitions[~same[transitions]]
    if bool(same[0]):
        starts = np.concatenate((np.zeros(1, dtype=np.int64), starts))
    if bool(same[-1]):
        stops = np.concatenate((stops, np.asarray([len(same)], dtype=np.int64)))
    # Dense many-image graphs can have millions of tiny pair runs.  Use an
    # exact lexsort for that case rather than a Python loop per pair.
    # This changes performance only; both paths return the same permutation
    # for the duplicate-free records accepted by the caller.
    if len(starts) > 4096:
        return np.lexsort((-image[:, 2], -image[:, 1], -image[:, 0], dst, src))
    for start, stop in zip(starts, stops):
        reverse[start:stop + 1] = reverse[start:stop + 1][::-1]
    return reverse


def _validate_integer_graph(edge_index, shifts, atoms):
    """Exact integer checks without constructing Python objects per edge."""
    import numpy as np
    src, dst = edge_index.numpy()
    image = shifts.numpy()
    if _canonical_integer_order(src, dst, image):
        ordered_src, ordered_dst, ordered_image = src, dst, image
    else:
        order = np.lexsort((image[:, 2], image[:, 1], image[:, 0], src, dst))
        ordered_src, ordered_dst, ordered_image = src[order], dst[order], image[order]
    chunk = 262144
    for start in range(1, len(src), chunk):
        stop = min(start + chunk, len(src))
        left, right = slice(start - 1, stop - 1), slice(start, stop)
        duplicate = ((ordered_src[left] == ordered_src[right]) & (ordered_dst[left] == ordered_dst[right]) &
                     (ordered_image[left] == ordered_image[right]).all(axis=1))
        _require(not bool(duplicate.any()), "NEIGHBOR_DUPLICATE_IMAGE")
    _require(bool(((src >= 0) & (src < atoms) & (dst >= 0) & (dst < atoms)).all()),
             "NEIGHBOR_EDGE_OUTSIDE_GLOBAL_ROWS")
    _require(bool(((src != dst) | (image != 0).any(axis=1)).all()), "NEIGHBOR_ZERO_SHIFT_SELF_EDGE")
    # Bound before negation so int64 overflow cannot make a false reciprocal.
    _require(bool(((image >= -(2**31)) & (image < 2**31)).all()), "NEIGHBOR_SHIFT_INT32_OVERFLOW")
    reverse = _reciprocal_order_canonical(ordered_src, ordered_dst, ordered_image)
    for start in range(0, len(src), chunk):
        selected, reversed_rows = slice(start, start + chunk), reverse[start:start + chunk]
        _require(np.array_equal(ordered_src[selected], ordered_dst[reversed_rows]) and
                 np.array_equal(ordered_dst[selected], ordered_src[reversed_rows]) and
                 np.array_equal(ordered_image[selected], -ordered_image[reversed_rows]),
                 "NEIGHBOR_RECIPROCAL_IMAGE_REQUIRED")


def _owned_view(torch, radius, edge_index, shifts, incoming, metadata, rank, capacity, padding_multiple):
    rows = metadata.indices_by_rank[rank]
    local = torch.tensor(rows, dtype=torch.int64)
    inverse = torch.full((metadata.n_global,), -1, dtype=torch.int64)
    inverse[local] = torch.arange(len(rows), dtype=torch.int64)
    neighbors = torch.full((len(rows), capacity), -1, dtype=torch.int64)
    base_shifts = torch.zeros((len(rows), capacity, 3), dtype=torch.int32)
    counts = torch.zeros(len(rows), dtype=torch.int32)
    if isinstance(incoming, _IncomingColumns):
        # Stable destination grouping gives the same row order as appending edges in input order.
        # No sorting is applied to EFM source/destination arrays below.
        import numpy as np
        src, dst = edge_index.numpy()
        edge_local_rows = inverse.numpy()[dst]
        selected_rows = np.flatnonzero(edge_local_rows >= 0)
        local_rows = edge_local_rows[selected_rows]
        del edge_local_rows
        columns = incoming.columns[selected_rows]
        _require(not len(columns) or int(columns.max()) < capacity, "NEIGHBOR_CAPACITY_INVALID")
        neighbors.numpy()[local_rows, columns] = src[selected_rows]
        # Assignment performs the same exact integer cast without allocating
        # another selected int32 array; the full int32-range guard ran above.
        base_shifts.numpy()[local_rows, columns] = shifts.numpy()[selected_rows]
        counts.numpy()[:] = incoming.counts[np.asarray(rows, dtype=np.int64)]
        number = len(selected_rows)
    else:
        # List-of-rows input form, used by independent CPU checks.
        for owned, global_row in enumerate(rows):
            entries = incoming[global_row]
            counts[owned] = len(entries)
            if entries:
                values = torch.tensor(entries, dtype=torch.int64)
                neighbors[owned, :len(entries)] = values[:, 0]
                base_shifts[owned, :len(entries)] = values[:, 1:].to(torch.int32)
        selected = inverse[edge_index[1]] >= 0
        number = int(selected.sum())
    padded = ((number + padding_multiple - 1) // padding_multiple) * padding_multiple
    source = torch.full((padded,), -1, dtype=torch.int64)
    destination = torch.full((padded,), -1, dtype=torch.int64)
    valid = torch.zeros(padded, dtype=torch.bool)
    images = torch.zeros((padded, 3), dtype=torch.int64)
    if isinstance(incoming, _IncomingColumns):
        source.numpy()[:number] = src[selected_rows]
        destination.numpy()[:number] = local_rows
        images.numpy()[:number] = shifts.numpy()[selected_rows]
    else:
        source[:number] = edge_index[0, selected]
        destination[:number] = inverse[edge_index[1, selected]]
        images[:number] = shifts[selected]
    valid[:number] = True
    return OwnedRadiusView(radius, rank, local, inverse, neighbors, base_shifts, counts, capacity,
                           source, destination, valid, images, number, padded)


def explicit_fp64_squared_distance(positions, cell, edge_index, integer_shifts):
    """Shared CPU/device IEEE FP64 predicate input, using separate eager ops.

    The operation order is fixed: shift=(s0*c0+s1*c1)+s2*c2;
    displacement=(p_src-p_dst)+shift; squared=(dx*dx+dy*dy)+dz*dz.
    Each arithmetic operation materializes a tensor before the next one. This
    is preparation outside torch.compile/capture, with no matmul or norm.
    """
    import torch
    _require(not torch.compiler.is_compiling(), "NEIGHBOR_EXPLICIT_PREDICATE_EAGER_REQUIRED")
    points, fixed_cell, images = positions.double(), cell.double(), integer_shifts.double()
    first = images[:, 0:1] * fixed_cell[0]
    second = images[:, 1:2] * fixed_cell[1]
    third = images[:, 2:3] * fixed_cell[2]
    first_two = first + second
    shift = first_two + third
    difference = points[edge_index[0].long()] - points[edge_index[1].long()]
    displacement = difference + shift
    x_squared = displacement[:, 0] * displacement[:, 0]
    y_squared = displacement[:, 1] * displacement[:, 1]
    z_squared = displacement[:, 2] * displacement[:, 2]
    xy_squared = x_squared + y_squared
    return xy_squared + z_squared


def explicit_fp64_cutoff_squared(radius, skin):
    """Same host binary64 addition/multiplication for CPU and device filters."""
    threshold = radius + skin
    squared = threshold * threshold
    _require(math.isfinite(squared), "NEIGHBOR_SQUARED_CUTOFF_NONFINITE")
    return squared


def prepare_owned_neighbor_epoch(global_positions, edge_index, integer_shifts, cell, *, global_ids,
                                 radii, skin, supplied_radius, world_size, epoch=0, previous=None,
                                 rank_order=None, padding_multiple=8, reserve_factor=1.2):
    """Prepare private CPU metadata from an explicitly complete reciprocal graph.

    global_ids must already be in strictly increasing canonical row order.
    radii contains one entry per relevant layer/operator; repeats weight its
    incoming skin-edge work. rank_order only permutes the resulting greedy bins.
    A replacement requires a strictly newer epoch. Existing epochs are not
    mutated or retired: the caller must collectively adopt the new generation.
    """
    import torch
    from hyff.multigpu.collectives import edge_balanced_metadata, make_ownership_metadata
    _require(type(world_size) is int and world_size > 0, "NEIGHBOR_WORLD_SIZE_INVALID")
    _require(type(epoch) is int and epoch >= 0, "NEIGHBOR_EPOCH_INVALID")
    _require(previous is None or isinstance(previous, OwnedNeighborEpoch), "NEIGHBOR_PREVIOUS_EPOCH_INVALID")
    if previous is not None:
        previous.assert_integrity()
        _require(epoch > previous.epoch, "NEIGHBOR_EPOCH_MUST_INCREASE")
    _require(type(skin) in (int, float) and math.isfinite(skin) and skin >= 0.0,
             "NEIGHBOR_SKIN_INVALID")
    layer_radii = tuple(radii)
    _require(bool(layer_radii) and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                                     for value in layer_radii), "NEIGHBOR_RADII_INVALID")
    layer_radii = tuple(map(float, layer_radii))
    _require(type(supplied_radius) in (int, float) and math.isfinite(supplied_radius) and
             supplied_radius >= max(layer_radii) + skin, "NEIGHBOR_SUPPLIED_RADIUS_TOO_SMALL")
    _require(type(padding_multiple) is int and padding_multiple > 0 and
             type(reserve_factor) in (int, float) and math.isfinite(reserve_factor) and reserve_factor >= 1.0,
             "NEIGHBOR_CAPACITY_SETTINGS_INVALID")
    order = tuple(range(world_size)) if rank_order is None else tuple(rank_order)
    _require(len(order) == world_size and all(type(value) is int for value in order) and
             set(order) == set(range(world_size)), "NEIGHBOR_RANK_ORDER_INVALID")
    if isinstance(global_ids, torch.Tensor):
        _require(global_ids.device.type == "cpu" and global_ids.dtype == torch.int64 and global_ids.ndim == 1,
                 "NEIGHBOR_CANONICAL_IDS_REQUIRED")
        ids = tuple(global_ids.tolist())
    else:
        ids = tuple(global_ids)
    _require(all(type(value) is int for value in ids) and all(a < b for a, b in zip(ids, ids[1:])),
             "NEIGHBOR_CANONICAL_IDS_REQUIRED")
    _cpu_float(torch, global_positions, (len(ids), 3), "NEIGHBOR_POSITION_METADATA_INVALID")
    _cpu_float(torch, cell, (3, 3), "NEIGHBOR_CELL_METADATA_INVALID")
    _require(float(torch.linalg.det(cell.double()).abs()) > 0.0, "NEIGHBOR_SINGULAR_CELL")
    _require(isinstance(edge_index, torch.Tensor) and edge_index.device.type == "cpu" and
             edge_index.dtype == torch.int64 and edge_index.ndim == 2 and edge_index.shape[0] == 2 and
             isinstance(integer_shifts, torch.Tensor) and integer_shifts.device.type == "cpu" and
             integer_shifts.dtype == torch.int64 and tuple(integer_shifts.shape) == (edge_index.shape[1], 3),
             "NEIGHBOR_INTEGER_GRAPH_REQUIRED")
    with torch.no_grad():
        _validate_integer_graph(edge_index, integer_shifts, len(ids))
        positions, edges, shifts, fixed_cell = (value.detach().clone().contiguous() for value in
                                               (global_positions, edge_index, integer_shifts, cell))
        distance_squared = explicit_fp64_squared_distance(positions, fixed_cell, edges, shifts)
        supplied_counts = tuple(torch.bincount(edges[1], minlength=len(ids)).tolist())
        choices = []
        weighted = [0] * len(ids)
        for radius in sorted(set(layer_radii)):
            selected = distance_squared < explicit_fp64_cutoff_squared(radius, skin)
            subset, images = edges[:, selected].contiguous(), shifts[selected].contiguous()
            incoming = _incoming_columns(subset[1].numpy(), len(ids))
            counts = tuple(map(int, incoming.counts))
            multiplicity = layer_radii.count(radius)
            weighted = [value + multiplicity * count for value, count in zip(weighted, counts)]
            choices.append((radius, multiplicity, subset, images, incoming, counts))
        balanced = edge_balanced_metadata(ids, weighted, world_size=world_size)
        metadata = make_ownership_metadata(ids, tuple(order[rank] for rank in balanced.owners),
                                            world_size=world_size, edge_counts=weighted)
        graphs = []
        for radius, multiplicity, subset, images, incoming, counts in choices:
            required = max(counts, default=0)
            # Same reserve/8-slot capacity rule as base_messages.prepare_owned_graph.
            capacity = max(8, ((int(required * reserve_factor) + 1 + 7) // 8) * 8)
            views = tuple(_owned_view(torch, radius, subset, images, incoming, metadata, rank,
                                      capacity, padding_multiple) for rank in range(world_size))
            graphs.append(RadiusSkinGraph(radius, multiplicity, subset, images, counts, views))
        buffers = [positions, edges, shifts, fixed_cell]
        for graph in graphs:
            buffers.extend((graph.edge_index, graph.integer_shifts))
            for view in graph.owned:
                buffers.extend(view.buffers())
        return OwnedNeighborEpoch(epoch, 1 if previous is None else previous.generation + 1,
            metadata, positions, edges, shifts, fixed_cell, layer_radii, float(skin), float(supplied_radius),
            supplied_counts, tuple(weighted), order, padding_multiple, float(reserve_factor), tuple(graphs),
            tuple((value, value._version) for value in buffers))


def build_owned_base_graph(prepared, radius, rank, *, device):
    """Caller-run FP32 CUDA adapter; no silent reference geometry downcast.

    CPU FP64 epochs remain useful for mathematical checks but are not accepted
    by this bridge. This dtype guard does not validate GPU cutoff rounding.
    """
    _require(isinstance(prepared, OwnedNeighborEpoch), "NEIGHBOR_PREPARED_EPOCH_REQUIRED")
    graph = prepared.radius_graph(radius)
    view = prepared.radius_view(radius, rank)
    _require(prepared.metadata.n_global > 0, "NEIGHBOR_EMPTY_GLOBAL_BASE_UNSUPPORTED")
    import torch
    _require(prepared.global_positions.dtype == prepared.cell.dtype == torch.float32,
             "NEIGHBOR_FP32_RUNTIME_GEOMETRY_REQUIRED")
    _require(not prepared.cell.requires_grad and bool(torch.isfinite(prepared.cell).all()),
             "NEIGHBOR_FIXED_FINITE_CELL_REQUIRED")
    from hyff.multigpu.base_messages import OwnedImageGraph, _eager
    device = torch.device(device)
    _eager(device)
    # The epoch has already checked the complete reciprocal integer graph and
    # constructed these exact global-capacity, input-order-preserving rows, so
    # they are copied to the device directly instead of rebuilding edge lists.
    prepared.assert_integrity()
    _require(type(rank) is int and 0 <= rank < prepared.metadata.world_size,
             "NEIGHBOR_RANK_INVALID")
    required = max(graph.incoming_counts, default=0)
    capacity = max(8, ((int(required * prepared.reserve_factor) + 1 + 7) // 8) * 8)
    _require(view.base_capacity == capacity and view.rank == rank and
             tuple(view.local_to_global.tolist()) == prepared.metadata.indices_by_rank[rank],
             "NEIGHBOR_PREPARED_VIEW_INVALID")
    return OwnedImageGraph(prepared.metadata, rank, view.local_to_global.to(device,copy=True),
        view.neighbors.to(device,copy=True), view.base_integer_shifts.to(device,copy=True),
        view.counts.to(device,copy=True), prepared.cell.to(device=device,dtype=torch.float32,copy=True).contiguous(),
        capacity)


__all__ = ["NeighborEpochError", "OwnedRadiusView", "RadiusSkinGraph", "NeighborSkinDecision",
           "OwnedNeighborEpoch", "prepare_owned_neighbor_epoch", "build_owned_base_graph",
           "explicit_fp64_squared_distance", "explicit_fp64_cutoff_squared"]
