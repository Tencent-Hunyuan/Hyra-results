"""Owned-output EFM term that reuses the single-GPU edge computation unchanged.

Inputs S/V/T are already projected GLOBAL source rows. Destination indices are
LOCAL and ``n_owned`` is explicit. The caller owns projection, differentiable
source/position gathers, and their single SUM transpose. This module performs
no collective, model installation, process-group setup, or device selection.

Only the seven compact inputs and integer/mask metadata survive forward. A
first-order backward replays one tile and returns all seven input cotangents.
The exit modules execute outside the custom aggregate, on owned rows only.
Input validation is eager, so this term does not run under CUDA graph capture.
"""
from __future__ import annotations

import math

import torch

from hyff.engines.capture.tiled_recompute import _edge_outputs
from hyff.kernels.efm_rotated import frame_matrices
from hyff.models.l2_potential import smooth_switch


MAX_TILE_EDGES = 32768


class OwnedProjectedEFMError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise OwnedProjectedEFMError(code)


def _metadata(source, destination, valid, n_global, n_owned, device):
    _require(type(n_owned) is int and 0 <= n_owned <= n_global, "EFM_OWNED_COUNT_INVALID")
    _require(all(isinstance(value, torch.Tensor) and value.device == device
                 for value in (source, destination, valid)), "EFM_METADATA_DEVICE_INVALID")
    _require(source.dtype == destination.dtype == torch.int64 and valid.dtype == torch.bool,
             "EFM_METADATA_DTYPE_INVALID")
    _require(source.ndim == 1 and source.shape == destination.shape == valid.shape,
             "EFM_METADATA_SHAPE_INVALID")
    # Validation is deliberately eager: boolean indexing reads values on the
    # host, so this path cannot run under CUDA graph capture.
    active_source, active_destination = source[valid], destination[valid]
    _require(bool(((active_source >= 0) & (active_source < n_global)).all()),
             "EFM_SOURCE_OUT_OF_RANGE")
    _require(bool(((active_destination >= 0) & (active_destination < n_owned)).all()),
             "EFM_DESTINATION_OUT_OF_RANGE")


def _input_shapes(efm, values, source, destination, valid, n_owned):
    S, V, T, rbf, R, D, envelope = values
    _require(all(isinstance(value, torch.Tensor) for value in values), "EFM_TENSOR_REQUIRED")
    _require(S.ndim == 2 and S.shape[1] == efm.channels, "EFM_SOURCE_SHAPE_INVALID")
    _require(S.dtype in (torch.float32, torch.float64) and
             all(value.dtype == S.dtype and value.device == S.device for value in values),
             "EFM_INPUT_TYPE_INVALID")
    n_global, count = S.shape[0], source.numel()
    _metadata(source, destination, valid, n_global, n_owned, S.device)
    if efm.m_blocks:
        _require(V.shape == (n_global, 3, efm.channels) and T.shape == (n_global, 5, efm.channels),
                 "EFM_SOURCE_SHAPE_INVALID")
    else:
        _require(V.shape == T.shape == (0,), "EFM_SCALAR_PLACEHOLDER_INVALID")
    _require(rbf.ndim == 2 and rbf.shape == (count, efm.radial[0].in_features) and
             R.shape == (count, 3, 3) and D.shape == (count, 5, 5) and envelope.shape == (count,),
             "EFM_GEOMETRY_SHAPE_INVALID")
    _require(all(parameter.device == S.device and parameter.dtype == S.dtype
                 for parameter in efm.parameters()), "EFM_PARAMETER_TYPE_INVALID")
    _require(all(bool(torch.isfinite(value).all()) for value in values[:3]) and
             all(bool(torch.isfinite(value[valid]).all()) for value in values[3:]),
             "EFM_ACTIVE_INPUT_NONFINITE")


def _masked(value, valid, fallback=0.0):
    mask = valid.reshape(valid.shape[0], *((1,) * (value.ndim - 1)))
    return torch.where(mask, value, value.new_full((), fallback))


def _tile_inputs(values, source, valid, part, has_m):
    S, V, T, rbf, R, D, envelope = values
    # Invalid indices may be negative or arbitrarily large; sanitize before
    # every gather. A globally empty input still has a private safe read row.
    index = torch.where(valid, source, torch.zeros_like(source))

    def gather(value):
        readable = value if value.shape[0] else value.new_zeros((1, *value.shape[1:]))
        return _masked(readable.index_select(0, index), valid)

    return (gather(S), gather(V) if has_m else V, gather(T) if has_m else T,
            _masked(rbf[part], valid), _masked(R[part], valid),
            _masked(D[part], valid), _masked(envelope[part], valid))


def make_owned_projected_term(efm, tile_edges=MAX_TILE_EDGES, compile_tiles=False):
    """Return ``term(S,V,T,rbf,R,D,envelope,src,dst_local,valid,n_owned)``.

    Scalar-source V/T must be explicit empty tensors of shape ``(0,)``.
    Invalid edges may contain arbitrary indices and nonfinite geometry; they
    produce exactly zero outputs and all seven cotangents. Valid inputs must
    be finite. Metadata validation synchronizes with the host.

    The caller installs this term on ALL active EFM blocks with a tile size
    (<= 32768) fixed per construction and epoch. Path selection belongs to the
    caller; this factory never chooses a path from n_owned or estimates memory
    fit from N/P. Scalar-source tiles are never compiled and use the eager edge
    function.
    """
    _require(type(tile_edges) is int and 0 < tile_edges <= MAX_TILE_EDGES, "EFM_TILE_SIZE_INVALID")
    _require(not any(parameter.requires_grad for parameter in efm.parameters()),
             "EFM_FROZEN_PARAMETERS_REQUIRED")
    _require(efm.aggregation == "sum" and efm.frame == "projector" and
             not efm.complex_structure and getattr(efm, "receiver_in", None) is None,
             "EFM_VARIANT_UNSUPPORTED")
    has_m = bool(efm.m_blocks)
    compile_requested = bool(compile_tiles)
    compiled = compile_requested and not bool(efm.input_is_scalar)

    def native_edges(*values):
        return _edge_outputs(efm, *values)

    edge_function = (torch.compile(native_edges, fullgraph=True, dynamic=False,
                                   options={"triton.cudagraphs": False})
                     if compiled else native_edges)

    class OwnedAggregate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, S, V, T, rbf, R, D, envelope, source, destination, valid, n_owned):
            values = (S, V, T, rbf, R, D, envelope)
            ctx.save_for_backward(*values, source, destination, valid)
            ctx.n_owned = n_owned
            C = efm.channels
            # The final scratch row absorbs padding, including n_owned == 0.
            totals = (S.new_zeros((n_owned + 1, C)), S.new_zeros((n_owned + 1, 3, C)),
                      S.new_zeros((n_owned + 1, 5, C)))
            for begin in range(0, source.numel(), tile_edges):
                part = slice(begin, begin + tile_edges)
                mask = valid[part]
                dst = torch.where(mask, destination[part], torch.full_like(destination[part], n_owned))
                local = _tile_inputs(values, source[part], mask, part, has_m)
                for total, edge in zip(totals, edge_function(*local)):
                    total.index_add_(0, dst, _masked(edge, mask))
            return tuple(total[:n_owned] for total in totals)

        @staticmethod
        def backward(ctx, grad_s, grad_v, grad_t):
            if torch.is_grad_enabled():
                raise RuntimeError("EFM_FIRST_DERIVATIVE_ONLY")
            saved = ctx.saved_tensors
            values, source, destination, valid = saved[:7], saved[7], saved[8], saved[9]
            needs = ctx.needs_input_grad[:7]
            gradients = [torch.zeros_like(value) if needed else None
                         for value, needed in zip(values, needs)]
            # Empty output cotangents get the same harmless sentinel row.
            cotangent_rows = tuple(torch.cat((gradient, gradient.new_zeros((1, *gradient.shape[1:]))), 0)
                                   for gradient in (grad_s, grad_v, grad_t))
            for begin in range(0, source.numel(), tile_edges):
                part = slice(begin, begin + tile_edges)
                mask = valid[part]
                src = torch.where(mask, source[part], torch.zeros_like(source[part]))
                dst = torch.where(mask, destination[part], torch.full_like(destination[part], ctx.n_owned))
                local = _tile_inputs(values, source[part], mask, part, has_m)
                with torch.enable_grad():
                    leaves = tuple(value.detach().requires_grad_(True) for value in local)
                    outputs = tuple(_masked(edge, mask) for edge in edge_function(*leaves))
                    cotangents = tuple(gradient.index_select(0, dst) for gradient in cotangent_rows)
                    local_gradients = torch.autograd.grad(outputs, leaves, cotangents,
                                                          allow_unused=True, create_graph=False)
                for index, (total, gradient) in enumerate(zip(gradients, local_gradients)):
                    if total is None or gradient is None:
                        continue
                    if index < 3:
                        if total.shape[0]:
                            total.index_add_(0, src, gradient)
                    else:
                        total[part].copy_(gradient)
            return (*gradients, None, None, None, None)

    def aggregate(S, V, T, rbf, R, D, envelope, source, destination, valid, n_owned):
        _require(not any(parameter.requires_grad for parameter in efm.parameters()),
                 "EFM_FROZEN_PARAMETERS_REQUIRED")
        values = (S, V, T, rbf, R, D, envelope)
        # Validation is control work: boolean indexing in finite-value checks
        # must not create even a temporary autograd tape or saved-tensor hook.
        with torch.no_grad():
            _input_shapes(efm, values, source, destination, valid, n_owned)
        return OwnedAggregate.apply(*values, source, destination, valid, n_owned)

    def term(S, V, T, rbf, R, D, envelope, source, destination, valid, n_owned):
        scalar, vector, tensor = aggregate(S, V, T, rbf, R, D, envelope,
                                            source, destination, valid, n_owned)
        return efm.exit_scalar(scalar), efm.exit_vector(vector), efm.exit_tensor(tensor)

    term.aggregate = aggregate
    term.owned_efm_metadata = {
        "schema": "LARGO_OWNED_PROJECTED_EFM", "tile_edges": tile_edges,
        "tile_edges_cap": MAX_TILE_EDGES, "block_policy": "ALL_ACTIVE_EFM_BLOCKS",
        "tile_policy": "FIXED_CONSTRUCTION_EPOCH_NO_OWNED_COUNT_THRESHOLD",
        "compile_tiles_requested": compile_requested, "compile_tiles": compiled,
        "compile_fallback": "KNOWN_SCALAR_SOURCE_BACKWARD_DEFECT" if compile_requested and not compiled else None,
        "global_projected_source_inputs": True, "owned_output_rows": True,
        "explicit_validity_mask": True, "all_seven_cotangents": True,
        "source_transpose": "CALLER_SINGLE_SUM", "first_derivative_only": True,
        "native_edge_outputs_imported_unchanged": True, "expanded_all_edge_tape_retained": False,
        "capture_validated": False, "gpu_validated": False,
    }
    return term


def owned_edge_geometry(positions, owned_global_indices, source, destination, valid, unit_shifts, cell,
                        *, centers, gamma, rbf_cutoff, radius, width):
    """Native geometry with global endpoints and safe explicit padding.

    ``destination`` is local; it is mapped through ``owned_global_indices``
    before geometry. ``unit_shifts`` are integer lattice shifts. Nonzero self
    images are valid; zero-shift self edges must not be marked valid.
    """
    _require(isinstance(positions, torch.Tensor) and positions.ndim == 2 and positions.shape[1] == 3,
             "EFM_POSITION_SHAPE_INVALID")
    _require(positions.dtype in (torch.float32, torch.float64), "EFM_INPUT_TYPE_INVALID")
    _require(isinstance(owned_global_indices, torch.Tensor) and owned_global_indices.ndim == 1 and
             owned_global_indices.dtype == torch.int64 and owned_global_indices.device == positions.device,
             "EFM_OWNER_MAP_INVALID")
    n_global, n_owned = positions.shape[0], owned_global_indices.numel()
    _metadata(source, destination, valid, n_global, n_owned, positions.device)
    _require(bool(((owned_global_indices >= 0) & (owned_global_indices < n_global)).all()) and
             torch.unique(owned_global_indices).numel() == n_owned, "EFM_OWNER_MAP_INVALID")
    _require(unit_shifts.shape == (source.numel(), 3) and unit_shifts.dtype == torch.int64 and
             unit_shifts.device == positions.device, "EFM_SHIFT_SHAPE_INVALID")
    _require(cell.shape == (3, 3) and cell.device == positions.device and cell.dtype == positions.dtype and
             centers.ndim == 1 and centers.device == positions.device and centers.dtype == positions.dtype,
             "EFM_GEOMETRY_TYPE_INVALID")
    _require(all(type(value) in (int, float) and math.isfinite(value) and value > 0
                 for value in (rbf_cutoff, radius, width)) and width <= radius,
             "EFM_CUTOFF_INVALID")
    safe_source = torch.where(valid, source, torch.zeros_like(source))
    owner_map = torch.cat((owned_global_indices, owned_global_indices.new_zeros(1)), 0)
    safe_local = torch.where(valid, destination, torch.full_like(destination, n_owned))
    global_destination = owner_map.index_select(0, safe_local)
    shifts = torch.where(valid[:, None], unit_shifts, torch.zeros_like(unit_shifts))
    _require(bool((~valid | (safe_source != global_destination) | (shifts != 0).any(dim=-1)).all()),
             "EFM_ZERO_SHIFT_SELF_EDGE")
    readable = positions if n_global else positions.new_zeros((1, 3))
    displacement = (readable.index_select(0, safe_source) -
                    readable.index_select(0, global_destination) + shifts.to(positions.dtype) @ cell)
    padding = positions.new_tensor((1.0, 0.0, 0.0))
    displacement = torch.where(valid[:, None], displacement, padding)
    distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-9)
    direction = displacement / distance[:, None]
    basis = torch.exp(-gamma * (distance[:, None] - centers[None, :]).square())
    radial_envelope = 0.5 * (torch.cos(math.pi * distance / rbf_cutoff) + 1.0)
    radial_envelope = radial_envelope * (distance < rbf_cutoff).to(distance.dtype)
    rbf = basis * radial_envelope[:, None]
    envelope = smooth_switch(distance, radius, width) * valid.to(distance.dtype)
    R, D = frame_matrices(direction)
    return rbf, R, D, envelope
