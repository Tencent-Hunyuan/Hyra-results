"""Smooth conservative l<=2 Cartesian tensor potential (the HyFF L2 architecture) for MD.

One C2 radial switch multiplies every edge message, including biased radial
filters and the pair energy. The model accepts checked periodic/Verlet graphs.
No prediction correction is used.

The model carries scalar, vector, and symmetric-traceless rank-2 atom features.
After the first interaction, its edge messages include explicit l=1<->l=2
paths and the l=1->l=0 / l=2->l=0 directional contractions.  These
parameter-efficient scalar closures share their radial gates with the matching
cross-order paths.  All geometry enters through relative vectors, distances,
and equivariant contractions, so energies are rigid-motion invariant and
forces are conservative by construction.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


HIDDEN = 160
N_RBF = 32
N_INTERACTIONS = 4
CUTOFF = 5.0


@dataclass(frozen=True)
class L2Config:
    hidden: int = HIDDEN
    n_rbf: int = N_RBF
    interactions: int = N_INTERACTIONS
    cutoff: float = CUTOFF
    switch_width: float = 1.0
    edge_hidden: int | None = None
    tensor_hidden: int | None = None
    atomic_refinements: int = 0
    layer_cutoffs: tuple[float, ...] | None = None
    mlp_hidden: int | None = None
    scalar_source_blocks: tuple[int, ...] = ()
    higher_invariants: bool = False
    tensor_mixing_rank: int | None = None

    def __post_init__(self):
        if self.hidden < 1 or self.n_rbf < 2 or self.interactions < 2:
            raise ValueError('Invalid L2 dimensions')
        if not 0 < self.switch_width <= self.cutoff:
            raise ValueError('A positive C2 switching interval is required')
        if self.edge_hidden is not None and not 0 < self.edge_hidden <= self.hidden:
            raise ValueError('Edge width must be positive and no larger than atom width')
        if self.tensor_hidden is not None and not 0 < self.tensor_hidden <= self.hidden:
            raise ValueError('Tensor width must be positive and no larger than scalar width')
        if not isinstance(self.atomic_refinements, int) or self.atomic_refinements < 0:
            raise ValueError('The atom-only refinement count must be a nonnegative integer')
        if self.mlp_hidden is not None and (not isinstance(self.mlp_hidden, int) or self.mlp_hidden < 1):
            raise ValueError('MLP width must be a positive integer')
        scalar_blocks = tuple(self.scalar_source_blocks)
        if (any(type(index) is not int or not 0 <= index < self.interactions for index in scalar_blocks)
                or len(set(scalar_blocks)) != len(scalar_blocks)):
            raise ValueError('Scalar-source blocks must be distinct valid interaction indices')
        object.__setattr__(self, 'scalar_source_blocks', scalar_blocks)
        if type(self.higher_invariants) is not bool:
            raise ValueError('Higher invariants must be a boolean')
        if self.tensor_mixing_rank is not None and (
                type(self.tensor_mixing_rank) is not int or
                not 0 < self.tensor_mixing_rank <= (self.tensor_hidden or self.hidden)):
            raise ValueError('Tensor mixing rank must be a positive integer no larger than tensor width')
        if self.layer_cutoffs is not None:
            radii = tuple(float(radius) for radius in self.layer_cutoffs)
            if len(radii) != self.interactions or any(not math.isfinite(radius) or
                    not self.switch_width <= radius <= self.cutoff for radius in radii):
                raise ValueError('Each layer radius must be between switch width and full cutoff')
            object.__setattr__(self, 'layer_cutoffs', radii)


def smooth_switch(distance, cutoff, width):
    x = ((cutoff - distance) / width).clamp(0.0, 1.0)
    return x.pow(3) * (10.0 - 15.0 * x + 6.0 * x.square())


def _linear(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.xavier_uniform_(layer.weight)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


class _RowSelectedLinear(nn.Module):
    """Linear layer containing selected output chunks of a larger initialization.

    The discarded rows correspond to branches that are provably unreachable in
    the first or last interaction.  Initializing the full parent layer before
    slicing preserves both its active initial values and all downstream random
    number consumption, while the smaller stored matrix avoids paying for dead
    rows during training.
    """

    def __init__(
        self,
        in_features: int,
        full_out_features: int,
        rows: torch.Tensor,
        bias: bool = True,
    ):
        super().__init__()
        parent = _linear(in_features, full_out_features, bias=bias)
        self.weight = nn.Parameter(parent.weight.detach()[rows].clone())
        if bias:
            self.bias = nn.Parameter(parent.bias.detach()[rows].clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight, self.bias)


def _selected_chunk_linear(
    in_features: int,
    hidden: int,
    full_chunks: int,
    selected_chunks: tuple[int, ...],
) -> _RowSelectedLinear:
    rows = torch.cat(
        [torch.arange(chunk * hidden, (chunk + 1) * hidden) for chunk in selected_chunks]
    )
    return _RowSelectedLinear(in_features, full_chunks * hidden, rows)


class RadialBasis(nn.Module):
    def __init__(self, count: int = N_RBF, cutoff: float = CUTOFF):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, count)
        spacing = cutoff / (count - 1)
        self.register_buffer("centers", centers)
        self.register_buffer("gamma", torch.tensor(0.5 / (spacing * spacing)))
        self.cutoff = float(cutoff)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        delta = distance[:, None] - self.centers[None, :]
        basis = torch.exp(-self.gamma * delta.square())
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)
        envelope = envelope * (distance < self.cutoff).to(distance.dtype)
        return basis * envelope[:, None]


def _quadrupole(direction: torch.Tensor) -> torch.Tensor:
    """Five orthonormal coordinates of uu^T-I/3 for each unit vector."""
    x, y, z = direction.unbind(dim=-1)
    sqrt2 = math.sqrt(2.0)
    sqrt6 = math.sqrt(6.0)
    return torch.stack(
        (
            (x.square() - y.square()) / sqrt2,
            sqrt2 * x * y,
            sqrt2 * x * z,
            sqrt2 * y * z,
            (x.square() + y.square() - 2.0 * z.square()) / sqrt6,
        ),
        dim=-1,
    )


def _tensor_dot_direction(
    tensor: torch.Tensor, direction: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``T @ u`` and the same path's scalar closure ``u . T @ u``."""
    a, b, c, d, e = tensor.unbind(dim=1)
    x, y, z = direction.unbind(dim=-1)
    x = x[:, None]
    y = y[:, None]
    z = z[:, None]
    sqrt2 = math.sqrt(2.0)
    inv_sqrt6 = 1.0 / math.sqrt(6.0)
    txx = a / sqrt2 + e * inv_sqrt6
    tyy = -a / sqrt2 + e * inv_sqrt6
    tzz = -2.0 * e * inv_sqrt6
    txy = b / sqrt2
    txz = c / sqrt2
    tyz = d / sqrt2
    wx = txx * x + txy * y + txz * z
    wy = txy * x + tyy * y + tyz * z
    wz = txz * x + tyz * y + tzz * z
    vector = torch.stack((wx, wy, wz), dim=1)
    scalar = wx * x + wy * y + wz * z
    return vector, scalar


def _vector_tensor_product(
    vector: torch.Tensor, direction: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``ST(v tensor u)`` and its scalar closure ``v . u``."""
    vx, vy, vz = vector.unbind(dim=1)
    x, y, z = direction.unbind(dim=-1)
    x = x[:, None]
    y = y[:, None]
    z = z[:, None]
    sqrt2 = math.sqrt(2.0)
    sqrt6 = math.sqrt(6.0)
    vxx = vx * x
    vyy = vy * y
    vzz = vz * z
    tensor = torch.stack(
        (
            (vxx - vyy) / sqrt2,
            (vx * y + vy * x) / sqrt2,
            (vx * z + vz * x) / sqrt2,
            (vy * z + vz * y) / sqrt2,
            (vxx + vyy - 2.0 * vzz) / sqrt6,
        ),
        dim=1,
    )
    return tensor, vxx + vyy + vzz


def _channel_tensor_vector(
    tensor: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Channelwise l=2 x l=1 -> l=1 contraction for atom features."""
    a, b, c, d, e = tensor.unbind(dim=1)
    x, y, z = vector.unbind(dim=1)
    sqrt2 = math.sqrt(2.0)
    inv_sqrt6 = 1.0 / math.sqrt(6.0)
    txx = a / sqrt2 + e * inv_sqrt6
    tyy = -a / sqrt2 + e * inv_sqrt6
    tzz = -2.0 * e * inv_sqrt6
    txy = b / sqrt2
    txz = c / sqrt2
    tyz = d / sqrt2
    return torch.stack(
        (
            txx * x + txy * y + txz * z,
            txy * x + tyy * y + tyz * z,
            txz * x + tyz * y + tzz * z,
        ),
        dim=1,
    )


def _channel_vector_tensor(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Channelwise symmetric-traceless l=1 x l=1 -> l=2 product."""
    lx, ly, lz = left.unbind(dim=1)
    rx, ry, rz = right.unbind(dim=1)
    sqrt2 = math.sqrt(2.0)
    sqrt6 = math.sqrt(6.0)
    return torch.stack(
        (
            (lx * rx - ly * ry) / sqrt2,
            (lx * ry + ly * rx) / sqrt2,
            (lx * rz + lz * rx) / sqrt2,
            (ly * rz + lz * ry) / sqrt2,
            (lx * rx + ly * ry - 2.0 * lz * rz) / sqrt6,
        ),
        dim=1,
    )


def _channel_tensor_square(tensor):
    """STF(T @ T) in the five orthonormal Cartesian l=2 coordinates."""
    a, b, c, d, e = tensor.unbind(dim=1)
    xx = a / math.sqrt(2.) + e / math.sqrt(6.)
    yy = -a / math.sqrt(2.) + e / math.sqrt(6.)
    zz = -2 * e / math.sqrt(6.)
    xy, xz, yz = b / math.sqrt(2.), c / math.sqrt(2.), d / math.sqrt(2.)
    qxx = xx.square() + xy.square() + xz.square()
    qyy = yy.square() + xy.square() + yz.square()
    qzz = zz.square() + xz.square() + yz.square()
    qxy = xy * (xx + yy) + xz * yz
    qxz = xz * (xx + zz) + xy * yz
    qyz = yz * (yy + zz) + xy * xz
    return torch.stack(((qxx - qyy) / math.sqrt(2.), math.sqrt(2.) * qxy,
                        math.sqrt(2.) * qxz, math.sqrt(2.) * qyz,
                        (qxx + qyy - 2 * qzz) / math.sqrt(6.)), dim=1)


class L2AtomicRefinement(nn.Module):
    """Higher body order through atom-only equivariant polynomial updates.

    The scalar input contains the five invariants of one vector and one STF
    tensor per mixed channel. Norms keep magnitudes; normalized polynomial
    invariants remain bounded without clipping. A zero output layer starts
    this map at the identity, so adding it does not change the function at
    initialization. No new neighbors, geometry features, or pair lists are used.
    """
    def __init__(self, scalar_hidden, tensor_hidden):
        super().__init__()
        self.scalar_hidden, self.tensor_hidden = scalar_hidden, tensor_hidden
        self.scalar_norm = nn.LayerNorm(scalar_hidden)
        self.vector_mix = _linear(tensor_hidden, tensor_hidden, bias=False)
        self.tensor_mix = _linear(tensor_hidden, tensor_hidden, bias=False)
        self.update = nn.Sequential(_linear(scalar_hidden + 5 * tensor_hidden, scalar_hidden),
                                    nn.SiLU(), _linear(scalar_hidden, scalar_hidden + 2 * tensor_hidden))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.coupling = nn.Parameter(torch.full((3, tensor_hidden), .05))

    def forward(self, scalar, vector, tensor):
        v, t = self.vector_mix(vector), self.tensor_mix(tensor)
        vn = torch.sqrt(1. + v.square().sum(dim=1))
        tn = torch.sqrt(1. + t.square().sum(dim=1))
        vu, tu = v / vn[:, None, :], t / tn[:, None, :]
        tv = _channel_tensor_vector(tu, vu)
        tt = _channel_tensor_square(tu)
        invariants = (vn - 1., tn - 1., (tt * tu).sum(dim=1),
                      (vu * tv).sum(dim=1), tv.square().sum(dim=1))
        delta = self.update(torch.cat((self.scalar_norm(scalar), *invariants), dim=-1))
        ds, gv, gt = delta.split((self.scalar_hidden, self.tensor_hidden, self.tensor_hidden), dim=-1)
        gains = torch.tanh(self.coupling)
        dv = v + gains[0][None, None, :] * vn[:, None, :] * tv
        dt = t + tn[:, None, :] * (gains[1][None, None, :] * tt +
                 gains[2][None, None, :] * _channel_vector_tensor(vu, vu))
        return scalar + ds, vector + gv[:, None, :] * dv, tensor + gt[:, None, :] * dt


class L2Interaction(nn.Module):
    """Scalar, vector, and rank-2 equivariant message/update block."""

    def __init__(
        self,
        hidden: int,
        n_rbf: int,
        cross_order: bool,
        local_cross: bool,
        input_is_scalar: bool = False,
        terminal: bool = False,
        edge_hidden: int | None = None,
        tensor_hidden: int | None = None,
        cutoff: float = CUTOFF,
        switch_width: float = 1.,
        mlp_hidden: int | None = None,
        higher_invariants: bool = False,
        tensor_mixing_rank: int | None = None,
    ):
        super().__init__()
        self.cross_order = bool(cross_order)
        self.cutoff = float(cutoff)
        self.switch_width = float(switch_width)
        self.local_cross = bool(local_cross)
        self.input_is_scalar = bool(input_is_scalar)
        # This flag describes the source message, not the persistent node
        # state. Later scalar-source blocks still keep and update local tensors.
        self.higher_invariants = bool(higher_invariants)
        self.tensor_mixing_rank = tensor_mixing_rank
        self.terminal = bool(terminal)
        self.message_hidden = edge_hidden or hidden
        self.tensor_hidden = tensor_hidden or hidden
        self.scalar_hidden = hidden
        self.bottleneck = self.message_hidden != hidden
        self.tensor_bottleneck = self.message_hidden != self.tensor_hidden
        self.narrow_tensors = self.tensor_hidden != hidden
        message_width = self.message_hidden
        tensor_width = self.tensor_hidden
        mlp_width = mlp_hidden or hidden
        invariant_width = (5 if self.higher_invariants else 2) * tensor_width
        if self.input_is_scalar and self.cross_order:
            raise ValueError("a scalar-only input block cannot use cross-order paths")
        if self.terminal and self.local_cross:
            raise ValueError("terminal vector/tensor updates would be unobservable")
        five_h = 5 * hidden
        message_h = (7 if self.cross_order else 5) * message_width
        self.scalar_norm = nn.LayerNorm(hidden)
        self.update_norm = nn.LayerNorm(hidden)
        if self.input_is_scalar:
            # Scalar-source messages keep scalar, 0->1, and 0->2 paths. At
            # depth zero this is exact dead-code removal; at later depths it is
            # an explicitly configured architecture change.
            self.scalar_message = nn.Sequential(
                _linear(hidden, mlp_width),
                nn.SiLU(),
                _selected_chunk_linear(mlp_width, message_width, 5, (0, 1, 3)),
            )
            self.radial_filter = nn.Sequential(
                _linear(n_rbf, message_width),
                nn.SiLU(),
                _selected_chunk_linear(message_width, message_width, 5, (0, 1, 3)),
            )
        else:
            self.scalar_message = nn.Sequential(
                _linear(hidden, mlp_width), nn.SiLU(), _linear(mlp_width, message_h)
            )
            self.radial_filter = nn.Sequential(
                _linear(n_rbf, message_width), nn.SiLU(), _linear(message_width, message_h)
            )

        # Bias-free channel mixing commutes with rotations.  Besides the
        # same-order paths, source tensors can feed vector messages through T@u and
        # source vectors can feed tensors through ST(v tensor u).  These are the
        # inexpensive Cartesian l=2 x l=1 Clebsch-Gordan couplings.
        projection_width = tensor_mixing_rank or tensor_width
        if tensor_mixing_rank is not None:
            # U/V share one learned input basis within each irreducible order.
            # Persistent tensor states keep their full multiplicity; only the
            # local channel transformations use this optional low-rank form.
            self.vector_projection_in = _linear(tensor_width, projection_width, bias=False)
            self.tensor_projection_in = _linear(tensor_width, projection_width, bias=False)
        self.vector_u = _linear(projection_width, tensor_width, bias=False)
        self.vector_v = _linear(projection_width, tensor_width, bias=False)
        self.tensor_u = _linear(projection_width, tensor_width, bias=False)
        self.tensor_v = _linear(projection_width, tensor_width, bias=False)
        if self.local_cross:
            # Two atom-local Clebsch-Gordan update branches. They add the
            # non-radial l=1 x l=2 -> l=1 and l=1 x l=1 -> l=2 products that
            # complement the edge-aligned cross-order messages. The first
            # coupling starts gently active; the second starts at zero because
            # its product is quadratic in vector-state magnitude.
            self.vector_tensor_gain = nn.Parameter(
                torch.full((tensor_width,), math.atanh(0.05))
            )
            self.vector_vector_gain = nn.Parameter(torch.zeros(tensor_width))

        if self.narrow_tensors:
            # Multiplicity can differ between irreducible orders. Keep a wide
            # scalar state but pay tensor channel mixing only at tensor_width.
            # Invariant contractions are mapped back to scalar channels locally.
            self.update_mlp = nn.Sequential(
                _linear(hidden + invariant_width, mlp_width), nn.SiLU(),
                _linear(mlp_width, hidden + (2 if self.terminal else 4) * tensor_width),
            )
            self.invariant_vector_out = _linear(tensor_width, hidden, bias=False)
            self.invariant_tensor_out = _linear(tensor_width, hidden, bias=False)
        elif self.terminal:
            # Only the scalar update reaches the immediate terminal readout.  The
            # parent output chunks are [s, vdot, v, tdot, t]; keep 0, 1, and 3.
            self.update_mlp = nn.Sequential(
                _linear(hidden + invariant_width, mlp_width),
                nn.SiLU(),
                _selected_chunk_linear(mlp_width, hidden, 5, (0, 1, 3)),
            )
        else:
            self.update_mlp = nn.Sequential(
                _linear(hidden + invariant_width, mlp_width), nn.SiLU(), _linear(mlp_width, five_h)
            )
        self.message_scale = 0.25
        if self.tensor_bottleneck:
            if not self.input_is_scalar:
                self.edge_vector_in = _linear(tensor_width, message_width, bias=False)
                self.edge_tensor_in = _linear(tensor_width, message_width, bias=False)
        if self.bottleneck:
            self.edge_scalar_out = _linear(message_width, hidden, bias=False)
        if self.tensor_bottleneck:
            self.edge_vector_out = _linear(message_width, tensor_width, bias=False)
            self.edge_tensor_out = _linear(message_width, tensor_width, bias=False)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        tensor: torch.Tensor,
        radial: torch.Tensor,
        direction: torch.Tensor,
        quadrupole: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        edge_envelope: torch.Tensor,
        radial_output: torch.Tensor | None = None,
        csr=None,
        reference=None,
        node_message=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_atoms = scalar.shape[0]
        hidden = self.message_hidden
        if self.tensor_bottleneck:
            if self.input_is_scalar:
                message_vector = vector.new_zeros((n_atoms, 3, hidden))
                message_tensor = tensor.new_zeros((n_atoms, 5, hidden))
            else:
                message_vector = self.edge_vector_in(vector)
                message_tensor = self.edge_tensor_in(tensor)
        else:
            message_vector, message_tensor = vector, tensor
        if node_message is None:
            node_message = self.scalar_message(self.scalar_norm(scalar))
        if csr is not None:
            positions, plan, table = csr
            from hyff.kernels.fused_messages import messages
            aggregate_s, aggregate_v, aggregate_t = messages(
                positions, plan.cell, plan.neighbors, plan.shifts, plan.counts, node_message,
                message_vector, message_tensor, table, self.cutoff, self.switch_width, self.input_is_scalar,
                plan.model.fused_message_settings)
            return self._finish_messages(scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t)
        if reference is not None:
            from hyff.models.l2_chunked_reference import reference_messages
            displacement,valid,table,rbf,cutoff,width,chunk=reference
            aggregate_s,aggregate_v,aggregate_t=reference_messages(
                displacement,source,destination,node_message,
                message_vector,message_tensor,valid,table,self.radial_filter,rbf,cutoff,width,self.input_is_scalar,chunk)
            return self._finish_messages(scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t)
        edge_message = node_message[source]
        radial_output = self.radial_filter(radial) if radial_output is None else radial_output
        edge_message = (
            edge_message * radial_output
            * (self.message_scale * edge_envelope[:, None])
        )
        if self.input_is_scalar:
            msg_s, msg_vr, msg_tr = edge_message.chunk(3, dim=-1)
            edge_vector = msg_vr[:, None, :] * direction[:, :, None]
            edge_tensor = msg_tr[:, None, :] * quadrupole[:, :, None]
        elif self.cross_order:
            source_vector = message_vector[source]
            source_tensor = message_tensor[source]
            (
                msg_s,
                msg_vr,
                msg_v,
                msg_tv,
                msg_tr,
                msg_t,
                msg_vt,
            ) = edge_message.chunk(7, dim=-1)

            # Form each cross-order tensor product once and reuse its scalar
            # closure. This is algebraically identical to v.u and T:Q(u), while
            # avoiding two redundant edge contractions in both reverse passes.
            tensor_vector, tensor_projection = _tensor_dot_direction(
                source_tensor, direction
            )
            vector_tensor, vector_projection = _vector_tensor_product(
                source_vector, direction
            )
            msg_s = (
                msg_s
                + msg_vt * vector_projection
                + msg_tv * tensor_projection
            )

            edge_vector = (
                msg_v[:, None, :] * source_vector
                + msg_vr[:, None, :] * direction[:, :, None]
                + msg_tv[:, None, :] * tensor_vector
            )
            edge_tensor = (
                msg_t[:, None, :] * source_tensor
                + msg_tr[:, None, :] * quadrupole[:, :, None]
                + msg_vt[:, None, :] * vector_tensor
            )
        else:
            source_vector = message_vector[source]
            source_tensor = message_tensor[source]
            msg_s, msg_vr, msg_v, msg_tr, msg_t = edge_message.chunk(5, dim=-1)
            edge_vector = (
                msg_v[:, None, :] * source_vector
                + msg_vr[:, None, :] * direction[:, :, None]
            )
            edge_tensor = (
                msg_t[:, None, :] * source_tensor
                + msg_tr[:, None, :] * quadrupole[:, :, None]
            )

        aggregate_s = scalar.new_zeros((n_atoms, hidden))
        aggregate_v = vector.new_zeros((n_atoms, 3, hidden))
        aggregate_t = tensor.new_zeros((n_atoms, 5, hidden))
        aggregate_s.index_add_(0, destination, msg_s)
        aggregate_v.index_add_(0, destination, edge_vector)
        aggregate_t.index_add_(0, destination, edge_tensor)
        return self._finish_messages(scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t)

    def _finish_messages(self, scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t):
        if self.bottleneck:
            aggregate_s = self.edge_scalar_out(aggregate_s)
        if hasattr(self, 'folded_first_vector'):
            vector, vu, vv = self.folded_first_vector(aggregate_v).chunk(3, dim=-1)
            tensor, tu, tv = self.folded_first_tensor(aggregate_t).chunk(3, dim=-1)
            return self._node_update(scalar + aggregate_s, vector, tensor, (vu, vv, tu, tv))
        if self.tensor_bottleneck:
            aggregate_v = self.edge_vector_out(aggregate_v)
            aggregate_t = self.edge_tensor_out(aggregate_t)
        return self._node_update(scalar + aggregate_s, vector + aggregate_v, tensor + aggregate_t)

    def _node_update(self, scalar, vector, tensor, projections=None):
        if projections is not None:
            vu, vv, tu, tv = projections
        else:
            vector_basis = self.vector_projection_in(vector) if self.tensor_mixing_rank is not None else vector
            tensor_basis = self.tensor_projection_in(tensor) if self.tensor_mixing_rank is not None else tensor
            if hasattr(self, 'joined_vector'):
                vu, vv = self.joined_vector(vector_basis).chunk(2, dim=-1)
                tu, tv = self.joined_tensor(tensor_basis).chunk(2, dim=-1)
            else:
                vu = self.vector_u(vector_basis)
                vv = self.vector_v(vector_basis)
                tu = self.tensor_u(tensor_basis)
                tv = self.tensor_v(tensor_basis)
        vector_norm = torch.sqrt(vv.square().sum(dim=1) + 1.0e-8)
        tensor_norm = torch.sqrt(tv.square().sum(dim=1) + 1.0e-8)
        invariants = (vector_norm, tensor_norm)
        if self.higher_invariants:
            # The five standard invariants of a vector and an STF tensor:
            # |v|, |T|, tr(T^3), v.T.v and v.T^2.v. The additional three use
            # smooth normalization; magnitudes remain available in the norms.
            # All products are atom-local and reuse the existing channel mixes.
            v = vv / torch.sqrt(1. + vv.square().sum(dim=1))[:, None, :]
            t = tv / torch.sqrt(1. + tv.square().sum(dim=1))[:, None, :]
            vt = _channel_tensor_vector(t, v)
            tt = _channel_tensor_square(t)
            invariants += ((tt * t).sum(dim=1), (v * vt).sum(dim=1), vt.square().sum(dim=1))
        update = self.update_mlp(
            torch.cat((self.update_norm(scalar), *invariants), dim=-1)
        )
        if self.terminal:
            # No later block consumes vector/tensor output, so form only the two
            # invariant closures needed by the scalar update and final readout.
            if self.narrow_tensors:
                upd_s, upd_vdot, upd_tdot = update.split(
                    (self.scalar_hidden, self.tensor_hidden, self.tensor_hidden), dim=-1)
            else:
                upd_s, upd_vdot, upd_tdot = update.chunk(3, dim=-1)
            vector_dot = (vu * vv).sum(dim=1)
            tensor_dot = (tu * tv).sum(dim=1)
            scalar = self._scalar_update(scalar, upd_s, upd_vdot * vector_dot, upd_tdot * tensor_dot)
            return scalar, vector, tensor


        if self.narrow_tensors:
            upd_s, upd_vdot, upd_v, upd_tdot, upd_t = update.split(
                (self.scalar_hidden,) + (self.tensor_hidden,) * 4, dim=-1)
        else:
            upd_s, upd_vdot, upd_v, upd_tdot, upd_t = update.chunk(5, dim=-1)
        vector_dot = (vu * vv).sum(dim=1)
        tensor_dot = (tu * tv).sum(dim=1)
        scalar = self._scalar_update(scalar, upd_s, upd_vdot * vector_dot, upd_tdot * tensor_dot)
        if self.local_cross:
            # Use the independently mixed copies already paid for by the
            # invariant update. Small bounded gains allow local angular orders
            # to interact before the next edge pass.
            vector_cross = _channel_tensor_vector(tu, vv)
            tensor_cross = _channel_vector_tensor(vu, vv)
            vector = vector + upd_v[:, None, :] * (
                vu + torch.tanh(self.vector_tensor_gain)[None, None, :] * vector_cross
            )
            tensor = tensor + upd_t[:, None, :] * (
                tu + torch.tanh(self.vector_vector_gain)[None, None, :] * tensor_cross
            )
        else:
            vector = vector + upd_v[:, None, :] * vu
            tensor = tensor + upd_t[:, None, :] * tu
        return scalar, vector, tensor

    def _scalar_update(self, scalar, update, vector_invariant, tensor_invariant):
        if self.narrow_tensors:
            if hasattr(self, 'joined_invariants'):
                return scalar + update + self.joined_invariants(
                    torch.cat((vector_invariant, tensor_invariant), dim=-1))
            vector_invariant = self.invariant_vector_out(vector_invariant)
            tensor_invariant = self.invariant_tensor_out(tensor_invariant)
        return scalar + update + vector_invariant + tensor_invariant


class L2Potential(nn.Module):
    def __init__(self, config: L2Config | None = None):
        super().__init__()
        self.config = config or L2Config()
        hidden, n_rbf, interactions = self.config.hidden, self.config.n_rbf, self.config.interactions
        self.cutoff = self.config.cutoff
        self.layer_cutoffs = self.config.layer_cutoffs or (self.cutoff,) * interactions
        self.hidden = hidden
        self.tensor_hidden = self.config.tensor_hidden or hidden
        self.register_buffer('padding_displacement', torch.tensor([1., 0., 0.]), persistent=False)
        self.radial_tables = None
        self.fused_messages = False
        self.reference_chunk_size = 0
        self.embedding = nn.Embedding(119, hidden)
        nn.init.normal_(self.embedding.weight, std=0.32)
        self.rbf = RadialBasis(n_rbf, self.cutoff)
        self.blocks = nn.ModuleList(
            [
                L2Interaction(
                    hidden,
                    n_rbf,
                    cross_order=(index > 0 and index not in self.config.scalar_source_blocks),
                    local_cross=(index == interactions - 2),
                    input_is_scalar=(index == 0 or index in self.config.scalar_source_blocks),
                    terminal=(index == interactions - 1),
                    edge_hidden=self.config.edge_hidden,
                    tensor_hidden=self.config.tensor_hidden,
                    cutoff=self.layer_cutoffs[index],
                    switch_width=self.config.switch_width,
                    mlp_hidden=self.config.mlp_hidden,
                    higher_invariants=self.config.higher_invariants,
                    tensor_mixing_rank=self.config.tensor_mixing_rank,
                )
                for index in range(interactions)
            ]
        )
        # A readout at every body-order/depth gives short gradient paths and lets
        # shallow and deep local contributions coexist.
        self.readout_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(interactions)]
        )
        self.readouts = nn.ModuleList(
            [
                nn.Sequential(
                    _linear(hidden, self.config.mlp_hidden or hidden), nn.SiLU(),
                    _linear(self.config.mlp_hidden or hidden, 1)
                )
                for _ in range(interactions)
            ]
        )
        for readout in self.readouts:
            nn.init.normal_(readout[-1].weight, std=1.0e-3)
            nn.init.zeros_(readout[-1].bias)

        self.element_offset = nn.Embedding(119, 1)
        nn.init.zeros_(self.element_offset.weight)

        # An explicit symmetric two-body term learns bond stretching quickly,
        # leaving the equivariant network to model environment dependence.
        self.pair_coeff = nn.Embedding(64, n_rbf)
        nn.init.zeros_(self.pair_coeff.weight)
        lut = torch.full((119,), -1, dtype=torch.long)
        for index, atomic_number in enumerate((1, 6, 7, 8, 9, 15, 16, 17)):
            lut[atomic_number] = index
        self.register_buffer("element_lut", lut)
        # Construct after every other parameter, so that with zero atom-only
        # depth the RNG sequence is that of a model without refinements.
        self.refinements = nn.ModuleList([
            nn.ModuleList([L2AtomicRefinement(hidden, self.tensor_hidden)
                           for _ in range(self.config.atomic_refinements)])
            for _ in range(interactions - 1)
        ]) if self.config.atomic_refinements else nn.ModuleList()

    def forward(self, batch: Dict[str, torch.Tensor], csr=None, stable_sum=False,
                prefix_depth: int | None = None,
                return_prefix_depth: int | None = None):
        # An optional block prefix is an integrator energy group. It uses the same
        # readout scale and pair term; without a prefix the full energy is returned.
        if prefix_depth is not None and (not isinstance(prefix_depth, int) or
                                        not 1 <= prefix_depth <= len(self.blocks)):
            raise ValueError('The energy prefix must contain 1 through all interaction blocks')
        if return_prefix_depth is not None:
            if prefix_depth is not None:
                raise ValueError('Request a prefix or the full energy with a prefix, not both')
            if not isinstance(return_prefix_depth, int) or not 1 <= return_prefix_depth <= len(self.blocks):
                raise ValueError('Invalid jointly evaluated prefix depth')
        positions = batch["positions"]
        if csr is not None:
            csr.refresh_subgraphs(positions, prefix_depth or len(self.blocks))
        edge_index = batch["edge_index"]
        source, destination = edge_index[0], edge_index[1]
        if csr is None:
            displacement = positions[source] - positions[destination]
            if "shifts" in batch:
                displacement = displacement + batch["shifts"]
            # Inactive padded edges have a nonzero harmless displacement so the
            # norm Hessian remains defined during force-loss differentiation.
            valid = source != destination
            if 'edge_valid' in batch:
                valid = batch['edge_valid']
            elif 'shifts' in batch:
                valid = valid | (batch['shifts'] != 0).any(-1)
            displacement = torch.where(valid[:, None], displacement, self.padding_displacement)
            distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-9)
            direction = displacement / distance[:, None]
            quadrupole = _quadrupole(direction)
            radial = self.rbf(distance)
            envelope = smooth_switch(distance, self.cutoff, self.config.switch_width) * valid.to(distance.dtype)
        else:
            # Prepared kernels consume current coordinates directly. In this
            # path no E-by-radial or E-by-angular PyTorch arrays are needed.
            radial = direction = quadrupole = envelope = None

        atomic_numbers = batch["atomic_numbers"]
        scalar = self.embedding(atomic_numbers)
        vector = scalar.new_zeros((scalar.shape[0], 3, self.tensor_hidden))
        tensor = scalar.new_zeros((scalar.shape[0], 5, self.tensor_hidden))
        atomic_energy = scalar.new_zeros((scalar.shape[0],))
        prefix_atomic_energy = None
        readout_scale = 1.0 / math.sqrt(len(self.blocks))
        for block_index, (block, norm, readout) in enumerate(zip(
            self.blocks, self.readout_norms, self.readouts
        )):
            if prefix_depth is not None and block_index >= prefix_depth:
                break
            scalar, vector, tensor = block(
                scalar,
                vector,
                tensor,
                radial,
                direction,
                quadrupole,
                source,
                destination,
                (envelope if self.layer_cutoffs[block_index] == self.cutoff else
                 smooth_switch(distance, self.layer_cutoffs[block_index], self.config.switch_width) * valid)
                    if csr is None else None,
                self.radial_tables[block_index](distance) if self.radial_tables is not None and csr is None and not self.reference_chunk_size else None,
                (positions, csr.radius_view(self.layer_cutoffs[block_index]), self.radial_tables[block_index]) if csr is not None else None,
                (displacement,valid,self.radial_tables[block_index] if self.radial_tables is not None else None,
                 self.rbf,self.layer_cutoffs[block_index],self.config.switch_width,self.reference_chunk_size)
                if self.reference_chunk_size and csr is None else None,
                self.frozen_first_messages[atomic_numbers] if block_index == 0 and
                    hasattr(self, 'frozen_first_messages') else None,
            )
            if block_index < len(self.refinements):
                for refinement in self.refinements[block_index]:
                    scalar, vector, tensor = refinement(scalar, vector, tensor)
            atomic_energy = atomic_energy + readout_scale * readout(
                norm(scalar)
            ).squeeze(-1)
            if block_index + 1 == return_prefix_depth:
                prefix_atomic_energy = atomic_energy

        elemental = self.element_offset(atomic_numbers).squeeze(-1)
        atomic_energy = atomic_energy + elemental
        if prefix_atomic_energy is not None:
            prefix_atomic_energy = prefix_atomic_energy + elemental
        if stable_sum:
            # Extensive energies can sum hundreds of thousands of signed pair
            # terms. Accumulate them in FP64; keep learned operations/forces at
            # the model dtype. Without stable_sum the sum keeps the training order.
            atomic_energy = atomic_energy.double()
            if prefix_atomic_energy is not None:
                prefix_atomic_energy = prefix_atomic_energy.double()

        graph_ids = batch["batch"]
        n_graphs = batch["ptr"].numel() - 1
        elements = self.element_lut[atomic_numbers]
        if csr is not None:
            from hyff.kernels.fused_messages import pair_energy
            atomic_pair = pair_energy(positions, csr.radius_view(self.cutoff), elements, self.radial_tables[-1])
            atomic_energy = atomic_energy + atomic_pair
            if prefix_atomic_energy is not None:
                prefix_atomic_energy = prefix_atomic_energy + atomic_pair
        else:
            low = torch.minimum(elements[source], elements[destination])
            high = torch.maximum(elements[source], elements[destination])
            pair_index = low * 8 + high
            pair_values = 0.5 * (self.pair_coeff(pair_index) * radial).sum(dim=-1) * envelope
        def reduce_energy(atom_values):
            if stable_sum and n_graphs == 1:
                # Parallel tree reduction avoids contention on one global
                # accumulator for the large single-system MD case.
                energy = atom_values.sum().reshape(1)
                if csr is None:
                    energy = energy + pair_values.double().sum().reshape(1)
                return energy
            energy = atom_values.new_zeros((n_graphs,))
            energy.index_add_(0, graph_ids, atom_values)
            if csr is None:
                energy.index_add_(0, graph_ids[source], pair_values.to(energy.dtype))
            return energy

        energy = reduce_energy(atomic_energy)
        return energy if prefix_atomic_energy is None else (energy, reduce_energy(prefix_atomic_energy))

    def energy(self, batch, edges=None, csr=None, stable_sum=True, prefix_depth=None):
        from hyff.graph import PeriodicEdges, lattice_translation, radius_edges
        if edges is None:
            edges = radius_edges(batch, self.cutoff)
        inputs = {**batch}
        if isinstance(edges, PeriodicEdges):
            inputs['edge_index'] = edges.edge_index
            if csr is None:
                inputs['shifts'] = lattice_translation(batch, edges.unit_shifts, edges.edge_index[1], batch['positions'].dtype)
                inputs['edge_valid'] = (edges.edge_index[0] != edges.edge_index[1]) | (edges.unit_shifts != 0).any(-1)
        else:
            inputs['edge_index'] = edges
            inputs.pop('shifts', None)
        return self(inputs, csr=csr, stable_sum=stable_sum, prefix_depth=prefix_depth)

    def compile_radial_tables(self, intervals=200, order=5):
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError('Radial compilation requires frozen model parameters')
        from hyff.kernels.radial_table import CubicRadialTable, QuinticRadialTable
        if order not in (3, 5):
            raise ValueError('Supported C2 table orders are cubic and quintic')
        table_class = CubicRadialTable if order == 3 else QuinticRadialTable
        parameter = next(self.parameters())
        self.radial_tables = nn.ModuleList([
            table_class.compile(self.rbf, block.radial_filter, intervals, table_cutoff=block.cutoff).to(parameter)
            for block in self.blocks
        ])
        # The final table contains the 64 pair-type radial functions. A meta
        # linear avoids consuming the RNG during frozen inference preparation.
        pair_linear = nn.Linear(self.config.n_rbf, 64, bias=False, device='meta')
        pair_linear.weight = nn.Parameter(self.pair_coeff.weight.detach(), requires_grad=False)
        self.radial_tables.append(table_class.compile(self.rbf, [pair_linear], intervals).to(parameter))
        self.radial_table_order = order
        return self

    def enable_fused_messages(self, intervals=200, settings=(32,8,4,False), radial_order=5,
                              filter_skin=False):
        if self.radial_tables is None:
            self.compile_radial_tables(intervals, order=radial_order)
        self.fused_messages = True
        if settings[0] not in (8,16,32,64) or settings[1] not in (4,8,16,32) or settings[2] not in (4,8):
            raise ValueError('Unsupported inference tiling')
        self.fused_message_settings = tuple(settings)
        self.filter_skin_neighbors = bool(filter_skin)
        self.fused = True
        self.inference_backend = 'l2_csr_cubic' if self.radial_tables[0].coefficients.shape[1] == 4 else 'l2_csr_quintic'
        return self

    def enable_fast_nodes(self, precision='tf32x3'):
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError('Freeze model parameters before selecting an inference backend')
        from hyff.kernels.fused_linear import FastLinear, enable_fast_nodes
        def convert_selected(module):
            for name, child in list(module.named_children()):
                if isinstance(child, _RowSelectedLinear):
                    child.in_features, child.out_features = child.weight.shape[1], child.weight.shape[0]
                    setattr(module, name, FastLinear(child, precision))
                else:
                    convert_selected(child)
        convert_selected(self)
        enable_fast_nodes(self, fuse_activations=True, strided_inputs=True, precision=precision)
        from hyff.kernels.layernorm import enable_frozen_layernorm
        enable_frozen_layernorm(self)
        self.fast_layernorm_width_tiles = {128: 16, 160: 8}
        return self

    def enable_compiled_nodes(self):
        """Fuse atom-local projections, invariants and gates with AOTAutograd.

        The model's number of atoms is fixed during MD. Neighbor traversal and
        its independent reference remain outside these compiled functions.
        """
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError('Atom compilation requires frozen parameters')
        for block in self.blocks:
            if getattr(block, 'compiled_nodes', False):
                continue
            block.uncompiled_finish_messages = block._finish_messages
            block._finish_messages = torch.compile(
                block._finish_messages, fullgraph=True, dynamic=False,
                options={'triton.cudagraphs': False})
            block.compiled_nodes = True
        for group in self.refinements:
            for refinement in group:
                if getattr(refinement, 'compiled_nodes', False):
                    continue
                refinement.uncompiled_forward = refinement.forward
                refinement.forward = torch.compile(refinement.forward, fullgraph=True, dynamic=False,
                    options={'triton.cudagraphs': False})
                refinement.compiled_nodes = True
        self.compiled_nodes = True
        return self

    def prepare_energy(self, batch, edges):
        if self.fused_messages:
            from hyff.kernels.fused_messages import PreparedL2Neighbors
            return PreparedL2Neighbors(self, batch, edges)
        return lambda positions: self.energy({**batch, 'positions': positions}, edges)


def load_l2_checkpoint(path, device='cpu'):
    raise ValueError('Checkpoint loading is not available here; use hyff.weights.load_model')
