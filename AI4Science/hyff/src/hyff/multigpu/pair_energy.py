"""Owned-atom pair-energy kernels using the species-general polynomial pair table.

The full directed graph must be reciprocal, including image shifts. Only
owned rows are computed/stored; global positions/species/upstream weights are
read. Backward returns a COMPLETE owned position derivative, not a partial
field for SUM reduce-scatter. CUDA graph capture of these kernels is not tested.
The single-GPU pair kernels are reused unchanged; no device or process group is
selected here.
"""
from dataclasses import dataclass
import math
import torch
import triton as tr
import triton.language as tl
from hyff.kernels.pair_energy import _pair_value
from hyff.multigpu.base_messages import OwnedImageGraph, _owned_geometry
from hyff.multigpu.collectives import DISJOINT_OUTPUT, gather_disjoint_output_adjoints


@tr.jit
def _owned_pair_forward(P,Cell,Neighbors,Shifts,Counts,OwnedRows,Elements,Coeff,Energy,
                  K,INTERVALS:tl.constexpr,CUTOFF:tl.constexpr,WIDTH:tl.constexpr,BN:tl.constexpr,NC:tl.constexpr=6, NE:tl.constexpr=17):
    node=tl.program_id(0)
    global_node=tl.load(OwnedRows+node)
    own=tl.load(Elements+global_node)
    total=tl.full((BN,),0.,tl.float32)
    for start in range(tl.cdiv(tl.load(Counts+node),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,valid,r,_,_,_=_owned_geometry(P,Cell,Neighbors,Shifts,OwnedRows,node,slot,K)
        other=tl.load(Elements+j)
        pair=tl.minimum(own,other)*NE+tl.maximum(own,other)
        value,_=_pair_value(Coeff,r,pair,valid,INTERVALS,CUTOFF,WIDTH,NC,NE)
        total+=.5*value
    tl.store(Energy+node,tl.sum(total,0))


@tr.jit
def _owned_pair_backward(P,Cell,Neighbors,Shifts,Counts,OwnedRows,Elements,Coeff,Gradient,GP,
                   K,INTERVALS:tl.constexpr,CUTOFF:tl.constexpr,WIDTH:tl.constexpr,BN:tl.constexpr,NC:tl.constexpr=6, NE:tl.constexpr=17):
    node=tl.program_id(0)
    global_node=tl.load(OwnedRows+node)
    own=tl.load(Elements+global_node)
    g=tl.load(Gradient+global_node)
    gx=tl.full((BN,),0.,tl.float32)
    gy,gz=gx,gx
    for start in range(tl.cdiv(tl.load(Counts+node),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,valid,r,x,y,z=_owned_geometry(P,Cell,Neighbors,Shifts,OwnedRows,node,slot,K)
        other=tl.load(Elements+j)
        pair=tl.minimum(own,other)*NE+tl.maximum(own,other)
        _,slope=_pair_value(Coeff,r,pair,valid,INTERVALS,CUTOFF,WIDTH,NC,NE)
        # Both endpoint energies contain half of this symmetric pair. This
        # expression also supports nonuniform atomic upstream derivatives.
        scale=-.5*(g+tl.load(Gradient+j))*slope
        gx+=scale*x;gy+=scale*y;gz+=scale*z
    tl.store(GP+node*3,tl.sum(gx,0))
    tl.store(GP+node*3+1,tl.sum(gy,0))
    tl.store(GP+node*3+2,tl.sum(gz,0))


class OwnedPairError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(value, code):
    if not value:
        raise OwnedPairError(code)


@dataclass(frozen=True)
class OwnedPairOperator:
    graph: OwnedImageGraph
    elements: torch.Tensor
    coefficients: torch.Tensor
    cutoff: float
    switch_width: float
    n_elements: int
    fma: bool
    adjoint_spec: object = None


@dataclass(frozen=True)
class PairSnapshot:
    owned_positions: torch.Tensor
    global_positions: torch.Tensor
    operator: OwnedPairOperator
    origin: str


def prepare_pair_operator(graph, elements, coefficients, *, cutoff, switch_width,
                          n_elements, fma=False, adjoint_spec=None):
    _require(isinstance(graph, OwnedImageGraph), 'PREPARED_IMAGE_GRAPH_REQUIRED')
    device = graph.cell.device
    _require(device.type == 'cuda' and device.index == torch.cuda.current_device() and
             not torch.cuda.is_current_stream_capturing(), 'EAGER_CURRENT_DEVICE_REQUIRED')
    _require(type(n_elements) is int and n_elements > 0 and type(fma) is bool,
             'STATIC_PAIR_SETTINGS_REQUIRED')
    _require(all(type(v) in (int,float) and math.isfinite(v) and v > 0
                 for v in (cutoff,switch_width)) and switch_width <= cutoff,
             'PAIR_RADIAL_BOUNDS_INVALID')
    _require(elements.device == device and elements.dtype == torch.int64 and
             elements.shape == (graph.n_global,) and elements.is_contiguous() and
             bool(((elements >= 0) & (elements < n_elements)).all()), 'GLOBAL_SPECIES_INDICES_REQUIRED')
    _require(coefficients.device == device and coefficients.dtype == torch.float32 and
             coefficients.ndim == 3 and coefficients.shape[0] > 0 and coefficients.shape[1] in (4,6) and
             coefficients.shape[2] == n_elements*n_elements and coefficients.is_contiguous() and
             not coefficients.requires_grad and coefficients.grad_fn is None and
             bool(torch.isfinite(coefficients).all()), 'FROZEN_GENERAL_PAIR_TABLE_REQUIRED')
    if adjoint_spec is not None:
        _require(adjoint_spec.contract == DISJOINT_OUTPUT and adjoint_spec.tail_shape == () and
                 adjoint_spec.dtype == torch.float32 and adjoint_spec.runtime.metadata == graph.metadata and
                 adjoint_spec.runtime.rank == graph.rank and adjoint_spec.runtime.device == device,
                 'PAIR_ADJOINT_SPEC_MISMATCH')
    return OwnedPairOperator(graph,elements,coefficients,float(cutoff),float(switch_width),
                             n_elements,fma,adjoint_spec)


def _position_metadata(positions, graph, *, local=False, detached=False):
    _require(isinstance(positions,torch.Tensor) and positions.device == graph.cell.device and
             positions.dtype == torch.float32 and positions.shape == (graph.n_local if local else graph.n_global,3)
             and positions.is_contiguous(), 'PAIR_POSITION_METADATA_MISMATCH')
    if detached:
        _require(not positions.requires_grad and positions.grad_fn is None, 'DETACHED_PAIR_CACHE_REQUIRED')


@torch.no_grad()
def verify_pair_snapshot(owned_positions, global_positions, operator):
    graph=operator.graph
    _require(not torch.cuda.is_current_stream_capturing(), 'EAGER_PAIR_SNAPSHOT_REQUIRED')
    _position_metadata(owned_positions,graph,local=True)
    _position_metadata(global_positions,graph,detached=True)
    _require(bool(torch.isfinite(global_positions).all()) and torch.equal(
        owned_positions,global_positions.index_select(0,graph.local_rows)), 'PAIR_SOURCE_SNAPSHOT_MISMATCH')
    return PairSnapshot(owned_positions,global_positions,operator,'eager_verified_prefilled')


def _snapshot_metadata(snapshot):
    _require(isinstance(snapshot,PairSnapshot) and snapshot.origin in ('eager_verified_prefilled', 'detached_collectives'),
             'PAIR_SOURCE_ORIGIN_REQUIRED')
    _position_metadata(snapshot.owned_positions,snapshot.operator.graph,local=True)
    _position_metadata(snapshot.global_positions,snapshot.operator.graph,detached=True)


@torch.no_grad()
def owned_pair_forward(snapshot):
    _snapshot_metadata(snapshot)
    op=snapshot.operator; graph=op.graph
    result=snapshot.owned_positions.new_empty(graph.n_local)
    if graph.n_local:
        _owned_pair_forward[(graph.n_local,)](snapshot.global_positions,graph.cell,graph.neighbors,
            graph.shifts,graph.counts,graph.local_rows,op.elements,op.coefficients,result,
            graph.capacity,op.coefficients.shape[0],op.cutoff,op.switch_width,32,
            NC=op.coefficients.shape[1],NE=op.n_elements,num_warps=4,enable_fp_fusion=op.fma)
    return result


@torch.no_grad()
def owned_pair_vjp(snapshot, global_atomic_cotangent):
    _snapshot_metadata(snapshot)
    op=snapshot.operator; graph=op.graph
    _require(global_atomic_cotangent.device == graph.cell.device and
             global_atomic_cotangent.dtype == torch.float32 and
             global_atomic_cotangent.shape == (graph.n_global,) and
             not global_atomic_cotangent.requires_grad, 'DETACHED_GLOBAL_ATOMIC_COTANGENT_REQUIRED')
    gradient=snapshot.owned_positions.new_empty((graph.n_local,3))
    if graph.n_local:
        _owned_pair_backward[(graph.n_local,)](snapshot.global_positions,graph.cell,graph.neighbors,
            graph.shifts,graph.counts,graph.local_rows,op.elements,op.coefficients,
            global_atomic_cotangent.contiguous(),gradient,graph.capacity,op.coefficients.shape[0],
            op.cutoff,op.switch_width,32,NC=op.coefficients.shape[1],NE=op.n_elements,
            num_warps=4,enable_fp_fusion=op.fma)
    return gradient


class _OwnedPairEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owned_positions, snapshot):
        op=snapshot.operator; graph=op.graph
        ctx.operator=op
        ctx.origin=snapshot.origin
        ctx.save_for_backward(owned_positions,snapshot.global_positions,op.elements,op.coefficients,
                              graph.cell,graph.neighbors,graph.shifts,graph.counts,graph.local_rows)
        return owned_pair_forward(snapshot)

    @staticmethod
    def backward(ctx, local_cotangent):
        _require(not torch.is_grad_enabled(), 'PAIR_INFERENCE_FIRST_DERIVATIVE_ONLY')
        saved=ctx.saved_tensors
        snapshot=PairSnapshot(saved[0],saved[1],ctx.operator,ctx.origin)
        cotangent=gather_disjoint_output_adjoints(local_cotangent,ctx.operator.adjoint_spec)
        return owned_pair_vjp(snapshot,cotangent),None


def owned_pair_energy(snapshot):
    _snapshot_metadata(snapshot)
    _require(snapshot.operator.adjoint_spec is not None and snapshot.owned_positions.requires_grad,
             'PAIR_OWNED_GRADIENT_AND_ADJOINT_SPEC_REQUIRED')
    return _OwnedPairEnergy.apply(snapshot.owned_positions,snapshot)


__all__=['OwnedPairOperator','PairSnapshot','OwnedPairError','prepare_pair_operator',
         'verify_pair_snapshot','owned_pair_forward','owned_pair_vjp','owned_pair_energy']
