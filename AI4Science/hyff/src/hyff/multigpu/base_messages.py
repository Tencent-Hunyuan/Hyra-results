"""Owned-row version of the fused L2 message kernels for multi-GPU largo (C84 channels).

The only arithmetic changes relative to the single-GPU kernels are local/global
index selection. Complete owned gradients return directly to local inputs. Full
source caches are detached. Preparation and origin checks are eager-only.
Only the 84-channel layout is supported; process groups come from the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import triton as tr
import triton.language as tl

from hyff.kernels.fused_messages import _filters, _vjp, _load_states, _load_message, _q, _matvec, _stv
from hyff.multigpu.collectives import (
    OwnershipMetadata, DETACHED_SOURCE, DISJOINT_OUTPUT, gather_detached_sources,
    gather_disjoint_output_adjoints,
)


@tr.jit
def _owned_geometry(P,Cell,Neighbors,Shifts,OwnedRows,node,slot,K):
    global_node=tl.load(OwnedRows+node)
    neighbor=tl.load(Neighbors+node*K+slot,slot<K,-1)
    valid=(slot<K)&(neighbor>=0)
    neighbor=tl.maximum(neighbor,0)
    sx=tl.load(Shifts+(node*K+slot)*3,slot<K,0).to(tl.float32)
    sy=tl.load(Shifts+(node*K+slot)*3+1,slot<K,0).to(tl.float32)
    sz=tl.load(Shifts+(node*K+slot)*3+2,slot<K,0).to(tl.float32)
    dx=tl.load(P+neighbor*3)-tl.load(P+global_node*3)+sx*tl.load(Cell)+sy*tl.load(Cell+3)+sz*tl.load(Cell+6)
    dy=tl.load(P+neighbor*3+1)-tl.load(P+global_node*3+1)+sx*tl.load(Cell+1)+sy*tl.load(Cell+4)+sz*tl.load(Cell+7)
    dz=tl.load(P+neighbor*3+2)-tl.load(P+global_node*3+2)+sx*tl.load(Cell+2)+sy*tl.load(Cell+5)+sz*tl.load(Cell+8)
    dx,dy,dz=tl.where(valid,dx,1.),tl.where(valid,dy,0.),tl.where(valid,dz,0.)
    r=tl.sqrt(dx*dx+dy*dy+dz*dz)
    return neighbor,valid,r,dx/r,dy/r,dz/r


@tr.jit
def _owned_forward(P,Cell,Neighbors,Shifts,Counts,OwnedRows,M,V,T,Coeff,Sout,Vout,Tout,
                   C:tl.constexpr,K,INTERVALS:tl.constexpr,CUTOFF:tl.constexpr,WIDTH:tl.constexpr,
                   FIRST:tl.constexpr,BC:tl.constexpr,BN:tl.constexpr,NC:tl.constexpr=6):
    node=tl.program_id(0)
    ch=tl.program_id(1)*BC+tl.arange(0,BC)
    ss=tl.full((BC,),0.,tl.float32)
    svx,svy,svz,st0,st1,st2,st3,st4=ss,ss,ss,ss,ss,ss,ss,ss
    for start in range(tl.cdiv(tl.load(Counts+node),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,mask,r,ux,uy,uz=_owned_geometry(P,Cell,Neighbors,Shifts,OwnedRows,node,slot,K)
        u,v,w=ux[:,None],uy[:,None],uz[:,None]
        channel=ch[None,:]
        (w0,w1,w2,w3,w4,w5,w6,_,_,_,_,_,_,_)=_filters(Coeff,r[:,None],channel,mask[:,None],C,INTERVALS,CUTOFF,WIDTH,FIRST,NC)
        m0,m1,m2,m3,m4,m5,m6=_load_message(M,j[:,None],channel,C,FIRST)
        a0,a1,a2,a3,a4,a5,a6=m0*w0,m1*w1,m2*w2,m3*w3,m4*w4,m5*w5,m6*w6
        q0,q1,q2,q3,q4=_q(u,v,w)
        if FIRST:
            es=a0
            evx,evy,evz=a1*u,a1*v,a1*w
            et0,et1,et2,et3,et4=a4*q0,a4*q1,a4*q2,a4*q3,a4*q4
        else:
            vx,vy,vz,t0,t1,t2,t3,t4=_load_states(V,T,j[:,None],channel,C)
            tx,ty,tz=_matvec(t0,t1,t2,t3,t4,u,v,w)
            b0,b1,b2,b3,b4=_stv(vx,vy,vz,u,v,w)
            es=a0+a6*(vx*u+vy*v+vz*w)+a3*(tx*u+ty*v+tz*w)
            evx,evy,evz=a2*vx+a1*u+a3*tx,a2*vy+a1*v+a3*ty,a2*vz+a1*w+a3*tz
            et0,et1,et2,et3,et4=a5*t0+a4*q0+a6*b0,a5*t1+a4*q1+a6*b1,a5*t2+a4*q2+a6*b2,a5*t3+a4*q3+a6*b3,a5*t4+a4*q4+a6*b4
        ss+=tl.sum(es,axis=0)
        svx+=tl.sum(evx,axis=0);svy+=tl.sum(evy,axis=0);svz+=tl.sum(evz,axis=0)
        st0+=tl.sum(et0,axis=0);st1+=tl.sum(et1,axis=0);st2+=tl.sum(et2,axis=0);st3+=tl.sum(et3,axis=0);st4+=tl.sum(et4,axis=0)
    tl.store(Sout+node*C+ch,ss,ch<C)
    tl.store(Vout+node*3*C+ch,svx,ch<C);tl.store(Vout+node*3*C+C+ch,svy,ch<C);tl.store(Vout+node*3*C+2*C+ch,svz,ch<C)
    tl.store(Tout+node*5*C+ch,st0,ch<C);tl.store(Tout+node*5*C+C+ch,st1,ch<C);tl.store(Tout+node*5*C+2*C+ch,st2,ch<C);tl.store(Tout+node*5*C+3*C+ch,st3,ch<C);tl.store(Tout+node*5*C+4*C+ch,st4,ch<C)


@tr.jit
def _owned_backward(P,Cell,Neighbors,Shifts,Counts,OwnedRows,M,V,T,Coeff,GS,GV,GT,GM,GVI,GTI,GP,
                    C:tl.constexpr,K,INTERVALS:tl.constexpr,CUTOFF:tl.constexpr,WIDTH:tl.constexpr,
                    FIRST:tl.constexpr,NEED_NODE:tl.constexpr,BC:tl.constexpr,BN:tl.constexpr,NC:tl.constexpr=6):
    node=tl.program_id(0);tile=tl.program_id(1)
    global_node=tl.load(OwnedRows+node)
    ch=tile*BC+tl.arange(0,BC);channel=ch[None,:]
    own0,own1,own2,own3,own4,own5,own6=_load_message(M,global_node,channel,C,FIRST)
    ovx,ovy,ovz,ot0,ot1,ot2,ot3,ot4=_load_states(V,T,global_node,channel,C)
    ogs=tl.load(GS+global_node*C+channel,channel<C,0)
    ogx,ogy,ogz,og0,og1,og2,og3,og4=_load_states(GV,GT,global_node,channel,C)
    z=tl.full((BC,),0.,tl.float32)
    sm0,sm1,sm2,sm3,sm4,sm5,sm6,svx,svy,svz,st0,st1,st2,st3,st4,spx,spy,spz=z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z
    for start in range(tl.cdiv(tl.load(Counts+node),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,mask,r,ux,uy,uz=_owned_geometry(P,Cell,Neighbors,Shifts,OwnedRows,node,slot,K)
        u,v,w=ux[:,None],uy[:,None],uz[:,None]
        w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6=_filters(Coeff,r[:,None],channel,mask[:,None],C,INTERVALS,CUTOFF,WIDTH,FIRST,NC)
        gs=tl.load(GS+j[:,None]*C+channel,channel<C,0)
        gx,gy,gz,g0,g1,g2,g3,g4=_load_states(GV,GT,j[:,None],channel,C)
        a0,a1,a2,a3,a4,a5,a6,avx,avy,avz,at0,at1,at2,at3,at4,apx,apy,apz=_vjp(
            own0,own1,own2,own3,own4,own5,own6,ovx,ovy,ovz,ot0,ot1,ot2,ot3,ot4,
            gs,gx,gy,gz,g0,g1,g2,g3,g4,-u,-v,-w,r[:,None],
            w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6,FIRST)
        if NEED_NODE:
            sm0+=tl.sum(a0,0);sm1+=tl.sum(a1,0);sm2+=tl.sum(a2,0);sm3+=tl.sum(a3,0);sm4+=tl.sum(a4,0);sm5+=tl.sum(a5,0);sm6+=tl.sum(a6,0)
            svx+=tl.sum(avx,0);svy+=tl.sum(avy,0);svz+=tl.sum(avz,0)
            st0+=tl.sum(at0,0);st1+=tl.sum(at1,0);st2+=tl.sum(at2,0);st3+=tl.sum(at3,0);st4+=tl.sum(at4,0)
        m0,m1,m2,m3,m4,m5,m6=_load_message(M,j[:,None],channel,C,FIRST)
        vx,vy,vz,t0,t1,t2,t3,t4=_load_states(V,T,j[:,None],channel,C)
        _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,bpx,bpy,bpz=_vjp(
            m0,m1,m2,m3,m4,m5,m6,vx,vy,vz,t0,t1,t2,t3,t4,
            ogs,ogx,ogy,ogz,og0,og1,og2,og3,og4,u,v,w,r[:,None],
            w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6,FIRST)
        spx+=tl.sum(apx-bpx,0);spy+=tl.sum(apy-bpy,0);spz+=tl.sum(apz-bpz,0)
    if NEED_NODE:
        O:tl.constexpr=(3 if FIRST else 7)*C
        tl.store(GM+node*O+ch,sm0,ch<C);tl.store(GM+node*O+C+ch,sm1,ch<C)
        if FIRST:
            tl.store(GM+node*O+2*C+ch,sm4,ch<C)
        else:
            tl.store(GM+node*O+2*C+ch,sm2,ch<C);tl.store(GM+node*O+3*C+ch,sm3,ch<C);tl.store(GM+node*O+4*C+ch,sm4,ch<C);tl.store(GM+node*O+5*C+ch,sm5,ch<C);tl.store(GM+node*O+6*C+ch,sm6,ch<C)
        tl.store(GVI+node*3*C+ch,svx,ch<C);tl.store(GVI+node*3*C+C+ch,svy,ch<C);tl.store(GVI+node*3*C+2*C+ch,svz,ch<C)
        tl.store(GTI+node*5*C+ch,st0,ch<C);tl.store(GTI+node*5*C+C+ch,st1,ch<C);tl.store(GTI+node*5*C+2*C+ch,st2,ch<C);tl.store(GTI+node*5*C+3*C+ch,st3,ch<C);tl.store(GTI+node*5*C+4*C+ch,st4,ch<C)
    TILES:tl.constexpr=tr.cdiv(C,BC)
    tl.store(GP+(node*3)*TILES+tile,tl.sum(spx,0))
    tl.store(GP+(node*3+1)*TILES+tile,tl.sum(spy,0))
    tl.store(GP+(node*3+2)*TILES+tile,tl.sum(spz,0))


@tr.jit
def _owned_backward_scalar(P,Cell,Neighbors,Shifts,Counts,OwnedRows,M,Coeff,GS,GV,GT,GM,GP,
                           C:tl.constexpr,K,INTERVALS:tl.constexpr,
                           CUTOFF:tl.constexpr,WIDTH:tl.constexpr,NEED_NODE:tl.constexpr,
                           BC:tl.constexpr,BN:tl.constexpr,N,
                           NC:tl.constexpr=6,BM:tl.constexpr=1):
    raw_node=tl.program_id(0)*BM+tl.arange(0,BM);tile=tl.program_id(1)
    node=tl.minimum(raw_node,N-1)
    global_node=tl.load(OwnedRows+node)
    ch=tile*BC+tl.arange(0,BC);channel=ch[None,None,:]
    own=global_node[:,None,None]
    own0,own1,own2,own3,own4,own5,own6=_load_message(M,own,channel,C,True)
    ogs=tl.load(GS+own*C+channel,channel<C,0)
    ogx,ogy,ogz,og0,og1,og2,og3,og4=_load_states(GV,GT,own,channel,C)
    z=tl.full((BM,BC),0.,tl.float32)
    sm0,sm1,sm4,spx,spy,spz=z,z,z,z,z,z
    for start in range(tl.cdiv(tl.max(tl.load(Counts+node),0),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,mask,r,ux,uy,uz=_owned_geometry(P,Cell,Neighbors,Shifts,OwnedRows,node[:,None],slot[None,:],K)
        u,v,w=ux[:,:,None],uy[:,:,None],uz[:,:,None]
        w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6=_filters(
            Coeff,r[:,:,None],channel,mask[:,:,None],C,INTERVALS,CUTOFF,WIDTH,True,NC)
        gs=tl.load(GS+j[:,:,None]*C+channel,channel<C,0)
        gx,gy,gz,g0,g1,g2,g3,g4=_load_states(GV,GT,j[:,:,None],channel,C)
        m0,m1,m2,m3,m4,m5,m6=_load_message(M,j[:,:,None],channel,C,True)
        bx,by,bz=m1*ogx-own1*gx,m1*ogy-own1*gy,m1*ogz-own1*gz
        a0,a1,a2,a3,a4=m4*og0+own4*g0,m4*og1+own4*g1,m4*og2+own4*g2,m4*og3+own4*g3,m4*og4+own4*g4
        q0,q1,q2,q3,q4=_q(u,v,w)
        vector_projection=bx*u+by*v+bz*w
        tensor_projection=a0*q0+a1*q1+a2*q2+a3*q3+a4*q4
        ax,ay,az=_matvec(a0,a1,a2,a3,a4,u,v,w)
        radial=d0*(own0*gs+m0*ogs)+d1*vector_projection+d4*tensor_projection
        angular_projection=w1*vector_projection+2*w4*tensor_projection
        spx-=tl.sum(radial*u+(w1*bx+2*w4*ax-angular_projection*u)/r[:,:,None],1)
        spy-=tl.sum(radial*v+(w1*by+2*w4*ay-angular_projection*v)/r[:,:,None],1)
        spz-=tl.sum(radial*w+(w1*bz+2*w4*az-angular_projection*w)/r[:,:,None],1)
        if NEED_NODE:
            sm0+=tl.sum(w0*gs,1)
            sm1-=tl.sum(w1*(gx*u+gy*v+gz*w),1)
            sm4+=tl.sum(w4*(g0*q0+g1*q1+g2*q2+g3*q3+g4*q4),1)
    if NEED_NODE:
        offset=raw_node[:,None]*3*C+ch[None,:]
        mask=(raw_node[:,None]<N)&(ch[None,:]<C)
        tl.store(GM+offset,sm0,mask)
        tl.store(GM+offset+C,sm1,mask)
        tl.store(GM+offset+2*C,sm4,mask)
    TILES:tl.constexpr=tr.cdiv(C,BC)
    tl.store(GP+raw_node*3*TILES+tile,tl.sum(spx,1),raw_node<N)
    tl.store(GP+(raw_node*3+1)*TILES+tile,tl.sum(spy,1),raw_node<N)
    tl.store(GP+(raw_node*3+2)*TILES+tile,tl.sum(spz,1),raw_node<N)


class OwnedBaseGPUError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code):
    if not condition:
        raise OwnedBaseGPUError(code)


def _eager(device):
    _require(device.type == "cuda" and device.index == torch.cuda.current_device(), "CURRENT_CUDA_DEVICE_REQUIRED")
    _require(not torch.cuda.is_current_stream_capturing(), "EAGER_PREPARATION_REQUIRED")


@dataclass(frozen=True)
class BaseInputs:
    positions: torch.Tensor
    message: torch.Tensor
    vector: torch.Tensor
    tensor: torch.Tensor

    def tensors(self):
        return self.positions, self.message, self.vector, self.tensor


@dataclass(frozen=True)
class OwnedImageGraph:
    metadata: OwnershipMetadata
    rank: int
    local_rows: torch.Tensor
    neighbors: torch.Tensor
    shifts: torch.Tensor
    counts: torch.Tensor
    cell: torch.Tensor
    capacity: int

    @property
    def n_local(self):
        return self.metadata.counts[self.rank]

    @property
    def n_global(self):
        return self.metadata.n_global


def prepare_owned_graph(edge_index, unit_shifts, cell, metadata, rank, *, device, reserve_factor=1.2):
    """Eager full symmetric CPU image-graph guard, then owned device rows.

    This constructor also works for a simulated partition without a process
    group. It does not check neighbor completeness against the geometry and
    never updates a graph in place; rebuild outside capture when graph
    membership or capacity changes.
    """
    device = torch.device(device)
    _eager(device)
    _require(isinstance(metadata, OwnershipMetadata) and metadata.n_global > 0 and
             type(rank) is int and 0 <= rank < metadata.world_size, "OWNERSHIP_REQUIRED")
    _require(edge_index.device.type == unit_shifts.device.type == cell.device.type == "cpu" and
             edge_index.dtype == unit_shifts.dtype == torch.int64 and
             edge_index.ndim == 2 and edge_index.shape[0] == 2 and
             unit_shifts.shape == (edge_index.shape[1], 3) and cell.shape == (3, 3), "CPU_IMAGE_GRAPH_REQUIRED")
    _require(not cell.requires_grad and bool(torch.isfinite(cell).all()), "FIXED_FINITE_CELL_REQUIRED")
    _require(math.isfinite(reserve_factor) and reserve_factor >= 1.0, "RESERVE_FACTOR_INVALID")
    rows = torch.cat((edge_index.T, unit_shifts), dim=1).tolist()
    keys = {tuple(row) for row in rows}
    _require(len(keys) == len(rows), "DUPLICATE_IMAGE_EDGE")
    _require(all(0 <= i < metadata.n_global and 0 <= j < metadata.n_global for i,j,*_ in keys), "EDGE_OUTSIDE_ROWS")
    _require(all(i != j or (a,b,c) != (0,0,0) for i,j,a,b,c in keys), "ZERO_SHIFT_SELF_EDGE")
    _require(all((j,i,-a,-b,-c) in keys for i,j,a,b,c in keys), "RECIPROCAL_IMAGE_REQUIRED")
    _require(all(-(2**31) <= shift < 2**31 for row in rows for shift in row[2:]), "SHIFT_INT32_OVERFLOW")
    incoming = [[] for _ in range(metadata.n_global)]
    for source,destination,a,b,c in rows:
        incoming[destination].append((source,a,b,c))
    # Capacity is global so each partition uses the same native neighborhood
    # reserve policy. The input edge order within each row is preserved.
    required = max(map(len, incoming))
    capacity = max(8, tr.cdiv(int(required * reserve_factor) + 1, 8) * 8)
    local_ids = metadata.indices_by_rank[rank]
    neighbors = torch.full((len(local_ids), capacity), -1, dtype=torch.int64)
    shifts = torch.zeros((len(local_ids), capacity, 3), dtype=torch.int32)
    counts = torch.zeros(len(local_ids), dtype=torch.int32)
    for local, global_row in enumerate(local_ids):
        count = len(incoming[global_row])
        counts[local] = count
        if count:
            row = torch.tensor(incoming[global_row], dtype=torch.int64)
            neighbors[local,:count] = row[:,0]
            shifts[local,:count] = row[:,1:].to(torch.int32)
    return OwnedImageGraph(metadata, rank, torch.tensor(local_ids,dtype=torch.int64,device=device),
                           neighbors.to(device), shifts.to(device), counts.to(device),
                           cell.to(device=device,dtype=torch.float32).contiguous(), capacity)


@dataclass(frozen=True)
class PreparedBaseOperator:
    graph: OwnedImageGraph
    coefficients: torch.Tensor
    cutoff: float
    switch_width: float
    first: bool
    settings: tuple[int, int, int, bool]
    adjoint_specs: tuple | None


def _matching_spec(spec, graph, tail, contract):
    _require(spec.contract == contract and spec.dtype == torch.float32 and spec.tail_shape == tail and
             spec.runtime.device == graph.cell.device and spec.runtime.rank == graph.rank and
             spec.runtime.metadata == graph.metadata, "COLLECTIVE_SPEC_MISMATCH")


def prepare_base_operator(graph, coefficients, *, cutoff, switch_width, first,
                          settings=(32,8,4,False), adjoint_specs=None):
    _eager(graph.cell.device)
    _require(type(first) is bool and len(settings) == 4, "STATIC_SETTINGS_REQUIRED")
    bc,bn,warps,fma = settings
    _require(all(type(v) is int and v > 0 and v & (v-1) == 0 for v in (bc,bn,warps)) and
             warps in (1,2,4,8,16,32) and type(fma) is bool, "KERNEL_SETTINGS_INVALID")
    _require(all(math.isfinite(v) and v > 0 for v in (cutoff,switch_width)), "RADIAL_BOUNDS_INVALID")
    _require(coefficients.device == graph.cell.device and coefficients.dtype == torch.float32 and
             coefficients.ndim == 3 and coefficients.shape[0] > 0 and coefficients.shape[1] in (4,6) and
             coefficients.shape[2] == (3 if first else 7)*84 and coefficients.is_contiguous() and
             not coefficients.requires_grad and coefficients.grad_fn is None and
             bool(torch.isfinite(coefficients).all()), "FIXED_C84_COEFFICIENT_TABLE_REQUIRED")
    if adjoint_specs is not None:
        _require(len(adjoint_specs) == 3, "THREE_ADJOINT_SPECS_REQUIRED")
        for spec,tail in zip(adjoint_specs,((84,),(3,84),(5,84))):
            _matching_spec(spec,graph,tail,DISJOINT_OUTPUT)
    return PreparedBaseOperator(graph,coefficients,float(cutoff),float(switch_width),first,
                                 tuple(settings),None if adjoint_specs is None else tuple(adjoint_specs))


@dataclass(frozen=True)
class SourceSnapshot:
    owned: BaseInputs
    cache: BaseInputs
    operator: PreparedBaseOperator
    origin: str


def _check_inputs(inputs, operator, *, global_rows=False, detached=False):
    graph = operator.graph
    count = graph.n_global if global_rows else graph.n_local
    tails = ((3,),((3 if operator.first else 7)*84,),(3,84),(5,84))
    for value,tail in zip(inputs.tensors(),tails):
        _require(value.device == graph.cell.device and value.dtype == torch.float32 and
                 value.shape == (count,*tail), "SOURCE_METADATA_MISMATCH")
        if global_rows:
            _require(value.is_contiguous(), "CONTIGUOUS_GLOBAL_CACHE_REQUIRED")
        if detached:
            _require(not value.requires_grad and value.grad_fn is None, "DETACHED_GLOBAL_CACHE_REQUIRED")


@torch.no_grad()
def verify_prefilled_snapshot(owned, cache, operator):
    """Eager origin check for prepared/full-system/simulated-rank source data."""
    _eager(operator.graph.cell.device)
    _check_inputs(owned,operator)
    _check_inputs(cache,operator,global_rows=True,detached=True)
    for local,full in zip(owned.tensors(),cache.tensors()):
        _require(bool(torch.isfinite(full).all()) and torch.equal(
            local.detach(),full.index_select(0,operator.graph.local_rows)), "SOURCE_SNAPSHOT_MISMATCH")
    return SourceSnapshot(owned,cache,operator,"eager_verified_prefilled")


def gather_source_snapshot(owned, operator, source_specs):
    """Tensor path: detached gather origin by construction, without packing."""
    _check_inputs(owned,operator)
    tails = ((3,),((3 if operator.first else 7)*84,),(3,84),(5,84))
    _require(len(source_specs) == 4, "FOUR_SOURCE_SPECS_REQUIRED")
    for spec,tail in zip(source_specs,tails):
        _matching_spec(spec,operator.graph,tail,DETACHED_SOURCE)
    cache = BaseInputs(*(gather_detached_sources(value,spec) for value,spec in zip(owned.tensors(),source_specs)))
    return SourceSnapshot(owned,cache,operator,"detached_collectives")


def _check_snapshot(snapshot):
    _require(isinstance(snapshot,SourceSnapshot) and snapshot.origin in
             ("eager_verified_prefilled","detached_collectives"), "SOURCE_ORIGIN_REQUIRED")
    _check_inputs(snapshot.owned,snapshot.operator)
    _check_inputs(snapshot.cache,snapshot.operator,global_rows=True,detached=True)


@torch.no_grad()
def owned_base_forward(snapshot):
    """Forward for a prepared snapshot without autograd; also used for simulated ranks."""
    _check_snapshot(snapshot)
    op,cache = snapshot.operator,snapshot.cache
    graph = op.graph
    n,c = graph.n_local,84
    outputs = tuple(cache.positions.new_empty((n,*tail)) for tail in ((c,),(3,c),(5,c)))
    if n:
        bc,bn,warps,fma = op.settings
        _owned_forward[(n,tr.cdiv(c,bc))](cache.positions,graph.cell,graph.neighbors,graph.shifts,graph.counts,
            graph.local_rows,cache.message,cache.vector,cache.tensor,op.coefficients,*outputs,
            c,graph.capacity,op.coefficients.shape[0],op.cutoff,op.switch_width,op.first,bc,bn,
            NC=op.coefficients.shape[1],num_warps=warps,enable_fp_fusion=fma)
    return outputs


@torch.no_grad()
def owned_base_vjp(snapshot, global_adjoints, *, needs=(True,True,True,True)):
    """Complete owned VJP from READY global cotangents; no communication.

    This explicit primitive is the simulated-rank path. The caller supplies the
    actual global cotangents; it is not the derivative of an arbitrary local
    loss. The normal autograd wrapper obtains these cotangents by all-gather.
    """
    _check_snapshot(snapshot)
    _require(len(needs) == 4 and all(type(v) is bool for v in needs), "FOUR_GRADIENT_FLAGS_REQUIRED")
    op,cache = snapshot.operator,snapshot.cache
    graph = op.graph
    _require(len(global_adjoints) == 3, "THREE_GLOBAL_ADJOINTS_REQUIRED")
    for value,tail in zip(global_adjoints,((84,),(3,84),(5,84))):
        _require(value.device == graph.cell.device and value.dtype == torch.float32 and
                 value.shape == (graph.n_global,*tail) and not value.requires_grad and value.grad_fn is None,
                 "DETACHED_GLOBAL_ADJOINT_REQUIRED")
    gs,gv,gt = (value.contiguous() for value in global_adjoints)
    n,c = graph.n_local,84
    bc,bn,warps,fma = op.settings
    gp = cache.positions.new_empty((n,3,tr.cdiv(c,bc)))
    need_node = needs[1] if op.first else any(needs[1:])
    if op.first:
        gm = snapshot.owned.message.new_empty(snapshot.owned.message.shape) if need_node else cache.positions
        gvi,gti = None,None
    else:
        gm,gvi,gti = (tuple(value.new_empty(value.shape) for value in snapshot.owned.tensors()[1:])
                      if need_node else (cache.positions,cache.positions,cache.positions))
    if n:
        if op.first:
            _owned_backward_scalar[(n,tr.cdiv(c,bc))](cache.positions,graph.cell,graph.neighbors,graph.shifts,
                graph.counts,graph.local_rows,cache.message,op.coefficients,gs,gv,gt,gm,gp,
                c,graph.capacity,op.coefficients.shape[0],op.cutoff,op.switch_width,need_node,bc,bn,n,
                NC=op.coefficients.shape[1],BM=1,num_warps=warps,enable_fp_fusion=fma)
        else:
            _owned_backward[(n,tr.cdiv(c,bc))](cache.positions,graph.cell,graph.neighbors,graph.shifts,graph.counts,
                graph.local_rows,cache.message,cache.vector,cache.tensor,op.coefficients,gs,gv,gt,gm,gvi,gti,gp,
                c,graph.capacity,op.coefficients.shape[0],op.cutoff,op.switch_width,False,need_node,bc,bn,
                NC=op.coefficients.shape[1],num_warps=warps,enable_fp_fusion=fma)
    if op.first:
        gvi = torch.zeros_like(snapshot.owned.vector) if needs[2] else None
        gti = torch.zeros_like(snapshot.owned.tensor) if needs[3] else None
    return (gp.sum(-1) if needs[0] else None,gm if needs[1] else None,
            gvi if needs[2] else None,gti if needs[3] else None)


class _OwnedBaseMessages(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, message, vector, tensor, snapshot):
        ctx.set_materialize_grads(False)
        graph,op = snapshot.operator.graph,snapshot.operator
        ctx.operator,ctx.origin = op,snapshot.origin
        ctx.save_for_backward(positions,message,vector,tensor,*snapshot.cache.tensors(),
                              graph.local_rows,graph.neighbors,graph.shifts,graph.counts,graph.cell,op.coefficients)
        return owned_base_forward(snapshot)

    @staticmethod
    def backward(ctx, gs, gv, gt):
        _require(not torch.is_grad_enabled(), "INFERENCE_FIRST_DERIVATIVES_ONLY")
        saved = ctx.saved_tensors  # Validate versions before using immutable saved containers.
        snapshot = SourceSnapshot(BaseInputs(*saved[:4]),BaseInputs(*saved[4:8]),ctx.operator,ctx.origin)
        adjoints = tuple(gather_disjoint_output_adjoints(value,spec) for value,spec in
                         zip((gs,gv,gt),snapshot.operator.adjoint_specs))
        gradients = owned_base_vjp(snapshot,adjoints,needs=tuple(ctx.needs_input_grad[:4]))
        return (*gradients,None)


def owned_base_messages(snapshot):
    """Normal autograd entry: direct owned inputs, detached cache, no SUM VJP."""
    _check_snapshot(snapshot)
    _require(snapshot.operator.adjoint_specs is not None, "PREPARED_ADJOINT_EXCHANGE_REQUIRED")
    _require(any(value.requires_grad for value in snapshot.owned.tensors()), "OWNED_GRADIENT_DEPENDENCY_REQUIRED")
    return _OwnedBaseMessages.apply(*snapshot.owned.tensors(),snapshot)


__all__ = ["BaseInputs","OwnedImageGraph","PreparedBaseOperator","SourceSnapshot","OwnedBaseGPUError",
           "prepare_owned_graph","prepare_base_operator","verify_prefilled_snapshot","gather_source_snapshot",
           "owned_base_forward","owned_base_vjp","owned_base_messages"]
