"""Native symmetric pair energy with explicit 17-element checkpoint stride."""
import torch
import triton as tr
import triton.language as tl
from hyff.kernels.fused_messages import _geometry, _poly

@tr.jit
def _pair_value(Coeff, r, pair, mask, INTERVALS:tl.constexpr,
                CUTOFF:tl.constexpr, WIDTH:tl.constexpr, NC:tl.constexpr=6, NE:tl.constexpr=17):
    coordinate=tl.minimum(tl.maximum(r*(INTERVALS/CUTOFF),0.),INTERVALS)
    interval=tl.minimum(coordinate.to(tl.int32),INTERVALS-1)
    value,derivative=_poly(Coeff,interval,coordinate-interval,pair,NE*NE,NE*NE,INTERVALS/CUTOFF,0,NC)
    x=tl.minimum(tl.maximum((CUTOFF-r)/WIDTH,0.),1.)
    envelope=x*x*x*(10.-15.*x+6.*x*x)*mask
    slope=-30.*x*x*(1.-x)*(1.-x)/WIDTH*mask
    return value*envelope,derivative*envelope+value*slope


@tr.jit
def _pair_forward(P,Cell,Neighbors,Shifts,Counts,Elements,Coeff,Energy,
                  K:tl.constexpr,INTERVALS:tl.constexpr,CUTOFF:tl.constexpr,WIDTH:tl.constexpr,BN:tl.constexpr,NC:tl.constexpr=6, NE:tl.constexpr=17):
    node=tl.program_id(0)
    own=tl.load(Elements+node)
    total=tl.full((BN,),0.,tl.float32)
    for start in range(tl.cdiv(tl.load(Counts+node),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,valid,r,_,_,_=_geometry(P,Cell,Neighbors,Shifts,node,slot,K)
        other=tl.load(Elements+j)
        pair=tl.minimum(own,other)*NE+tl.maximum(own,other)
        value,_=_pair_value(Coeff,r,pair,valid,INTERVALS,CUTOFF,WIDTH,NC,NE)
        total+=.5*value
    tl.store(Energy+node,tl.sum(total,0))


@tr.jit
def _pair_backward(P,Cell,Neighbors,Shifts,Counts,Elements,Coeff,Gradient,GP,
                   K:tl.constexpr,INTERVALS:tl.constexpr,CUTOFF:tl.constexpr,WIDTH:tl.constexpr,BN:tl.constexpr,NC:tl.constexpr=6, NE:tl.constexpr=17):
    node=tl.program_id(0)
    own=tl.load(Elements+node)
    g=tl.load(Gradient+node)
    gx=tl.full((BN,),0.,tl.float32)
    gy,gz=gx,gx
    for start in range(tl.cdiv(tl.load(Counts+node),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,valid,r,x,y,z=_geometry(P,Cell,Neighbors,Shifts,node,slot,K)
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


class _PairEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx,position,cell,neighbors,shifts,counts,elements,coefficients,cutoff,width,fma):
        n=len(position)
        result=position.new_empty(n)
        _pair_forward[(n,)](position,cell,neighbors,shifts,counts,elements,coefficients,result,
            neighbors.shape[1],coefficients.shape[0],cutoff,width,32,NC=coefficients.shape[1],num_warps=4,enable_fp_fusion=fma)
        ctx.save_for_backward(position,cell,neighbors,shifts,counts,elements,coefficients)
        ctx.settings=cutoff,width,fma
        return result

    @staticmethod
    def backward(ctx,gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Fused L2 pairs are inference-only; use the plain energy for force training')
        position,cell,neighbors,shifts,counts,elements,coefficients=ctx.saved_tensors
        cutoff,width,fma=ctx.settings
        gp=torch.empty_like(position)
        _pair_backward[(len(position),)](position,cell,neighbors,shifts,counts,elements,coefficients,
            gradient.contiguous(),gp,neighbors.shape[1],coefficients.shape[0],cutoff,width,32,
            NC=coefficients.shape[1],num_warps=4,enable_fp_fusion=fma)
        return gp,None,None,None,None,None,None,None,None,None


def pair_energy(position,plan,elements,table):
    return _PairEnergy.apply(position,plan.cell,plan.neighbors,plan.shifts,plan.counts,elements,
        table.coefficients,plan.model.cutoff,plan.model.config.switch_width,plan.model.fused_message_settings[3])
