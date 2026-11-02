"""Atom-owned full-tensor reverse reductions with alternative GPU layouts.

This module uses the same radial tables and complete l<=2 contractions as
hyff.kernels.fused_messages, with no approximated forces or geometry reuse
between steps. The fused_messages backward remains the general path. Every
coordinate's two directed contributions are kept. The default dispatch uses
this module only for the tuned C = 16 quintic configuration.
"""
import triton as tr
import triton.language as tl

from hyff.kernels.fused_messages import (
    _filters, _geometry, _load_message, _load_states, _matvec, _q, _stv, _vjp,
)


@tr.jit
def _tensor_vjp_parts(m0,m1,m2,m3,m4,m5,m6,vx,vy,vz,t0,t1,t2,t3,t4,
                      gs,gx,gy,gz,g0,g1,g2,g3,g4,u,v,w,
                      w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6):
    """Node adjoints and unprojected radial/angular coordinate adjoints."""
    q0,q1,q2,q3,q4=_q(u,v,w)
    f0=gs
    f1=gx*u+gy*v+gz*w
    f4=g0*q0+g1*q1+g2*q2+g3*q3+g4*q4
    gux,guy,guz=_matvec(g0,g1,g2,g3,g4,u,v,w)
    tux,tuy,tuz=_matvec(t0,t1,t2,t3,t4,u,v,w)
    tgx,tgy,tgz=_matvec(t0,t1,t2,t3,t4,gx,gy,gz)
    gvx,gvy,gvz=_matvec(g0,g1,g2,g3,g4,vx,vy,vz)
    f2=gx*vx+gy*vy+gz*vz
    f3=gs*(tux*u+tuy*v+tuz*w)+gx*tux+gy*tuy+gz*tuz
    f5=g0*t0+g1*t1+g2*t2+g3*t3+g4*t4
    f6=gs*(vx*u+vy*v+vz*w)+vx*gux+vy*guy+vz*guz
    a1,a2,a3,a4,a5,a6=m1*w1,m2*w2,m3*w3,m4*w4,m5*w5,m6*w6
    dvx=a2*gx+a6*(gs*u+gux)
    dvy=a2*gy+a6*(gs*v+guy)
    dvz=a2*gz+a6*(gs*w+guz)
    b0,b1,b2,b3,b4=_stv(gx,gy,gz,u,v,w)
    dt0,dt1,dt2,dt3,dt4=a5*g0+a3*(gs*q0+b0),a5*g1+a3*(gs*q1+b1),a5*g2+a3*(gs*q2+b2),a5*g3+a3*(gs*q3+b3),a5*g4+a3*(gs*q4+b4)
    dx=a1*gx+2*a4*gux+a3*(2*gs*tux+tgx)+a6*(gs*vx+gvx)
    dy=a1*gy+2*a4*guy+a3*(2*gs*tuy+tgy)+a6*(gs*vy+gvy)
    dz=a1*gz+2*a4*guz+a3*(2*gs*tuz+tgz)+a6*(gs*vz+gvz)
    dr=m0*d0*f0+m1*d1*f1+m4*d4*f4+m2*d2*f2+m3*d3*f3+m5*d5*f5+m6*d6*f6
    return w0*f0,w1*f1,w2*f2,w3*f3,w4*f4,w5*f5,w6*f6,dvx,dvy,dvz,dt0,dt1,dt2,dt3,dt4,dr,dx,dy,dz


@tr.jit
def _backward_tensor(P,Cell,Neighbors,Shifts,Counts,M,V,T,Coeff,GS,GV,GT,GM,GVI,GTI,GP,
                     C:tl.constexpr,K:tl.constexpr,INTERVALS:tl.constexpr,
                     CUTOFF:tl.constexpr,WIDTH:tl.constexpr,NEED_NODE:tl.constexpr,
                     BC:tl.constexpr,BN:tl.constexpr,N:tl.constexpr,
                     NC:tl.constexpr=6,BM:tl.constexpr=1,JOINT:tl.constexpr=True):
    raw_node=tl.program_id(0)*BM+tl.arange(0,BM)
    node=tl.minimum(raw_node,N-1)
    tile=tl.program_id(1)
    ch=tile*BC+tl.arange(0,BC)
    channel=ch[None,None,:]
    own=node[:,None,None]
    own0,own1,own2,own3,own4,own5,own6=_load_message(M,own,channel,C,False)
    ovx,ovy,ovz,ot0,ot1,ot2,ot3,ot4=_load_states(V,T,own,channel,C)
    ogs=tl.load(GS+own*C+channel,channel<C,0)
    ogx,ogy,ogz,og0,og1,og2,og3,og4=_load_states(GV,GT,own,channel,C)
    z=tl.full((BM,BC),0.,tl.float32)
    sm0,sm1,sm2,sm3,sm4,sm5,sm6,svx,svy,svz,st0,st1,st2,st3,st4,spx,spy,spz=z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z
    for start in range(tl.cdiv(tl.max(tl.load(Counts+node),0),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,mask,r,ux,uy,uz=_geometry(P,Cell,Neighbors,Shifts,node[:,None],slot[None,:],K)
        u,v,w=ux[:,:,None],uy[:,:,None],uz[:,:,None]
        w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6=_filters(Coeff,r[:,:,None],channel,mask[:,:,None],C,INTERVALS,CUTOFF,WIDTH,False,NC)
        gs=tl.load(GS+j[:,:,None]*C+channel,channel<C,0)
        gx,gy,gz,g0,g1,g2,g3,g4=_load_states(GV,GT,j[:,:,None],channel,C)
        if JOINT:
            a0,a1,a2,a3,a4,a5,a6,avx,avy,avz,at0,at1,at2,at3,at4,ar,ax,ay,az=_tensor_vjp_parts(
                own0,own1,own2,own3,own4,own5,own6,ovx,ovy,ovz,ot0,ot1,ot2,ot3,ot4,
                gs,gx,gy,gz,g0,g1,g2,g3,g4,-u,-v,-w,
                w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6)
        else:
            a0,a1,a2,a3,a4,a5,a6,avx,avy,avz,at0,at1,at2,at3,at4,apx,apy,apz=_vjp(
                own0,own1,own2,own3,own4,own5,own6,ovx,ovy,ovz,ot0,ot1,ot2,ot3,ot4,
                gs,gx,gy,gz,g0,g1,g2,g3,g4,-u,-v,-w,r[:,:,None],
                w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6,False)
        if NEED_NODE:
            sm0+=tl.sum(a0,1);sm1+=tl.sum(a1,1);sm2+=tl.sum(a2,1);sm3+=tl.sum(a3,1);sm4+=tl.sum(a4,1);sm5+=tl.sum(a5,1);sm6+=tl.sum(a6,1)
            svx+=tl.sum(avx,1);svy+=tl.sum(avy,1);svz+=tl.sum(avz,1)
            st0+=tl.sum(at0,1);st1+=tl.sum(at1,1);st2+=tl.sum(at2,1);st3+=tl.sum(at3,1);st4+=tl.sum(at4,1)
        m0,m1,m2,m3,m4,m5,m6=_load_message(M,j[:,:,None],channel,C,False)
        vx,vy,vz,t0,t1,t2,t3,t4=_load_states(V,T,j[:,:,None],channel,C)
        if JOINT:
            unused0,unused1,unused2,unused3,unused4,unused5,unused6,unusedvx,unusedvy,unusedvz,unusedt0,unusedt1,unusedt2,unusedt3,unusedt4,br,bx,by,bz=_tensor_vjp_parts(
                m0,m1,m2,m3,m4,m5,m6,vx,vy,vz,t0,t1,t2,t3,t4,
                ogs,ogx,ogy,ogz,og0,og1,og2,og3,og4,u,v,w,
                w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6)
            # The first contribution differentiates -u; its radius derivative
            # has the opposite sign. Both angular tangents use the same plane.
            dx,dy,dz=bx-ax,by-ay,bz-az
            projection=dx*u+dy*v+dz*w
            radial=ar+br
            spx-=tl.sum(radial*u+(dx-projection*u)/r[:,:,None],1)
            spy-=tl.sum(radial*v+(dy-projection*v)/r[:,:,None],1)
            spz-=tl.sum(radial*w+(dz-projection*w)/r[:,:,None],1)
        else:
            unused0,unused1,unused2,unused3,unused4,unused5,unused6,unusedvx,unusedvy,unusedvz,unusedt0,unusedt1,unusedt2,unusedt3,unusedt4,bpx,bpy,bpz=_vjp(
                m0,m1,m2,m3,m4,m5,m6,vx,vy,vz,t0,t1,t2,t3,t4,
                ogs,ogx,ogy,ogz,og0,og1,og2,og3,og4,u,v,w,r[:,:,None],
                w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6,False)
            spx+=tl.sum(apx-bpx,1);spy+=tl.sum(apy-bpy,1);spz+=tl.sum(apz-bpz,1)
    output_node=raw_node[:,None]
    output_ch=ch[None,:]
    output_mask=(output_node<N)&(output_ch<C)
    if NEED_NODE:
        base=output_node*7*C+output_ch
        tl.store(GM+base,sm0,output_mask);tl.store(GM+base+C,sm1,output_mask)
        tl.store(GM+base+2*C,sm2,output_mask);tl.store(GM+base+3*C,sm3,output_mask)
        tl.store(GM+base+4*C,sm4,output_mask);tl.store(GM+base+5*C,sm5,output_mask);tl.store(GM+base+6*C,sm6,output_mask)
        base=output_node*3*C+output_ch
        tl.store(GVI+base,svx,output_mask);tl.store(GVI+base+C,svy,output_mask);tl.store(GVI+base+2*C,svz,output_mask)
        base=output_node*5*C+output_ch
        tl.store(GTI+base,st0,output_mask);tl.store(GTI+base+C,st1,output_mask)
        tl.store(GTI+base+2*C,st2,output_mask);tl.store(GTI+base+3*C,st3,output_mask);tl.store(GTI+base+4*C,st4,output_mask)
    TILES:tl.constexpr=tr.cdiv(C,BC)
    tl.store(GP+raw_node*3*TILES+tile,tl.sum(spx,1),raw_node<N)
    tl.store(GP+(raw_node*3+1)*TILES+tile,tl.sum(spy,1),raw_node<N)
    tl.store(GP+(raw_node*3+2)*TILES+tile,tl.sum(spz,1),raw_node<N)
