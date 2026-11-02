"""Inference-only L2 neighbor reductions without materializing edge tensors.

The forward reduces each atom's neighbors directly. The exact reverse uses
the symmetric radius graph to reduce node adjoints and both roles of each
coordinate locally, avoiding channel-sized atomic scatter arrays. Radial
functions are C2 polynomial tables of the model's radial filters, not new force heads.
"""
import math

import torch
import triton as tr
import triton.language as tl

KERNEL_RESOURCES = {}


def configure_neighbor_reserve(model, factor=1.2):
    """Reserve slots, not interactions; overflow continues to force recapture."""
    factor = float(factor)
    if not math.isfinite(factor) or factor < 1.:
        raise ValueError('Neighbor reserve factor must be finite and at least one')
    model._l2_neighbor_reserve_factor = factor
    return model


def _record(kernel, key):
    # Resource records are plain values so that they can be serialized to JSON.
    if isinstance(key, tuple):
        key = ':'.join(map(str, key))
    if key not in KERNEL_RESOURCES:
        KERNEL_RESOURCES[key] = dict(registers=kernel.n_regs, spills=kernel.n_spills,
                                     shared_bytes=kernel.metadata.shared)


@tr.jit
def _matvec(a, b, c, d, e, x, y, z):
    xx = .7071067811865475 * a + .4082482904638631 * e
    yy = -.7071067811865475 * a + .4082482904638631 * e
    zz = -.8164965809277261 * e
    xy = .7071067811865475 * b
    xz = .7071067811865475 * c
    yz = .7071067811865475 * d
    return xx*x + xy*y + xz*z, xy*x + yy*y + yz*z, xz*x + yz*y + zz*z


@tr.jit
def _stv(x, y, z, u, v, w):
    return (.7071067811865475*(x*u-y*v), .7071067811865475*(x*v+y*u),
            .7071067811865475*(x*w+z*u), .7071067811865475*(y*w+z*v),
            .4082482904638631*(x*u+y*v-2*z*w))


@tr.jit
def _q(u, v, w):
    return (.7071067811865475*(u*u-v*v), 1.4142135623730951*u*v,
            1.4142135623730951*u*w, 1.4142135623730951*v*w,
            .4082482904638631*(u*u+v*v-2*w*w))


@tr.jit
def _poly(Coeff, interval, x, channel, C: tl.constexpr, O: tl.constexpr, HINV: tl.constexpr, PATH: tl.constexpr, NC: tl.constexpr = 6):
    base = interval * NC * O + PATH * C + channel
    a3 = tl.load(Coeff + base + 3*O, channel < C, 0)
    a2 = tl.load(Coeff + base + 2*O, channel < C, 0)
    a1 = tl.load(Coeff + base + O, channel < C, 0)
    a0 = tl.load(Coeff + base, channel < C, 0)
    if NC == 6:
        a5 = tl.load(Coeff + base + 5*O, channel < C, 0)
        a4 = tl.load(Coeff + base + 4*O, channel < C, 0)
        value = (((((a5*x+a4)*x+a3)*x+a2)*x+a1)*x+a0)
        derivative = ((((5*a5*x+4*a4)*x+3*a3)*x+2*a2)*x+a1)*HINV
    else:
        tl.static_assert(NC == 4)
        value = ((a3*x+a2)*x+a1)*x+a0
        derivative = ((3*a3*x+2*a2)*x+a1)*HINV
    return value, derivative


@tr.jit
def _filters(Coeff, r, ch, mask, C: tl.constexpr, INTERVALS: tl.constexpr,
             CUTOFF: tl.constexpr, WIDTH: tl.constexpr, FIRST: tl.constexpr, NC: tl.constexpr = 6):
    O: tl.constexpr = (3 if FIRST else 7)*C
    HINV: tl.constexpr = INTERVALS/CUTOFF
    coordinate = tl.minimum(tl.maximum(r*HINV, 0.), INTERVALS)
    interval = tl.minimum(coordinate.to(tl.int32), INTERVALS-1)
    x = coordinate - interval
    y = tl.minimum(tl.maximum((CUTOFF-r)/WIDTH, 0.), 1.)
    envelope = y*y*y*(10.-15.*y+6.*y*y)*mask*.25
    derivative = -30.*y*y*(1.-y)*(1.-y)/WIDTH*mask*.25
    p0, d0 = _poly(Coeff, interval, x, ch, C, O, HINV, 0, NC)
    p1, d1 = _poly(Coeff, interval, x, ch, C, O, HINV, 1, NC)
    if FIRST:
        p4, d4 = _poly(Coeff, interval, x, ch, C, O, HINV, 2, NC)
        p2 = tl.full(r.shape, 0., tl.float32)
        p3, p5, p6, d2, d3, d5, d6 = p2, p2, p2, p2, p2, p2, p2
    else:
        p2, d2 = _poly(Coeff, interval, x, ch, C, O, HINV, 2, NC)
        p3, d3 = _poly(Coeff, interval, x, ch, C, O, HINV, 3, NC)
        p4, d4 = _poly(Coeff, interval, x, ch, C, O, HINV, 4, NC)
        p5, d5 = _poly(Coeff, interval, x, ch, C, O, HINV, 5, NC)
        p6, d6 = _poly(Coeff, interval, x, ch, C, O, HINV, 6, NC)
    return (p0*envelope, p1*envelope, p2*envelope, p3*envelope, p4*envelope, p5*envelope, p6*envelope,
            d0*envelope+p0*derivative, d1*envelope+p1*derivative, d2*envelope+p2*derivative,
            d3*envelope+p3*derivative, d4*envelope+p4*derivative, d5*envelope+p5*derivative, d6*envelope+p6*derivative)


@tr.jit
def _load_message(M, node, ch, C: tl.constexpr, FIRST: tl.constexpr):
    O: tl.constexpr = (3 if FIRST else 7)*C
    m0 = tl.load(M+node*O+ch, ch<C, 0)
    m1 = tl.load(M+node*O+C+ch, ch<C, 0)
    if FIRST:
        m4 = tl.load(M+node*O+2*C+ch, ch<C, 0)
        m2 = tl.full(m0.shape, 0., tl.float32)
        m3, m5, m6 = m2, m2, m2
    else:
        m2 = tl.load(M+node*O+2*C+ch, ch<C, 0)
        m3 = tl.load(M+node*O+3*C+ch, ch<C, 0)
        m4 = tl.load(M+node*O+4*C+ch, ch<C, 0)
        m5 = tl.load(M+node*O+5*C+ch, ch<C, 0)
        m6 = tl.load(M+node*O+6*C+ch, ch<C, 0)
    return m0,m1,m2,m3,m4,m5,m6


@tr.jit
def _load_states(V,T,node,ch,C:tl.constexpr):
    vx=tl.load(V+node*3*C+ch,ch<C,0)
    vy=tl.load(V+node*3*C+C+ch,ch<C,0)
    vz=tl.load(V+node*3*C+2*C+ch,ch<C,0)
    t0=tl.load(T+node*5*C+ch,ch<C,0)
    t1=tl.load(T+node*5*C+C+ch,ch<C,0)
    t2=tl.load(T+node*5*C+2*C+ch,ch<C,0)
    t3=tl.load(T+node*5*C+3*C+ch,ch<C,0)
    t4=tl.load(T+node*5*C+4*C+ch,ch<C,0)
    return vx,vy,vz,t0,t1,t2,t3,t4


@tr.jit
def _geometry(P,Cell,Neighbors,Shifts,node,slot,K:tl.constexpr):
    neighbor=tl.load(Neighbors+node*K+slot,slot<K,-1)
    valid=(slot<K)&(neighbor>=0)
    neighbor=tl.maximum(neighbor,0)
    sx=tl.load(Shifts+(node*K+slot)*3,slot<K,0).to(tl.float32)
    sy=tl.load(Shifts+(node*K+slot)*3+1,slot<K,0).to(tl.float32)
    sz=tl.load(Shifts+(node*K+slot)*3+2,slot<K,0).to(tl.float32)
    dx=tl.load(P+neighbor*3)-tl.load(P+node*3)+sx*tl.load(Cell)+sy*tl.load(Cell+3)+sz*tl.load(Cell+6)
    dy=tl.load(P+neighbor*3+1)-tl.load(P+node*3+1)+sx*tl.load(Cell+1)+sy*tl.load(Cell+4)+sz*tl.load(Cell+7)
    dz=tl.load(P+neighbor*3+2)-tl.load(P+node*3+2)+sx*tl.load(Cell+2)+sy*tl.load(Cell+5)+sz*tl.load(Cell+8)
    dx,dy,dz=tl.where(valid,dx,1.),tl.where(valid,dy,0.),tl.where(valid,dz,0.)
    r=tl.sqrt(dx*dx+dy*dy+dz*dz)
    return neighbor,valid,r,dx/r,dy/r,dz/r


@tr.jit
def _radius_select(P, Cell, Neighbors, Shifts, OutNeighbors, OutShifts, OutCounts,
                   K: tl.constexpr, B: tl.constexpr, CUTOFF: tl.constexpr):
    node = tl.program_id(0)
    slot = tl.arange(0, B)
    neighbor, valid, distance, _, _, _ = _geometry(P, Cell, Neighbors, Shifts, node, slot, K)
    keep = valid & (distance < CUTOFF)
    count = tl.sum(keep.to(tl.int32), 0)
    rank = tl.cumsum(keep.to(tl.int32), 0) - 1
    # Tail and kept scatter destinations are disjoint, so there is no
    # cross-thread clear/scatter race. Every tail entry is reset each step.
    tail = (slot < K) & (slot >= count)
    tl.store(OutNeighbors + node*K + slot, -1, tail)
    for axis in tl.static_range(3):
        shift = tl.load(Shifts + (node*K + slot)*3 + axis, slot < K, 0)
        tl.store(OutShifts + (node*K + slot)*3 + axis, 0, tail)
        tl.store(OutShifts + (node*K + rank)*3 + axis, shift, keep)
    tl.store(OutNeighbors + node*K + rank, neighbor, keep)
    tl.store(OutCounts + node, count)


@tr.jit
def _forward(P,Cell,Neighbors,Shifts,Counts,M,V,T,Coeff,Sout,Vout,Tout,
             C:tl.constexpr,K:tl.constexpr,INTERVALS:tl.constexpr,CUTOFF:tl.constexpr,WIDTH:tl.constexpr,
             FIRST:tl.constexpr,BC:tl.constexpr,BN:tl.constexpr,NC:tl.constexpr=6):
    node=tl.program_id(0)
    ch=tl.program_id(1)*BC+tl.arange(0,BC)
    ss=tl.full((BC,),0.,tl.float32)
    svx,svy,svz,st0,st1,st2,st3,st4=ss,ss,ss,ss,ss,ss,ss,ss
    for start in range(tl.cdiv(tl.load(Counts+node),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,mask,r,ux,uy,uz=_geometry(P,Cell,Neighbors,Shifts,node,slot,K)
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
def _vjp(m0,m1,m2,m3,m4,m5,m6,vx,vy,vz,t0,t1,t2,t3,t4,
         gs,gx,gy,gz,g0,g1,g2,g3,g4,u,v,w,r,
         w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6,FIRST:tl.constexpr):
    q0,q1,q2,q3,q4=_q(u,v,w)
    f0=gs
    f1=gx*u+gy*v+gz*w
    f4=g0*q0+g1*q1+g2*q2+g3*q3+g4*q4
    gux,guy,guz=_matvec(g0,g1,g2,g3,g4,u,v,w)
    a1,a4=m1*w1,m4*w4
    dx,dy,dz=a1*gx+2*a4*gux,a1*gy+2*a4*guy,a1*gz+2*a4*guz
    dr=m0*d0*f0+m1*d1*f1+m4*d4*f4
    zero=tl.full(f1.shape,0.,tl.float32)
    if FIRST:
        dm2,dm3,dm5,dm6,dvx,dvy,dvz,dt0,dt1,dt2,dt3,dt4=zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero,zero
    else:
        tux,tuy,tuz=_matvec(t0,t1,t2,t3,t4,u,v,w)
        tgx,tgy,tgz=_matvec(t0,t1,t2,t3,t4,gx,gy,gz)
        gvx,gvy,gvz=_matvec(g0,g1,g2,g3,g4,vx,vy,vz)
        f2=gx*vx+gy*vy+gz*vz
        f3=gs*(tux*u+tuy*v+tuz*w)+gx*tux+gy*tuy+gz*tuz
        f5=g0*t0+g1*t1+g2*t2+g3*t3+g4*t4
        f6=gs*(vx*u+vy*v+vz*w)+vx*gux+vy*guy+vz*guz
        dm2,dm3,dm5,dm6=w2*f2,w3*f3,w5*f5,w6*f6
        a2,a3,a5,a6=m2*w2,m3*w3,m5*w5,m6*w6
        dvx,dvy,dvz=a2*gx+a6*(gs*u+gux),a2*gy+a6*(gs*v+guy),a2*gz+a6*(gs*w+guz)
        b0,b1,b2,b3,b4=_stv(gx,gy,gz,u,v,w)
        dt0,dt1,dt2,dt3,dt4=a5*g0+a3*(gs*q0+b0),a5*g1+a3*(gs*q1+b1),a5*g2+a3*(gs*q2+b2),a5*g3+a3*(gs*q3+b3),a5*g4+a3*(gs*q4+b4)
        dx+=a3*(2*gs*tux+tgx)+a6*(gs*vx+gvx)
        dy+=a3*(2*gs*tuy+tgy)+a6*(gs*vy+gvy)
        dz+=a3*(2*gs*tuz+tgz)+a6*(gs*vz+gvz)
        dr+=m2*d2*f2+m3*d3*f3+m5*d5*f5+m6*d6*f6
    projection=dx*u+dy*v+dz*w
    px,py,pz=dr*u+(dx-projection*u)/r,dr*v+(dy-projection*v)/r,dr*w+(dz-projection*w)/r
    return w0*f0,w1*f1,dm2,dm3,w4*f4,dm5,dm6,dvx,dvy,dvz,dt0,dt1,dt2,dt3,dt4,px,py,pz


@tr.jit
def _backward(P,Cell,Neighbors,Shifts,Counts,M,V,T,Coeff,GS,GV,GT,GM,GVI,GTI,GP,
              C:tl.constexpr,K:tl.constexpr,INTERVALS:tl.constexpr,CUTOFF:tl.constexpr,WIDTH:tl.constexpr,
              FIRST:tl.constexpr,NEED_NODE:tl.constexpr,BC:tl.constexpr,BN:tl.constexpr,NC:tl.constexpr=6):
    node=tl.program_id(0);tile=tl.program_id(1)
    ch=tile*BC+tl.arange(0,BC);channel=ch[None,:]
    own0,own1,own2,own3,own4,own5,own6=_load_message(M,node,channel,C,FIRST)
    ovx,ovy,ovz,ot0,ot1,ot2,ot3,ot4=_load_states(V,T,node,channel,C)
    ogs=tl.load(GS+node*C+channel,channel<C,0)
    ogx,ogy,ogz,og0,og1,og2,og3,og4=_load_states(GV,GT,node,channel,C)
    z=tl.full((BC,),0.,tl.float32)
    sm0,sm1,sm2,sm3,sm4,sm5,sm6,svx,svy,svz,st0,st1,st2,st3,st4,spx,spy,spz=z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z
    for start in range(tl.cdiv(tl.load(Counts+node),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,mask,r,ux,uy,uz=_geometry(P,Cell,Neighbors,Shifts,node,slot,K)
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
def _backward_scalar(P,Cell,Neighbors,Shifts,Counts,M,Coeff,GS,GV,GT,GM,GP,
                     C:tl.constexpr,K:tl.constexpr,INTERVALS:tl.constexpr,
                     CUTOFF:tl.constexpr,WIDTH:tl.constexpr,NEED_NODE:tl.constexpr,
                     BC:tl.constexpr,BN:tl.constexpr,N:tl.constexpr,
                     NC:tl.constexpr=6,BM:tl.constexpr=1):
    """Joint coordinate VJP of both directed scalar-source messages.

    At a fixed reverse-mode step the upstream atom features are constants.
    Combining the odd vector path by subtraction and the even scalar/tensor
    paths by addition permits one angular projection per undirected pair in
    each node-owned traversal. The message-state adjoint is still accumulated
    separately. No atomics, geometry caching or force approximation is used.
    """
    raw_node=tl.program_id(0)*BM+tl.arange(0,BM);tile=tl.program_id(1)
    # Clamp only inactive final-tile lanes; their stores are masked below.
    node=tl.minimum(raw_node,N-1)
    ch=tile*BC+tl.arange(0,BC);channel=ch[None,None,:]
    own=node[:,None,None]
    own0,own1,own2,own3,own4,own5,own6=_load_message(M,own,channel,C,True)
    ogs=tl.load(GS+own*C+channel,channel<C,0)
    ogx,ogy,ogz,og0,og1,og2,og3,og4=_load_states(GV,GT,own,channel,C)
    z=tl.full((BM,BC),0.,tl.float32)
    sm0,sm1,sm4,spx,spy,spz=z,z,z,z,z,z
    for start in range(tl.cdiv(tl.max(tl.load(Counts+node),0),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,mask,r,ux,uy,uz=_geometry(P,Cell,Neighbors,Shifts,node[:,None],slot[None,:],K)
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


class _Messages(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, cell, neighbors, shifts, counts, message, vector, tensor, coefficients, cutoff, width, first, settings):
        n,c=vector.shape[0],vector.shape[-1]
        if position.dtype != torch.float32 or position.device.type != 'cuda':
            raise ValueError('Fused L2 inference requires CUDA float32')
        message,vector,tensor=message.contiguous(),vector.contiguous(),tensor.contiguous()
        out_s=torch.empty((n,c),device=position.device,dtype=position.dtype)
        out_v=torch.empty_like(vector);out_t=torch.empty_like(tensor)
        bc,bn,warps,fma=settings
        if c==16 and coefficients.shape[1]==6 and (bc,bn,warps,fma)==(16,8,4,True):
            # Timings on real neighborhoods favor eight atom-owned rows for
            # both scalar and full tensor sources. Untuned configurations
            # keep the default layout.
            from hyff.kernels.message_tiles import _forward_atoms
            kernel=_forward_atoms[(tr.cdiv(n,8),1)](
                position,cell,neighbors,shifts,counts,message,vector,tensor,coefficients,out_s,out_v,out_t,
                c,neighbors.shape[1],coefficients.shape[0],cutoff,width,first,bc,1,n,
                NC=6,BM=8,num_warps=warps,enable_fp_fusion=fma)
            _record(kernel, f'forward_C{c}_K{neighbors.shape[1]}_first{first}_atoms8_neighbors1_coeff6_{settings}')
        else:
            kernel=_forward[(n,tr.cdiv(c,bc))](position,cell,neighbors,shifts,counts,message,vector,tensor,coefficients,out_s,out_v,out_t,
                c,neighbors.shape[1],coefficients.shape[0],cutoff,width,first,bc,bn,NC=coefficients.shape[1],num_warps=warps,enable_fp_fusion=fma)
            _record(kernel, f'forward_C{c}_K{neighbors.shape[1]}_first{first}_coeff{coefficients.shape[1]}_{settings}')
        ctx.save_for_backward(position,cell,neighbors,shifts,counts,message,vector,tensor,coefficients)
        ctx.settings=(cutoff,width,first,settings)
        return out_s,out_v,out_t

    @staticmethod
    def backward(ctx, gs, gv, gt):
        if torch.is_grad_enabled():
            raise RuntimeError('Fused L2 messages are inference-only; use the reference for force training')
        position,cell,neighbors,shifts,counts,message,vector,tensor,coefficients=ctx.saved_tensors
        cutoff,width,first,settings=ctx.settings
        bc,bn,warps,fma=settings
        n,c=vector.shape[0],vector.shape[-1]
        gp=torch.empty((n,3,tr.cdiv(c,bc)),device=position.device,dtype=position.dtype)
        gs=gs.contiguous() if gs is not None else torch.zeros((n,c),device=position.device)
        gv=gv.contiguous() if gv is not None else torch.zeros_like(vector)
        gt=gt.contiguous() if gt is not None else torch.zeros_like(tensor)
        if first:
            need_message=ctx.needs_input_grad[5]
            gm=torch.empty_like(message) if need_message else position
            # Timings on water and GroEL at 3.5/4.5 A favor four atoms per
            # program when narrow source adjoints are needed.
            # Keep the single-atom layout for the frozen element-only source
            # and other untuned configurations.
            atom_tile=4 if need_message and c<=16 and (bc,bn,warps,fma)==(16,8,4,True) else 1
            neighbor_tile=bn//atom_tile
            kernel=_backward_scalar[(tr.cdiv(n,atom_tile),tr.cdiv(c,bc))](
                position,cell,neighbors,shifts,counts,message,coefficients,gs,gv,gt,gm,gp,
                c,neighbors.shape[1],coefficients.shape[0],cutoff,width,need_message,bc,neighbor_tile,n,
                NC=coefficients.shape[1],BM=atom_tile,num_warps=warps,enable_fp_fusion=fma)
            _record(kernel,f'backward_scalar_C{c}_K{neighbors.shape[1]}_nodes{need_message}_atoms{atom_tile}_neighbors{neighbor_tile}_coeff{coefficients.shape[1]}_{settings}')
            return (gp.sum(-1),None,None,None,None,gm if need_message else None,
                    torch.zeros_like(vector) if ctx.needs_input_grad[6] else None,
                    torch.zeros_like(tensor) if ctx.needs_input_grad[7] else None,
                    None,None,None,None,None)
        need_node=any(ctx.needs_input_grad[5:8])
        # The first block's element embeddings and zero tensor states have no
        # coordinate dependency. Their adjoints can be omitted exactly.
        gm,gv_in,gt_in=((torch.empty_like(message),torch.empty_like(vector),torch.empty_like(tensor))
                        if need_node else (position,position,position))
        if c==16 and coefficients.shape[1]==6 and (bc,bn,warps,fma)==(16,8,4,True):
            # Timings on water and GroEL at 3.5/4.5 A favor eight atom-owned
            # rows. Untuned widths/tables keep the default layout. Both
            # directed coordinate roles and all source
            # adjoints remain active; this changes execution only.
            from hyff.kernels.tensor_vjp import _backward_tensor
            kernel=_backward_tensor[(tr.cdiv(n,8),1)](
                position,cell,neighbors,shifts,counts,message,vector,tensor,coefficients,
                gs,gv,gt,gm,gv_in,gt_in,gp,c,neighbors.shape[1],coefficients.shape[0],
                cutoff,width,need_node,bc,1,n,NC=6,BM=8,JOINT=True,
                num_warps=warps,enable_fp_fusion=fma)
            _record(kernel,f'backward_tensor_C{c}_K{neighbors.shape[1]}_nodes{need_node}_atoms8_neighbors1_coeff6_{settings}')
        else:
            kernel=_backward[(n,tr.cdiv(c,bc))](position,cell,neighbors,shifts,counts,message,vector,tensor,coefficients,gs,gv,gt,gm,gv_in,gt_in,gp,
                c,neighbors.shape[1],coefficients.shape[0],cutoff,width,first,need_node,bc,bn,NC=coefficients.shape[1],num_warps=warps,enable_fp_fusion=fma)
            _record(kernel, f'backward_C{c}_K{neighbors.shape[1]}_first{first}_nodes{need_node}_coeff{coefficients.shape[1]}_{settings}')
        return (gp.sum(-1),None,None,None,None,gm if need_node else None,
                gv_in if need_node else None,gt_in if need_node else None,None,None,None,None,None)


def messages(position, cell, neighbors, shifts, counts, message, vector, tensor, table, cutoff, width, first, settings=(32,8,4,False)):
    return _Messages.apply(position,cell,neighbors,shifts,counts,message,vector,tensor,table.coefficients,cutoff,width,first,settings)


@tr.jit
def _pair_value(Coeff, r, pair, mask, INTERVALS:tl.constexpr,
                CUTOFF:tl.constexpr, WIDTH:tl.constexpr, NC:tl.constexpr=6):
    coordinate=tl.minimum(tl.maximum(r*(INTERVALS/CUTOFF),0.),INTERVALS)
    interval=tl.minimum(coordinate.to(tl.int32),INTERVALS-1)
    value,derivative=_poly(Coeff,interval,coordinate-interval,pair,64,64,INTERVALS/CUTOFF,0,NC)
    x=tl.minimum(tl.maximum((CUTOFF-r)/WIDTH,0.),1.)
    envelope=x*x*x*(10.-15.*x+6.*x*x)*mask
    slope=-30.*x*x*(1.-x)*(1.-x)/WIDTH*mask
    return value*envelope,derivative*envelope+value*slope


@tr.jit
def _pair_forward(P,Cell,Neighbors,Shifts,Counts,Elements,Coeff,Energy,
                  K:tl.constexpr,INTERVALS:tl.constexpr,CUTOFF:tl.constexpr,WIDTH:tl.constexpr,BN:tl.constexpr,NC:tl.constexpr=6):
    node=tl.program_id(0)
    own=tl.load(Elements+node)
    total=tl.full((BN,),0.,tl.float32)
    for start in range(tl.cdiv(tl.load(Counts+node),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,valid,r,_,_,_=_geometry(P,Cell,Neighbors,Shifts,node,slot,K)
        other=tl.load(Elements+j)
        pair=tl.minimum(own,other)*8+tl.maximum(own,other)
        value,_=_pair_value(Coeff,r,pair,valid,INTERVALS,CUTOFF,WIDTH,NC)
        total+=.5*value
    tl.store(Energy+node,tl.sum(total,0))


@tr.jit
def _pair_backward(P,Cell,Neighbors,Shifts,Counts,Elements,Coeff,Gradient,GP,
                   K:tl.constexpr,INTERVALS:tl.constexpr,CUTOFF:tl.constexpr,WIDTH:tl.constexpr,BN:tl.constexpr,NC:tl.constexpr=6):
    node=tl.program_id(0)
    own=tl.load(Elements+node)
    g=tl.load(Gradient+node)
    gx=tl.full((BN,),0.,tl.float32)
    gy,gz=gx,gx
    for start in range(tl.cdiv(tl.load(Counts+node),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,valid,r,x,y,z=_geometry(P,Cell,Neighbors,Shifts,node,slot,K)
        other=tl.load(Elements+j)
        pair=tl.minimum(own,other)*8+tl.maximum(own,other)
        _,slope=_pair_value(Coeff,r,pair,valid,INTERVALS,CUTOFF,WIDTH,NC)
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


class _RadiusView:
    def __init__(self, parent):
        self.model, self.cell = parent.model, parent.cell
        self.neighbors = torch.empty_like(parent.neighbors)
        self.shifts = torch.empty_like(parent.shifts)
        self.counts = torch.empty_like(parent.counts)


class PreparedL2Neighbors:
    """Checked, untruncated matrix from the symmetric physical radius graph.

    Native GPU radius edges are already grouped by receiver, making normal
    preparation O(N+E). An arbitrary reference ordering can be sorted once.
    The graph's symmetry is checked when this prepared object is initialized.
    """
    def __init__(self, model, batch, edges):
        from hyff.graph import PeriodicEdges
        self.model,self.batch,self.edges=model,batch,edges
        self.n=len(batch['positions'])
        self.periodic=isinstance(edges,PeriodicEdges)
        self.cell=batch['cell'][0] if 'cell' in batch else batch['positions'].new_zeros((3,3))
        self.reserve_factor=getattr(model,'_l2_neighbor_reserve_factor',1.2)
        if not math.isfinite(self.reserve_factor) or self.reserve_factor<1.:
            raise ValueError('Neighbor reserve factor must be finite and at least one')
        self.capacity=None
        self._update(check_symmetry=True)
        radii = set(model.layer_cutoffs)
        self.filter_skin = getattr(model, 'filter_skin_neighbors', False)
        if self.filter_skin:radii.add(model.cutoff)
        self.radius_views = {radius: _RadiusView(self) for radius in radii
                             if radius < model.cutoff or self.filter_skin}

    def refresh_subgraphs(self, positions, active_depth):
        # Filter the checked full-radius graph at CURRENT coordinates on every
        # energy evaluation. Thus subgraphs require no extra skin assumptions
        # and remain valid during CUDA graph replay and neighbor rebuilds.
        radii = set(self.model.layer_cutoffs[:active_depth])
        if self.filter_skin:radii.add(self.model.cutoff)
        for radius in sorted(radii):
            if radius in self.radius_views:
                view = self.radius_views[radius]
                kernel = _radius_select[(self.n,)](positions, self.cell, self.neighbors,
                    self.shifts, view.neighbors, view.shifts, view.counts, self.capacity,
                    tr.next_power_of_2(self.capacity), radius, num_warps=4)
                _record(kernel, ('radius_select', self.capacity, radius))

    def radius_view(self, radius):
        return self.radius_views.get(radius, self)

    @torch.no_grad()
    def _update(self, check_symmetry=False):
        index=self.edges.edge_index if self.periodic else self.edges
        unit=self.edges.unit_shifts if self.periodic else torch.zeros((index.shape[1],3),device=index.device,dtype=torch.int32)
        valid=(index[0]!=index[1])|(unit!=0).any(-1)
        source,destination=index[:,valid]
        unit=unit[valid]
        if check_symmetry:
            # An independent CPU set check includes lattice images, not merely
            # endpoint indices. It is setup-only, and recorded by the profiler.
            rows=torch.cat((source[:,None],destination[:,None],unit),dim=1).cpu().tolist()
            directed={tuple(row) for row in rows}
            if len(directed)!=len(rows) or any((j,i,-a,-b,-c) not in directed for i,j,a,b,c in directed):
                raise ValueError('Fused reverse requires a duplicate-free symmetric radius graph')
        counts=torch.bincount(destination,minlength=self.n)
        required=int(counts.max()) if len(counts) else 0
        if self.capacity is None:
            self.capacity=max(8,tr.cdiv(int(required*self.reserve_factor)+1,8)*8)
            self.neighbors=torch.full((self.n,self.capacity),-1,device=index.device,dtype=torch.int64)
            self.shifts=torch.zeros((self.n,self.capacity,3),device=index.device,dtype=torch.int32)
            self.counts=torch.zeros(self.n,device=index.device,dtype=torch.int32)
        elif required>self.capacity:
            return False
        if len(destination)>1 and not bool((destination[1:]>=destination[:-1]).all()):
            order=torch.argsort(destination,stable=True)
            source,destination,unit=source[order],destination[order],unit[order]
        prefix=counts.cumsum(0)-counts
        slot=torch.arange(len(source),device=index.device)-prefix[destination]
        self.neighbors.fill_(-1);self.shifts.zero_()
        self.counts.copy_(counts)
        self.neighbors[destination,slot]=source
        self.shifts[destination,slot]=unit
        return True

    def update_edges(self):
        return self._update()

    def __call__(self, positions):
        return self.model.energy({**self.batch,'positions':positions},self.edges,csr=self)
