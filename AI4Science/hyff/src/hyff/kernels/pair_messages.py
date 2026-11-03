"""Frozen symmetric-bank messages on the atom-owned contractions of hyff.kernels.fused_messages.

The kernels keep all source/coordinate adjoints. Compared with fused_messages
they add symmetric element-pair table addressing, explicit inactive count
masks and common R8/R16 atom tiles. With BANKS=False the same kernels run with
the bank addressing compiled out (one shared radial table).
"""
import torch
import triton as tr
import triton.language as tl

from hyff.kernels.fused_messages import (
    _filters, _geometry, _load_message, _load_states, _matvec, _q, _stv, _vjp, _record,
)
from hyff.kernels.tensor_vjp import _tensor_vjp_parts


@tr.jit
def _symmetric_pointer(Coeff, Elements, own, neighbor, C: tl.constexpr,
                       INTERVALS: tl.constexpr, FIRST: tl.constexpr, BANKS: tl.constexpr):
    O: tl.constexpr = (3 if FIRST else 7) * C
    if BANKS:
        a = tl.load(Elements + own).to(tl.int32)
        b = tl.load(Elements + neighbor).to(tl.int32)
        low, high = tl.minimum(a, b), tl.maximum(a, b)
        bank = low * (15 - low) // 2 + high
    else:
        bank = tl.full(neighbor.shape, 0, tl.int32)
    return Coeff + bank * INTERVALS * 6 * O


@tr.jit
def _forward_pair(P, Cell, Neighbors, Shifts, Counts, M, V, T, Coeff, Elements, Sout, Vout, Tout,
                   C: tl.constexpr, K: tl.constexpr, INTERVALS: tl.constexpr,
                   CUTOFF: tl.constexpr, WIDTH: tl.constexpr, FIRST: tl.constexpr,
                   BC: tl.constexpr, BN: tl.constexpr, N: tl.constexpr,
                   NC: tl.constexpr = 6, BM: tl.constexpr = 1, BANKS: tl.constexpr = True):
    raw_node = tl.program_id(0) * BM + tl.arange(0, BM)
    node = tl.minimum(raw_node, N - 1)
    ch = tl.program_id(1) * BC + tl.arange(0, BC)
    channel = ch[None, None, :]
    counts = tl.load(Counts + node)
    ss = tl.full((BM, BC), 0., tl.float32)
    svx, svy, svz, st0, st1, st2, st3, st4 = ss, ss, ss, ss, ss, ss, ss, ss
    for start in range(tl.cdiv(tl.max(counts, 0), BN)):
        slot = start * BN + tl.arange(0, BN)
        j, mask, r, ux, uy, uz = _geometry(
            P, Cell, Neighbors, Shifts, node[:, None], slot[None, :], K)
        # Different rows can have different counts. The shared loop must
        # never interpret another row's tail capacity as physical neighbors.
        mask = mask & (slot[None, :] < counts[:, None])
        # Stale tail indices can be zero-distance self edges. Zero filters
        # cannot cancel a NaN direction, so give inactive geometry finite
        # values before using it in the contractions.
        r = tl.where(mask, r, 1.)
        ux, uy, uz = tl.where(mask, ux, 0.), tl.where(mask, uy, 0.), tl.where(mask, uz, 0.)
        u, v, w = ux[:, :, None], uy[:, :, None], uz[:, :, None]
        w0, w1, w2, w3, w4, w5, w6, d0, d1, d2, d3, d4, d5, d6 = _filters(
            _symmetric_pointer(Coeff, Elements, node[:, None], j, C, INTERVALS, FIRST, BANKS)[:, :, None], r[:, :, None], channel, mask[:, :, None],
            C, INTERVALS, CUTOFF, WIDTH, FIRST, NC)
        m0, m1, m2, m3, m4, m5, m6 = _load_message(M, j[:, :, None], channel, C, FIRST)
        a0, a1, a2, a3, a4, a5, a6 = m0*w0, m1*w1, m2*w2, m3*w3, m4*w4, m5*w5, m6*w6
        q0, q1, q2, q3, q4 = _q(u, v, w)
        if FIRST:
            es = a0
            evx, evy, evz = a1*u, a1*v, a1*w
            et0, et1, et2, et3, et4 = a4*q0, a4*q1, a4*q2, a4*q3, a4*q4
        else:
            vx, vy, vz, t0, t1, t2, t3, t4 = _load_states(V, T, j[:, :, None], channel, C)
            tx, ty, tz = _matvec(t0, t1, t2, t3, t4, u, v, w)
            b0, b1, b2, b3, b4 = _stv(vx, vy, vz, u, v, w)
            es = a0 + a6*(vx*u + vy*v + vz*w) + a3*(tx*u + ty*v + tz*w)
            evx, evy, evz = a2*vx+a1*u+a3*tx, a2*vy+a1*v+a3*ty, a2*vz+a1*w+a3*tz
            et0, et1, et2, et3, et4 = (a5*t0+a4*q0+a6*b0, a5*t1+a4*q1+a6*b1,
                a5*t2+a4*q2+a6*b2, a5*t3+a4*q3+a6*b3, a5*t4+a4*q4+a6*b4)
        ss += tl.sum(es, 1)
        svx += tl.sum(evx, 1)
        svy += tl.sum(evy, 1)
        svz += tl.sum(evz, 1)
        st0 += tl.sum(et0, 1)
        st1 += tl.sum(et1, 1)
        st2 += tl.sum(et2, 1)
        st3 += tl.sum(et3, 1)
        st4 += tl.sum(et4, 1)
    output_node, output_ch = raw_node[:, None], ch[None, :]
    valid = (output_node < N) & (output_ch < C)
    base = output_node*C + output_ch
    tl.store(Sout + base, ss, valid)
    base = output_node*3*C + output_ch
    tl.store(Vout + base, svx, valid)
    tl.store(Vout + base+C, svy, valid)
    tl.store(Vout + base+2*C, svz, valid)
    base = output_node*5*C + output_ch
    tl.store(Tout + base, st0, valid)
    tl.store(Tout + base+C, st1, valid)
    tl.store(Tout + base+2*C, st2, valid)
    tl.store(Tout + base+3*C, st3, valid)
    tl.store(Tout + base+4*C, st4, valid)


@tr.jit
def _backward_pair_scalar(P,Cell,Neighbors,Shifts,Counts,M,Coeff,Elements,GS,GV,GT,GM,GP,
                     C:tl.constexpr,K:tl.constexpr,INTERVALS:tl.constexpr,
                     CUTOFF:tl.constexpr,WIDTH:tl.constexpr,NEED_NODE:tl.constexpr,
                     BC:tl.constexpr,BN:tl.constexpr,N:tl.constexpr,
                     NC:tl.constexpr=6,BM:tl.constexpr=1,BANKS:tl.constexpr=True):
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
    counts=tl.load(Counts+node)
    for start in range(tl.cdiv(tl.max(counts,0),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,mask,r,ux,uy,uz=_geometry(P,Cell,Neighbors,Shifts,node[:,None],slot[None,:],K)
        mask=mask & (slot[None,:] < counts[:,None])
        r=tl.where(mask,r,1.)
        ux,uy,uz=tl.where(mask,ux,0.),tl.where(mask,uy,0.),tl.where(mask,uz,0.)
        u,v,w=ux[:,:,None],uy[:,:,None],uz[:,:,None]
        w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6=_filters(
            _symmetric_pointer(Coeff,Elements,node[:,None],j,C,INTERVALS,True,BANKS)[:,:,None],r[:,:,None],channel,mask[:,:,None],C,INTERVALS,CUTOFF,WIDTH,True,NC)
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


@tr.jit
def _backward_pair_tensor(P,Cell,Neighbors,Shifts,Counts,M,V,T,Coeff,Elements,GS,GV,GT,GM,GVI,GTI,GP,
                     C:tl.constexpr,K:tl.constexpr,INTERVALS:tl.constexpr,
                     CUTOFF:tl.constexpr,WIDTH:tl.constexpr,NEED_NODE:tl.constexpr,
                     BC:tl.constexpr,BN:tl.constexpr,N:tl.constexpr,
                     NC:tl.constexpr=6,BM:tl.constexpr=1,JOINT:tl.constexpr=True,BANKS:tl.constexpr=True):
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
    counts=tl.load(Counts+node)
    for start in range(tl.cdiv(tl.max(counts,0),BN)):
        slot=start*BN+tl.arange(0,BN)
        j,mask,r,ux,uy,uz=_geometry(P,Cell,Neighbors,Shifts,node[:,None],slot[None,:],K)
        mask=mask & (slot[None,:] < counts[:,None])
        r=tl.where(mask,r,1.)
        ux,uy,uz=tl.where(mask,ux,0.),tl.where(mask,uy,0.),tl.where(mask,uz,0.)
        u,v,w=ux[:,:,None],uy[:,:,None],uz[:,:,None]
        w0,w1,w2,w3,w4,w5,w6,d0,d1,d2,d3,d4,d5,d6=_filters(_symmetric_pointer(Coeff,Elements,node[:,None],j,C,INTERVALS,False,BANKS)[:,:,None],r[:,:,None],channel,mask[:,:,None],C,INTERVALS,CUTOFF,WIDTH,False,NC)
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


class _PairMessages(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, cell, neighbors, shifts, counts, elements,
                message, vector, tensor, coefficients, cutoff, width, first, settings):
        if position.dtype != torch.float32 or not position.is_cuda:
            raise ValueError('Symmetric fused inference requires CUDA FP32')
        n, c = vector.shape[0], vector.shape[-1]
        banks = coefficients.ndim == 4
        intervals = coefficients.shape[1] if banks else coefficients.shape[0]
        if (c not in (8, 16) or tuple(settings) != (16, 8, 4, True)
                or (banks and coefficients.shape[0] != 36)
                or coefficients.shape[-2] != 6
                or coefficients.shape[-1] != (3 if first else 7) * c):
            raise ValueError('Expected declared R8/R16 quintic table and tile shapes')
        if any(x.requires_grad for x in (cell, coefficients)):
            raise ValueError('Cell/table training is unsupported by the frozen force backend')
        message, vector, tensor = message.contiguous(), vector.contiguous(), tensor.contiguous()
        out_s = position.new_empty((n, c))
        out_v, out_t = torch.empty_like(vector), torch.empty_like(tensor)
        bc, _, warps, fma = settings
        kernel = _forward_pair[(tr.cdiv(n, 8), tr.cdiv(c, bc))](
            position, cell, neighbors, shifts, counts, message, vector, tensor, coefficients,
            elements, out_s, out_v, out_t, c, neighbors.shape[1], intervals, cutoff, width,
            first, bc, 1, n, NC=6, BM=8, BANKS=banks, num_warps=warps, enable_fp_fusion=fma)
        _record(kernel, f'pair_forward_C{c}_K{neighbors.shape[1]}_first{first}_banks{banks}_atoms8')
        ctx.save_for_backward(position, cell, neighbors, shifts, counts, elements,
                              message, vector, tensor, coefficients)
        ctx.settings = (cutoff, width, first, settings, banks, intervals)
        return out_s, out_v, out_t

    @staticmethod
    def backward(ctx, gs, gv, gt):
        if torch.is_grad_enabled():
            raise RuntimeError('Pair fused messages are inference-only; use the reference for force training')
        position, cell, neighbors, shifts, counts, elements, message, vector, tensor, coefficients = ctx.saved_tensors
        cutoff, width, first, settings, banks, intervals = ctx.settings
        bc, bn, warps, fma = settings
        n, c = vector.shape[0], vector.shape[-1]
        gp = position.new_empty((n, 3, tr.cdiv(c, bc)))
        gs = gs.contiguous() if gs is not None else position.new_zeros((n, c))
        gv = gv.contiguous() if gv is not None else torch.zeros_like(vector)
        gt = gt.contiguous() if gt is not None else torch.zeros_like(tensor)
        if first:
            need_message = ctx.needs_input_grad[6]
            gm = torch.empty_like(message) if need_message else position
            atom_tile = 4 if need_message else 1
            kernel = _backward_pair_scalar[(tr.cdiv(n, atom_tile), tr.cdiv(c, bc))](
                position, cell, neighbors, shifts, counts, message, coefficients, elements,
                gs, gv, gt, gm, gp, c, neighbors.shape[1], intervals, cutoff, width,
                need_message, bc, bn // atom_tile, n, NC=6, BM=atom_tile, BANKS=banks,
                num_warps=warps, enable_fp_fusion=fma)
            _record(kernel, f'pair_scalar_vjp_C{c}_K{neighbors.shape[1]}_nodes{need_message}_banks{banks}_atoms{atom_tile}')
            return (gp.sum(-1), None, None, None, None, None,
                    gm if need_message else None,
                    torch.zeros_like(vector) if ctx.needs_input_grad[7] else None,
                    torch.zeros_like(tensor) if ctx.needs_input_grad[8] else None,
                    None, None, None, None, None)
        need_node = any(ctx.needs_input_grad[6:9])
        gm, gvi, gti = ((torch.empty_like(message), torch.empty_like(vector), torch.empty_like(tensor))
                        if need_node else (position, position, position))
        kernel = _backward_pair_tensor[(tr.cdiv(n, 8), tr.cdiv(c, bc))](
            position, cell, neighbors, shifts, counts, message, vector, tensor, coefficients,
            elements, gs, gv, gt, gm, gvi, gti, gp, c, neighbors.shape[1], intervals,
            cutoff, width, need_node, bc, 1, n, NC=6, BM=8, JOINT=True, BANKS=banks,
            num_warps=warps, enable_fp_fusion=fma)
        _record(kernel, f'pair_tensor_vjp_C{c}_K{neighbors.shape[1]}_nodes{need_node}_banks{banks}_atoms8')
        return (gp.sum(-1), None, None, None, None, None, gm if need_node else None,
                gvi if need_node else None, gti if need_node else None,
                None, None, None, None, None)


def messages(position, cell, neighbors, shifts, counts, elements, message, vector, tensor,
             table, cutoff, width, first, settings=(16, 8, 4, True)):
    return _PairMessages.apply(position, cell, neighbors, shifts, counts, elements,
        message, vector, tensor, table.coefficients, cutoff, width, first, settings)
