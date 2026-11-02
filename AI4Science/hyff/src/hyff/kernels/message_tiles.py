"""Atom-owned layouts for the L2 message forward.

The C = 16 dispatch of hyff.kernels.fused_messages uses the eight-atom layout;
the C = 8 engines select their own layouts, and others remain available for
numerical and workload diagnostics. These kernels change only the GPU
reduction layout.
"""
import triton as tr
import triton.language as tl

from hyff.kernels.fused_messages import (
    _filters, _geometry, _load_message, _load_states, _matvec, _q, _stv,
)


@tr.jit
def _forward_atoms(P, Cell, Neighbors, Shifts, Counts, M, V, T, Coeff, Sout, Vout, Tout,
                   C: tl.constexpr, K: tl.constexpr, INTERVALS: tl.constexpr,
                   CUTOFF: tl.constexpr, WIDTH: tl.constexpr, FIRST: tl.constexpr,
                   BC: tl.constexpr, BN: tl.constexpr, N: tl.constexpr,
                   NC: tl.constexpr = 6, BM: tl.constexpr = 1):
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
            Coeff, r[:, :, None], channel, mask[:, :, None],
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
