"""Fused local features with centered FP32 scalar LayerNorm and input VJP.

The operator has no parameters of its own. It evaluates the block's affine
LayerNorm, two tensor norms, optional three normalized polynomial invariants,
and two dot closures. The plain model never enables it; the native engines opt
in. Force training uses the ordinary differentiable PyTorch operations.
"""
from types import MethodType

import torch
import triton as tr
import triton.language as tl

from hyff.kernels.fused_messages import _matvec, _q, _stv


@tr.jit
def _load_vector(P, row, channel, N, T: tl.constexpr,
                 STRIDE0: tl.constexpr, STRIDE1: tl.constexpr, STRIDE2: tl.constexpr):
    base = row[:, None] * STRIDE0 + channel[None, :] * STRIDE2
    valid = (row[:, None] < N) & (channel[None, :] < T)
    a = tl.load(P + base, valid, 0.)
    b = tl.load(P + base + STRIDE1, valid, 0.)
    c = tl.load(P + base + 2 * STRIDE1, valid, 0.)
    return a, b, c


@tr.jit
def _load_tensor(P, row, channel, N, T: tl.constexpr,
                 STRIDE0: tl.constexpr, STRIDE1: tl.constexpr, STRIDE2: tl.constexpr):
    a, b, c = _load_vector(P, row, channel, N, T, STRIDE0, STRIDE1, STRIDE2)
    base = row[:, None]*STRIDE0 + channel[None, :]*STRIDE2
    valid = (row[:, None] < N) & (channel[None, :] < T)
    d = tl.load(P + base + 3 * STRIDE1, valid, 0.)
    e = tl.load(P + base + 4 * STRIDE1, valid, 0.)
    return a, b, c, d, e


@tr.jit
def _square_order2(a, b, c, d, e):
    xx = .7071067811865475 * a + .4082482904638631 * e
    yy = -.7071067811865475 * a + .4082482904638631 * e
    zz = -.8164965809277261 * e
    xy, xz, yz = .7071067811865475*b, .7071067811865475*c, .7071067811865475*d
    qxx, qyy, qzz = xx*xx+xy*xy+xz*xz, yy*yy+xy*xy+yz*yz, zz*zz+xz*xz+yz*yz
    qxy, qxz, qyz = xy*(xx+yy)+xz*yz, xz*(xx+zz)+xy*yz, yz*(yy+zz)+xy*xz
    return (.7071067811865475*(qxx-qyy), 1.4142135623730951*qxy,
            1.4142135623730951*qxz, 1.4142135623730951*qyz,
            .4082482904638631*(qxx+qyy-2*qzz))


@tr.jit
def _features_forward(X, W, B, VU, VV, TU, TV, Out, Dots, Mean, Rstd,
                      N, S: tl.constexpr, T: tl.constexpr, EPS: tl.constexpr,
                      HIGHER: tl.constexpr,
                      VUS0: tl.constexpr, VUS1: tl.constexpr, VUS2: tl.constexpr,
                      VVS0: tl.constexpr, VVS1: tl.constexpr, VVS2: tl.constexpr,
                      TUS0: tl.constexpr, TUS1: tl.constexpr, TUS2: tl.constexpr,
                      TVS0: tl.constexpr, TVS1: tl.constexpr, TVS2: tl.constexpr,
                      BM: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    ch = tl.arange(0, BC)
    valid_s = (row[:, None] < N) & (ch[None, :] < S)
    valid_t = (row[:, None] < N) & (ch[None, :] < T)
    x = tl.load(X + row[:, None]*S + ch[None, :], valid_s, 0.).to(tl.float32)
    centered = x
    mean = tl.sum(centered, 1)/S
    deviation = tl.where(ch[None, :] < S, centered-mean[:, None], 0.)
    rstd = tl.rsqrt(tl.sum(deviation*deviation, 1)/S + EPS)
    weight, bias = tl.load(W+ch, ch < S, 0.).to(tl.float32), tl.load(B+ch, ch < S, 0.).to(tl.float32)
    normalized = deviation*rstd[:, None]*weight[None, :] + bias[None, :]
    WIDTH: tl.constexpr = S + (5 if HIGHER else 2)*T
    base = row[:, None]*WIDTH + ch[None, :]
    tl.store(Out+base, normalized, valid_s)
    tl.store(Mean+row, mean, row < N)
    tl.store(Rstd+row, rstd, row < N)
    ux, uy, uz = _load_vector(VU, row, ch, N, T, VUS0, VUS1, VUS2)
    vx, vy, vz = _load_vector(VV, row, ch, N, T, VVS0, VVS1, VVS2)
    u0, u1, u2, u3, u4 = _load_tensor(TU, row, ch, N, T, TUS0, TUS1, TUS2)
    t0, t1, t2, t3, t4 = _load_tensor(TV, row, ch, N, T, TVS0, TVS1, TVS2)
    v2 = vx*vx + vy*vy + vz*vz
    t2norm = t0*t0 + t1*t1 + t2*t2 + t3*t3 + t4*t4
    vn, tn = tl.sqrt(v2+1.e-8), tl.sqrt(t2norm+1.e-8)
    tl.store(Out+base+S, vn, valid_t)
    tl.store(Out+base+S+T, tn, valid_t)
    dot_base = row[:, None]*2*T + ch[None, :]
    tl.store(Dots+dot_base, ux*vx+uy*vy+uz*vz, valid_t)
    tl.store(Dots+dot_base+T, u0*t0+u1*t1+u2*t2+u3*t3+u4*t4, valid_t)
    if HIGHER:
        vr, trstd = tl.rsqrt(1.+v2), tl.rsqrt(1.+t2norm)
        x, y, z = vx*vr, vy*vr, vz*vr
        a, b, c, d, e = t0*trstd, t1*trstd, t2*trstd, t3*trstd, t4*trstd
        tx, ty, tz = _matvec(a, b, c, d, e, x, y, z)
        q0, q1, q2, q3, q4 = _square_order2(a, b, c, d, e)
        tl.store(Out+base+S+2*T, q0*a+q1*b+q2*c+q3*d+q4*e, valid_t)
        tl.store(Out+base+S+3*T, x*tx+y*ty+z*tz, valid_t)
        tl.store(Out+base+S+4*T, tx*tx+ty*ty+tz*tz, valid_t)


@tr.jit
def _features_backward(X, W, VU, VV, TU, TV, Features, Mean, Rstd, GOut, GDots,
                       DX, DVU, DVV, DTU, DTV, N, S: tl.constexpr, T: tl.constexpr,
                       HIGHER: tl.constexpr,
                       VUS0: tl.constexpr, VUS1: tl.constexpr, VUS2: tl.constexpr,
                       VVS0: tl.constexpr, VVS1: tl.constexpr, VVS2: tl.constexpr,
                       TUS0: tl.constexpr, TUS1: tl.constexpr, TUS2: tl.constexpr,
                       TVS0: tl.constexpr, TVS1: tl.constexpr, TVS2: tl.constexpr,
                       BM: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    ch = tl.arange(0, BC)
    valid_s = (row[:, None] < N) & (ch[None, :] < S)
    valid_t = (row[:, None] < N) & (ch[None, :] < T)
    x = tl.load(X+row[:, None]*S+ch[None, :], valid_s, 0.).to(tl.float32)
    mean, rstd = tl.load(Mean+row, row < N, 0.), tl.load(Rstd+row, row < N, 0.)
    norm = tl.where(ch[None, :] < S, (x-mean[:, None])*rstd[:, None], 0.)
    WIDTH: tl.constexpr = S + (5 if HIGHER else 2)*T
    base = row[:, None]*WIDTH+ch[None, :]
    weight = tl.load(W+ch, ch < S, 0.).to(tl.float32)
    gradient = tl.load(GOut+base, valid_s, 0.).to(tl.float32)*weight[None, :]
    avg, projected = tl.sum(gradient, 1)/S, tl.sum(gradient*norm, 1)/S
    dx = (gradient-avg[:, None]-norm*projected[:, None])*rstd[:, None]
    tl.store(DX+row[:, None]*S+ch[None, :], dx, valid_s)
    ux, uy, uz = _load_vector(VU, row, ch, N, T, VUS0, VUS1, VUS2)
    vx, vy, vz = _load_vector(VV, row, ch, N, T, VVS0, VVS1, VVS2)
    u0, u1, u2, u3, u4 = _load_tensor(TU, row, ch, N, T, TUS0, TUS1, TUS2)
    t0, t1, t2, t3, t4 = _load_tensor(TV, row, ch, N, T, TVS0, TVS1, TVS2)
    vn = tl.load(Features+base+S, valid_t, 1.)
    tn = tl.load(Features+base+S+T, valid_t, 1.)
    gv = tl.load(GOut+base+S, valid_t, 0.)/vn
    gt = tl.load(GOut+base+S+T, valid_t, 0.)/tn
    dot_base = row[:, None]*2*T + ch[None, :]
    gdv, gdt = tl.load(GDots+dot_base, valid_t, 0.), tl.load(GDots+dot_base+T, valid_t, 0.)
    dvx, dvy, dvz = gv*vx+gdv*ux, gv*vy+gdv*uy, gv*vz+gdv*uz
    dt0, dt1, dt2, dt3, dt4 = (gt*t0+gdt*u0, gt*t1+gdt*u1, gt*t2+gdt*u2,
                              gt*t3+gdt*u3, gt*t4+gdt*u4)
    if HIGHER:
        vr = tl.rsqrt(1.+vx*vx+vy*vy+vz*vz)
        trstd = tl.rsqrt(1.+t0*t0+t1*t1+t2*t2+t3*t3+t4*t4)
        x, y, z = vx*vr, vy*vr, vz*vr
        a, b, c, d, e = t0*trstd, t1*trstd, t2*trstd, t3*trstd, t4*trstd
        tx, ty, tz = _matvec(a, b, c, d, e, x, y, z)
        ttx, tty, ttz = _matvec(a, b, c, d, e, tx, ty, tz)
        q0, q1, q2, q3, q4 = _square_order2(a, b, c, d, e)
        vv0, vv1, vv2, vv3, vv4 = _q(x, y, z)
        tv0, tv1, tv2, tv3, tv4 = _stv(tx, ty, tz, x, y, z)
        g3 = tl.load(GOut+base+S+2*T, valid_t, 0.)
        gvt = tl.load(GOut+base+S+3*T, valid_t, 0.)
        gtt = tl.load(GOut+base+S+4*T, valid_t, 0.)
        gx, gy, gz = 2*(gvt*tx+gtt*ttx), 2*(gvt*ty+gtt*tty), 2*(gvt*tz+gtt*ttz)
        ga, gb, gc, gd, ge = (3*g3*q0+gvt*vv0+2*gtt*tv0, 3*g3*q1+gvt*vv1+2*gtt*tv1,
            3*g3*q2+gvt*vv2+2*gtt*tv2, 3*g3*q3+gvt*vv3+2*gtt*tv3, 3*g3*q4+gvt*vv4+2*gtt*tv4)
        vp, tp = gx*x+gy*y+gz*z, ga*a+gb*b+gc*c+gd*d+ge*e
        dvx, dvy, dvz = dvx+(gx-vp*x)*vr, dvy+(gy-vp*y)*vr, dvz+(gz-vp*z)*vr
        dt0, dt1, dt2, dt3, dt4 = (dt0+(ga-tp*a)*trstd, dt1+(gb-tp*b)*trstd,
            dt2+(gc-tp*c)*trstd, dt3+(gd-tp*d)*trstd, dt4+(ge-tp*e)*trstd)
    vbase, tbase = row[:, None]*3*T+ch[None, :], row[:, None]*5*T+ch[None, :]
    tl.store(DVU+vbase, gdv*vx, valid_t)
    tl.store(DVU+vbase+T, gdv*vy, valid_t)
    tl.store(DVU+vbase+2*T, gdv*vz, valid_t)
    tl.store(DVV+vbase, dvx, valid_t)
    tl.store(DVV+vbase+T, dvy, valid_t)
    tl.store(DVV+vbase+2*T, dvz, valid_t)
    tl.store(DTU+tbase, gdt*t0, valid_t)
    tl.store(DTU+tbase+T, gdt*t1, valid_t)
    tl.store(DTU+tbase+2*T, gdt*t2, valid_t)
    tl.store(DTU+tbase+3*T, gdt*t3, valid_t)
    tl.store(DTU+tbase+4*T, gdt*t4, valid_t)
    tl.store(DTV+tbase, dt0, valid_t)
    tl.store(DTV+tbase+T, dt1, valid_t)
    tl.store(DTV+tbase+2*T, dt2, valid_t)
    tl.store(DTV+tbase+3*T, dt3, valid_t)
    tl.store(DTV+tbase+4*T, dt4, valid_t)


class _FrozenLocalFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scalar, vu, vv, tu, tv, weight, bias, epsilon, higher, atom_tile):
        x = scalar.contiguous()
        n, s = x.shape
        t = vu.shape[-1]
        features = x.new_empty(n, s+(5 if higher else 2)*t)
        dots = x.new_empty(n, 2*t)
        mean, rstd = x.new_empty(n, dtype=torch.float32), x.new_empty(n, dtype=torch.float32)
        strides = tuple(stride for value in (vu, vv, tu, tv) for stride in value.stride())
        if n:
            _features_forward[(tr.cdiv(n, atom_tile),)](x, weight, bias, vu, vv, tu, tv,
                features, dots, mean, rstd, n, s, t, epsilon, higher, *strides,
                atom_tile, tr.next_power_of_2(max(s, t)), num_warps=4, enable_fp_fusion=True)
        ctx.save_for_backward(x, vu, vv, tu, tv, weight, features, mean, rstd)
        ctx.higher, ctx.atom_tile, ctx.strides = higher, atom_tile, strides
        return features, dots

    @staticmethod
    def backward(ctx, gradient, dot_gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Force training needs differentiable reference invariants')
        x, vu, vv, tu, tv, weight, features, mean, rstd = ctx.saved_tensors
        outputs = tuple(value.new_empty(value.shape) for value in (x, vu, vv, tu, tv))
        n, s = x.shape
        t = vu.shape[-1]
        if n:
            _features_backward[(tr.cdiv(n, ctx.atom_tile),)](x, weight, vu, vv, tu, tv,
                features, mean, rstd, gradient.contiguous(), dot_gradient.contiguous(),
                *outputs, n, s, t, ctx.higher, *ctx.strides, ctx.atom_tile,
                tr.next_power_of_2(max(s, t)), num_warps=4, enable_fp_fusion=True)
        return (*outputs, None, None, None, None, None)


def frozen_local_features(scalar, vu, vv, tu, tv, norm, higher=False, atom_tile=4):
    """Return the node-update MLP features and the concatenated vector/tensor closures."""
    if norm.weight.requires_grad or norm.bias.requires_grad:
        raise ValueError('Freeze affine normalization before selecting this backend')
    if atom_tile not in (1, 4, 8, 16):
        raise ValueError('Unsupported atom tile')
    if any(value.device.type != 'cuda' or value.dtype != torch.float32
           for value in (scalar, vu, vv, tu, tv, norm.weight, norm.bias)):
        raise ValueError('Frozen local features require CUDA FP32 inputs')
    if scalar.ndim != 2 or vu.ndim != 3:
        raise ValueError('Expected atom, component and channel dimensions')
    n, s = scalar.shape
    t = vu.shape[-1]
    if (vu.shape != (n, 3, t) or vv.shape != vu.shape or tu.shape != (n, 5, t)
            or tv.shape != tu.shape or norm.weight.shape != (s,) or norm.bias.shape != (s,)):
        raise ValueError('Incompatible local feature dimensions')
    return _FrozenLocalFeatures.apply(scalar, vu, vv, tu, tv, norm.weight, norm.bias,
                                     norm.eps, higher, atom_tile)


def _frozen_node_update(block, scalar, vector, tensor, projections=None):
    """L2Interaction node update with the fused local features.

    Projection, scalar and equivariant state update operations below are the
    L2Interaction operations; only feature construction is substituted.
    """
    from hyff.models.l2_potential import _channel_tensor_vector, _channel_vector_tensor

    if projections is not None:
        vu, vv, tu, tv = projections
    else:
        vector_basis = block.vector_projection_in(vector) if block.tensor_mixing_rank is not None else vector
        tensor_basis = block.tensor_projection_in(tensor) if block.tensor_mixing_rank is not None else tensor
        if hasattr(block, 'joined_vector'):
            vu, vv = block.joined_vector(vector_basis).chunk(2, dim=-1)
            tu, tv = block.joined_tensor(tensor_basis).chunk(2, dim=-1)
        else:
            vu, vv = block.vector_u(vector_basis), block.vector_v(vector_basis)
            tu, tv = block.tensor_u(tensor_basis), block.tensor_v(tensor_basis)
    features, dots = frozen_local_features(scalar, vu, vv, tu, tv, block.update_norm,
        block.higher_invariants, block.frozen_local_feature_atom_tile)
    update = block.update_mlp(features)
    vector_dot, tensor_dot = dots.chunk(2, dim=-1)
    if block.terminal:
        if block.narrow_tensors:
            upd_s, upd_vdot, upd_tdot = update.split(
                (block.scalar_hidden, block.tensor_hidden, block.tensor_hidden), dim=-1)
        else:
            upd_s, upd_vdot, upd_tdot = update.chunk(3, dim=-1)
        scalar = block._scalar_update(scalar, upd_s, upd_vdot*vector_dot, upd_tdot*tensor_dot)
        return scalar, vector, tensor
    if block.narrow_tensors:
        upd_s, upd_vdot, upd_v, upd_tdot, upd_t = update.split(
            (block.scalar_hidden,)+(block.tensor_hidden,)*4, dim=-1)
    else:
        upd_s, upd_vdot, upd_v, upd_tdot, upd_t = update.chunk(5, dim=-1)
    scalar = block._scalar_update(scalar, upd_s, upd_vdot*vector_dot, upd_tdot*tensor_dot)
    if block.local_cross:
        vector_cross = _channel_tensor_vector(tu, vv)
        tensor_cross = _channel_vector_tensor(vu, vv)
        vector = vector + upd_v[:, None, :]*(
            vu+torch.tanh(block.vector_tensor_gain)[None, None, :]*vector_cross)
        tensor = tensor + upd_t[:, None, :]*(
            tu+torch.tanh(block.vector_vector_gain)[None, None, :]*tensor_cross)
    else:
        vector = vector+upd_v[:, None, :]*vu
        tensor = tensor+upd_t[:, None, :]*tu
    return scalar, vector, tensor


def enable_frozen_local_features(model, atom_tile=4):
    """Install the fused local features (an opt-in of the native engines; the plain model never calls this)."""
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze the model before selecting local feature fusion')
    if getattr(model, 'compiled_nodes', False):
        raise ValueError('Select local feature fusion before compiling atom updates')
    if getattr(model, 'frozen_local_features', False):
        raise ValueError('Local feature fusion is already enabled')
    if atom_tile not in (1, 4, 8, 16):
        raise ValueError('Unsupported atom tile')
    for block in model.blocks:
        block.frozen_local_feature_atom_tile = atom_tile
        block._node_update = MethodType(_frozen_node_update, block)
    model.frozen_local_feature_statistics = 'centered_fp32'
    model.frozen_local_features = True
    model.frozen_local_feature_atom_tile = atom_tile
    return model
