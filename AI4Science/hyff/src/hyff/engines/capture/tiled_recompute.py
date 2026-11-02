"""Frozen EFM full-chain edge tiles with a bounded first VJP (memory route for very large systems).

The tiled term has the nine-argument signature of the native prepared EFM term.
Only node projections, compact geometry and indices survive forward; no expanded
edge-chain tape does.
Backward replays one tile and accumulates every node/geometry cotangent. This is
an inference-only choice: first derivatives, frozen parameters, sum/projector
EFM. Selected with capacity_policy tiled/partial (experimental).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _edge_outputs(efm, s, Vs, Ts, rbf, R, D, envelope):
    """Native blocked algebra on already gathered projected source rows."""
    C = efm.channels
    rad = efm.radial(rbf)
    inputs = [s, rad[:, :C]]
    f = tt = None
    if efm.m_blocks:
        f = [R[:, i, 0, None] * Vs[:, 0] + R[:, i, 1, None] * Vs[:, 1]
             + R[:, i, 2, None] * Vs[:, 2] for i in range(3)]
        tt = [sum(D[:, m, k, None] * Ts[:, k] for k in range(5)) for m in range(5)]
        inputs += [f[0], tt[0]]
    s1, a1, c1 = efm.m0(torch.cat(inputs, -1)).split(C, -1)
    gates = torch.sigmoid(efm.gate(s1)).split(C, -1)
    h, a, c = F.silu(s1), a1 * gates[0], c1 * gates[1]
    X1 = Z = None
    if 1 in efm.m_blocks:
        X = torch.stack((torch.cat((f[1], tt[1]), -1),
                         torch.cat((f[2], tt[2]), -1)), 1) * rad[:, None, C:3*C]
        X1 = efm.m1_a(X) * torch.cat((gates[2], gates[3]), -1)[:, None, :]
    if 2 in efm.m_blocks:
        Z = efm.m2_a(torch.stack((tt[3], tt[4]), 1) * rad[:, None, 3*C:4*C]) * gates[4][:, None, :]
    for layer in getattr(efm, 'layers', ()):
        s2, a2, c2 = layer.m0(torch.cat((h, a, c), -1)).split(C, -1)
        g = torch.sigmoid(layer.gate(s2)).split(C, -1)
        h, a, c = h + F.silu(s2), a + a2*g[0], c + c2*g[1]
        if X1 is not None:
            X1 = X1 + layer.m1(X1) * torch.cat((g[2], g[3]), -1)[:, None, :]
        if Z is not None:
            Z = Z + layer.m2(Z) * g[4][:, None, :]
    fv = [a] + ([X1[:, 0, :C], X1[:, 1, :C]] if X1 is not None else [])
    ft = [c] + ([X1[:, 0, C:], X1[:, 1, C:]] if X1 is not None else []) + ([Z[:, 0], Z[:, 1]] if Z is not None else [])
    v = torch.stack([sum(R[:, i, j, None]*fv[i] for i in range(len(fv))) for j in range(3)], 1)
    t = torch.stack([sum(D[:, m, k, None]*ft[m] for m in range(len(ft))) for k in range(5)], 1)
    w = efm.message_scale * envelope
    return h*w[:, None], v*w[:, None, None], t*w[:, None, None]


def make_tiled_term(efm, tile_edges=32768, compile_tiles=False):
    """Return a tiled EFM callable with the native nine-argument signature (first VJP only).

    Requested compilation applies to non-scalar-source chains only: the torch
    2.7.1 compiled backward of the scalar-source chain is wrong (see efm_engine),
    so that chain stays eager.
    """
    if type(tile_edges) is not int or tile_edges < 1:
        raise ValueError('tile_edges must be a positive integer')
    if any(p.requires_grad for p in efm.parameters()):
        raise ValueError('Frozen EFM parameters required')
    if (efm.aggregation != 'sum' or efm.frame != 'projector' or efm.complex_structure
            or getattr(efm, 'receiver_in', None) is not None):
        raise ValueError('Only sum/projector real EFM without receiver input supported')

    compile_requested = bool(compile_tiles)
    compile_tiles = compile_requested and not bool(efm.input_is_scalar)

    def edge_outputs(*args):
        return _edge_outputs(efm, *args)

    edge_function = (torch.compile(edge_outputs, fullgraph=True, dynamic=False,
                                  options={'triton.cudagraphs': False})
                     if compile_tiles else edge_outputs)
    has_m = bool(efm.m_blocks)

    class TiledAggregate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, S, V, T, rbf, R, D, envelope, src, dst):
            ctx.save_for_backward(S, V, T, rbf, R, D, envelope, src, dst)
            n, C = S.shape
            aggregate = (S.new_zeros((n, C)), S.new_zeros((n, 3, C)), S.new_zeros((n, 5, C)))
            for begin in range(0, src.numel(), tile_edges):
                part = slice(begin, begin + tile_edges)
                source, destination = src[part], dst[part]
                local = (S.index_select(0, source),
                         V.index_select(0, source) if has_m else V,
                         T.index_select(0, source) if has_m else T,
                         rbf[part], R[part], D[part], envelope[part])
                outputs = edge_function(*local)
                for total, edge in zip(aggregate, outputs):
                    total.index_add_(0, destination, edge)
            return aggregate

        @staticmethod
        def backward(ctx, grad_s, grad_v, grad_t):
            if torch.is_grad_enabled():
                raise RuntimeError('Tiled frozen EFM supports first derivatives only')
            saved = ctx.saved_tensors
            S, V, T, rbf, R, D, envelope, src, dst = saved
            needs = ctx.needs_input_grad[:7]
            gradients = [torch.zeros_like(value) if needed else None
                         for value, needed in zip(saved[:7], needs)]
            for begin in range(0, src.numel(), tile_edges):
                part = slice(begin, begin + tile_edges)
                source, destination = src[part], dst[part]
                gathered = (S.index_select(0, source),
                            V.index_select(0, source) if has_m else V,
                            T.index_select(0, source) if has_m else T,
                            rbf[part], R[part], D[part], envelope[part])
                # Explicit detached tile leaves prevent replay from entering model,
                # electronic-embedding or latent-charge hooks, while returning all upstream cotangents.
                with torch.enable_grad():
                    leaves = tuple(value.detach().requires_grad_(True) for value in gathered)
                    outputs = edge_function(*leaves)
                    cotangents = tuple(g.index_select(0, destination) for g in (grad_s, grad_v, grad_t))
                    local_grad = torch.autograd.grad(outputs, leaves, cotangents,
                                                     allow_unused=True, create_graph=False)
                for i, (total, gradient) in enumerate(zip(gradients, local_grad)):
                    if total is None or gradient is None:
                        continue
                    if i < 3:
                        total.index_add_(0, source, gradient)
                    else:
                        total[part].copy_(gradient)
            return (*gradients, None, None)

    def term(normalized, vector, tensor, rbf, R, D, envelope, src, dst):
        S = efm.node_scalar(normalized)
        V = efm.node_vector(vector) if has_m else S.new_empty((0,))
        T = efm.node_tensor(tensor) if has_m else S.new_empty((0,))
        aggregate = TiledAggregate.apply(S, V, T, rbf, R, D, envelope, src, dst)
        return (efm.exit_scalar(aggregate[0]), efm.exit_vector(aggregate[1]),
                efm.exit_tensor(aggregate[2]))

    term.tiled_efm_metadata = dict(tile_edges=tile_edges, compile_tiles=bool(compile_tiles),
        compile_tiles_requested=compile_requested,
        compile_fallback=('known_scalar_source_backward_defect'
                          if compile_requested and not compile_tiles else None),
        frozen=True, first_derivative_only=True, full_chain_recompute=True,
        expanded_all_edge_tape_retained=False)
    return term


def install_tiled_terms(state, tile_edges=32768, compile_tiles=True):
    """Replace the EFM terms of one state instance; the replaced callables are kept in ``_tiled_efm_original_terms``."""
    if getattr(state, '_tiled_efm_original_terms', None) is not None:
        raise ValueError('Tiled terms already installed')
    replacements = {index: make_tiled_term(efm, tile_edges, compile_tiles)
                    for index, efm in state.efms.items()}
    state._tiled_efm_original_terms = dict(state.compiled_term)
    state._tiled_efm_original_backends = dict(state.term_backend)
    state.compiled_term.update(replacements)
    for index in replacements:
        state.term_backend[index] = 'EXPERIMENTAL_full_chain_tiled_recompute'
    return dict(tile_edges=tile_edges, compile_tiles=bool(compile_tiles),
                blocks=sorted(replacements),
                block_execution={index: function.tiled_efm_metadata
                                 for index, function in replacements.items()})
