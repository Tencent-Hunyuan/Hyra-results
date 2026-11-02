"""Edge-frame mixing (EFM) blocks inside the captured CUDA-graph force path.

Where the EFM enters. In the EFM model's block forward the EFM exits are added
to the node features after the message aggregation and before the node update
(s, v, T = s + A_s + ds, v + A_v + dv, T + A_T + dt; then _node_update). The EFM
is therefore not an additive energy term; it changes the features that every
later block and readout sees. On the fused CSR path the same boundary is
L2Interaction._finish_messages (called by the CSR forward with the aggregates),
so this module

  * wraps the model forward to stash the batch / CSR plan and to clear a per-step geometry cache,
  * wraps each EFM block's forward: LN(s) -> EFM term on the EFM edge set (blocked compiled core) -> stash ->
    original forward (CSR Triton messages) with the node message passed in,
  * replaces each EFM block's _finish_messages by a compiled function that receives the stashed (ds, dv, dt)
    as explicit arguments and adds them at the same position as the eager model. For the folded first block
    the exits are added to the folded state and their projections joined_vector(dv) / joined_tensor(dt) to
    the folded projections; algebraically the eager order.

Edge set. Each EFM block runs on the edges inside its EFM radius (the layer cutoff or the per-block
efm_edge_cutoff). The CSR path refreshes a radius view of the Verlet matrix at the current coordinates on
every energy evaluation (hyff.kernels.fused_messages._radius_select writes the kept neighbours compactly into
the first count slots), so the EFM edge list is a fixed-capacity compaction of the first `slots` columns of
that view (cumsum + scatter into an edge buffer of capacity E_cap = reserve x initial count; unused entries
are zero-weight self edges spread over atoms). Everything is static-shaped and runs inside the captured CUDA
graph; a device-side flag records any overflow of E_cap or of the slot window and is read on the host after
every chunk (never silently truncated). In the eager path (csr None: the reference evaluation and the
endpoint checks) the edges inside the radius are selected with nonzero() and padded to the same capacity
(or a 4096 bucket).

Arithmetic. The base path is unchanged (tf32x3 or fp32). The EFM GEMMs are cuBLAS TF32 when
efm_precision='tf32': torch.backends.cuda.matmul.allow_tf32 stays False, so the flag is raised only around
the compiled EFM forward call and, for the backward, by pre/post hooks on the compiled function's single
CompiledFunctionBackward node; the flag is a host-side cuBLAS setting read at launch time, so the captured
graph replays exactly the recorded TF32 kernels while every base-path cuBLAS call stays FP32.
efm_precision='fp32' leaves the flag alone. The andante and largo engines use fp32.
"""
from __future__ import annotations

import math

import torch

from hyff.kernels.efm_blocked import blocked_core
from hyff.kernels.efm_rotated import frame_matrices
from hyff.models.l2_potential import smooth_switch

COMPILE_OPTIONS = dict(fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
EFM_PRECISIONS = ('tf32', 'fp32')


def set_tf32(value):
    torch.backends.cuda.matmul.allow_tf32 = bool(value)


# ---------------------------------------------------------------------------------------------------------------------
# compiled pieces
# ---------------------------------------------------------------------------------------------------------------------
def _geometry(positions, src, dst, translation, centers, gamma, rbf_cutoff: float, radius: float, width: float, padding):
    """Per-edge geometry of a padded edge list: rbf features, frame matrices R (E,3,3) / D (E,5,5), r_e envelope."""
    displacement = positions.index_select(0, src) - positions.index_select(0, dst) + translation
    valid = (src != dst) | (translation != 0).any(-1)
    displacement = torch.where(valid[:, None], displacement, padding)
    distance = torch.linalg.vector_norm(displacement, dim=-1).clamp_min(1.0e-9)
    direction = displacement / distance[:, None]
    delta = distance[:, None] - centers[None, :]
    basis = torch.exp(-gamma * delta.square())
    rbf_envelope = 0.5 * (torch.cos(math.pi * distance / rbf_cutoff) + 1.0) * (distance < rbf_cutoff).to(distance.dtype)
    rbf = basis * rbf_envelope[:, None]
    envelope = smooth_switch(distance, radius, width) * valid.to(distance.dtype)
    R, D = frame_matrices(direction)
    return rbf, R, D, envelope


def _efm_term(efm, normalized, vector, tensor, rbf, R, D, envelope, src, dst):
    """Node projections, radial MLP, blocked per-edge chain, sum aggregation and the zero-initialised exits."""
    S = efm.node_scalar(normalized)
    V = T = None
    if efm.m_blocks:
        V = efm.node_vector(vector)
        T = efm.node_tensor(tensor)
    rad = efm.radial(rbf)
    w = efm.message_scale * envelope
    agg_s, agg_v, agg_t = blocked_core(efm, S, V, T, rad, R, D, w, src, dst, normalized.shape[0])
    return efm.exit_scalar(agg_s), efm.exit_vector(agg_v), efm.exit_tensor(agg_t)


class _PrecisionLinear(torch.autograd.Function):
    """F.linear with the cuBLAS TF32 flag raised around its own forward and backward products (frozen weight)."""

    @staticmethod
    def forward(ctx, x, weight, bias, tf32):
        ctx.save_for_backward(weight)
        ctx.tf32 = tf32
        previous = torch.backends.cuda.matmul.allow_tf32
        set_tf32(tf32 or previous)
        try:
            return torch.nn.functional.linear(x, weight, bias)
        finally:
            set_tf32(previous)

    @staticmethod
    def backward(ctx, gradient):
        (weight,) = ctx.saved_tensors
        previous = torch.backends.cuda.matmul.allow_tf32
        set_tf32(ctx.tf32 or previous)
        try:
            dx = gradient @ weight
        finally:
            set_tf32(previous)
        return dx, None, None, None


def _lin(layer, x, tf32):
    return _PrecisionLinear.apply(x, layer.weight, layer.bias, tf32)


def _chain(efm, normalized, vector, tensor, rbf, R, D, src, tf32):
    """The per-edge chain of blocked_core up to the last stacked layer, with explicit-precision linears (eager use)."""
    C = efm.channels
    S = _lin(efm.node_scalar, normalized, tf32)
    s = S.index_select(0, src)
    rad = _lin(efm.radial[2], torch.nn.functional.silu(_lin(efm.radial[0], rbf, tf32)), tf32)
    inputs = [s, rad[:, :C]]
    f = tt = None
    if efm.m_blocks:
        Vs = _lin(efm.node_vector, vector, tf32).index_select(0, src)
        Ts = _lin(efm.node_tensor, tensor, tf32).index_select(0, src)
        f = [R[:, i, 0, None] * Vs[:, 0] + R[:, i, 1, None] * Vs[:, 1] + R[:, i, 2, None] * Vs[:, 2] for i in range(3)]
        tt = [sum(D[:, m, k, None] * Ts[:, k] for k in range(5)) for m in range(5)]
        inputs += [f[0], tt[0]]
    s1, a1, c1 = _lin(efm.m0, torch.cat(inputs, -1), tf32).split(C, -1)
    gates = torch.sigmoid(_lin(efm.gate, s1, tf32)).split(C, -1)
    h = torch.nn.functional.silu(s1)
    a = a1 * gates[0]
    c = c1 * gates[1]
    X1 = Z = None
    if 1 in efm.m_blocks:
        X = torch.stack((torch.cat((f[1], tt[1]), -1), torch.cat((f[2], tt[2]), -1)), 1) * rad[:, None, C:3 * C]
        X1 = _lin(efm.m1_a, X, tf32) * torch.cat((gates[2], gates[3]), -1)[:, None, :]
    if 2 in efm.m_blocks:
        Z = _lin(efm.m2_a, torch.stack((tt[3], tt[4]), 1) * rad[:, None, 3 * C:4 * C], tf32) * gates[4][:, None, :]
    for layer in getattr(efm, 'layers', ()):
        s2, a2, c2 = _lin(layer.m0, torch.cat((h, a, c), -1), tf32).split(C, -1)
        g = torch.sigmoid(_lin(layer.gate, s2, tf32)).split(C, -1)
        h = h + torch.nn.functional.silu(s2)
        a = a + a2 * g[0]
        c = c + c2 * g[1]
        if X1 is not None:
            X1 = X1 + _lin(layer.m1, X1, tf32) * torch.cat((g[2], g[3]), -1)[:, None, :]
        if Z is not None:
            Z = Z + _lin(layer.m2, Z, tf32) * g[4][:, None, :]
    return h, a, c, X1, Z


def _aggregate(efm, h, a, c, X1, Z, R, D, envelope, dst, n_atoms):
    """Rotate back, sum-aggregate and apply the exits (the tail of blocked_core; compiled)."""
    C = efm.channels
    w = efm.message_scale * envelope
    fv = [a] + ([X1[:, 0, :C], X1[:, 1, :C]] if X1 is not None else [])
    ft = [c] + ([X1[:, 0, C:], X1[:, 1, C:]] if X1 is not None else []) + ([Z[:, 0], Z[:, 1]] if Z is not None else [])
    v_e = torch.stack([sum(R[:, i, j, None] * fv[i] for i in range(len(fv))) for j in range(3)], 1)
    t_e = torch.stack([sum(D[:, m, k, None] * ft[m] for m in range(len(ft))) for k in range(5)], 1)
    agg_s = h.new_zeros((n_atoms, C)).index_add_(0, dst, h * w[:, None])
    agg_v = h.new_zeros((n_atoms, 3, C)).index_add_(0, dst, v_e * w[:, None, None])
    agg_t = h.new_zeros((n_atoms, 5, C)).index_add_(0, dst, t_e * w[:, None, None])
    return efm.exit_scalar(agg_s), efm.exit_vector(agg_v), efm.exit_tensor(agg_t)


def _finish_with_efm(block, scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t, ds, dv, dt):
    """L2Interaction._finish_messages with the EFM exits added at the eager-model position."""
    if block.bottleneck:
        aggregate_s = block.edge_scalar_out(aggregate_s)
    if hasattr(block, 'folded_first_vector'):
        vector, vu, vv = block.folded_first_vector(aggregate_v).chunk(3, dim=-1)
        tensor, tu, tv = block.folded_first_tensor(aggregate_t).chunk(3, dim=-1)
        dvu, dvv = block.joined_vector(dv).chunk(2, dim=-1)
        dtu, dtv = block.joined_tensor(dt).chunk(2, dim=-1)
        return block._node_update((scalar + aggregate_s) + ds, vector + dv, tensor + dt, (vu + dvu, vv + dvv, tu + dtu, tv + dtv))
    if block.tensor_bottleneck:
        aggregate_v = block.edge_vector_out(aggregate_v)
        aggregate_t = block.edge_tensor_out(aggregate_t)
    return block._node_update((scalar + aggregate_s) + ds, (vector + aggregate_v) + dv, (tensor + aggregate_t) + dt)


# ---------------------------------------------------------------------------------------------------------------------
# per-model state
# ---------------------------------------------------------------------------------------------------------------------
class EFMState:
    def __init__(self, model, efms, info, *, efm_precision, slots, edge_reserve, device):
        if efm_precision not in EFM_PRECISIONS:
            raise ValueError(f'efm_precision must be one of {EFM_PRECISIONS}')
        self.model, self.efms, self.info = model, efms, info
        self.efm_precision, self.slots, self.reserve = efm_precision, int(slots), float(edge_reserve)
        self.device = torch.device(device)
        self.enabled = True
        self.batch = self.csr = None
        self.geometry = {}
        self.term = {}
        self.capacity = None
        self.flags = torch.zeros(2, dtype=torch.int32, device=self.device)   # [E_cap overflow, slot overflow], max over steps
        self.centers = model.rbf.centers.detach()
        self.gamma = model.rbf.gamma.detach()
        self.rbf_cutoff = float(model.rbf.cutoff)
        self.width = float(info['switch_width'])
        self.padding = model.padding_displacement.detach()
        self.pad_cache = {}
        self.stats = dict(csr_geometry_calls=0, eager_geometry_calls=0, buckets=[], backward_nodes={}, initial_edges=None,
                          initial_max_count=None, tf32_windows=0)
        self.compiled_geometry = torch.compile(_geometry, **COMPILE_OPTIONS)
        self.compiled_term, self.term_backend = {}, {}
        for index, efm in efms.items():
            # The Inductor backward of the scalar-source (block 0) edge chain is wrong on torch 2.7.1 (forward exact,
            # position/feature gradients off by O(1); the aggregation compiles correctly, aot_eager and eager are
            # exact). Every block's compiled forward AND backward are therefore checked against the eager blocked
            # core on a random problem before use, in this order: Inductor single graph, eager chain with
            # explicit-precision linears + Inductor aggregation, fully eager.
            self.compiled_term[index], self.term_backend[index] = self._checked_compile(index, efm)
        self.efm_finish, self.base_finish, self.original_forward = {}, {}, {}

    # -- compiled-term self-check -----------------------------------------------------------------------------------------
    def _check_problem(self, efm, seed=11):
        """A small random periodic problem in the padded-edge layout (positions require grad)."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        n, side = 48, 9.5
        pos = torch.rand(n, 3, device=self.device, generator=g) * side
        i, j = torch.meshgrid(torch.arange(n, device=self.device), torch.arange(n, device=self.device), indexing='ij')
        src_all, dst_all = j.reshape(-1), i.reshape(-1)
        d = pos[src_all] - pos[dst_all]
        shift = -side * torch.round(d / side)
        keep = ((d + shift).norm(dim=-1) < 3.4) & (src_all != dst_all)
        src, dst, trans = src_all[keep], dst_all[keep], shift[keep]
        E = int(src.numel()) + 9
        pad = torch.arange(E, device=self.device) % n
        s2 = pad.clone(); s2[:src.numel()] = src
        d2 = pad.clone(); d2[:src.numel()] = dst
        t2 = torch.zeros(E, 3, device=self.device); t2[:src.numel()] = trans
        Hs = efm.node_scalar.in_features
        Ht = efm.node_vector.in_features if hasattr(efm, 'node_vector') else Hs
        normalized = torch.randn(n, Hs, device=self.device, generator=g)
        vector = torch.randn(n, 3, Ht, device=self.device, generator=g)
        tensor = torch.randn(n, 5, Ht, device=self.device, generator=g)
        return dict(pos=pos, src=s2, dst=d2, trans=t2, normalized=normalized, vector=vector, tensor=tensor, generator=g)

    def _run_check(self, term_fn, P, weights, tf32):
        previous = torch.backends.cuda.matmul.allow_tf32
        set_tf32(tf32)
        try:
            leaves = [P['pos'].clone().requires_grad_(True), P['normalized'].clone().requires_grad_(True),
                      P['vector'].clone().requires_grad_(True), P['tensor'].clone().requires_grad_(True)]
            rbf, R, D, env = _geometry(leaves[0], P['src'], P['dst'], P['trans'], self.centers, self.gamma, self.rbf_cutoff, 3.5,
                                       self.width, self.padding)
            out = term_fn(leaves[1], leaves[2], leaves[3], rbf, R, D, env, P['src'], P['dst'])
            if weights is None:
                weights = [torch.randn(o.shape, device=self.device, generator=P['generator']) for o in out]
            loss = sum((o * w).sum() for o, w in zip(out, weights))
            grads = torch.autograd.grad(loss, leaves, allow_unused=True)
        finally:
            set_tf32(previous)
        return [o.detach() for o in out], [None if x is None else x.detach() for x in grads], weights

    def _term_choices(self, efm):
        """(name, factory(tf32) -> term(normalized, vector, tensor, rbf, R, D, envelope, src, dst)) in preference order."""
        def single(normalized, vector, tensor, rbf, R, D, envelope, src, dst):
            return _efm_term(efm, normalized, vector, tensor, rbf, R, D, envelope, src, dst)

        def aggregate(h, a, c, X1, Z, R, D, envelope, dst, n):
            return _aggregate(efm, h, a, c, X1, Z, R, D, envelope, dst, n)

        def split_factory(tf32, aggregate_fn):
            def term(normalized, vector, tensor, rbf, R, D, envelope, src, dst):
                h, a, c, X1, Z = _chain(efm, normalized, vector, tensor, rbf, R, D, src, tf32)
                return aggregate_fn(h, a, c, X1, Z, R, D, envelope, dst, normalized.shape[0])
            return term

        compiled_single = torch.compile(single, **COMPILE_OPTIONS)
        compiled_aggregate = torch.compile(aggregate, **COMPILE_OPTIONS)
        return [('inductor', lambda tf32: compiled_single),
                ('eager_chain_explicit_precision+inductor_aggregate', lambda tf32: split_factory(tf32, compiled_aggregate)),
                ('eager_explicit_precision', lambda tf32: split_factory(tf32, aggregate))], single

    def _checked_compile(self, index, efm):
        """First choice whose forward and input gradients reproduce the eager blocked core (at the engine's precision)."""
        tf32 = self.efm_precision == 'tf32'
        tolerance = 2e-2 if tf32 else 1e-3      # TF32 GEMMs move results by ~1e-3 relative; the Inductor bug is O(1)
        P = self._check_problem(efm)
        choices, reference_fn = self._term_choices(efm)
        ref_out, ref_grad, weights = self._run_check(reference_fn, P, None, False)
        report = {}
        names = ('positions', 'normalized', 'vector', 'tensor')
        for name, factory in choices:
            term = factory(tf32)
            out, grad, _ = self._run_check(term, P, weights, tf32)
            worst = 0.
            for a, b in zip(out, ref_out):
                worst = max(worst, float((a - b).abs().max()) / max(float(b.abs().max()), 1e-30))
            grad_errors = {}
            for leaf, a, b in zip(names, grad, ref_grad):
                if b is None or a is None:
                    grad_errors[leaf] = None if (a is None and b is None) else 'mismatch'
                    if grad_errors[leaf] == 'mismatch':
                        worst = float('inf')
                    continue
                grad_errors[leaf] = float((a - b).abs().max()) / max(float(b.abs().max()), 1e-30)
                worst = max(worst, grad_errors[leaf])
            report[name] = dict(worst_relative_error=worst, grad_relative_errors=grad_errors)
            if worst < tolerance:
                self.stats.setdefault('term_self_check', {})[index] = dict(selected=name, tolerance=tolerance, checked_in_tf32=tf32, choices=report)
                return term, name
        raise RuntimeError(f'block {index}: no EFM term implementation reproduces the eager blocked core: {report}')

    # -- per forward ---------------------------------------------------------------------------------------------------
    def begin(self, batch, csr):
        self.batch, self.csr = batch, csr
        self.geometry = {}

    def end(self):
        self.batch = self.csr = None
        self.geometry = {}
        self.term = {}

    def pad_nodes(self, length, n_atoms):
        key = (length, n_atoms)
        if key not in self.pad_cache:
            self.pad_cache[key] = torch.arange(length, device=self.device, dtype=torch.int64) % n_atoms
        return self.pad_cache[key]

    def set_capacity(self, count):
        self.capacity = max(1024, int(math.ceil(count * self.reserve / 1024.)) * 1024)
        self.stats['initial_edges'] = int(count)

    def geometry_for(self, radius, positions):
        if radius not in self.geometry:
            if self.csr is not None:
                self.geometry[radius] = self._csr_geometry(radius, positions)
            else:
                self.geometry[radius] = self._eager_geometry(radius)
        return self.geometry[radius]

    def _csr_geometry(self, radius, positions):
        parent = self.csr
        view = parent.radius_view(radius)
        # the view has the parent's capacity as columns; a small system can have fewer than `slots` columns, and the
        # flattened slot index below must use the real column count
        n, columns = view.neighbors.shape
        K = min(self.slots, columns)
        neighbors = view.neighbors[:, :K]
        shifts = view.shifts[:, :K]
        mask = (neighbors >= 0).reshape(-1)
        rank = torch.cumsum(mask.to(torch.int64), 0) - 1
        total = rank[-1] + 1
        if self.capacity is None or self.stats['initial_max_count'] is None:
            # first CSR evaluation: happens in the engine's uncaptured warm-up, so a host read is allowed here
            torch.cuda.synchronize()
            if self.capacity is None:
                self.set_capacity(int(total))
            self.stats['initial_max_count'] = int(view.counts.max())
            self.stats['initial_csr_edges'] = int(total)
            if self.stats['initial_max_count'] > K:
                raise RuntimeError(f'EFM slot window {K} below the initial neighbour count {self.stats["initial_max_count"]}')
        E = self.capacity
        # masked-out slots and any edges beyond the capacity go to the trash slot E (a capacity overflow therefore
        # gives a wrong but crash-free step; flags[0] below records it and the driver fails the run at the chunk end)
        position = torch.where(mask, rank.clamp_max(E), torch.full_like(rank, E))
        pad = self.pad_nodes(E + 1, n)
        src = pad.clone().scatter_(0, position, neighbors.reshape(-1).clamp_min(0))
        dst_index = torch.arange(n, device=self.device, dtype=torch.int64)[:, None].expand(n, K).reshape(-1)
        dst = pad.clone().scatter_(0, position, dst_index)
        shift = torch.zeros((E + 1, 3), dtype=shifts.dtype, device=self.device).scatter_(0, position[:, None].expand(-1, 3), shifts.reshape(-1, 3))
        self.flags[0].copy_(torch.maximum(self.flags[0], (total - E).to(torch.int32)))
        self.flags[1].copy_(torch.maximum(self.flags[1], view.counts.max() - K))
        src, dst, shift = src[:E], dst[:E], shift[:E]
        translation = shift.to(positions.dtype) @ parent.cell.to(positions.dtype)
        self.stats['csr_geometry_calls'] += 1
        return self._finish_geometry(positions, src, dst, translation, radius)

    def _eager_geometry(self, radius):
        batch = self.batch
        positions = batch['positions']
        src_all, dst_all = batch['edge_index'][0], batch['edge_index'][1]
        translation_all = batch.get('shifts')
        with torch.no_grad():
            displacement = positions[src_all] - positions[dst_all]
            if translation_all is not None:
                displacement = displacement + translation_all
            valid = batch['edge_valid'] if 'edge_valid' in batch else (src_all != dst_all)
            if 'edge_valid' not in batch and translation_all is not None:
                valid = valid | (translation_all != 0).any(-1)
            distance = torch.linalg.vector_norm(torch.where(valid[:, None], displacement, self.padding), dim=-1)
            active = torch.nonzero(valid & (distance < radius), as_tuple=False).squeeze(-1)
        count = int(active.numel())
        if self.capacity is None:
            self.set_capacity(count)
        E = self.capacity if count <= self.capacity else int(math.ceil(count / 4096.)) * 4096
        if E not in self.stats['buckets']:
            self.stats['buckets'].append(E)
        n = positions.shape[0]
        pad = self.pad_nodes(E, n)
        src = pad.clone()
        src[:count] = src_all.index_select(0, active)
        dst = pad.clone()
        dst[:count] = dst_all.index_select(0, active)
        translation = positions.new_zeros((E, 3))
        if translation_all is not None:
            translation = torch.cat((translation_all.index_select(0, active).to(positions.dtype), positions.new_zeros((E - count, 3))), 0)
        self.stats['eager_geometry_calls'] += 1
        return self._finish_geometry(positions, src, dst, translation, radius)

    def _finish_geometry(self, positions, src, dst, translation, radius):
        rbf, R, D, envelope = self.compiled_geometry(positions, src, dst, translation, self.centers, self.gamma,
                                                    self.rbf_cutoff, float(radius), self.width, self.padding)
        return dict(rbf=rbf, R=R, D=D, envelope=envelope, src=src, dst=dst)

    # -- the EFM term of one block ---------------------------------------------------------------------------------------
    def run_term(self, index, normalized, vector, tensor, geometry):
        fn = self.compiled_term[index]
        args = (normalized, vector, tensor, geometry['rbf'], geometry['R'], geometry['D'], geometry['envelope'], geometry['src'], geometry['dst'])
        if self.efm_precision != 'tf32':
            return fn(*args)
        previous = torch.backends.cuda.matmul.allow_tf32
        set_tf32(True)
        try:
            out = fn(*args)
        finally:
            set_tf32(previous)
        self.stats['tf32_windows'] += 1
        nodes = []
        for value in out:
            node = value.grad_fn
            if node is not None and all(node is not other for other in nodes):
                nodes.append(node)
        names = [node.name() for node in nodes]
        self.stats['backward_nodes'][index] = names
        for node in nodes:
            node.register_prehook(lambda grad_outputs: set_tf32(True))
            node.register_hook(lambda grad_inputs, grad_outputs, previous=previous: set_tf32(previous))
        return out

    def check_overflow(self):
        flags = self.flags.tolist()
        if flags[0] > 0 or flags[1] > 0:
            raise RuntimeError(f'EFM edge buffer overflow: capacity exceeded by {flags[0]} edges, slot window by {flags[1]}')
        return flags

    def report(self):
        return dict(efm_precision=self.efm_precision, enabled=self.enabled, slots=self.slots, edge_reserve=self.reserve,
                    capacity=self.capacity, blocks=self.info['blocks'], radii_A=self.info['radii_A'],
                    channels=self.info['channels'], layers=self.info['layers'], m_blocks=self.info['m_blocks'],
                    parameters=self.info['parameters'], exit_weight_rms=self.info['exit_weight_rms'],
                    term_backend=dict(self.term_backend),
                    edge_set='fixed-capacity compaction of the first `slots` columns of the 3.5 A radius view (separate '
                             'static edge buffer rebuilt every step inside the captured graph), not a masked view',
                    overflow_flags=self.flags.tolist(), **self.stats)


# ---------------------------------------------------------------------------------------------------------------------
# instance patches
# ---------------------------------------------------------------------------------------------------------------------
def _model_forward_with_efm(self, batch, *args, **kwargs):
    state = self.__dict__['_efm_state']
    state.begin(batch, args[0] if args else kwargs.get('csr'))
    try:
        return state.inner_forward(batch, *args, **kwargs)
    finally:
        state.end()


def _block_forward_with_efm(self, scalar, vector, tensor, radial, direction, quadrupole, source, destination, edge_envelope,
                            radial_output=None, csr=None, reference=None, node_message=None):
    state = self.__dict__['_efm_state']
    index = self.__dict__['_efm_index']
    original = state.original_forward[index]
    if not state.enabled:
        return original(scalar, vector, tensor, radial, direction, quadrupole, source, destination, edge_envelope,
                        radial_output, csr, reference, node_message)
    normalized = self.scalar_norm(scalar)
    if node_message is None:
        node_message = self.scalar_message(normalized)
    positions = csr[0] if csr is not None else state.batch['positions']
    geometry = state.geometry_for(state.info['radii_A'][index], positions)
    state.term[index] = state.run_term(index, normalized, vector, tensor, geometry)
    try:
        return original(scalar, vector, tensor, radial, direction, quadrupole, source, destination, edge_envelope,
                        radial_output, csr, reference, node_message)
    finally:
        state.term.pop(index, None)


def _finish_dispatch(self, scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t):
    state = self.__dict__['_efm_state']
    index = self.__dict__['_efm_index']
    term = state.term.get(index)
    if term is None:
        return state.base_finish[index](scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t)
    return state.efm_finish[index](scalar, vector, tensor, aggregate_s, aggregate_v, aggregate_t, *term)
