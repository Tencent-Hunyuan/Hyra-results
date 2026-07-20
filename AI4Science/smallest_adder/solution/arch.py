import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_W = None
METADATA = None

VOCAB = 10
D_MODEL = 3
HEAD_DIM = 4
ROPE_THETA = 3.0
RMS_EPS = 1e-6
OUTPUT_LEN = 11
INPUT_LEN = 24
ARC_RADIUS = 160.0
ARC_START = 1.0
ARC_STRIDE = math.pi / 48.0
ATTN_ANGLE = 0.5
DOWN_ANGLE = 1.8


def encode(a, b):
    pa = f"{a:010d}"; pb = f"{b:010d}"
    return ([0] + [int(c) for c in reversed(pa)] + [0, 0]
            + [int(c) for c in reversed(pb)] + [0])


def _rope_tables(head_dim, max_len, theta, device):
    freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_len, device=device, dtype=torch.float32)
    ph = torch.outer(pos, freq)
    return ph.cos(), ph.sin()


def _apply_rope(x, cos, sin):
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0); sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


def _rmsnorm(x, weight):
    scale = x.float().pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
    return (x.float() * scale).to(x.dtype) * weight


class Adder(nn.Module):
    def __init__(self):
        super().__init__()
        W = _W
        self.norm_free = nn.Parameter(torch.tensor(W["norm"]["free"], dtype=torch.float32))
        self.q_free = nn.Parameter(torch.tensor(W["q"]["free"], dtype=torch.float32))
        self.gate_free = nn.Parameter(torch.tensor(W["gate"]["free"], dtype=torch.float32))
        self.up_free = nn.Parameter(torch.tensor(W["up"]["free"], dtype=torch.float32))
        qkf = W["qk"]["free"]
        if len(qkf) > 0:
            self.qk_free = nn.Parameter(torch.tensor(qkf, dtype=torch.float32))
        else:
            self.register_parameter("qk_free", None)
        self._norm_assign = [tuple(a) for a in W["norm"]["assign"]]
        self._norm_shape = tuple(W["norm"]["shape"])
        self._q_assign = [tuple(a) for a in W["q"]["assign"]]
        self._q_shape = tuple(W["q"]["shape"])
        self._gate_assign = [tuple(a) for a in W["gate"]["assign"]]
        self._gate_shape = tuple(W["gate"]["shape"])
        self._up_assign = [tuple(a) for a in W["up"]["assign"]]
        self._up_shape = tuple(W["up"]["shape"])
        self._qk_free_dims = list(W["qk"]["free_dims"])
        self._qk_fix = dict(W["qk"]["fix"])

    @staticmethod
    def _assemble(free, assign, shape):
        n = 1
        for s in shape:
            n = n * s
        flat = free.new_zeros(n)
        for pos, idx in assign:
            flat[pos] = free[idx]
        if len(shape) == 1:
            return flat.view(shape[0])
        return flat.view(shape[0], shape[1])

    def qkn(self):
        parts = []; j = 0
        for d in range(HEAD_DIM):
            if d in self._qk_free_dims:
                parts.append(self.qk_free[j]); j += 1
            elif d in self._qk_fix:
                parts.append(self.norm_free.new_full((), self._qk_fix[d]))
            else:
                parts.append(self.norm_free.new_zeros(()))
        return torch.stack(parts)

    def _emb(self):
        dev = self.norm_free.device; dt = self.norm_free.dtype
        d = torch.arange(VOCAB, device=dev, dtype=dt)
        ang = ARC_START + d * ARC_STRIDE
        c = ARC_RADIUS * torch.cos(ang); s = ARC_RADIUS * torch.sin(ang)
        return torch.stack([c, s, torch.zeros_like(c)], 1)

    def _attn(self, x, mask, cos, sin):
        B, T, _ = x.shape
        qm = self._assemble(self.q_free, self._q_assign, self._q_shape)
        sh = F.linear(x, qm)
        ct, st = math.cos(ATTN_ANGLE), math.sin(ATTN_ANGLE)
        q = sh
        x0, x1 = sh[..., 0::2], sh[..., 1::2]
        k = torch.stack([x0 * ct - x1 * st, x0 * st + x1 * ct], -1).flatten(-2)
        v = sh
        q = q.view(B, T, 1, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, 1, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, 1, HEAD_DIM).transpose(1, 2)
        w = self.qkn()
        q = _rmsnorm(q, w); k = _rmsnorm(k, w)
        q = _apply_rope(q, cos, sin); k = _apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM)) + mask[:T, :T]
        attn = F.softmax(scores, -1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, HEAD_DIM)
        return F.linear(out, qm.t())

    def _mlp(self, x):
        gm = self._assemble(self.gate_free, self._gate_assign, self._gate_shape)
        um = self._assemble(self.up_free, self._up_assign, self._up_shape)
        gate = F.linear(x, gm); up = F.linear(x, um)
        h = F.silu(gate) * up
        ct, st = math.cos(DOWN_ANGLE), math.sin(DOWN_ANGLE)
        h0, h1 = h[..., 0::2], h[..., 1::2]
        h = torch.stack([h0 * ct - h1 * st, h0 * st + h1 * ct], -1).flatten(-2)
        return F.linear(h, um.t())

    def forward(self, tokens):
        table = self._emb()
        x = table[tokens]
        T = tokens.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        cos, sin = _rope_tables(HEAD_DIM, T, ROPE_THETA, x.device)
        nrm = self._assemble(self.norm_free, self._norm_assign, self._norm_shape)
        x = x + self._attn(_rmsnorm(x, nrm), mask, cos, sin)
        x = x + self._mlp(_rmsnorm(x, nrm))
        x = _rmsnorm(x, nrm)
        return F.linear(x, table)


def build_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Adder().to(device)
    model.eval()
    return model, METADATA


def add(model, a, b):
    device = next(model.parameters()).device
    x = torch.tensor([encode(a, b)], dtype=torch.long, device=device)
    digits = []
    with torch.no_grad():
        for _ in range(OUTPUT_LEN):
            nxt = model(x)[0, -1].argmax().item()
            digits.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], 1)
    return sum(d * (10 ** i) for i, d in enumerate(digits))
