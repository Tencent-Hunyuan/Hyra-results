"""The HyCell network: a set-transformer encoder and a count decoder.

A cell enters as a set of tokens. Each expressed gene contributes its gene embedding plus a small network
applied to its log1p(counts per 10,000); measured zeros enter the same way with value 0, and a [CLS] token
leads the set. There is no positional encoding. Six pre-RMSNorm transformer blocks with SwiGLU feed-forward
layers process the set, and the embedding z is a linear projection of the final [CLS] state.

The decoder maps z to a vector h and scores every gene g by h . t_g + b_g, with t_g the gene's input
embedding. During training it also received a learned code for the dataset and donor of each cell; for new
data the learned "unknown dataset and donor" code is used.

Two attention paths compute the same function: ``packed`` runs cells packed into one sequence with
flash-attention's variable-length kernel (CUDA, flash-attn installed), ``padded`` runs a padded batch with
PyTorch's scaled_dot_product_attention (any device).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError:  # optional dependency
    flash_attn_varlen_qkvpacked_func = None


@dataclass(frozen=True)
class HyCellConfig:
    n_genes: int = 36495     # gene vocabulary; token ids >= n_genes are special ([CLS] = n_genes)
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 704
    d_embed: int = 512       # cell embedding z
    n_reserved: int = 64     # unused token ids after the special tokens
    d_value: int = 64        # hidden width of the value network
    d_decoder: int = 1024
    d_code: int = 64         # dataset / donor code width
    n_datasets: int = 555    # training datasets with a learned code
    n_donors: int = 12533    # training donors with a learned code

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def flash_available(device) -> bool:
    return flash_attn_varlen_qkvpacked_func is not None and torch.device(device).type == 'cuda'


class Block(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.h = n_heads
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.w12 = nn.Linear(d, 2 * d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)

    def _ffn(self, x):
        u, v = self.w12(self.n2(x)).chunk(2, dim=-1)
        return x + self.w3(F.silu(u) * v)

    def forward_packed(self, x, cu, max_len):
        """x (tokens, d) of cells packed back to back; cu (cells + 1,) int32 offsets."""
        n, d = x.shape
        qkv = self.qkv(self.n1(x)).view(n, 3, self.h, d // self.h)
        a = flash_attn_varlen_qkvpacked_func(qkv, cu, max_len, 0.0, causal=False)
        return self._ffn(x + self.o(a.reshape(n, d)))

    def forward_padded(self, x, keep):
        """x (cells, length, d); keep (cells, length) marks real tokens."""
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        return self._ffn(x + self.o(a.transpose(1, 2).reshape(b, n, d)))


class HyCellModel(nn.Module):
    def __init__(self, cfg: HyCellConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # encoder
        self.tok = nn.Embedding(cfg.n_genes + 3 + cfg.n_reserved, d)
        self.val = nn.Sequential(nn.Linear(1, cfg.d_value), nn.SiLU(), nn.Linear(cfg.d_value, d))
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(d)
        self.proj = nn.Linear(d, cfg.d_embed)
        # decoder
        self.dec = nn.Sequential(nn.Linear(cfg.d_embed, cfg.d_decoder), nn.SiLU(), nn.Linear(cfg.d_decoder, d))
        self.gene_bias = nn.Parameter(torch.zeros(cfg.n_genes))
        self.log_theta = nn.Parameter(torch.zeros(cfg.n_genes))   # negative-binomial dispersion per gene
        self.ds_emb = nn.Embedding(cfg.n_datasets, cfg.d_code)
        self.dd_emb = nn.Embedding(cfg.n_donors, cfg.d_code)
        self.c_in = nn.Linear(cfg.d_code, cfg.d_decoder, bias=False)
        self.c_out = nn.Linear(cfg.d_code, d, bias=False)
        self.n0 = nn.Parameter(torch.zeros(cfg.d_code))             # code of an unknown dataset and donor

    @property
    def cls_id(self) -> int:
        return self.cfg.n_genes

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ encoder
    def _inputs(self, tok, val):
        tok = tok.long()
        x = self.tok(tok)
        is_gene = (tok < self.cfg.n_genes).unsqueeze(-1)
        return x + self.val(val.unsqueeze(-1).to(x.dtype)) * is_gene

    def encode_packed(self, tok, val, cu, max_len):
        """Embeddings of cells packed back to back (flash-attention). Returns (cells, d_embed)."""
        x = self._inputs(tok, val)
        for blk in self.blocks:
            x = blk.forward_packed(x, cu, max_len)
        return self.proj(self.norm(x)[cu[:-1].long()])

    def encode_padded(self, tok, val, keep):
        """Embeddings of a padded batch whose first token is [CLS]. Returns (cells, d_embed)."""
        x = self._inputs(tok, val)
        for blk in self.blocks:
            x = blk.forward_padded(x, keep)
        return self.proj(self.norm(x)[:, 0])

    def encode_cells(self, cells, attention):
        """Embeddings of a list of (tokens, values) arrays from tokenize_cell: packed back to back for 'flash',
        padded to the longest cell for 'sdpa'. Differentiable; run under autocast for bfloat16."""
        dev = self.tok.weight.device
        lens = np.array([len(t) for t, _ in cells])
        if attention == 'flash':
            tok = torch.from_numpy(np.concatenate([t for t, _ in cells])).to(dev)
            val = torch.from_numpy(np.concatenate([v for _, v in cells])).to(dev)
            cu = torch.from_numpy(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)).to(dev)
            return self.encode_packed(tok, val, cu, int(lens.max()))
        width = int(lens.max())
        tok = np.full((len(cells), width), self.cls_id, np.int32)
        val = np.zeros((len(cells), width), np.float32)
        keep = np.zeros((len(cells), width), bool)
        for r, (t, v) in enumerate(cells):
            tok[r, :len(t)], val[r, :len(t)], keep[r, :len(t)] = t, v, True
        return self.encode_padded(torch.from_numpy(tok).to(dev), torch.from_numpy(val).to(dev),
                                  torch.from_numpy(keep).to(dev))

    # ------------------------------------------------------------------ decoder
    def gene_logits(self, z, code=None, genes=None):
        """Decoder scores (cells, genes) in float32. code: (cells, d_code), default the unknown-dataset code;
        genes: vocabulary ids to score (default all)."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            e = self.n0.float().expand(len(z), -1) if code is None else code.float()
            h = self.dec[2](self.dec[1](self.dec[0](z) + self.c_in(e))) + self.c_out(e)
            t = self.tok.weight[: self.cfg.n_genes].float()
            b = self.gene_bias.float()
            if genes is not None:
                t, b = t[genes], b[genes]
            return h @ t.T + b
