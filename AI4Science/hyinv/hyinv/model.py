"""HyInv backbone graph model: 4-layer encoder + 3-layer decoder, width 128, 48 neighbors per residue,
1,957,573 parameters in total.

The modules follow ProteinMPNN (MIT license, Dauparas et al.), keeping only what inference needs:
distance features on edges are computed only for the selected neighbors, and residues with missing
atoms are excluded from every neighbor list.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

AA = "ACDEFGHIKLMNPQRSTVWY"
MASK = 20  # "unknown" token in the decoding state


@dataclass
class Backbone:
    """B copies of the same backbone (one per design). coords [B,L,4,3] (N, CA, C, O), atom_mask [B,L,4]."""
    coords: Tensor
    atom_mask: Tensor
    residue_index: Tensor   # [B,L] long
    chain_id: Tensor        # [B,L] long
    slot_index: Tensor      # [B,L] long

    @property
    def geometry_valid(self) -> Tensor:
        return (self.atom_mask & torch.isfinite(self.coords).all(-1)).all(-1)

    @property
    def safe_coords(self) -> Tensor:
        return torch.where(self.geometry_valid[..., None, None], self.coords, 0.0)


def gather_nodes(nodes: Tensor, idx: Tensor) -> Tensor:
    """[B,N,C] gathered at the neighbor indices [B,N,K] -> [B,N,K,C]."""
    flat = idx.reshape(idx.shape[0], -1).unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, flat).reshape(*idx.shape, -1)


def cat_neighbors_nodes(h_nodes: Tensor, h_neighbors: Tensor, idx: Tensor) -> Tensor:
    return torch.cat([h_neighbors, gather_nodes(h_nodes, idx)], -1)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.W_in, self.W_out, self.act = nn.Linear(dim, hidden), nn.Linear(hidden, dim), nn.GELU()

    def forward(self, x):
        return self.W_out(self.act(self.W_in(x)))


def _mlp(dim: int, n_in: int):
    return nn.Linear(dim + n_in, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)


class EncLayer(nn.Module):
    def __init__(self, dim: int, n_in: int, scale: int = 30):
        super().__init__()
        self.scale = scale
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim) for _ in range(3))
        self.W1, self.W2, self.W3 = _mlp(dim, n_in)
        self.W11, self.W12, self.W13 = _mlp(dim, n_in)
        self.act = nn.GELU()
        self.dense = FeedForward(dim, dim * 4)

    def forward(self, h_V, h_E, E_idx, mask_V, mask_attend):
        def message(h_V, a, b, c):
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_EV = torch.cat([h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1), h_EV], -1)
            return c(self.act(b(self.act(a(h_EV)))))
        h_message = mask_attend.unsqueeze(-1) * message(h_V, self.W1, self.W2, self.W3)
        h_V = self.norm1(h_V + torch.sum(h_message, -2) / self.scale)
        h_V = self.norm2(h_V + self.dense(h_V))
        h_V = mask_V.unsqueeze(-1) * h_V
        h_E = self.norm3(h_E + message(h_V, self.W11, self.W12, self.W13))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, dim: int, n_in: int, scale: int = 30):
        super().__init__()
        self.scale = scale
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.W1, self.W2, self.W3 = _mlp(dim, n_in)
        self.act = nn.GELU()
        self.dense = FeedForward(dim, dim * 4)

    def forward(self, h_V, h_E, mask_V, mask_attend):
        h_EV = torch.cat([h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1), h_E], -1)
        h_message = mask_attend.unsqueeze(-1) * self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        h_V = self.norm1(h_V + torch.sum(h_message, -2) / self.scale)
        h_V = self.norm2(h_V + self.dense(h_V))
        return mask_V.unsqueeze(-1) * h_V


class _Positional(nn.Module):
    def __init__(self, dim: int, max_relative: int = 32):
        super().__init__()
        self.max_relative = max_relative
        self.linear = nn.Linear(2 * max_relative + 2, dim)

    def forward(self, offset, same_chain):
        m = self.max_relative
        d = torch.clip(offset + m, 0, 2 * m) * same_chain + (1 - same_chain) * (2 * m + 1)
        return self.linear(torch.nn.functional.one_hot(d, 2 * m + 2).float())


# Order of the 25 atom-pair distance features (CA-CA is handled separately, as the first group)
_PAIRS = ("N N,C C,O O,B B,A N,A C,A O,A B,N C,N O,N B,B C,B O,O C,"
          "N A,C A,O A,B A,C N,O N,B N,C B,O B,C O").split(",")


class Features(nn.Module):
    def __init__(self, dim: int, top_k: int, num_rbf: int = 16, num_pos: int = 16):
        super().__init__()
        self.top_k, self.num_rbf = top_k, num_rbf
        self.embeddings = _Positional(num_pos)
        self.edge_embedding = nn.Linear(num_pos + num_rbf * 25, dim, bias=False)
        self.norm_edges = nn.LayerNorm(dim)

    def _rbf(self, D):
        mu = torch.linspace(2.0, 22.0, self.num_rbf, device=D.device).view(1, 1, 1, -1)
        return torch.exp(-((D.unsqueeze(-1) - mu) / (20.0 / self.num_rbf)) ** 2)

    def forward(self, X, mask, residue_idx, chain_id):
        N, Ca, C, O = X[:, :, 0], X[:, :, 1], X[:, :, 2], X[:, :, 3]
        b, c = Ca - N, C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
        atoms = {"N": N, "A": Ca, "C": C, "O": O, "B": Cb}
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = Ca.unsqueeze(1) - Ca.unsqueeze(2)
        D = torch.sqrt(torch.sum(dX ** 2, 3) + 1e-6).masked_fill(~mask_2D.bool(), float("inf"))
        D_nb, E_idx = torch.topk(D, min(self.top_k, X.shape[1]), dim=-1, largest=False)
        rbf = [self._rbf(D_nb)]
        for pair in _PAIRS:
            p, q = atoms[pair[0]], atoms[pair[2]]
            dist = torch.sqrt(torch.sum((p[:, :, None, :] - gather_nodes(q, E_idx)) ** 2, -1) + 1e-6)
            rbf.append(self._rbf(dist))
        offset = residue_idx[:, :, None] - gather_nodes(residue_idx[..., None], E_idx).squeeze(-1)
        same = (chain_id[:, :, None] == gather_nodes(chain_id[..., None], E_idx).squeeze(-1)).long()
        E = torch.cat((self.embeddings(offset.long(), same), torch.cat(rbf, -1)), -1)
        return self.norm_edges(self.edge_embedding(E)), E_idx


def priority(residue_index: Tensor, chain_id: Tensor, slot_index: Tensor, order_seed: int) -> Tensor:
    """Integer decoding priority (smaller decodes first): a fixed hash of (residue index, chain), ties broken by position."""
    key = ((residue_index.long() + 4099 * chain_id.long() + order_seed) * 1103515245 + 12345) % 2147483647
    return key * (1 << 26) + slot_index.long()


class HyInv(nn.Module):
    def __init__(self, dim=128, encoder_layers=4, decoder_layers=3, neighbors=48, order_seed=20260925):
        super().__init__()
        self.dim, self.order_seed = dim, order_seed
        self.features = Features(dim, neighbors)
        self.W_e = nn.Linear(dim, dim)
        self.W_s = nn.Embedding(21, dim)
        self.encoder_layers = nn.ModuleList(EncLayer(dim, dim * 2) for _ in range(encoder_layers))
        self.decoder_layers = nn.ModuleList(DecLayer(dim, dim * 3) for _ in range(decoder_layers))
        self.W_out = nn.Linear(dim, 21)

    @classmethod
    def from_pretrained(cls, path=None, device="cpu") -> "HyInv":
        from safetensors.torch import load_file
        path = Path(path) if path else Path(__file__).resolve().parent.parent / "weights"
        cfg = json.loads((path / "config.json").read_text())["architecture"]
        model = cls(**cfg)
        model.load_state_dict(load_file(path / "model.safetensors"), strict=True)
        return model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, bb: Backbone) -> dict:
        """The backbone-only part; computed once per backbone."""
        valid = bb.geometry_valid
        features, nbr = self.features(bb.safe_coords, valid.float(), bb.residue_index, bb.chain_id)
        h = torch.zeros((*features.shape[:2], self.dim), device=features.device)
        edge_valid = gather_nodes(valid[..., None], nbr).squeeze(-1) & valid[..., None]
        edge = self.W_e(features) * edge_valid[..., None]
        for layer in self.encoder_layers:
            h, edge = layer(h, edge, nbr, valid.float(), edge_valid.float())
            edge = edge * edge_valid[..., None]
        values = cat_neighbors_nodes(h, cat_neighbors_nodes(torch.zeros_like(h), edge, nbr), nbr)
        prio = priority(bb.residue_index, bb.chain_id, bb.slot_index, self.order_seed)
        return dict(h=h, edge=edge, nbr=nbr, edge_valid=edge_valid, values=values, prio=prio, valid=valid)

    @torch.no_grad()
    def decode(self, enc: dict, state: Tensor) -> Tensor:
        """state [B,L] (0..19 are amino acids, 20 is unknown) -> logits [B,L,20].
        Each position sees the known neighbors that precede it in decoding order; all other
        neighbors contribute structure only."""
        h, edge, nbr, valid = enc["h"], enc["edge"], enc["nbr"], enc["valid"]
        visible = state.ne(MASK) & valid
        earlier = gather_nodes(enc["prio"][..., None], nbr).squeeze(-1) < enc["prio"][..., None]
        backward = gather_nodes(visible[..., None], nbr).squeeze(-1) & (earlier | ~visible[..., None]) & enc["edge_valid"]
        bw = backward[..., None].to(h.dtype)
        fw = (enc["edge_valid"] & ~backward)[..., None].to(h.dtype)
        seq_edges = cat_neighbors_nodes(self.W_s(state), edge, nbr)
        node_mask, edge_mask = valid.float(), enc["edge_valid"].float()
        for layer in self.decoder_layers:
            h = layer(h, bw * cat_neighbors_nodes(h, seq_edges, nbr) + fw * enc["values"], node_mask, edge_mask)
        return self.W_out(h)[..., :20]
