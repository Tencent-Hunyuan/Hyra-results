"""User-facing API: load the model, embed cells, decode embeddings back to expression, fine-tune and save."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import open_counts
from .genes import load_genes, map_ensembl, map_symbols, _is_ensembl
from .model import HyCellConfig, HyCellModel, flash_available
from .tokenize import MAX_TOKENS, ZERO_RATIO, cell_rng, tokenize_cell
from .weights import resolve, sha256

BLOCK_CELLS = 8192   # cells tokenized together before the padded path sorts them by length


class _Progress:
    def __init__(self, total, enabled):
        self.bar = None
        if enabled:
            try:
                from tqdm.auto import tqdm
                self.bar = tqdm(total=total, unit='cell', desc='hycell')
            except ImportError:
                self.total, self.done, self.next = total, 0, 0.0
                self.bar = False

    def update(self, n):
        if self.bar:
            self.bar.update(n)
        elif self.bar is False:
            self.done += n
            if self.done >= self.next or self.done == self.total:
                print(f'hycell: {self.done:,} / {self.total:,} cells', file=sys.stderr, flush=True)
                self.next = self.done + max(self.total // 20, 1)

    def close(self):
        if self.bar:
            self.bar.close()


class HyCell:
    """A loaded HyCell model. Create it with :func:`hycell.load`."""

    def __init__(self, model: HyCellModel, meta: dict, device, attention: str, precision: str, head=None):
        self.model, self.meta, self.device = model, meta, torch.device(device)
        self.attention, self.precision = attention, precision
        self.head = head                                  # classification head of a fine-tuned model
        self.classes = list(meta.get('classes', [])) or None
        self.genes = load_genes()
        tok = meta.get('tokenizer', {})
        self.max_tokens = int(tok.get('max_tokens', MAX_TOKENS))
        self.zero_ratio = float(tok.get('zero_ratio', ZERO_RATIO))
        if len(self.genes) != model.cfg.n_genes:
            raise ValueError('the gene table does not match the model')

    def __repr__(self):
        return (f'HyCell({self.meta.get("model_id", "hycell")}, {self.model.n_parameters():,} parameters, '
                f'device={self.device}, attention={self.attention}, precision={self.precision})')

    @property
    def embedding_dim(self) -> int:
        return self.model.cfg.d_embed

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.precision == 'bf16')

    # ------------------------------------------------------------------ embedding
    @torch.no_grad()
    def embed(self, data, layer: str | None = None, use_raw: bool = False, gene_key: str | None = None,
              seed: int = 0, rows=None, batch_tokens: int = 65536, progress: bool = False) -> np.ndarray:
        """Embed every cell of ``data`` (an AnnData or a path to an .h5ad file) from its raw counts.

        layer / use_raw: where the counts are (default X). gene_key: the var column identifying genes (default:
        detected, Ensembl ids preferred). seed, rows: each cell's tie breaks and measured-zero draws come from
        a generator seeded with (seed, row); rows defaults to 0..n-1, and passing the cells' row numbers in a larger
        file reproduces the embeddings of that file for a subset. batch_tokens: tokens per forward pass.
        Returns a float32 array (cells, 512) in input order."""
        src = open_counts(data, self.genes, layer=layer, use_raw=use_raw, gene_key=gene_key)
        cls = self.model.cls_id
        if rows is not None:
            rows = np.asarray(rows, dtype=np.int64)
            if rows.shape != (src.n_cells,):
                raise ValueError(f'rows must have one entry per cell ({src.n_cells})')
        bar = _Progress(src.n_cells, progress)

        def tokens():
            for row, g, x in src.cells():
                r = row if rows is None else rows[row]
                yield tokenize_cell(g, x, src.zero_pool, cell_rng(seed, r), cls, self.max_tokens, self.zero_ratio)

        run = self._embed_flash if self.attention == 'flash' else self._embed_sdpa
        z = run(tokens(), src.n_cells, batch_tokens, bar)
        bar.close()
        return z

    def _embed_flash(self, stream, n, budget, bar):
        out, seqs, used = [], [], 0

        def flush():
            with self._autocast():
                out.append(self.model.encode_cells(seqs, 'flash').float().cpu().numpy())
            bar.update(len(seqs))

        for t, v in stream:
            if seqs and used + len(t) > budget:
                flush()
                seqs, used = [], 0
            seqs.append((t, v))
            used += len(t)
        if seqs:
            flush()
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.embedding_dim), np.float32)

    def _embed_sdpa(self, stream, n, budget, bar):
        z = np.zeros((n, self.embedding_dim), np.float32)
        row = 0
        while row < n:
            block = [next(stream) for _ in range(min(BLOCK_CELLS, n - row))]
            lens = np.array([len(t) for t, _ in block])
            order = np.argsort(lens, kind='stable')
            i = 0
            while i < len(order):
                j = i + 1
                while j < len(order) and lens[order[j]] * (j + 1 - i) <= budget:
                    j += 1
                idx = order[i:j]
                with self._autocast():
                    zb = self.model.encode_cells([block[k] for k in idx], 'sdpa')
                z[row + idx] = zb.float().cpu().numpy()
                bar.update(len(idx))
                i = j
            row += len(block)
        return z

    # ------------------------------------------------------------------ genes and decoding
    def gene_ids(self, genes) -> np.ndarray:
        """Vocabulary ids of a list of Ensembl ids or gene symbols; KeyError for genes outside the vocabulary."""
        genes = np.asarray(list(genes), dtype=object)
        ids = map_ensembl(genes, self.genes) if _is_ensembl(genes) >= 0.5 else map_symbols(genes, self.genes)
        if (ids < 0).any():
            missing = genes[ids < 0]
            raise KeyError(f'{len(missing)} genes are not in the HyCell vocabulary, e.g. {list(missing[:5])}')
        return ids

    def measured_genes(self, data, layer: str | None = None, use_raw: bool = False,
                       gene_key: str | None = None) -> np.ndarray:
        """Ensembl ids of the vocabulary genes that ``data`` measures, in vocabulary order."""
        src = open_counts(data, self.genes, layer=layer, use_raw=use_raw, gene_key=gene_key)
        ids = np.unique(src.gene_map.vocab_id[src.gene_map.vocab_id >= 0])
        return self.genes.ensembl_id.values[ids]

    @torch.no_grad()
    def decode(self, z, genes=None, library_size=1e4, batch_size: int = 1024) -> np.ndarray:
        """Expression decoded from embeddings: the expected counts of each gene at ``library_size`` total counts,
        normalised over ``genes`` (Ensembl ids or symbols; default the whole vocabulary). Pass the genes a
        dataset measures (:meth:`measured_genes`) to decode in that dataset's gene space. Returns float32
        (cells, genes); with the default library size the values are counts per 10,000."""
        ids = None if genes is None else torch.from_numpy(self.gene_ids(genes)).long().to(self.device)
        z = np.asarray(z, np.float32)
        lib = np.broadcast_to(np.asarray(library_size, np.float64), (len(z),))
        out = []
        for a in range(0, len(z), batch_size):
            zb = torch.from_numpy(z[a:a + batch_size]).to(self.device)
            p = torch.softmax(self.model.gene_logits(zb, genes=ids), dim=-1)
            out.append((p.double().cpu().numpy() * lib[a:a + batch_size, None]).astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, 0 if ids is None else len(ids)), np.float32)

    def dispersion(self, genes=None) -> np.ndarray:
        """Negative-binomial inverse dispersion theta of each gene (variance = mu + mu^2 / theta)."""
        t = self.model.log_theta.detach().float().exp().cpu().numpy()
        return t if genes is None else t[self.gene_ids(genes)]

    # ------------------------------------------------------------------ fine-tuning
    def finetune(self, data, label_key: str, **kwargs) -> 'HyCell':
        """Fine-tune encoder and a classification head on the labels in obs[label_key]; returns a new model with
        :meth:`predict`. See :func:`hycell.finetune.finetune` for the options."""
        from .finetune import finetune
        return finetune(self, data, label_key, **kwargs)

    @torch.no_grad()
    def predict(self, data, return_probabilities: bool = False, **embed_kwargs):
        """Labels predicted by the classification head of a fine-tuned model. Returns an array of
        class names, or a DataFrame of class probabilities (cells x classes) with return_probabilities."""
        if self.head is None:
            raise ValueError('this model has no classification head; fine-tune it with labels first')
        z = torch.from_numpy(self.embed(data, **embed_kwargs)).to(self.device)
        p = torch.softmax(self.head(z).float(), -1).cpu().numpy()
        if return_probabilities:
            return pd.DataFrame(p, columns=self.classes)
        return np.asarray(self.classes, dtype=object)[p.argmax(1)]

    def save(self, directory) -> Path:
        """Write weights.safetensors and its weights.json sidecar into ``directory``; load with hycell.load(directory)."""
        from safetensors.torch import save_file
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().float().cpu().contiguous() for k, v in self.model.state_dict().items()}
        if self.head is not None:
            state.update({f'head.{k}': v.detach().float().cpu().contiguous() for k, v in self.head.state_dict().items()})
        path = out / 'weights.safetensors'
        save_file(state, str(path), metadata={'format': 'hycell-weights', 'model_id': str(self.meta.get('model_id'))})
        meta = dict(self.meta, weights_sha256=sha256(path))
        (out / 'weights.json').write_text(json.dumps(meta, indent=1) + '\n')
        return out


def load(weights=None, device=None, attention: str = 'auto', precision: str = 'auto') -> HyCell:
    """Load HyCell.

    weights: None (the released model, shipped with the package), a model name, a .safetensors file with its .json
    sidecar, or a directory holding both. device: 'cuda', 'cpu', ... (default: CUDA when available).
    attention: 'flash' (flash-attn, CUDA), 'sdpa' (PyTorch) or 'auto'. precision: 'bf16' (CUDA default, as in
    training and evaluation), 'fp32' (CPU default) or 'auto'."""
    from safetensors.torch import load_file
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if precision == 'auto':
        precision = 'bf16' if device.type == 'cuda' else 'fp32'
    if precision not in ('bf16', 'fp32'):
        raise ValueError("precision must be 'bf16', 'fp32' or 'auto'")
    if attention == 'auto':
        attention = 'flash' if flash_available(device) and precision == 'bf16' else 'sdpa'
    if attention == 'flash' and not (flash_available(device) and precision == 'bf16'):
        raise ValueError("attention='flash' needs flash-attn, a CUDA device and precision='bf16'")
    if attention not in ('flash', 'sdpa'):
        raise ValueError("attention must be 'flash', 'sdpa' or 'auto'")
    path, meta = resolve(weights)
    model = HyCellModel(HyCellConfig.from_dict(meta['model_config']))
    state = load_file(str(path))
    head_state = {k[5:]: state.pop(k) for k in list(state) if k.startswith('head.')}
    model.load_state_dict(state)
    head = None
    if head_state:
        head = nn.Linear(model.cfg.d_embed, len(meta['classes']))
        head.load_state_dict(head_state)
        head = head.to(device).eval()
    return HyCell(model.to(device).eval(), meta, device, attention, precision, head=head)


def embed(data, weights=None, device=None, **kwargs) -> np.ndarray:
    """One-call embedding: ``hycell.embed(adata)`` loads the default model and returns (cells, 512)."""
    return load(weights, device).embed(data, **kwargs)
