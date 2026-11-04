"""Fine-tune HyCell for cell-type annotation.

The encoder and a linear classification head on the embedding are trained together with cross-entropy on
labelled cells, starting from the pretrained weights. Each step tokenizes a random set of cells with the
pretraining augmentations (random gene panel, binomial thinning, gene cap drawn by value). A random
``validation_fraction`` of the labelled cells is held out; the validation loss is evaluated every
``eval_every`` steps, the weights with the lowest loss are kept, and training stops after ``patience``
evaluations without improvement. The fine-tuned model embeds cells like the base model and predicts labels
with :meth:`HyCell.predict`.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import open_counts
from .tokenize import augment_counts, cell_rng, tokenize_cell


def _labels(adata, key):
    if key not in adata.obs.columns:
        raise KeyError(f'label_key {key!r} is not an obs column')
    lab = adata.obs[key]
    ok = lab.notna().to_numpy()
    s = lab.astype(str).to_numpy()
    classes = sorted(set(s[ok]))
    if len(classes) < 2:
        raise ValueError(f'obs[{key!r}] needs at least two classes')
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut[v] if o else -1 for v, o in zip(s, ok)], np.int64), classes


class _Steps:
    def __init__(self, total, enabled):
        self.bar = None
        if enabled:
            try:
                from tqdm.auto import tqdm
                self.bar = tqdm(total=total, unit='step', desc='hycell finetune')
            except ImportError:
                self.bar = False

    def update(self, step, info):
        if self.bar:
            self.bar.update(1)
            self.bar.set_postfix(info, refresh=False)
        elif self.bar is False and 'val_loss' in info:
            print(f'hycell finetune: step {step} ' + ' '.join(f'{k} {v}' for k, v in info.items()),
                  file=sys.stderr, flush=True)

    def close(self):
        if self.bar:
            self.bar.close()


def finetune(model, data, label_key: str, *, layer: str | None = None, use_raw: bool = False,
             gene_key: str | None = None, steps: int = 2000, cells_per_step: int = 64, lr: float = 1e-5,
             head_lr: float = 1e-3, validation_fraction: float = 0.1, eval_every: int = 100, patience: int = 5,
             seed: int = 0, progress: bool = True):
    """Fine-tune a loaded :class:`HyCell` to annotate cells with the labels in ``obs[label_key]``.

    data: AnnData or .h5ad path with raw counts; cells without a label are ignored. steps, cells_per_step:
    optimisation steps and cells per step. lr: learning rate of the network; head_lr: of the classification head.
    validation_fraction, eval_every, patience: early stopping on held-out labelled cells (0 trains for all steps).
    Returns a new HyCell with a classification head; the model passed in is not changed."""
    from .api import HyCell
    import anndata as ad
    adata = ad.read_h5ad(data) if isinstance(data, (str, Path)) else data
    y, classes = _labels(adata, label_key)
    src = open_counts(adata, model.genes, layer=layer, use_raw=use_raw, gene_key=gene_key)
    cells = [(g, x) for _, g, x in src.cells()]

    idx = np.flatnonzero((y >= 0) & np.array([len(g) > 0 for g, _ in cells]))
    rng = np.random.default_rng(seed)
    idx = idx[rng.permutation(len(idx))]
    n_val = int(round(validation_fraction * len(idx))) if validation_fraction > 0 else 0
    val_idx, train_idx = np.sort(idx[:n_val]), idx[n_val:]
    if len(train_idx) < 2:
        raise ValueError(f'too few labelled cells to fine-tune ({len(train_idx)} after the validation split)')

    dev, cls = model.device, model.model.cls_id
    net = copy.deepcopy(model.model).train()
    for p in net.parameters():
        p.requires_grad_(True)
    head = nn.Linear(net.cfg.d_embed, len(classes)).to(dev)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    params = [{'params': list(net.parameters()), 'lr': lr}, {'params': list(head.parameters()), 'lr': head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    y_t = torch.from_numpy(y).to(dev)

    def loss_of(rows, step_rng):
        toks = []
        for i in rows:
            g, x = cells[i]
            if step_rng is None:   # validation: inference tokenization
                toks.append(tokenize_cell(g, x, src.zero_pool, cell_rng(seed, i), cls, model.max_tokens,
                                          model.zero_ratio))
            else:
                ga, xa = augment_counts(g, x, step_rng)
                toks.append(tokenize_cell(ga, xa, src.zero_pool, step_rng, cls, model.max_tokens, model.zero_ratio,
                                          expressed=g, sample=True))
        with model._autocast():
            z = net.encode_cells(toks, model.attention)
        logit = head(z.float())
        yr = y_t[torch.from_numpy(np.asarray(rows)).to(dev)]
        return F.cross_entropy(logit, yr), (logit.argmax(-1) == yr).float().mean()

    @torch.no_grad()
    def validate():
        net.eval()
        loss, acc = 0.0, 0.0
        for a in range(0, len(val_idx), cells_per_step):
            rows = val_idx[a:a + cells_per_step]
            l, c = loss_of(rows, None)
            loss, acc = loss + float(l) * len(rows), acc + float(c) * len(rows)
        net.train()
        return loss / len(val_idx), acc / len(val_idx)

    def snapshot():
        return ({k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})

    history, best, best_step, stale, step = [], None, 0, 0, 0
    best_state = snapshot()
    if n_val:
        best, acc = validate()
        history.append({'step': 0, 'val_loss': round(best, 5), 'val_accuracy': round(acc, 5)})
    bar = _Steps(steps, progress)
    order, pos = train_idx[rng.permutation(len(train_idx))], 0
    for step in range(1, steps + 1):
        if pos + cells_per_step > len(order):
            order, pos = train_idx[rng.permutation(len(train_idx))], 0
        rows = order[pos:pos + cells_per_step]
        pos += cells_per_step
        loss, acc = loss_of(rows, np.random.default_rng([seed, 1, step]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in params for p in g['params']], 1.0)
        opt.step()
        info = {'loss': f'{float(loss):.4f}'}
        if n_val and (step % eval_every == 0 or step == steps):
            v, va = validate()
            history.append({'step': step, 'train_loss': round(float(loss), 5), 'val_loss': round(v, 5),
                            'val_accuracy': round(va, 5)})
            info['val_loss'] = f'{v:.4f}'
            if v < best:
                best, best_step, stale = v, step, 0
                best_state = snapshot()
            else:
                stale += 1
            if stale >= patience:
                bar.update(step, info)
                break
        bar.update(step, info)
    bar.close()
    if not n_val:
        best_state, best_step = snapshot(), step
    net.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    net.eval()
    head.eval()
    for p in list(net.parameters()) + list(head.parameters()):
        p.requires_grad_(False)

    meta = copy.deepcopy(model.meta)
    meta['model_id'] = f'{meta.get("model_id", "hycell")}-finetuned'
    meta['classes'] = classes
    meta['finetune'] = {
        'base_model': model.meta.get('model_id'), 'base_weights_sha256': model.meta.get('weights_sha256'),
        'label_key': label_key, 'n_classes': len(classes), 'n_cells': int(len(cells)), 'n_train': int(len(train_idx)),
        'n_validation': int(n_val), 'steps_run': int(step), 'best_step': int(best_step),
        'cells_per_step': cells_per_step, 'lr': lr, 'head_lr': head_lr, 'seed': seed, 'history': history}
    return HyCell(net, meta, dev, model.attention, model.precision, head=head)
