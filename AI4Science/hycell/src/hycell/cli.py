"""Command line: ``hycell embed``, ``hycell finetune``, ``hycell predict``, ``hycell info``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _embed(a):
    import anndata as ad
    from .api import load
    model = load(a.weights, a.device, a.attention, a.precision)
    out = Path(a.output)
    if out.suffix == '.h5ad':
        adata = ad.read_h5ad(a.input)
        z = model.embed(adata, layer=a.layer, use_raw=a.use_raw, gene_key=a.gene_key, seed=a.seed,
                        batch_tokens=a.batch_tokens, progress=True)
        adata.obsm[a.key] = z
        adata.write_h5ad(out)
    else:
        if out.suffix != '.npy':
            raise SystemExit('--output must end in .npy or .h5ad')
        z = model.embed(a.input, layer=a.layer, use_raw=a.use_raw, gene_key=a.gene_key, seed=a.seed,
                        batch_tokens=a.batch_tokens, progress=True)
        np.save(out, z)
        names = ad.read_h5ad(a.input, backed='r').obs_names
        out.with_suffix('.obs_names.txt').write_text('\n'.join(map(str, names)) + '\n')
    print(json.dumps({'output': str(out), 'cells': int(z.shape[0]), 'dim': int(z.shape[1]),
                      'attention': model.attention, 'precision': model.precision}))


def _finetune(a):
    from .api import load
    model = load(a.weights, a.device, a.attention, a.precision)
    tuned = model.finetune(a.input, a.label_key, layer=a.layer, use_raw=a.use_raw, gene_key=a.gene_key,
                           steps=a.steps, cells_per_step=a.cells_per_step, lr=a.lr, head_lr=a.head_lr,
                           validation_fraction=a.validation_fraction, seed=a.seed)
    out = tuned.save(a.output)
    f = tuned.meta['finetune']
    print(json.dumps({'output': str(out), 'steps_run': f['steps_run'], 'best_step': f['best_step'],
                      'last': f['history'][-1] if f['history'] else None}))


def _predict(a):
    import anndata as ad
    import pandas as pd
    from .api import load
    model = load(a.weights, a.device, a.attention, a.precision)
    p = model.predict(a.input, return_probabilities=True, layer=a.layer, use_raw=a.use_raw, gene_key=a.gene_key)
    p.insert(0, 'prediction', p.idxmax(axis=1))
    p.index = pd.Index(ad.read_h5ad(a.input, backed='r').obs_names, name='cell')
    p.to_csv(a.output)
    print(json.dumps({'output': a.output, 'cells': len(p), 'classes': len(model.classes)}))


def _info(a):
    from .model import flash_available
    from .weights import index, packaged, verify
    import torch
    idx = index()
    rows = {name: {'sha256': m['sha256'], 'size_MB': round(m['size'] / 1e6, 1), 'path': str(packaged(name))}
            for name, m in idx['models'].items()}
    if a.verify:
        for name in rows:
            rows[name]['verified'] = verify(name)
    print(json.dumps({'models': rows, 'default': idx['default'],
                      'cuda': torch.cuda.is_available(), 'flash_attn': flash_available('cuda')}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='hycell', description='HyCell single-cell embeddings')
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('embed', help='embed the cells of an .h5ad file')
    e.add_argument('input', help='.h5ad file with raw counts')
    e.add_argument('-o', '--output', required=True, help='.npy (embeddings + .obs_names.txt) or .h5ad (input + obsm)')
    e.add_argument('--key', default='X_hycell', help='obsm key for .h5ad output')
    e.add_argument('--layer', help='layer holding the raw counts (default X)')
    e.add_argument('--use-raw', action='store_true', help='read the counts from adata.raw')
    e.add_argument('--gene-key', help="var column identifying genes ('index' for var_names; default: detected)")
    e.add_argument('--weights', help='model name, .safetensors file or directory (default: the released model)')
    e.add_argument('--device', help='cuda, cpu, cuda:1, ... (default: CUDA when available)')
    e.add_argument('--attention', default='auto', choices=['auto', 'flash', 'sdpa'])
    e.add_argument('--precision', default='auto', choices=['auto', 'bf16', 'fp32'])
    e.add_argument('--seed', type=int, default=0)
    e.add_argument('--batch-tokens', type=int, default=65536)
    e.set_defaults(fn=_embed)
    f = sub.add_parser('finetune', help='fine-tune a cell-type annotator on labelled cells and save it')
    f.add_argument('input', help='.h5ad file with raw counts and labels')
    f.add_argument('-o', '--output', required=True, help='directory for weights.safetensors + weights.json')
    f.add_argument('--label-key', required=True, help='obs column of cell labels')
    f.add_argument('--steps', type=int, default=2000)
    f.add_argument('--cells-per-step', type=int, default=64)
    f.add_argument('--lr', type=float, default=1e-5)
    f.add_argument('--head-lr', type=float, default=1e-3)
    f.add_argument('--validation-fraction', type=float, default=0.1)
    p = sub.add_parser('predict', help='predict labels with a fine-tuned model')
    p.add_argument('input', help='.h5ad file with raw counts')
    p.add_argument('-o', '--output', required=True, help='.csv with the prediction and the class probabilities')
    for q in (f, p):
        q.add_argument('--layer')
        q.add_argument('--use-raw', action='store_true')
        q.add_argument('--gene-key')
        q.add_argument('--weights', help='model to start from (finetune) or the fine-tuned model directory (predict)')
        q.add_argument('--device')
        q.add_argument('--attention', default='auto', choices=['auto', 'flash', 'sdpa'])
        q.add_argument('--precision', default='auto', choices=['auto', 'bf16', 'fp32'])
    f.add_argument('--seed', type=int, default=0)
    f.set_defaults(fn=_finetune)
    p.set_defaults(fn=_predict)
    i = sub.add_parser('info', help='models and hardware')
    i.add_argument('--verify', action='store_true', help='check the sha256 of the packaged weights')
    i.set_defaults(fn=_info)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
