# Fine-tuning

Fine-tuning starts from released (or earlier fine-tuned) weights, trains on your reference energies and
forces, and exports weights that run like the released ones: `hyff.load`, `hyff.create_engine`,
`hyff.Simulation`, `hyff predict` and the ASE calculator use them on the model's fast engines.

```python
import hyff

summary = hyff.finetune("presto", train="train.extxyz", output="ft/", epochs=20)
model = hyff.load(weights=summary["best"])
```

```bash
hyff finetune --model presto --train train.extxyz --output ft/ --epochs 20
hyff md --weights ft/best --system system.npz --output run/
```

`docs/examples/finetune_example.py --output DIR` runs the whole workflow: it writes 400 perturbed
molecules and water clusters labelled by another HyFF model (a stand-in for reference calculations),
fine-tunes presto and loads the result in the ASE calculator.

## Data

- **Extended XYZ** or any other file that `ase.io.read` reads (needs ASE). Each frame carries its energy
  and forces, as written by ASE from a `SinglePointCalculator` (`energy=` in the comment line and a
  `forces` column); other names are selected with `--energy-key` (an `atoms.info` entry) and
  `--forces-key` (an `atoms.arrays` entry). `atoms.info["charge"]` and `atoms.info["multiplicity"]`
  (or `"spin"`, read as the multiplicity) default to 0 and 1. Frames with `pbc=True` are periodic in
  their cell, frames with `pbc=False` isolated.
- **NPZ** (no ASE needed): the keys of a molecule collection plus the labels.

| Key | Shape | Meaning |
|---|---|---|
| `positions` | (ΣN, 3) | positions of all structures, concatenated, Å |
| `numbers` | (ΣN,) | atomic numbers 1–83 |
| `offsets` | (M+1,) | structure boundaries |
| `energies` | (M,) | energies, eV |
| `forces` | (ΣN, 3) | forces, eV/Å |
| `total_charge`, `multiplicity` | (M,), optional | defaults 0 and 1 |
| `cells` | (M, 3, 3), optional | periodic cells (rows are lattice vectors); makes every structure periodic |

Energies may use any consistent reference, for example absolute total energies of another
electronic-structure method: before training, the per-element energy offsets of the model are refitted by
least squares on the training set, and the shifts are printed with the training energy error before and
after (`--no-refit-element-energies` keeps the offsets). Validation structures come from `--valid` or,
by default, a random 5% of the training set (`--valid-fraction`).

## Training

The plain PyTorch model is trained with forces in the loss (double backpropagation):
`energy_weight × mean((ΔE/N)²) + force_weight × mean(ΔF²)`, with energies per atom in eV and force
components in eV/Å. The optimizer is AdamW without weight decay, with a linear warm-up over the first 5% of
the steps, cosine decay to 10% of the peak learning rate and gradient clipping at norm 10. Batches hold up to
`--batch-atoms` atoms (default 2048); periodic and isolated structures are batched separately.

After every epoch the run reports the validation energy MAE (meV/atom) and force MAE (meV/Å), writes
`checkpoint.pt` and, when the validation loss improved, the `best/` weights.

| Option | Default | Meaning |
|---|---|---|
| `epochs` | 10 | passes over the training set |
| `lr` | per model, see below | peak learning rate |
| `energy_weight`, `force_weight` | 10, 1 | loss weights |
| `batch_atoms` | 2048 | atoms per batch |
| `freeze`, `train_only` | none | parameter name patterns (below) |
| `refit_element_energies` | on | refit the per-element energy offsets first |
| `ema_decay` | none | also keep an exponential moving average of the weights and export it |
| `valid`, `valid_fraction` | none, 0.05 | validation structures |
| `device` | cuda if available | training device (one GPU, or the CPU for small tests) |
| `gpus` | 1 | data-parallel GPUs of this machine (below) |
| `seed` | 0 | validation split and batch order |

## Learning rate

The default peak learning rate depends on the model: 3 × 10⁻⁵ for presto, 1 × 10⁻⁵ for vivace, moderato
and andante, and 3 × 10⁻⁶ for largo. These are the largest of 10⁻⁴, 3 × 10⁻⁵, 10⁻⁵ and 3 × 10⁻⁶ that
trained stably in short test runs (6 epochs on 960 molecules and water clusters labelled by another HyFF
model); at higher rates the validation error of the larger models rose. If the validation error rises over
the first epochs, lower `lr`; with much data far from the training domain of the released models a higher
rate may train faster. `best/` holds the weights with the lowest validation loss among all epochs and the
starting weights, so it is never worse on the validation set than the start.

## Multiple GPUs

```bash
hyff finetune --model andante --train train.extxyz --output ft/ --gpus 8
```

`gpus=N` (`--gpus N`) trains data-parallel on N GPUs of one machine; HyFF starts one process per GPU with
`torch.distributed.run`. Every GPU holds the whole model and takes every N-th batch; after each step the
gradients of all GPUs are summed, with the loss normalized by the structure and force counts of all of
them, so a step equals a single-GPU step on one batch holding all their structures. `batch_atoms` is per
GPU, so a step sees N times as many structures and an epoch has N times fewer steps; the learning rate is
not rescaled. When the batches do not divide evenly, the first ones of the (shuffled) epoch are repeated so
that every GPU takes the same number of steps. On two GPUs the summed gradient agrees with the gradient of
the combined batch to 10⁻⁵ (relative), the FP32 rounding of the different summation order.

Seconds per epoch (the second of two; the first includes start-up) with 24,000 structures of 5–18 atoms
(215,580 atoms), 2,048 atoms per batch and GPU, on H20 GPUs of one machine:

| Model | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|---|---:|---:|---:|---:|
| presto | 6.0 | 3.5 (1.7×) | 1.8 (3.3×) | 1.0 (6.0×) |
| andante | 23.6 | 12.1 (2.0×) | 6.4 (3.7×) | 3.5 (6.7×) |
| largo | 32.3 | 16.6 (1.9×) | 8.6 (3.8×) | 4.6 (7.0×) |

The larger the model, the smaller the share of the gradient exchange in a step. Every process loads the
model and the data, so the start of a largo run takes minutes on any number of GPUs.

Each GPU must hold the model, its gradients and the optimizer state for its own batch: data parallelism
does not split one structure over GPUs. A run resumes with the same number of GPUs (the batch order
depends on it).

## Choosing what to train

All parameters train by default. `hyff finetune --model presto --list-parameters` prints the parameter
groups; `--freeze` keeps parameters matching name patterns fixed and `--train-only` trains only the
matching ones (shell-style patterns on the parameter names), for example

```bash
hyff finetune ... --train-only "readouts.*" "readout_norms.*" "element_offset.*"   # output heads only
hyff finetune ... --freeze "embedding.*" "blocks.0.*"                                 # keep the first block
```

Each element has its own update network; the networks of elements absent from the training data receive
no gradient and stay unchanged. The latent-charge term of largo always stays fixed.

## Output and resuming

| File | Content |
|---|---|
| `best/weights.safetensors`, `best/weights.json` | weights with the lowest validation loss |
| `final/weights.safetensors`, `final/weights.json` | weights after the last epoch |
| `finetune.json` | options, starting weights, element shifts and the per-epoch history |
| `checkpoint.pt` | model, optimizer and run state for `--resume` (a `torch.load` pickle written by this tool) |
| `checkpoint.json` | epoch, step and number of GPUs of the checkpoint |

`--resume` (or `resume=True`) continues the run in the same output directory from its checkpoint: the
optimizer state, the learning-rate schedule and the data order continue, and the training and
validation files, the starting weights and the number of GPUs must be unchanged. A larger `--epochs` on
resume extends the run; the cosine schedule is then recomputed for the new length. In the tests a resumed run reproduces an
uninterrupted one bit for bit on the CPU. An output directory that already holds a checkpoint is refused
without `--resume`.

## Cost

Measured on one H20 with 960 training structures of
5–18 atoms (8,600 atoms per epoch) and the default batch of 2,048 atoms (median epoch of three):

| Model | Seconds per epoch | Peak GPU memory |
|---|---:|---:|
| presto | 0.4 | 0.3 GB |
| vivace | 0.6 | 0.6 GB |
| moderato | 0.8 | 1.8 GB |
| andante | 1.3 | 7.1 GB |
| largo | 1.6 | 22.6 GB |

Loading largo and writing its 4.8 GB weight files add minutes to a run. With `docs/examples/finetune_example.py`
(380 training and 20 validation structures, 10 epochs, defaults) the validation force MAE of presto
against the reference labels fell from 47.2 to 39.0 meV/Å and the energy MAE from 2.07 to 1.15 meV/atom.
