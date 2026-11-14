"""Fine-tuning of HyFF models on reference energies and forces.

Starts from released (or earlier fine-tuned) weights, trains the plain PyTorch model on structures with
reference energies and forces, and exports the result in the HyFF weight format. The exported weights load
with ``hyff.load(weights=...)`` and run on the model's fast engines, in ``hyff.Simulation`` and in the ASE
calculator.

    import hyff
    result = hyff.finetune("presto", train="train.extxyz", output="ft/", epochs=20)
    model = hyff.load(weights=result["best"])

Data: an ASE-readable file (extended XYZ, ASE database, ...) whose frames carry an energy and forces, or an
NPZ (see ``read_structures``). Units are Angstrom, eV and eV/Angstrom. Before training, the per-element
energy offsets are refitted by least squares so that a different electronic-structure method or energy
reference does not start as a large energy error.

``gpus=N`` trains data-parallel on N GPUs of one machine: every GPU holds the whole model, takes its share of
the batches, and the gradients are summed so that a step equals a single-GPU step on the union of the
batches (``batch_atoms`` is per GPU).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

MAX_Z = 83
CHECKPOINT = 'checkpoint.pt'
ALWAYS_FROZEN = ('latent_charge_energy.*',)
# Peak learning rates: the largest of 1e-4, 3e-5, 1e-5, 3e-6 that trained stably in short test runs.
DEFAULT_LR = dict(presto=3e-5, vivace=1e-5, moderato=1e-5, andante=1e-5, largo=3e-6)


# ----------------------------------------------------------------------------- data

@dataclass
class Structures:
    """Structures with reference energies and forces, stored as concatenated arrays."""
    positions: np.ndarray          # (sum N, 3) Angstrom
    numbers: np.ndarray            # (sum N,)
    offsets: np.ndarray            # (M + 1,)
    energies: np.ndarray           # (M,) eV
    forces: np.ndarray             # (sum N, 3) eV/Angstrom
    charges: np.ndarray            # (M,)
    multiplicities: np.ndarray     # (M,)
    cells: np.ndarray              # (M, 3, 3); rows of isolated structures are zero
    periodic: np.ndarray           # (M,) bool

    def __len__(self):
        return len(self.energies)

    @property
    def sizes(self):
        return np.diff(self.offsets)

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        atoms = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in indices])
        return Structures(self.positions[atoms], self.numbers[atoms],
                          np.concatenate([[0], np.cumsum(self.sizes[indices])]), self.energies[indices],
                          self.forces[atoms], self.charges[indices], self.multiplicities[indices],
                          self.cells[indices], self.periodic[indices])


def _check(data, name):
    def require(condition, message):
        if not condition:
            raise ValueError(f'{name}: {message}')
    m, n = len(data.energies), len(data.numbers)
    require(m > 0, 'no structures')
    require(data.offsets[0] == 0 and data.offsets[-1] == n and np.all(np.diff(data.offsets) > 0),
            'offsets must start at 0, increase and end at the atom count')
    require(data.positions.shape == (n, 3) and np.isfinite(data.positions).all(), 'positions must be finite (N, 3)')
    require(data.forces.shape == (n, 3) and np.isfinite(data.forces).all(), 'forces must be finite (N, 3)')
    require(np.isfinite(data.energies).all(), 'energies must be finite')
    bad = sorted(set(data.numbers[(data.numbers < 1) | (data.numbers > MAX_Z)].tolist()))
    require(not bad, f'atomic numbers outside 1-83 (H to Bi): {bad[:5]}')
    require(np.isfinite(data.charges).all(), 'charges must be finite')
    require(np.all(data.multiplicities >= 1), 'multiplicities must be integers 2S+1 >= 1')
    volumes = np.abs(np.linalg.det(data.cells[data.periodic])) if data.periodic.any() else np.ones(1)
    require(np.all(volumes > 0), 'periodic cells must have nonzero volume')
    return data


def read_structures(path, *, energy_key=None, forces_key=None):
    """Read training structures.

    NPZ: ``positions`` (sum N, 3), ``numbers`` (sum N,), ``offsets`` (M + 1,), ``energies`` (M,) and
    ``forces`` (sum N, 3), optionally ``total_charge`` and ``multiplicity`` (M,) and ``cells`` (M, 3, 3),
    which makes every structure periodic.

    Any other file is read with ``ase.io.read(path, index=":")``. The energy and forces come from the
    frame's calculator results (extended XYZ ``energy=`` and ``forces`` columns), or from
    ``atoms.info[energy_key]`` and ``atoms.arrays[forces_key]``. ``atoms.info["charge"]`` and
    ``atoms.info["multiplicity"]`` (or ``"spin"``, the OMol25 convention for the multiplicity) default to
    0 and 1. Frames must have ``pbc`` all True (periodic) or all False (isolated).
    """
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            missing = [key for key in ('positions', 'numbers', 'offsets', 'energies', 'forces') if key not in z]
            if missing:
                raise ValueError(f'{path.name} is missing {", ".join(missing)}')
            offsets = z['offsets'].astype(np.int64)
            m = len(offsets) - 1
            cells = z['cells'].astype(np.float64) if 'cells' in z else np.zeros((m, 3, 3))
            data = Structures(z['positions'].astype(np.float64), z['numbers'].astype(np.int64), offsets,
                              z['energies'].astype(np.float64).reshape(-1), z['forces'].astype(np.float64),
                              z['total_charge'].astype(np.float64).reshape(-1) if 'total_charge' in z else np.zeros(m),
                              z['multiplicity'].astype(np.int64).reshape(-1) if 'multiplicity' in z else np.ones(m, np.int64),
                              cells.reshape(m, 3, 3), np.full(m, 'cells' in z))
        return _check(data, path.name)
    from ase.io import read
    frames = read(str(path), index=':')
    if not isinstance(frames, list):
        frames = [frames]
    positions, numbers, energies, forces, charges, multiplicities, cells, periodic = ([] for _ in range(8))
    for k, atoms in enumerate(frames):
        where = f'{path.name} frame {k}'
        results = getattr(atoms.calc, 'results', {}) if atoms.calc is not None else {}
        if energy_key is not None:
            if energy_key not in atoms.info:
                raise ValueError(f'{where}: no atoms.info[{energy_key!r}]')
            energy = atoms.info[energy_key]
        elif 'energy' in results:
            energy = results['energy']
        else:
            raise ValueError(f'{where}: no energy; give energy_key for an atoms.info entry')
        if forces_key is not None:
            if forces_key not in atoms.arrays:
                raise ValueError(f'{where}: no atoms.arrays[{forces_key!r}]')
            frame_forces = atoms.arrays[forces_key]
        elif 'forces' in results:
            frame_forces = results['forces']
        else:
            raise ValueError(f'{where}: no forces; give forces_key for an atoms.arrays entry')
        info = atoms.info
        if 'multiplicity' in info and 'spin' in info and info['multiplicity'] != info['spin']:
            raise ValueError(f'{where}: atoms.info has different "multiplicity" and "spin"')
        pbc = np.asarray(atoms.pbc, dtype=bool)
        if pbc.any() and not pbc.all():
            raise ValueError(f'{where}: pbc must be all True (periodic) or all False (isolated)')
        multiplicity = float(info.get('multiplicity', info.get('spin', 1)))
        if multiplicity != int(multiplicity):
            raise ValueError(f'{where}: the multiplicity 2S+1 must be an integer')
        positions.append(atoms.positions)
        numbers.append(atoms.numbers)
        energies.append(float(energy))
        forces.append(np.asarray(frame_forces, dtype=np.float64).reshape(-1, 3))
        charges.append(float(info.get('charge', 0)))
        multiplicities.append(int(multiplicity))
        cells.append(atoms.cell.array if pbc.all() else np.zeros((3, 3)))
        periodic.append(bool(pbc.all()))
    data = Structures(np.concatenate(positions).astype(np.float64), np.concatenate(numbers).astype(np.int64),
                      np.concatenate([[0], np.cumsum([len(x) for x in numbers])]).astype(np.int64),
                      np.array(energies), np.concatenate(forces), np.array(charges),
                      np.array(multiplicities, dtype=np.int64), np.array(cells, dtype=np.float64).reshape(-1, 3, 3),
                      np.array(periodic))
    return _check(data, path.name)


def _batches(data, indices, batch_atoms, rng):
    """Groups of structures with at most ``batch_atoms`` atoms (a larger structure forms its own group);
    periodic and isolated structures are never mixed."""
    sizes, groups = data.sizes, []
    for periodic in (False, True):
        pool = [int(i) for i in indices if data.periodic[i] == periodic]
        if rng is not None:
            pool = [pool[i] for i in rng.permutation(len(pool))]
        current, count = [], 0
        for i in pool:
            if current and count + sizes[i] > batch_atoms:
                groups.append(current)
                current, count = [], 0
            current.append(i)
            count += int(sizes[i])
        if current:
            groups.append(current)
    if rng is not None:
        groups = [groups[i] for i in rng.permutation(len(groups))]
    return groups


def shard(groups, rank, world):
    """The batches of one of ``world`` processes: every process gets the same number of batches; when the
    count does not divide evenly the first batches are repeated (each epoch reshuffles them)."""
    if world == 1:
        return list(groups)
    groups = list(groups)
    groups += [groups[i % len(groups)] for i in range((-len(groups)) % world)]
    return groups[rank::world]


def _collate(data, group, device):
    import torch
    atoms = np.concatenate([np.arange(data.offsets[i], data.offsets[i + 1]) for i in group])
    sizes = data.sizes[group]
    batch = dict(positions=torch.as_tensor(data.positions[atoms], dtype=torch.float32, device=device),
                 atomic_numbers=torch.as_tensor(data.numbers[atoms], dtype=torch.int64, device=device),
                 batch=torch.repeat_interleave(torch.arange(len(group), device=device),
                                               torch.as_tensor(sizes, device=device)),
                 ptr=torch.as_tensor(np.concatenate([[0], np.cumsum(sizes)]), dtype=torch.int64, device=device),
                 n_atoms=torch.as_tensor(sizes, dtype=torch.float32, device=device),
                 total_charge=torch.as_tensor(data.charges[group], dtype=torch.float32, device=device),
                 multiplicity=torch.as_tensor(data.multiplicities[group], dtype=torch.int64, device=device))
    if data.periodic[group[0]]:
        batch['cell'] = torch.as_tensor(data.cells[group], dtype=torch.float32, device=device)
        batch['pbc'] = torch.ones((len(group), 3), dtype=torch.bool, device=device)
    targets = dict(energy=torch.as_tensor(data.energies[group], dtype=torch.float64, device=device),
                   forces=torch.as_tensor(data.forces[atoms], dtype=torch.float32, device=device))
    return batch, targets


def _predict(model, batch, create_graph):
    import torch
    from hyff.graph import radius_edges
    positions = batch['positions'].requires_grad_(True)
    with torch.enable_grad():
        energy = model.energy(batch, radius_edges(batch, model.cutoff))
        forces = -torch.autograd.grad(energy.sum(), positions, create_graph=create_graph)[0]
    return energy, forces


# ----------------------------------------------------------------------------- configuration

@dataclass
class FineTuneConfig:
    """Options of :func:`finetune` (also the keys of ``finetune.json``)."""
    train: str
    model: str | None = None            # model name; None: presto, or inferred from weights
    weights: str | None = None          # starting checkpoint, e.g. an earlier fine-tune
    valid: str | None = None            # validation file; None: a random valid_fraction of train
    valid_fraction: float = 0.05
    output: str = 'hyff_finetune'
    epochs: int = 10
    lr: float | None = None             # peak learning rate (AdamW); None: the model's DEFAULT_LR
    min_lr_ratio: float = 0.1           # cosine decay to lr * min_lr_ratio
    warmup_fraction: float = 0.05       # linear warm-up over this fraction of training
    batch_atoms: int = 2048             # atoms per batch
    energy_weight: float = 10.0         # weight of the squared energy error per atom (eV/atom)^2
    force_weight: float = 1.0           # weight of the squared force error per component (eV/A)^2
    freeze: list = field(default_factory=list)       # parameter name patterns kept fixed
    train_only: list = field(default_factory=list)   # if given, only parameters matching these train
    refit_element_energies: bool = True
    ema_decay: float | None = None      # exponential moving average of the weights (e.g. 0.999)
    clip_grad_norm: float | None = 10.0
    energy_key: str | None = None
    forces_key: str | None = None
    device: str | None = None           # default: cuda if available, else cpu
    gpus: int = 1                       # data-parallel GPUs of this machine (batch_atoms is per GPU)
    seed: int = 0
    resume: bool = False

    def __post_init__(self):
        for name in ('epochs', 'batch_atoms', 'gpus'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')
        for name in ('energy_weight', 'force_weight'):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
                raise ValueError(f'{name} must be a finite number >= 0, got {value!r}')
        if self.lr is not None and not (isinstance(self.lr, (int, float)) and math.isfinite(self.lr) and self.lr > 0):
            raise ValueError(f'lr must be a positive number or None, got {self.lr!r}')
        if not 0 <= self.valid_fraction < 1:
            raise ValueError('valid_fraction must be in [0, 1)')
        if not 0 <= self.min_lr_ratio <= 1 or not 0 <= self.warmup_fraction < 1:
            raise ValueError('min_lr_ratio must be in [0, 1] and warmup_fraction in [0, 1)')
        if self.ema_decay is not None and not 0 < self.ema_decay < 1:
            raise ValueError('ema_decay must be in (0, 1)')
        if self.clip_grad_norm is not None and not self.clip_grad_norm > 0:
            raise ValueError('clip_grad_norm must be positive or None')
        if self.gpus > 1 and self.device not in (None, 'cuda'):
            raise ValueError(f'gpus={self.gpus} trains on CUDA devices; device must be cuda or None')
        self.freeze, self.train_only = list(self.freeze), list(self.train_only)
        if self.model is None and self.weights is None:
            self.model = 'presto'


# ----------------------------------------------------------------------------- parameters

def parameter_groups(model):
    """[(group, parameter count, number of tensors)] with groups ``blocks.<i>.<module>`` and top-level modules."""
    groups = {}
    for name, parameter in model.named_parameters():
        parts = name.split('.')
        key = '.'.join(parts[:3]) if parts[0] == 'blocks' else parts[0]
        count, tensors = groups.get(key, (0, 0))
        groups[key] = (count + parameter.numel(), tensors + 1)
    return [(key, count, tensors) for key, (count, tensors) in groups.items()]


def select_trainable(model, freeze=(), train_only=()):
    """Set requires_grad from name patterns (fnmatch); returns (trainable, total) parameter counts.

    With ``train_only`` only matching parameters train; ``freeze`` then fixes matching ones. The
    latent-charge term of largo always stays fixed (its learned scale is ~0, and periodic structures would
    need its periodic sum).
    """
    names = [name for name, _ in model.named_parameters()]
    for pattern in list(freeze) + list(train_only):
        if not any(fnmatch(name, pattern) for name in names):
            raise ValueError(f'Pattern {pattern!r} matches no parameter (list them with hyff finetune --list-parameters)')
    trainable = 0
    for name, parameter in model.named_parameters():
        train = any(fnmatch(name, p) for p in train_only) if train_only else True
        if any(fnmatch(name, p) for p in list(freeze) + list(ALWAYS_FROZEN)):
            train = False
        parameter.requires_grad_(train)
        trainable += parameter.numel() if train else 0
    if not trainable:
        raise ValueError('No parameter is trainable with these freeze/train_only patterns')
    return trainable, sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- weights

def export_weights(model, source_weights, directory):
    """Write ``weights.safetensors`` and ``weights.json`` (the source architecture) into ``directory``."""
    from safetensors.torch import save_file
    from hyff.weights import config_path, entry_path, sha256
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().to('cpu').contiguous().clone() for name, value in model.state_dict().items()}
    target = directory / 'weights.safetensors'
    temporary = directory / 'weights.safetensors.pending'
    save_file(state, str(temporary))
    os.replace(temporary, target)
    source = json.loads(config_path(entry_path(source_weights)).read_text())
    config = {key: source[key] for key in ('format', 'model_id', 'model_config', 'numerical_execution') if key in source}
    config['weights_sha256'] = sha256(target)
    (directory / 'weights.json').write_text(json.dumps(config, indent=1) + '\n')
    return directory


# ----------------------------------------------------------------------------- training

def _sum_gradients(parameters, bucket_elements=1 << 26):
    """Sum the gradients of all processes in place (bucketed). A parameter keeps ``grad=None``, and so is skipped
    by the optimizer as on one GPU, only when no process has a gradient for it."""
    import torch
    import torch.distributed as dist
    device = parameters[0].device
    present = torch.tensor([p.grad is not None for p in parameters], dtype=torch.float32, device=device)
    dist.all_reduce(present)
    bucket = []

    def flush():
        flat = torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), dtype=p.dtype, device=device)
                          for p in bucket])
        dist.all_reduce(flat)
        offset = 0
        for p in bucket:
            p.grad = flat[offset:offset + p.numel()].view_as(p).clone()
            offset += p.numel()
        bucket.clear()

    size = 0
    for p, used in zip(parameters, present.tolist()):
        if not used:
            p.grad = None
            continue
        bucket.append(p)
        size += p.numel()
        if size >= bucket_elements:
            flush()
            size = 0
    if bucket:
        flush()


class _EMA:
    def __init__(self, parameters, decay):
        import torch
        self.parameters, self.decay = list(parameters), decay
        self.shadow = [p.detach().clone() for p in self.parameters]
        self._torch = torch

    def update(self):
        with self._torch.no_grad():
            for shadow, parameter in zip(self.shadow, self.parameters):
                shadow.lerp_(parameter.detach(), 1 - self.decay)

    def swap(self):
        with self._torch.no_grad():
            for shadow, parameter in zip(self.shadow, self.parameters):
                saved = parameter.detach().clone()
                parameter.copy_(shadow)
                shadow.copy_(saved)


class FineTune:
    """One fine-tuning run; ``run()`` trains and returns the summary (also written to ``finetune.json``)."""

    def __init__(self, config: FineTuneConfig, verbose=True):
        import torch
        from hyff.registry import resolve
        from hyff.weights import load_model, sha256
        self.config = config
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.rank = int(os.environ.get('RANK', '0'))
        self.verbose = verbose and self.rank == 0
        self.out = Path(config.output)
        if (self.out / CHECKPOINT).exists() and not config.resume:
            raise FileExistsError(f'{self.out} already holds a fine-tuning checkpoint; pass resume=True '
                                  '(--resume) to continue it or choose a fresh output directory')
        if config.resume and not (self.out / CHECKPOINT).exists():
            raise FileNotFoundError(f'resume=True but {self.out / CHECKPOINT} does not exist')
        if self.world > 1:
            import torch.distributed as dist
            if self.world != config.gpus:
                raise ValueError(f'Started with {self.world} processes for gpus={config.gpus}')
            local = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local)
            self.device = f'cuda:{local}'
            if not dist.is_initialized():
                dist.init_process_group('nccl')
        else:
            self.device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        selection = resolve(config.model, config.weights, engine='reference')
        self.source = dict(model=selection.name, weights=selection.weights, sha256=selection.sha256)
        torch.set_float32_matmul_precision('highest')
        self.model, _ = load_model(selection.weights, self.device)
        self.trainable, self.total = select_trainable(self.model, config.freeze, config.train_only)
        self.data = read_structures(config.train, energy_key=config.energy_key, forces_key=config.forces_key)
        files = [config.train] + ([config.valid] if config.valid else [])
        self.data_sha256 = [sha256(path) for path in files]
        rng = np.random.default_rng(config.seed)
        if config.valid:
            self.valid = read_structures(config.valid, energy_key=config.energy_key, forces_key=config.forces_key)
            self.train_indices = np.arange(len(self.data))
        else:
            order = rng.permutation(len(self.data))
            count = int(round(config.valid_fraction * len(self.data)))
            count = min(max(count, 1 if config.valid_fraction > 0 and len(self.data) > 1 else 0), len(self.data) - 1)
            self.valid = self.data.subset(np.sort(order[:count])) if count else None
            self.train_indices = np.sort(order[count:])
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.lr = config.lr if config.lr is not None else DEFAULT_LR.get(selection.name, 1e-5)
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.lr, weight_decay=0.0)
        self.ema = _EMA(self.parameters, config.ema_decay) if config.ema_decay else None
        self.rng = np.random.default_rng(config.seed + 1)
        self.epoch, self.step, self.history, self.best = 0, 0, [], None
        self.element_shifts = None

    # -- helpers
    def say(self, text):
        if self.verbose:
            print(text, flush=True)

    def all_sum(self, values):
        """Sum a float64 vector over the processes (identity on one process)."""
        import torch
        tensor = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        if self.world > 1:
            import torch.distributed as dist
            dist.all_reduce(tensor)
        return tensor

    def barrier(self):
        if self.world > 1:
            import torch.distributed as dist
            dist.barrier()

    def broadcast_state(self):
        """Copy the model weights of process 0 to all processes."""
        if self.world > 1:
            import torch.distributed as dist
            for value in self.model.state_dict().values():
                dist.broadcast(value, 0)

    def learning_rate(self, progress):
        c = self.config
        if progress < c.warmup_fraction:
            return self.lr * max(progress / c.warmup_fraction, 0.01)
        t = (progress - c.warmup_fraction) / (1 - c.warmup_fraction)
        return self.lr * (c.min_lr_ratio + (1 - c.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    def evaluate(self, data, indices=None):
        """Energy MAE (meV/atom), force MAE (meV/A) and the loss on ``data``."""
        import torch
        indices = np.arange(len(data)) if indices is None else indices
        e_abs, e_sq, f_abs, f_sq, structures, components = 0., 0., 0., 0., 0, 0
        groups = _batches(data, indices, self.config.batch_atoms, None)
        for group in groups[self.rank::self.world]:
            batch, targets = _collate(data, group, self.device)
            energy, forces = _predict(self.model, batch, create_graph=False)
            de = ((energy.detach() - targets['energy']) / batch['n_atoms'].double()).cpu().numpy()
            df = (forces.detach() - targets['forces']).double()
            e_abs += float(np.abs(de).sum()); e_sq += float((de ** 2).sum())
            f_abs += float(df.abs().sum()); f_sq += float(df.square().sum())
            structures += len(group); components += df.numel()
            del energy, forces
        e_abs, e_sq, f_abs, f_sq, structures, components = self.all_sum(
            [e_abs, e_sq, f_abs, f_sq, structures, components]).tolist()
        structures, components = int(structures), int(components)
        c = self.config
        loss = c.energy_weight * e_sq / structures + c.force_weight * f_sq / components
        return dict(energy_mae_meV_atom=1e3 * e_abs / structures, force_mae_meV_A=1e3 * f_abs / components,
                    loss=loss, structures=structures)

    def refit_element_energies(self):
        """Least-squares per-element shifts of the energy offsets, from the training structures."""
        import torch
        data, indices = self.data, self.train_indices
        from hyff.graph import radius_edges
        predicted = np.zeros(len(data))
        with torch.no_grad():
            groups = _batches(data, indices, self.config.batch_atoms, None)
            for group in groups[self.rank::self.world]:
                batch, _ = _collate(data, group, self.device)
                predicted[group] = self.model.energy(batch, radius_edges(batch, self.model.cutoff)).double().cpu().numpy()
        predicted = self.all_sum(predicted).cpu().numpy()
        counts = np.zeros((len(indices), MAX_Z + 1))
        for row, i in enumerate(indices):
            np.add.at(counts[row], data.numbers[data.offsets[i]:data.offsets[i + 1]], 1)
        present = np.flatnonzero(counts.sum(0))
        residual = data.energies[indices] - predicted[indices]
        delta = np.linalg.lstsq(counts[:, present], residual, rcond=None)[0]
        weight = self.model.element_offset.weight
        with torch.no_grad():
            weight[torch.as_tensor(present, device=weight.device), 0] += torch.as_tensor(delta, dtype=weight.dtype,
                                                                                         device=weight.device)
        if self.ema is not None:
            self.ema.shadow = [p.detach().clone() for p in self.ema.parameters]
        before = np.abs(residual / data.sizes[indices]).mean()
        after = np.abs((residual - counts[:, present] @ delta) / data.sizes[indices]).mean()
        self.element_shifts = {int(z): float(d) for z, d in zip(present, delta)}
        self.say(f'refitted element energies (eV): ' + ', '.join(f'Z={z}: {d:+.4f}' for z, d in self.element_shifts.items())
                 + f'; training energy MAE {1e3 * before:.2f} -> {1e3 * after:.2f} meV/atom')

    # -- checkpoints
    def save_checkpoint(self):
        import torch
        if self.rank != 0:
            self.barrier()
            return
        self.out.mkdir(parents=True, exist_ok=True)
        state = dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                     ema=None if self.ema is None else self.ema.shadow, epoch=self.epoch, step=self.step,
                     history=self.history, best=self.best, rng=self.rng.bit_generator.state,
                     config=asdict(self.config), source=self.source, data_sha256=self.data_sha256,
                     train_indices=self.train_indices, element_shifts=self.element_shifts, gpus=self.world)
        temporary = self.out / (CHECKPOINT + '.pending')
        torch.save(state, temporary)
        os.replace(temporary, self.out / CHECKPOINT)
        # Small summary of the checkpoint, read by the launcher before it starts the processes.
        (self.out / 'checkpoint.json').write_text(json.dumps(dict(epoch=self.epoch, step=self.step, gpus=self.world)))
        self.barrier()

    def load_checkpoint(self):
        import torch
        state = torch.load(self.out / CHECKPOINT, map_location=self.device, weights_only=False)
        if state['source']['sha256'] != self.source['sha256']:
            raise ValueError('The checkpoint started from other weights than this configuration')
        if state['data_sha256'] != self.data_sha256:
            raise ValueError('The training or validation files differ from those of the checkpoint')
        if state.get('gpus', 1) != self.world:
            raise ValueError(f'The checkpoint was written with gpus={state.get("gpus", 1)}; resume with the same '
                             'number of GPUs (the batch order depends on it)')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.ema is not None and state['ema'] is not None:
            self.ema.shadow = [t.to(self.device) for t in state['ema']]
        self.epoch, self.step, self.history, self.best = state['epoch'], state['step'], state['history'], state['best']
        self.rng.bit_generator.state = state['rng']
        self.train_indices = np.asarray(state['train_indices'])
        self.element_shifts = state['element_shifts']

    def export(self, name):
        target = self.out / name
        if self.rank == 0:
            if self.ema is not None:
                self.ema.swap()
            try:
                export_weights(self.model, self.source['weights'], target)
            finally:
                if self.ema is not None:
                    self.ema.swap()
        self.barrier()
        return str(target)

    def validate(self):
        if self.valid is None:
            return None
        if self.ema is not None:
            self.ema.swap()
        try:
            return self.evaluate(self.valid)
        finally:
            if self.ema is not None:
                self.ema.swap()

    def train_step(self, group):
        """Gradients of the loss over the batches of all processes (``group`` is this process's batch); returns
        the loss. The loss is normalized by the global structure and force-component counts, so the summed
        gradient equals the gradient of one batch holding all of them."""
        import torch
        c = self.config
        batch, targets = _collate(self.data, group, self.device)
        energy, forces = _predict(self.model, batch, create_graph=True)
        de = (energy - targets['energy']) / batch['n_atoms'].double()
        energy_sum = de.square().sum()
        force_sum = (forces - targets['forces']).square().sum()
        counts = self.all_sum([len(group), forces.numel()])
        loss = c.energy_weight * energy_sum / counts[0] + c.force_weight * force_sum / counts[1]
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.world > 1:
            _sum_gradients(self.parameters)
        return float(self.all_sum(loss.detach().reshape(1))[0])

    # -- run
    def run(self):
        import torch
        c = self.config
        self.out.mkdir(parents=True, exist_ok=True)
        self.say(f'fine-tuning {self.source["model"] or self.source["weights"]}: {len(self.train_indices)} training and '
                 f'{0 if self.valid is None else len(self.valid)} validation structures, '
                 f'{self.trainable:,} of {self.total:,} parameters trainable, peak lr {self.lr:g}, '
                 + (f'{self.world} GPUs' if self.world > 1 else f'device {self.device}'))
        if c.resume:
            self.load_checkpoint()
            self.say(f'resumed after epoch {self.epoch} (step {self.step})')
        else:
            if c.refit_element_energies:
                self.refit_element_energies()
            self.broadcast_state()
            if self.ema is not None:
                self.ema.shadow = [p.detach().clone() for p in self.ema.parameters]
            start = self.validate()
            self.history.append(dict(epoch=0, step=0, valid=start))
            if start is not None:
                # The starting weights compete for best/: it is never worse on validation than the start.
                self.best = dict(epoch=0, score=start['loss'])
                self.export('best')
                self.say(f'epoch 0: valid energy MAE {start["energy_mae_meV_atom"]:.3f} meV/atom, '
                         f'force MAE {start["force_mae_meV_A"]:.2f} meV/A')
        while self.epoch < c.epochs:
            started = time.perf_counter()
            groups = shard(_batches(self.data, self.train_indices, c.batch_atoms, self.rng), self.rank, self.world)
            total = 0.
            for k, group in enumerate(groups):
                lr = self.learning_rate((self.epoch + k / len(groups)) / c.epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                total += self.train_step(group)
                if c.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_([p for p in self.parameters if p.grad is not None], c.clip_grad_norm)
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update()
                self.step += 1
            self.epoch += 1
            valid = self.validate()
            record = dict(epoch=self.epoch, step=self.step, lr=lr, seconds=time.perf_counter() - started,
                          train_loss=total / len(groups), valid=valid)
            self.history.append(record)
            score = valid['loss'] if valid is not None else record['train_loss']
            improved = self.best is None or score < self.best['score']
            if improved:
                self.best = dict(epoch=self.epoch, score=score)
                self.export('best')
            message = f'epoch {self.epoch}/{c.epochs}: train loss {record["train_loss"]:.3g}'
            if valid is not None:
                message += (f', valid energy MAE {valid["energy_mae_meV_atom"]:.3f} meV/atom, '
                            f'force MAE {valid["force_mae_meV_A"]:.2f} meV/A')
            self.say(message + f', lr {lr:.2e}, {record["seconds"]:.1f} s' + (' (best)' if improved else ''))
            self.save_checkpoint()
        final = self.export('final')
        summary = dict(config=asdict(c), source=self.source, device=self.device, gpus=self.world, lr=self.lr,
                       trainable_parameters=self.trainable,
                       total_parameters=self.total, element_shifts_eV=self.element_shifts,
                       best=str(self.out / 'best'), best_epoch=self.best['epoch'], final=final,
                       history=self.history)
        if self.rank == 0:
            (self.out / 'finetune.json').write_text(json.dumps(summary, indent=1) + '\n')
        self.barrier()
        return summary

    def close(self):
        if self.world > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


def finetune(model=None, *, train, output='hyff_finetune', weights=None, verbose=True, **options):
    """Fine-tune a model on reference energies and forces; see :class:`FineTuneConfig` for the options.

    Returns the summary written to ``output/finetune.json``; ``summary["best"]`` (lowest validation loss)
    and ``summary["final"]`` are weight directories for ``hyff.load(weights=...)``, ``hyff.Simulation``
    and the ASE calculator.
    """
    config = FineTuneConfig(train=str(train), model=model, weights=None if weights is None else str(weights),
                            output=str(output), **options)
    if config.gpus > 1 and 'LOCAL_RANK' not in os.environ:
        return _launch(config, verbose)
    run = FineTune(config, verbose=verbose)
    try:
        return run.run()
    finally:
        run.close()


def _launch(config, verbose):
    """Run a multi-GPU fine-tune: one process per GPU under torchrun; returns the summary of process 0."""
    import torch
    out = Path(config.output)
    if (out / CHECKPOINT).exists() and not config.resume:
        raise FileExistsError(f'{out} already holds a fine-tuning checkpoint; pass resume=True '
                              '(--resume) to continue it or choose a fresh output directory')
    available = torch.cuda.device_count()
    if available < config.gpus:
        raise ValueError(f'gpus={config.gpus} but {available} CUDA devices are visible')
    if config.resume and (out / 'checkpoint.json').is_file():
        written = json.loads((out / 'checkpoint.json').read_text()).get('gpus', 1)
        if written != config.gpus:
            raise ValueError(f'The checkpoint was written with gpus={written}; resume with the same number of GPUs '
                             '(the batch order depends on it)')
    out.mkdir(parents=True, exist_ok=True)
    request = out / 'finetune_request.json'
    request.write_text(json.dumps(dict(config=asdict(config), verbose=verbose), indent=1) + '\n')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={config.gpus}',
               '-m', 'hyff.finetuning', str(request)]
    completed = subprocess.run(command)
    if completed.returncode:
        raise RuntimeError(f'multi-GPU fine-tuning failed (exit code {completed.returncode}); see the output above')
    return json.loads((out / 'finetune.json').read_text())


if __name__ == '__main__':
    # Process entry under torchrun (see _launch).
    request = json.loads(Path(sys.argv[1]).read_text())
    run = FineTune(FineTuneConfig(**request['config']), verbose=request['verbose'])
    try:
        run.run()
    finally:
        run.close()
