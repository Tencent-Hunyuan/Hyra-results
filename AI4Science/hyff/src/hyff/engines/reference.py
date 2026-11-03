"""Reference loader and single-system batches for the plain PyTorch HyFF models.

Builds a model from a weight file and its architecture JSON
(hyff.weights.read_adapter_export), checks that every loaded tensor is
byte-identical to the weight file and never casts a tensor.
The global state enters as in training: the features [Q/N, (M-1)/N] pass through
``state_projection`` and are added to the atomic embedding
(hyff.models.state_model.StatePotential.initial_scalar).

``bind_global_state`` fixes Q, M and N for CUDA-graph execution through the
``prepare_energy`` hook of hyff.engines.force_engine.ForceEngine: the species
partition and the initial scalar are prepared outside capture, because
``physical()`` strips ``multiplicity``/``n_atoms`` from engine batches and the
training ``initial_scalar`` performs host synchronizations.
"""
from __future__ import annotations

from pathlib import Path
from types import MethodType

import numpy as np
import torch

from hyff.weights import sha256 as digest
from hyff.models.element_model import prepare_batch
from hyff.models.base_model import POLICY as LAYERNORM_POLICY
from hyff.models.base_model import install as install_fp64_layernorm
from hyff.data import physical
from hyff.engines.force_engine import ForceEngine
from hyff.graph import NeighborCache, energy_forces
from hyff.weights import read_adapter_export

FACTORIES = {
    'hyff.models.base_model:build_model': 'large',
}
EXPORT_KEYS = ('architecture', 'model_factory', 'model_config', 'model_state', 'numerical_execution')
ELEMENTS = tuple(range(1, 84))
STATE_FEATURES = ('Q/N', '(M-1)/N')


def read_export(path):
    saved = read_adapter_export(path)
    missing = [key for key in EXPORT_KEYS if key not in saved]
    if missing:
        raise KeyError(f'Not a HyFF weight export; missing {missing}')
    if saved['model_factory'] not in FACTORIES:
        raise ValueError(f'Unsupported model factory {saved["model_factory"]!r}; expected one of {sorted(FACTORIES)}')
    return saved


def family(saved):
    return FACTORIES[saved['model_factory']]


def _tensor_bytes(value):
    return value.detach().cpu().contiguous().numpy().tobytes()


def verify_state(model, saved):
    """Bytewise comparison of every export tensor with the live model; returns the count."""
    actual = model.state_dict()
    export = saved['model_state']
    unexpected = sorted(set(export).difference(actual))
    if unexpected:
        raise ValueError('Export tensors absent from the model: ' + str(unexpected))
    changed = []
    for name, value in export.items():
        live = actual[name]
        if live.dtype != value.dtype or live.shape != value.shape or _tensor_bytes(live) != _tensor_bytes(value):
            changed.append(name)
    if changed:
        raise ValueError('Loaded tensors differ from the export: ' + str(changed))
    return len(export)


def build(saved):
    from hyff.models.factory import build_model
    model = build_model(dict(model_id=saved['model_factory'], model_config=saved['model_config']))
    policy = saved.get('numerical_execution')
    if not policy or policy.get('policy') != LAYERNORM_POLICY:
        raise ValueError(f'Export records an unknown numerical execution policy: {policy!r}')
    if getattr(model, 'numerical_execution', None) is None:
        # Constructors without the FP64-internal LayerNorm policy get it installed here.
        install_fp64_layernorm(model)
    if model.numerical_execution['layernorm_modules'] != policy['layernorm_modules']:
        raise ValueError('LayerNorm module set differs from the export policy')
    live = model.state_dict()
    export = saved['model_state']
    if set(live) != set(export):
        raise ValueError(f'State schema differs: missing {sorted(set(live) - set(export))}, unexpected {sorted(set(export) - set(live))}')
    for name, value in export.items():
        if live[name].dtype != value.dtype or live[name].shape != value.shape:
            raise ValueError(f'{name}: factory tensor {live[name].dtype}{tuple(live[name].shape)} vs export {value.dtype}{tuple(value.shape)}; no cast is performed')
    model.load_state_dict(export, strict=True)
    verify_state(model, saved)
    if tuple(model.atomic_numbers) != ELEMENTS or not getattr(model, 'condition_spin', False):
        raise ValueError('Explicit 83-element Q/M model required (the trainers enforce the same)')
    return model.eval().requires_grad_(False)


def dependency_radius(model):
    return max(float(model.cutoff), float(sum(model.layer_cutoffs)))


def load(checkpoint, device='cpu'):
    """Build the exported model on `device` with unchanged parameter dtypes (FP32 learned tensors)."""
    path = Path(checkpoint).resolve()
    saved = read_export(path)
    model = build(saved).to(device)
    verified = verify_state(model, saved)
    model.export_metadata = dict(
        checkpoint=str(path), checkpoint_sha256=digest(path), family=family(saved),
        architecture=saved['architecture'], model_factory=saved['model_factory'],
        selection_kind=saved.get('selection_kind'), step=saved.get('step'),
        frame_epochs=saved.get('frame_epochs'), energy_output=saved.get('energy_output'),
        atomic_numbers=list(model.atomic_numbers), condition_spin=True, state_features=list(STATE_FEATURES),
        cutoff_A=float(model.cutoff), layer_cutoffs_A=[float(r) for r in model.layer_cutoffs],
        dependency_radius_A=dependency_radius(model),
        numerical_execution=model.numerical_execution,
        parameters=sum(p.numel() for p in model.parameters()),
        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        float64_buffers=sorted(name for name, b in model.named_buffers() if b.dtype == torch.float64),
        checkpoint_tensors_verified=verified, learned_weights_changed=False)
    return model, saved


def state_features(n_atoms, total_charge=0., multiplicity=1):
    """The intensive inputs of StatePotential.initial_scalar for one system."""
    n = int(n_atoms)
    m = int(multiplicity)
    if n < 1 or m < 1 or m != multiplicity or not np.isfinite(total_charge):
        raise ValueError('Positive atom count, finite charge and integer multiplicity M=2S+1 >= 1 required')
    return [float(total_charge) / n, (m - 1.) / n]


def system_batch(positions, atomic_numbers, cell=None, total_charge=0., multiplicity=1,
                 device='cpu', dtype=torch.float32):
    """One physical system in the training batch layout (single graph)."""
    positions = torch.as_tensor(np.asarray(positions), dtype=dtype, device=device)
    numbers = torch.as_tensor(np.asarray(atomic_numbers), dtype=torch.int64, device=device)
    if positions.ndim != 2 or positions.shape[1] != 3 or numbers.ndim != 1 or len(numbers) != len(positions):
        raise ValueError('positions must be (N,3) and atomic_numbers (N,)')
    n = len(numbers)
    state_features(n, total_charge, multiplicity)
    batch = dict(positions=positions, atomic_numbers=numbers,
                 batch=torch.zeros(n, dtype=torch.int64, device=device),
                 ptr=torch.tensor([0, n], dtype=torch.int64, device=device),
                 n_atoms=torch.tensor([n], dtype=torch.float32, device=device),
                 total_charge=torch.tensor([float(total_charge)], dtype=torch.float32, device=device),
                 multiplicity=torch.tensor([int(multiplicity)], dtype=torch.int64, device=device))
    if cell is not None:
        cell = torch.as_tensor(np.asarray(cell), dtype=dtype, device=device).reshape(1, 3, 3)
        batch.update(cell=cell, pbc=torch.ones((1, 3), dtype=torch.bool, device=device))
    return batch


# ---------------------------------------------------------------------------------------------
# Engine path: fixed whole-system state, prepared outside CUDA graph capture.

def _initial_scalar(self, batch):
    if '_engine_initial_scalar' in batch:
        return batch['_engine_initial_scalar']
    return self._original_initial_scalar(batch)


def complete_state_batch(model, batch):
    """Add the bound state fields to an engine batch exactly as the training collate supplies them."""
    context = model._engine_global_context
    n = len(batch['positions'])
    if len(batch['ptr']) != 2 or context['n_atoms'] != n:
        raise ValueError('The bound global state describes one physical system with a fixed atom count')
    device = batch['positions'].device
    charge = torch.tensor([context['total_charge']], dtype=torch.float32, device=device)
    if 'total_charge' in batch and not torch.equal(batch['total_charge'].to(torch.float32), charge):
        raise ValueError('Batch total charge differs from the bound global state')
    completed = {**batch, 'total_charge': charge,
                 'multiplicity': torch.tensor([context['multiplicity']], dtype=torch.int64, device=device)}
    completed.setdefault('n_atoms', torch.tensor([n], dtype=torch.float32, device=device))
    return completed


def _prepare_energy(self, static_batch, static_edges):
    """ForceEngine hook: partition and initial scalar once, outside capture; then capturable energy."""
    prepared = prepare_batch(complete_state_batch(self, static_batch))
    with torch.no_grad():
        prepared['_engine_initial_scalar'] = self._original_initial_scalar(prepared).detach()

    def evaluate(positions):
        return self.energy({**prepared, 'positions': positions}, static_edges)
    return evaluate


def bind_global_state(model, n_atoms, total_charge=0., multiplicity=1):
    """Fix Q, M and N for graph-captured execution; the batch-field path stays available."""
    features = state_features(n_atoms, total_charge, multiplicity)
    context = dict(n_atoms=int(n_atoms), total_charge=float(total_charge), multiplicity=int(multiplicity),
                   state_features=features)
    model._engine_global_context = context
    if not hasattr(model, '_original_initial_scalar'):
        model._original_initial_scalar = model.initial_scalar
    model.initial_scalar = MethodType(_initial_scalar, model)
    model.prepare_energy = MethodType(_prepare_energy, model)
    return context


__all__ = ['read_export', 'family', 'build', 'load', 'verify_state', 'state_features', 'system_batch',
           'bind_global_state', 'complete_state_batch']
