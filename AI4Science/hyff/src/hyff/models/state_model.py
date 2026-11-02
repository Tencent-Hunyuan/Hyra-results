"""Minimal, topology-free conditioning on the total charge and spin multiplicity.

Only intensive scalar state features are added to the initial atomic scalar
embedding. Forces remain exact derivatives of the conditioned energy. No
atomic formal-charge assignments or connectivity enter this model.
This is not a long-range electrostatics or arbitrary-fragment charge solution.
"""
from dataclasses import asdict

import torch
from torch import nn

from hyff.models.element_model import (
    AtomicNumberPartition, ElementConfig, ElementPotential, build_model as build_base, prepare_batch,
)
from hyff.models.experts import ElementPartition
from hyff.models.state_forward import reference_forward


ARCHITECTURE = "hyff_state_density_pilot"


class StatePotential(ElementPotential):
    architecture = ARCHITECTURE

    def __init__(self, config: ElementConfig, condition_spin=False):
        super().__init__(config)
        self.condition_spin = bool(condition_spin)
        self.state_projection = nn.Linear(2 if condition_spin else 1, config.hidden, bias=False)
        nn.init.zeros_(self.state_projection.weight)

    def initial_scalar(self, batch):
        scalar = self.embedding(batch["atomic_numbers"])
        count = (batch["ptr"][1:] - batch["ptr"][:-1]).to(scalar)
        charge = batch["total_charge"].to(scalar)
        if charge.shape != count.shape:
            raise ValueError("Supply one total_charge per system")
        if not bool(torch.isfinite(charge).all() & (count > 0).all()):
            raise ValueError("State conditioning requires finite charge and nonempty systems")
        features = [charge / count]
        if self.condition_spin:
            multiplicity = batch["multiplicity"].to(scalar)
            if multiplicity.shape != count.shape:
                raise ValueError("Supply one multiplicity M=2S+1 per system")
            if not bool((torch.isfinite(multiplicity) & (multiplicity >= 1) &
                         (multiplicity == multiplicity.round())).all()):
                raise ValueError("Multiplicity M=2S+1 must be a positive integer")
            features.append((multiplicity - 1.) / count)
        elif "multiplicity" in batch:
            # A charge-only checkpoint cannot stand in for a trained spin model.
            if not bool((batch["multiplicity"] == 1).all()):
                raise ValueError("This charge-only model has no trained non-singlet response")
        state = self.state_projection(torch.stack(features, dim=-1))
        return scalar + state[batch["batch"]]

    def forward(self, batch, **kwargs):
        if kwargs.get("csr") is not None or self.fused_messages or self.reference_chunk_size:
            raise ValueError("The state model supports conservative reference execution only")
        batch = prepare_batch(batch)
        partition = batch["_hyff_atomic_number_partition"]
        if not isinstance(partition, AtomicNumberPartition):
            raise TypeError("Invalid species partition")
        unsupported = set(partition.groups).difference(self.atomic_numbers)
        if unsupported:
            raise ValueError(f"Unsupported elements: {sorted(unsupported)}")
        groups = tuple(partition.groups.get(z, partition.empty) for z in self.atomic_numbers)
        bound = ElementPartition(groups, len(batch["atomic_numbers"]), partition.empty,
                                 partition.empty, partition.empty)
        for index in self.config.expert_blocks:
            self.blocks[index].update_mlp.partition = bound
        return reference_forward(self, batch, **kwargs)

    def expert_accounting(self):
        base = super().expert_accounting()
        return dict(base, architecture=ARCHITECTURE,
                    charge_conditioning=True, spin_conditioning=self.condition_spin,
                    input_fields=base["input_fields"] + ["total_charge"] +
                        (["multiplicity"] if self.condition_spin else []),
                    state_features=["Q/N", "(M-1)/N"] if self.condition_spin else ["Q/N"],
                    total_parameters=sum(p.numel() for p in self.parameters()))


def build_model(config):
    config = dict(config)
    spin = bool(config.pop("condition_spin", False))
    return StatePotential(ElementConfig(**asdict(build_base(config).config)), condition_spin=spin)


def from_base_checkpoint(saved, conditioned=True):
    config = saved["model_config"]
    model = build_model(config) if conditioned else build_base(config)
    model.to(dtype=saved["model_state"]["embedding.weight"].dtype)
    missing, unexpected = model.load_state_dict(saved["model_state"], strict=False)
    expected = ["state_projection.weight"] if conditioned else []
    if missing != expected or unexpected:
        raise ValueError(f"Unexpected checkpoint conversion: {missing}, {unexpected}")
    return model
