"""M26/L4A0 geometric backbones with explicitly configurable chemical species.

All species have distinct scalar embeddings, node experts, element offsets and
unordered-pair radial coefficients. Charge entries of the batch are not used:
this is an E(Z, R) potential (hyff.models.state_model adds charge and spin
conditioning). No fixed covalent graph or prediction correction is used.

The scalar/vector/STF blocks are the L2Interaction implementation. The energy
evaluation in ``hyff.models.element_forward`` is a copy of L2Potential.forward
with only its pair-table stride generalized. This reference supports training
and conservative eager inference; the eight-species fused kernels are not
dispatched here.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn

from hyff.models.experts import ElementPartition, _network
from hyff.models.l2_potential import L2Config, L2Potential
from hyff.models.radial_capacity import expand_network


ARCHITECTURE = "hyff_element_complete"
LEGACY_ELEMENTS = (1, 6, 7, 8, 9, 15, 16, 17)
# Default 17-element species list of ElementConfig; a model_config can list
# any species explicitly.
DEFAULT_ELEMENTS = (1, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 35, 53)


@dataclass(frozen=True)
class ElementConfig(L2Config):
    backbone: str = "m26"
    atomic_numbers: tuple[int, ...] = DEFAULT_ELEMENTS
    expert_routing: str = "element"
    expert_count: int | None = None
    expert_hidden: int = 64
    expert_blocks: tuple[int, ...] | None = None
    radial_conditioning: str = "shared"
    radial_hidden: int | None = None

    def __post_init__(self):
        super().__post_init__()
        elements = tuple(self.atomic_numbers)
        if (not elements or len(set(elements)) != len(elements) or
                any(type(z) is not int or not 1 <= z <= 118 for z in elements)):
            raise ValueError("atomic_numbers must contain distinct supported atomic numbers")
        object.__setattr__(self, "atomic_numbers", elements)
        if self.backbone not in ("m26", "l4a0", "custom"):
            raise ValueError("Expected m26, l4a0, or an explicitly custom backbone")
        if self.expert_routing != "element" or self.radial_conditioning != "shared":
            raise ValueError("This implementation requires element experts and shared radial filters")
        if self.expert_count is not None and self.expert_count != len(elements):
            raise ValueError("expert_count must match the explicit atomic_numbers list")
        object.__setattr__(self, "expert_count", len(elements))
        if type(self.expert_hidden) is not int or self.expert_hidden < 1:
            raise ValueError("expert_hidden must be a positive integer")
        blocks = tuple(range(self.interactions)) if self.expert_blocks is None else tuple(self.expert_blocks)
        if (not blocks or len(set(blocks)) != len(blocks) or
                any(type(i) is not int or not 0 <= i < self.interactions for i in blocks)):
            raise ValueError("expert_blocks must contain distinct valid block indices")
        object.__setattr__(self, "expert_blocks", blocks)
        initial = self.edge_hidden or self.hidden
        width = initial if self.radial_hidden is None else self.radial_hidden
        if type(width) is not int or width < initial:
            raise ValueError("radial_hidden must be at least the message width")
        object.__setattr__(self, "radial_hidden", width)


@dataclass(frozen=True)
class AtomicNumberPartition:
    atomic_numbers: torch.Tensor
    version: int
    groups: Mapping[int, torch.Tensor]
    empty: torch.Tensor

    def matches(self, numbers):
        return numbers is self.atomic_numbers and numbers._version == self.version


@torch.no_grad()
def prepare_batch(batch: dict) -> dict:
    """Cache an exact species partition; leave positions and labels untouched.

    This setup step accepts a ``Z`` alias and is independent of a model's
    configured species order. Its host transfer happens once per prepared
    batch, never inside a force/parameter derivative. Cached species mutations
    are rejected instead of silently reusing an incorrect routing partition.
    """
    if "atomic_numbers" not in batch:
        if "Z" not in batch:
            raise KeyError("Expected atomic_numbers or Z")
        batch = {**batch, "atomic_numbers": batch["Z"]}
    numbers = batch["atomic_numbers"]
    if numbers.ndim != 1 or numbers.dtype != torch.int64:
        raise ValueError("Expected a one-dimensional int64 atomic_numbers tensor")
    cached = batch.get("_hyff_atomic_number_partition")
    if cached is not None:
        if not isinstance(cached, AtomicNumberPartition) or not cached.matches(numbers):
            raise ValueError("Stale species partition: prepare current atomic_numbers")
        return batch
    cpu = numbers.detach().cpu()
    present = tuple(int(z) for z in torch.unique(cpu).tolist())
    if any(not 1 <= z <= 118 for z in present):
        raise ValueError("Atomic numbers must lie in 1..118")
    groups = {z: torch.nonzero(cpu == z, as_tuple=False).flatten().to(numbers.device) for z in present}
    partition = AtomicNumberPartition(numbers, numbers._version, groups,
                                     torch.empty(0, dtype=torch.long, device=numbers.device))
    return {**batch, "_hyff_atomic_number_partition": partition}


class ElementExpertMLP(nn.Module):
    def __init__(self, inputs, hidden, outputs, count):
        super().__init__()
        first = _network(inputs, hidden, outputs)
        self.experts = nn.ModuleList([first, *(copy.deepcopy(first) for _ in range(count - 1))])
        self.out_features = outputs
        self.partition = None

    def forward(self, values):
        if self.partition is None or self.partition.atoms != values.shape[0]:
            raise ValueError("Bind the exact species partition before an atom update")
        output = values.new_zeros((values.shape[0], self.out_features))
        for expert, indices in zip(self.experts, self.partition.groups):
            if indices.numel():
                output = output.index_copy(0, indices, expert(values.index_select(0, indices)))
        return output


class ElementPotential(L2Potential):
    architecture = ARCHITECTURE

    def __init__(self, config: ElementConfig):
        if not isinstance(config, ElementConfig):
            raise TypeError("Expected an explicit ElementConfig")
        # This keeps the initialization order of the eight-element model,
        # including all geometric blocks and readout heads.
        super().__init__(config)
        self.n_elements = len(config.atomic_numbers)
        self.atomic_numbers = config.atomic_numbers
        for index in config.expert_blocks:
            old = self.blocks[index].update_mlp
            self.blocks[index].update_mlp = ElementExpertMLP(
                old[0].weight.shape[1], config.expert_hidden,
                old[-1].weight.shape[0], self.n_elements)
        generator = torch.Generator(device="cpu")
        generator.set_state(torch.random.get_rng_state())
        for block in self.blocks:
            block.radial_filter = expand_network(block.radial_filter, config.radial_hidden, generator)

        lut = torch.full((119,), -1, dtype=torch.long)
        lut[list(self.atomic_numbers)] = torch.arange(self.n_elements)
        self.element_lut = lut
        if self.n_elements != 8:
            pair = nn.Embedding(self.n_elements * self.n_elements, config.n_rbf, device="meta")
            pair.weight = nn.Parameter(torch.zeros(self.n_elements * self.n_elements, config.n_rbf))
            self.pair_coeff = pair
        # The low*n_elements+high layout keeps the state-dict layout of the
        # eight-element model. Lower-triangle rows are inactive;
        # n*(n+1)/2 distinct unordered species pairs are reachable.

    def forward(self, batch, **kwargs):
        if kwargs.get("csr") is not None or self.fused_messages or self.reference_chunk_size:
            raise ValueError("Element-complete reference has no generalized fused inference backend")
        batch = prepare_batch(batch)
        z_partition = batch["_hyff_atomic_number_partition"]
        unsupported = set(z_partition.groups).difference(self.atomic_numbers)
        if unsupported:
            raise ValueError(f"Elements absent from explicit model config: {sorted(unsupported)}")
        groups = tuple(z_partition.groups.get(z, z_partition.empty) for z in self.atomic_numbers)
        # The plain experts read only the groups and the atom count; the other
        # partition fields are left empty, and no eight-element kernel runs.
        partition = ElementPartition(groups, len(batch["atomic_numbers"]),
                                     z_partition.empty, z_partition.empty, z_partition.empty)
        for index in self.config.expert_blocks:
            self.blocks[index].update_mlp.partition = partition
        from hyff.models.element_forward import reference_forward
        return reference_forward(self, batch, **kwargs)

    def compile_radial_tables(self, intervals=200, order=5):
        if order != 5:
            raise ValueError("Only the quintic radial table approximation (order=5) is supported")
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError("Freeze parameters before compiling radial tables")
        from hyff.kernels.radial_table import QuinticRadialTable
        parameter = next(self.parameters())
        tables = [QuinticRadialTable.compile(self.rbf, block.radial_filter, intervals,
                  table_cutoff=block.cutoff).to(parameter) for block in self.blocks]
        pair_linear = nn.Linear(self.config.n_rbf, self.n_elements ** 2, bias=False, device="meta")
        pair_linear.weight = nn.Parameter(self.pair_coeff.weight.detach(), requires_grad=False)
        tables.append(QuinticRadialTable.compile(self.rbf, [pair_linear], intervals).to(parameter))
        self.radial_tables = nn.ModuleList(tables)
        self.radial_table_order = order
        return self

    def enable_fused_messages(self, *args, **kwargs):
        raise NotImplementedError("Generalized-species frozen kernels require separate validation")

    def expert_accounting(self):
        return dict(architecture=ARCHITECTURE, atomic_numbers=list(self.atomic_numbers),
                    geometric_aggregations=len(self.blocks), requires_covalent_topology=False,
                    input_fields=["atomic_numbers", "positions", "geometric_neighbors"],
                    charge_conditioning=False, experts=self.n_elements,
                    active_experts_per_atom=1,
                    unordered_pair_functions=self.n_elements * (self.n_elements + 1) // 2,
                    total_parameters=sum(p.numel() for p in self.parameters()))


def build_model(model_config: dict) -> ElementPotential:
    values = dict(model_config)
    backbone = values.get("backbone", "m26")
    if backbone in ("m26", "l4a0"):
        large = backbone == "l4a0"
        defaults = dict(hidden=128 if large else 64, tensor_hidden=64 if large else 32,
                        edge_hidden=16 if large else 8, n_rbf=32, interactions=4 if large else 3,
                        cutoff=4.5, switch_width=1., layer_cutoffs=(3.5, 3.5, 4.5, 4.5) if large
                        else (3.5, 3.5, 4.5), atomic_refinements=0, higher_invariants=True,
                        expert_hidden=128 if large else 64, radial_hidden=128 if large else 64)
        values = {**defaults, **values}
    return ElementPotential(ElementConfig(**values))


def load_checkpoint(path, device="cpu"):
    raise ValueError('Checkpoint loading is not available here; use hyff.weights.load_model')
