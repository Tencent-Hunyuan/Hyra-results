"""Experimental sparse invariant atom networks (experts) over the L2 message transport.

Element routing is independent of coordinates. Environment routing uses a
periodic cardinal cubic B-spline: exactly four neighboring experts contribute,
and entering/leaving coefficients and their first two derivatives vanish.
Routing derivatives are part of the energy gradient. No force head, detached
router, auxiliary fit, capacity truncation, or geometry-dependent hard top-k
is used. This module does not modify the L2 implementation.

The grouped implementation is the differentiable reference/training path.
Environment grouping includes a device-to-host count transfer, so its speed is
not that of a fixed-shape CUDA-graph backend.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn

from hyff.models.l2_potential import L2Config, L2Potential, _linear


ELEMENTS = (1, 6, 7, 8, 9, 15, 16, 17)


@dataclass(frozen=True)
class L2ExpertConfig(L2Config):
    expert_routing: str = "element"
    expert_count: int = 8
    expert_hidden: int = 64
    expert_blocks: tuple[int, ...] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.expert_routing not in ("dense", "element", "spline"):
            raise ValueError("Expected dense, element, or spline routing")
        if type(self.expert_hidden) is not int or self.expert_hidden < 1:
            raise ValueError("Expert width must be a positive integer")
        if type(self.expert_count) is not int:
            raise ValueError("Expert count must be an integer")
        if self.expert_routing == "dense" and self.expert_count != 1:
            raise ValueError("Dense routing has exactly one network")
        if self.expert_routing == "element" and self.expert_count != len(ELEMENTS):
            raise ValueError("Element routing covers the same eight eligible elements")
        if self.expert_routing == "spline" and self.expert_count < 4:
            raise ValueError("Cubic spline routing requires at least four experts")
        blocks = tuple(range(self.interactions)) if self.expert_blocks is None else tuple(self.expert_blocks)
        if (not blocks or len(set(blocks)) != len(blocks) or
                any(type(i) is not int or not 0 <= i < self.interactions for i in blocks)):
            raise ValueError("Expert blocks must be distinct interaction indices")
        object.__setattr__(self, "expert_blocks", blocks)


def cubic_weights(t: torch.Tensor):
    """Four cardinal coefficients on one closed interval, including its ends."""
    return torch.stack(((1. - t).pow(3),
        3. * t.pow(3) - 6. * t.square() + 4.,
        -3. * t.pow(3) + 3. * t.square() + 3. * t + 1.,
        t.pow(3)), dim=-1) / 6.


def cubic_routes(coordinate: torch.Tensor, count: int):
    """Return four periodic expert indices and C2 partition-of-unity weights.

    ``floor`` is used only to choose polynomial pieces. Differentiating through
    ``coordinate - floor(coordinate)`` keeps the derivative within a piece;
    the cardinal basis matches through order two at each integer boundary.
    """
    cell = torch.floor(coordinate)
    t = coordinate - cell
    weights = cubic_weights(t)
    offsets = torch.arange(-1, 3, device=coordinate.device)
    indices = (cell.long()[..., None] + offsets) % count
    return indices, weights


@dataclass(frozen=True)
class ElementPartition:
    groups: tuple[torch.Tensor, ...]
    atoms: int
    order: torch.Tensor
    starts: torch.Tensor
    tiles: torch.Tensor


@torch.no_grad()
def element_partition(atomic_numbers: torch.Tensor) -> ElementPartition:
    """Setup-only exact partition; no coordinates or molecule identities."""
    if atomic_numbers.ndim != 1:
        raise ValueError("Atomic numbers must be a vector")
    # This transfer occurs once per cached training batch or prepared MD graph.
    numbers = atomic_numbers.detach().cpu()
    cpu_groups = tuple(torch.nonzero(numbers == z, as_tuple=False).flatten() for z in ELEMENTS)
    counts = [len(group) for group in cpu_groups]
    if sum(counts) != len(numbers):
        raise ValueError("Unsupported atomic number in element router")
    starts = [0]
    tiles = [0]
    for count in counts:
        starts.append(starts[-1] + count)
        tiles.append(tiles[-1] + (count + 63) // 64)
    order = torch.cat(cpu_groups).to(atomic_numbers.device)
    # Views share the owned order buffer. Frozen products consume its constant
    # shape plus exact counts, so validation batches with differing element
    # populations do not trigger a different compiled graph for each group.
    groups = tuple(order[a:b] for a, b in zip(starts, starts[1:]))
    return ElementPartition(groups, len(numbers), order,
        torch.tensor(starts, dtype=torch.int32, device=atomic_numbers.device),
        torch.tensor(tiles, dtype=torch.int32, device=atomic_numbers.device))


def prepare_expert_batch(batch: dict, routing: str) -> dict:
    if routing == "element" and "_element_partition" not in batch:
        return {**batch, "_element_partition": element_partition(batch["atomic_numbers"])}
    return batch


def _network(inputs, hidden, outputs):
    return nn.Sequential(_linear(inputs, hidden), nn.SiLU(), _linear(hidden, outputs))


class ElementExpertMLP(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        # All experts start as the same function, so element routing starts from
        # the dense model of the same width; element gradients then separate them.
        first = _network(inputs, hidden, outputs)
        self.experts = nn.ModuleList([first, *(copy.deepcopy(first) for _ in ELEMENTS[1:])])
        self.out_features = outputs
        self.partition: ElementPartition | None = None

    def forward(self, values):
        if self.partition is None or self.partition.atoms != values.shape[0]:
            raise ValueError("Bind the exact element partition before an atom update")
        output = values.new_zeros((values.shape[0], self.out_features))
        for expert, indices in zip(self.experts, self.partition.groups):
            if indices.numel():
                output = output.index_copy(0, indices, expert(values.index_select(0, indices)))
        return output


class SplineExpertMLP(nn.Module):
    def __init__(self, inputs, hidden, outputs, count):
        super().__init__()
        self.count, self.out_features = count, outputs
        # Replicated smooth initial functions avoid random routing jumps even
        # before learning. Gradients differ with spline weight and environment.
        first = _network(inputs, hidden, outputs)
        self.experts = nn.ModuleList([first, *(copy.deepcopy(first) for _ in range(count - 1))])
        self.router = nn.Linear(inputs, 1)
        nn.init.normal_(self.router.weight, std=1. / math.sqrt(inputs))
        nn.init.zeros_(self.router.bias)

    def routes(self, values):
        # A smooth bounded normalization keeps large invariant magnitudes from
        # immediately saturating the router. No cross-atom normalization.
        normalized = values * torch.rsqrt(1. + values.square().mean(-1, keepdim=True))
        q = self.count * torch.sigmoid(self.router(normalized).squeeze(-1))
        indices, weights = cubic_routes(q, self.count)
        return q, indices, weights

    def forward_with_routes(self, values, indices, weights):
        """Sparse reference: evaluate only selected (atom, expert) pairs."""
        if indices.shape != (len(values), 4) or weights.shape != indices.shape:
            raise ValueError("Expected four spline contributions per atom")
        flat_experts = indices.reshape(-1)
        order = torch.argsort(flat_experts, stable=True)
        counts = torch.bincount(flat_experts, minlength=self.count).detach().cpu().tolist()
        output = values.new_zeros((len(values), self.out_features))
        start = 0
        for expert, count in zip(self.experts, counts):
            if count:
                assignments = order[start:start + count]
                atoms = assignments // 4
                coefficient = weights.reshape(-1).index_select(0, assignments)
                part = expert(values.index_select(0, atoms)) * coefficient[:, None]
                output = output.index_add(0, atoms, part)
            start += count
        return output

    def forward(self, values):
        _, indices, weights = self.routes(values)
        return self.forward_with_routes(values, indices, weights)


class L2ExpertPotential(L2Potential):
    def __init__(self, config: L2ExpertConfig):
        if not isinstance(config, L2ExpertConfig):
            raise TypeError("Expected an L2ExpertConfig")
        # Construct all common tensors first, keeping their initial values
        # and random draw sequence the same for every routing choice.
        super().__init__(config)
        for index in config.expert_blocks:
            old = self.blocks[index].update_mlp
            inputs, outputs = old[0].weight.shape[1], old[-1].weight.shape[0]
            if config.expert_routing == "dense":
                replacement = _network(inputs, config.expert_hidden, outputs)
            elif config.expert_routing == "element":
                replacement = ElementExpertMLP(inputs, config.expert_hidden, outputs)
            else:
                replacement = SplineExpertMLP(inputs, config.expert_hidden, outputs, config.expert_count)
            self.blocks[index].update_mlp = replacement

    def forward(self, batch, **kwargs):
        if self.config.expert_routing == "element":
            partition = batch.get("_element_partition")
            if partition is None:
                partition = element_partition(batch["atomic_numbers"])
            for index in self.config.expert_blocks:
                self.blocks[index].update_mlp.partition = partition
        return super().forward(batch, **kwargs)

    def prepare_energy(self, batch, edges):
        # The prepared callable owns the routed batch and hence its index
        # tensors for the whole CUDA graph lifetime. Eager calls may rebind
        # module.partition without invalidating a previously captured graph.
        routed = prepare_expert_batch(batch, self.config.expert_routing)
        return super().prepare_energy(routed, edges)

    def expert_accounting(self):
        active = 4 if self.config.expert_routing == "spline" else 1
        rows = []
        for index in self.config.expert_blocks:
            layer = self.blocks[index].update_mlp
            subnet = layer.experts[0] if hasattr(layer, "experts") else layer
            inputs, outputs = subnet[0].weight.shape[1], subnet[-1].weight.shape[0]
            h = self.config.expert_hidden
            rows.append(dict(block=index, input_width=inputs, output_width=outputs,
                total_expert_parameters=sum(p.numel() for p in layer.parameters()),
                active_experts_per_atom=active, hidden_per_expert=h,
                active_affine_macs_per_atom=active * h * (inputs + outputs),
                router_macs_per_atom=inputs if self.config.expert_routing == "spline" else 0))
        return dict(routing=self.config.expert_routing, experts=self.config.expert_count,
            total_parameters=sum(p.numel() for p in self.parameters()), blocks=rows,
            caution="Per-atom arithmetic excludes routing, memory movement, tensor transport and its coordinate VJP; measure full E+F and MD.")


def load_l2_expert_checkpoint(path, device="cpu"):
    raise ValueError('Checkpoint loading is not available here; use hyff.weights.load_model')
