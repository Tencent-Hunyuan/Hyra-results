"""Experimental symmetric FiLM radial filters over the element atom experts.

This reference has no fused message backend and is not used by the HyFF named
models. Chemical affine conditioning can be materialized once for each
unordered pair; force, training and performance checks are incomplete.
"""
from dataclasses import dataclass
import copy

import torch
from torch import nn

from hyff.models.experts import ELEMENTS, L2ExpertConfig, L2ExpertPotential
from hyff.kernels.radial_table import QuinticRadialTable


PAIR_ELEMENTS = tuple((a, b) for a in range(8) for b in range(a, 8))


@dataclass(frozen=True)
class PairRadialConfig(L2ExpertConfig):
    radial_conditioning: str = 'symmetric_film'

    def __post_init__(self):
        super().__post_init__()
        if self.radial_conditioning not in ('shared', 'symmetric_film'):
            raise ValueError('Expected shared or symmetric_film radial filters')
        if self.expert_routing not in ('dense', 'element'):
            raise ValueError('This variant uses only static atom networks')


@dataclass(frozen=True)
class PairContext:
    low: torch.Tensor
    high: torch.Tensor
    table_ids: torch.Tensor
    atomic_numbers: torch.Tensor
    edge_index: torch.Tensor
    numbers_version: int
    edges_version: int

    def matches(self, batch):
        return (self.atomic_numbers is batch['atomic_numbers']
            and self.edge_index is batch['edge_index']
            and self.numbers_version == self.atomic_numbers._version
            and self.edges_version == self.edge_index._version)


@torch.no_grad()
def prepare_pair_radial_batch(batch):
    """Cache only static element-pair indices; positions remain differentiable."""
    if '_pair_radial_context' in batch:
        if not batch['_pair_radial_context'].matches(batch):
            raise ValueError('Stale radial context: prepare current atom types and edges')
        return batch
    numbers, edges = batch['atomic_numbers'], batch['edge_index']
    if (numbers.ndim != 1 or edges.ndim != 2 or edges.shape[0] != 2
            or numbers.dtype != torch.int64 or edges.dtype != torch.int64
            or numbers.device != edges.device):
        raise ValueError('Expected same-device int64 atom numbers and two-row edges')
    if not set(numbers.detach().cpu().tolist()).issubset(ELEMENTS):
        raise ValueError('Unsupported atomic number')
    lut = torch.full((119,), -1, dtype=torch.int64, device=numbers.device)
    lut[list(ELEMENTS)] = torch.arange(8, device=numbers.device)
    types = lut[numbers]
    a, b = types[edges[0]], types[edges[1]]
    low, high = torch.minimum(a, b), torch.maximum(a, b)
    table_ids = low * (15 - low) // 2 + high
    context = PairContext(low, high, table_ids, numbers, edges,
        numbers._version, edges._version)
    return {**batch, '_pair_radial_context': context}


class SymmetricFiLMRadial(nn.Module):
    def __init__(self, shared_network):
        super().__init__()
        if len(shared_network) != 3 or not isinstance(shared_network[1], nn.SiLU):
            raise ValueError('Expected an affine/SiLU/affine radial network')
        self.base = shared_network
        width = shared_network[0].weight.shape[0]
        # No random draws: both common initial tensors and RNG state are kept.
        self.scale = nn.Parameter(shared_network[0].weight.new_zeros((8, width)))
        self.shift = nn.Parameter(shared_network[0].weight.new_zeros((8, width)))
        self.context = None

    def bind(self, context):
        self.context = context

    def forward(self, radial):
        if self.context is None or len(radial) != len(self.context.low):
            raise ValueError('Bind the exact current pair context before evaluation')
        low, high = self.context.low, self.context.high
        gain = 1. + self.scale[low] + self.scale[high]
        shift = self.shift[low] + self.shift[high]
        hidden = self.base[0](radial)
        return self.base[2](self.base[1](hidden * gain + shift))

    @torch.no_grad()
    def materialize(self, a, b):
        """CPU FP64 network for numerical radial compilation, not retraining.

        Form effective matrices in FP64 from the actual learned parameters.
        Ordinary runtime reassociation can differ by floating-point roundoff;
        comparisons must check values and both radial derivatives.
        """
        if not 0 <= a < 8 or not 0 <= b < 8:
            raise ValueError('Expected two eligible element IDs')
        network = copy.deepcopy(self.base).cpu().double()
        scale = self.scale.detach().cpu().double()
        shift = self.shift.detach().cpu().double()
        gain = 1. + scale[a] + scale[b]
        network[0].weight.mul_(gain[:, None])
        network[0].bias.mul_(gain).add_(shift[a] + shift[b])
        return network.requires_grad_(False)


class PairRadialTable(nn.Module):
    def __init__(self, coefficients, cutoff):
        super().__init__()
        if coefficients.ndim != 4 or coefficients.shape[0] != 36 or coefficients.shape[2] != 6:
            raise ValueError('Expected 36 unordered-pair quintic tables')
        self.register_buffer('coefficients', coefficients)
        self.intervals, self.cutoff = coefficients.shape[1], float(cutoff)
        self.inverse_spacing = self.intervals / self.cutoff
        self.context = None

    @classmethod
    def compile(cls, rbf, bank, intervals=200, table_cutoff=None):
        tables = [QuinticRadialTable.compile(rbf, bank.materialize(a, b),
            intervals, table_cutoff) for a, b in PAIR_ELEMENTS]
        return cls(torch.stack([t.coefficients for t in tables]), tables[0].cutoff)

    def bind(self, context):
        self.context = context

    def forward(self, distance):
        if self.context is None or distance.numel() != self.context.table_ids.numel():
            raise ValueError('Bind the current pair context before table evaluation')
        coordinate = (distance * self.inverse_spacing).clamp(0., float(self.intervals))
        interval = coordinate.floor().long().clamp(max=self.intervals - 1)
        x = (coordinate - interval.to(coordinate.dtype))[:, None]
        pair = self.context.table_ids
        value = self.coefficients[pair, interval, 5]
        for power in range(4, -1, -1):
            value = value * x + self.coefficients[pair, interval, power]
        return value


class PairRadialPotential(L2ExpertPotential):
    def __init__(self, config):
        if not isinstance(config, PairRadialConfig):
            raise TypeError('Expected an explicit PairRadialConfig')
        super().__init__(config)
        if config.radial_conditioning == 'symmetric_film':
            for block in self.blocks:
                block.radial_filter = SymmetricFiLMRadial(block.radial_filter)

    def forward(self, batch, **kwargs):
        if kwargs.get('csr') is not None or self.reference_chunk_size or self.fused_messages:
            raise ValueError('This variant has no fused or chunked force backend')
        if self.config.radial_conditioning == 'symmetric_film':
            batch = prepare_pair_radial_batch(batch)
            context = batch['_pair_radial_context']
            for block in self.blocks:
                block.radial_filter.bind(context)
            if self.radial_tables is not None:
                for table in self.radial_tables[:-1]:
                    table.bind(context)
        return super().forward(batch, **kwargs)

    def compile_radial_tables(self, intervals=200, order=5):
        if order != 5:
            raise ValueError('This reference supports quintic tables only')
        if self.config.radial_conditioning == 'shared':
            return super().compile_radial_tables(intervals, order)
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError('Freeze parameters before compiling radial functions')
        parameter = next(self.parameters())
        tables = [PairRadialTable.compile(self.rbf, block.radial_filter,
            intervals, block.cutoff).to(parameter) for block in self.blocks]
        pair_linear = nn.Linear(self.config.n_rbf, 64, bias=False, device='meta')
        pair_linear.weight = nn.Parameter(self.pair_coeff.weight.detach(), requires_grad=False)
        tables.append(QuinticRadialTable.compile(self.rbf, [pair_linear], intervals).to(parameter))
        self.radial_tables = nn.ModuleList(tables)
        self.radial_table_order = order
        return self

    def enable_fused_messages(self, *args, **kwargs):
        raise NotImplementedError('No fused symmetric-radial implementation is available')
