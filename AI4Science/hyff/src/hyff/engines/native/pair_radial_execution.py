"""Inference execution adapter for the pair-radial models; the model classes are not modified.

Only each instance's message dispatch is replaced. Parameter names and all
atom-local functions are inherited unchanged. Table lookup is symmetric in
the two elements; complete coordinate and feature adjoints use that same
table in both directed roles. No edge-sized feature tensor is formed.
"""
from types import MethodType

import torch

from hyff.models.experts import L2ExpertPotential, prepare_expert_batch
from hyff.kernels.fused_messages import PreparedL2Neighbors
from hyff.models.l2_potential import L2Interaction

from hyff.models.pair_radials import PairRadialConfig, PairRadialPotential, prepare_pair_radial_batch
from hyff.models.pair_radials_io import ARCHITECTURE


def _message_forward(self, scalar, vector, tensor, radial, direction, quadrupole,
                     source, destination, edge_envelope, radial_output=None,
                     csr=None, reference=None, node_message=None):
    if csr is None and (reference is None or not hasattr(self.radial_filter, 'scale')):
        return L2Interaction.forward(self, scalar, vector, tensor, radial, direction,
            quadrupole, source, destination, edge_envelope, radial_output, csr, reference, node_message)
    n_atoms, hidden = scalar.shape[0], self.message_hidden
    if self.tensor_bottleneck:
        if self.input_is_scalar:
            message_vector = vector.new_zeros((n_atoms, 3, hidden))
            message_tensor = tensor.new_zeros((n_atoms, 5, hidden))
        else:
            message_vector = self.edge_vector_in(vector)
            message_tensor = self.edge_tensor_in(tensor)
    else:
        message_vector, message_tensor = vector, tensor
    if node_message is None:
        node_message = self.scalar_message(self.scalar_norm(scalar))
    if csr is not None:
        from hyff.kernels.pair_messages import messages
        positions, plan, table = csr
        values = messages(positions, plan.cell, plan.neighbors, plan.shifts, plan.counts,
            plan.elements, node_message, message_vector, message_tensor, table,
            self.cutoff, self.switch_width, self.input_is_scalar, plan.model.fused_message_settings)
    else:
        from hyff.kernels.pair_messages_reference import reference_messages
        displacement, valid, table, rbf, cutoff, width, chunk = reference
        values = reference_messages(displacement, source, destination, node_message,
            message_vector, message_tensor, valid, table, self.radial_filter, rbf,
            cutoff, width, self.input_is_scalar, chunk)
    return self._finish_messages(scalar, vector, tensor, *values)


class PreparedPairNeighbors(PreparedL2Neighbors):
    def __init__(self, model, batch, edges):
        self.atomic_numbers = batch['atomic_numbers']
        self.numbers_version = self.atomic_numbers._version
        if self.atomic_numbers.dtype != torch.int64:
            raise ValueError('Expected int64 atomic numbers')
        # prepare_expert_batch validates the supported elements during setup.
        if not set(self.atomic_numbers.detach().cpu().tolist()).issubset((1, 6, 7, 8, 9, 15, 16, 17)):
            raise ValueError('Unsupported atom type for a symmetric radial bank')
        self.elements = model.element_lut[self.atomic_numbers].contiguous()
        super().__init__(model, batch, edges)
        for view in self.radius_views.values():
            view.elements = self.elements

    def check_types(self, numbers):
        if numbers is not self.atomic_numbers or numbers._version != self.numbers_version:
            raise ValueError('Stale prepared atom types: build a new pair-neighbor plan')

    def __call__(self, positions):
        self.check_types(self.batch['atomic_numbers'])
        return super().__call__(positions)


class FrozenPairPotential(PairRadialPotential):
    def __init__(self, config):
        super().__init__(config)
        for block in self.blocks:
            block.forward = MethodType(_message_forward, block)

    def forward(self, batch, **kwargs):
        csr = kwargs.get('csr')
        if csr is not None:
            if not self.fused_messages or not isinstance(csr, PreparedPairNeighbors):
                raise ValueError('Expected the explicit prepared pair-neighbor backend')
            csr.check_types(batch['atomic_numbers'])
        elif self.config.radial_conditioning == 'symmetric_film':
            batch = prepare_pair_radial_batch(batch)
            context = batch['_pair_radial_context']
            for block in self.blocks:
                block.radial_filter.bind(context)
            if self.radial_tables is not None:
                for table in self.radial_tables[:-1]:
                    table.bind(context)
        return L2ExpertPotential.forward(self, batch, **kwargs)

    def enable_fused_messages(self, intervals=200, settings=(16, 8, 4, True),
                              radial_order=5, filter_skin=False):
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError('Freeze pair model parameters before enabling inference')
        if self.config.edge_hidden not in (8, 16) or radial_order != 5:
            raise ValueError('This adapter supports R8/R16 quintic execution only')
        if tuple(settings) != (16, 8, 4, True):
            raise ValueError('The declared pair layout uses settings (16,8,4,True)')
        if self.radial_tables is None:
            self.compile_radial_tables(intervals, order=radial_order)
        self.fused_messages = self.fused = True
        self.fused_message_settings = tuple(settings)
        self.filter_skin_neighbors = bool(filter_skin)
        self.inference_backend = 'l2_pair_csr_quintic'
        return self

    def prepare_energy(self, batch, edges):
        batch = prepare_expert_batch(batch, self.config.expert_routing)
        if self.fused_messages:
            return PreparedPairNeighbors(self, batch, edges)
        return lambda positions: self.energy({**batch, 'positions': positions}, edges)


def load_checkpoint(path, device='cpu'):
    raise ValueError('Checkpoint loading is not available here; use hyff.weights.load_model')


def enable_family_nodes(model, precision):
    from hyff.kernels.sparse_experts import enable_sparse_expert_nodes
    if precision != 'fp16w2':
        raise ValueError('The pair adapter requires precision fp16w2')
    return enable_sparse_expert_nodes(model)


def family_node_description(model):
    return dict(family=ARCHITECTURE, routing=model.config.expert_routing,
        radial_conditioning=model.config.radial_conditioning,
        neighbor_width=model.config.edge_hidden,
        active_atom_experts=1, implementation='frozen element FP16w2 plus symmetric radial banks')
