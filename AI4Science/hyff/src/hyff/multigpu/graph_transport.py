"""Bounded root-to-local device neighbor transport; imports are stdlib-only.

Only the JSON header and scalar control packets reach CPU. Global edge/shift/
membership payloads stay on the caller-selected device, in bounded chunks.
Correctness relies on a trusted broadcast of the root-validated stream plus
independent local builder checks; the device graph itself is not hashed.
"""
from __future__ import annotations

import hashlib
import json
import math


HEADER_SCHEMA = 'LARGO_DEVICE_NEIGHBOR_HEADER'
WIRE_SCHEMA = 'LARGO_DEVICE_NEIGHBOR_TRANSPORT'
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_CHUNK_BYTES = 64 * 1024 * 1024
MAX_RADII = 256
MAX_CHUNK_SIZE = 1048576
_DESCRIPTOR_SIZE = 15
_HEADER_FIELDS = frozenset(('schema', 'epoch', 'generation', 'global_ids', 'world_size',
    'owners', 'layer_radii', 'skin', 'supplied_radius', 'rank_order', 'padding_multiple',
    'reserve_factor', 'global_edge_count', 'supplied_incoming_counts',
    'weighted_incoming_counts', 'radii', 'certificate'))
_RADIUS_FIELDS = frozenset(('radius', 'multiplicity', 'incoming_counts', 'edge_count',
    'base_capacity', 'per_rank_edge_counts', 'per_rank_edge_capacities'))


class DeviceGraphTransportError(RuntimeError):
    def __init__(self, code, *, collective_safe=True):
        self.code = code
        self.collective_safe = collective_safe
        self.state_advance_allowed = False
        super().__init__(code)


def _need(condition):
    if not condition:
        raise ValueError('DEVICE_GRAPH_LOCAL_CONTRACT')


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _local(control, code, operation):
    """Every fallible local phase joins this scalar gate before more traffic."""
    result, failed = None, False
    try:
        result = operation()
    except Exception:
        failed = True
    try:
        shared = control.max_code(int(failed))
    except BaseException as error:
        raise DeviceGraphTransportError('MD_DEVICE_GRAPH_CONTROL_FAILED', collective_safe=False) from error
    if shared:
        raise DeviceGraphTransportError(code)
    return result


def _broadcast(control, tensor):
    if control.world_size > 1:
        try:
            control.dist.broadcast(tensor, src=control.root, group=control.group)
        except BaseException as error:
            # A failed collective is fatal. Do not enqueue a recovery collective
            # whose peers may never reach it; the launcher's timeouts handle that case.
            raise DeviceGraphTransportError('MD_DEVICE_GRAPH_BROADCAST_FAILED', collective_safe=False) from error


def _json_certificate(value, depth=0, budget=None):
    if budget is None:
        budget = [4096]
    budget[0] -= 1
    _need(depth <= 8 and budget[0] >= 0)
    if value is None or type(value) is bool:
        return
    if type(value) is str:
        _need(len(value) <= 4096)
    elif type(value) is int:
        _need(-(2**63) <= value < 2**63)
    elif type(value) is float:
        _need(math.isfinite(value))
    elif type(value) is list:
        for item in value:
            _json_certificate(item, depth+1, budget)
    elif type(value) is dict:
        for name, item in value.items():
            _need(type(name) is str and len(name) <= 128)
            _json_certificate(item, depth+1, budget)
    else:
        raise ValueError('DEVICE_GRAPH_NON_JSON_CERTIFICATE')


def _counts(values, length, maximum):
    _need(type(values) is list and len(values) == length and
          all(_integer(value, 0, maximum) for value in values))


def _validate_header(header, *, atoms, world_size, max_matrix_entries, expected_epoch, expected_generation):
    _need(type(header) is dict and set(header) == _HEADER_FIELDS and header['schema'] == HEADER_SCHEMA)
    _need(type(header['epoch']) is int and header['epoch'] == expected_epoch and
          type(header['generation']) is int and header['generation'] == expected_generation and
          type(header['world_size']) is int and header['world_size'] == world_size)
    ids, owners = header['global_ids'], header['owners']
    _need(type(ids) is list and len(ids) == atoms and
          all(_integer(value, -(2**63), 2**63-1) for value in ids) and len(set(ids)) == atoms)
    _counts(owners, atoms, world_size-1)
    order = header['rank_order']
    _need(type(order) is list and len(order) == world_size and
          all(type(value) is int for value in order) and sorted(order) == list(range(world_size)))
    _need(type(header['padding_multiple']) is int and header['padding_multiple'] == 8 and
          type(header['reserve_factor']) is float and header['reserve_factor'] == 1.2)
    layers, rows = header['layer_radii'], header['radii']
    _need(type(layers) is list and 1 <= len(layers) <= 4096 and all(_finite(value, positive=True) for value in layers))
    _need(type(rows) is list and 1 <= len(rows) <= MAX_RADII and
          _finite(header['skin']) and _finite(header['supplied_radius'], positive=True))
    edges = header['global_edge_count']
    _need(_integer(edges, 0, max_matrix_entries))
    supplied, weighted = header['supplied_incoming_counts'], header['weighted_incoming_counts']
    _counts(supplied, atoms, edges)
    _counts(weighted, atoms, max_matrix_entries*len(layers))
    _need(sum(supplied) == edges)
    accumulated = [0]*atoms
    unique_radii = set()
    total_multiplicity = 0
    capacity_bound = 8 + (6*max_matrix_entries+4)//5
    for row in rows:
        _need(type(row) is dict and set(row) == _RADIUS_FIELDS and _finite(row['radius'], positive=True))
        _need(row['radius'] not in unique_radii)
        unique_radii.add(row['radius'])
        _need(_integer(row['multiplicity'], 1, len(layers)))
        total_multiplicity += row['multiplicity']
        _need(_integer(row['edge_count'], 0, edges))
        _counts(row['incoming_counts'], atoms, edges)
        _counts(row['per_rank_edge_counts'], world_size, edges)
        _counts(row['per_rank_edge_capacities'], world_size, capacity_bound)
        _need(sum(row['incoming_counts']) == row['edge_count'] == sum(row['per_rank_edge_counts']))
        expected = [0]*world_size
        for index, count in enumerate(row['incoming_counts']):
            _need(count <= supplied[index])
            expected[owners[index]] += count
            accumulated[index] += row['multiplicity']*count
        _need(expected == row['per_rank_edge_counts'])
        _need(_integer(row['base_capacity'], 0, capacity_bound) and row['base_capacity'] % 8 == 0)
        _need(all(capacity >= count and capacity % 8 == 0
                  for capacity, count in zip(row['per_rank_edge_capacities'], row['per_rank_edge_counts'])))
    _need(total_multiplicity == len(layers) and accumulated == weighted)
    _need(type(header['certificate']) is dict)
    _json_certificate(header['certificate'])
    return len(rows), edges


def _graph_metadata(graph, *, atoms, max_matrix_entries, max_attempts):
    capacity, degree, bins = graph.capacity, graph.maximum_degree, graph.spatial_bins
    attempts = graph.attempts
    _need(_integer(capacity, 1, max_matrix_entries//atoms) and _integer(degree, 0, capacity)
          and _integer(bins, 1, 2**63-1) and type(attempts) in (list, tuple)
          and 1 <= len(attempts) <= max_attempts)
    history = []
    for pair in attempts:
        _need(type(pair) in (list, tuple) and len(pair) == 2 and
              _integer(pair[0], 1, max_matrix_entries//atoms) and _integer(pair[1], 0, max_matrix_entries))
        history.append(list(pair))
    _need(history[-1] == [capacity, degree])
    return dict(capacity=capacity, maximum_degree=degree, attempts=len(history), growths=len(history)-1,
                spatial_bins=bins, capacity_history=history)


def _decoded_graph_metadata(value, *, atoms, max_matrix_entries, max_attempts):
    _need(type(value) is dict and set(value) == {'capacity', 'maximum_degree', 'attempts', 'growths',
                                               'spatial_bins', 'capacity_history'})
    class Graph:
        pass
    graph = Graph()
    graph.capacity, graph.maximum_degree = value['capacity'], value['maximum_degree']
    graph.spatial_bins, graph.attempts = value['spatial_bins'], value['capacity_history']
    canonical = _graph_metadata(graph, atoms=atoms, max_matrix_entries=max_matrix_entries, max_attempts=max_attempts)
    _need(type(value['attempts']) is int and type(value['growths']) is int and canonical == value)
    return canonical


def _decode(payload):
    def pairs(items):
        result = {}
        for name, value in items:
            _need(name not in result)
            result[name] = value
        return result
    def invalid(_value):
        raise ValueError('DEVICE_GRAPH_NONFINITE_JSON')
    return json.loads(payload.decode('ascii'), object_pairs_hook=pairs, parse_constant=invalid)


def _digest_words(payload):
    digest = hashlib.sha256(payload).digest()
    return [int.from_bytes(digest[index:index+8], 'little', signed=True) for index in range(0, 32, 8)]


def broadcast_device_neighbor_epoch(control, root_graph, *, global_positions, cell, atoms,
                                    max_matrix_entries, max_attempts, expected_epoch,
                                    expected_generation, chunk_size=1048576):
    """Return one rank's epoch and shared graph metadata, or fail collectively.

    Local failures join the same scalar max_code boundary before any following
    broadcast. A collective/control transport failure itself is fatal and is
    left to the caller's finite process-group/whole-job timeout.
    """
    def arguments():
        _need(_integer(atoms, 1, 2**31-1) and _integer(max_matrix_entries, atoms, 2**31-1)
              and _integer(max_attempts, 1, 32) and _integer(expected_epoch, 0, 2**31-1)
              and _integer(expected_generation, 0, 2**31-1) and _integer(chunk_size, 1, MAX_CHUNK_SIZE))
        _need(type(control.world_size) is int and control.world_size in (1, 2, 4, 8)
              and _integer(control.rank, 0, control.world_size-1) and control.root == 0)
        _need(control.device.type in ('cpu', 'cuda'))
        from hyff.multigpu.device_neighbor_epoch import (
            DeviceNeighborChunk, DeviceNeighborLocalBuilder,
        )
        return DeviceNeighborChunk, DeviceNeighborLocalBuilder
    chunk_class, builder_class = _local(control, 'MD_DEVICE_GRAPH_ARGUMENTS', arguments)
    torch = control.torch
    descriptor = _local(control, 'MD_DEVICE_GRAPH_DESCRIPTOR_ALLOCATION',
                        lambda: torch.empty(_DESCRIPTOR_SIZE, dtype=torch.int64, device=control.device))
    local_arguments = [atoms, max_matrix_entries, max_attempts, expected_epoch,
                       expected_generation, chunk_size, control.world_size]
    limits = dict(atoms=atoms, world_size=control.world_size, max_matrix_entries=max_matrix_entries,
                  expected_epoch=expected_epoch, expected_generation=expected_generation)

    def root_header():
        if control.rank != 0:
            _need(root_graph is None)
            return None
        _need(root_graph is not None)
        snapshot = root_graph.snapshot
        snapshot.assert_integrity()
        header = snapshot.header
        radii, edges = _validate_header(header, **limits)
        graph = _graph_metadata(root_graph, atoms=atoms, max_matrix_entries=max_matrix_entries, max_attempts=max_attempts)
        _need(max(header['supplied_incoming_counts']) == graph['maximum_degree'])
        capacity = min(chunk_size, max(1, edges), MAX_CHUNK_BYTES//(4*(5+radii)))
        _need(capacity >= 1)
        payload = json.dumps(dict(schema=WIRE_SCHEMA, header=header, graph=graph),
                             sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
        header_bound = min(MAX_HEADER_BYTES, 65536+32*(atoms*(radii+4)+radii*(2*control.world_size+12)))
        _need(0 < len(payload) <= header_bound)
        return snapshot, payload, [len(payload), edges, radii, capacity], _digest_words(payload)
    root = _local(control, 'MD_DEVICE_GRAPH_ROOT_HEADER', root_header)

    def descriptor_fill():
        descriptor.zero_()
        if control.rank == 0:
            for index, value in enumerate(local_arguments+root[2]+root[3]):
                descriptor[index] = value
    _local(control, 'MD_DEVICE_GRAPH_DESCRIPTOR_COPY', descriptor_fill)
    _broadcast(control, descriptor)

    def descriptor_read():
        words = descriptor.cpu().tolist()
        _need(words[:7] == local_arguments)
        # Anchor the agreed digest and dimensions to the root-validated bytes.
        if control.rank == 0:
            _need(words[7:15] == root[2] + root[3])
        length, edges, radii, capacity = words[7:11]
        _need(_integer(radii, 1, MAX_RADII) and _integer(edges, 0, max_matrix_entries)
              and capacity == min(chunk_size, max(1, edges), MAX_CHUNK_BYTES//(4*(5+radii))) and capacity > 0)
        bound = min(MAX_HEADER_BYTES, 65536+32*(atoms*(radii+4)+radii*(2*control.world_size+12)))
        _need(_integer(length, 1, bound))
        return length, edges, radii, capacity, words[11:15]
    length, edges, radii, capacity, header_digest = _local(control, 'MD_DEVICE_GRAPH_DESCRIPTOR', descriptor_read)

    if control.world_size > 1:
        # Agree the fixed-size descriptor before any size-dependent collective.
        # A valid-range corruption of just one received length must not let
        # peers enter broadcasts with different shapes. Only 15 scalar words
        # per rank are exchanged here, never graph or header payloads.
        copies = _local(control, 'MD_DEVICE_GRAPH_DESCRIPTOR_AGREEMENT_ALLOCATION',
            lambda: [torch.empty_like(descriptor) for _ in range(control.world_size)])
        try:
            control.dist.all_gather(copies, descriptor, group=control.group)
        except BaseException as error:
            raise DeviceGraphTransportError('MD_DEVICE_GRAPH_DESCRIPTOR_AGREEMENT_FAILED',
                                            collective_safe=False) from error
        _local(control, 'MD_DEVICE_GRAPH_DESCRIPTOR_DISAGREEMENT',
               lambda: _need(all(torch.equal(value, descriptor) for value in copies)))

    def allocate():
        return (torch.empty(length, dtype=torch.uint8, device=control.device),
                torch.empty((5+radii)*capacity, dtype=torch.int32, device=control.device),
                torch.empty(3*capacity, dtype=torch.int32, device=control.device),
                torch.empty(radii*capacity, dtype=torch.bool, device=control.device))
    header_buffer, packed_buffer, shifts_buffer, membership_buffer = _local(
        control, 'MD_DEVICE_GRAPH_BUFFER_ALLOCATION', allocate)

    def header_copy():
        if control.rank == 0:
            header_buffer.copy_(torch.frombuffer(bytearray(root[1]), dtype=torch.uint8))
    _local(control, 'MD_DEVICE_GRAPH_HEADER_COPY', header_copy)
    _broadcast(control, header_buffer)

    def header_decode():
        payload = bytes(header_buffer.cpu().tolist())
        _need(_digest_words(payload) == header_digest)
        envelope = _decode(payload)
        _need(type(envelope) is dict and set(envelope) == {'schema', 'header', 'graph'}
              and envelope['schema'] == WIRE_SCHEMA)
        header = envelope['header']
        # The full envelope digest above binds this fresh JSON tree to the
        # root-validated bytes under descriptor-agreed local arguments.
        # Keep the descriptor/header dimensions binding without repeating
        # the same O(R*N) validation. Builder-specific checks remain below.
        _need((len(header['radii']), header['global_edge_count']) == (radii, edges))
        graph = _decoded_graph_metadata(envelope['graph'], atoms=atoms,
            max_matrix_entries=max_matrix_entries, max_attempts=max_attempts)
        _need(max(header['supplied_incoming_counts']) == graph['maximum_degree'])
        return header, graph
    header, graph_meta = _local(control, 'MD_DEVICE_GRAPH_HEADER_DECODE', header_decode)
    builder = _local(control, 'MD_DEVICE_GRAPH_LOCAL_ALLOCATION',
        lambda: builder_class._from_transport_owned_header(header, rank=control.rank,
            device=control.device, global_positions=global_positions, cell=cell))
    # _decode created this tree locally; no caller-owned header alias escapes.
    del header
    iterator = _local(control, 'MD_DEVICE_GRAPH_STREAM_START',
        lambda: iter(root[0].iter_chunks(chunk_size=capacity)) if control.rank == 0 else None)

    for offset in range(0, edges, capacity):
        width = min(capacity, edges-offset)
        def prepare_chunk():
            packed = packed_buffer[:(5+radii)*width].view(5+radii, width)
            if control.rank == 0:
                chunk = next(iterator)
                _need(type(chunk.offset) is int and chunk.offset == offset)
                for value, dtype, shape in ((chunk.edge_index, torch.int32, (2, width)),
                        (chunk.integer_shifts, torch.int32, (width, 3)),
                        (chunk.membership, torch.bool, (radii, width))):
                    _need(isinstance(value, torch.Tensor) and value.dtype == dtype and
                          tuple(value.shape) == shape and value.device == control.device and not value.requires_grad)
                packed[:2].copy_(chunk.edge_index)
                packed[2:5].copy_(chunk.integer_shifts.transpose(0, 1))
                packed[5:].copy_(chunk.membership)
            return packed
        packed = _local(control, 'MD_DEVICE_GRAPH_CHUNK_PREPARE', prepare_chunk)
        _broadcast(control, packed)

        def append_chunk():
            # Conversion must not silently reinterpret corrupted integer flags.
            _need(not bool(((packed[5:] != 0) & (packed[5:] != 1)).any().item()))
            shifts = shifts_buffer[:3*width].view(width, 3)
            membership = membership_buffer[:radii*width].view(radii, width)
            shifts.copy_(packed[2:5].transpose(0, 1))
            membership.copy_(packed[5:])
            builder.append(chunk_class(offset=offset, edge_index=packed[:2],
                integer_shifts=shifts, membership=membership))
        _local(control, 'MD_DEVICE_GRAPH_LOCAL_APPEND', append_chunk)

    def end_stream():
        if control.rank == 0:
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError('DEVICE_GRAPH_EXTRA_CHUNK')
            root[0].assert_integrity()
    _local(control, 'MD_DEVICE_GRAPH_STREAM_END', end_stream)

    def finish():
        epoch = builder.finish()
        metadata = dict(graph_meta, transport_schema=WIRE_SCHEMA,
            transport_policy='TRUSTED_BROADCAST_ROOT_FULL_VALIDATION_INDEPENDENT_LOCAL_COUNTS',
            payload_layout='INT32_SRC_DST_SHIFT_XYZ_MEMBERSHIP', global_edge_count=edges,
            chunk_capacity=capacity, chunks=(edges+capacity-1)//capacity,
            packed_buffer_bytes=(5+radii)*capacity*4,
            decode_buffer_bytes=(3*4+radii)*capacity,
            header_bytes=length, header_sha256_scope='JSON_HEADER_AND_QUERY_METADATA_ONLY',
            descriptor_agreement_scope='FIFTEEN_INT64_CONTROL_WORDS_ONLY',
            graph_payload_sha256=None, global_topology_cpu_copy=False,
            global_snapshot_scope='ROOT_ONLY', non_root_global_topology_retained=False,
            all_rank_global_topology_retained=False, graph_minmax_collectives=0)
        return epoch, metadata
    return _local(control, 'MD_DEVICE_GRAPH_LOCAL_FINISH', finish)


__all__ = ['broadcast_device_neighbor_epoch', 'DeviceGraphTransportError']
