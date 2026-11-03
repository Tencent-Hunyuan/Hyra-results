"""Multi-GPU force-service policy: the arithmetic and kernel options of one service configuration.

``native_nodes``: tf32x3 node products with grouped experts (otherwise all-fp32 nodes);
``compiled_efm``: compiled fixed-shape edge-frame-mixing tiles; ``compact_efm_edges``: skip inactive
edge-frame-mixing edges (requires ``compiled_efm``); ``compact_base_pair``: skip inactive base/pair
rows (requires ``compact_efm_edges``); ``efm_tile_edges``: edges per compiled tile. All ranks must
agree on the resulting dictionary.
"""


class ServicePolicyError(ValueError):
    pass


def _require(condition, code):
    if not condition:
        raise ServicePolicyError(code)


def service_policy(*, native_nodes=False, compiled_efm=True, compact_efm_edges=True,
                   compact_base_pair=True, efm_tile_edges=32768):
    """The full service policy dictionary for the given options."""
    for value in (native_nodes, compiled_efm, compact_efm_edges, compact_base_pair):
        _require(type(value) is bool, 'SERVICE_OPTION_INVALID')
    _require(not compact_efm_edges or compiled_efm, 'SERVICE_COMPACT_EFM_EDGES_REQUIRES_COMPILED_EFM')
    _require(not compact_base_pair or compact_efm_edges, 'SERVICE_COMPACT_BASE_PAIR_REQUIRES_COMPACT_EFM_EDGES')
    _require(type(efm_tile_edges) is int and efm_tile_edges in (8192, 16384, 32768), 'SERVICE_EFM_TILE_EDGES_INVALID')
    _require(efm_tile_edges == 32768 or compiled_efm, 'SERVICE_EFM_TILE_EDGES_REQUIRES_COMPILED_EFM')
    return dict(node_precision='tf32x3' if native_nodes else 'fp32', efm_precision='fp32',
                parameter_dtype='float32', grouped_experts=native_nodes, compile_efm_tiles=compiled_efm,
                active_efm_compaction=compact_efm_edges, active_base_pair_compaction=compact_base_pair,
                efm_tile_edges=efm_tile_edges, float32_matmul_precision='highest',
                cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False, compiled_nodes=False,
                joined_projections=False, folded_first_projections=False, frozen_layernorm=False,
                fused_local_features=False, learned_weights_changed=False)
