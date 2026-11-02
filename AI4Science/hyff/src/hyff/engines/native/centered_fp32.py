"""Force-time FP32 LayerNorm statistics for the vivace and moderato native engines.

The default ("fp64") statistics evaluate every frozen LayerNorm and the fused
local-feature normalization with centered FP64 internals. With "fp32" the same
operators use centered FP32 statistics at force time; the setup-time first
message cache keeps FP64 statistics. Parameters and epsilon are unchanged.
"""
from __future__ import annotations

STATISTICS = ('fp64', 'fp32')


def normalization_installers(statistics, precision):
    """Return the (LayerNorm, local feature) installers for the requested statistics."""
    if statistics not in STATISTICS:
        raise ValueError('statistics must be fp64 or fp32')
    if statistics == 'fp64':
        from hyff.kernels.layernorm_fp64 import enable_frozen_layernorm
        from hyff.kernels.local_features_fp64 import enable_frozen_local_features
        return enable_frozen_layernorm, enable_frozen_local_features
    if precision != 'tf32x3':
        raise ValueError('FP32 statistics require tf32x3 products')
    from hyff.kernels.layernorm_fp32 import enable_frozen_layernorm
    from hyff.kernels.local_features_fp32 import enable_frozen_local_features
    return enable_frozen_layernorm, enable_frozen_local_features


def check_fp32_statistics(model, expected_norms=None):
    """Confirm that every frozen LayerNorm and the fused features use FP32 statistics."""
    from hyff.kernels.layernorm_fp32 import FrozenLayerNorm32
    count = sum(isinstance(m, FrozenLayerNorm32) for m in model.modules())
    if count <= 0 or count != model.deployment_metadata.get('frozen_layernorm_modules'):
        raise ValueError('FP32 statistics conversion missed a frozen LayerNorm')
    if expected_norms is not None and count != expected_norms:
        raise ValueError('Unexpected number of frozen LayerNorms for FP32 statistics')
    if getattr(model, 'frozen_local_feature_statistics', None) != 'centered_fp32':
        raise ValueError('FP32 scalar normalization is absent from the fused local features')
    model.deployment_metadata = dict(model.deployment_metadata, layernorm_statistics='fp32',
                                     normalization_policy='centered_FP32_internal_same_epsilon',
                                     fused_feature_normalization_precision='FP32')
    return model
