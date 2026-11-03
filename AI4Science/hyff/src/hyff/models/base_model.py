"""Model build that evaluates LayerNorm internally in float64; same potential and parameters.

Large but finite scalar features can have variances above float32's range.
No clipping, epsilon change, feature rescaling, force correction or new learned
parameter is introduced. Input/output and stored parameter dtypes are kept.
Both coordinate and mixed force/parameter derivatives pass through the casts.
"""
from types import MethodType

import torch

from hyff.models.state_model import build_model as build_original
from hyff.models.state_model import prepare_batch


POLICY = 'layernorm_centered_fp64'


def forward_fp64(self, value):
    if not value.is_floating_point():
        raise TypeError('LayerNorm requires floating-point inputs')
    dims = tuple(range(value.ndim-len(self.normalized_shape), value.ndim))
    x = value.double()
    centered = x-x.mean(dim=dims, keepdim=True)
    variance = centered.square().mean(dim=dims, keepdim=True)
    result = centered*torch.rsqrt(variance+self.eps)
    if self.weight is not None:
        result = result*self.weight.double()
    if self.bias is not None:
        result = result+self.bias.double()
    return result.to(value.dtype)


def install(model):
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.forward = MethodType(forward_fp64, module)
            names.append(name)
    if not names:
        raise ValueError('No ordinary LayerNorm modules found; wrong model factory')
    model.numerical_execution = dict(policy=POLICY, layernorm_modules=names,
                                    parameter_changes=False, mathematical_model_changes=False)
    return model


def build_model(config):
    return install(build_original(config))
