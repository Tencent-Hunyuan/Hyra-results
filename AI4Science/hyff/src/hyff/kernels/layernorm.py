"""Row-tiled frozen LayerNorm and its full input derivative.

Automatic fast-node conversion covers the tuned 128/160-channel layouts only.
Parameters and epsilon keep their learned values; force training uses
ordinary differentiable PyTorch.
"""
import torch
from torch import nn
import triton as tr
import triton.language as tl


@tr.jit
def _norm_forward(X, Weight, Bias, Out, Mean, Rstd, N, H: tl.constexpr,
                  EPS: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    channel = tl.arange(0, BH)
    valid = (row[:, None] < N) & (channel[None, :] < H)
    x = tl.load(X + row[:, None]*H + channel[None, :], valid, 0.)
    anchor = tl.load(X + row*H, row < N, 0.)
    centered = tl.where(channel[None, :] < H, x-anchor[:, None], 0.)
    mean = tl.sum(centered, 1) / H
    deviation = tl.where(channel[None, :] < H, centered-mean[:, None], 0.)
    variance = tl.sum(deviation*deviation, 1) / H
    rstd = tl.rsqrt(variance + EPS)
    weight = tl.load(Weight + channel, channel < H, 0.)
    bias = tl.load(Bias + channel, channel < H, 0.)
    out = deviation * rstd[:, None] * weight[None, :] + bias[None, :]
    tl.store(Out + row[:, None]*H + channel[None, :], out, valid)
    tl.store(Mean + row, mean, row < N)
    tl.store(Rstd + row, rstd, row < N)


@tr.jit
def _norm_backward(X, Weight, Mean, Rstd, Gradient, Out, N, H: tl.constexpr,
                   BM: tl.constexpr, BH: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    channel = tl.arange(0, BH)
    valid = (row[:, None] < N) & (channel[None, :] < H)
    x = tl.load(X + row[:, None]*H + channel[None, :], valid, 0.)
    anchor = tl.load(X + row*H, row < N, 0.)
    mean = tl.load(Mean + row, row < N, 0.)
    rstd = tl.load(Rstd + row, row < N, 0.)
    norm = tl.where(channel[None, :] < H, (x-anchor[:, None]-mean[:, None])*rstd[:, None], 0.)
    weight = tl.load(Weight + channel, channel < H, 0.)
    gradient = tl.load(Gradient + row[:, None]*H + channel[None, :], valid, 0.)
    scaled = gradient * weight[None, :]
    mean_scaled = tl.sum(scaled, 1) / H
    mean_projected = tl.sum(scaled*norm, 1) / H
    out = (scaled-mean_scaled[:, None]-norm*mean_projected[:, None])*rstd[:, None]
    tl.store(Out + row[:, None]*H + channel[None, :], out, valid)


class _FrozenNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight, bias, epsilon, atom_tile):
        shape = values.shape
        x = values.reshape(-1, shape[-1]).contiguous()
        n, h = x.shape
        out = torch.empty_like(x)
        mean, rstd = x.new_empty(n), x.new_empty(n)
        if n:
            _norm_forward[(tr.cdiv(n, atom_tile),)](x, weight, bias, out, mean, rstd,
                n, h, epsilon, atom_tile, tr.next_power_of_2(h), num_warps=4, enable_fp_fusion=True)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.shape, ctx.atom_tile = shape, atom_tile
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Force training needs differentiable reference normalization')
        x, weight, mean, rstd = ctx.saved_tensors
        n, h = x.shape
        out = torch.empty_like(x)
        if n:
            _norm_backward[(tr.cdiv(n, ctx.atom_tile),)](x, weight, mean, rstd,
                gradient.contiguous(), out, n, h, ctx.atom_tile, tr.next_power_of_2(h),
                num_warps=4, enable_fp_fusion=True)
        return out.reshape(ctx.shape), None, None, None, None


class FrozenLayerNorm(nn.Module):
    def __init__(self, original, atom_tile=8):
        super().__init__()
        if len(original.normalized_shape) != 1 or not original.elementwise_affine or original.bias is None:
            raise ValueError('Frozen normalization requires one affine channel dimension')
        if any(p.requires_grad for p in original.parameters()):
            raise ValueError('Freeze normalization parameters before selecting this backend')
        if atom_tile not in (1, 4, 8, 16):
            raise ValueError('Unsupported atom tile')
        self.weight, self.bias = original.weight, original.bias
        self.normalized_shape, self.eps = original.normalized_shape, original.eps
        self.atom_tile = atom_tile

    def forward(self, values):
        if self.weight.requires_grad or self.bias.requires_grad:
            raise RuntimeError('Frozen normalization does not train parameters')
        if values.device.type != 'cuda' or values.dtype != torch.float32:
            raise ValueError('Frozen normalization requires CUDA FP32 activations')
        if self.weight.dtype != torch.float32 or self.bias.dtype != torch.float32:
            raise ValueError('Frozen normalization requires FP32 affine parameters')
        if values.shape[-1] != self.normalized_shape[0]:
            raise ValueError('Incompatible normalization channels')
        return _FrozenNorm.apply(values, self.weight, self.bias, self.eps, self.atom_tile)


def enable_frozen_layernorm(model, atom_tile=None):
    """Convert the LayerNorms of the tuned widths (or all, with atom_tile); training modules are unchanged."""
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze the model before selecting an inference backend')
    for name, child in list(model.named_children()):
        if isinstance(child, nn.LayerNorm):
            tile = atom_tile
            if tile is None:
                if child.normalized_shape == (128,):
                    tile = 16
                elif child.normalized_shape == (160,):
                    tile = 8
                else:
                    continue
            setattr(model, name, FrozenLayerNorm(child, tile))
        else:
            enable_frozen_layernorm(child, atom_tile)
    return model
