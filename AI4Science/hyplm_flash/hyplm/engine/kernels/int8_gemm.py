"""INT8 W8A8 GEMM with per-(token, 128-column) activation scales and per-channel weight scales.

Each 128-wide K block is multiplied on INT8 tensor cores into INT32, promoted to FP32 and scaled by its
activation scale before accumulation; the per-output-channel weight scale is applied in the epilogue.
"""

import torch
import triton
import triton.language as tl

from .qstore import GROUP

BLOCK_K = GROUP

# (BM, BN, warps, stages) per (M bucket, N, K), autotuned once on H20; a fixed table avoids per-process
# autotuning and keeps launches CUDA-graph friendly.
_SMALL, _LARGE = (64, 64, 4, 5), (128, 64, 4, 4)
TUNED = {
    (1024, 3840, 1280): (64, 128, 4, 4), (1024, 7168, 1280): (64, 128, 4, 4),
    (2048, 1280, 1280): (64, 128, 4, 4), (2048, 3840, 1280): _LARGE, (2048, 7168, 1280): _LARGE,
    (4096, 3840, 1280): _LARGE, (4096, 7168, 1280): _LARGE,
    (8192, 1280, 1280): (64, 128, 4, 4), (8192, 3840, 1280): (64, 128, 4, 4), (8192, 7168, 1280): _LARGE,
}


def _bucket(m):
    return 1 << max(m - 1, 1).bit_length()


def _config(m, n, k):
    bucket = _bucket(m)
    if bucket >= 16384:
        return _LARGE
    return TUNED.get((max(bucket, 1024), n, k), _SMALL)


@triton.jit
def _int8_gemm_kernel(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, M, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = pid // (GROUP_M * num_n)
    first_m = group * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % (GROUP_M * num_n)) % size_m
    pid_n = (pid % (GROUP_M * num_n)) // size_m
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :].to(tl.int64) * K + offs_k[:, None]
    kb_total = K // BK
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(kb_total):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0)
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0)
        part = tl.dot(a, b, out_dtype=tl.int32)
        scale = tl.load(as_ptr + offs_m * kb_total + kb, mask=mask_m, other=0.0)
        acc += part.to(tl.float32) * scale[:, None]
        a_ptrs += BK
        b_ptrs += BK
    acc *= tl.load(bs_ptr + offs_n, mask=mask_n, other=0.0)[None, :]
    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def int8_gemm(a, a_scale, weight):
    """a: INT8 [M, K]; a_scale: FP32 [M, K // 128]; weight: Int8Weight [N, K] -> BF16 [M, N]."""
    m, k = a.shape
    n = weight.q.shape[0]
    c = torch.empty(m, n, device=a.device, dtype=torch.bfloat16)
    bm, bn, warps, stages = _config(m, n, k)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _int8_gemm_kernel[grid](a, a_scale, weight.q, weight.scale, c, m, n, k, BM=bm, BN=bn, BK=BLOCK_K, GROUP_M=8,
                            num_warps=warps, num_stages=stages)
    return c


class Int8Weight:
    """Per-output-channel symmetric INT8 weight [N, K]."""

    def __init__(self, weight):
        weight = weight.float()
        self.scale = (weight.abs().amax(dim=1).clamp_min(1e-12) / 127.0).contiguous()
        self.q = torch.round(weight / self.scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
