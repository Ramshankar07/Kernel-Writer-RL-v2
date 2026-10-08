"""
AutoKernel -- The file the agent modifies.

starters_v2 fixes (2026-09-30), Triton 3.2 / bench v2 (L4):
  * float32 inputs: tl.dot defaulted to TF32 (max_abs_err ~4e-2 vs tol 1e-4). fp32 now uses
    input_precision="ieee" per BLOCK_SIZE_K tile, and the per-tile partial products are summed
    in float64 across K, so the result is ~exact (fp32 is not a timed dtype).
  * fp16/bf16: fp32 accumulation inside tensor-core MMAs truncates, and over K=4096 at |A|,|B|~8
    the bias exceeded atol=1e-2 on near-zero outputs (bench v2 adversarial big_scale_K4096).
    Each group of GROUP_K=2 tiles (unrolled with tl.static_range, so the K loop still
    pipelines: 339 us vs 330 us upstream at 'large') is accumulated in a fresh fp32 block, and
    the blocks are added with ordinary round-to-nearest fp32 adds.

Current kernel: Matrix Multiplication
Target metric: throughput_tflops (higher is better)
Secondary: correctness must ALWAYS pass

The agent can change anything in this file:
  - Block sizes, warps, stages
  - Tiling strategy, memory access patterns
  - Split-K, persistent kernels, autotune configs
  - Any Triton feature or trick

The agent CANNOT change bench.py, reference.py, or the evaluation.
"""

KERNEL_TYPE = "matmul"  # must match a key in bench.py KERNEL_CONFIGS

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    IEEE_FP32: tl.constexpr,
):
    """Basic tiled matmul. The agent improves this."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if IEEE_FP32:
        acc64 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
        for k in range(0, K, BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            part = tl.dot(a, b, input_precision="ieee")
            acc64 += part.to(tl.float64)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            offs_k += BLOCK_SIZE_K
        tl.store(c_ptrs, acc64.to(tl.float32), mask=mask)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_SIZE_K * GROUP_K):
            blk = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in tl.static_range(GROUP_K):
                a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
                blk += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk
                offs_k += BLOCK_SIZE_K
            acc += blk
        c = acc.to(C_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=mask)


def kernel_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.matmul_ref signature."""
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_K = 2   # tiles per fp32 partial block (see header)

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_K=GROUP_K,
        IEEE_FP32=(A.dtype == torch.float32),
    )
    return C
