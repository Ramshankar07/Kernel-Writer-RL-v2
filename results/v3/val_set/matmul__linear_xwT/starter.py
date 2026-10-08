"""
Validation starter: matmul / linear_xwT.
x @ w.T: w is read with swapped strides (no transpose copy).
"""

KERNEL_TYPE = "matmul"

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_K: tl.constexpr, IEEE_FP32: tl.constexpr,
):
    """Tiled matmul (same accumulation scheme as the fixed AutoKernel matmul starter):
    fp16/bf16: fp32 tensor-core partials per GROUP_K tiles, summed with fp32 adds;
    fp32: input_precision="ieee" tiles summed in fp64."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if IEEE_FP32:
        acc64 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
        for k in range(0, K, BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc64 += tl.dot(a, b, input_precision="ieee").to(tl.float64)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            offs_k += BLOCK_SIZE_K
        tl.store(c_ptrs, acc64.to(tl.float32), mask=c_mask)
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
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


def kernel_fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    A, Bm = x, w
    C = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BM, BN, BK = 64, 64, 32
    matmul_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, Bm, C, M, N, K, A.stride(0), A.stride(1), Bm.stride(1), Bm.stride(0), C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK, GROUP_K=2,
        IEEE_FP32=(x.dtype == torch.float32))
    return C
