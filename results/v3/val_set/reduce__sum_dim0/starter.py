"""
Validation starter: reduce / sum_dim0.
Column sum: each program owns BLOCK_N columns and loops over rows.
"""

KERNEL_TYPE = "reduce"

import torch
import triton
import triton.language as tl


@triton.jit
def colsum_kernel(X_ptr, OUT_ptr, M, N, stride_b, stride_m, stride_n, stride_ob,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Sum over the M axis of a [batch, M, N] view. Program (column block, batch)."""
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.arange(0, BLOCK_M)
    base = X_ptr + pid_b * stride_b
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, M, BLOCK_M):
        r = start + rows
        m = (r[:, None] < M) & (cols[None, :] < N)
        acc += tl.load(base + r[:, None] * stride_m + cols[None, :] * stride_n, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT_ptr + pid_b * stride_ob + cols, tl.sum(acc, axis=0), mask=cols < N)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    x3 = x.contiguous().unsqueeze(0)
    Bt, M, N = x3.shape
    out = torch.empty((Bt, N), device=x.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N = 32, 128
    grid = (triton.cdiv(N, BLOCK_N), Bt)
    colsum_kernel[grid](x3, out, M, N, x3.stride(0), x3.stride(1), x3.stride(2), out.stride(0),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out[0].to(x.dtype)
