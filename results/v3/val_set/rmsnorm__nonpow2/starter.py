"""
Validation starter: rmsnorm / nonpow2.
RMSNorm, non-power-of-2 N.
"""

KERNEL_TYPE = "rmsnorm"

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(X_ptr, R_ptr, W_ptr, OUT_ptr, N, stride_row, stride_out, eps,
                   BLOCK_SIZE: tl.constexpr):
    """One program per row; fp32 sum of squares over one block."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT_ptr + row * stride_out + offs, (x / rms) * w, mask=mask)


def kernel_fn(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    out = torch.empty_like(x2)
    rmsnorm_kernel[(x2.shape[0],)](x2, x2, weight, out, N, x2.stride(0), out.stride(0),
                                   1e-6, BLOCK_SIZE=triton.next_power_of_2(N))
    return out.view(x.shape)
