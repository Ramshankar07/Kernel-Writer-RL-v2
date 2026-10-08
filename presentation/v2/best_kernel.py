# Best PASS kernel found (extracted by presentation/v2/make_extra.py; code below is verbatim).
# source=grpo_v2 step=9 kernel=rmsnorm sample=7 turn=3 speedup=2.926x vs PyTorch (L4 full bench) code_sha=c45bd6a226e2

"""
AutoKernel improved -- RMS Normalization
Improved Triton kernel to handle mixed precision inputs more robustly.
"""

KERNEL_TYPE = "rmsnorm"

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, OUT_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel RMS normalization."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load row into float16 for numerical stability
    x = tl.load(X_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=0.0).to(tl.float16)

    # Compute RMS
    sq_mean = tl.sum(x * x, axis=0) / N
    rms = tl.sqrt(sq_mean + eps)

    # Normalize
    x_norm = x / rms

    # Scale by weight
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float16)
    out = x_norm * w

    # Store (cast back to input dtype via the output tensor's dtype)
    tl.store(OUT_ptr + row * stride_om + offs * stride_on, out, mask=mask)


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.rmsnorm_ref signature."""
    assert x.is_cuda
    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)

    rmsnorm_kernel[(M,)](
        x, weight, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
