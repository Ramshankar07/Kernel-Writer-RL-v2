"""
Validation starter: layernorm / no_affine.
LayerNorm without weight/bias.
"""

KERNEL_TYPE = "layernorm"

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(X_ptr, R_ptr, Y_ptr, W_ptr, B_ptr, stride_x_row, stride_y_row, N, eps,
                     BLOCK_SIZE: tl.constexpr):
    """One program per row, two-pass mean/variance in fp32 on one block."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    x = tl.load(X_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    y = xc / tl.sqrt(var + eps)
    tl.store(Y_ptr + row_idx * stride_y_row + col_offsets, y, mask=mask)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    y = torch.empty_like(x2)
    BLOCK_SIZE = triton.next_power_of_2(N)
    layernorm_kernel[(x2.shape[0],)](x2, x2, y, x2, x2, x2.stride(0), y.stride(0), N, 1e-5,
                                     BLOCK_SIZE=BLOCK_SIZE)
    return y.view(x.shape)
