"""
Validation starter: layernorm / large_D.
Rows of 100k-300k elements: three looped passes
(mean, variance of x - mean, normalise) over BLOCK_SIZE chunks, fp32.
"""

KERNEL_TYPE = "layernorm"

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_looped_kernel(X_ptr, Y_ptr, W_ptr, B_ptr, stride_x_row, stride_y_row, N, eps,
                            BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = X_ptr + row * stride_x_row
    y_row = Y_ptr + row * stride_y_row
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cols = start + offs
        acc += tl.load(x_row + cols, mask=cols < N, other=0.0).to(tl.float32)
    mean = tl.sum(acc, axis=0) / N
    acc2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cols = start + offs
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        xc = tl.where(mask, x - mean, 0.0)
        acc2 += xc * xc
    rstd = 1.0 / tl.sqrt(tl.sum(acc2, axis=0) / N + eps)
    for start in range(0, N, BLOCK_SIZE):
        cols = start + offs
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_row + cols, (x - mean) * rstd * w + b, mask=mask)


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    y = torch.empty_like(x2)
    layernorm_looped_kernel[(x2.shape[0],)](x2, y, weight, bias, x2.stride(0), y.stride(0), N, 1e-5,
                                            BLOCK_SIZE=4096)
    return y.view(x.shape)
