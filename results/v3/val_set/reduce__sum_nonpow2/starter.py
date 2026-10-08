"""
Validation starter: reduce / sum_nonpow2.
Row sum over the last dim.
"""

KERNEL_TYPE = "reduce"

import torch
import triton
import triton.language as tl


@triton.jit
def reduce_rows_kernel(X_ptr, OUT_ptr, N, stride_row, BLOCK_SIZE: tl.constexpr):
    """One program per row; loops over the row in BLOCK_SIZE chunks, fp32 accumulator."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cols = start + offs
        x = tl.load(X_ptr + row * stride_row + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x
    tl.store(OUT_ptr + row, tl.sum(acc, axis=0))


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    out = torch.empty(x2.shape[0], device=x.device, dtype=torch.float32)
    BLOCK_SIZE = min(triton.next_power_of_2(N), 8192)
    reduce_rows_kernel[(x2.shape[0],)](x2, out, N, x2.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return out.to(x.dtype).view(x.shape[:-1])
