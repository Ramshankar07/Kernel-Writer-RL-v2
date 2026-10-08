"""
Validation starter: rotary_embedding / head_dim_96.
Interleaved RoPE; head_dim 96 (half = 48, masked block).
"""

KERNEL_TYPE = "rotary_embedding"

import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(X_ptr, COS_ptr, SIN_ptr, OUT_ptr, S, H, half, BLOCK_SIZE: tl.constexpr):
    """One program per (row of head_dim elements); position = row % S."""
    row = tl.program_id(0).to(tl.int64)
    p = row % S
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < half
    x_row = X_ptr + row * (2 * half)
    o_row = OUT_ptr + row * (2 * half)
    x1 = tl.load(x_row + offs * 2, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_row + offs * 2 + 1, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(COS_ptr + p * half + offs, mask=mask, other=1.0).to(tl.float32)
    s = tl.load(SIN_ptr + p * half + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_row + offs * 2, x1 * c - x2 * s, mask=mask)
    tl.store(o_row + offs * 2 + 1, x1 * s + x2 * c, mask=mask)


def kernel_fn(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    Bt, Hh, S, D = x.shape
    x = x.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    out = torch.empty_like(x)
    n_rows = x.numel() // D
    rope_kernel[(n_rows,)](x, cos, sin, out, S, Hh, D // 2, BLOCK_SIZE=triton.next_power_of_2(D // 2))
    return out
