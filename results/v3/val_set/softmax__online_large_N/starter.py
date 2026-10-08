"""
Validation starter: softmax / online_large_N.
Rows are too long for one block: single-pass online max/sum over
BLOCK_SIZE-wide lanes, then a second pass writes exp(x - m) / s.
"""

KERNEL_TYPE = "softmax"

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_online_kernel(x_ptr, out_ptr, n_cols, stride_x, stride_o, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride_x
    o_row = out_ptr + row * stride_o
    offs = tl.arange(0, BLOCK_SIZE)
    m_vec = tl.full((BLOCK_SIZE,), -1e30, dtype=tl.float32)   # per-lane running max
    s_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)          # per-lane running sum
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + offs
        x = tl.load(x_row + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m_vec, x)
        s_vec = s_vec * tl.exp(m_vec - m_new) + tl.exp(x - m_new)
        m_vec = m_new
    m = tl.max(m_vec, axis=0)
    s = tl.sum(s_vec * tl.exp(m_vec - m), axis=0)
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + offs
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
        tl.store(o_row + cols, tl.exp(x - m) / s, mask=mask)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x2.shape
    out = torch.empty_like(x2)
    softmax_online_kernel[(n_rows,)](x2, out, n_cols, x2.stride(0), out.stride(0), BLOCK_SIZE=4096)
    return out.view(x.shape)
