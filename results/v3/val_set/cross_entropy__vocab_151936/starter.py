"""
Validation starter: cross_entropy / vocab_151936.
Qwen-size vocab: one program per row with an
online (running max / rescaled sum) log-sum-exp over BLOCK_SIZE chunks.
"""

KERNEL_TYPE = "cross_entropy"

import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_online_kernel(logits_ptr, targets_ptr, losses_ptr, n_cols, stride_row,
                                BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    row_start = logits_ptr + row * stride_row
    offs = tl.arange(0, BLOCK_SIZE)
    m_vec = tl.full((BLOCK_SIZE,), -1e30, dtype=tl.float32)
    s_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + offs
        x = tl.load(row_start + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m_vec, x)
        s_vec = s_vec * tl.exp(m_vec - m_new) + tl.exp(x - m_new)
        m_vec = m_new
    m = tl.max(m_vec, axis=0)
    lse = m + tl.log(tl.sum(s_vec * tl.exp(m_vec - m), axis=0))
    target = tl.load(targets_ptr + row)
    target_logit = tl.load(row_start + target).to(tl.float32)
    tl.store(losses_ptr + row, lse - target_logit)


def kernel_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    V = logits.shape[-1]
    l2 = logits.reshape(-1, V)
    t2 = targets.reshape(-1)
    losses = torch.empty(l2.shape[0], device=logits.device, dtype=torch.float32)
    cross_entropy_online_kernel[(l2.shape[0],)](l2, t2, losses, V, l2.stride(0), BLOCK_SIZE=4096)
    return losses.mean().to(logits.dtype)
