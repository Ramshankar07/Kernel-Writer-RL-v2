"""
Validation starter: cross_entropy / 3d_logits.
Mean cross-entropy on [batch, seq, vocab] logits.
"""

KERNEL_TYPE = "cross_entropy"

import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(logits_ptr, targets_ptr, losses_ptr, n_cols, stride_row,
                         BLOCK_SIZE: tl.constexpr):
    """One program per row; whole row in one block; fp32 log-sum-exp."""
    row_idx = tl.program_id(0)
    row_start = logits_ptr + row_idx.to(tl.int64) * stride_row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    logits = tl.load(row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    lse = row_max + tl.log(tl.sum(tl.exp(logits - row_max), axis=0))
    target = tl.load(targets_ptr + row_idx)
    target_logit = tl.load(row_start + target).to(tl.float32)
    tl.store(losses_ptr + row_idx, lse - target_logit)


def kernel_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    V = logits.shape[-1]
    l2 = logits.reshape(-1, V)
    t2 = targets.reshape(-1)
    losses = torch.empty(l2.shape[0], device=logits.device, dtype=torch.float32)
    cross_entropy_kernel[(l2.shape[0],)](l2, t2, losses, V, l2.stride(0),
                                         BLOCK_SIZE=triton.next_power_of_2(V))
    return losses.mean().to(logits.dtype)
