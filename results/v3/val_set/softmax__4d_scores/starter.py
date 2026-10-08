"""
Validation starter: softmax / 4d_scores.
Row softmax over the last dim of a [B, H, Sq, Sk] tensor.
"""

KERNEL_TYPE = "softmax"

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr, n_cols, stride_input_row, stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row; the whole row is one block (BLOCK_SIZE >= n_cols)."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(input_ptr + row_idx * stride_input_row + col_offsets, mask=mask,
                  other=float("-inf")).to(tl.float32)
    # (no pre-scaling)
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    tl.store(output_ptr + row_idx * stride_output_row + col_offsets, result, mask=mask)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    xt = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = xt.shape
    out = torch.empty_like(xt)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](xt, out, n_cols, xt.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return out.view(x.shape)
