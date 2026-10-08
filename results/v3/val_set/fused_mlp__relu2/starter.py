"""
Validation starter: fused_mlp / relu2.
Fused gate/up with a squared-ReLU gate, fp32 hidden.
"""

KERNEL_TYPE = "fused_mlp"

import torch
import triton
import triton.language as tl


@triton.jit
def gate_up_kernel(X_ptr, Wg_ptr, Wu_ptr, Out_ptr, M, N, K,
                   stride_xm, stride_xk, stride_wk, stride_wn, stride_om, stride_on,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   IEEE_FP32: tl.constexpr):
    """h = act(x @ w_gate.T) * (x @ w_up.T), act = relu2. W is [N, K], read transposed."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        x = tl.load(X_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)
        w_off = kk[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = (kk[:, None] < K) & (offs_n[None, :] < N)
        wg = tl.load(Wg_ptr + w_off, mask=w_mask, other=0.0)
        wu = tl.load(Wu_ptr + w_off, mask=w_mask, other=0.0)
        if IEEE_FP32:
            acc_g += tl.dot(x, wg, input_precision="ieee")
            acc_u += tl.dot(x, wu, input_precision="ieee")
        else:
            acc_g += tl.dot(x, wg)
            acc_u += tl.dot(x, wu)
    g = acc_g
    h = (tl.maximum(g, 0.0) * tl.maximum(g, 0.0)) * acc_u
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, h, mask=out_mask)  # float32 hidden


def kernel_fn(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = w_gate.shape[0]
    hidden = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gate_up_kernel[grid](x2, w_gate, w_up, hidden, M, N, K,
                         x2.stride(0), x2.stride(1), w_gate.stride(1), w_gate.stride(0),
                         hidden.stride(0), hidden.stride(1),
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                         IEEE_FP32=(x.dtype == torch.float32))
    out = (hidden @ w_down.to(torch.float32).t()).to(x.dtype)  # down projection in fp32
    return out.view(*lead, out.shape[-1])
