"""
Validation starter: flash_attention / head_dim_32.
Causal attention, head_dim 32.
"""

KERNEL_TYPE = "flash_attention"

import torch
import triton
import triton.language as tl


import math


@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    M_size, N_size, GROUP, sm_scale,
    D: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Online-softmax attention (fixed AutoKernel starter). Program = (query block, head, batch).
    K/V head = head // GROUP (GQA); keys may be longer than queries (N_size != M_size)."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)
    kv_h = pid_h // GROUP
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q_ptr + pid_z * stride_qz + pid_h * stride_qh + offs_m[:, None] * stride_qm
                + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < M_size, other=0.0)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    if IS_CAUSAL:
        kv_end = tl.minimum(N_size, (pid_m + 1) * BLOCK_M)
    else:
        kv_end = N_size
    k_base = K_ptr + pid_z * stride_kz + kv_h * stride_kh
    v_base = V_ptr + pid_z * stride_vz + kv_h * stride_vh
    for start_n in range(0, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
                    mask=offs_n[:, None] < N_size, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        qk = tl.where(offs_n[None, :] < N_size, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                    mask=offs_n[:, None] < N_size, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O_ptr + pid_z * stride_oz + pid_h * stride_oh + offs_m[:, None] * stride_om
             + offs_d[None, :] * stride_ok, acc.to(O_ptr.dtype.element_ty), mask=offs_m[:, None] < M_size)


def kernel_fn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    Z, Hq, M_size, D = Q.shape
    Hkv, N_size = K.shape[1], K.shape[2]
    O = torch.empty_like(Q)
    BLOCK_M = 64
    BLOCK_N, extra = (32, {"num_stages": 2}) if D >= 128 else (64, {})  # L4 shared memory
    grid = (triton.cdiv(M_size, BLOCK_M), Hq, Z)
    flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        M_size, N_size, Hq // Hkv, 1.0 / math.sqrt(D),
        D=D, IS_CAUSAL=True, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, **extra,
    )
    return O
