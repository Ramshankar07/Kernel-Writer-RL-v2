import torch
import torch.nn.functional as F


def reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Causal grouped-query attention. Q: [b, h, s, d]; K, V: [b, h_kv, s, d], h % h_kv == 0."""
    K = K.repeat_interleave(Q.shape[1] // K.shape[1], dim=1)
    V = V.repeat_interleave(Q.shape[1] // V.shape[1], dim=1)
    attn = torch.matmul(Q, K.transpose(-2, -1)) * (Q.shape[-1] ** -0.5)
    s_q, s_k = Q.shape[-2], K.shape[-2]
    mask = torch.triu(torch.ones(s_q, s_k, device=Q.device, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(mask, float('-inf'))
    return torch.matmul(F.softmax(attn, dim=-1), V)
