import torch
import torch.nn.functional as F


def reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Causal scaled dot-product attention, Q/K/V: [batch, heads, seq, head_dim]."""
    attn = torch.matmul(Q, K.transpose(-2, -1)) * (Q.shape[-1] ** -0.5)
    s_q, s_k = Q.shape[-2], K.shape[-2]
    mask = torch.triu(torch.ones(s_q, s_k, device=Q.device, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(mask, float('-inf'))
    return torch.matmul(F.softmax(attn, dim=-1), V)
