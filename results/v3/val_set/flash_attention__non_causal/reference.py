import torch
import torch.nn.functional as F


def reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Non-causal scaled dot-product attention, Q/K/V: [batch, heads, seq, head_dim]."""
    attn = torch.matmul(Q, K.transpose(-2, -1)) * (Q.shape[-1] ** -0.5)
    return torch.matmul(F.softmax(attn, dim=-1), V)
