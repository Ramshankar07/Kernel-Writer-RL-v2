import torch
import torch.nn.functional as F


def reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE on x: [batch, seq, heads, head_dim]; cos, sin: [seq, head_dim // 2]."""
    c = cos[:, None, :]
    s = sin[:, None, :]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)
