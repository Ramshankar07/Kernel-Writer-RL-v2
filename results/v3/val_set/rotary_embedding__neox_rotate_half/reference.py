import torch
import torch.nn.functional as F


def reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half (NeoX) RoPE. x: [batch, heads, seq, head_dim]; cos, sin: [seq, head_dim // 2]."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
