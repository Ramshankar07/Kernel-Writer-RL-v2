import torch
import torch.nn.functional as F


def reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE. x: [batch, heads, seq, head_dim]; cos, sin: [seq, head_dim // 2]."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)
