import torch
import torch.nn.functional as F


def reference(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """Fused residual add + RMSNorm over the last dim."""
    h = x + residual
    rms = torch.sqrt(torch.mean(h ** 2, dim=-1, keepdim=True) + eps)
    return (h / rms) * weight
