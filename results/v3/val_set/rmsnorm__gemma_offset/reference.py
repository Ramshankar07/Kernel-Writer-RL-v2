import torch
import torch.nn.functional as F


def reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Gemma-style RMSNorm: the learned scale is (1 + weight)."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * (1.0 + weight)
