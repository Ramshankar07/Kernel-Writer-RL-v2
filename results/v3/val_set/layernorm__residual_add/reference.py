import torch
import torch.nn.functional as F


def reference(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
              bias: torch.Tensor) -> torch.Tensor:
    """Fused residual add + LayerNorm: layer_norm(x + residual) over the last dim (eps=1e-5)."""
    return F.layer_norm(x + residual, x.shape[-1:], weight, bias, 1e-5)
