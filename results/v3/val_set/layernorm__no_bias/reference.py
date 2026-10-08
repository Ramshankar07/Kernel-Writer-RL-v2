import torch
import torch.nn.functional as F


def reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """LayerNorm over the last dim with a weight and NO bias (eps=1e-5)."""
    return F.layer_norm(x, x.shape[-1:], weight, None, 1e-5)
