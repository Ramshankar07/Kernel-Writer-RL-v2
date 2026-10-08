import torch
import torch.nn.functional as F


def reference(x: torch.Tensor) -> torch.Tensor:
    """Sum over dim 1 of a [batch, n, d] tensor -> [batch, d]."""
    return x.sum(dim=1)
