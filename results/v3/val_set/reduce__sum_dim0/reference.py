import torch
import torch.nn.functional as F


def reference(x: torch.Tensor) -> torch.Tensor:
    """Sum over dim 0 (column sums)."""
    return x.sum(dim=0)
