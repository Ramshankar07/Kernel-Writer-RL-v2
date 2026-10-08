import torch
import torch.nn.functional as F


def reference(x: torch.Tensor) -> torch.Tensor:
    """Softmax over dim 0 (each column sums to 1)."""
    return F.softmax(x, dim=0)
