import torch
import torch.nn.functional as F


def reference(x: torch.Tensor) -> torch.Tensor:
    """Sum of squares over the last dim."""
    return torch.sum(x * x, dim=-1)
