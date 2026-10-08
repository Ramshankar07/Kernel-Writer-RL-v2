import torch
import torch.nn.functional as F


def reference(x: torch.Tensor) -> torch.Tensor:
    """Sum over the last dim."""
    return x.sum(dim=-1)
