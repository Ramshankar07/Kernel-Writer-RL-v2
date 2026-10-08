import torch
import torch.nn.functional as F


def reference(x: torch.Tensor) -> torch.Tensor:
    """Mean over the last dim."""
    return x.mean(dim=-1)
