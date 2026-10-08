import torch
import torch.nn.functional as F


def reference(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim."""
    return F.softmax(x, dim=-1)
