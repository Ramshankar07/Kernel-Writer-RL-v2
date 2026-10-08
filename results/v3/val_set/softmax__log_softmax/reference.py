import torch
import torch.nn.functional as F


def reference(x: torch.Tensor) -> torch.Tensor:
    """Log-softmax over the last dim."""
    return F.log_softmax(x, dim=-1)
