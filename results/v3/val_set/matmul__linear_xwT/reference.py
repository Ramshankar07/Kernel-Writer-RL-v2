import torch
import torch.nn.functional as F


def reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Linear layer without bias: x @ w.T, x: [M, K], w: [N, K]."""
    return x @ w.T
