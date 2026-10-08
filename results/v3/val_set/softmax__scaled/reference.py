import torch
import torch.nn.functional as F


SCALE = 0.08838834764831845  # 1 / sqrt(128)


def reference(x: torch.Tensor) -> torch.Tensor:
    """Scaled softmax over the last dim: softmax(x * SCALE)."""
    return F.softmax(x * SCALE, dim=-1)
