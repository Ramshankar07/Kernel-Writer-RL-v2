import torch
import torch.nn.functional as F


def reference(x: torch.Tensor) -> torch.Tensor:
    """Max over the last dim (values only)."""
    return x.max(dim=-1).values
