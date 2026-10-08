import torch
import torch.nn.functional as F


def reference(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """relu(A @ B + bias), bias: [N]."""
    return torch.relu(torch.matmul(A, B) + bias)
