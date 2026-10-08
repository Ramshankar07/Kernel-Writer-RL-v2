import torch
import torch.nn.functional as F


def reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A @ B, A: [M, K], B: [K, N]."""
    return torch.matmul(A, B)
