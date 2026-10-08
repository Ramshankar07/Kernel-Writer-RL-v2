import torch
import torch.nn.functional as F


def reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched matmul: A [b, M, K] @ B [b, K, N]."""
    return torch.bmm(A, B)
