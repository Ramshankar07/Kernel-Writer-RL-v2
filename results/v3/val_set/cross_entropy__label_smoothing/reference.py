import torch
import torch.nn.functional as F


def reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy with label smoothing 0.1."""
    return F.cross_entropy(logits, targets, label_smoothing=0.1)
