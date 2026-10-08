import torch
import torch.nn.functional as F


def reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over rows whose target != -100 (ignore_index)."""
    return F.cross_entropy(logits, targets, ignore_index=-100)
