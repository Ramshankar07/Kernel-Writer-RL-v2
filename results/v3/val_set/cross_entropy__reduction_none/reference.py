import torch
import torch.nn.functional as F


def reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-row cross-entropy losses (no reduction)."""
    return F.cross_entropy(logits, targets, reduction="none")
