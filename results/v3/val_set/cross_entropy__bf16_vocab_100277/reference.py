import torch
import torch.nn.functional as F


def reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy loss over all rows."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
