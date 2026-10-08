import torch
import torch.nn.functional as F


def reference(x: torch.Tensor) -> torch.Tensor:
    """LayerNorm over the last dim, no elementwise affine (eps=1e-5)."""
    return F.layer_norm(x, x.shape[-1:], None, None, 1e-5)
