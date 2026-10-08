import torch
import torch.nn.functional as F


def reference(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU gate/up without the down projection -> [M, hidden]."""
    return F.silu(x @ w_gate.T) * (x @ w_up.T)
