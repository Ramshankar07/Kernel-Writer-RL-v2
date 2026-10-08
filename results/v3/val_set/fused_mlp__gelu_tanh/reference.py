import torch
import torch.nn.functional as F


def reference(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
              w_down: torch.Tensor) -> torch.Tensor:
    """down(gelu_tanh(x @ w_gate.T) * (x @ w_up.T)); weights [out, in]."""
    gate = F.gelu(x @ w_gate.T, approximate="tanh")
    up = x @ w_up.T
    return (gate * up) @ w_down.T
