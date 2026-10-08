import torch
import torch.nn.functional as F


def reference(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
              w_down: torch.Tensor) -> torch.Tensor:
    """down(relu(x @ w_gate.T)**2 * (x @ w_up.T)); weights [out, in]."""
    gate = F.relu(x @ w_gate.T) ** 2
    up = x @ w_up.T
    return (gate * up) @ w_down.T
