import math
import torch
import torch.nn as nn

from causal_rl.algo.imitation.gail.core_net import ContinuousActor


class SACQNetwork(nn.Module):
    """Twin-ready Q-network for SAC.  Takes (state, action) → scalar Q-value."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return self.net(torch.cat([state, action], dim=-1))


__all__ = ["SACQNetwork", "ContinuousActor"]
