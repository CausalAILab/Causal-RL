import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_rl.algo.imitation.gail.core_net import ContinuousActor, ResidualBlock


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


class SQILQNetwork(nn.Module):
    """Residual-block Q-network for SQIL.  Takes (state, action) → scalar Q-value.

    Same architecture as IQLearnQNetwork (residual blocks with LayerNorm, SiLU,
    dropout) but without compute_v, which is IQ-Learn-specific.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_blocks: int = 3, dropout: float = 0.05, layernorm: bool = True):
        super().__init__()
        input_dim = state_dim + action_dim

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        nn.init.orthogonal_(self.input_layer.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.input_layer.bias)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout, layernorm=layernorm)
            for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.output_layer.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        h = self.input_layer(torch.cat([state, action], dim=-1))
        for blk in self.blocks:
            h = blk(h)
        h = F.silu(h)
        return self.output_layer(h)


__all__ = ["SACQNetwork", "SQILQNetwork", "ContinuousActor"]
