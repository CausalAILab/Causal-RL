"""
Network architectures for SQIL (Soft Q Imitation Learning).

This module defines Q-networks for SAC-based SQIL and imports the actor
network from GAIL for reuse.
"""

import numpy as np
import torch
import torch.nn as nn

# Reuse ContinuousActor from GAIL
from causal_rl.algo.imitation.gail.core_net import ContinuousActor


class SACQNetwork(nn.Module):
    """
    Q-network for Soft Actor-Critic.

    Takes concatenated [state, action] as input and outputs a scalar Q-value.
    Uses orthogonal initialization for better gradient flow.

    Args:
        state_dim: Dimension of state encoding
        action_dim: Dimension of action space
        hidden_dim: Hidden layer size (default: 256)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        input_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Orthogonal initialization (matching TD3 style)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Q-network.

        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
            action: Action tensor [batch_size, action_dim] or [action_dim]

        Returns:
            Q-value tensor [batch_size, 1] or [1]
        """
        # Handle single samples (add batch dimension)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


__all__ = ['SACQNetwork', 'ContinuousActor']
