"""
Network architectures for IQ-Learn (Inverse Q-Learning).

This module defines Q-networks with soft V-function computation and imports
the actor network from GAIL for reuse.
"""

import numpy as np
import torch
import torch.nn as nn

# Reuse ContinuousActor from GAIL
from causal_rl.algo.imitation.gail.core_net import ContinuousActor


class IQLearnQNetwork(nn.Module):
    """
    Q-network for IQ-Learn with soft V-function computation.

    Takes concatenated [state, action] as input and outputs a scalar Q-value.
    Includes method to compute soft V-function via Monte Carlo sampling:
    V(s) = log E_{a~π}[exp(Q(s,a))]

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

        # Orthogonal initialization (matching SQIL/SAC style)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s,a).

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

    def compute_v(
        self,
        state: torch.Tensor,
        actor: ContinuousActor,
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        Compute soft V-function via Monte Carlo sampling.

        V(s) = log E_{a~π}[exp(Q(s,a))]
             ≈ log(1/k Σ exp(Q(s,a_i)))  where a_i ~ π(·|s)
             = logsumexp({Q(s,a_i)}) - log(k)

        Uses LogSumExp for numerical stability.

        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
            actor: Policy network to sample actions from
            num_samples: Number of action samples for MC approximation (default: 10)

        Returns:
            V-value tensor [batch_size, 1] or [1]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size = state.size(0)

        # Sample actions from policy: [batch_size * num_samples, action_dim]
        # Expand state to match number of samples
        state_expanded = state.unsqueeze(1).expand(batch_size, num_samples, -1)
        state_flat = state_expanded.reshape(batch_size * num_samples, -1)

        # Sample actions (no gradient through actor for V computation)
        with torch.no_grad():
            actions_flat, _, _ = actor.act(state_flat, deterministic=False)

        # Compute Q-values: [batch_size * num_samples, 1]
        q_values = self.forward(state_flat, actions_flat)
        q_values = q_values.reshape(batch_size, num_samples)

        # Compute V via LogSumExp for numerical stability:
        # log(1/k * Σ exp(Q_i)) = log(Σ exp(Q_i)) - log(k)
        v_values = torch.logsumexp(q_values, dim=1, keepdim=True) - np.log(num_samples)

        return v_values


__all__ = ['IQLearnQNetwork', 'ContinuousActor']
