import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_rl.algo.imitation.gail.core_net import ContinuousActor, ResidualBlock


class IQLearnQNetwork(nn.Module):
    """Q-network for IQ-Learn with entropy-regularised V(s) computation.

    Takes (state, action) → scalar Q-value.
    Provides `compute_v` which estimates
        V(s) = E_{a~π}[ Q(s,a) - α log π(a|s) ]
    via Monte Carlo sampling from the current actor.

    Uses residual blocks matching the actor architecture for sufficient
    representational capacity.
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

    def compute_v(
        self,
        state: torch.Tensor,
        actor: ContinuousActor,
        alpha: float,
    ) -> torch.Tensor:
        """Estimate V(s) = Q(s,a) - α log π(a|s) with a single policy sample.

        Uses the same single-sample estimate as SAC target computation,
        avoiding the high-variance MC estimate that degrades learning in
        high-dimensional action spaces.

        Actor sampling is done without gradient to avoid backprop through the
        actor when this is used inside the critic loss.

        Args:
            state:       (B, state_dim)
            actor:       policy network
            alpha:       current entropy coefficient

        Returns:
            (B, 1)  V-values
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            a, lp, _ = actor.act(state, deterministic=False)

        q = self.forward(state, a)              # (B, 1)
        v = q - alpha * lp.unsqueeze(-1)        # (B, 1)
        return v


__all__ = ["IQLearnQNetwork", "ContinuousActor"]
