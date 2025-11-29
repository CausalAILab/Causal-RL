import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Independent

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05, layernorm: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = F.silu(h)
        h = self.fc1(h)
        h = self.ln2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class ContinuousActor(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_size: int = 256,
        std: float = 0.0,
        action_low: float = -1.0,
        action_high: float = 1.0,
        num_blocks: int = 3,
        dropout: float = 0.05,
        layernorm: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.low = float(action_low)
        self.high = float(action_high)

        # backbone (ContinuousPolicyNN-style)
        self.hidden = nn.Linear(num_inputs, hidden_size)
        nn.init.orthogonal_(self.hidden.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.hidden.bias)

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, dropout=dropout, layernorm=layernorm) for _ in range(num_blocks)]
        )

        self.mu_head = nn.Linear(hidden_size, num_outputs)
        nn.init.uniform_(self.mu_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mu_head.bias)

        # log std in pre-tanh space
        self.log_std_actor = nn.Parameter(torch.ones(num_outputs) * std)

    def _backbone(self, z: torch.Tensor) -> torch.Tensor:
        h = self.hidden(z)
        for blk in self.blocks:
            h = blk(h)
        h = F.silu(h)
        return h

    def forward(self, z: torch.Tensor) -> Independent:
        h = self._backbone(z)
        mu = self.mu_head(h)
        std = self.log_std_actor.exp().expand_as(mu)
        dist = Normal(mu, std)
        return Independent(dist, 1)

    @torch.no_grad()
    def act(self, z: torch.Tensor, deterministic: bool = False):
        base = self.forward(z)
        u = base.base_dist.loc if deterministic else base.rsample()

        a_tanh = torch.tanh(u)
        action = (a_tanh + 1) * 0.5 * (self.high - self.low) + self.low

        log_det_tanh = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(dim=-1)
        log_det_scale = u.shape[-1] * torch.log(
            torch.as_tensor((self.high - self.low) / 2.0, device=z.device, dtype=z.dtype)
        ).sum()

        log_prob = base.log_prob(u) - (log_det_tanh + log_det_scale)
        entropy = base.entropy()
        return action, log_prob, entropy

    def evaluate_actions(self, z: torch.Tensor, action: torch.Tensor):
        base = self.forward(z)

        a_tanh = ((action - self.low) / (self.high - self.low) * 2) - 1.0
        a_tanh = torch.clamp(a_tanh, -0.999, 0.999)

        u = 0.5 * (torch.log1p(a_tanh + 1e-6) - torch.log1p(-a_tanh + 1e-6))

        log_det_tanh = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(dim=-1)
        log_det_scale = u.shape[-1] * torch.log(
            torch.as_tensor((self.high - self.low) / 2.0, device=z.device, dtype=z.dtype)
        ).sum()

        log_prob = base.log_prob(u) - (log_det_tanh + log_det_scale)
        entropy = base.entropy()
        return log_prob, entropy

    def forward(self, z):
        out1 = self.activation_fn(self.linear1(z))
        out2 = self.activation_fn(self.linear2(out1))
        out3 = self.activation_fn(self.linear3(out1 + out2))

        mu = self.linear4(out1 + out2 + out3)
        std = self.log_std_actor.exp().expand_as(mu)
        dist = Normal(mu, std)
        return Independent(dist, 1)

    @torch.no_grad()
    def act(self, z, deterministic=False):
        base = self.forward(z)
        u = base.base_dist.loc if deterministic else base.rsample()

        a_tanh = torch.tanh(u)
        action = (a_tanh + 1) * 0.5 * (self.high - self.low) + self.low

        log_det_tanh = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(dim=-1)
        log_det_scale = u.shape[-1] * torch.log(torch.as_tensor((self.high - self.low)/2.0, device=z.device, dtype=z.dtype)).sum()

        return action, (base.log_prob(u) - (log_det_tanh + log_det_scale)), base.entropy()

    def evaluate_actions(self, z, action):
        base = self.forward(z)

        a_tanh = ((action - self.low) / (self.high - self.low) * 2) - 1.0
        a_tanh = torch.clamp(a_tanh, -0.999, 0.999)

        u = 0.5 * (torch.log1p(a_tanh + 1e-6) - torch.log1p(-a_tanh + 1e-6))

        log_det_tanh = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(dim=-1)
        log_det_scale = u.shape[-1] * torch.log(torch.as_tensor((self.high - self.low)/2.0, device=z.device, dtype=z.dtype)).sum()

        return (base.log_prob(u) - (log_det_tanh + log_det_scale)), base.entropy()

class DiscreteActor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=1024, activation='relu'):
        super(DiscreteActor, self).__init__()

        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'leakyrelu':
            self.activation_fn = F.leaky_relu
        elif activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'sigmoid':
            self.activation_fn = torch.sigmoid

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.linear4 = nn.Linear(hidden_size, num_outputs)
        self.linear4.weight.data.mul_(0.1)
        self.linear4.bias.data.mul_(0.0)

    def forward(self, z):
        out1 = self.activation_fn(self.linear1(z))
        out2 = self.activation_fn(self.linear2(out1))
        out3 = self.activation_fn(self.linear3(out1 + out2))

        logits = self.linear4(out1 + out2 + out3)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, z, deterministic=False):
        dist = self.forward(z)
        action = dist.sample() if not deterministic else torch.argmax(dist.logits, dim=-1)
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, z, action):
        dist = self.forward(z)
        return dist.log_prob(action), dist.entropy()

class Critic(nn.Module):
    def __init__(self, num_inputs, num_outputs=1, hidden_size=1024, activation='relu'):
        super(Critic, self).__init__()

        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'leakyrelu':
            self.activation_fn = F.leaky_relu
        elif activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'sigmoid':
            self.activation_fn = torch.sigmoid

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.linear4 = nn.Linear(hidden_size, num_outputs)
        self.linear4.weight.data.mul_(0.1)
        self.linear4.bias.data.mul_(0.0)

    def forward(self, z):
        out1 = self.activation_fn(self.linear1(z))
        out2 = self.activation_fn(self.linear2(out1))
        out3 = self.activation_fn(self.linear3(out1 + out2))

        value = self.linear4(out1 + out2 + out3)
        return value

class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=1024, activation='relu', dropout=0.2):
        super(Discriminator, self).__init__()

        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'leakyrelu':
            self.activation_fn = F.leaky_relu
        elif activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'sigmoid':
            self.activation_fn = torch.sigmoid

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        self.linear4.weight.data.mul_(0.1)
        self.linear4.bias.data.mul_(0.0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        out1 = self.activation_fn(self.linear1(z))
        out1 = self.dropout(out1)

        out2 = self.activation_fn(self.linear2(out1))
        out2 = self.dropout(out2)

        out3 = self.activation_fn(self.linear3(out1 + out2))
        out3 = self.dropout(out3)

        out4 = self.linear4(out1 + out2 + out3).squeeze(-1)
        return out4