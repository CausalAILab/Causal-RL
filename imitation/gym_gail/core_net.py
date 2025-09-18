import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Independent

class ContinuousActor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=1024, activation='relu', std=0.0, action_low: float = -1.0, action_high: float = 1.0):
        super(ContinuousActor, self).__init__()

        self.hidden_size = hidden_size

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

        self.log_std_actor = nn.Parameter(torch.ones(num_outputs,) * std)

        self.low = float(action_low)
        self.high = float(action_high)

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
        log_det_scale = u.shape[-1] *torch.log(torch.as_tensor((self.high - self.low)/2.0, device=z.device, dtype=z.dtype)).sum()

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
    def __init__(self, num_inputs, hidden_size, activation='relu', dropout=0.2):
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