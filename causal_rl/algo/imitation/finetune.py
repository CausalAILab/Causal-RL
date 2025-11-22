import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Callable
from gymnasium import spaces

from causal_gym import PCH
from causal_rl.algo.imitation.imitate import build_window_features

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self._ptr = 0
        self._full = False

    def __len__(self) -> int:
        return len(self.states)

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool):
        # add a new transition to the buffer
        # move tensors to cpu for storage
        state_cpu = state.detach().cpu().view(-1)
        next_state_cpu = next_state.detach().cpu().view(-1)
        action_cpu = action.detach().cpu().view(-1)

        if not self._full:
            self.states.append(state_cpu)
            self.actions.append(action_cpu)
            self.rewards.append(reward)
            self.next_states.append(next_state_cpu)
            self.dones.append(done)

            if len(self.states) >= self.capacity:
                self._full = True
                self._ptr = 0

        else:
            self.states[self._ptr] = state_cpu
            self.actions[self._ptr] = action_cpu
            self.rewards[self._ptr] = reward
            self.next_states[self._ptr] = next_state_cpu
            self.dones[self._ptr] = done

            self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # randomly sample a batch of transitions from the buffer
        batch_size = min(batch_size, len(self))
        indices = np.random.randint(0, len(self), size=batch_size)

        states = torch.stack([self.states[i] for i in indices], dim=0).to(device=device, dtype=torch.float32)
        actions = torch.stack([self.actions[i] for i in indices], dim=0).to(device=device, dtype=torch.float32)
        rewards = torch.tensor([self.rewards[i] for i in indices], device=device, dtype=torch.float32).unsqueeze(-1)
        next_states = torch.stack([self.next_states[i] for i in indices], dim=0).to(device=device, dtype=torch.float32)
        dones = torch.tensor([self.dones[i] for i in indices], device=device, dtype=torch.float32).unsqueeze(-1)

        return states, actions, rewards, next_states, dones

class QNetwork(nn.Module):
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

        # initialize weights
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        x = torch.cat([state, action], dim=-1)
        return self.net(x)

def soft_update(source: nn.Module, target: nn.Module, tau: float):
    with torch.no_grad():
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)

def polyak_update_all(q1: nn.Module, q2: nn.Module, target_q1: nn.Module, target_q2: nn.Module, tau: float):
    soft_update(q1, target_q1, tau)
    soft_update(q2, target_q2, tau)

def init_q_networks(state_dim: int, action_dim: int, hidden_dim: int, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)

    # initialize target networks
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())

    return q1, q2, target_q1, target_q2

def infer_time_index(obs: dict[str, list[np.ndarray]]) -> int:
    for v in obs.values():
        return len(v) - 1
    
    return 0

def build_state_feature(obs: dict[str, list[np.ndarray]], t: int, Z_trim: dict[str, set[str]], slots: list[tuple[str, int, int]], device: torch.device) -> torch.Tensor:
    # convert PCH observation history into 1D state feature tensor
    key = f'X{t}'
    Zt = Z_trim.get(key, set())

    x, m = build_window_features(obs, t, Zt, slots)
    xm = np.concatenate([x, m], axis=0).astype(np.float32)
    state = torch.from_numpy(xm).unsqueeze(0).to(device=device)
    return state

def select_action(
    actor,
    state: torch.Tensor,
    action_space: spaces.Box,
    noise_std: float,
    device: torch.device,
    deterministic: bool = False
) -> np.ndarray:
    with torch.no_grad():
        action = actor(state).squeeze(0).cpu().numpy()

    if not deterministic:
        noise = np.random.normal(0, noise_std, size=action.shape)
        action = action + noise

    # clip action to valid range
    action = np.clip(action, action_space.low, action_space.high).astype(np.float32)
    return action

def rollout_online_episode(
    env: PCH,
    actor,
    replay_buffer: ReplayBuffer,
    Z_trim: dict[str, set[str]],
    slots: list[tuple[str, int, int]],
    action_space: spaces.Box,
    max_steps: int,
    noise_std: float,
    device: torch.device,
    seed: int | None = None
) -> dict[str, Any]:
    # roll out one online episode w/ exploration noise
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    rewards = []
    terminated = False
    truncated = False

    for step in range(max_steps):
        state = build_state_feature(obs, step, Z_trim, slots, device)
        action = select_action(actor, state, action_space, noise_std, device, deterministic=False)

        obs, reward, terminated, truncated, info = env.do(lambda x: action, show_reward=True)

        total_reward += reward
        rewards.append(reward)

        next_state = build_state_feature(obs, step + 1, Z_trim, slots, device)
        done = terminated or truncated

        replay_buffer.push(state, torch.from_numpy(action).float(), reward, next_state.squeeze(0), done)

        if done:
            break

    return {
        'return': total_reward,
        'length': step + 1,
        'rewards': rewards,
        'terminated': terminated,
        'truncated': truncated
    }

def td3_update_critics(
    q1: QNetwork,
    q2: QNetwork,
    target_q1: QNetwork,
    target_q2: QNetwork,
    actor,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    critic_optimizer_1: torch.optim.Optimizer,
    critic_optimizer_2: torch.optim.Optimizer,
    device: torch.device,
    action_space: spaces.Box,
    target_policy_noise: float,
    target_noise_clip: float
) -> dict[str, float]:
    # sample batch of transitions from replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device)

    # perform TD3 critic update
    with torch.no_grad():
        # TD3 target policy smoothing
        next_actions = actor(next_states)
        noise = torch.normal(mean=0.0, std=target_policy_noise, size=next_actions.shape, device=device)
        noise = noise.clamp(-target_noise_clip, target_noise_clip)
        next_actions = next_actions + noise

        # clip next actions to valid range
        low = torch.as_tensor(action_space.low, device=device, dtype=torch.float32)
        high = torch.as_tensor(action_space.high, device=device, dtype=torch.float32)
        next_actions = torch.max(torch.min(next_actions, high), low)

        q1_next = target_q1(next_states, next_actions)
        q2_next = target_q2(next_states, next_actions)
        q_next = torch.min(q1_next, q2_next)

        y = rewards + gamma * (1.0 - dones) * q_next

    q1_pred = q1(states, actions)
    q2_pred = q2(states, actions)

    loss_q1 = F.mse_loss(q1_pred, y)
    loss_q2 = F.mse_loss(q2_pred, y)

    critic_optimizer_1.zero_grad(set_to_none=True)
    loss_q1.backward()
    critic_optimizer_1.step()

    critic_optimizer_2.zero_grad(set_to_none=True)
    loss_q2.backward()
    critic_optimizer_2.step()

    with torch.no_grad():
        mean_q1 = q1_pred.mean().item()
        mean_q2 = q2_pred.mean().item()

    return {
        'loss_q1': loss_q1.item(),
        'loss_q2': loss_q2.item(),
        'mean_q1': mean_q1,
        'mean_q2': mean_q2
    }

def td3_update_actor(
    actor,
    q1: QNetwork,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    actor_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    # only need states
    states, _, _, _, _ = replay_buffer.sample(batch_size, device)

    actions_pi = actor(states)
    q1_pi = q1(states, actions_pi)
    actor_loss = -q1_pi.mean()

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_optimizer.step()

    return float(actor_loss.item())

# container for online RL hyperparameters
class OnlineRLConfig:
    def __init__(
        self,
        total_env_steps: int,
        start_steps: int,
        max_episode_steps: int,
        batch_size: int,
        gamma: float,
        tau: float,
        policy_delay: int,
        actor_lr: float,
        critic_lr: float,
        noise_std: float,
        hidden_dim_q: int,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5
    ):
        self.total_env_steps = total_env_steps
        self.start_steps = start_steps
        self.max_episode_steps = max_episode_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.noise_std = noise_std
        self.hidden_dim_q = hidden_dim_q
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

def fine_tune_actor_td3(
    env: PCH,
    actor,
    Z_trim: dict[str, set[str]],
    slots: list[tuple[str, int, int]],
    state_dim: int,
    action_dim: int,
    config: OnlineRLConfig,
    device: torch.device,
    seed: int | None = None,
    log_callback: Callable[[int, dict[str, Any]], None] | None = None
) -> tuple[Any, dict[str, list[float]]]:
    # fine-tune pretrained continuous actor using TD3-style online RL
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    action_space = env.env.action_space
    replay_buffer = ReplayBuffer(capacity=1_000_000)
    q1, q2, target_q1, target_q2 = init_q_networks(state_dim, action_dim, config.hidden_dim_q, device)

    actor = actor.to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer_1 = torch.optim.Adam(q1.parameters(), lr=config.critic_lr)
    critic_optimizer_2 = torch.optim.Adam(q2.parameters(), lr=config.critic_lr)

    logs = {
        'episode_returns': [],
        'episode_lengths': [],
        'critic_loss_q1': [],
        'critic_loss_q2': [],
        'actor_loss': []
    }

    env_steps = 0
    ep = 0

    rng = np.random.default_rng(seed) if seed is not None else None

    while env_steps < config.total_env_steps:
        ep_seed = int(rng.integers(0, 1e6)) if rng is not None else None

        ep_data = rollout_online_episode(
            env=env,
            actor=actor,
            replay_buffer=replay_buffer,
            Z_trim=Z_trim,
            slots=slots,
            action_space=action_space,
            max_steps=config.max_episode_steps,
            noise_std=config.noise_std,
            device=device,
            seed=ep_seed
        )

        ep_return = ep_data['return']
        ep_length = ep_data['length']
        logs['episode_returns'].append(ep_return)
        logs['episode_lengths'].append(ep_length)

        env_steps += ep_length
        ep += 1

        # update critics
        if len(replay_buffer) > config.start_steps:
            n_updates = ep_length

            for update_iter in range(n_updates):
                critic_metrics = td3_update_critics(
                    q1=q1,
                    q2=q2,
                    target_q1=target_q1,
                    target_q2=target_q2,
                    actor=actor,
                    replay_buffer=replay_buffer,
                    batch_size=config.batch_size,
                    gamma=config.gamma,
                    critic_optimizer_1=critic_optimizer_1,
                    critic_optimizer_2=critic_optimizer_2,
                    device=device,
                    action_space=action_space,
                    target_policy_noise=config.target_policy_noise,
                    target_noise_clip=config.target_noise_clip
                )

                logs['critic_loss_q1'].append(critic_metrics['loss_q1'])
                logs['critic_loss_q2'].append(critic_metrics['loss_q2'])

                # delayed policy updates
                actor_loss_val = None
                if update_iter % config.policy_delay == 0:
                    actor_loss_val = td3_update_actor(
                        actor=actor,
                        q1=q1,
                        replay_buffer=replay_buffer,
                        batch_size=config.batch_size,
                        actor_optimizer=actor_optimizer,
                        device=device
                    )
                    logs['actor_loss'].append(actor_loss_val)

                    # soft-update target networks
                    polyak_update_all(q1, q2, target_q1, target_q2, config.tau)

        if log_callback is not None:
            log_callback({
                'episode': ep,
                'env_steps': env_steps,
                'return': ep_return,
                'length': ep_length,
                'buffer_size': len(replay_buffer)
            })

        if env_steps >= config.total_env_steps:
            break

    return actor, logs