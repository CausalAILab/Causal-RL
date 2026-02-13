"""
SQIL (Soft Q Imitation Learning) implementation with causal integration.

SQIL treats imitation learning as reinforcement learning with binary rewards:
- Expert demonstrations get reward +1.0
- Policy samples get reward 0.0

Uses SAC (Soft Actor-Critic) as the base RL algorithm for continuous action spaces.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple
from gymnasium import spaces

from causal_gym import PCH
from causal_rl.algo.imitation.gail.causal_gail import (
    build_z_encoder,
    calc_categorical_dims
)
from causal_rl.algo.imitation.gail.core_net import ContinuousActor
from .core_net import SACQNetwork


# ====================================================================================
# Replay Buffer
# ====================================================================================

class ReplayBuffer:
    """
    Basic replay buffer for storing transitions.
    Stores tensors on CPU and transfers to GPU during sampling.
    """

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

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: float,
             next_state: torch.Tensor, done: bool):
        """Add transition to buffer (stores on CPU)."""
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

    def sample(self, batch_size: int, device: torch.device
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample batch and transfer to device."""
        batch_size = min(batch_size, len(self))
        indices = np.random.randint(0, len(self), size=batch_size)

        states = torch.stack([self.states[i] for i in indices], dim=0).to(device=device, dtype=torch.float32)
        actions = torch.stack([self.actions[i] for i in indices], dim=0).to(device=device, dtype=torch.float32)
        rewards = torch.tensor([self.rewards[i] for i in indices], device=device, dtype=torch.float32).unsqueeze(-1)
        next_states = torch.stack([self.next_states[i] for i in indices], dim=0).to(device=device, dtype=torch.float32)
        dones = torch.tensor([self.dones[i] for i in indices], device=device, dtype=torch.float32).unsqueeze(-1)

        return states, actions, rewards, next_states, dones


class SQILReplayBuffer:
    """
    SQIL Replay Buffer with separate expert and policy sub-buffers.

    Expert transitions are labeled with reward=+1.0, policy transitions with reward=0.0.
    Supports mixed sampling from both buffers for balanced learning.
    """

    def __init__(self, capacity: int, expert_capacity_ratio: float = 0.5):
        """
        Args:
            capacity: Total buffer capacity
            expert_capacity_ratio: Fraction of capacity for expert buffer (default: 0.5)
        """
        self.capacity = capacity
        self.expert_capacity = int(capacity * expert_capacity_ratio)
        self.policy_capacity = capacity - self.expert_capacity

        self.expert_buffer = ReplayBuffer(self.expert_capacity)
        self.policy_buffer = ReplayBuffer(self.policy_capacity)

    def push_expert(self, state: torch.Tensor, action: torch.Tensor,
                   next_state: torch.Tensor, done: bool):
        """Add expert transition with reward=+1.0."""
        self.expert_buffer.push(state, action, reward=1.0, next_state=next_state, done=done)

    def push_policy(self, state: torch.Tensor, action: torch.Tensor,
                   next_state: torch.Tensor, done: bool):
        """Add policy transition with reward=0.0."""
        self.policy_buffer.push(state, action, reward=0.0, next_state=next_state, done=done)

    def sample(self, batch_size: int, device: torch.device, expert_ratio: float = 0.5
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample mixed batch from expert and policy buffers.

        Args:
            batch_size: Total batch size
            device: Target device
            expert_ratio: Fraction of batch from expert buffer (default: 0.5)

        Returns:
            (states, actions, rewards, next_states, dones)
        """
        # Calculate samples from each buffer
        n_expert = int(batch_size * expert_ratio)
        n_policy = batch_size - n_expert

        # Adapt if buffers are empty or small
        if len(self.expert_buffer) == 0:
            n_expert = 0
            n_policy = batch_size
        elif len(self.policy_buffer) == 0:
            n_expert = batch_size
            n_policy = 0
        else:
            n_expert = min(n_expert, len(self.expert_buffer))
            n_policy = min(n_policy, len(self.policy_buffer))

        # Sample from each buffer
        if n_expert > 0 and n_policy > 0:
            e_states, e_actions, e_rewards, e_next_states, e_dones = self.expert_buffer.sample(n_expert, device)
            p_states, p_actions, p_rewards, p_next_states, p_dones = self.policy_buffer.sample(n_policy, device)

            # Concatenate
            states = torch.cat([e_states, p_states], dim=0)
            actions = torch.cat([e_actions, p_actions], dim=0)
            rewards = torch.cat([e_rewards, p_rewards], dim=0)
            next_states = torch.cat([e_next_states, p_next_states], dim=0)
            dones = torch.cat([e_dones, p_dones], dim=0)

        elif n_expert > 0:
            states, actions, rewards, next_states, dones = self.expert_buffer.sample(n_expert, device)
        else:
            states, actions, rewards, next_states, dones = self.policy_buffer.sample(n_policy, device)

        return states, actions, rewards, next_states, dones


# ====================================================================================
# Expert Buffer Initialization
# ====================================================================================

def initialize_expert_buffer(
    expert_records: List[Dict[str, Any]],
    encode: Callable,
    sqil_buffer: SQILReplayBuffer,
    device: torch.device
) -> None:
    """
    Pre-populate expert buffer from expert demonstration records.

    Converts expert trajectories to (s, a, s', done) format and stores them
    with reward=+1.0 in the expert buffer.

    Args:
        expert_records: List of expert demonstration records from collect_expert_trajectories
        encode: Encoding function from build_z_encoder (obs, t) -> state_features
        sqil_buffer: SQIL replay buffer to populate
        device: Device for tensor operations
    """
    # Group records by episode
    episodes = {}
    for record in expert_records:
        ep = record['episode']
        if ep not in episodes:
            episodes[ep] = []
        episodes[ep].append(record)

    # Process each episode
    for ep_id, ep_records in episodes.items():
        # Sort by step
        ep_records = sorted(ep_records, key=lambda r: r['step'])

        for i, record in enumerate(ep_records):
            t = record['step']
            obs = record['obs']
            action = np.asarray(record['action'], dtype=np.float32)

            # Encode current state
            state = torch.from_numpy(encode(obs, t)).float().to(device)
            action_tensor = torch.from_numpy(action).float().to(device)

            # Determine next state and done flag
            terminated = record.get('terminated', False)
            truncated = record.get('truncated', False)
            done = terminated or truncated

            if i < len(ep_records) - 1:
                # Not last step - get next observation
                next_record = ep_records[i + 1]
                next_obs = next_record['obs']
                next_t = next_record['step']
                next_state = torch.from_numpy(encode(next_obs, next_t)).float().to(device)
            else:
                # Last step - use current state (won't be used due to done=True)
                next_state = state

            # Push to expert buffer (reward automatically set to +1.0)
            sqil_buffer.push_expert(state, action_tensor, next_state, done)

    print(f"Initialized expert buffer with {len(sqil_buffer.expert_buffer)} transitions from {len(episodes)} episodes")


sqil_init_expert_buffer = initialize_expert_buffer


# ====================================================================================
# Helper Functions
# ====================================================================================

def soft_update(source: nn.Module, target: nn.Module, tau: float):
    """
    Polyak averaging: target = tau * source + (1 - tau) * target

    Args:
        source: Source network (current)
        target: Target network (lagging)
        tau: Polyak averaging coefficient
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# ====================================================================================
# SAC Update Functions
# ====================================================================================

def sac_update_critics(
    q1: SACQNetwork,
    q2: SACQNetwork,
    target_q1: SACQNetwork,
    target_q2: SACQNetwork,
    actor: ContinuousActor,
    replay_buffer: SQILReplayBuffer,
    batch_size: int,
    gamma: float,
    alpha: float,
    critic_optimizer_1: torch.optim.Optimizer,
    critic_optimizer_2: torch.optim.Optimizer,
    device: torch.device,
    action_low: float,
    action_high: float
) -> Dict[str, float]:
    """
    SAC critic update with entropy regularization.

    Bellman target: y = r + γ * (min(Q1(s',a'), Q2(s',a')) - α * log π(a'|s'))

    Args:
        q1, q2: Current Q-networks
        target_q1, target_q2: Target Q-networks (slowly updated)
        actor: Policy network
        replay_buffer: SQIL replay buffer
        batch_size: Batch size for sampling
        gamma: Discount factor
        alpha: Entropy coefficient
        critic_optimizer_1, critic_optimizer_2: Optimizers for Q1, Q2
        device: Device
        action_low, action_high: Action space bounds

    Returns:
        Dictionary of training metrics
    """
    # Sample batch from replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device)

    # Compute target Q-values using target networks
    with torch.no_grad():
        # Sample next actions from current policy
        next_action_dist = actor(next_states)
        next_actions, next_log_probs, _ = actor.act(next_states, deterministic=False)

        # Compute target Q-values (use minimum of two critics)
        q1_next = target_q1(next_states, next_actions)
        q2_next = target_q2(next_states, next_actions)
        q_next = torch.min(q1_next, q2_next)

        # SAC target: r + γ * (Q(s',a') - α * log π(a'|s'))
        target_q = rewards + gamma * (1.0 - dones) * (q_next - alpha * next_log_probs.unsqueeze(-1))

    # Compute current Q-values
    q1_pred = q1(states, actions)
    q2_pred = q2(states, actions)

    # MSE loss for both critics
    loss_q1 = F.mse_loss(q1_pred, target_q)
    loss_q2 = F.mse_loss(q2_pred, target_q)

    # Update Q1
    critic_optimizer_1.zero_grad()
    loss_q1.backward()
    critic_optimizer_1.step()

    # Update Q2
    critic_optimizer_2.zero_grad()
    loss_q2.backward()
    critic_optimizer_2.step()

    return {
        'loss_q1': loss_q1.item(),
        'loss_q2': loss_q2.item(),
        'mean_q1': q1_pred.mean().item(),
        'mean_q2': q2_pred.mean().item(),
        'mean_target_q': target_q.mean().item(),
        'mean_reward': rewards.mean().item()
    }


def sac_update_actor(
    actor: ContinuousActor,
    q1: SACQNetwork,
    q2: SACQNetwork,
    replay_buffer: SQILReplayBuffer,
    batch_size: int,
    alpha: float,
    actor_optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    SAC actor update: maximize Q-values and entropy.

    Actor objective: max E[Q(s,a) - α * log π(a|s)]

    Args:
        actor: Policy network
        q1, q2: Q-networks (frozen during actor update)
        replay_buffer: SQIL replay buffer
        batch_size: Batch size
        alpha: Entropy coefficient
        actor_optimizer: Actor optimizer
        device: Device

    Returns:
        Dictionary of training metrics
    """
    # Sample states only (actions will be sampled from current policy)
    states, _, _, _, _ = replay_buffer.sample(batch_size, device)

    # Sample actions from current policy WITH gradients (reparameterization trick).
    # NOTE: actor.act() is decorated @torch.no_grad, which severs the gradient
    # chain from Q(s,a) back through 'a' to the actor parameters, so we must
    # call forward() and apply the tanh squashing manually.
    dist = actor(states)
    u = dist.rsample()
    a_tanh = torch.tanh(u)
    actions = (a_tanh + 1) * 0.5 * (actor.high - actor.low) + actor.low

    log_det_tanh = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(dim=-1)
    log_det_scale = u.shape[-1] * np.log((actor.high - actor.low) / 2.0)
    log_probs = dist.log_prob(u) - (log_det_tanh + log_det_scale)

    # Compute Q-values (use minimum of two critics)
    q1_pi = q1(states, actions)
    q2_pi = q2(states, actions)
    q_pi = torch.min(q1_pi, q2_pi)

    # SAC actor loss: E[α * log π(a|s) - Q(s,a)]
    actor_loss = (alpha * log_probs.unsqueeze(-1) - q_pi).mean()

    # Update actor
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return {
        'actor_loss': actor_loss.item(),
        'mean_log_prob': log_probs.mean().item(),
        'mean_q_pi': q_pi.mean().item(),
        'mean_entropy': -log_probs.mean().item()
    }


# ====================================================================================
# Rollout and Evaluation
# ====================================================================================

def rollout_sqil_episode(
    env: PCH,
    actor: ContinuousActor,
    sqil_buffer: SQILReplayBuffer,
    encode: Callable,
    max_steps: int,
    device: torch.device,
    deterministic: bool = False,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Rollout one episode and collect transitions into SQIL policy buffer.

    Args:
        env: CausalGym PCH environment
        actor: Policy network
        sqil_buffer: SQIL replay buffer
        encode: State encoding function
        max_steps: Maximum episode length
        device: Device
        deterministic: Use deterministic actions (default: False)
        seed: Random seed

    Returns:
        Episode statistics
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    for step in range(max_steps):
        # Encode state
        state = torch.from_numpy(encode(obs, step)).float().to(device)

        # Sample action from policy
        with torch.no_grad():
            action, _, _ = actor.act(state.unsqueeze(0), deterministic=deterministic)

        action_np = action.squeeze(0).cpu().numpy()

        # Step environment
        next_obs, reward, terminated, truncated, next_info = env.do(lambda x: action_np, show_reward=True)

        total_reward += reward
        done = terminated or truncated
        steps += 1

        # Encode next state
        next_state = torch.from_numpy(encode(next_obs, step + 1)).float().to(device)

        # Push to policy buffer (reward=0.0 automatically)
        sqil_buffer.push_policy(state, action.squeeze(0), next_state, done)

        obs = next_obs

        if done:
            break

    return {
        'episode_return': total_reward,
        'episode_length': steps,
        'terminated': terminated,
        'truncated': truncated
    }


def evaluate_sqil_policy(
    env: PCH,
    actor: ContinuousActor,
    encode: Callable,
    max_steps: int,
    device: torch.device,
    num_episodes: int = 10,
    seed: Optional[int] = None
) -> float:
    """
    Evaluate SQIL policy deterministically.

    Args:
        env: Environment
        actor: Policy network
        encode: State encoding function
        max_steps: Max steps per episode
        device: Device
        num_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        Average episode return
    """
    total_returns = []

    for ep in range(num_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        ep_return = 0.0

        for step in range(max_steps):
            state = torch.from_numpy(encode(obs, step)).float().to(device)

            with torch.no_grad():
                action, _, _ = actor.act(state.unsqueeze(0), deterministic=True)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.do(lambda x: action_np, show_reward=True)

            ep_return += reward
            obs = next_obs

            if terminated or truncated:
                break

        total_returns.append(ep_return)

    return np.mean(total_returns)


# ====================================================================================
# Main Training Function
# ====================================================================================

def train_sqil(
    env: PCH,
    expert_records: List[Dict[str, Any]],
    device: torch.device,
    # Hyperparameters
    total_timesteps: int = 1_000_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    alpha: float = 0.2,
    tau: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    hidden_dim: int = 256,
    buffer_capacity: int = 1_000_000,
    expert_capacity_ratio: float = 0.5,
    expert_sampling_ratio: float = 0.5,
    updates_per_step: int = 1,
    start_steps: int = 10_000,
    max_episode_steps: int = 1000,
    eval_freq: int = 10_000,
    eval_episodes: int = 10,
    seed: Optional[int] = None,
    log_callback: Optional[Callable] = None
) -> Tuple[ContinuousActor, Dict[str, List]]:
    """
    Train SQIL policy using SAC algorithm.

    Args:
        env: CausalGym PCH environment
        expert_records: Expert demonstration records
        device: Device for training
        total_timesteps: Total training timesteps
        batch_size: Batch size for SAC updates
        gamma: Discount factor
        alpha: Entropy coefficient
        tau: Polyak averaging rate
        actor_lr: Actor learning rate
        critic_lr: Critic learning rate
        hidden_dim: Hidden dimension for networks
        buffer_capacity: Total replay buffer capacity
        expert_capacity_ratio: Fraction of buffer for expert data
        expert_sampling_ratio: Fraction of batch from expert buffer
        updates_per_step: SAC updates per environment step
        start_steps: Random exploration before training
        max_episode_steps: Max steps per episode
        eval_freq: Evaluation frequency (in timesteps)
        eval_episodes: Number of episodes for evaluation
        seed: Random seed
        log_callback: Optional callback for logging

    Returns:
        (trained_actor, logs)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Setup encoder from expert data
    sample_obs = expert_records[0]['obs']
    encode, z_dim, _, _ = build_z_encoder({}, sample_obs, calc_categorical_dims(env))
    state_dim = z_dim

    action_space = env.env.action_space
    action_dim = action_space.shape[0]
    action_low = float(action_space.low[0])
    action_high = float(action_space.high[0])

    print(f"SQIL Training Setup:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Action bounds: [{action_low}, {action_high}]")

    # Initialize networks
    actor = ContinuousActor(
        num_inputs=state_dim,
        num_outputs=action_dim,
        hidden_size=hidden_dim,
        action_low=action_low,
        action_high=action_high
    ).to(device)

    q1 = SACQNetwork(state_dim, action_dim, hidden_dim).to(device)
    q2 = SACQNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_q1 = copy.deepcopy(q1).to(device)
    target_q2 = copy.deepcopy(q2).to(device)

    # Freeze target networks
    for p in target_q1.parameters():
        p.requires_grad = False
    for p in target_q2.parameters():
        p.requires_grad = False

    # Optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer_1 = torch.optim.Adam(q1.parameters(), lr=critic_lr)
    critic_optimizer_2 = torch.optim.Adam(q2.parameters(), lr=critic_lr)

    # Replay buffer
    sqil_buffer = SQILReplayBuffer(buffer_capacity, expert_capacity_ratio)

    # Pre-populate expert buffer
    print("Initializing expert buffer...")
    initialize_expert_buffer(expert_records, encode, sqil_buffer, device)

    # Training loop
    timesteps = 0
    episode = 0
    logs = {
        'episode_returns': [],
        'episode_lengths': [],
        'critic_loss_q1': [],
        'critic_loss_q2': [],
        'actor_loss': [],
        'eval_returns': [],
        'eval_timesteps': []
    }

    print(f"\nStarting training for {total_timesteps} timesteps...")

    while timesteps < total_timesteps:
        # Rollout episode
        use_random = timesteps < start_steps

        if use_random:
            # Random exploration
            ep_data = rollout_sqil_episode(
                env, actor, sqil_buffer, encode,
                max_episode_steps, device, deterministic=False, seed=seed
            )
        else:
            # Policy rollout
            ep_data = rollout_sqil_episode(
                env, actor, sqil_buffer, encode,
                max_episode_steps, device, deterministic=False, seed=seed
            )

        timesteps += ep_data['episode_length']
        episode += 1

        logs['episode_returns'].append(ep_data['episode_return'])
        logs['episode_lengths'].append(ep_data['episode_length'])

        # Update networks (only after warmup and if enough policy data)
        if timesteps > start_steps and len(sqil_buffer.policy_buffer) >= batch_size:
            for _ in range(ep_data['episode_length'] * updates_per_step):
                # Update critics
                critic_metrics = sac_update_critics(
                    q1, q2, target_q1, target_q2, actor,
                    sqil_buffer, batch_size, gamma, alpha,
                    critic_optimizer_1, critic_optimizer_2,
                    device, action_low, action_high
                )
                logs['critic_loss_q1'].append(critic_metrics['loss_q1'])
                logs['critic_loss_q2'].append(critic_metrics['loss_q2'])

                # Update actor
                actor_metrics = sac_update_actor(
                    actor, q1, q2, sqil_buffer, batch_size, alpha,
                    actor_optimizer, device
                )
                logs['actor_loss'].append(actor_metrics['actor_loss'])

                # Soft update target networks
                soft_update(q1, target_q1, tau)
                soft_update(q2, target_q2, tau)

        # Evaluation
        if timesteps % eval_freq == 0 or timesteps >= total_timesteps:
            eval_return = evaluate_sqil_policy(
                env, actor, encode, max_episode_steps,
                device, eval_episodes, seed
            )
            logs['eval_returns'].append(eval_return)
            logs['eval_timesteps'].append(timesteps)

            print(f"Timestep {timesteps}/{total_timesteps} | Episode {episode} | "
                  f"Eval Return: {eval_return:.2f} | "
                  f"Train Return: {np.mean(logs['episode_returns'][-10:]):.2f}")

            if log_callback:
                log_callback({
                    'timesteps': timesteps,
                    'episode': episode,
                    'eval_return': eval_return
                })

    print("\nTraining complete!")
    return actor, logs


__all__ = [
    'train_sqil',
    'SQILReplayBuffer',
    'evaluate_sqil_policy',
    'rollout_sqil_episode',
    'initialize_expert_buffer',
    'sqil_init_expert_buffer',
    'sac_update_critics',
    'sac_update_actor',
    'soft_update',
]
