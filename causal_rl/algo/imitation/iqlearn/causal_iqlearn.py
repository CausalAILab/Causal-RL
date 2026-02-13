"""
IQ-Learn (Inverse Q-Learning) implementation with causal integration.

IQ-Learn learns Q-functions that satisfy Bellman optimality on expert data
through an inverse soft-Q learning objective. Unlike SQIL, it learns implicit
rewards through the Q-function rather than using explicit binary rewards.

Key innovation: Combines expert Bellman consistency with policy regularization
to learn from both expert demonstrations and self-collected data.
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
from .core_net import IQLearnQNetwork


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


class IQLearnReplayBuffer:
    """
    IQ-Learn Replay Buffer with separate expert and policy sub-buffers.

    Expert transitions are labeled with reward=+1.0, policy transitions with reward=0.0.
    The reward field is used to identify expert vs policy samples for loss computation,
    not for actual Q-learning rewards (IQ-Learn learns implicit rewards).
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
        """Add expert transition (marked with reward=+1.0 for identification)."""
        self.expert_buffer.push(state, action, reward=1.0, next_state=next_state, done=done)

    def push_policy(self, state: torch.Tensor, action: torch.Tensor,
                   next_state: torch.Tensor, done: bool):
        """Add policy transition (marked with reward=0.0 for identification)."""
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
            Note: rewards are used only to identify expert (1.0) vs policy (0.0) samples
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

def iq_init_expert_buffer(
    expert_records: List[Dict[str, Any]],
    encode: Callable,
    iqlearn_buffer: IQLearnReplayBuffer,
    device: torch.device
) -> None:
    """
    Pre-populate expert buffer from expert demonstration records.

    Converts expert trajectories to (s, a, s', done) format and stores them
    in the expert buffer (marked with reward=1.0 for identification).

    Args:
        expert_records: List of expert demonstration records
        encode: Encoding function from build_z_encoder (obs, t) -> state_features
        iqlearn_buffer: IQ-Learn replay buffer to populate
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

            # Push to expert buffer (marked with reward=1.0)
            iqlearn_buffer.push_expert(state, action_tensor, next_state, done)

    print(f"Initialized expert buffer with {len(iqlearn_buffer.expert_buffer)} transitions from {len(episodes)} episodes")


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
# IQ-Learn Update Functions
# ====================================================================================

def iqlearn_update_critic(
    q_network: IQLearnQNetwork,
    target_q_network: IQLearnQNetwork,
    actor: ContinuousActor,
    replay_buffer: IQLearnReplayBuffer,
    batch_size: int,
    gamma: float,
    lambda_reg: float,
    critic_optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_v_samples: int = 10
) -> Dict[str, float]:
    """
    IQ-Learn critic update with Bellman consistency and regularization.

    Loss: L_Q = E_expert[(Q(s,a) - γV(s'))²]
                + λ * E_policy[(log π(a|s) - Q(s,a) + V(s))²]

    Args:
        q_network: Current Q-network
        target_q_network: Target Q-network (for V(s') computation)
        actor: Policy network
        replay_buffer: IQ-Learn replay buffer
        batch_size: Batch size for sampling
        gamma: Discount factor
        lambda_reg: Regularization coefficient (λ)
        critic_optimizer: Q-network optimizer
        device: Device
        num_v_samples: Number of samples for V(s) computation

    Returns:
        Dictionary of training metrics
    """
    # Sample mixed batch (50% expert, 50% policy)
    states, actions, rewards, next_states, dones = replay_buffer.sample(
        batch_size, device, expert_ratio=0.5
    )

    # Identify expert vs policy samples (expert have reward=1.0)
    is_expert = (rewards == 1.0).squeeze(-1)
    is_policy = ~is_expert

    # Compute V(s') using target network (for stability)
    with torch.no_grad():
        v_next = target_q_network.compute_v(next_states, actor, num_v_samples)

    # Current Q(s,a)
    q_values = q_network(states, actions)

    # Expert loss: maximize implicit reward on expert data
    # IQ-Learn objective: maximize E_expert[Q(s,a) - γV(s')]
    # The quantity Q(s,a) - γV(s') is the implicit reward; it should be high for expert data.
    if is_expert.any():
        expert_q = q_values[is_expert]
        expert_v_next = v_next[is_expert]
        expert_dones = dones[is_expert]

        implicit_reward = expert_q - gamma * (1.0 - expert_dones) * expert_v_next
        expert_loss = -implicit_reward.mean()
    else:
        expert_loss = torch.tensor(0.0, device=device)

    # Policy regularization loss: (log π(a|s) - Q(s,a) + V(s))²
    if is_policy.any():
        policy_states = states[is_policy]
        policy_actions = actions[is_policy]
        policy_q = q_values[is_policy]

        # Compute log π(a|s) using evaluate_actions, which correctly
        # inverts the tanh squashing and applies the Jacobian correction.
        with torch.no_grad():
            log_prob, _ = actor.evaluate_actions(policy_states, policy_actions)
        log_prob = log_prob.unsqueeze(-1)  # [n_policy, 1]

        # Compute V(s) for policy states
        v_policy = q_network.compute_v(policy_states, actor, num_v_samples)

        # Regularization: (log π - Q + V)²
        # Target is zero (perfect consistency)
        reg_residual = log_prob - policy_q + v_policy
        reg_loss = F.mse_loss(reg_residual, torch.zeros_like(reg_residual))

        # Chi-squared divergence regularization on policy implicit rewards
        # Prevents Q-value divergence by penalizing large implicit rewards on policy data
        policy_v_next = v_next[is_policy]
        policy_dones = dones[is_policy]
        chi2_residual = policy_q - gamma * (1.0 - policy_dones) * policy_v_next
        chi2_loss = 0.5 * (chi2_residual ** 2).mean()
    else:
        reg_loss = torch.tensor(0.0, device=device)
        chi2_loss = torch.tensor(0.0, device=device)

    # Combined loss
    total_loss = expert_loss + lambda_reg * reg_loss + chi2_loss

    # Update Q-network
    critic_optimizer.zero_grad()
    total_loss.backward()
    critic_optimizer.step()

    return {
        'critic_loss': total_loss.item(),
        'expert_loss': expert_loss.item(),
        'reg_loss': reg_loss.item(),
        'chi2_loss': chi2_loss.item(),
        'mean_q': q_values.mean().item(),
        'mean_v_next': v_next.mean().item()
    }


def iqlearn_update_actor(
    actor: ContinuousActor,
    q_network: IQLearnQNetwork,
    replay_buffer: IQLearnReplayBuffer,
    batch_size: int,
    alpha: float,
    actor_optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    IQ-Learn actor update: standard SAC objective.

    Actor objective: max E[Q(s,a) - α * log π(a|s)]

    Args:
        actor: Policy network
        q_network: Q-network (frozen during actor update)
        replay_buffer: IQ-Learn replay buffer
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

    # Compute Q-values
    q_pi = q_network(states, actions)

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

def rollout_iqlearn_episode(
    env: PCH,
    actor: ContinuousActor,
    iqlearn_buffer: IQLearnReplayBuffer,
    encode: Callable,
    max_steps: int,
    device: torch.device,
    deterministic: bool = False,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Rollout one episode and collect transitions into IQ-Learn policy buffer.

    Args:
        env: CausalGym PCH environment
        actor: Policy network
        iqlearn_buffer: IQ-Learn replay buffer
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

        # Push to policy buffer (marked with reward=0.0)
        iqlearn_buffer.push_policy(state, action.squeeze(0), next_state, done)

        obs = next_obs

        if done:
            break

    return {
        'episode_return': total_reward,
        'episode_length': steps,
        'terminated': terminated,
        'truncated': truncated
    }


def evaluate_iqlearn_policy(
    env: PCH,
    actor: ContinuousActor,
    encode: Callable,
    max_steps: int,
    device: torch.device,
    num_episodes: int = 10,
    seed: Optional[int] = None
) -> float:
    """
    Evaluate IQ-Learn policy deterministically.

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

def train_iqlearn(
    env: PCH,
    expert_records: List[Dict[str, Any]],
    device: torch.device,
    # Hyperparameters
    total_timesteps: int = 1_000_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    lambda_reg: float = 1.0,
    alpha: float = 0.2,
    tau: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    hidden_dim: int = 256,
    buffer_capacity: int = 1_000_000,
    expert_capacity_ratio: float = 0.5,
    expert_sampling_ratio: float = 0.5,
    num_v_samples: int = 10,
    updates_per_step: int = 1,
    start_steps: int = 10_000,
    max_episode_steps: int = 1000,
    eval_freq: int = 10_000,
    eval_episodes: int = 10,
    seed: Optional[int] = None,
    log_callback: Optional[Callable] = None
) -> Tuple[ContinuousActor, Dict[str, List]]:
    """
    Train IQ-Learn policy using inverse soft-Q learning.

    Args:
        env: CausalGym PCH environment
        expert_records: Expert demonstration records
        device: Device for training
        total_timesteps: Total training timesteps
        batch_size: Batch size for updates
        gamma: Discount factor
        lambda_reg: Policy regularization weight (λ)
        alpha: Entropy coefficient
        tau: Polyak averaging rate
        actor_lr: Actor learning rate
        critic_lr: Critic learning rate
        hidden_dim: Hidden dimension for networks
        buffer_capacity: Total replay buffer capacity
        expert_capacity_ratio: Fraction of buffer for expert data
        expert_sampling_ratio: Fraction of batch from expert buffer
        num_v_samples: Samples for V(s) computation
        updates_per_step: Updates per environment step
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

    print(f"IQ-Learn Training Setup:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Action bounds: [{action_low}, {action_high}]")
    print(f"  Lambda (reg weight): {lambda_reg}")
    print(f"  V(s) samples: {num_v_samples}")

    # Initialize networks
    actor = ContinuousActor(
        num_inputs=state_dim,
        num_outputs=action_dim,
        hidden_size=hidden_dim,
        action_low=action_low,
        action_high=action_high
    ).to(device)

    q_network = IQLearnQNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_q_network = copy.deepcopy(q_network).to(device)

    # Freeze target network
    for p in target_q_network.parameters():
        p.requires_grad = False

    # Optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(q_network.parameters(), lr=critic_lr)

    # Replay buffer
    iqlearn_buffer = IQLearnReplayBuffer(buffer_capacity, expert_capacity_ratio)

    # Pre-populate expert buffer
    print("Initializing expert buffer...")
    iq_init_expert_buffer(expert_records, encode, iqlearn_buffer, device)

    # Training loop
    timesteps = 0
    episode = 0
    logs = {
        'episode_returns': [],
        'episode_lengths': [],
        'critic_loss': [],
        'expert_loss': [],
        'reg_loss': [],
        'actor_loss': [],
        'eval_returns': [],
        'eval_timesteps': []
    }

    print(f"\nStarting IQ-Learn training for {total_timesteps} timesteps...")

    while timesteps < total_timesteps:
        # Rollout episode
        use_random = timesteps < start_steps

        if use_random:
            # Random exploration (still use actor but with high stochasticity)
            ep_data = rollout_iqlearn_episode(
                env, actor, iqlearn_buffer, encode,
                max_episode_steps, device, deterministic=False, seed=seed
            )
        else:
            # Policy rollout
            ep_data = rollout_iqlearn_episode(
                env, actor, iqlearn_buffer, encode,
                max_episode_steps, device, deterministic=False, seed=seed
            )

        timesteps += ep_data['episode_length']
        episode += 1

        logs['episode_returns'].append(ep_data['episode_return'])
        logs['episode_lengths'].append(ep_data['episode_length'])

        # Update networks (only after warmup and if enough policy data)
        if timesteps > start_steps and len(iqlearn_buffer.policy_buffer) >= batch_size:
            for _ in range(ep_data['episode_length'] * updates_per_step):
                # Update critic (IQ-Learn loss)
                critic_metrics = iqlearn_update_critic(
                    q_network, target_q_network, actor,
                    iqlearn_buffer, batch_size, gamma, lambda_reg,
                    critic_optimizer, device, num_v_samples
                )
                logs['critic_loss'].append(critic_metrics['critic_loss'])
                logs['expert_loss'].append(critic_metrics['expert_loss'])
                logs['reg_loss'].append(critic_metrics['reg_loss'])

                # Update actor (SAC objective)
                actor_metrics = iqlearn_update_actor(
                    actor, q_network, iqlearn_buffer, batch_size, alpha,
                    actor_optimizer, device
                )
                logs['actor_loss'].append(actor_metrics['actor_loss'])

                # Soft update target network
                soft_update(q_network, target_q_network, tau)

        # Evaluation
        if timesteps % eval_freq == 0 or timesteps >= total_timesteps:
            eval_return = evaluate_iqlearn_policy(
                env, actor, encode, max_episode_steps,
                device, eval_episodes, seed
            )
            logs['eval_returns'].append(eval_return)
            logs['eval_timesteps'].append(timesteps)

            # Print stats
            recent_expert_loss = np.mean(logs['expert_loss'][-100:]) if logs['expert_loss'] else 0.0
            recent_reg_loss = np.mean(logs['reg_loss'][-100:]) if logs['reg_loss'] else 0.0

            print(f"Timestep {timesteps}/{total_timesteps} | Episode {episode} | "
                  f"Eval Return: {eval_return:.2f} | "
                  f"Train Return: {np.mean(logs['episode_returns'][-10:]):.2f} | "
                  f"Expert Loss: {recent_expert_loss:.4f} | "
                  f"Reg Loss: {recent_reg_loss:.4f}")

            if log_callback:
                log_callback({
                    'timesteps': timesteps,
                    'episode': episode,
                    'eval_return': eval_return,
                    'expert_loss': recent_expert_loss,
                    'reg_loss': recent_reg_loss
                })

    print("\nIQ-Learn training complete!")
    return actor, logs


__all__ = [
    'train_iqlearn',
    'IQLearnReplayBuffer',
    'evaluate_iqlearn_policy',
    'rollout_iqlearn_episode',
    'iq_init_expert_buffer',
    'iqlearn_update_critic',
    'iqlearn_update_actor',
    'soft_update',
]
