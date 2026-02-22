"""
SQIL (Soft Q Imitation Learning) — rewritten from scratch.

Binary reward signal: expert transitions → reward = +1, policy transitions → reward = 0.
Base RL algorithm: SAC with twin Q-networks, automatic entropy tuning, gradient clipping.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple

from causal_gym import PCH
from causal_rl.algo.imitation.gail.core_net import ContinuousActor
from .core_net import SACQNetwork, SQILQNetwork


# ── Replay buffers ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-capacity ring buffer storing (s, a, r, s', done) on CPU."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.next_states: list[torch.Tensor] = []
        self.dones: list[float] = []
        self._ptr = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else len(self.states)

    def push(self, state: torch.Tensor, action: torch.Tensor,
             reward: float, next_state: torch.Tensor, done: float):
        s = state.detach().cpu().view(-1)
        a = action.detach().cpu().view(-1)
        ns = next_state.detach().cpu().view(-1)
        if not self._full:
            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(reward)
            self.next_states.append(ns)
            self.dones.append(done)
            if len(self.states) >= self.capacity:
                self._full = True
                self._ptr = 0
        else:
            self.states[self._ptr] = s
            self.actions[self._ptr] = a
            self.rewards[self._ptr] = reward
            self.next_states[self._ptr] = ns
            self.dones[self._ptr] = done
            self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, n: int, device: torch.device):
        n = min(n, len(self))
        idx = np.random.randint(0, len(self), size=n)
        s = torch.stack([self.states[i] for i in idx]).to(device, dtype=torch.float32)
        a = torch.stack([self.actions[i] for i in idx]).to(device, dtype=torch.float32)
        r = torch.tensor([self.rewards[i] for i in idx], device=device, dtype=torch.float32).unsqueeze(-1)
        ns = torch.stack([self.next_states[i] for i in idx]).to(device, dtype=torch.float32)
        d = torch.tensor([self.dones[i] for i in idx], device=device, dtype=torch.float32).unsqueeze(-1)
        return s, a, r, ns, d


class SQILReplayBuffer:
    """Wrapper around two ReplayBuffers (expert / policy) with mixed sampling."""

    def __init__(self, capacity: int, expert_ratio: float = 0.5):
        exp_cap = int(capacity * expert_ratio)
        pol_cap = capacity - exp_cap
        self.expert_buffer = ReplayBuffer(exp_cap)
        self.policy_buffer = ReplayBuffer(pol_cap)

    def push_expert(self, s, a, ns, done):
        self.expert_buffer.push(s, a, 1.0, ns, float(done))

    def push_policy(self, s, a, ns, done):
        self.policy_buffer.push(s, a, 0.0, ns, float(done))

    def sample(self, batch_size: int, device: torch.device, expert_ratio: float = 0.5):
        n_e = int(batch_size * expert_ratio)
        n_p = batch_size - n_e
        if len(self.expert_buffer) == 0:
            n_e, n_p = 0, batch_size
        elif len(self.policy_buffer) == 0:
            n_e, n_p = batch_size, 0
        else:
            n_e = min(n_e, len(self.expert_buffer))
            n_p = min(n_p, len(self.policy_buffer))
        parts = []
        if n_e > 0:
            parts.append(self.expert_buffer.sample(n_e, device))
        if n_p > 0:
            parts.append(self.policy_buffer.sample(n_p, device))
        if len(parts) == 1:
            return parts[0]
        return tuple(torch.cat(ts, dim=0) for ts in zip(*parts))


# ── Expert buffer initialization ──────────────────────────────────────────────

def initialize_expert_buffer(
    expert_records: List[Dict[str, Any]],
    encode: Callable,
    sqil_buffer: SQILReplayBuffer,
    device: torch.device,
) -> None:
    """Fill the expert sub-buffer with (s, a, s', done) transitions, reward = +1."""
    episodes: Dict[int, list] = {}
    for rec in expert_records:
        episodes.setdefault(rec["episode"], []).append(rec)

    for ep_recs in episodes.values():
        ep_recs = sorted(ep_recs, key=lambda r: r["step"])
        for i, rec in enumerate(ep_recs):
            t = rec["step"]
            obs = rec["obs"]
            action = np.asarray(rec["action"], dtype=np.float32)
            state = torch.from_numpy(encode(obs, t)).float()
            action_t = torch.from_numpy(action).float()
            terminated = rec.get("terminated", False)
            truncated = rec.get("truncated", False)
            done = terminated or truncated
            if i < len(ep_recs) - 1:
                nr = ep_recs[i + 1]
                next_state = torch.from_numpy(encode(nr["obs"], nr["step"])).float()
            else:
                next_state = state  # terminal; won't be bootstrapped
            sqil_buffer.push_expert(state, action_t, next_state, done)

    print(f"Expert buffer: {len(sqil_buffer.expert_buffer)} transitions "
          f"from {len(episodes)} episodes")


sqil_init_expert_buffer = initialize_expert_buffer  # alias


# ── Helpers ───────────────────────────────────────────────────────────────────

def soft_update(src: nn.Module, tgt: nn.Module, tau: float):
    for tp, sp in zip(tgt.parameters(), src.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def _reparameterize_actor(actor: ContinuousActor, states: torch.Tensor):
    """Sample actions from actor WITH gradient flow (reparameterization trick).

    Returns (actions, log_probs) where actions are squashed to [low, high].
    """
    dist = actor(states)
    u = dist.rsample()
    a_tanh = torch.tanh(u)
    actions = (a_tanh + 1.0) * 0.5 * (actor.high - actor.low) + actor.low

    log_det_tanh = torch.log(1.0 - a_tanh.pow(2) + 1e-6).sum(dim=-1)
    log_det_scale = u.shape[-1] * np.log((actor.high - actor.low) / 2.0)
    log_probs = dist.log_prob(u) - (log_det_tanh + log_det_scale)
    return actions, log_probs


# ── SAC update step ───────────────────────────────────────────────────────────

def sac_update(
    q1: SACQNetwork,
    q2: SACQNetwork,
    tq1: SACQNetwork,
    tq2: SACQNetwork,
    actor: ContinuousActor,
    log_alpha: torch.Tensor,
    target_entropy: float,
    q1_opt: torch.optim.Optimizer,
    q2_opt: torch.optim.Optimizer,
    actor_opt: torch.optim.Optimizer,
    alpha_opt: torch.optim.Optimizer,
    buffer: SQILReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """One SAC update: critics → actor → alpha → soft-update targets."""
    alpha = log_alpha.exp().item()
    states, actions, rewards, next_states, dones = buffer.sample(batch_size, device)

    # ── Critic target ──
    with torch.no_grad():
        na, nlp = _reparameterize_actor(actor, next_states)
        tq1_val = tq1(next_states, na)
        tq2_val = tq2(next_states, na)
        target_q = rewards + gamma * (1.0 - dones) * (
            torch.min(tq1_val, tq2_val) - alpha * nlp.unsqueeze(-1)
        )

    # ── Update Q1, Q2 ──
    q1_pred = q1(states, actions)
    q2_pred = q2(states, actions)
    loss_q1 = F.mse_loss(q1_pred, target_q)
    loss_q2 = F.mse_loss(q2_pred, target_q)

    q1_opt.zero_grad(set_to_none=True)
    loss_q1.backward()
    torch.nn.utils.clip_grad_norm_(q1.parameters(), max_grad_norm)
    q1_opt.step()

    q2_opt.zero_grad(set_to_none=True)
    loss_q2.backward()
    torch.nn.utils.clip_grad_norm_(q2.parameters(), max_grad_norm)
    q2_opt.step()

    # ── Update actor ──
    a_pi, lp_pi = _reparameterize_actor(actor, states)
    q_pi = torch.min(q1(states, a_pi), q2(states, a_pi))
    actor_loss = (alpha * lp_pi.unsqueeze(-1) - q_pi).mean()

    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    actor_opt.step()

    # ── Update alpha (automatic entropy tuning) ──
    alpha_loss = -(log_alpha * (lp_pi.detach() + target_entropy)).mean()
    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()

    return {
        "loss_q1": loss_q1.item(),
        "loss_q2": loss_q2.item(),
        "actor_loss": actor_loss.item(),
        "alpha": log_alpha.exp().item(),
        "mean_q": q1_pred.mean().item(),
        "mean_reward": rewards.mean().item(),
    }


# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout_sqil_episode(
    env: PCH,
    actor: ContinuousActor,
    sqil_buffer: SQILReplayBuffer,
    encode: Callable,
    max_steps: int,
    device: torch.device,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Roll out one episode; push transitions to *policy* buffer with reward = 0."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    for t in range(max_steps):
        z_np = encode(obs, t)
        z = torch.from_numpy(z_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _ = actor.act(z, deterministic=deterministic)
        a_np = action.squeeze(0).cpu().numpy().astype(np.float32)

        next_obs, reward, terminated, truncated, _ = env.do(
            lambda _: a_np, show_reward=True
        )
        done = terminated or truncated
        total_reward += reward
        steps += 1

        nz_np = encode(next_obs, t + 1)
        nz = torch.from_numpy(nz_np).float()
        sqil_buffer.push_policy(
            torch.from_numpy(z_np).float(), action.squeeze(0).cpu(), nz, done
        )

        obs = next_obs
        if done:
            break

    return {
        "episode_return": total_reward,
        "episode_length": steps,
        "terminated": terminated,
        "truncated": truncated,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_sqil_policy(
    env: PCH,
    actor: ContinuousActor,
    encode: Callable,
    max_steps: int,
    device: torch.device,
    num_episodes: int = 10,
    seed: Optional[int] = None,
) -> float:
    """Deterministic evaluation; returns mean episode return."""
    returns: list[float] = []
    for ep in range(num_episodes):
        ep_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=ep_seed)
        ep_ret = 0.0
        for t in range(max_steps):
            z = torch.from_numpy(encode(obs, t)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = actor.act(z, deterministic=True)
            a_np = action.squeeze(0).cpu().numpy().astype(np.float32)
            obs, reward, terminated, truncated, _ = env.do(
                lambda _: a_np, show_reward=True
            )
            ep_ret += reward
            if terminated or truncated:
                break
        returns.append(ep_ret)
    return float(np.mean(returns))


# ── Legacy wrappers matching old API (used by existing notebook) ──────────────

def sac_update_critics(
    q1, q2, tq1, tq2, actor, buffer, batch_size, gamma, alpha,
    q1_opt, q2_opt, device, action_low, action_high,
    max_grad_norm: float = 1.0,
):
    """Backward-compat wrapper: critic-only update with fixed alpha."""
    states, actions, rewards, next_states, dones = buffer.sample(batch_size, device)
    with torch.no_grad():
        na, nlp = _reparameterize_actor(actor, next_states)
        target_q = rewards + gamma * (1.0 - dones) * (
            torch.min(tq1(next_states, na), tq2(next_states, na))
            - alpha * nlp.unsqueeze(-1)
        )
    q1_pred = q1(states, actions)
    q2_pred = q2(states, actions)
    loss_q1 = F.mse_loss(q1_pred, target_q)
    loss_q2 = F.mse_loss(q2_pred, target_q)

    q1_opt.zero_grad(set_to_none=True)
    loss_q1.backward()
    torch.nn.utils.clip_grad_norm_(q1.parameters(), max_grad_norm)
    q1_opt.step()

    q2_opt.zero_grad(set_to_none=True)
    loss_q2.backward()
    torch.nn.utils.clip_grad_norm_(q2.parameters(), max_grad_norm)
    q2_opt.step()

    return {
        "loss_q1": loss_q1.item(),
        "loss_q2": loss_q2.item(),
        "mean_q1": q1_pred.mean().item(),
        "mean_q2": q2_pred.mean().item(),
        "mean_target_q": target_q.mean().item(),
        "mean_reward": rewards.mean().item(),
    }


def sac_update_actor(
    actor, q1, q2, buffer, batch_size, alpha, actor_opt, device,
    max_grad_norm: float = 1.0,
):
    """Backward-compat wrapper: actor-only update with fixed alpha."""
    states, _, _, _, _ = buffer.sample(batch_size, device)
    a_pi, lp_pi = _reparameterize_actor(actor, states)
    q_pi = torch.min(q1(states, a_pi), q2(states, a_pi))
    actor_loss = (alpha * lp_pi.unsqueeze(-1) - q_pi).mean()

    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    actor_opt.step()

    return {
        "actor_loss": actor_loss.item(),
        "mean_log_prob": lp_pi.mean().item(),
        "mean_q_pi": q_pi.mean().item(),
        "mean_entropy": -lp_pi.mean().item(),
    }


# ── Main training loop ───────────────────────────────────────────────────────

def train_sqil(
    env: PCH,
    expert_records: List[Dict[str, Any]],
    encode: Callable,
    device: torch.device,
    *,
    state_dim: int,
    action_dim: int,
    action_low: float = -1.0,
    action_high: float = 1.0,
    total_timesteps: int = 1_000_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    alpha_lr: float = 3e-4,
    hidden_dim: int = 256,
    # Q-network architecture (defaults reproduce old SACQNetwork behavior)
    q_network_cls: str = "simple",       # "simple" = SACQNetwork, "residual" = SQILQNetwork
    num_blocks: int = 3,
    dropout: float = 0.05,
    layernorm: bool = True,
    # Scheduling (defaults reproduce old behavior)
    utd_ratio: Optional[float] = None,   # if set, overrides updates_per_step
    cosine_lr: bool = False,             # if True, wrap critic optimizers with CosineAnnealingLR
    buffer_capacity: int = 1_000_000,
    expert_capacity_ratio: float = 0.5,
    updates_per_step: int = 1,
    start_steps: int = 5_000,
    max_episode_steps: int = 1000,
    eval_freq: int = 10_000,
    eval_episodes: int = 10,
    max_grad_norm: float = 1.0,
    seed: Optional[int] = None,
    log_callback: Optional[Callable] = None,
) -> Tuple[ContinuousActor, Dict[str, List]]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ── Networks ──
    actor = ContinuousActor(
        num_inputs=state_dim, num_outputs=action_dim,
        hidden_size=hidden_dim, action_low=action_low, action_high=action_high,
    ).to(device)
    if q_network_cls == "residual":
        q1 = SQILQNetwork(state_dim, action_dim, hidden_dim,
                           num_blocks=num_blocks, dropout=dropout,
                           layernorm=layernorm).to(device)
        q2 = SQILQNetwork(state_dim, action_dim, hidden_dim,
                           num_blocks=num_blocks, dropout=dropout,
                           layernorm=layernorm).to(device)
    else:
        q1 = SACQNetwork(state_dim, action_dim, hidden_dim).to(device)
        q2 = SACQNetwork(state_dim, action_dim, hidden_dim).to(device)
    tq1 = copy.deepcopy(q1)
    tq2 = copy.deepcopy(q2)
    for p in tq1.parameters():
        p.requires_grad = False
    for p in tq2.parameters():
        p.requires_grad = False

    # ── Automatic entropy coefficient ──
    target_entropy = -float(action_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)

    # ── Optimizers ──
    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=critic_lr)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=critic_lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=alpha_lr)

    # ── Optional cosine LR schedule for critics ──
    q1_scheduler = None
    q2_scheduler = None
    if cosine_lr:
        effective_utd = utd_ratio if utd_ratio is not None else float(updates_per_step)
        T_max = max(1, int(total_timesteps * effective_utd))
        q1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(q1_opt, T_max=T_max)
        q2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(q2_opt, T_max=T_max)

    # ── Buffer ──
    buf = SQILReplayBuffer(buffer_capacity, expert_capacity_ratio)
    initialize_expert_buffer(expert_records, encode, buf, device)

    # ── Training ──
    timesteps = 0
    episode = 0
    logs: Dict[str, list] = {
        "episode_returns": [], "episode_lengths": [],
        "eval_returns": [], "eval_timesteps": [],
        "loss_q1": [], "actor_loss": [], "alpha": [],
    }

    print(f"SQIL training: state_dim={state_dim}, action_dim={action_dim}, "
          f"action=[{action_low}, {action_high}]")

    while timesteps < total_timesteps:
        ep_data = rollout_sqil_episode(
            env, actor, buf, encode, max_episode_steps, device,
            deterministic=False, seed=(seed + episode) if seed else None,
        )
        timesteps += ep_data["episode_length"]
        episode += 1
        logs["episode_returns"].append(ep_data["episode_return"])
        logs["episode_lengths"].append(ep_data["episode_length"])

        # Sanity check on first real episode
        if episode == 1:
            assert len(buf.policy_buffer) > 0, (
                "BUG: policy buffer empty after rollout!")

        if timesteps > start_steps and len(buf.policy_buffer) >= batch_size // 2:
            if utd_ratio is not None:
                n_updates = max(1, int(ep_data["episode_length"] * utd_ratio))
            else:
                n_updates = ep_data["episode_length"] * updates_per_step
            for _ in range(n_updates):
                metrics = sac_update(
                    q1, q2, tq1, tq2, actor, log_alpha, target_entropy,
                    q1_opt, q2_opt, actor_opt, alpha_opt,
                    buf, batch_size, gamma, device, max_grad_norm,
                )
                soft_update(q1, tq1, tau)
                soft_update(q2, tq2, tau)

                if q1_scheduler is not None:
                    q1_scheduler.step()
                    q2_scheduler.step()

                logs["loss_q1"].append(metrics["loss_q1"])
                logs["actor_loss"].append(metrics["actor_loss"])
                logs["alpha"].append(metrics["alpha"])

        if timesteps % eval_freq < ep_data["episode_length"] or timesteps >= total_timesteps:
            eval_ret = evaluate_sqil_policy(
                env, actor, encode, max_episode_steps,
                device, eval_episodes, seed=42,
            )
            logs["eval_returns"].append(eval_ret)
            logs["eval_timesteps"].append(timesteps)
            alpha_val = log_alpha.exp().item()
            print(
                f"[SQIL ep {episode}] ts={timesteps}, "
                f"eval={eval_ret:.2f}, "
                f"train={np.mean(logs['episode_returns'][-10:]):.2f}, "
                f"alpha={alpha_val:.4f}"
            )
            if log_callback:
                log_callback({"timesteps": timesteps, "episode": episode,
                              "eval_return": eval_ret})

    print("SQIL training complete.")
    return actor, logs


__all__ = [
    "train_sqil",
    "SQILReplayBuffer",
    "ReplayBuffer",
    "evaluate_sqil_policy",
    "rollout_sqil_episode",
    "initialize_expert_buffer",
    "sqil_init_expert_buffer",
    "sac_update",
    "sac_update_critics",
    "sac_update_actor",
    "soft_update",
]
