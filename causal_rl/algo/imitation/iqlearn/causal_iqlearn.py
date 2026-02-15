"""
IQ-Learn (Inverse soft-Q Learning) — rewritten from scratch.

Learns a Q-function whose implicit reward  r(s,a) = Q(s,a) − γ V(s')  is high
for expert state-action pairs.  Uses the chi-squared divergence formulation:

    L_critic = −E_expert[ Q(s,a) − γ V(s') ]
             + ½ E_all[ (Q(s,a) − γ V(s'))² ]

The actor is updated with the standard SAC objective.

Key differences from the previous (broken) implementation:
  • Twin Q-networks (Q1, Q2) for stability
  • V(s) = E_{a~π}[Q(s,a) − α log π(a|s)]  (entropy-regularised, NOT logsumexp)
  • Automatic entropy tuning (learnable log_alpha)
  • Gradient clipping to max norm 1.0
  • Chi-squared divergence variant of the IQ-Learn loss
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple

from causal_gym import PCH
from causal_rl.algo.imitation.gail.core_net import ContinuousActor
from .core_net import IQLearnQNetwork


# ── Replay buffers ────────────────────────────────────────────────────────────
# (Identical structure to SQIL — separate expert / policy sub-buffers)

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

    def push(self, state, action, reward, next_state, done):
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


class IQLearnReplayBuffer:
    """Expert / policy split buffer."""

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

    def sample_expert(self, n: int, device: torch.device):
        return self.expert_buffer.sample(n, device)

    def sample_policy(self, n: int, device: torch.device):
        return self.policy_buffer.sample(n, device)


# ── Expert buffer initialization ──────────────────────────────────────────────

def iq_init_expert_buffer(
    expert_records: List[Dict[str, Any]],
    encode: Callable,
    iq_buffer: IQLearnReplayBuffer,
    device: torch.device,
) -> None:
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
                next_state = state
            iq_buffer.push_expert(state, action_t, next_state, done)

    print(f"Expert buffer: {len(iq_buffer.expert_buffer)} transitions "
          f"from {len(episodes)} episodes")


initialize_expert_buffer = iq_init_expert_buffer  # alias


# ── Helpers ───────────────────────────────────────────────────────────────────

def soft_update(src: nn.Module, tgt: nn.Module, tau: float):
    for tp, sp in zip(tgt.parameters(), src.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def _reparameterize_actor(actor: ContinuousActor, states: torch.Tensor):
    """Sample actions WITH gradient flow. Returns (actions, log_probs)."""
    dist = actor(states)
    u = dist.rsample()
    a_tanh = torch.tanh(u)
    actions = (a_tanh + 1.0) * 0.5 * (actor.high - actor.low) + actor.low
    log_det_tanh = torch.log(1.0 - a_tanh.pow(2) + 1e-6).sum(dim=-1)
    log_det_scale = u.shape[-1] * np.log((actor.high - actor.low) / 2.0)
    log_probs = dist.log_prob(u) - (log_det_tanh + log_det_scale)
    return actions, log_probs


# ── IQ-Learn critic update ───────────────────────────────────────────────────

def iqlearn_update_critic(
    q1: IQLearnQNetwork,
    q2: IQLearnQNetwork,
    tq1: IQLearnQNetwork,
    tq2: IQLearnQNetwork,
    actor: ContinuousActor,
    alpha: float,
    buffer: IQLearnReplayBuffer,
    batch_size: int,
    gamma: float,
    q1_opt: torch.optim.Optimizer,
    q2_opt: torch.optim.Optimizer,
    device: torch.device,
    num_v_samples: int = 10,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Chi-squared IQ-Learn critic loss using twin Q-networks.

    L = −E_expert[r_imp] + ½ E_all[r_imp²]

    where r_imp(s,a,s') = Q(s,a) − γ(1−d)V(s')  is the implicit reward.
    V(s') is computed with the *target* Q-networks for stability.
    """
    half = batch_size // 2
    e_s, e_a, _, e_ns, e_d = buffer.sample_expert(half, device)
    p_s, p_a, _, p_ns, p_d = buffer.sample_policy(half, device)

    # Concatenate for the regularisation term (all data)
    all_s = torch.cat([e_s, p_s], dim=0)
    all_a = torch.cat([e_a, p_a], dim=0)
    all_ns = torch.cat([e_ns, p_ns], dim=0)
    all_d = torch.cat([e_d, p_d], dim=0)

    # V(s') from target networks (no grad through actor or target Q)
    with torch.no_grad():
        v_next_1 = tq1.compute_v(all_ns, actor, alpha, num_v_samples)
        v_next_2 = tq2.compute_v(all_ns, actor, alpha, num_v_samples)
        v_next = torch.min(v_next_1, v_next_2)

    n_expert = e_s.size(0)

    # ── Loss for Q1 ──
    q1_all = q1(all_s, all_a)
    r_imp_1 = q1_all - gamma * (1.0 - all_d) * v_next
    expert_reward_1 = r_imp_1[:n_expert]
    loss_q1 = -expert_reward_1.mean() + 0.5 * (r_imp_1 ** 2).mean()

    q1_opt.zero_grad(set_to_none=True)
    loss_q1.backward()
    torch.nn.utils.clip_grad_norm_(q1.parameters(), max_grad_norm)
    q1_opt.step()

    # ── Loss for Q2 ──
    q2_all = q2(all_s, all_a)
    r_imp_2 = q2_all - gamma * (1.0 - all_d) * v_next
    expert_reward_2 = r_imp_2[:n_expert]
    loss_q2 = -expert_reward_2.mean() + 0.5 * (r_imp_2 ** 2).mean()

    q2_opt.zero_grad(set_to_none=True)
    loss_q2.backward()
    torch.nn.utils.clip_grad_norm_(q2.parameters(), max_grad_norm)
    q2_opt.step()

    return {
        "critic_loss": 0.5 * (loss_q1.item() + loss_q2.item()),
        "expert_reward_mean": 0.5 * (expert_reward_1.mean().item()
                                      + expert_reward_2.mean().item()),
        "policy_reward_mean": 0.5 * (r_imp_1[n_expert:].mean().item()
                                      + r_imp_2[n_expert:].mean().item()),
        "mean_q": 0.5 * (q1_all.mean().item() + q2_all.mean().item()),
    }


# ── IQ-Learn actor update (standard SAC objective) ───────────────────────────

def iqlearn_update_actor(
    actor: ContinuousActor,
    q1: IQLearnQNetwork,
    q2: IQLearnQNetwork,
    log_alpha: torch.Tensor,
    target_entropy: float,
    actor_opt: torch.optim.Optimizer,
    alpha_opt: torch.optim.Optimizer,
    buffer: IQLearnReplayBuffer,
    batch_size: int,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """SAC actor update: maximise Q − α log π."""
    alpha = log_alpha.exp().item()
    states, _, _, _, _ = buffer.sample(batch_size, device)

    a_pi, lp_pi = _reparameterize_actor(actor, states)
    q_pi = torch.min(q1(states, a_pi), q2(states, a_pi))
    actor_loss = (alpha * lp_pi.unsqueeze(-1) - q_pi).mean()

    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    actor_opt.step()

    # ── Update alpha ──
    alpha_loss = -(log_alpha * (lp_pi.detach() + target_entropy)).mean()
    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()

    return {
        "actor_loss": actor_loss.item(),
        "alpha": log_alpha.exp().item(),
        "mean_log_prob": lp_pi.mean().item(),
        "mean_q_pi": q_pi.mean().item(),
    }


# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout_iqlearn_episode(
    env: PCH,
    actor: ContinuousActor,
    iqlearn_buffer: IQLearnReplayBuffer,
    encode: Callable,
    max_steps: int,
    device: torch.device,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
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
        iqlearn_buffer.push_policy(
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

def evaluate_iqlearn_policy(
    env: PCH,
    actor: ContinuousActor,
    encode: Callable,
    max_steps: int,
    device: torch.device,
    num_episodes: int = 10,
    seed: Optional[int] = None,
) -> float:
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


# ── Main training loop ───────────────────────────────────────────────────────

def train_iqlearn(
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
    buffer_capacity: int = 1_000_000,
    expert_capacity_ratio: float = 0.5,
    num_v_samples: int = 10,
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
    q1 = IQLearnQNetwork(state_dim, action_dim, hidden_dim).to(device)
    q2 = IQLearnQNetwork(state_dim, action_dim, hidden_dim).to(device)
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

    # ── Buffer ──
    buf = IQLearnReplayBuffer(buffer_capacity, expert_capacity_ratio)
    iq_init_expert_buffer(expert_records, encode, buf, device)

    # ── Training ──
    timesteps = 0
    episode = 0
    logs: Dict[str, list] = {
        "episode_returns": [], "episode_lengths": [],
        "eval_returns": [], "eval_timesteps": [],
        "critic_loss": [], "actor_loss": [], "alpha": [],
        "expert_reward_mean": [], "policy_reward_mean": [],
    }

    print(f"IQ-Learn training: state_dim={state_dim}, action_dim={action_dim}, "
          f"action=[{action_low}, {action_high}]")

    while timesteps < total_timesteps:
        ep_data = rollout_iqlearn_episode(
            env, actor, buf, encode, max_episode_steps, device,
            deterministic=False, seed=(seed + episode) if seed else None,
        )
        timesteps += ep_data["episode_length"]
        episode += 1
        logs["episode_returns"].append(ep_data["episode_return"])
        logs["episode_lengths"].append(ep_data["episode_length"])

        if episode == 1:
            assert len(buf.policy_buffer) > 0, (
                "BUG: policy buffer empty after rollout!")

        if timesteps > start_steps and len(buf.policy_buffer) >= batch_size // 2:
            alpha_val = log_alpha.exp().item()
            for _ in range(ep_data["episode_length"] * updates_per_step):
                alpha_val = log_alpha.exp().item()

                c_metrics = iqlearn_update_critic(
                    q1, q2, tq1, tq2, actor, alpha_val, buf,
                    batch_size, gamma, q1_opt, q2_opt, device,
                    num_v_samples, max_grad_norm,
                )
                a_metrics = iqlearn_update_actor(
                    actor, q1, q2, log_alpha, target_entropy,
                    actor_opt, alpha_opt, buf, batch_size, device,
                    max_grad_norm,
                )

                soft_update(q1, tq1, tau)
                soft_update(q2, tq2, tau)

                logs["critic_loss"].append(c_metrics["critic_loss"])
                logs["actor_loss"].append(a_metrics["actor_loss"])
                logs["alpha"].append(a_metrics["alpha"])
                logs["expert_reward_mean"].append(c_metrics["expert_reward_mean"])
                logs["policy_reward_mean"].append(c_metrics["policy_reward_mean"])

        if timesteps % eval_freq < ep_data["episode_length"] or timesteps >= total_timesteps:
            eval_ret = evaluate_iqlearn_policy(
                env, actor, encode, max_episode_steps,
                device, eval_episodes, seed=42,
            )
            logs["eval_returns"].append(eval_ret)
            logs["eval_timesteps"].append(timesteps)
            alpha_val = log_alpha.exp().item()
            er = np.mean(logs["expert_reward_mean"][-100:]) if logs["expert_reward_mean"] else 0.0
            pr = np.mean(logs["policy_reward_mean"][-100:]) if logs["policy_reward_mean"] else 0.0
            print(
                f"[IQ-Learn ep {episode}] ts={timesteps}, "
                f"eval={eval_ret:.2f}, "
                f"train={np.mean(logs['episode_returns'][-10:]):.2f}, "
                f"alpha={alpha_val:.4f}, "
                f"r_expert={er:.3f}, r_policy={pr:.3f}"
            )
            if log_callback:
                log_callback({"timesteps": timesteps, "episode": episode,
                              "eval_return": eval_ret})

    print("IQ-Learn training complete.")
    return actor, logs


__all__ = [
    "train_iqlearn",
    "IQLearnReplayBuffer",
    "ReplayBuffer",
    "evaluate_iqlearn_policy",
    "rollout_iqlearn_episode",
    "iq_init_expert_buffer",
    "initialize_expert_buffer",
    "iqlearn_update_critic",
    "iqlearn_update_actor",
    "soft_update",
]
