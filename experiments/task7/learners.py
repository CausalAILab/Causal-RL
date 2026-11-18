"""Task 7 learner training loops: BC, GAIL, SQIL, IQ-Learn (vanilla / causal / wmask inputs).

Each loop reproduces the SCIL paper notebooks (Causal-RL 9c8741c, examples/experiments/{c,n}{bc,gail,sqil,
iqlearn}_*.ipynb) with the B0 launch-check changes, all recorded in the protocol:
  - no best-checkpoint selection and no in-training evaluation on the training env (seed 42, dropout on);
    evaluation is common.evaluate on separate environments, every `eval_every` interactions (c=0, diagnostic);
  - SQIL/IQ-Learn stop at exactly `budget` interactions; an episode is cut at the 1M secondary endpoint and at
    the budget, and a cut transition is not a terminal (bootstraps);
  - the expert half of the replay holds every training transition (enlarged beyond 500k when needed);
  - BC trains on the 800 training episodes and early-stops on the 200 validation episodes (episode split).
Each function returns (policy_modules, act_fn, log) and calls
`on_checkpoint(tag, interactions, updates_or_rounds, modules, act)` at the
secondary endpoint and at the end so the runner can save weights and run the evaluation banks.
"""
from __future__ import annotations

import copy
import random
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from causal_rl.algo.imitation.data import WindowedExpertDataset
from causal_rl.algo.imitation.gail.causal_gail import make_expert_batch, one_training_round
from causal_rl.algo.imitation.gail.core_net import ContinuousActor, Critic, Discriminator
from causal_rl.algo.imitation.imitate import ContinuousPolicyNN, make_window_spec, shared_policy_fn_long_horizon
from causal_rl.algo.imitation.iqlearn import causal_iqlearn as IQ
from causal_rl.algo.imitation.iqlearn.core_net import IQLearnQNetwork
from causal_rl.algo.imitation.sqil import causal_sqil as SQ
from causal_rl.algo.imitation.sqil.core_net import SQILQNetwork

DEV = torch.device("cpu")

# SCIL notebook hyperparameters (paper Table 4 / notebooks at 9c8741c)
OFFPOLICY = dict(batch_size=256, gamma=0.99, tau=0.005, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, hidden=256,
                 buffer_capacity=1_000_000, expert_ratio=0.5, start_steps=5_000, max_grad_norm=1.0, blocks=3,
                 dropout=0.05, alpha_clamp=(0.001, 0.1))
GAIL = dict(gamma=0.99, gae_lambda=0.95, ppo_clip=0.2, minibatch=1024, entropy_coeff=1e-2, value_coeff=0.5,
            max_grad_norm=0.5, normalize_adv=True, loss_type="bce", gp_lambda=5.0, d_updates=2, d_minibatch=1024,
            hidden=256, actor_lr=1e-4, critic_lr=3e-4, disc_lr=3e-4, blocks=3, dropout=0.05, disc_dropout=0.2,
            disc_step=100, disc_gamma=0.5)
BC = dict(hidden=256, lr=3e-4, batch_size=2048, blocks=4, dropout=0.0)


def seed_everything(s: int):
    random.seed(s)
    np.random.seed(s % 2**32)
    torch.manual_seed(s)


def make_actor(dim: int, A: int, low: float, high: float, p=OFFPOLICY) -> ContinuousActor:
    return ContinuousActor(num_inputs=dim, num_outputs=A, hidden_size=p["hidden"], std=0.0, action_low=low,
                           action_high=high, num_blocks=p["blocks"], dropout=p["dropout"], layernorm=True).to(DEV)


# ── SQIL / IQ-Learn ─────────────────────────────────────────────────────────────────────────────

def train_offpolicy(kind: str, name: str, encoder: C.Encoder, demos: C.Demos, train_idx, run_seed: int, *,
                    budget: int = 2_000_000, secondary: Optional[int] = 1_000_000, eval_every: int = 200_000,
                    eval_episodes: int = 20, on_checkpoint: Callable = lambda *a: None,
                    on_eval: Callable = lambda *a: None, p: Dict[str, Any] = OFFPOLICY,
                    schedule_budget: Optional[int] = None):
    """schedule_budget: interactions the cosine LR schedule is sized for (default `budget`). A shortened run with
    schedule_budget=2M follows a full 2M run exactly up to its own budget."""
    assert kind in ("sqil", "iq")
    s = C.spec(name)
    seed_everything(C.seed_of("torch", s["index"], run_seed, 0))
    env = C.make_env(name, "train", seed=C.seed_of("train_env", s["index"], run_seed, 0))
    A = s["dims"]["X"]
    low, high = float(env.env.action_space.low.min()), float(env.env.action_space.high.max())
    actor = make_actor(encoder.dim, A, low, high, p)
    Q = SQILQNetwork if kind == "sqil" else IQLearnQNetwork
    q1, q2 = (Q(encoder.dim, A, p["hidden"], num_blocks=p["blocks"], dropout=p["dropout"], layernorm=True).to(DEV)
              for _ in range(2))
    tq1, tq2 = copy.deepcopy(q1), copy.deepcopy(q2)
    for prm in list(tq1.parameters()) + list(tq2.parameters()):
        prm.requires_grad = False
    actor_opt = torch.optim.Adam(actor.parameters(), lr=p["actor_lr"])
    q1_opt = torch.optim.Adam(q1.parameters(), lr=p["critic_lr"])
    q2_opt = torch.optim.Adam(q2.parameters(), lr=p["critic_lr"])
    t_max = int((schedule_budget or budget) * s["utd"])  # SCIL: estimated_total_updates = total_timesteps * utd_ratio
    q1_sch = torch.optim.lr_scheduler.CosineAnnealingLR(q1_opt, T_max=t_max)
    q2_sch = torch.optim.lr_scheduler.CosineAnnealingLR(q2_opt, T_max=t_max)
    log_alpha = torch.zeros(1, requires_grad=True, device=DEV)
    alpha_opt = torch.optim.Adam([log_alpha], lr=p["alpha_lr"])
    target_entropy = -float(A)

    mod = SQ if kind == "sqil" else IQ
    Buffer = SQ.SQILReplayBuffer if kind == "sqil" else IQ.IQLearnReplayBuffer
    buf = Buffer(p["buffer_capacity"], p["expert_ratio"])
    n_expert = demos.n_transitions(train_idx)
    if n_expert > len_capacity(buf.expert_buffer):  # launch check 10: never overwrite expert data
        buf.expert_buffer = mod.ReplayBuffer(n_expert)
    init = SQ.initialize_expert_buffer if kind == "sqil" else IQ.iq_init_expert_buffer
    init(demos.records(train_idx), encoder, buf, DEV)
    assert len(buf.expert_buffer) == n_expert
    rollout = SQ.rollout_sqil_episode if kind == "sqil" else IQ.rollout_iqlearn_episode
    lo_clip, hi_clip = np.log(p["alpha_clamp"][0]), np.log(p["alpha_clamp"][1])

    def update():
        if kind == "sqil":
            m = SQ.sac_update(q1, q2, tq1, tq2, actor, log_alpha, target_entropy, q1_opt, q2_opt, actor_opt,
                              alpha_opt, buf, p["batch_size"], p["gamma"], DEV, p["max_grad_norm"])
        else:
            m = IQ.iqlearn_update_critic(q1, q2, tq1, tq2, actor, log_alpha.exp().item(), buf, p["batch_size"],
                                         p["gamma"], q1_opt, q2_opt, DEV, p["max_grad_norm"],
                                         num_v_samples=s["num_v_samples"])
            m.update(IQ.iqlearn_update_actor(actor, q1, q2, log_alpha, target_entropy, actor_opt, alpha_opt, buf,
                                             p["batch_size"], DEV, p["max_grad_norm"]))
        IQ.soft_update(q1, tq1, p["tau"])
        IQ.soft_update(q2, tq2, p["tau"])
        q1_sch.step()
        q2_sch.step()
        with torch.no_grad():
            log_alpha.clamp_(min=lo_clip, max=hi_clip)
        return m

    act = C.actor_policy(actor, encoder, DEV)
    log: Dict[str, List] = {"episodes": [], "evals": [], "updates": 0}
    ts, ep, next_eval, t0 = 0, 0, eval_every, time.time()
    stops = sorted({x for x in (secondary, budget) if x})
    while ts < budget:
        stop = next(x for x in stops if x > ts)
        cap = min(s["H"], stop - ts)
        d = rollout(env, actor, buf, encoder, cap, DEV, deterministic=False,
                    seed=C.seed_of("train_env", s["index"], run_seed, ep + 1))
        ts += d["episode_length"]
        ep += 1
        n_upd = 0
        if ts > p["start_steps"] and len(buf.policy_buffer) >= p["batch_size"] // 2:
            n_upd = max(1, int(d["episode_length"] * s["utd"]))
            for _ in range(n_upd):
                m = update()
            log["updates"] += n_upd
        log["episodes"].append([ts, d["episode_length"], float(d["episode_return"]), bool(d["terminated"]),
                                n_upd, float(log_alpha.exp().item())])
        print(f"progress done={ts} total={budget} unit=interactions updates={log['updates']} "
              f"elapsed={time.time() - t0:.0f}", flush=True)
        while ts >= next_eval and next_eval <= budget:
            r = C.evaluate(name, act, [C.seed_of("checkpoint_eval", s["index"], run_seed, next_eval // eval_every * 1000 + i)
                                       for i in range(eval_episodes)], c=0.0, modules=[actor])
            r = {k: v for k, v in r.items() if k != "per_episode"}
            r.update(interactions=ts, target=next_eval, updates=log["updates"], seconds=round(time.time() - t0, 1))
            log["evals"].append(r)
            on_eval(r)
            next_eval += eval_every
        if ts in stops:
            on_checkpoint("final" if ts == budget else "secondary", ts, log["updates"], [actor, q1, q2], act)
    log.update(interactions=ts, seconds=round(time.time() - t0, 1))
    return [actor, q1, q2], act, log


def len_capacity(rb) -> int:
    return rb.capacity


# ── GAIL ────────────────────────────────────────────────────────────────────────────────────────

def train_gail(name: str, encoder: C.Encoder, demos: C.Demos, train_idx, run_seed: int, *,
               rounds: Optional[int] = None, episodes_per_round: Optional[int] = None,
               secondary_interactions: Optional[int] = 2_000_000, eval_every: int = 200_000, eval_episodes: int = 20,
               on_checkpoint: Callable = lambda *a: None, on_eval: Callable = lambda *a: None, p=GAIL):
    s = C.spec(name)
    rounds = rounds or s["gail_rounds"]
    episodes_per_round = episodes_per_round or s["gail_episodes"]
    seed_everything(C.seed_of("torch", s["index"], run_seed, 0))
    env = C.make_env(name, "train", seed=C.seed_of("train_env", s["index"], run_seed, 0))
    A = s["dims"]["X"]
    low, high = float(env.env.action_space.low.min()), float(env.env.action_space.high.max())
    actor = make_actor(encoder.dim, A, low, high, p)
    critic = Critic(num_inputs=encoder.dim, hidden_size=p["hidden"]).to(DEV)
    disc = Discriminator(num_inputs=encoder.dim + A, hidden_size=p["hidden"], dropout=p["disc_dropout"]).to(DEV)
    a_opt = torch.optim.Adam(actor.parameters(), lr=p["actor_lr"])
    c_opt = torch.optim.Adam(critic.parameters(), lr=p["critic_lr"])
    d_opt = torch.optim.Adam(disc.parameters(), lr=p["disc_lr"])
    d_sch = torch.optim.lr_scheduler.StepLR(d_opt, step_size=p["disc_step"], gamma=p["disc_gamma"])
    _, _, X_e = make_expert_batch(demos.records(train_idx), encoder)
    act = C.actor_policy(actor, encoder, DEV)
    log: Dict[str, List] = {"rounds": [], "evals": []}
    ts, next_eval, secondary_done, t0 = 0, eval_every, secondary_interactions is None, time.time()
    for it in range(1, rounds + 1):
        st = one_training_round(
            env=env, actor=actor, critic=critic, discriminator=disc, actor_optim=a_opt, critic_optim=c_opt,
            discriminator_optim=d_opt, encode=encoder, X_e=X_e, expert_records=None, gamma=p["gamma"],
            gae_lambda=p["gae_lambda"], ppo_clip=p["ppo_clip"], epochs=s["ppo_epochs"], minibatch_size=p["minibatch"],
            entropy_coeff=p["entropy_coeff"], value_coeff=p["value_coeff"], max_grad_norm=p["max_grad_norm"],
            normalize_adv=p["normalize_adv"], loss_type=p["loss_type"], gp_lambda=p["gp_lambda"],
            d_updates=p["d_updates"], d_minibatch_size=p["d_minibatch"], use_gp=True, instance_noise_std=0.0,
            label_smoothing=0.0, max_steps=s["H"], num_episodes=episodes_per_round,
            seed=C.seed_of("train_env", s["index"], run_seed, it * 1000))
        d_sch.step()
        ts += st["n_steps"]
        log["rounds"].append([it, ts, st["avg_env_return"], st["D_loss"], st["ppo_actor_loss"], st["ppo_approx_kl"]])
        print(f"progress done={it} total={rounds} unit=rounds interactions={ts} elapsed={time.time() - t0:.0f}",
              flush=True)
        while ts >= next_eval:
            r = C.evaluate(name, act, [C.seed_of("checkpoint_eval", s["index"], run_seed, next_eval // eval_every * 1000 + i)
                                       for i in range(eval_episodes)], c=0.0, modules=[actor, critic, disc])
            r = {k: v for k, v in r.items() if k != "per_episode"}
            r.update(interactions=ts, target=next_eval, round=it, seconds=round(time.time() - t0, 1))
            log["evals"].append(r)
            on_eval(r)
            next_eval += eval_every
        if not secondary_done and ts >= secondary_interactions:
            secondary_done = True
            on_checkpoint("secondary", ts, it, [actor, critic, disc], act)
    on_checkpoint("final", ts, rounds, [actor, critic, disc], act)
    log.update(interactions=ts, seconds=round(time.time() - t0, 1))
    return [actor, critic, disc], act, log


# ── BC ──────────────────────────────────────────────────────────────────────────────────────────

def train_bc(name: str, encoding: str, demos: C.Demos, train_idx, val_idx, run_seed: int, *,
             epochs: Optional[int] = None, patience: Optional[int] = None, p=BC,
             on_checkpoint: Callable = lambda *a: None):
    """SCIL's long-horizon BC (train_single_policy_long_horizon) with an episode-level train/validation split.
    Input = window of lags 1..k of every public variable plus a mask of the Z-set slots (SCIL design;
    no current-time slot - kept as SCIL by decision on 2026-11-09)."""
    s = C.spec(name)
    seed_everything(C.seed_of("torch", s["index"], run_seed, 0))
    Z = C.z_sets(name, "vanilla" if encoding == "wmask" else encoding)
    include = [v for v in demos.public] + ["X"]
    dims = {v: s["dims"][v] for v in include}
    slots = make_window_spec(include_vars=include, dims=dims, lookback=s["k"])
    if encoding == "wmask":
        Z = {k: {v for v in vs if not v.startswith("W")} for k, vs in Z.items()}
    tr = WindowedExpertDataset(demos.records(train_idx), Z, slots, action_var="action", continuous=True)
    va = WindowedExpertDataset(demos.records(val_idx), Z, slots, action_var="action", continuous=True)
    g = torch.Generator().manual_seed(C.seed_of("torch", s["index"], run_seed, 1))
    tl = torch.utils.data.DataLoader(tr, batch_size=p["batch_size"], shuffle=True, generator=g)
    vl = torch.utils.data.DataLoader(va, batch_size=p["batch_size"], shuffle=False)
    env = C.make_env(name, "train", num_steps=10, seed=0)
    model = ContinuousPolicyNN(input_dim=tr.x.shape[1], action_dim=s["dims"]["X"], hidden_dim=p["hidden"],
                               num_blocks=p["blocks"], dropout=p["dropout"], layernorm=True, final_tanh=True,
                               action_bounds=(env.env.action_space.low, env.env.action_space.high)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    loss_fn = torch.nn.HuberLoss()
    epochs, patience = epochs or s["bc_epochs"], patience or s["bc_patience"]
    best, best_state, bad, hist, t0 = float("inf"), None, 0, [], time.time()
    for epoch in range(epochs):
        model.train()
        tot = n = 0
        for xb, yb in tl:
            loss = loss_fn(model(xb), yb.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            vl_loss = sum(loss_fn(model(xb), yb.float()).item() * len(xb) for xb, yb in vl) / len(va)
        hist.append([epoch + 1, tot / n, vl_loss])
        print(f"progress done={epoch + 1} total={epochs} unit=epochs val_loss={vl_loss:.5f} "
              f"elapsed={time.time() - t0:.0f}", flush=True)
        if vl_loss + 1e-6 < best:  # SCIL's improvement rule
            best, bad = vl_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    pi = shared_policy_fn_long_horizon(model, slots, Z, continuous=True, device=DEV)
    on_checkpoint("final", 0, len(hist), [model], lambda o, t: pi(o))
    log = {"epochs": hist, "best_val": best, "stopped_epoch": len(hist), "train_samples": len(tr),
           "val_samples": len(va), "seconds": round(time.time() - t0, 1)}
    return [model], (lambda o, t: pi(o)), log
