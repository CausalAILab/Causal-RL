"""Task 7 speed / memory benchmark built from the SCIL notebook training loops.

Nothing here trains a usable policy: stand-in "expert" data are two untrained-policy episodes, and only
wall-clock time and memory are measured.

usage: python bench_task7.py ENV ALGO ENC [--device cpu|cuda|mps] [--updates N]
  ENV  antmed | antlarge | hummed | humlarge
  ALGO sqil | iq | gail | eval
  ENC  causal | naive
Parallel runs: set BENCH_BARRIER_DIR and BENCH_BARRIER_N so that all processes start timing together.
Prints one JSON line.
"""
import argparse, copy, json, os, platform, random, resource, sys, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("env"); p.add_argument("algo"); p.add_argument("enc")
p.add_argument("--device", default="cpu"); p.add_argument("--updates", type=int, default=0)
args = p.parse_args()
ENV, ALGO, ENC = args.env, args.algo, args.enc
torch.set_num_threads(1)
dev = torch.device(args.device)

from causal_gym import AntMazePCH, HumanoidMazePCH
from causal_rl.algo.imitation.imitate import *  # parse_graph, find_sequential_pi_backdoor, trim_Z_sets, collectors
from causal_rl.algo.imitation.gail.core_net import ContinuousActor, Critic, Discriminator
from causal_rl.algo.imitation.gail.causal_gail import build_windowed_z_encoder, make_gail_policy, make_expert_batch, one_training_round
from causal_rl.algo.imitation.sqil.core_net import SQILQNetwork
from causal_rl.algo.imitation.sqil.causal_sqil import SQILReplayBuffer, initialize_expert_buffer, rollout_sqil_episode, sac_update, soft_update
from causal_rl.algo.imitation.iqlearn.core_net import IQLearnQNetwork
from causal_rl.algo.imitation.iqlearn.causal_iqlearn import (IQLearnReplayBuffer, iq_init_expert_buffer, iqlearn_update_critic,
                                                           iqlearn_update_actor, rollout_iqlearn_episode)

seed = 0
random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
ANT = ENV.startswith("ant")
if ANT:
    env_id = f"antmaze-{'medium' if ENV == 'antmed' else 'large'}-navigate-singletask-task1-v0"
    H, k, hid = 1000, 10, {"O"}; utd, nv = 0.25, 16; ppo_epochs, n_ep, rounds = 4, 20, 500
    mk = lambda **kw: AntMazePCH(env_id=env_id, **kw)
    dims = {"P": 3, "A": 8, "L": 3, "T": 3, "J": 8, "W": 2, "X": 8}
else:
    env_id = f"humanoidmaze-{'medium' if ENV == 'hummed' else 'large'}-navigate-singletask-task1-v0"
    H, k, hid = 2000, 2, {"C"}; utd, nv = 0.5, 1; ppo_epochs, n_ep, rounds = 2, 10, 200
    mk = lambda **kw: HumanoidMazePCH(env_id=env_id, **kw)
    dims = {"P": 2, "A": 21, "H": 1, "E": 12, "V": 3, "J": 27, "W": 2, "X": 21}


def sync():
    if dev.type == "cuda": torch.cuda.synchronize()
    elif dev.type == "mps": torch.mps.synchronize()


t0 = time.perf_counter()
train_env = mk(num_steps=H, expert_mode=True, custom_hidden=hid, seed=seed)
eval_env = mk(num_steps=H, expert_mode=False, seed=seed)
small = k + 1
G = parse_graph(mk(num_steps=small, seed=seed).get_graph)
obs_prefix = train_env.env.observed_unobserved_vars[0]
if ENC == "causal":
    Z = find_sequential_pi_backdoor(G, {f"X{t}" for t in range(small)}, f"Y{small}", obs_prefix)
    base = Z[f"X{small - 1}"]
    for i in range(small, H):
        Z[f"X{i}"] = {f"{v[0]}{int(v[1:]) + i - k}" for v in base}
else:  # full public history over the lookback window (SCIL n* notebooks)
    Z = {}
    for i in range(H):
        c = set()
        for j in range(i - k, i + 1):
            if j >= 0: c.update({f"{o}{j}" for o in set(obs_prefix) - {"X"}})
        for j in range(i - k, i):
            if j >= 0: c.add(f"X{j}")
        Z[f"X{i}"] = c
encode, z_dim, _ = build_windowed_z_encoder(trim_Z_sets(Z, lookback=k), dims=dims, lookback=k)
setup_s = time.perf_counter() - t0

A = train_env.env.action_space.shape[0]
lo, hi = float(train_env.env.action_space.low.min()), float(train_env.env.action_space.high.max())
actor = ContinuousActor(num_inputs=z_dim, num_outputs=A, hidden_size=256, std=0.0, action_low=lo, action_high=hi,
                        num_blocks=3, dropout=0.05, layernorm=True).to(dev)
records = collect_imitator_trajectories(env=mk(num_steps=H, expert_mode=True),
                                        policies=make_shared_policy_dict(make_gail_policy(actor, encode, device=dev, deterministic=False)),
                                        num_episodes=2, max_steps=H, show_progress=False, seed=1)
out = {"env": ENV, "algo": ALGO, "enc": ENC, "device": str(dev), "z_dim": int(z_dim), "setup_s": round(setup_s, 1),
       "parallel": int(os.environ.get("BENCH_BARRIER_N", "1")),
       "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "cpu": platform.processor() or platform.machine()}


def barrier():
    d, n = os.environ.get("BENCH_BARRIER_DIR"), int(os.environ.get("BENCH_BARRIER_N", "1"))
    if not d or n <= 1: return
    open(os.path.join(d, f"ready_{os.getpid()}"), "w").close()
    while sum(f.startswith("ready_") for f in os.listdir(d)) < n:
        time.sleep(0.05)


def off_policy(kind):
    Q = SQILQNetwork if kind == "sqil" else IQLearnQNetwork
    q1, q2 = (Q(z_dim, A, 256, num_blocks=3, dropout=0.05, layernorm=True).to(dev) for _ in range(2))
    tq1, tq2 = copy.deepcopy(q1), copy.deepcopy(q2)
    for prm in list(tq1.parameters()) + list(tq2.parameters()): prm.requires_grad = False
    ao, o1, o2 = (torch.optim.Adam(m.parameters(), lr=3e-4) for m in (actor, q1, q2))
    s1 = torch.optim.lr_scheduler.CosineAnnealingLR(o1, T_max=500000); s2 = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=500000)
    la = torch.zeros(1, requires_grad=True, device=dev); lao = torch.optim.Adam([la], lr=3e-4)
    if kind == "sqil":
        buf = SQILReplayBuffer(1_000_000, 0.5); initialize_expert_buffer(records, encode, buf, dev); roll = rollout_sqil_episode
    else:
        buf = IQLearnReplayBuffer(1_000_000, 0.5); iq_init_expert_buffer(records, encode, buf, dev); roll = rollout_iqlearn_episode
    barrier()
    sync(); t = time.perf_counter(); steps = 0
    for ep in range(2):
        steps += roll(train_env, actor, buf, encode, H, dev, deterministic=False, seed=100 + ep)["episode_length"]
    sync(); r = (time.perf_counter() - t) / steps
    for i in range(10):  # warm-up (CUDA kernels, allocator); needs policy data in the buffer
        step(kind, q1, q2, tq1, tq2, ao, o1, o2, s1, s2, la, lao, buf)
    n = args.updates or (150 if (kind == "iq" and nv > 1 and dev.type == "cpu") else 300)
    sync(); t = time.perf_counter()
    for _ in range(n):
        step(kind, q1, q2, tq1, tq2, ao, o1, o2, s1, s2, la, lao, buf)
    sync(); u = (time.perf_counter() - t) / n
    per = r + utd * u
    out.update(rollout_ms_per_step=round(r * 1e3, 3), update_ms=round(u * 1e3, 3), ms_per_interaction=round(per * 1e3, 3),
               hours_2M=round(2e6 * per / 3600, 2))


def step(kind, q1, q2, tq1, tq2, ao, o1, o2, s1, s2, la, lao, buf):
    if kind == "sqil":
        sac_update(q1, q2, tq1, tq2, actor, la, -float(A), o1, o2, ao, lao, buf, 256, 0.99, dev, 1.0)
    else:
        iqlearn_update_critic(q1, q2, tq1, tq2, actor, la.exp().item(), buf, 256, 0.99, o1, o2, dev, 1.0, num_v_samples=nv)
        iqlearn_update_actor(actor, q1, q2, la, -float(A), ao, lao, buf, 256, dev, 1.0)
    soft_update(q1, tq1, 0.005); soft_update(q2, tq2, 0.005); s1.step(); s2.step()
    with torch.no_grad(): la.clamp_(min=np.log(.001), max=np.log(.1))


if ALGO in ("sqil", "iq"):
    off_policy(ALGO)
elif ALGO == "gail":
    critic = Critic(num_inputs=z_dim, hidden_size=256).to(dev)
    disc = Discriminator(num_inputs=z_dim + A, hidden_size=256, dropout=0.2).to(dev)
    ao, co, do_ = (torch.optim.Adam(m.parameters(), lr=lr) for m, lr in ((actor, 1e-4), (critic, 3e-4), (disc, 3e-4)))
    _, _, X_e = make_expert_batch(records, encode)
    X_e = X_e.to(dev)
    kw = dict(env=train_env, actor=actor, critic=critic, discriminator=disc, actor_optim=ao, critic_optim=co, discriminator_optim=do_,
              encode=encode, X_e=X_e, expert_records=None, gamma=.99, gae_lambda=.95, ppo_clip=.2, epochs=ppo_epochs,
              minibatch_size=1024, entropy_coeff=1e-2, value_coeff=.5, max_grad_norm=.5, normalize_adv=True, loss_type="bce",
              gp_lambda=5.0, d_updates=2, d_minibatch_size=1024, use_gp=True, instance_noise_std=0.0, label_smoothing=0.0, max_steps=H)
    barrier()
    eps = 2
    sync(); t = time.perf_counter()
    one_training_round(num_episodes=eps, seed=7, **kw)
    sync(); dt = time.perf_counter() - t
    steps = eps * H  # untrained policy almost never reaches the goal: full-length episodes (worst case)
    out.update(ms_per_interaction=round(dt / steps * 1e3, 3), hours_full_schedule_worst=round(rounds * n_ep * H * dt / steps / 3600, 2))
elif ALGO == "eval":
    det = make_shared_policy_dict(make_gail_policy(actor, encode, device=dev, deterministic=True))
    barrier()
    t = time.perf_counter()
    recs = collect_imitator_trajectories(env=eval_env, policies=det, num_episodes=2, max_steps=H, hidden_dims=hid,
                                         show_progress=False, seed=90210)
    out.update(ms_per_eval_step=round((time.perf_counter() - t) / len(recs) * 1e3, 3))
else:
    sys.exit(f"unknown algo {ALGO}")

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
out["peak_rss_MB"] = round(rss / 2**20 if sys.platform == "darwin" else rss / 1024)
if dev.type == "cuda":
    out["peak_gpu_MB"] = round(torch.cuda.max_memory_allocated() / 2**20)
print(json.dumps(out), flush=True)
