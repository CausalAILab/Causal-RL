"""Task 7 shared pieces: environment specs, inputs (Z sets / encoders), compact demonstrations,
expert loading and collection, isolated evaluation, and seed streams.

Conventions
- An episode is stored as arrays: every state variable has T+1 rows (s_0..s_T), the action X has T rows.
  A step-t training sample uses only rows <= t of state variables and rows < t of X, which is exactly
  what the SCIL notebooks' per-step history records exposed to the encoders.
- Learner environments are built with the B0 launch-check flags of Causal-Gymnasium:
  seed_base_reset=True, success_after_step=True, playback=False.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from causal_gym import AntMazePCH, HumanoidMazePCH
from causal_rl.algo.imitation.imitate import (
    ContinuousPolicyNN, find_sequential_pi_backdoor, parse_graph, shared_policy_fn_long_horizon, trim_Z_sets,
)
from causal_rl.algo.imitation.gail.causal_gail import build_windowed_z_encoder

# ── Environment specs (SCIL notebooks; Hum-Large radius 15 as in SCIL, decided 2026-11-09) ─────────

ANT_DIMS = {"P": 3, "O": 4, "A": 8, "L": 3, "T": 3, "J": 8, "W": 2, "X": 8}
HUM_DIMS = {"P": 2, "A": 21, "H": 1, "E": 12, "V": 3, "C": 3, "J": 27, "W": 2, "X": 21}

SPECS: Dict[str, Dict[str, Any]] = {
    "antmed": dict(env_id="antmaze-medium-navigate-singletask-task1-v0", family="ant", H=1000, k=10,
                   hidden="O", radius=5.0, utd=0.25, num_v_samples=16, ppo_epochs=4, gail_rounds=500,
                   gail_episodes=20, bc_epochs=100, bc_patience=15),
    "antlarge": dict(env_id="antmaze-large-navigate-singletask-task1-v0", family="ant", H=1000, k=10,
                     hidden="O", radius=5.0, utd=0.25, num_v_samples=16, ppo_epochs=4, gail_rounds=500,
                     gail_episodes=20, bc_epochs=100, bc_patience=15),
    "hummed": dict(env_id="humanoidmaze-medium-navigate-singletask-task1-v0", family="hum", H=2000, k=2,
                   hidden="C", radius=10.0, utd=0.5, num_v_samples=1, ppo_epochs=2, gail_rounds=200,
                   gail_episodes=10, bc_epochs=100, bc_patience=30),
    "humlarge": dict(env_id="humanoidmaze-large-navigate-singletask-task1-v0", family="hum", H=2000, k=2,
                     hidden="C", radius=15.0, utd=0.5, num_v_samples=1, ppo_epochs=2, gail_rounds=200,
                     gail_episodes=10, bc_epochs=200, bc_patience=15),
}
ENV_NAMES = tuple(SPECS)
ENCODINGS = ("causal", "vanilla", "wmask")


def spec(name: str) -> Dict[str, Any]:
    s = dict(SPECS[name])
    s["name"] = name
    s["dims"] = dict(ANT_DIMS if s["family"] == "ant" else HUM_DIMS)
    s["state_vars"] = [v for v in s["dims"] if v != "X"]
    s["index"] = ENV_NAMES.index(name)
    return s


def make_env(name: str, role: str, *, seed: Optional[int] = None, c: float = 1.0, num_steps: Optional[int] = None,
             legacy: bool = False, flags: Optional[Dict[str, bool]] = None):
    """role: 'expert'  - everything visible, W unrotated (expert development and collection);
             'train'   - hidden O/C, W unrotated (learner training, = c 0; SCIL's train_env);
             'eval'    - hidden O/C, W rotated by c*pi (c=0 in-distribution, c=1 runtime)."""
    s = spec(name)
    cls = AntMazePCH if s["family"] == "ant" else HumanoidMazePCH
    kw = dict(env_id=s["env_id"], num_steps=num_steps or s["H"], success_radius=s["radius"], seed=seed,
              seed_base_reset=not legacy, success_after_step=not legacy, playback=False)  # legacy: SCIL's env behavior
    kw.update(flags or {})
    if role == "expert":
        return cls(expert_mode=True, **kw)
    if role == "train":
        return cls(expert_mode=True, custom_hidden={s["hidden"]}, **kw)
    if role == "eval":
        return cls(expert_mode=False, confound_strength=float(c), **kw)
    raise ValueError(role)


# ── Inputs: Z sets and encoders ─────────────────────────────────────────────────────────────────

def z_sets(name: str, encoding: str) -> Dict[str, set]:
    """Per-action conditioning sets for t = 0..H-1, built exactly as the SCIL notebooks do.
    causal: sequential pi-backdoor on a (k+1)-step graph, slid forward; vanilla/wmask: all public
    variables over the lookback window plus past actions."""
    s = spec(name)
    H, k = s["H"], s["k"]
    small = k + 1
    env = make_env(name, "train", num_steps=50, seed=0)
    obs_prefix = env.env.observed_unobserved_vars[0]
    if encoding == "causal":
        G = parse_graph(make_env(name, "eval", num_steps=small, seed=0).get_graph)
        Z = find_sequential_pi_backdoor(G, {f"X{t}" for t in range(small)}, f"Y{small}", obs_prefix)
        base = Z[f"X{small - 1}"]
        for i in range(small, H):
            Z[f"X{i}"] = {f"{v[0]}{int(v[1:]) + i - k}" for v in base}
    elif encoding in ("vanilla", "wmask"):
        Z = {}
        public = set(obs_prefix) - {"X"}
        for i in range(H):
            cond = set()
            for j in range(i - k, i + 1):
                if j >= 0:
                    cond.update(f"{o}{j}" for o in public)
            for j in range(i - k, i):
                if j >= 0:
                    cond.add(f"X{j}")
            Z[f"X{i}"] = cond
    else:
        raise ValueError(encoding)
    missing = [t for t in range(H) if f"X{t}" not in Z]
    assert not missing, f"Z sets do not cover t={missing[:5]}"  # launch check 7
    return trim_Z_sets(Z, lookback=k)


class Encoder:
    """z = encode(obs, t) for one environment and input type. 'wmask' = the vanilla layout with every
    W slot set to zero (the Ant-Medium mechanism control)."""

    def __init__(self, name: str, encoding: str):
        s = spec(name)
        self.name, self.encoding = name, encoding
        self.Z = z_sets(name, encoding)
        self._enc, self.dim, self.slots = build_windowed_z_encoder(self.Z, dims=s["dims"], lookback=s["k"])
        self.mask = None
        if encoding == "wmask":
            mask, pos = np.ones(self.dim, dtype=np.float32), 0
            for v, _lag in self.slots:
                d = s["dims"][v]
                if v == "W":
                    mask[pos:pos + d] = 0.0
                pos += d
            self.mask = mask

    def __call__(self, obs, t: int) -> np.ndarray:
        z = self._enc(obs, t)
        return z * self.mask if self.mask is not None else z

    def schema(self) -> Dict[str, Any]:
        return {"encoding": self.encoding, "dim": int(self.dim), "slots": [list(x) for x in self.slots]}


# ── Compact demonstrations ──────────────────────────────────────────────────────────────────────

AUDIT_KEYS = ("reward", "U")  # stored for audit only; never exposed to learners


def save_episodes(path: Path, name: str, episodes: List[Dict[str, np.ndarray]], meta: Dict[str, Any]) -> str:
    """episodes: dicts of var -> (T+1, d) for states, 'X' -> (T, A), 'reward' (T,), 'U' (T+1, 2),
    'terminated' bool. Saved as one npz with offsets; returns its sha256."""
    s = spec(name)
    arrays: Dict[str, np.ndarray] = {}
    lengths = np.array([len(e["X"]) for e in episodes], dtype=np.int64)
    arrays["lengths"] = lengths
    arrays["terminated"] = np.array([bool(e["terminated"]) for e in episodes])
    for v in s["state_vars"] + ["U"]:
        arrays[v] = np.concatenate([np.asarray(e[v], dtype=np.float32).reshape(len(e[v]), -1) for e in episodes])
    arrays["X"] = np.concatenate([np.asarray(e["X"], dtype=np.float32) for e in episodes])
    arrays["reward"] = np.concatenate([np.asarray(e["reward"], dtype=np.float32) for e in episodes])
    arrays["meta"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return sha256(path)


class Demos:
    """Read-only view of a saved demo file. obs(i) gives the learner view of episode i: a dict of arrays
    with the hidden variable, U and rewards removed."""

    def __init__(self, path: Path, name: str):
        s = spec(name)
        with np.load(path) as f:
            self.data = {k: f[k] for k in f.files}
        self.meta = json.loads(bytes(self.data.pop("meta")).decode())
        self.lengths = self.data["lengths"]
        self.terminated = self.data["terminated"]
        self.state_off = np.concatenate([[0], np.cumsum(self.lengths + 1)])
        self.act_off = np.concatenate([[0], np.cumsum(self.lengths)])
        self.public = [v for v in s["state_vars"] if v != s["hidden"]]
        self.hidden = s["hidden"]

    def __len__(self):
        return len(self.lengths)

    def obs(self, i: int, learner: bool = True) -> Dict[str, np.ndarray]:
        a, b = self.state_off[i], self.state_off[i + 1]
        out = {v: self.data[v][a:b] for v in (self.public if learner else self.public + [self.hidden])}
        out["X"] = self.data["X"][self.act_off[i]:self.act_off[i + 1]]
        return out

    def actions(self, i: int) -> np.ndarray:
        return self.data["X"][self.act_off[i]:self.act_off[i + 1]]

    def records(self, idx: Iterable[int]) -> List[Dict[str, Any]]:
        """Light per-step records in the format the library initializers expect
        (initialize_expert_buffer, iq_init_expert_buffer, make_expert_batch, BC datasets).
        Each record's 'obs' is a shared view of the episode arrays, so no history is copied."""
        recs = []
        for i in idx:
            o, acts, T = self.obs(i), self.actions(i), int(self.lengths[i])
            term = bool(self.terminated[i])
            for t in range(T):
                last = t == T - 1
                recs.append({"episode": int(i), "step": t, "obs": o, "action": acts[t],
                             "terminated": last and term, "truncated": last and not term})
        return recs

    def n_transitions(self, idx: Iterable[int]) -> int:
        return int(sum(self.lengths[i] for i in idx))


# ── Expert ──────────────────────────────────────────────────────────────────────────────────────

# SCIL expert checkpoints, extracted from Causal-RL eylam-imitation at 02ac8ce (blobs identical to the paper
# notebook commit 9c8741c) into runs/task7/experts/scil/.
EXPERT_FILES = {"antmed": "antmaze_expert_finetuned.pt", "antlarge": "antmaze_large_expert_finetuned.pt",
                "hummed": "humanoidmaze_medium_expert_finetuned.pt",
                "humlarge": "humanoidmaze_large_expert_finetuned_k5.pt"}


def load_expert(path: Path, device=torch.device("cpu")):
    """SCIL expert checkpoint (ContinuousPolicyNN + window spec) -> (policy(obs_history) -> action, ckpt)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ContinuousPolicyNN(
        # hidden width from the weights: the Humanoid-Medium checkpoint records hidden_dim=512 but holds 256-wide
        # layers (SCIL's notebooks hard-coded 256 when loading it).
        input_dim=ck["input_dim"], action_dim=ck["num_actions"], hidden_dim=int(ck["state_dict"]["head.weight"].shape[1]),
        num_blocks=ck["num_blocks"], dropout=ck["dropout"], layernorm=ck["layernorm"], final_tanh=ck["final_tanh"],
        action_bounds=(ck["action_bounds_low"], ck["action_bounds_high"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return shared_policy_fn_long_horizon(model, ck["slots"], ck["Z_trim"], continuous=True, device=device), ck


def run_episode(env, policy: Callable[[Dict[str, Any], int], np.ndarray], seed: int, H: int) -> Dict[str, Any]:
    """One episode with do(); policy(obs_history, t) -> action. Returns the full arrays of the episode."""
    obs, _ = env.reset(seed=seed)
    terminated = False
    for t in range(H):
        a = np.asarray(policy(obs, t), dtype=np.float32)
        obs, _r, terminated, truncated, _ = env.do(lambda _o, a=a: a, show_reward=True)
        if terminated or truncated:
            break
    scm = env.env
    names = [v for v in ("P", "O", "A", "L", "T", "J", "H", "E", "V", "C", "W") if hasattr(scm, v)]
    ep = {v: np.asarray(getattr(scm, v), dtype=np.float32) for v in names}
    ep["X"] = np.asarray(scm.X, dtype=np.float32)
    ep["reward"] = np.asarray(scm._Y, dtype=np.float32)
    ep["U"] = np.asarray(scm._U, dtype=np.float32)
    ep["terminated"] = bool(terminated)
    ep["success"] = bool(terminated)
    ep["length"] = len(scm.X)
    ep["return"] = float(np.sum(scm._Y))
    return ep


def collect(name: str, policy, seeds: Sequence[int], role: str = "expert", c: float = 1.0) -> List[Dict[str, Any]]:
    env = make_env(name, role, seed=int(seeds[0]) if len(seeds) else 0, c=c)
    H = spec(name)["H"]
    return [run_episode(env, policy, int(s), H) for s in seeds]


# ── Isolated evaluation ─────────────────────────────────────────────────────────────────────────

def summarize_episodes(eps: List[Dict[str, Any]], H: int) -> Dict[str, Any]:
    succ = np.array([e["success"] for e in eps], dtype=float)
    lens = np.array([e["length"] for e in eps], dtype=float)
    rets = np.array([e["return"] for e in eps], dtype=float)
    return {"episodes": len(eps), "success_rate": float(succ.mean()), "return_mean": float(rets.mean()),
            "length_mean": float(lens.mean()), "timeout_rate": float(np.mean((~succ.astype(bool)) & (lens >= H))),
            "capped_time_to_goal": float(np.mean(np.where(succ.astype(bool), lens, H))),
            "per_episode": [{"success": bool(e["success"]), "length": int(e["length"]), "return": float(e["return"])}
                            for e in eps]}


def evaluate(name: str, act: Callable[[Dict[str, Any], int], np.ndarray], seeds: Sequence[int], c: float,
             modules: Sequence[torch.nn.Module] = ()) -> Dict[str, Any]:
    """Evaluate on a fresh eval environment. Modules are put in eval() mode and restored; global
    torch/numpy RNG states are saved and restored, so training randomness is untouched (launch check 11)."""
    was_training = [m.training for m in modules]
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    try:
        for m in modules:
            m.eval()
        env = make_env(name, "eval", seed=int(seeds[0]) if len(seeds) else 0, c=c)
        H = spec(name)["H"]
        with torch.no_grad():
            eps = [run_episode(env, act, int(s), H) for s in seeds]
        env.close()
        return summarize_episodes(eps, H)
    finally:
        for m, w in zip(modules, was_training):
            m.train(w)
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)


def actor_policy(actor, encoder: Encoder, device=torch.device("cpu")):
    """Deterministic action (squashed mean) of a ContinuousActor on encoded history."""
    def act(obs, t):
        z = torch.from_numpy(encoder(obs, t)).float().unsqueeze(0).to(device)
        a, _, _ = actor.act(z, deterministic=True)
        return a.squeeze(0).cpu().numpy().astype(np.float32)
    return act


# ── Seed streams ────────────────────────────────────────────────────────────────────────────────
# Disjoint integer ranges per purpose; within a stream the seed depends only on (env, run seed, item).

STREAMS = {"train_env": 1, "torch": 2, "checkpoint_eval": 3, "final_eval": 4, "secondary_eval": 5,
           "demo": 6, "expert_dev": 7, "expert_confirm": 8}


def seed_of(stream: str, env_index: int, run_seed: int, item: int) -> int:
    assert 0 <= item < 10_000_000 and 0 <= run_seed < 100_000
    return STREAMS[stream] * 10**13 + env_index * 10**12 + run_seed * 10**7 + item


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_mb() -> float:
    import resource, sys
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024
