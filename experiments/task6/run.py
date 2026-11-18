#!/usr/bin/env python3
"""Task 6 only: State vs State+Natural (counterfactual) training on the frozen suite.

Commands
  prepare      freeze sources/config into <output>/snapshot, run preflight checks
  run          train/evaluate pending jobs in parallel CPU worker processes
  status       count job states
  summarize    collect per-run results into results.csv / checkpoints.csv
  pause        stop launching new jobs (running jobs finish); clear-pause undoes it
  worker       (internal) one job in its own process

Neural backbones are the unmodified CleanRL dqn/c51/ppo scripts. A vector-env
hook saves the learner's weights at each checkpoint step without touching
training; all evaluation happens after training on separate environments.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import fcntl
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TASK = 6
NEURAL = ("dqn", "c51", "ppo")
REFERENCES = ("random", "follow_natural")
BANDITS = ("ucb", "ctf_ucb")
MODEL_NAME = {"dqn": "q_network", "c51": "q_network", "ppo": "agent"}
# Relative per-transition cost from the 3 November pilot (ms/transition), used only to order jobs.
COST = {"c51": 0.9, "dqn": 0.5, "ppo": 0.3}
GIB = 1024 ** 3
ENV_ID = "Task6-v0"

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


# ── Small utilities ──────────────────────────────────────────────────────────

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_write(path, writer, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="."+path.name)
    try:
        with os.fdopen(fd, mode) as f:
            writer(f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path, value):
    atomic_write(path, lambda f: json.dump(value, f, indent=2, allow_nan=False))


def write_csv_gz(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["empty"]

    def writer(f):
        with gzip.open(f, "wt", newline="") as g:
            w = csv.DictWriter(g, fieldnames=keys)
            w.writeheader()
            w.writerows({k: row.get(k, "") for k in keys} for row in rows)
    atomic_write(path, writer, "wb")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["empty"]

    def writer(f):
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: row.get(k, "") for k in keys} for row in rows)
    atomic_write(path, writer)


def free_gib(path):
    return shutil.disk_usage(path).free / GIB


# ── Configuration ────────────────────────────────────────────────────────────

def checkpoints_for(schedule):
    if "checkpoints" in schedule:
        return list(schedule["checkpoints"])
    every = schedule["checkpoint_every"]
    return list(range(0, schedule["steps"] + 1, every))


def configuration(path, mode="main"):
    cfg = read_json(path)
    if mode == "smoke":
        cfg["seeds"] = cfg["seeds"][:1]
        cfg["schedules"] = {"short": {"steps": 4096, "checkpoints": [0, 2048, 4096]},
                            "long": {"steps": 8192, "checkpoint_every": 4096}}
        cfg["evaluation"] = {**cfg["evaluation"], "checkpoint_episodes": 5, "final_episodes": 20}
        cfg["xor"] = {**cfg["xor"], "steps": 500, "checkpoints": [100, 500]}
        cfg["hyperparameters"] = copy.deepcopy(cfg["hyperparameters"])
        for algo in ("dqn", "c51"):
            cfg["hyperparameters"][algo]["learning_starts"] = 1000
    ppo = cfg["hyperparameters"]["ppo"]
    batch = ppo["num_envs"] * ppo["num_steps"]
    for name, schedule in cfg["schedules"].items():
        schedule["checkpoints"] = checkpoints_for(schedule)
        schedule.pop("checkpoint_every", None)
        cps = schedule["checkpoints"]
        if cps[0] != 0 or cps[-1] != schedule["steps"] or cps != sorted(set(cps)):
            raise ValueError(f"Schedule {name}: checkpoints must start at 0, end at the budget and increase")
        if any(c % batch for c in cps):
            raise ValueError(f"Schedule {name}: every checkpoint must be a whole PPO batch ({batch})")
    ids = [e["id"] for e in cfg["environments"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate environment id")
    for e in cfg["environments"]:
        if e["schedule"] not in cfg["schedules"]:
            raise ValueError(f"Unknown schedule for {e['id']}")
    if not set(cfg["random_hint"]["environments"]) <= set(ids):
        raise ValueError("random_hint lists an unknown environment")
    return cfg


def enabled(cfg):
    return [(i, e) for i, e in enumerate(cfg["environments"]) if e["enabled"]]


def spec_of(cfg, env_id):
    for i, e in enumerate(cfg["environments"]):
        if e["id"] == env_id:
            return i, e
    raise KeyError(env_id)


# ── Sources, snapshot, manifest ──────────────────────────────────────────────

def source_paths(gym_source=None):
    gym = ROOT if (ROOT / "causal_gym").is_dir() else Path(
        gym_source or os.environ.get("CAUSAL_GYM_SOURCE", ROOT.parent / "Causal-Gymnasium")).resolve()
    if not (gym / "causal_gym").is_dir():
        raise ValueError(f"Missing local environment source: {gym}")
    for path in (ROOT / "third_party/cleanrl", gym, ROOT):
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))
    return gym


def source_files(gym):
    files = {}
    for package, parent in (("causal_rl", ROOT), ("causal_gym", gym)):
        for p in sorted((parent / package).rglob("*.py")):
            files[str(p.relative_to(parent))] = p
    for p in sorted((gym / "causal_gym/envs/assets").rglob("*")):
        if p.is_file():
            files[str(p.relative_to(gym))] = p
    clean = ROOT / "third_party/cleanrl"
    for rel in ("cleanrl/dqn.py", "cleanrl/c51.py", "cleanrl/ppo.py", "PROVENANCE.json"):
        files["third_party/cleanrl/" + rel] = clean / rel
    for p in sorted((clean / "cleanrl_utils").rglob("*.py")):
        files[str(p.relative_to(ROOT))] = p
    for name in ("run.py", "config.json"):
        files["experiments/task6/" + name] = HERE / name
    return files


def runtime_versions():
    names = ("numpy", "gymnasium", "torch", "tyro", "tensorboard", "greenlet", "pygame")
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def verify_snapshot(output, check_runtime=True):
    output = Path(output)
    if not (output / "manifest.json").exists():
        raise ValueError(f"No Task 6 manifest in {output}; run prepare first")
    saved = read_json(output / "manifest.json")
    payload = {k: saved.get(k) for k in ("task", "mode", "config", "sources", "runtime")}
    if saved.get("task") != TASK or digest(payload) != saved.get("identity"):
        raise ValueError("Not a valid Task 6 manifest")
    if check_runtime and runtime_versions() != saved["runtime"]:
        raise ValueError("Runtime changed; restore the saved Python/library versions")
    for name, expected in saved["sources"].items():
        if hashlib.sha256((output / "snapshot" / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source changed: " + name)
    if read_json(output / "config.json") != saved["config"]:
        raise ValueError("Saved Task 6 configuration changed")
    return saved


def freeze(output, cfg, mode, gym):
    output = Path(output)
    if (output / "manifest.json").exists():
        saved = verify_snapshot(output)
        if saved["config"] != cfg or saved["mode"] != mode:
            raise ValueError("Output belongs to a different protocol; choose a new directory")
        return saved
    if any(p.name != ".task6.lock" for p in output.iterdir()):
        raise ValueError("Output is nonempty without a Task 6 manifest; refusing to overwrite")
    hashes = {}
    for name, path in source_files(gym).items():
        content = path.read_bytes()
        atomic_write(output / "snapshot" / name, lambda f, b=content: f.write(b), "wb")
        hashes[name] = hashlib.sha256(content).hexdigest()
    payload = {"task": TASK, "mode": mode, "config": cfg, "sources": hashes, "runtime": runtime_versions()}
    saved = {**payload, "identity": digest(payload),
             "source_directories": {"rl": str(ROOT), "gym": str(gym)},
             "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "seed_design": seed_design()}
    write_json(output / "config.json", cfg)
    write_json(output / "manifest.json", saved)
    return saved


def seed_design():
    return ("Training: CleanRL --seed = 1e6*(env_index+1) + 100*seed, shared by do/ctf/random_hint "
            "within a backbone and seed. Evaluation reset seeds: stream base + 1e8*env_index + "
            "1e4*seed + episode, streams 3e9 (checkpoints) and 4e9 (final); identical across "
            "backbones, modes and references for the same environment and seed, disjoint across "
            "seeds, environments and streams. XOR: 3e7 + 1e4*seed + t.")


# ── Jobs ─────────────────────────────────────────────────────────────────────

def jobs(cfg):
    out = []
    for index, spec in enabled(cfg):
        steps = cfg["schedules"][spec["schedule"]]["steps"]
        for algo in cfg["algorithms"]:
            modes = list(cfg["modes"])
            if algo in cfg["random_hint"]["algorithms"] and spec["id"] in cfg["random_hint"]["environments"]:
                modes.append("random_hint")
            for mode, seed in itertools.product(modes, cfg["seeds"]):
                out.append(dict(kind="neural", env=spec["id"], algorithm=algo, mode=mode, seed=seed,
                                steps=steps, cost=steps * COST[algo]))
        for algo, seed in itertools.product(cfg["references"], cfg["seeds"]):
            out.append(dict(kind="reference", env=spec["id"], algorithm=algo, mode="ctf", seed=seed,
                            steps=0, cost=1.0))
    for algo, seed in itertools.product(BANDITS, cfg["seeds"]):
        out.append(dict(kind="bandit", env="casino_xor", algorithm=algo, mode="native", seed=seed,
                        steps=cfg["xor"]["steps"], cost=1.0))
    for job in out:
        job["key"] = f"{job['env']}/{job['algorithm']}-{job['mode']}/seed_{job['seed']}"
    out.sort(key=lambda j: -j["cost"])
    return out


def job_dir(output, key):
    return Path(output) / "jobs" / key


def job_identity(saved, job):
    return digest({"experiment": saved["identity"],
                   **{k: job[k] for k in ("env", "algorithm", "mode", "seed", "steps")}})


def statuses(output, saved):
    counts, rows = Counter(), []
    for job in jobs(saved["config"]):
        path = job_dir(output, job["key"]) / "status.json"
        state = read_json(path) if path.exists() else {"status": "pending"}
        if state.get("status") != "pending" and state.get("identity") != job_identity(saved, job):
            state = {"status": "foreign"}
        counts[state["status"]] += 1
        rows.append({**job, **state})
    return dict(counts), rows


# ── Environments ─────────────────────────────────────────────────────────────

def native(spec):
    kind = spec["kind"]
    if kind == "robotwalk":
        from causal_gym.envs.robowalk import RobotWalkPCH
        return RobotWalkPCH(length=spec["length"], timelimit=spec["horizon"], put=spec["put"])
    if kind == "mdp":
        from causal_gym.envs.mdp import MDPPCH
        env = MDPPCH(init_dist=spec["init_dist"], max_step=spec["horizon"])
        # Native MDP has no constructor options for its noise probabilities.
        for name, p in zip(("_u1", "_u2", "_u3"), spec["noise_probs"]):
            setattr(env.env, name, lambda p=p, env=env: env.env.rng.choice(2, p=[1 - p, p]))
        return env
    if kind == "drone":
        from causal_gym.envs.rescue_drone import RescueDronePCH

        class DistanceDrone(RescueDronePCH):
            # Native random-pair sampling, with the reset distance independent of H.
            def reset(self, *, seed=None, options=None):
                horizon = self.env.max_steps
                self.env.max_steps = self.distance
                try:
                    return super().reset(seed=seed, options=options)
                finally:
                    self.env.max_steps = horizon

        env = DistanceDrone(grid_size=spec["grid_size"], max_steps=spec["horizon"],
                            wind_probs=spec["wind_probs"], random_start=spec["random_start"],
                            drone_start=spec.get("drone_start", [0, 0]),
                            victim_start=spec.get("victim_start", [0, 2]))
        env.distance = spec["distance"]
        return env
    if kind == "cartpole":
        from causal_gym.envs.cartpole_wind import CartPoleWindPCH
        w = [float(v) for v in spec["behavior"]["weights"]]
        k = spec["behavior"]["wind_on_index"]

        def behavior(obs, wind):
            s = [float(v) for v in obs]
            s[k] += float(wind or 0.0)
            return int(sum(a * b for a, b in zip(w, s)) >= 0)

        return CartPoleWindPCH(max_episode_steps=spec["horizon"], wind_mean=0.0, wind_std=spec["wind_std"],
                               init_theta_mean=0.0, init_theta_std=spec["init_theta_std"], policy=behavior)
    raise NotImplementedError(f"{spec['id']}: {spec.get('disabled_reason', 'no native adapter')}")


def make_env(spec, mode):
    import gymnasium as gym
    import numpy as np
    from causal_rl.wrappers import CTFWrapper

    class Inputs(gym.ObservationWrapper):
        """Normalize public inputs to [0,1]; optionally add an uninformative random hint."""

        def __init__(self, env, random_hint=False):
            super().__init__(env)
            self.random_hint = random_hint
            low = env.observation_space.low.astype(np.float64)
            high = env.observation_space.high.astype(np.float64)
            bounds = spec.get("obs_bounds")
            for i in range(len(low)):
                if not (np.isfinite(low[i]) and np.isfinite(high[i])):
                    if bounds is None or i >= len(bounds):
                        raise ValueError(f"{spec['id']}: unbounded input {i} needs obs_bounds")
                    low[i], high[i] = -bounds[i], bounds[i]
            self.low, self.scale = low, np.maximum(1e-8, high - low)
            size = env.observation_space.shape[0] + (env.action_space.n if random_hint else 0)
            self.observation_space = gym.spaces.Box(0, 1, (size,), dtype=np.float32)

        @property
        def spec(self):
            # gym.make() writes its spec onto env.unwrapped; some native PCHs (CartPoleWind)
            # expose an inner gym env as `unwrapped`, so the chain above it reports None.
            s = super().spec
            return s if s is not None else self.unwrapped.spec

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.reset_rng = np.random.default_rng(seed)
            elif not hasattr(self, "reset_rng"):
                self.reset_rng = np.random.default_rng()
            episode_seed = int(self.reset_rng.integers(0, 2 ** 63 - 1))
            self.hint_rng = np.random.default_rng(episode_seed ^ 0x12345678)
            return super().reset(seed=episode_seed, options=options)

        def observation(self, obs):
            obs = np.clip((obs - self.low) / self.scale, 0.0, 1.0)
            if self.random_hint:
                hint = np.zeros(self.action_space.n, dtype=np.float32)
                if obs[-1] > 0:
                    hint[self.hint_rng.integers(self.action_space.n)] = 1
                obs = np.concatenate((obs[:-1], hint, obs[-1:]))
            return obs.astype(np.float32)

    wrapped = CTFWrapper(native(spec), mode="do" if mode == "random_hint" else mode, horizon=spec["horizon"])
    return Inputs(wrapped, random_hint=(mode == "random_hint"))


def native_scm(env):
    """Inputs -> CTFWrapper -> native PCH -> its SCM (analysis/tests only)."""
    return env.env.env.env


def success_of(spec, env, info, term_native, length):
    kind = spec["kind"]
    if kind == "robotwalk":
        return int(env.unwrapped.current_location == spec["length"])
    if kind == "drone":
        return int(bool(info.get("rescued", False)))
    if kind == "cartpole":
        return int(length >= spec["horizon"] and not term_native)
    return ""


# ── Evaluation ───────────────────────────────────────────────────────────────

def eval_seed(env_index, seed, stream, episode):
    base = {"checkpoint": 3_000_000_000, "final": 4_000_000_000}[stream]
    return base + 100_000_000 * env_index + 10_000 * seed + episode


def evaluate(spec, env_index, mode, seed, stream, episodes, act, label):
    import numpy as np
    env = make_env(spec, mode)
    n = env.action_space.n
    rows = []
    for episode in range(episodes):
        reset_seed = eval_seed(env_index, seed, stream, episode)
        obs, info = env.reset(seed=reset_seed)
        rng = np.random.default_rng(reset_seed ^ 0x55555555)
        total, agree, known, optimal, counts = 0.0, 0, 0, 0, [0] * n
        native_term, t = False, 0
        for t in range(spec["horizon"]):
            intention = info.get("next_natural_action")
            action = int(act(obs, info, rng))
            counts[action] += 1
            if intention is not None:
                known += 1
                agree += int(action == int(intention))
                if spec["kind"] == "mdp":
                    # Native MDP's expected-reward-optimal action is 1-I (analysis only).
                    optimal += int(action == 1 - int(intention))
            obs, reward, term, trunc, info = env.step(action)
            total += float(reward)
            if term or trunc:
                native_term = bool(info.get("native_terminated", term and not info.get("task_horizon_reached")))
                break
        length = t + 1
        success = success_of(spec, env, info, native_term, length)
        row = dict(label=label, episode=episode, reset_seed=reset_seed, return_=total, length=length,
                   success=success,
                   capped_time=(length if success == 1 else spec["horizon"]) if success != "" else "",
                   hint_agreement=agree / known if known else "",
                   optimal_action_rate=optimal / known if spec["kind"] == "mdp" and known else "")
        row.update({f"action_{a}": c for a, c in enumerate(counts)})
        rows.append(row)
    env.close()
    return rows


def summary(rows):
    import numpy as np
    out = {"episodes": len(rows), "mean_return": float(np.mean([r["return_"] for r in rows])),
           "mean_length": float(np.mean([r["length"] for r in rows]))}
    for key in ("success", "capped_time", "hint_agreement", "optimal_action_rate"):
        vals = [r[key] for r in rows if r[key] != ""]
        out[key if key != "success" else "success_rate"] = float(np.mean(vals)) if vals else None
    return out


def greedy_actor(algorithm, model):
    import torch

    def act(obs, info, rng):
        with torch.no_grad():
            x = torch.as_tensor(obs[None], dtype=torch.float32)
            if algorithm == "ppo":
                return int(model.actor(x).argmax(1))
            if algorithm == "c51":
                return int(model.get_action(x)[0])
            return int(model(x).argmax(1))
    return act


def sampled_ppo_actor(model):
    import torch

    def act(obs, info, rng):
        with torch.no_grad():
            logits = model.actor(torch.as_tensor(obs[None], dtype=torch.float32))[0]
            p = torch.softmax(logits.double(), 0).numpy()
            return int(rng.choice(len(p), p=p / p.sum()))
    return act


# ── Training ─────────────────────────────────────────────────────────────────

def cleanrl_argv(cfg, spec, job, env_index):
    hp = cfg["hyperparameters"]
    seed = 1_000_000 * (env_index + 1) + 100 * job["seed"]
    argv = [job["algorithm"], "--env-id", ENV_ID, "--seed", str(seed), "--no-cuda",
            "--total-timesteps", str(job["steps"])]
    values = {**hp["common"], **hp[job["algorithm"]]}
    if job["algorithm"] == "c51":
        values.update(v_min=spec["support"][0], v_max=spec["support"][1])
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            argv.append(flag if value else "--no-" + key.replace("_", "-"))
        else:
            argv += [flag, str(value)]
    return argv


def train_neural(cfg, spec, env_index, job, folder):
    import gymnasium as gym
    import numpy as np
    import runpy
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:        # already set in this process (only the first call is allowed)
        pass
    checkpoints = cfg["schedules"][spec["schedule"]]["checkpoints"]
    ckpt_dir = folder / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = MODEL_NAME[job["algorithm"]]
    saved_steps = []

    def save(step, model):
        torch.save({k: v.detach().clone() for k, v in model.state_dict().items()}, ckpt_dir / f"step_{step:08d}.pt")
        saved_steps.append(step)

    base = gym.vector.sync_vector_env.SyncVectorEnv     # the original class, never a previous patch

    class Vector(base):
        # CleanRL expects same-step autoresets and a list of final_info records.
        def __init__(self, factories, **kwargs):
            super().__init__(factories, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP, **kwargs)
            self.executed_steps = 0

        def step(self, actions):
            if self.executed_steps in checkpoints and self.executed_steps not in saved_steps:
                # Weights after all updates for the first `executed_steps` transitions.
                frame = sys._getframe(1)
                while frame is not None and name not in frame.f_globals:
                    frame = frame.f_back
                if frame is None:
                    raise RuntimeError("Checkpoint hook could not find the CleanRL model")
                save(self.executed_steps, frame.f_globals[name])
            obs, rewards, terms, truncs, infos = super().step(actions)
            self.executed_steps += self.num_envs
            if truncs.any():
                raise RuntimeError("All configured task horizons must be terminal")
            if "final_info" in infos:
                def unpack(batch, i):
                    return {k: unpack(v, i) if isinstance(v, dict) else v[i]
                            for k, v in batch.items() if not k.startswith("_")}
                infos["final_info"] = [unpack(infos["final_info"], i) if terms[i] else None
                                       for i in range(self.num_envs)]
                infos["final_observation"] = infos.pop("final_obs")
            return obs, rewards, terms, truncs, infos

    gym.register(ENV_ID, entry_point=lambda: make_env(spec, job["mode"]),
                 disable_env_checker=True, order_enforce=False)
    gym.vector.SyncVectorEnv = Vector
    sys.argv = cleanrl_argv(cfg, spec, job, env_index)
    os.chdir(folder)
    state = runpy.run_path(str(ROOT / "third_party/cleanrl/cleanrl" / (job["algorithm"] + ".py")),
                           run_name="__main__")
    executed = state["envs"].executed_steps
    if executed != job["steps"]:
        raise RuntimeError(f"Executed {executed} transitions, expected {job['steps']}")
    model = state[name]
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Non-finite model parameters")
    save(job["steps"], model)
    if saved_steps != checkpoints:
        raise RuntimeError(f"Checkpoint steps {saved_steps} != {checkpoints}")
    model.eval()
    torch.save({"weights": model.state_dict(), "args": vars(state["args"])}, folder / "model.pt")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    scalars = []
    for events in (folder / "runs").iterdir():
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0}).Reload()
        for tag in acc.Tags()["scalars"]:
            values = acc.Scalars(tag)
            if tag.startswith("losses/") and tag != "losses/explained_variance":
                if not all(np.isfinite(v.value) for v in values):
                    raise RuntimeError(f"Non-finite training scalar {tag}")
            scalars.extend(dict(tag=tag, step=v.step, value=v.value) for v in values)
    write_csv_gz(folder / "scalars.csv.gz", scalars)
    shutil.rmtree(folder / "runs")          # TensorBoard events duplicate scalars.csv.gz
    return model, checkpoints


def run_neural(cfg, spec, env_index, job, folder):
    import torch
    model, checkpoints = train_neural(cfg, spec, env_index, job, folder)
    ev = cfg["evaluation"]
    act = greedy_actor(job["algorithm"], model)
    curve = []
    for step in checkpoints:
        model.load_state_dict(torch.load(folder / "checkpoints" / f"step_{step:08d}.pt"))
        model.eval()
        rows = evaluate(spec, env_index, job["mode"], job["seed"], "checkpoint",
                        ev["checkpoint_episodes"], act, f"step_{step}")
        curve.append({"step": step, **summary(rows)})
        for r in rows:
            r["step"] = step
        write_csv_gz(folder / "eval" / f"checkpoint_{step:08d}.csv.gz", rows)
    write_csv(folder / "checkpoints.csv", curve)
    final = evaluate(spec, env_index, job["mode"], job["seed"], "final", ev["final_episodes"], act, "greedy")
    write_csv_gz(folder / "eval" / "final.csv.gz", final)
    result = {"final": summary(final), "train_steps": job["steps"]}
    if job["algorithm"] == "ppo" and ev.get("ppo_sampled_final"):
        sampled = evaluate(spec, env_index, job["mode"], job["seed"], "final", ev["final_episodes"],
                           sampled_ppo_actor(model), "sampled")
        write_csv_gz(folder / "eval" / "final_sampled.csv.gz", sampled)
        result["final_sampled"] = summary(sampled)
    return result


def run_reference(cfg, spec, env_index, job, folder):
    probe = make_env(spec, "ctf")
    n_actions = probe.action_space.n
    probe.close()

    def act(obs, info, rng):
        if job["algorithm"] == "follow_natural":
            return int(info["next_natural_action"])
        return int(rng.integers(n_actions))

    rows = evaluate(spec, env_index, "ctf", job["seed"], "final", cfg["evaluation"]["final_episodes"],
                    act, job["algorithm"])
    write_csv_gz(folder / "eval" / "final.csv.gz", rows)
    return {"final": summary(rows), "train_steps": 0}


def run_bandit(cfg, job, folder):
    from causal_gym.envs.greedy_casino import MABPCH
    from causal_rl.algo.ctf_do.ctfucb import UCB, CtfUCB
    env = MABPCH()
    ctf = job["algorithm"] == "ctf_ucb"
    alpha = cfg["xor"]["alpha"]
    learner = CtfUCB([0, 1], alpha=alpha) if ctf else UCB([0, 1], alpha=alpha)
    rows, cumulative, own = [], 0.0, 0.0
    for t in range(job["steps"]):
        env.reset(seed=30_000_000 + job["seed"] * 10_000 + t)
        transition = env.ctf_do(learner.select_arm) if ctf else env.do(learner.select_arm)
        _, reward, _, _, info = transition
        action = int(info["action"])
        intention = int(info["natural_action"]) if ctf else int(env.env.action())
        if ctf:
            learner.update(intention, action, reward)
        else:
            learner.update(action, reward)
        # Oracle means (analysis only): .45 when A != I, .15 when A == I.
        # Without the intention both arms have mean .30, so ordinary UCB has zero own-information regret.
        mean = 0.45 if action == 1 - intention else 0.15
        cumulative += 0.45 - mean
        own += (0.45 - mean) if ctf else 0.0
        rows.append(dict(step=t + 1, action=action, intention=intention, reward=int(reward),
                         conditional_oracle_regret=cumulative, own_information_regret=own))
    write_csv_gz(folder / "interactions.csv.gz", rows)
    write_json(folder / "learner.json", {"counts": {str(k): v for k, v in learner.counts.items()},
                                         "values": {str(k): v for k, v in learner.values.items()}})
    env.close()
    cps = {str(c): {"conditional_oracle_regret": rows[c - 1]["conditional_oracle_regret"],
                    "own_information_regret": rows[c - 1]["own_information_regret"]}
           for c in cfg["xor"]["checkpoints"]}
    return {"train_steps": job["steps"], "checkpoints": cps,
            "mean_reward": sum(r["reward"] for r in rows) / len(rows),
            "conditional_oracle_regret": cumulative, "own_information_regret": own}


def run_worker(output, key):
    output = Path(output)
    saved = verify_snapshot(output, check_runtime=True)
    cfg = saved["config"]
    job = next(j for j in jobs(cfg) if j["key"] == key)
    folder = job_dir(output, key)
    identity = job_identity(saved, job)
    for stale in ("checkpoints", "eval", "runs", "scalars.csv.gz", "model.pt", "result.json", "checkpoints.csv",
                  "interactions.csv.gz", "learner.json"):
        p = folder / stale
        shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    base = {k: job[k] for k in ("key", "env", "algorithm", "mode", "seed", "steps")}
    write_json(folder / "status.json", {**base, "identity": identity, "status": "running",
                                        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    start = time.monotonic()
    try:
        if job["kind"] == "bandit":
            result = run_bandit(cfg, job, folder)
        else:
            env_index, spec = spec_of(cfg, job["env"])
            runner = run_neural if job["kind"] == "neural" else run_reference
            result = runner(cfg, spec, env_index, job, folder)
        seconds = time.monotonic() - start
        write_json(folder / "result.json", {**base, **result, "seconds": seconds, "identity": identity})
        write_json(folder / "status.json", {**base, "identity": identity, "status": "complete", "seconds": seconds})
        return 0
    except Exception as exc:
        write_json(folder / "status.json", {**base, "identity": identity, "status": "failed",
                                            "error": repr(exc), "traceback": traceback.format_exc()})
        raise


# ── Preflight ────────────────────────────────────────────────────────────────

def preflight(cfg, episodes=12):
    import numpy as np
    report = []
    for index, spec in enabled(cfg):
        modes = ["do", "ctf"] + (["random_hint"] if spec["id"] in cfg["random_hint"]["environments"] else [])
        for seed in range(episodes):
            envs = {m: make_env(spec, m) for m in modes}
            obs = {m: envs[m].reset(seed=seed) for m in modes}
            n = envs["do"].action_space.n
            state = {m: o for m, (o, _) in obs.items()}
            infos = {m: i for m, (_, i) in obs.items()}
            for t in range(spec["horizon"] + 1):
                d, c = state["do"], state["ctf"]
                # Same public state in both conditions; ctf adds exactly one one-hot intention.
                np.testing.assert_array_equal(d[:-1], c[:len(d) - 1])
                np.testing.assert_array_equal(d[-1:], c[-1:])
                if "random_hint" in state:
                    r = state["random_hint"]
                    np.testing.assert_array_equal(d[:-1], r[:len(d) - 1])
                for m in modes:
                    assert envs[m].observation_space.contains(state[m]), (spec["id"], m)
                assert infos["do"].get("next_natural_action") is None, "do mode must not expose intention"
                if t == 0 and spec["kind"] == "drone":
                    u = envs["do"].unwrapped
                    assert abs(u.drone_pos - u.victim_pos).sum() == spec["distance"]
                if spec["kind"] == "cartpole" and spec["wind_std"] == 0 and infos["ctf"].get("next_natural_action") is not None:
                    s = native_scm(envs["ctf"]).observation()
                    w = spec["behavior"]["weights"]
                    assert infos["ctf"]["next_natural_action"] == int(sum(a * b for a, b in zip(w, s)) >= 0)
                action = (t * 7 + seed) % n
                out = {m: envs[m].step(action) for m in modes}
                rewards = {m: o[1] for m, o in out.items()}
                dones = {m: (o[2], o[3]) for m, o in out.items()}
                assert len(set(rewards.values())) == 1 and len(set(dones.values())) == 1, spec["id"]
                state = {m: o[0] for m, o in out.items()}
                infos = {m: o[4] for m, o in out.items()}
                if dones["do"][1]:
                    raise AssertionError(f"{spec['id']}: native truncation before the task horizon")
                if dones["do"][0]:
                    assert t + 1 <= spec["horizon"]
                    break
            for e in envs.values():
                e.close()
        report.append({"environment": spec["id"], "paired_episodes": episodes, "modes": modes})
    return report


# ── Dispatcher ───────────────────────────────────────────────────────────────

def dispatch_run(output, saved, args):
    if not (output / "preflight.json").exists() or read_json(output / "preflight.json")["identity"] != saved["identity"]:
        raise ValueError("Run prepare first; preflight is missing or belongs to another experiment")
    if (output / "PAUSE").exists():
        raise ValueError("Experiment is paused; run clear-pause before continuing")
    cfg = saved["config"]
    _, rows = statuses(output, saved)
    if any(r["status"] == "foreign" for r in rows):
        raise ValueError("A job folder belongs to another experiment identity; inspect before running")
    selected = [r for r in rows if r["status"] != "complete"
                and (not args.envs or r["env"] in args.envs)
                and (not args.algorithms or r["algorithm"] in args.algorithms)
                and (not args.modes or r["mode"] in args.modes)
                and (not args.seeds or r["seed"] in args.seeds)]
    minimum = cfg["disk"]["run_min_free_gib"]
    snapshot_run = output / "snapshot/experiments/task6/run.py"
    print(f"TASK 6: {len(selected)} jobs, {args.workers} worker processes (1 Torch thread each); "
          f"free disk {free_gib(output):.1f} GiB", flush=True)

    def launch(job):
        if (output / "PAUSE").exists():
            return job["key"], "not-started"
        if free_gib(output) < minimum:
            (output / "PAUSE").touch()
            print(f"PAUSED: free disk below {minimum} GiB", flush=True)
            return job["key"], "not-started"
        folder = job_dir(output, job["key"])
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / "train.log"
        with log.open("w") as f:
            code = subprocess.call([sys.executable, str(snapshot_run), "worker", "--output", str(output),
                                    "--job", job["key"]], stdout=f, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=str(folder))
        with log.open("rb") as src, gzip.open(folder / "train.log.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        log.unlink()
        return job["key"], "complete" if code == 0 else "failed"

    previous = {s: signal.signal(s, lambda *_: (output / "PAUSE").touch()) for s in (signal.SIGINT, signal.SIGTERM)}
    counts = Counter()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(launch, job) for job in selected]
            for done, future in enumerate(as_completed(futures), 1):
                key, state = future.result()
                counts[state] += 1
                if state != "not-started":
                    print(f"{done}/{len(futures)} {state.upper()} {key}", flush=True)
    finally:
        for s, h in previous.items():
            signal.signal(s, h)
    totals, _ = statuses(output, saved)
    write_json(output / "status.json", {"task": TASK, "identity": saved["identity"], "counts": totals})
    print(json.dumps(totals, sort_keys=True), flush=True)
    return 1 if counts["failed"] else 0


def summarize(output, saved):
    results, curves = [], []
    for job in jobs(saved["config"]):
        folder = job_dir(output, job["key"])
        if not (folder / "result.json").exists():
            continue
        res = read_json(folder / "result.json")
        base = {k: job[k] for k in ("env", "algorithm", "mode", "seed", "steps")}
        row = {**base, "seconds": res["seconds"]}
        for block in ("final", "final_sampled"):
            for k, v in res.get(block, {}).items():
                row[f"{block}_{k}"] = v
        for k in ("conditional_oracle_regret", "own_information_regret", "mean_reward"):
            if k in res:
                row[k] = res[k]
        results.append(row)
        if (folder / "checkpoints.csv").exists():
            with (folder / "checkpoints.csv").open() as f:
                curves.extend({**base, **r} for r in csv.DictReader(f))
    write_csv(output / "results.csv", results)
    write_csv(output / "checkpoints.csv", curves)
    print(f"SUMMARIZED {len(results)} completed jobs -> {output/'results.csv'}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "status", "summarize", "pause", "clear-pause", "worker"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path)
        p.add_argument("--mode", choices=("main", "smoke"), default="main")
        if command == "prepare":
            p.add_argument("--config", type=Path, default=HERE / "config.json")
            p.add_argument("--gym-source", type=Path)
        if command == "run":
            p.add_argument("--workers", type=int, default=7)
            p.add_argument("--envs", nargs="+")
            p.add_argument("--algorithms", nargs="+")
            p.add_argument("--modes", nargs="+")
            p.add_argument("--seeds", nargs="+", type=int)
        if command == "worker":
            p.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    output = (args.output or ROOT / "runs/task6" / args.mode).resolve()
    target = output / "snapshot/experiments/task6/run.py"

    if args.command == "prepare":
        cfg = configuration(args.config, args.mode)
        gym = source_paths(args.gym_source)
        if (output / "manifest.json").exists():
            verify_snapshot(output)
        elif output.exists() and any(p.name != ".task6.lock" for p in output.iterdir()):
            raise ValueError("Nonempty unrelated output directory; refusing to write Task 6 files")
        output.mkdir(parents=True, exist_ok=True)
        need = cfg["disk"]["prepare_min_free_gib"]
        if free_gib(output) < need:
            raise ValueError(f"Only {free_gib(output):.1f} GiB free; Task 6 requires at least {need} GiB")
        with (output / ".task6.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            saved = freeze(output, cfg, args.mode, gym)
            if Path(__file__).resolve() == target:
                report = preflight(saved["config"])
                write_json(output / "preflight.json", {"identity": saved["identity"], "checks": report})
                counts, rows = statuses(output, saved)
                print(f"PREPARED {args.mode}: {len(rows)} jobs; preflight passed for "
                      f"{len(report)} environments; no training started: {output}", flush=True)
                return 0
        return subprocess.call([sys.executable, str(target), "prepare", "--output", str(output),
                                "--mode", args.mode, "--config", str(args.config.resolve())])

    saved = verify_snapshot(output)
    if Path(__file__).resolve() != target:
        forwarded = list(argv if argv is not None else sys.argv[1:])
        if "--output" not in forwarded:
            forwarded += ["--output", str(output)]
        return subprocess.call([sys.executable, str(target), *forwarded])
    source_paths()
    if args.command == "worker":
        return run_worker(output, args.job)
    if args.command == "status":
        counts, rows = statuses(output, saved)
        write_json(output / "status.json", {"task": TASK, "identity": saved["identity"], "counts": counts})
        print(json.dumps({"mode": saved["mode"], "planned": len(rows), "counts": counts}, indent=2))
        return 0
    if args.command == "summarize":
        summarize(output, saved)
        return 0
    if args.command in ("pause", "clear-pause"):
        (output / "PAUSE").touch() if args.command == "pause" else (output / "PAUSE").unlink(missing_ok=True)
        print(args.command, output)
        return 0
    if not 1 <= args.workers <= 12:
        raise ValueError("workers must be between 1 and 12")
    with (output / ".task6.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return dispatch_run(output, saved, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise SystemExit(1)
