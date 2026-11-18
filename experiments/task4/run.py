#!/usr/bin/env python3
"""Task 4 only: frozen local learners, shared data, deterministic replay recovery."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ("ucb", "direct", "cool")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(path, lambda f: f.write((json.dumps(value, sort_keys=True, indent=2,
                                                   allow_nan=False) + "\n").encode()))


def write_csv(path, rows):
    if not rows:
        Path(path).unlink(missing_ok=True)
        return
    import io
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, lambda f: f.write(text.getvalue().encode()))


def configuration(path, mode="main"):
    cfg = read_json(path)
    if (cfg["methods"] != list(METHODS) or cfg["horizon"] != 1
            or cfg["offline_samples"] != 5000 or cfg["online_steps"] != 10000
            or cfg["seeds"] != list(range(2000, 2020)) or cfg["bound_alpha"] != .05
            or cfg["design"] != "2026-10-31-replacement"):
        raise ValueError("Protocol requires N=5000, T=10000, seeds 2000--2019, H=1, alpha=.05")
    expected = [
        ("mabcool", (.4, .15), 1., .8, False),
        ("mabcool", (.4, .12), 1., .8, False),
        ("mabcool", (.4, .1), 1., .8, False),
        ("mab", (.7, .6), 1., .97, False),
        ("mab", (.6, .7), 1., .97, True),
        ("mab", (.8, .6), .5, .9, False),
        ("mab", (.6, .8), .5, .9, True)]
    actual = [(e["kind"], tuple(e["means"]), e["rho"], e["q"], e["mirror"])
              for e in cfg["environments"]]
    if actual != expected or len({e["id"] for e in cfg["environments"]}) != 7:
        raise ValueError("Expected three native MABCOOL and four confounded MAB configurations")
    for spec in cfg["environments"]:
        if spec["group"] != spec["kind"] or not spec["id"].replace("_", "").isalnum():
            raise ValueError("Invalid family or environment ID")
        if spec["kind"] == "mabcool" and not np.allclose(
                spec["means"], [.4, .4-spec["delta"]], atol=1e-12, rtol=0):
            raise ValueError("Native MABCOOL delta/means mismatch")
        if theoretical_diagnostics(spec, cfg)["robust_separation_margin"] <= 0:
            raise ValueError("Population bound does not separate the suboptimal arm at this N")
    if (cfg["checkpoints"] != [500, 1000, 2000, 5000, 10000]
            or cfg["checkpoint_every"] != 500 or cfg["optimal_arm_window"] != 1000):
        raise ValueError("Unexpected checkpoints or final-window protocol")
    if mode == "smoke":
        cfg = copy.deepcopy(cfg)
        cfg.update(offline_samples=128, online_steps=64, seeds=[42],
                   checkpoint_every=16, checkpoints=[16, 32, 64], optimal_arm_window=32)
    return cfg


def theoretical_diagnostics(spec, cfg):
    """Population analysis only; never passed to action selection or data filtering."""
    good, bad = max(spec["means"]), min(spec["means"])
    q, rho = spec["q"], spec["rho"]
    coefficient = rho + (1-rho)*q
    upper_bad = 1-q+coefficient*bad
    epsilon = float(np.sqrt(np.log(8/cfg["bound_alpha"])/(2*cfg["offline_samples"])))
    return {"environment": spec["id"], "kind": spec["kind"],
            "population_suboptimal_upper": upper_bad, "epsilon": epsilon,
            "robust_suboptimal_upper": upper_bad+2*epsilon,
            "optimal_mean": good, "robust_separation_margin": good-upper_bad-2*epsilon}


def source_paths(gym_source=None):
    gym = ROOT if (ROOT / "causal_gym").is_dir() else Path(
        gym_source or os.environ.get("CAUSAL_GYM_SOURCE", ROOT.parent / "Causal-Gymnasium")
    ).resolve()
    if not (gym / "causal_gym").is_dir():
        raise ValueError(f"Missing local environment source: {gym}")
    # The worktree/frozen sources take precedence over editable installs elsewhere.
    for path in (gym, ROOT):
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))
    return gym


def make_environment(spec):
    if spec["kind"] == "mabcool":
        from causal_gym.envs.mab_cool import MABCOOLPCH
        return MABCOOLPCH(delta=spec["delta"])
    from causal_gym.envs.mab import MABPCH
    env = MABPCH(confounding_strength=spec["rho"], arms_probs=spec["means"])
    q = spec["q"]
    env.env.change_policy(lambda u: int(u >= q) if spec["mirror"] else int(u < q))
    return env


def episode_seeds(env_index, seed, stream, size):
    rng = np.random.default_rng(np.random.SeedSequence([0xCA04C1, env_index, seed, stream]))
    return rng.integers(0, 2**64, size=size, dtype=np.uint64)


def data_digest(actions, rewards, seeds):
    h = hashlib.sha256()
    for array, dtype in ((actions, "u1"), (rewards, "u1"), (seeds, "<u8")):
        h.update(np.asarray(array, dtype=dtype).tobytes())
    return h.hexdigest()


def bound_statistics(actions, rewards, alpha):
    joint = np.zeros((2, 2))
    for a in (0, 1):
        for y in (0, 1):
            joint[a, y] = np.mean((actions == a) & (rewards == y))
    epsilon = float(np.sqrt(np.log(8/alpha)/(2*len(actions))))
    counts = np.bincount(actions, minlength=2)
    sums = np.bincount(actions, weights=rewards, minlength=2)
    conditional = [float(sums[a]/counts[a]) if counts[a] else None for a in (0, 1)]
    return {"joint": joint.tolist(), "epsilon": epsilon,
            "lower": np.maximum(0, joint[:, 1]-epsilon).tolist(),
            "upper": np.minimum(1, 1-joint[:, 0]+epsilon).tolist(),
            "counts": counts.tolist(), "sums": sums.tolist(),
            "conditional_means": conditional,
            "direct_initial_indices": (sums/(counts+1)+np.sqrt(np.log(1e4)/(counts+1))).tolist()}


def runtime_versions():
    return {"python": platform.python_version(), **{
        n: importlib.metadata.version(n) for n in ("numpy", "gymnasium", "networkx")}}


def source_files(gym):
    files = {}
    for package, parent in (("causal_rl", ROOT), ("causal_gym", gym)):
        for p in sorted((parent/package).rglob("*.py")):
            files[str(p.relative_to(parent))] = p
    # Eager causal_gym imports reference rendering assets even in bandit runs.
    for p in sorted((gym/"causal_gym/envs/assets").rglob("*")):
        if p.is_file():
            files[str(p.relative_to(gym))] = p
    for name in ("run.py", "config.json"):
        files["experiments/task4/"+name] = HERE/name
    return files


def verify_snapshot(output, check_runtime=True):
    saved = read_json(output/"manifest.json")
    if saved.get("task") != 4:
        raise ValueError("Not a Task 4 experiment; Task 5 directories cannot be used")
    payload = {k: saved[k] for k in ("task", "mode", "config", "sources", "runtime")}
    if saved["task"] != 4 or digest(payload) != saved["identity"]:
        raise ValueError("Not a valid Task 4 manifest (Task 5 directories cannot be used)")
    if check_runtime and runtime_versions() != saved["runtime"]:
        raise ValueError("Runtime changed; restore saved Python/NumPy/Gymnasium/NetworkX versions")
    for name, expected in saved["sources"].items():
        if hashlib.sha256((output/"snapshot"/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source changed: "+name)
    if read_json(output/"config.json") != saved["config"]:
        raise ValueError("Saved Task 4 configuration changed")
    return saved


def freeze(output, cfg, mode, gym):
    if (output/"manifest.json").exists():
        saved = verify_snapshot(output)
        if saved["config"] != cfg or saved["mode"] != mode:
            raise ValueError("Output belongs to a different protocol; choose a new directory")
        return saved
    if any(p.name != ".task4.lock" for p in output.iterdir()):
        raise ValueError("Output is nonempty without a Task 4 manifest; refusing to overwrite")
    hashes = {}
    for name, path in source_files(gym).items():
        content = path.read_bytes()
        atomic_write(output/"snapshot"/name, lambda f, b=content: f.write(b))
        hashes[name] = hashlib.sha256(content).hexdigest()
    payload = {"task": 4, "mode": mode, "config": cfg, "sources": hashes,
               "runtime": runtime_versions()}
    saved = {**payload, "identity": digest(payload),
             "source_directories": {"rl": str(ROOT), "gym": str(gym)},
             "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "seed_design": "Independent offline/online streams by environment and seed; common online unit seeds across the three methods within each comparison.",
             "recovery": "Reconstruct original learner by replaying saved feedback; no extra environment interactions during prefix replay.",
             "algorithm_functions": {"ucb": "causal_rl.algo.cool.cool.cool_mab_ucb (no bounds)",
                                     "direct": "causal_rl.algo.cool.cool.mab_ucb_direct_transfer",
                                     "cool": "causal_rl.algo.cool.cool.cool_mab_ucb (empirical bounds)"}}
    write_json(output/"config.json", cfg)
    write_json(output/"manifest.json", saved)
    return saved


def dataset_paths(output, env_id, seed):
    family = "mabcool" if env_id.startswith("mabcool_") else "mab"
    folder = output/"data"/family/env_id/str(seed)
    return folder/"offline.npz", folder/"metadata.json"


def prepare_datasets(output):
    saved = verify_snapshot(output)
    cfg = saved["config"]
    data_hashes = {}
    for index, spec in enumerate(cfg["environments"]):
        for seed in cfg["seeds"]:
            path, metadata = dataset_paths(output, spec["id"], seed)
            if path.exists() and metadata.exists():
                _, _, existing = load_dataset(output, spec, index, seed, cfg)
                data_hashes[f"{spec['id']}/{seed}"] = existing["data_sha256"]
                continue
            if path.exists() or metadata.exists():
                raise ValueError(f"Incomplete dataset pair: {path.parent}; inspect before recovery")
            seeds = episode_seeds(index, seed, 0, cfg["offline_samples"])
            actions, rewards = [], []
            env = make_environment(spec)
            try:
                for value in seeds:
                    env.reset(seed=int(value))
                    _, reward, _, _, info = env.see()
                    actions.append(info["natural_action"])
                    rewards.append(reward)
            finally:
                env.close()
            actions, rewards = np.asarray(actions, dtype="uint8"), np.asarray(rewards, dtype="uint8")
            stats = bound_statistics(actions, rewards, cfg["bound_alpha"])
            meta = {"identity": saved["identity"], "environment": spec["id"], "seed": seed,
                    "n": len(actions), "data_sha256": data_digest(actions, rewards, seeds), **stats}
            atomic_write(path, lambda f: np.savez_compressed(f, actions=actions, rewards=rewards,
                                                            episode_seeds=seeds))
            write_json(metadata, meta)
            data_hashes[f"{spec['id']}/{seed}"] = meta["data_sha256"]
        print("DATA READY", spec["id"], len(cfg["seeds"]), "shared datasets", flush=True)
    write_csv(output/"design_diagnostics.csv", [theoretical_diagnostics(e, cfg) for e in cfg["environments"]])
    write_json(output/"prepared.json", {"identity": saved["identity"],
                                       "datasets": len(cfg["environments"])*len(cfg["seeds"]),
                                       "data_hashes": data_hashes})


def load_dataset(output, spec, index, seed, cfg):
    path, meta_path = dataset_paths(output, spec["id"], seed)
    meta = read_json(meta_path)
    with np.load(path, allow_pickle=False) as data:
        actions, rewards, seeds = (data[n].copy() for n in ("actions", "rewards", "episode_seeds"))
    identity = read_json(output/"manifest.json")["identity"]
    if (meta["identity"] != identity or meta["environment"] != spec["id"] or meta["seed"] != seed
            or not actions.shape == rewards.shape == seeds.shape == (cfg["offline_samples"],)
            or not np.isin(actions, (0, 1)).all() or not np.isin(rewards, (0, 1)).all()
            or meta["data_sha256"] != data_digest(actions, rewards, seeds)
            or not np.array_equal(seeds, episode_seeds(index, seed, 0, len(actions)))
            or meta["n"] != len(actions)):
        raise ValueError("Offline dataset identity/payload mismatch: "+str(path))
    stats = bound_statistics(actions, rewards, cfg["bound_alpha"])
    if any(meta[k] != v for k, v in stats.items()):
        raise ValueError("Offline bounds/statistics changed: "+str(meta_path))
    if (output/"prepared.json").exists():
        prepared = read_json(output/"prepared.json")
        if (prepared["identity"] != identity or
                prepared["data_hashes"].get(f"{spec['id']}/{seed}") != meta["data_sha256"]):
            raise ValueError("Dataset differs from the prepared data inventory")
    return actions, rewards, meta


class Paused(Exception):
    pass


class ReplayAdapter:
    """Only feedback/RNG/logging: the repository function always chooses actions."""
    def __init__(self, env, offline, seeds, prefix=None, save=None, pause=None):
        self.env = env
        self.offline_actions, self.offline_rewards = offline
        self.offline_index = 0
        self.seeds = seeds
        self.actions, self.rewards = [], []
        self.prefix = prefix or ([], [])
        self.replayed_steps = len(self.prefix[0])
        self.save, self.pause = save, pause
        self.pending_reset = False

    def reset(self):
        if self.pending_reset:
            raise ValueError("Reset without consuming previous SCM unit")
        self.pending_reset = True
        if self.offline_index < len(self.offline_actions) or len(self.actions) < self.replayed_steps:
            return None, {}
        if len(self.actions) >= len(self.seeds):
            raise ValueError("Online budget exceeded")
        return self.env.reset(seed=int(self.seeds[len(self.actions)]))

    def see(self):
        if not self.pending_reset or self.offline_index >= len(self.offline_actions):
            raise ValueError("Unexpected offline access")
        i = self.offline_index
        self.offline_index += 1
        self.pending_reset = False
        return None, int(self.offline_rewards[i]), True, False, {
            "natural_action": int(self.offline_actions[i])}

    def do(self, policy):
        if not self.pending_reset or self.offline_index != len(self.offline_actions):
            raise ValueError("Unexpected online access")
        self.pending_reset = False
        step = len(self.actions)
        if step < self.replayed_steps:
            action = int(policy())
            if action != self.prefix[0][step]:
                raise ValueError(f"Saved feedback does not reproduce original learner at step {step+1}")
            reward = int(self.prefix[1][step])
            result = None, reward, True, False, {"action": action}
        else:
            if self.pause and self.pause(step):
                if self.save:
                    self.save(self)
                raise Paused("Pause requested at an online interaction boundary")
            result = self.env.do(policy)
            action, reward = int(result[4]["action"]), result[1]
            if action not in (0, 1) or reward not in (0, 1):
                raise ValueError("Task 4 expects two arms and binary rewards")
        self.actions.append(action)
        self.rewards.append(int(reward))
        if self.save and len(self.actions) > self.replayed_steps:
            self.save(self)
        return result


def execute_local(spec, method, offline_actions, offline_rewards, online_seeds, bounds,
                  prefix=None, save=None, pause=None):
    from causal_rl.algo.cool.cool import cool_mab_ucb, mab_ucb_direct_transfer
    env = make_environment(spec)
    records = (offline_actions, offline_rewards) if method == "direct" else ([], [])
    adapter = ReplayAdapter(env, records, online_seeds, prefix, save, pause)
    try:
        if method == "direct":
            returned = mab_ucb_direct_transfer(adapter, do_timesteps=len(online_seeds),
                                               see_timesteps=len(offline_actions))
        elif method in ("ucb", "cool"):
            lower, upper = (np.asarray(bounds[k]) for k in ("lower", "upper"))
            returned = cool_mab_ucb(adapter, lower=lower if method == "cool" else None,
                                    upper=upper if method == "cool" else None,
                                    timesteps=len(online_seeds))
        else:
            raise ValueError("Unknown method: "+method)
        if returned != adapter.rewards:
            raise ValueError("Original learner return and recorded rewards disagree")
        return adapter
    finally:
        env.close()


def job_path(output, env_id, method, seed):
    family = "mabcool" if env_id.startswith("mabcool_") else "mab"
    return output/"jobs"/family/env_id/method/str(seed)


def job_identity(saved, spec, method, seed, meta):
    return digest({"experiment": saved["identity"], "environment": spec,
                   "method": method, "seed": seed, "data": meta["data_sha256"]})


def load_trace(path, identity, seeds, budget, means=None):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        actions, rewards, used_seeds = (data[n].copy() for n in ("actions", "rewards", "episode_seeds"))
        regret = data["pseudo_regret"].copy()
    n = len(actions)
    if (meta["identity"] != identity or not 0 <= n <= budget or meta["steps"] != n
            or actions.shape != (n,) or rewards.shape != (n,) or used_seeds.shape != (n,)
            or not np.isin(actions, (0, 1)).all() or not np.isin(rewards, (0, 1)).all()
            or not np.array_equal(used_seeds, seeds[:n])
            or meta["trace_sha256"] != data_digest(actions, rewards, used_seeds)):
        raise ValueError("Checkpoint identity, trace checksum or seed schedule mismatch: "+str(path))
    if means is not None:
        means = np.asarray(means)
        if regret.shape != (n,) or not np.array_equal(regret, means.max()-means[actions]):
            raise ValueError("Saved per-pull regret differs from the action trace")
    return actions.tolist(), rewards.tolist()


def metrics_for_trace(spec, method, actions, rewards, offline, cfg):
    """Post-training evaluation only; none of these oracle values selects actions."""
    actions, rewards = np.asarray(actions, dtype=int), np.asarray(rewards, dtype=int)
    means = np.asarray(spec["means"])
    best, gap = float(means.max()), float(np.ptp(means))
    initial_counts, initial_sums = np.ones(2), np.zeros(2)
    if method == "direct":
        initial_counts += np.asarray(offline["counts"])
        initial_sums += np.asarray(offline["sums"])
    regret = gap*np.cumsum(means[actions] < best)
    metrics = []
    for step in cfg["checkpoints"]:
        if step > len(actions):
            continue
        counts = np.bincount(actions[:step], minlength=2)
        sums = np.bincount(actions[:step], weights=rewards[:step], minlength=2)
        estimates = (initial_sums+sums)/(initial_counts+counts)
        if method == "cool":
            estimates = np.clip(estimates, offline["lower"], offline["upper"])
        recommendation = int(np.argmax(estimates))
        window = min(step, cfg["optimal_arm_window"])
        metrics.append({"steps": step, "cumulative_pseudo_regret": float(regret[step-1]),
                        "normalized_score": float(1-regret[step-1]/(step*gap)),
                        "simple_regret": float(best-means[recommendation]),
                        "recommended_arm": recommendation,
                        "optimal_arm_fraction": float(np.mean(means[actions[step-window:step]] == best)),
                        "optimal_arm_window": window,
                        "online_arm0_count": int(counts[0]), "online_arm1_count": int(counts[1]),
                        "observed_mean_reward": float(rewards[:step].mean())})
    return metrics


@functools.lru_cache(maxsize=4)
def verified_manifest(output_string):
    return verify_snapshot(Path(output_string))


def run_job(output_string, env_id, method, seed, stop_after=None):
    output = Path(output_string)
    saved = verified_manifest(str(output.resolve()))
    cfg = saved["config"]
    source_paths()
    index, spec = next((i, e) for i, e in enumerate(cfg["environments"]) if e["id"] == env_id)
    if method not in METHODS or seed not in cfg["seeds"]:
        raise ValueError("Job outside the frozen protocol")
    offline_actions, offline_rewards, meta = load_dataset(output, spec, index, seed, cfg)
    identity = job_identity(saved, spec, method, seed, meta)
    seeds = episode_seeds(index, seed, 1, cfg["online_steps"])
    folder = job_path(output, env_id, method, seed)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "busy", "job": [env_id, method, seed]}
        status = read_json(folder/"status.json") if (folder/"status.json").exists() else {}
        path = folder/"trace.npz"
        prefix = load_trace(path, identity, seeds, cfg["online_steps"], spec["means"]) if path.exists() else ([], [])
        if status and status.get("identity") != identity:
            raise ValueError("Job status identity mismatch")
        if status and not path.exists():
            raise ValueError("Previously started job has no checkpoint; no silent restart")
        if status.get("status") == "complete":
            if len(prefix[0]) != cfg["online_steps"] or not (folder/"summary.json").exists():
                raise ValueError("Completed job has missing/incomplete artifacts")
            return {"status": "skipped", "job": [env_id, method, seed]}
        write_json(folder/"status.json", {"identity": identity, "status": "running", "steps": len(prefix[0])})
        write_json(folder/"execution.json", {"pid": os.getpid(), "replayed_steps": len(prefix[0]),
                   "recovery": "exact saved feedback replay; no environment calls for replayed prefix"})
        saved_step = [-1]

        def persist(adapter, force=False):
            step = len(adapter.actions)
            if not force and step != cfg["online_steps"] and step % cfg["checkpoint_every"]:
                return
            if saved_step[0] == step:
                return
            values = {"identity": identity, "steps": step,
                      "trace_sha256": data_digest(adapter.actions, adapter.rewards, seeds[:step])}
            atomic_write(path, lambda f: np.savez_compressed(
                f, meta=np.array(json.dumps(values, sort_keys=True)),
                actions=np.asarray(adapter.actions, dtype="uint8"),
                rewards=np.asarray(adapter.rewards, dtype="uint8"), episode_seeds=seeds[:step],
                pseudo_regret=np.max(spec["means"])-np.asarray(spec["means"])[
                    np.asarray(adapter.actions, dtype=int)]))
            saved_step[0] = step
            write_json(folder/"status.json", {"identity": identity, "status": "running", "steps": step})

        stop = [False]
        previous = {}
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, lambda *_: stop.__setitem__(0, True))
        adapter_ref = [None]

        def save(adapter):
            adapter_ref[0] = adapter
            persist(adapter)

        def should_pause(step):
            return stop[0] or (output/"PAUSE").exists() or (stop_after is not None and step >= stop_after)

        try:
            # Save an empty valid trace first, so even an early interrupt is recoverable.
            empty = ReplayAdapter(None, ([], []), seeds)
            empty.actions, empty.rewards = list(prefix[0]), list(prefix[1])
            persist(empty, force=True)
            adapter = execute_local(spec, method, offline_actions, offline_rewards, seeds, meta,
                                    prefix, save, should_pause)
            persist(adapter, force=True)
            means = np.asarray(spec["means"])
            lower, upper = np.asarray(meta["lower"]), np.asarray(meta["upper"])
            bad = int(np.argmin(means))
            summary = {"identity": identity, "environment": env_id, "group": spec["group"],
                       "kind": spec["kind"],
                       "environment_class": "MABCOOLPCH" if spec["kind"] == "mabcool" else "MABPCH",
                       "method": method, "seed": seed, "offline_samples": cfg["offline_samples"],
                       "offline_sha256": meta["data_sha256"], "online_steps": len(adapter.actions),
                       "metrics": metrics_for_trace(spec, method, adapter.actions, adapter.rewards, meta, cfg),
                       "bounds": {"lower": meta["lower"], "upper": meta["upper"],
                                  "widths": (upper-lower).tolist(),
                                  "covers_true_means": bool(np.all(lower <= means) and np.all(means <= upper)),
                                  "suboptimal_upper_below_optimal_mean": bool(upper[bad] < means.max())},
                       "oracle_used_by_learner": False}
            write_json(folder/"summary.json", summary)
            write_csv(folder/"checkpoints.csv", summary["metrics"])
            state = {"identity": identity, "status": "complete", "steps": len(adapter.actions)}
        except Paused as exc:
            adapter = adapter_ref[0] or empty
            persist(adapter, force=True)
            state = {"identity": identity, "status": "paused", "steps": len(adapter.actions), "reason": str(exc)}
        except BaseException as exc:
            if adapter_ref[0] is not None:
                persist(adapter_ref[0], force=True)
            write_json(folder/"status.json", {"identity": identity, "status": "failed",
                       "reason": repr(exc), "traceback": traceback.format_exc()})
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        write_json(folder/"status.json", state)
        return {**state, "job": [env_id, method, seed]}


def statuses(output, saved):
    counts, rows = Counter(), []
    for spec in saved["config"]["environments"]:
        for method in METHODS:
            for seed in saved["config"]["seeds"]:
                path = job_path(output, spec["id"], method, seed)/"status.json"
                state = read_json(path) if path.exists() else {"status": "pending", "steps": 0}
                counts[state["status"]] += 1
                rows.append({"environment": spec["id"], "method": method, "seed": seed, **state})
    return dict(counts), rows


def dispatch_run(output, saved, args):
    if not (output/"prepared.json").exists():
        raise ValueError("Run prepare first; no training before offline datasets are frozen")
    if read_json(output/"prepared.json")["identity"] != saved["identity"]:
        raise ValueError("Prepared data belongs to a different experiment")
    if (output/"PAUSE").exists():
        raise ValueError("Experiment is paused; inspect status and run clear-pause before continuing")
    ids = {e["id"] for e in saved["config"]["environments"]}
    if args.envs and not set(args.envs) <= ids:
        raise ValueError("Unknown environment selection")
    if args.seeds and not set(args.seeds) <= set(saved["config"]["seeds"]):
        raise ValueError("Seed outside the saved experiment")
    _, rows = statuses(output, saved)
    jobs = [(str(output), r["environment"], r["method"], r["seed"]) for r in rows
            if r["status"] != "complete" and (not args.envs or r["environment"] in args.envs)
            and (not args.seeds or r["seed"] in args.seeds)
            and (not args.methods or r["method"] in args.methods)]
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: (output/"PAUSE").touch())
    failures = 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = {pool.submit(run_job, *job): job for job in jobs}
            for future in as_completed(pending):
                try:
                    result = future.result()
                    print(result["status"].upper(), *result["job"], flush=True)
                except Exception as exc:
                    failures += 1
                    print("FAILED", pending[future][1:], repr(exc), flush=True)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    counts, _ = statuses(output, saved)
    print(json.dumps(counts, sort_keys=True))
    return 1 if failures else (2 if any(k != "complete" and v for k, v in counts.items()) else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "status", "pause", "clear-pause"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path)
        p.add_argument("--mode", choices=("main", "smoke"), default="main")
        if command == "prepare":
            p.add_argument("--config", type=Path, default=HERE/"config.json")
            p.add_argument("--gym-source", type=Path)
        if command == "run":
            p.add_argument("--workers", type=int, default=2)
            p.add_argument("--envs", nargs="+")
            p.add_argument("--seeds", nargs="+", type=int)
            p.add_argument("--methods", nargs="+", choices=METHODS)
    args = parser.parse_args(argv)
    output = (args.output or ROOT/"runs/task4"/args.mode).resolve()
    if args.command == "prepare":
        cfg = configuration(args.config, args.mode)
        gym = source_paths(args.gym_source)
        # Reject unrelated experiments before even creating a Task 4 lock file.
        if (output/"manifest.json").exists():
            verify_snapshot(output)
        elif output.exists() and any(p.name != ".task4.lock" for p in output.iterdir()):
            raise ValueError("Nonempty unrelated output directory; refusing to write Task 4 files")
        output.mkdir(parents=True, exist_ok=True)
        with (output/".task4.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            freeze(output, cfg, args.mode, gym)
            target = output/"snapshot/experiments/task4/run.py"
            if Path(__file__).resolve() == target:
                prepare_datasets(output)
                print("PREPARED; no online training started:", output)
                return 0
        target = output/"snapshot/experiments/task4/run.py"
        return subprocess.call([sys.executable, str(target), "prepare", "--output", str(output),
                                "--mode", args.mode, "--config", str(args.config.resolve())])
    saved = verify_snapshot(output)
    target = output/"snapshot/experiments/task4/run.py"
    if Path(__file__).resolve() != target:
        forwarded = list(argv if argv is not None else sys.argv[1:])
        if "--output" not in forwarded:
            forwarded += ["--output", str(output)]
        return subprocess.call([sys.executable, str(target), *forwarded])
    source_paths()
    if args.command == "status":
        counts, rows = statuses(output, saved)
        write_json(output/"status.json", {"task": 4, "identity": saved["identity"], "counts": counts,
                                          "planned_runs": len(rows), "mode": saved["mode"]})
        print(json.dumps({"mode": saved["mode"], "planned": len(rows), "counts": counts}, indent=2))
        return 0
    if args.command in ("pause", "clear-pause"):
        if args.command == "pause":
            (output/"PAUSE").touch()
        else:
            (output/"PAUSE").unlink(missing_ok=True)
        print(args.command, output)
        return 0
    if args.workers < 1:
        raise ValueError("workers must be positive")
    with (output/".task4.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return dispatch_run(output, saved, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise SystemExit(1)
