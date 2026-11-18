#!/usr/bin/env python3
"""Task 5 orchestration. Learners and structural equations come from local repos."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("full", "mis", "pomis", "all_at_once")


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
    atomic_write(path, lambda f: f.write((json.dumps(value, indent=2, sort_keys=True,
                                                   allow_nan=False) + "\n").encode()))


def source_paths(gym_source=None):
    """Prefer the frozen package tree when executing a saved experiment."""
    gym = ROOT if (ROOT / "causal_gym").is_dir() else Path(
        gym_source or os.environ.get("CAUSAL_GYM_SOURCE", ROOT.parent / "Causal-Gymnasium")
    ).resolve()
    if not (gym / "causal_gym").is_dir():
        raise ValueError(f"Missing local Causal-Gymnasium source: {gym}")
    for path in (gym, ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return gym


def configuration(path, mode="main"):
    cfg = read_json(path)
    if cfg["methods"] != list(METHODS) or cfg["horizon"] != 1 or cfg["alpha"] != 1:
        raise ValueError("Task 5 requires the four declared methods, H=1, alpha=1")
    ids = [e["id"] for e in cfg["environments"]]
    if len(ids) != len(set(ids)) or len(cfg["seeds"]) != len(set(cfg["seeds"])):
        raise ValueError("Duplicate environment IDs or seeds")
    if mode == "smoke":
        cfg = copy.deepcopy(cfg)
        cfg["seeds"] = [20]
        cfg["common_steps"] = 1024
        cfg["checkpoint_every"] = 256
        cfg["optimal_arm_window"] = 1000
        for e in cfg["environments"]:
            e["budget"] = 1024
            e["checkpoints"] = [256, 512, 1024]
    for e in cfg["environments"]:
        if (e["budget"] < cfg["common_steps"] or e["checkpoints"][-1] != e["budget"]
                or e["checkpoints"] != sorted(set(e["checkpoints"]))
                or e["checkpoints"][0] < 1 or cfg["common_steps"] not in e["checkpoints"]):
            raise ValueError(f"Invalid checkpoints/budget: {e['id']}")
    return cfg


def make_environment(spec):
    # These imports resolve to the frozen local implementations during training.
    if spec["kind"] == "example":
        from causal_gym.envs.wheredo_example import ExampleSCM_9_5
        return ExampleSCM_9_5()
    if spec["kind"] == "six_variable":
        from causal_gym.envs.wheredo_experiemnt import WhereDoSCM
        return WhereDoSCM()
    from causal_gym.envs.wheredo_synthetic import SyntheticWhereDoSCM
    return SyntheticWhereDoSCM(structure=spec["structure"], n_actions=spec["d"])


def execute_intervention(env, spec, arm, episode_seed):
    """A fresh independent SCM unit per pull; no context is passed to UCB."""
    env.reset(seed=int(episode_seed))
    if spec["kind"] == "example":
        # The textbook environment exposes the older explicit-noise step API.
        u1, u2 = env.sample_u()
        _, reward, terminated, truncated, _ = env.step(
            u1, u2, arm.get("X1"), arm.get("X2"))
    else:
        _, reward, terminated, truncated, _ = env.step(dict(arm))
    if not (terminated or truncated) or reward not in (0, 1):
        raise ValueError("Task 5 expects one-step episodes and binary reward")
    return int(reward)


def exact_arm_means(spec, arms):
    """Evaluation-only exhaustive integration of the actual local SCM equations."""
    env = make_environment(spec)
    if spec["kind"] == "example":
        probabilities = {"U1": .5, "U2": .5}
    elif spec["kind"] == "six_variable":
        from causal_gym.envs.wheredo_experiemnt import NOISE_PROBS
        probabilities = dict(NOISE_PROBS)
    else:
        probabilities = dict(env.noise_probs)
    means = np.zeros(len(arms), dtype=np.float64)
    for values in itertools.product((0, 1), repeat=len(probabilities)):
        unit = dict(zip(probabilities, values))
        weight = np.prod([p if unit[u] else 1 - p for u, p in probabilities.items()])
        for i, arm in enumerate(arms):
            if spec["kind"] == "example":
                outcome = env.step(unit["U1"], unit["U2"], arm.get("X1"), arm.get("X2"))[1]
            else:
                env.u = unit
                outcome = env.F(arm)["Y"]
            means[i] += weight * outcome
    return means, probabilities


def prepare_environment(spec):
    from causal_rl.algo.where_do.graph import graph_to_vizdag
    from causal_rl.algo.where_do.where_do import WhereDo
    env = make_environment(spec)
    targets = ("X1", "X2") if spec["kind"] == "example" else tuple(env.intervenable)
    arms = [{n: v for n, v in zip(targets, values) if v is not None}
            for values in itertools.product((None, 0, 1), repeat=len(targets))]
    if spec["kind"] == "six_variable":
        # The local unrestricted solver assumes every non-outcome is manipulable.
        # Use the legal reference scopes fixed in the reviewed protocol.
        mis = [(), ("B",), ("D",), ("E",), ("B", "D"), ("B", "E"), ("D", "E")]
        pomis = [s for s in mis if s != ("B", "E")]
        scope_source = "legal reference scopes + local UCB"
    else:
        graph = env.get_graph
        graph = graph() if callable(graph) else graph
        graph = graph_to_vizdag(graph)
        solver = WhereDo()
        mis, pomis = solver.MISs(graph, "Y"), solver.POMISs(graph, "Y")
        scope_source = "local WhereDo.MISs / WhereDo.POMISs"
    scopes = {"mis": {frozenset(s) for s in mis}, "pomis": {frozenset(s) for s in pomis}}
    if any(not s <= set(targets) for ss in scopes.values() for s in ss):
        raise ValueError("Candidate set contains an illegal intervention target")
    candidates = {"full": list(range(len(arms))),
                  "all_at_once": [i for i, a in enumerate(arms) if len(a) == len(targets)]}
    for method in ("mis", "pomis"):
        candidates[method] = [i for i, a in enumerate(arms) if frozenset(a) in scopes[method]]
    counts = [len(candidates[m]) for m in METHODS]
    if counts != spec["expected_counts"]:
        raise ValueError(f"{spec['id']}: candidate count drift: {counts}")
    means, probabilities = exact_arm_means(spec, arms)
    optimum = float(means.max())
    if not np.isclose(optimum, spec["expected_optimum"], atol=5e-8, rtol=0):
        raise ValueError(f"{spec['id']}: SCM optimum drift: {optimum}")
    for method in ("mis", "pomis"):
        if not np.isclose(means[candidates[method]].max(), optimum, atol=1e-12, rtol=0):
            raise ValueError(f"{spec['id']}: {method} discards every optimal intervention")
    return {"spec": spec, "targets": list(targets), "arms": arms, "candidates": candidates,
            "means": means.tolist(), "noise_probabilities": probabilities,
            "scope_source": scope_source, "scopes": {
                m: sorted([sorted(s) for s in ss], key=lambda s: (len(s), s))
                for m, ss in scopes.items()},
            "optimum": optimum, "minimum": float(means.min()),
            "set_optima": {m: float(means[candidates[m]].max()) for m in METHODS}}


def runtime_versions():
    return {"python": platform.python_version(), **{
        n: importlib.metadata.version(n) for n in ("numpy", "gymnasium", "networkx")}}


def source_files(gym):
    files = {}
    for package, parent in (("causal_rl", ROOT), ("causal_gym", gym)):
        for path in sorted((parent / package).rglob("*.py")):
            files[str(path.relative_to(parent))] = path
    # causal_gym imports its rendering constants eagerly, even for bandit tasks.
    for path in sorted((gym / "causal_gym/envs/assets").rglob("*")):
        if path.is_file():
            files[str(path.relative_to(gym))] = path
    files["experiments/task5.py"] = ROOT / "experiments/task5.py"
    return files


def verify_snapshot(output, check_runtime=True):
    saved = read_json(output / "manifest.json")
    identity_data = {k: saved[k] for k in ("config", "mode", "sources", "runtime")}
    if digest(identity_data) != saved["identity"]:
        raise ValueError("Experiment manifest identity mismatch")
    if check_runtime and saved["runtime"] != runtime_versions():
        raise ValueError("Python/NumPy/Gymnasium/NetworkX versions changed; restore the saved runtime")
    for rel, expected in saved["sources"].items():
        if hashlib.sha256((output / "snapshot" / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source hash mismatch: {rel}")
    if read_json(output / "snapshot/configs/task5.json") != saved["config"]:
        raise ValueError("Frozen configuration changed")
    return saved


def freeze(output, cfg, mode, gym):
    """Save source bytes once; existing runs always execute their saved sources."""
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        saved = verify_snapshot(output)
        if saved["config"] != cfg or saved["mode"] != mode:
            raise ValueError("Existing experiment has a different configuration; choose a new output")
        print("Using the saved source snapshot (including for resume).", flush=True)
        return saved
    if any(p.name != ".run.lock" for p in output.iterdir()):
        raise ValueError("Output is nonempty without a valid manifest; choose a new output")
    hashes = {}
    for rel, path in source_files(gym).items():
        data = path.read_bytes()
        target = output / "snapshot" / rel
        atomic_write(target, lambda f, data=data: f.write(data))
        hashes[rel] = hashlib.sha256(data).hexdigest()
    write_json(output / "snapshot/configs/task5.json", cfg)
    identity_data = {"config": cfg, "mode": mode, "sources": hashes, "runtime": runtime_versions()}
    saved = {**identity_data, "identity": digest(identity_data),
             "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "source_directories": {"rl": str(ROOT), "gym": str(gym)},
             "installed_packages": sorted(f"{d.metadata['Name']}=={d.version}"
                                          for d in importlib.metadata.distributions()),
             "seed_design": "Same exogenous unit stream within (environment, seed) across methods; independent streams across environment/seed. UCB is deterministic.",
             "stopping": "fixed budget; never convergence-based"}
    write_json(manifest_path, saved)
    return saved


def prepare_saved_experiment(output):
    saved = read_json(output / "manifest.json")
    cfg = saved["config"]
    expected = {}
    for spec in cfg["environments"]:
        plan = prepare_environment(spec)
        target = output / "environments" / (spec["id"] + ".json")
        if target.exists() and read_json(target) != plan:
            raise ValueError(f"Environment plan changed: {spec['id']}")
        write_json(target, plan)
        expected[spec["id"]] = digest(plan)
    record = {"identity": saved["identity"], "environment_hashes": expected}
    if (output / "preflight.json").exists() and read_json(output / "preflight.json") != record:
        raise ValueError("Preflight manifest changed")
    write_json(output / "preflight.json", record)
    return saved


def job_path(output, environment, method, seed):
    return Path(output) / "jobs" / environment / method / str(seed)


def metrics_for_trace(plan, method, actions, rewards, recommendations, cfg):
    means = np.asarray(plan["means"])
    global_star, set_star = plan["optimum"], plan["set_optima"][method]
    regret = np.cumsum(np.maximum(global_star - means[actions], 0))
    own = np.cumsum(np.maximum(set_star - means[actions], 0))
    metrics = []
    for step, recommendation in sorted((int(t), a) for t, a in recommendations.items()):
        if step > len(actions):
            continue
        window = min(cfg["optimal_arm_window"], step)
        metrics.append({"steps": step, "cumulative_pseudo_regret": float(regret[step - 1]),
                        "own_set_regret": float(own[step - 1]),
                        "candidate_set_value_gap": float(global_star - set_star),
                        "simple_regret": float(max(0, global_star - means[recommendation])),
                        "own_set_simple_regret": float(max(0, set_star - means[recommendation])),
                        "recommended_arm": int(recommendation),
                        "optimal_arm_fraction": float(np.isclose(means[actions[step-window:step]],
                                                                  global_star, atol=1e-12, rtol=0).mean()),
                        "optimal_arm_window": window,
                        "observed_mean_reward": float(rewards[:step].mean()),
                        "normalized_score": float(1 - regret[step-1] / (step * (global_star-plan["minimum"])))} )
    return metrics


def save_checkpoint(path, identity, agent, rng, actions, rewards, episode_seeds, recommendations, plan, method):
    meta = {"identity": identity, "steps": len(actions), "total": agent.total,
            "rng": rng.bit_generator.state, "recommendations": recommendations}
    executed_means = np.asarray(plan["means"])[np.asarray(actions, dtype=int)]
    atomic_write(path, lambda f: np.savez_compressed(
        f, meta=np.array(json.dumps(meta, sort_keys=True)), actions=np.asarray(actions, dtype=np.int16),
        rewards=np.asarray(rewards, dtype=np.uint8), episode_seeds=np.asarray(episode_seeds, dtype=np.uint64),
        pseudo_regret=np.maximum(plan["optimum"]-executed_means, 0),
        own_set_pseudo_regret=np.maximum(plan["set_optima"][method]-executed_means, 0),
        counts=np.asarray([agent.counts[a] for a in agent.arms], dtype=np.int64),
        values=np.asarray([agent.values[a] for a in agent.arms], dtype=np.float64)))


def load_checkpoint(path, identity, agent, rng, budget):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        actions, rewards, seeds = (data[k].copy() for k in ("actions", "rewards", "episode_seeds"))
        counts, values = data["counts"].copy(), data["values"].copy()
    step = meta["steps"]
    if (meta["identity"] != identity or not 0 <= step <= budget or meta["total"] != step
            or not len(actions) == len(rewards) == len(seeds) == step
            or counts.shape != (len(agent.arms),) or values.shape != counts.shape
            or np.any(counts < 0) or counts.sum() != step or not np.all(np.isfinite(values))
            or not np.isin(actions, agent.arms).all() or not np.isin(rewards, (0, 1)).all()):
        raise ValueError(f"Invalid checkpoint identity/payload: {path}")
    # Check the complete trajectory against the saved local UCB sufficient statistics.
    trace_counts = np.bincount(actions, minlength=max(agent.arms)+1)[agent.arms]
    trace_sums = np.bincount(actions, weights=rewards, minlength=max(agent.arms)+1)[agent.arms]
    empirical = trace_sums / np.maximum(trace_counts, 1)
    if not np.array_equal(trace_counts, counts) or not np.allclose(empirical, values, atol=1e-12, rtol=0):
        raise ValueError(f"Checkpoint counts/values do not match its trajectory: {path}")
    agent.total = int(meta["total"])
    agent.counts = dict(zip(agent.arms, map(int, counts)))
    agent.values = dict(zip(agent.arms, map(float, values)))
    rng.bit_generator.state = meta["rng"]
    return actions.tolist(), rewards.tolist(), seeds.tolist(), meta["recommendations"]


def run_job(output, environment, method, seed, stop_after=None):
    """stop_after is used only to test interruption/resume; not a CLI budget override."""
    from causal_rl.algo.ctf_do.ctfucb import UCB
    output = Path(output)
    saved = read_json(output / "manifest.json")
    cfg = saved["config"]
    plan = read_json(output / "environments" / (environment + ".json"))
    plan_hash = digest(plan)
    if plan_hash != read_json(output / "preflight.json")["environment_hashes"][environment]:
        raise ValueError("Environment plan hash mismatch")
    identity = digest({"experiment": saved["identity"], "environment": plan_hash,
                       "method": method, "seed": seed})
    directory = job_path(output, environment, method, seed)
    directory.mkdir(parents=True, exist_ok=True)
    spec, arms = plan["spec"], plan["candidates"][method]
    budget = spec["budget"]
    agent = UCB(arms, alpha=cfg["alpha"])  # No oracle values/context are passed here.
    env_index = next(i for i, e in enumerate(cfg["environments"]) if e["id"] == environment)
    rng = np.random.default_rng(np.random.SeedSequence([0xCA05A1, env_index, seed]))
    actions, rewards, episode_seeds, recommendations = [], [], [], {}
    checkpoint = directory / "checkpoint.npz"
    started = time.monotonic()
    prior_seconds = read_json(directory / "status.json").get("seconds", 0) if (directory / "status.json").exists() else 0

    def status(state, reason=None):
        value = {"identity": identity, "environment": environment, "method": method, "seed": seed,
                 "status": state, "steps": len(actions), "budget": budget,
                 "seconds": prior_seconds + time.monotonic() - started}
        if reason:
            value["reason"] = reason
        write_json(directory / "status.json", value)
        return value

    def save():
        save_checkpoint(checkpoint, identity, agent, rng, actions, rewards, episode_seeds,
                        recommendations, plan, method)

    try:
        if checkpoint.exists():
            actions, rewards, episode_seeds, recommendations = load_checkpoint(
                checkpoint, identity, agent, rng, budget)
        status("running")
        env = make_environment(spec)
        while len(actions) < budget:
            if (output / "PAUSE").exists() or (stop_after is not None and len(actions) >= stop_after):
                save()
                return status("paused")
            # Agent selects before reset and receives only the executed arm's reward.
            arm = agent.select_arm()
            episode_seed = int(rng.integers(0, 2**64, dtype=np.uint64))
            reward = execute_intervention(env, spec, plan["arms"][arm], episode_seed)
            agent.update(arm, reward)
            actions.append(arm)
            rewards.append(reward)
            episode_seeds.append(episode_seed)
            step = len(actions)
            if step in spec["checkpoints"]:
                # Same first-arm tie order as the local UCB. Never recommend an untried arm.
                observed = [a for a in arms if agent.counts[a]]
                recommendations[str(step)] = max(observed, key=lambda a: agent.values[a])
            if step % cfg["checkpoint_every"] == 0:
                save()
                status("running")
        save()
        trace_metrics = metrics_for_trace(plan, method, np.array(actions), np.array(rewards),
                                          recommendations, cfg)
        if [m["steps"] for m in trace_metrics] != spec["checkpoints"]:
            raise ValueError("Checkpoint recommendation history is incomplete")
        write_json(directory / "summary.json", {
            "identity": identity, "environment": environment, "method": method, "seed": seed,
            "steps": len(actions), "budget": budget, "candidate_count": len(arms),
            "metrics": trace_metrics})
        return status("complete")
    except KeyboardInterrupt:
        # Save only after a finished pull; an interrupted partial pull is replayed from last durable checkpoint.
        return status("interrupted", "KeyboardInterrupt; resume from durable checkpoint")
    except Exception:
        atomic_write(directory / "error.txt", lambda f: f.write(traceback.format_exc().encode()))
        status("failed", traceback.format_exc().splitlines()[-1])
        raise


def execute_saved(args):
    output = args.output.resolve()
    if ROOT != output / "snapshot":
        raise ValueError("Internal execution must use the saved source snapshot")
    verify_snapshot(output)
    source_paths()
    saved = prepare_saved_experiment(output)
    cfg = saved["config"]
    envs = args.environments or [e["id"] for e in cfg["environments"]]
    methods = args.methods or cfg["methods"]
    seeds = args.seeds or cfg["seeds"]
    if (set(envs) - {e["id"] for e in cfg["environments"]} or set(methods) - set(METHODS)
            or set(seeds) - set(cfg["seeds"])):
        raise ValueError("Selected jobs are outside the saved experiment plan")
    jobs = list(itertools.product(envs, methods, seeds))
    print(f"{saved['mode']}: {len(jobs)} selected jobs; workers={args.workers}; frozen local UCB", flush=True)
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
    futures = {pool.submit(run_job, str(output), *job): job for job in jobs}
    failures, paused, finished = [], 0, 0
    try:
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                paused += result["status"] != "complete"
            except Exception as exc:
                failures.append((job, str(exc)))
                print(f"FAILED {job}: {exc}", flush=True)
            finished += 1
            if finished <= 3 or finished % 20 == 0 or finished == len(jobs):
                print(f"Finished {finished}/{len(jobs)}; failed={len(failures)}; paused={paused}", flush=True)
    except KeyboardInterrupt:
        (output / "PAUSE").touch()
        print("Pause requested; finishing durable checkpoints.", flush=True)
        for future in futures:
            future.cancel()
        paused += 1
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return 1 if failures else 2 if paused else 0


def status_report(output):
    saved = read_json(output / "manifest.json")
    counts, steps = {}, 0
    for e, method, seed in itertools.product(saved["config"]["environments"], METHODS, saved["config"]["seeds"]):
        path = job_path(output, e["id"], method, seed) / "status.json"
        record = read_json(path) if path.exists() else {"status": "pending", "steps": 0}
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        steps += record["steps"]
    return {"mode": saved["mode"], "jobs": counts, "recorded_steps": steps,
            "pause_requested": (output / "PAUSE").exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "smoke", "status", "pause", "_run"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task5.json")
    parser.add_argument("--gym-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--environments", nargs="+")
    parser.add_argument("--methods", choices=METHODS, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--clear-pause", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    for name in ("environments", "methods", "seeds"):
        selected = getattr(args, name)
        if selected and len(selected) != len(set(selected)):
            parser.error(f"Duplicate --{name} entries would schedule the same job twice")
    default = "task5_smoke" if args.command == "smoke" else "task5"
    args.output = (args.output or ROOT / "runs/benchmark1.0" / default).resolve()
    if args.command == "plan":
        source_paths(args.gym_source)
        cfg = configuration(args.config)
        print("Environment         Full  MIS POMIS  All  Budget  Exact optimum")
        for spec in cfg["environments"]:
            plan = prepare_environment(spec)
            counts = " ".join(f"{len(plan['candidates'][m]):4d}" for m in METHODS)
            print(f"{spec['id']:19s} {counts} {spec['budget']:7d} {plan['optimum']:.9f}")
        print(f"{len(METHODS)*len(cfg['environments'])*len(cfg['seeds'])} runs; "
              f"{len(METHODS)*len(cfg['seeds'])*sum(e['budget'] for e in cfg['environments']):,} interactions")
        return 0
    if args.command == "status":
        print(json.dumps(status_report(args.output), indent=2))
        return 0
    if args.command == "pause":
        if not (args.output / "manifest.json").exists():
            parser.error("No experiment at output")
        (args.output / "PAUSE").touch()
        print("Pause requested.")
        return 0
    if args.command == "_run":
        return execute_saved(args)
    gym = source_paths(args.gym_source)
    mode = "smoke" if args.command == "smoke" else "main"
    cfg = configuration(args.config, mode)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another launcher owns this output directory") from None
        freeze(args.output, cfg, mode, gym)
        if args.clear_pause:
            (args.output / "PAUSE").unlink(missing_ok=True)
        if (args.output / "PAUSE").exists():
            raise ValueError("Experiment is paused; resume explicitly with --clear-pause")
        command = [sys.executable, str(args.output / "snapshot/experiments/task5.py"), "_run",
                   "--output", str(args.output), "--workers", str(args.workers)]
        for name in ("environments", "methods", "seeds"):
            if getattr(args, name):
                command += ["--" + name, *map(str, getattr(args, name))]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.setdefault("MPLCONFIGDIR", str(args.output / "cache/matplotlib"))
        env.setdefault("XDG_CACHE_HOME", str(args.output / "cache"))
        process = subprocess.Popen(command, env=env)
        try:
            return process.wait()
        except KeyboardInterrupt:
            # Keep the output lock until the child and workers have stopped writing.
            (args.output / "PAUSE").touch()
            print("Pause requested; waiting for workers to stop.", flush=True)
            while process.poll() is None:
                try:
                    process.wait(timeout=1)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    continue
            return 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
