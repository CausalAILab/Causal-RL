#!/usr/bin/env python3
"""Task 8 random reference: the return J_rand of the uniformly random policy on the target task.

analyze.py normalizes q = (J - J_rand) / (u - J_rand), as Section 6 of the report defines the
normalized score. As in Task 6, J_rand is the mean over the 10 seeds x 1000 episodes.

The environment is the run's frozen snapshot: the same evaluation sub-environment the trained
agents are scored on (UniformSampler on the target context bounds), restricted to do(). For each
seed it is the first evaluation sub-environment, SeedSequence([seed, 1]).generate_state(num_envs)[0],
and the actions are action_space.sample(), seeded with that sub-environment seed in make_env.

  python experiments/task8/random_reference.py --run runs/task8/main
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import random
import sys

EPISODES = 1000


def load_snapshot_runner(run):
    """Import the frozen run.py of the run and put the frozen sources on sys.path."""
    snapshot = run / "snapshot"
    spec = importlib.util.spec_from_file_location("task8_run", snapshot / "experiments/task8/run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for path in (snapshot / "experiments/task8", snapshot):
        sys.path.insert(0, str(path))
    return runner


def episode_returns(runner, cfg, env_id, seed, episodes):
    import numpy as np
    job = dict(env=env_id, curriculum="default", variant="original", seed=seed)
    exp = runner.experiment_class(env_id)(runner.run_args(cfg, job, "/nonexistent"), {})
    np.random.seed(seed)
    random.seed(seed)
    sub_seed = int(np.random.SeedSequence([seed, 1]).generate_state(cfg["ppo"]["num_envs"])[0])
    env = exp.make_env(True, sub_seed)()
    returns = []
    for _ in range(episodes):
        env.reset()
        total, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    return returns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path(__file__).resolve().parents[2] / "runs/task8/main")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    args = ap.parse_args()
    run = args.run.resolve()
    runner = load_snapshot_runner(run)
    saved = runner.verify_snapshot(run, check_runtime=True)
    cfg = saved["config"]
    import numpy as np

    rows, summary = [], {}
    for env in cfg["environments"]:
        per_seed = []
        for seed in cfg["seeds"]:
            r = episode_returns(runner, cfg, env["id"], seed, args.episodes)
            per_seed.append(r)
            rows.append({"env": env["id"], "seed": seed, "episodes": len(r), "mean_return": float(np.mean(r))})
            print(f"{env['id']:15s} seed {seed}: {np.mean(r):.4f}", flush=True)
        flat = np.concatenate(per_seed)
        seed_means = np.array([np.mean(r) for r in per_seed])
        summary[env["id"]] = {"j_rand": float(flat.mean()), "episodes": int(flat.size),
                              "seed_mean_sd": float(seed_means.std(ddof=1))}
    runner.write_csv(run / "random_reference.csv", rows)
    runner.write_json(run / "random_reference.json",
                      {"identity": saved["identity"], "episodes_per_seed": args.episodes, "policy": "uniform random",
                       "environments": summary})
    print(summary)


if __name__ == "__main__":
    main()
