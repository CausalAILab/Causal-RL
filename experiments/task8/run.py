#!/usr/bin/env python3
"""Task 8 only: Causally Aligned Curriculum Learning (Li, Zhang & Bareinboim, ICLR 2024).

Commands
  prepare      freeze sources/config into <output>/snapshot, run preflight checks
  run          train pending jobs in parallel CPU worker processes
  status       count job states
  summarize    collect per-run results into results.csv / curves.csv
  pause        stop launching new jobs (running jobs finish); clear-pause undoes it
  worker       (internal) one job in its own process

A job is one environment x curriculum x variant (original / causal) x seed. The worker
repeats the original run.py train(): build the experiment, seed torch / numpy / random,
train PPO with in-training evaluation on the target task every `log_interval` updates
(128 episodes, sampled actions), save the contexts used for training. After training it
keeps only the checkpoints named in config.checkpoints.keep and exports the logged scalars.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
import types

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TASK = 8
GIB = 1024 ** 3
EXPERIMENT_CLASS = {"button_maze": "ButtonMazeExperiment", "colored_sokoban": "ColoredSokobanExperiment"}
# Relative wall time per step from the 17 November timing run, used only to order jobs.
COST = {"button_maze": 1.45, "colored_sokoban": 1.0}
# C_{LOWER,UPPER}_CONTEXT_BOUNDS written by hand in the original experiment classes (f52fcea);
# preflight checks that FindMaxEdit on the Causal-Gym diagrams reproduces them.
PAPER_CAUSAL_BOUNDS = {
    "button_maze": ([1, 1, 1, 1, 1, 0, 0, 0], [1.999, 1.999, 1.999, 1.999, 1.999, 0.999, 0.25, 10.999]),
    "colored_sokoban": ([0, 0, 0], [2.999, 4.999, 0.999]),
}
# Fixed inside PPOInterface.train(): evaluate(mode='sample', ep=128)
PPO_EVALUATION = {"episodes": 128, "mode": "sample"}

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# ── Small utilities (as in Task 6) ───────────────────────────────────────────

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_write(path, writer, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, mode) as f:
            writer(f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path, value):
    atomic_write(path, lambda f: json.dump(value, f, indent=2, allow_nan=False))


def write_csv(path, rows, compress=False):
    keys = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["empty"]

    def put(f):
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: row.get(k, "") for k in keys} for row in rows)
    if compress:
        def writer(f):
            with gzip.open(f, "wt", newline="") as g:
                put(g)
        atomic_write(path, writer, "wb")
    else:
        atomic_write(path, put)


def free_gib(path):
    return shutil.disk_usage(path).free / GIB


# ── Configuration ────────────────────────────────────────────────────────────

def configuration(path, mode="main"):
    cfg = read_json(path)
    if mode == "smoke":
        # one seed, 10 updates: one in-training evaluation and the final checkpoint
        cfg["seeds"] = cfg["seeds"][:1]
        cfg["total_timesteps"] = 10 * cfg["ppo"]["num_envs"] * cfg["ppo"]["num_steps"]
    ppo = cfg["ppo"]
    batch = ppo["num_envs"] * ppo["num_steps"]
    if cfg["total_timesteps"] % batch:
        raise ValueError(f"total_timesteps must be a whole number of PPO batches ({batch})")
    if batch % ppo["num_minibatches"]:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    ev = cfg["evaluation"]
    if {k: ev[k] for k in PPO_EVALUATION} != PPO_EVALUATION:
        raise ValueError(f"evaluation episodes/mode are fixed by PPOInterface: {PPO_EVALUATION}")
    ids = [e["id"] for e in cfg["environments"]]
    if len(ids) != len(set(ids)) or not set(ids) <= set(EXPERIMENT_CLASS):
        raise ValueError(f"environments must be distinct ids from {sorted(EXPERIMENT_CLASS)}")
    if not set(cfg["curricula"]) <= {"default", "alp_gmm", "goal_gan", "plr", "currot"}:
        raise ValueError("Unknown curriculum")
    if not set(cfg["variants"]) <= {"original", "causal"}:
        raise ValueError("Unknown variant")
    if not set(cfg["checkpoints"]["keep"]) <= {"final", "all"}:
        raise ValueError("checkpoints.keep accepts 'final' or 'all'")
    return cfg


# ── Sources, snapshot, manifest ──────────────────────────────────────────────

def source_paths(gym_source=None):
    gym = ROOT if (ROOT / "causal_gym").is_dir() else Path(
        gym_source or os.environ.get("CAUSAL_GYM_SOURCE", ROOT.parent / "Causal-Gymnasium")).resolve()
    if not (gym / "causal_gym").is_dir():
        raise ValueError(f"Missing local environment source: {gym}")
    for path in (gym, ROOT, HERE):
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
    # compiled into the virtual environment; frozen here as a record
    for p in sorted((ROOT / "third_party/nadaraya_watson").rglob("*")):
        if p.is_file() and "build" not in p.relative_to(ROOT).parts and p.suffix in (".cpp", ".h", ".hpp", ".py", ".toml"):
            files[str(p.relative_to(ROOT))] = p
    for name in ("run.py", "config.json", "experiments.py"):
        files["experiments/task8/" + name] = HERE / name
    return files


def runtime_versions():
    names = ("numpy", "gymnasium", "torch", "tensorboard", "greenlet", "pygame", "minigrid",
             "scipy", "scikit-learn", "networkx", "nadaraya_watson", "imageio", "moviepy")
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def git_state(path):
    """Commit, branch and uncommitted files of the checkout the sources were frozen from."""
    def git(*args):
        out = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted": status.splitlines() if status else []}


def verify_snapshot(output, check_runtime=True):
    output = Path(output)
    if not (output / "manifest.json").exists():
        raise ValueError(f"No Task 8 manifest in {output}; run prepare first")
    saved = read_json(output / "manifest.json")
    payload = {k: saved.get(k) for k in ("task", "mode", "config", "sources", "runtime")}
    if saved.get("task") != TASK or digest(payload) != saved.get("identity"):
        raise ValueError("Not a valid Task 8 manifest")
    if check_runtime and runtime_versions() != saved["runtime"]:
        raise ValueError("Runtime changed; restore the saved Python/library versions")
    for name, expected in saved["sources"].items():
        if hashlib.sha256((output / "snapshot" / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source changed: " + name)
    if read_json(output / "config.json") != saved["config"]:
        raise ValueError("Saved Task 8 configuration changed")
    return saved


def freeze(output, cfg, mode, gym):
    output = Path(output)
    if (output / "manifest.json").exists():
        saved = verify_snapshot(output)
        if saved["config"] != cfg or saved["mode"] != mode:
            raise ValueError("Output belongs to a different protocol; choose a new directory")
        return saved
    if any(p.name != ".task8.lock" for p in output.iterdir()):
        raise ValueError("Output is nonempty without a Task 8 manifest; refusing to overwrite")
    hashes = {}
    for name, path in source_files(gym).items():
        content = path.read_bytes()
        atomic_write(output / "snapshot" / name, lambda f, b=content: f.write(b), "wb")
        hashes[name] = hashlib.sha256(content).hexdigest()
    payload = {"task": TASK, "mode": mode, "config": cfg, "sources": hashes, "runtime": runtime_versions()}
    saved = {**payload, "identity": digest(payload),
             "source_directories": {"rl": str(ROOT), "gym": str(gym)},
             "git": {"rl": git_state(ROOT), "gym": git_state(gym)},
             "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "seed_design": seed_design()}
    write_json(output / "config.json", cfg)
    write_json(output / "manifest.json", saved)
    return saved


def seed_design():
    return ("As the original run.py: torch.manual_seed, np.random.seed and random.seed are set to the "
            "job seed; the training and evaluation sub-environments get "
            "SeedSequence([seed, evaluation]).generate_state(num_envs). The same seed is shared by "
            "every curriculum and variant, so all of them see the same evaluation tasks.")


# ── Jobs ─────────────────────────────────────────────────────────────────────

def jobs(cfg):
    out = []
    for env in cfg["environments"]:
        for curriculum, variant, seed in itertools.product(cfg["curricula"], cfg["variants"], cfg["seeds"]):
            if curriculum == "default" and variant != "original":
                continue  # the default curriculum samples only target tasks: no causal variant
            out.append(dict(env=env["id"], curriculum=curriculum, variant=variant, seed=seed,
                            steps=cfg["total_timesteps"], cost=COST[env["id"]]))
    for job in out:
        job["key"] = f"{job['env']}/{job['curriculum']}-{job['variant']}/seed_{job['seed']}"
    out.sort(key=lambda j: -j["cost"])
    return out


def job_dir(output, key):
    return Path(output) / "jobs" / key


def job_identity(saved, job):
    return digest({"experiment": saved["identity"],
                   **{k: job[k] for k in ("env", "curriculum", "variant", "seed", "steps")}})


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


# ── One training run ─────────────────────────────────────────────────────────

def run_args(cfg, job, base_log_dir):
    """The argparse namespace of the original run.py for this job (its post-processing included)."""
    ppo = cfg["ppo"]
    env = next(e for e in cfg["environments"] if e["id"] == job["env"])
    args = types.SimpleNamespace(
        base_log_dir=str(base_log_dir), type=job["curriculum"], env=job["env"], seed=job["seed"],
        device=cfg["device"], device_id="0", learning_rate=env["learning_rate"],
        target_learning_rate=ppo["target_learning_rate"], total_timesteps=cfg["total_timesteps"],
        torch_deterministic=cfg["torch_deterministic"], debug=False,
        num_envs=ppo["num_envs"], num_steps=ppo["num_steps"], num_minibatches=ppo["num_minibatches"],
        update_epochs=ppo["update_epochs"], causal=job["variant"] == "causal",
        allow_target_task=cfg["allow_target_task"], eval_dir=None, eval_last=15,
        log_interval=cfg["evaluation"]["log_interval"], anneal_lr=ppo["anneal_lr"], gae=ppo["gae"],
        gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"], norm_adv=ppo["norm_adv"],
        clip_coef=ppo["clip_coef"], clip_vloss=ppo["clip_vloss"], ent_coef=ppo["ent_coef"],
        vf_coef=ppo["vf_coef"], max_grad_norm=ppo["max_grad_norm"], target_kl=ppo["target_kl"])
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def experiment_class(env_id):
    import experiments
    return getattr(experiments, EXPERIMENT_CLASS[env_id])


def train(args):
    """The original run.py train(), without its timestamped log directory."""
    import random
    import numpy as np
    import torch
    exp = experiment_class(args.env)(args, {})
    if args.torch_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(args.torch_deterministic)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    exp.train()
    return exp


def export_scalars(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = sorted(Path(run_dir).glob("events.out.tfevents.*"))
    if len(events) != 1:
        raise RuntimeError(f"expected one TensorBoard event file in {run_dir}, found {len(events)}")
    acc = EventAccumulator(str(events[0]), size_guidance={"scalars": 0, "tensors": 0, "images": 0})
    acc.Reload()
    rows = {}
    for tag in acc.Tags()["scalars"]:
        for e in acc.Scalars(tag):
            rows.setdefault(e.step, {"step": e.step})[tag] = e.value
    return [rows[s] for s in sorted(rows)]


def run_worker(output, key):
    import torch
    torch.set_num_threads(1)
    output = Path(output)
    saved = verify_snapshot(output, check_runtime=True)
    cfg = saved["config"]
    job = next(j for j in jobs(cfg) if j["key"] == key)
    folder = job_dir(output, key)
    identity = job_identity(saved, job)
    for stale in ("logs", "scalars.csv.gz", "contexts.json.gz", "result.json"):
        p = folder / stale
        shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    base = {k: job[k] for k in ("key", "env", "curriculum", "variant", "seed", "steps")}
    write_json(folder / "status.json", {**base, "identity": identity, "status": "running",
                                        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    start = time.monotonic()
    try:
        args = run_args(cfg, job, folder / "logs")
        exp = train(args)
        run_dir = Path(exp.get_log_dir())
        contexts = json.dumps(exp.train_contexts).encode()
        atomic_write(folder / "contexts.json.gz", lambda f: f.write(gzip.compress(contexts)), "wb")
        checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
        if "all" not in cfg["checkpoints"]["keep"]:
            for p in checkpoints:
                if p.name != "final.pt":
                    p.unlink()
        scalars = export_scalars(run_dir)
        write_csv(folder / "scalars.csv.gz", scalars, compress=True)
        evals = [r for r in scalars if "Evaluate/avg_target_return" in r]
        seconds = time.monotonic() - start
        result = {**base, "identity": identity, "seconds": seconds, "run_dir": str(run_dir.relative_to(folder)),
                  "updates": cfg["total_timesteps"] // args.batch_size, "evaluations": len(evals),
                  "last_eval_step": evals[-1]["step"] if evals else None,
                  "last_eval_return": evals[-1]["Evaluate/avg_target_return"] if evals else None,
                  "checkpoints_written": len(checkpoints),
                  "causal_bounds": [exp.LOWER_CONTEXT_BOUNDS.tolist(), exp.UPPER_CONTEXT_BOUNDS.tolist()]}
        write_json(folder / "result.json", result)
        write_json(folder / "status.json", {**base, "identity": identity, "status": "complete", "seconds": seconds})
        return 0
    except Exception as exc:
        write_json(folder / "status.json", {**base, "identity": identity, "status": "failed",
                                            "error": repr(exc), "traceback": traceback.format_exc()})
        raise


# ── Preflight ────────────────────────────────────────────────────────────────

def preflight(cfg, steps=40):
    """Every environment and job type: the bounds, the environments the teachers drive, and do-only access."""
    import numpy as np
    report = []
    for env in cfg["environments"]:
        cls = experiment_class(env["id"])
        for curriculum, variant in itertools.product(cfg["curricula"], cfg["variants"]):
            if curriculum == "default" and variant != "original":
                continue
            job = dict(env=env["id"], curriculum=curriculum, variant=variant, seed=cfg["seeds"][0])
            exp = cls(run_args(cfg, job, "/nonexistent"), {})
            if variant == "causal":
                lower, upper = PAPER_CAUSAL_BOUNDS[env["id"]]
                np.testing.assert_array_equal(exp.LOWER_CONTEXT_BOUNDS, lower)
                np.testing.assert_array_equal(exp.UPPER_CONTEXT_BOUNDS, upper)
            for evaluation in (False, True):
                e = exp.make_env(evaluation, 0)()
                if curriculum == "plr" and not evaluation:
                    # PLR scores levels with the learner's values (create_experiment sets the agent)
                    e.unwrapped.learner = types.SimpleNamespace(estimate_value=lambda x: np.zeros(len(x)))
                obs, _ = e.reset()
                assert obs.shape == (84, 84, 3) and obs.dtype == np.uint8, (env["id"], obs.shape, obs.dtype)
                for t in range(steps):
                    obs, r, term, trunc, _ = e.step(e.action_space.sample())
                    assert e.observation_space.contains(obs)
                    if term or trunc:
                        e.reset()
                e.close()
        native = cls(run_args(cfg, dict(env=env["id"], curriculum="default", variant="original",
                                        seed=0), "/nonexistent"), {}).make_native()
        native.reset(seed=0)
        try:
            native.step(0)
            raise AssertionError(f"{env['id']}: the native PCH accepted step(); it must allow do() only")
        except PermissionError:
            pass
        report.append({"environment": env["id"], "causal_bounds": PAPER_CAUSAL_BOUNDS[env["id"]],
                       "do_only": True, "steps_per_job_type": steps})
    return report


# ── Dispatcher (as in Task 6) ────────────────────────────────────────────────

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
                and (not args.curricula or r["curriculum"] in args.curricula)
                and (not args.variants or r["variant"] in args.variants)
                and (not args.seeds or r["seed"] in args.seeds)]
    minimum = cfg["disk"]["run_min_free_gib"]
    snapshot_run = output / "snapshot/experiments/task8/run.py"
    print(f"TASK 8: {len(selected)} jobs, {args.workers} worker processes (1 Torch thread each); "
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
        base = {k: job[k] for k in ("env", "curriculum", "variant", "seed", "steps")}
        results.append({**base, **{k: res[k] for k in ("seconds", "updates", "evaluations",
                                                       "last_eval_step", "last_eval_return")}})
        with gzip.open(folder / "scalars.csv.gz", "rt") as f:
            for r in csv.DictReader(f):
                curves.append({**base, "step": int(r["step"]),
                               "eval_return": r.get("Evaluate/avg_target_return", ""),
                               "train_return": r.get("Train/avg_epi_return", ""),
                               "num_episodes": r.get("Train/num_episodes", ""),
                               "misaligned": r.get("Train/misaligned_cnt", "")})
    write_csv(output / "results.csv", results)
    write_csv(output / "curves.csv", curves)
    print(f"SUMMARIZED {len(results)} completed jobs -> {output / 'results.csv'}", flush=True)


# ── CLI (as in Task 6) ───────────────────────────────────────────────────────

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
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--envs", nargs="+")
            p.add_argument("--curricula", nargs="+")
            p.add_argument("--variants", nargs="+")
            p.add_argument("--seeds", nargs="+", type=int)
        if command == "worker":
            p.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    output = (args.output or ROOT / "runs/task8" / args.mode).resolve()
    target = output / "snapshot/experiments/task8/run.py"

    if args.command == "prepare":
        cfg = configuration(args.config, args.mode)
        gym = source_paths(args.gym_source)
        if (output / "manifest.json").exists():
            verify_snapshot(output)
        elif output.exists() and any(p.name != ".task8.lock" for p in output.iterdir()):
            raise ValueError("Nonempty unrelated output directory; refusing to write Task 8 files")
        output.mkdir(parents=True, exist_ok=True)
        need = cfg["disk"]["prepare_min_free_gib"]
        if free_gib(output) < need:
            raise ValueError(f"Only {free_gib(output):.1f} GiB free; Task 8 requires at least {need} GiB")
        with (output / ".task8.lock").open("a") as lock:
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
    with (output / ".task8.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return dispatch_run(output, saved, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise SystemExit(1)
