#!/usr/bin/env python3
"""Task 7 runner: causal imitation (BC, GAIL, SQIL, IQ-Learn x causal/vanilla inputs, plus the Ant-Medium
W-mask control) on the four SCIL mazes, organised in seed batches B1-B5.

Commands
  prepare      freeze sources, config, expert checkpoints and demonstrations into <output>; run preflight
  run          train/evaluate pending jobs; select a batch with --batch B1 or --seeds; memory-aware scheduling
  status       count job states (optionally for one batch)
  summarize    collect per-run results into results.csv / curves.csv
  pause        stop launching new jobs (running jobs finish); clear-pause undoes it
  worker       (internal) one job in its own process

A job is one complete run (environment x method x seed): training, the secondary checkpoint and the final
evaluation banks. Jobs are never resumed: an interrupted job restarts from scratch with the same seeds.
The identity of an output directory does not depend on which batches run there, so batches can be split
between machines and their jobs/ folders merged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import importlib.metadata
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

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
TASK = 7
GIB = 1024 ** 3
# Repository root: two levels up from the live file, or the snapshot root inside an output directory.
ROOT = HERE.parents[1]


# ── Utilities ────────────────────────────────────────────────────────────────

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in keys} for r in rows)


def free_gib(path):
    return shutil.disk_usage(path).free / GIB


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def platform_info():
    import torch
    return {"host": platform.node(), "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "processor": platform.processor(), "cpus": os.cpu_count(),
            "python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
            "causal_gym": str(Path(sys.modules["causal_gym"].__file__).parent) if "causal_gym" in sys.modules else None,
            "causal_rl": str(Path(sys.modules["causal_rl"].__file__).parent) if "causal_rl" in sys.modules else None}


def runtime_versions():
    out = {"python": platform.python_version()}
    for n in ("numpy", "torch", "gymnasium", "mujoco", "ogbench", "networkx", "ananke-causal"):
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


# ── Configuration and sources ────────────────────────────────────────────────

def configuration(path, mode):
    cfg = read_json(path)
    cfg["mode"] = mode
    if mode in ("dev", "smoke"):
        m = cfg["modes"][mode]
        cfg["seeds"], cfg["envs"] = m["seeds"], m["envs"]
        cfg["batches"] = {"B0": m["seeds"]}
        cfg["controls"] = {k: v for k, v in cfg["controls"].items() if k in cfg["envs"]}
        cfg["budget"].update(m.get("budget_overrides", {}))
    if mode == "smoke":
        m = cfg["modes"]["smoke"]
        cfg["budget"] = {"offpolicy": m["budget_offpolicy"], "offpolicy_secondary": m["secondary_offpolicy"],
                         "gail_secondary": m["gail_secondary"]}
        cfg["evaluation"].update(checkpoint_every=m["checkpoint_every"], checkpoint_episodes=m["checkpoint_episodes"],
                                 final_c1={"default": m["final_episodes"]}, final_c0={"default": m["final_episodes"]},
                                 secondary={"default": m["final_episodes"]})
        cfg["demos"] = {"train_episodes": m["demo_episodes"][0], "val_episodes": m["demo_episodes"][1]}
    return cfg


def gym_root(explicit=None):
    gym = Path(explicit or os.environ.get("CAUSAL_GYM_SOURCE", ROOT.parent / "Causal-Gymnasium")).resolve()
    if (ROOT / "causal_gym").is_dir():  # inside a snapshot
        gym = ROOT
    if not (gym / "causal_gym").is_dir():
        raise ValueError(f"Missing Causal-Gymnasium source: {gym}")
    return gym


def use_sources(gym):
    for p in (gym, ROOT, HERE):
        if str(p) in sys.path:
            sys.path.remove(str(p))
        sys.path.insert(0, str(p))


def source_files(gym):
    files = {}
    for package, parent in (("causal_rl", ROOT), ("causal_gym", gym)):
        for p in sorted((parent / package).rglob("*.py")):
            files[str(p.relative_to(parent))] = p
    for p in sorted((gym / "causal_gym/envs/assets").rglob("*")):
        if p.is_file():
            files[str(p.relative_to(gym))] = p
    for name in ("run.py", "common.py", "learners.py", "config.json"):
        files["experiments/task7/" + name] = HERE / name
    return files


def data_files(cfg, data_root):
    """Expert checkpoints and demonstration files the experiment depends on (copied into <output>/data)."""
    import common as C
    out = {}
    for env in cfg["envs"]:
        out[f"experts/{C.EXPERT_FILES[env]}"] = data_root / "experts/scil" / C.EXPERT_FILES[env]
        for part in ("train", "val", "manifest"):
            ext = "json" if part == "manifest" else "npz"
            out[f"demos/{env}/{part}.{ext}"] = data_root / "demos" / env / f"{part}.{ext}"
    return out


def verify(output, check_runtime=True, check_data=False):
    output = Path(output)
    if not (output / "manifest.json").exists():
        raise ValueError(f"No Task 7 manifest in {output}; run prepare first")
    saved = read_json(output / "manifest.json")
    payload = {k: saved.get(k) for k in ("task", "mode", "config", "sources", "data", "runtime")}
    if saved.get("task") != TASK or digest(payload) != saved.get("identity"):
        raise ValueError("Not a valid Task 7 manifest")
    if check_runtime and runtime_versions() != saved["runtime"]:
        raise ValueError(f"Runtime changed: {runtime_versions()} != {saved['runtime']}")
    for name, expected in saved["sources"].items():
        if sha256_file(output / "snapshot" / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    if check_data:
        for name, expected in saved["data"].items():
            if sha256_file(output / "data" / name) != expected:
                raise ValueError("Frozen data changed: " + name)
    return saved


def freeze(output, cfg, gym, data_root):
    output = Path(output)
    if (output / "manifest.json").exists():
        saved = verify(output, check_data=True)
        if saved["config"] != cfg:
            raise ValueError("Output belongs to a different configuration; choose a new directory")
        return saved
    if any(p.name != ".task7.lock" for p in output.iterdir()):
        raise ValueError("Output is nonempty without a Task 7 manifest; refusing to overwrite")
    sources = {}
    for name, path in source_files(gym).items():
        dst = output / "snapshot" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        sources[name] = sha256_file(dst)
    data = {}
    for name, path in data_files(cfg, data_root).items():
        if not path.exists():
            raise ValueError(f"Missing {path}; extract the experts and run collect_demos.py first")
        dst = output / "data" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, dst)
        except OSError:
            shutil.copy2(path, dst)
        data[name] = sha256_file(dst)
    payload = {"task": TASK, "mode": cfg["mode"], "config": cfg, "sources": sources, "data": data,
               "runtime": runtime_versions()}
    saved = {**payload, "identity": digest(payload), "created_utc": now(),
             "source_directories": {"rl": str(ROOT), "gym": str(gym), "data": str(data_root)},
             "seed_design": "common.seed_of(stream, env_index, run_seed, item): streams train_env/torch/"
                            "checkpoint_eval/final_eval/secondary_eval are disjoint integer ranges; evaluation episode "
                            "IDs are shared by all methods and both c values within (env, seed), disjoint across seeds."}
    write_json(output / "manifest.json", saved)
    return saved


# ── Jobs ─────────────────────────────────────────────────────────────────────

def jobs(cfg):
    out = []
    family = {"antmed": "ant", "antlarge": "ant", "hummed": "hum", "humlarge": "hum"}
    sched = cfg["scheduler"]
    for env in cfg["envs"]:
        methods = [(a, e) for a in cfg["algorithms"] for e in cfg["encodings"]]
        ctrl = cfg["controls"].get(env)
        if ctrl:
            methods += [(a, ctrl["encoding"]) for a in ctrl["algorithms"]]
        for algo, enc in methods:
            for seed in cfg["seeds"]:
                out.append({"key": f"{env}/{algo}-{enc}/seed_{seed}", "env": env, "algorithm": algo, "encoding": enc,
                            "seed": seed, "memory_gb": sched["memory_estimate_gb"][env][algo],
                            "hours": sched["hours_estimate"][family[env]][algo]})
    out.sort(key=lambda j: (-j["hours"], j["key"]))
    return out


def batch_of(cfg, seed):
    return next((b for b, seeds in cfg["batches"].items() if seed in seeds), None)


def job_identity(saved, job):
    return digest({"experiment": saved["identity"], **{k: job[k] for k in ("env", "algorithm", "encoding", "seed")}})


def statuses(output, saved, seeds=None):
    counts, rows = Counter(), []
    for job in jobs(saved["config"]):
        if seeds and job["seed"] not in seeds:
            continue
        path = Path(output) / "jobs" / job["key"] / "status.json"
        state = read_json(path) if path.exists() else {"status": "pending"}
        if state.get("status") != "pending" and state.get("identity") != job_identity(saved, job):
            state = {"status": "foreign"}
        counts[state["status"]] += 1
        rows.append({**job, **state})
    return dict(counts), rows


# ── Worker: one complete run ─────────────────────────────────────────────────

def episodes_for(cfg, block, env):
    table = cfg["evaluation"][block]
    return table.get(env, table["default"])


def run_job(output, saved, job, folder):
    import numpy as np
    import torch
    import common as C
    import learners as Lr
    torch.set_num_threads(1)
    cfg = saved["config"]
    env, algo, enc, seed = job["env"], job["algorithm"], job["encoding"], job["seed"]
    s = C.spec(env)
    demos_tr = C.Demos(output / "data/demos" / env / "train.npz", env)
    demos_va = C.Demos(output / "data/demos" / env / "val.npz", env)
    ntr, nva = cfg["demos"]["train_episodes"], cfg["demos"]["val_episodes"]
    assert len(demos_tr) >= ntr and len(demos_va) >= nva
    train_idx, val_idx = range(ntr), range(nva)
    evals, curves = {}, []

    def bank(stream, n, c, act, modules):
        seeds = [C.seed_of(stream, s["index"], seed, i) for i in range(n)]
        r = C.evaluate(env, act, seeds, c=c, modules=modules)
        return r

    def on_checkpoint(tag, interactions, count, modules, act):
        torch.save({"state_dict": modules[0].state_dict(), "interactions": interactions, "count": count,
                    "encoder": None if algo == "bc" else enc}, folder / f"{tag}_policy.pt")
        if tag == "final":
            n1, n0 = episodes_for(cfg, "final_c1", env), episodes_for(cfg, "final_c0", env)
            evals["final"] = {"interactions": interactions, "count": count,
                              "c1": bank("final_eval", n1, 1.0, act, modules),
                              "c0": bank("final_eval", n0, 0.0, act, modules)}
        else:
            n = episodes_for(cfg, "secondary", env)
            evals["secondary"] = {"interactions": interactions, "count": count,
                                  "c1": bank("secondary_eval", n, 1.0, act, modules),
                                  "c0": bank("secondary_eval", n, 0.0, act, modules)}
        write_json(folder / "evals.json", evals)

    ev = cfg["evaluation"]
    smoke = cfg["mode"] == "smoke"
    m = cfg["modes"]["smoke"]
    if algo in ("sqil", "iq"):
        encoder = C.Encoder(env, enc)
        p = dict(Lr.OFFPOLICY, **({"start_steps": m["start_steps"]} if smoke else {}))
        budget = cfg["budget"].get(f"{algo}_offpolicy", cfg["budget"]["offpolicy"])
        secondary = cfg["budget"]["offpolicy_secondary"]
        _, _, log = Lr.train_offpolicy(
            algo, env, encoder, demos_tr, train_idx, seed, budget=budget,
            secondary=secondary if secondary < budget else None, schedule_budget=cfg["budget"]["offpolicy"],
            eval_every=ev["checkpoint_every"], eval_episodes=ev["checkpoint_episodes"], on_checkpoint=on_checkpoint,
            on_eval=curves.append, p=p)
        extra = {"interactions": log["interactions"], "updates": log["updates"], "z_dim": encoder.dim}
        episodes = log["episodes"]
    elif algo == "gail":
        encoder = C.Encoder(env, enc)
        _, _, log = Lr.train_gail(
            env, encoder, demos_tr, train_idx, seed, rounds=m["gail_rounds"] if smoke else None,
            episodes_per_round=m["gail_episodes"] if smoke else None,
            secondary_interactions=cfg["budget"]["gail_secondary"], eval_every=ev["checkpoint_every"],
            eval_episodes=ev["checkpoint_episodes"], on_checkpoint=on_checkpoint, on_eval=curves.append)
        extra = {"interactions": log["interactions"], "rounds": len(log["rounds"]), "z_dim": encoder.dim}
        episodes = log["rounds"]
    else:
        _, _, log = Lr.train_bc(env, enc, demos_tr, train_idx, val_idx, seed,
                                epochs=m["bc_epochs"] if smoke else None, on_checkpoint=on_checkpoint)
        extra = {"interactions": 0, "epochs": log["stopped_epoch"], "best_val": log["best_val"],
                 "train_samples": log["train_samples"], "val_samples": log["val_samples"]}
        episodes = log["epochs"]
    write_csv(folder / "curve.csv", curves)
    with open(folder / "train_log.json", "w") as f:
        json.dump(episodes, f)
    return {**extra, "final": {c: {k: v for k, v in evals["final"][c].items() if k != "per_episode"}
                               for c in ("c0", "c1")},
            "secondary": {c: {k: v for k, v in evals["secondary"][c].items() if k != "per_episode"}
                          for c in ("c0", "c1")} if "secondary" in evals else None}


def run_worker(output, key):
    output = Path(output)
    saved = verify(output)
    job = next(j for j in jobs(saved["config"]) if j["key"] == key)
    folder = output / "jobs" / key
    for stale in ("final_policy.pt", "secondary_policy.pt", "evals.json", "curve.csv", "train_log.json", "result.json"):
        (folder / stale).unlink(missing_ok=True)
    identity = job_identity(saved, job)
    base = {k: job[k] for k in ("key", "env", "algorithm", "encoding", "seed")}
    base["batch"] = batch_of(saved["config"], job["seed"])
    write_json(folder / "status.json", {**base, "identity": identity, "status": "running", "started_utc": now(),
                                        "platform": platform_info()})
    start = time.monotonic()
    try:
        result = run_job(output, saved, job, folder)
        import common as C
        seconds = time.monotonic() - start
        extra = {"seconds": round(seconds, 1), "peak_rss_mb": round(C.peak_rss_mb()), "platform": platform_info()}
        write_json(folder / "result.json", {**base, **result, **extra, "identity": identity})
        write_json(folder / "status.json", {**base, **extra, "identity": identity, "status": "complete",
                                            "finished_utc": now()})
        return 0
    except BaseException as exc:
        write_json(folder / "status.json", {**base, "identity": identity, "status": "failed", "error": repr(exc),
                                            "traceback": traceback.format_exc(), "platform": platform_info()})
        raise


# ── Preflight ────────────────────────────────────────────────────────────────

def preflight(output, cfg):
    import numpy as np
    import common as C
    report = {}
    for env in cfg["envs"]:
        dims = {e: C.Encoder(env, e).dim for e in ("causal", "vanilla", "wmask")}
        mf = read_json(output / "data/demos" / env / "manifest.json")
        tr = C.Demos(output / "data/demos" / env / "train.npz", env)
        va = C.Demos(output / "data/demos" / env / "val.npz", env)
        hidden = C.spec(env)["hidden"]
        assert hidden not in tr.obs(0) and "U" not in tr.obs(0) and "reward" not in tr.obs(0)
        assert len(tr) >= cfg["demos"]["train_episodes"] and len(va) >= cfg["demos"]["val_episodes"]
        assert mf["expert_sha256"] == read_json(output / "manifest.json")["data"][f"experts/{C.EXPERT_FILES[env]}"]
        enc = C.Encoder(env, "vanilla")
        z = enc(tr.obs(0), int(tr.lengths[0]) - 1)
        assert np.all(np.isfinite(z))
        report[env] = {"z_dim": dims, "demo_success_rate": mf["success_rate"],
                       "train_transitions": tr.n_transitions(range(cfg["demos"]["train_episodes"])),
                       "expert_sha256": mf["expert_sha256"]}
    return report


# ── Scheduler ────────────────────────────────────────────────────────────────

def dispatch(output, saved, args):
    cfg = saved["config"]
    if (output / "PAUSE").exists():
        raise ValueError("Experiment is paused; run clear-pause first")
    seeds = selected_seeds(cfg, args)
    _, rows = statuses(output, saved, seeds)
    if any(r["status"] == "foreign" for r in rows):
        raise ValueError("A job folder belongs to another experiment identity; inspect before running")
    pending = [r for r in rows if r["status"] != "complete" and (not args.envs or r["env"] in args.envs)
               and (not args.algorithms or r["algorithm"] in args.algorithms)]
    mem_budget, workers = args.memory_gb, args.workers
    snapshot_run = output / "snapshot/experiments/task7/run.py"
    print(f"TASK 7 [{saved['mode']}] seeds {seeds}: {len(pending)} jobs, up to {workers} processes, "
          f"memory budget {mem_budget} GB, free disk {free_gib(output):.1f} GiB", flush=True)
    running, counts, done_n = {}, Counter(), 0
    previous = {s: signal.signal(s, lambda *_: (output / "PAUSE").touch()) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        while pending or running:
            stop = (output / "PAUSE").exists()
            if free_gib(output) < cfg["disk"]["run_min_free_gib"] and not stop:
                (output / "PAUSE").touch()
                print("PAUSED: low disk", flush=True)
                stop = True
            used = sum(r["memory_gb"] for r in running.values())
            launched = False
            if not stop and len(running) < workers:
                for job in pending:  # longest first; skip jobs that do not fit the remaining memory
                    if used + job["memory_gb"] <= mem_budget or not running:
                        folder = output / "jobs" / job["key"]
                        folder.mkdir(parents=True, exist_ok=True)
                        log = open(folder / "train.log", "w")
                        proc = subprocess.Popen([sys.executable, str(snapshot_run), "worker", "--output", str(output),
                                                 "--job", job["key"]], stdout=log, stderr=subprocess.STDOUT,
                                                start_new_session=True, cwd=str(folder))
                        running[proc] = {**job, "log": log, "t0": time.time()}
                        pending.remove(job)
                        print(f"START {job['key']} (~{job['memory_gb']} GB)", flush=True)
                        launched = True
                        break
            if stop and not running:
                break
            if stop:
                pending = []
            for proc in [p for p in running if p.poll() is not None]:
                job = running.pop(proc)
                job["log"].close()
                state = "complete" if proc.returncode == 0 else "failed"
                counts[state] += 1
                done_n += 1
                print(f"{state.upper()} {job['key']} in {(time.time() - job['t0']) / 3600:.2f} h", flush=True)
            if not launched:
                time.sleep(args.poll)
    finally:
        for s, h in previous.items():
            signal.signal(s, h)
    totals, _ = statuses(output, saved, seeds)
    print(json.dumps(totals, sort_keys=True), flush=True)
    return 1 if counts["failed"] else 0


def selected_seeds(cfg, args):
    if getattr(args, "batch", None):
        return list(cfg["batches"][args.batch])
    if getattr(args, "seeds", None):
        return list(args.seeds)
    return list(cfg["seeds"])


def summarize(output, saved):
    results, curves = [], []
    for job in jobs(saved["config"]):
        folder = Path(output) / "jobs" / job["key"]
        if not (folder / "result.json").exists():
            continue
        r = read_json(folder / "result.json")
        row = {k: r[k] for k in ("env", "algorithm", "encoding", "seed", "batch", "interactions", "seconds",
                                 "peak_rss_mb") if k in r}
        row["host"] = r["platform"]["host"]
        row["system"] = r["platform"]["system"]
        for block in ("final", "secondary"):
            for c in ("c0", "c1"):
                for k in ("success_rate", "return_mean", "timeout_rate", "capped_time_to_goal", "episodes"):
                    if r.get(block):
                        row[f"{block}_{c}_{k}"] = r[block][c][k]
        results.append(row)
        if (folder / "curve.csv").exists():
            with open(folder / "curve.csv") as f:
                curves += [{**{k: job[k] for k in ("env", "algorithm", "encoding", "seed")}, **x}
                           for x in csv.DictReader(f)]
    write_csv(Path(output) / "results.csv", results)
    write_csv(Path(output) / "curves.csv", curves)
    print(f"SUMMARIZED {len(results)} completed jobs -> {Path(output) / 'results.csv'}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "status", "summarize", "pause", "clear-pause", "worker"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path)
        p.add_argument("--mode", choices=("main", "dev", "smoke"), default="main")
        if command == "prepare":
            p.add_argument("--config", type=Path, default=HERE / "config.json")
            p.add_argument("--gym-source", type=Path)
            p.add_argument("--data-root", type=Path, default=ROOT / "runs/task7")
        if command in ("run", "status"):
            p.add_argument("--batch")
            p.add_argument("--seeds", nargs="+", type=int)
        if command == "run":
            p.add_argument("--workers", type=int, default=None)
            p.add_argument("--memory-gb", type=float, default=None)
            p.add_argument("--envs", nargs="+")
            p.add_argument("--algorithms", nargs="+")
            p.add_argument("--poll", type=float, default=10.0)
        if command == "worker":
            p.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    output = (args.output or ROOT / "runs/task7" / args.mode).resolve()
    target = output / "snapshot/experiments/task7/run.py"

    if args.command == "prepare":
        cfg = configuration(args.config, args.mode)
        gym = gym_root(args.gym_source)
        use_sources(gym)
        output.mkdir(parents=True, exist_ok=True)
        if free_gib(output) < cfg["disk"]["prepare_min_free_gib"]:
            raise ValueError(f"Only {free_gib(output):.1f} GiB free")
        with (output / ".task7.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            saved = freeze(output, cfg, gym, args.data_root.resolve())
        report = preflight(output, saved["config"])
        write_json(output / "preflight.json", {"identity": saved["identity"], "checks": report})
        counts, rows = statuses(output, saved)
        print(f"PREPARED {args.mode}: {len(rows)} jobs, identity {saved['identity'][:12]}; preflight passed for "
              f"{len(report)} environments; no training started: {output}", flush=True)
        return 0

    saved = verify(output)
    if Path(__file__).resolve() != target:  # always execute the frozen copy
        forwarded = list(argv if argv is not None else sys.argv[1:])
        if "--output" not in forwarded:
            forwarded += ["--output", str(output)]
        return subprocess.call([sys.executable, str(target), *forwarded])
    use_sources(ROOT)
    if args.command == "worker":
        return run_worker(output, args.job)
    if args.command == "status":
        counts, rows = statuses(output, saved, selected_seeds(saved["config"], args))
        print(json.dumps({"mode": saved["mode"], "planned": len(rows), "counts": counts}, indent=2))
        return 0
    if args.command == "summarize":
        summarize(output, saved)
        return 0
    if args.command in ("pause", "clear-pause"):
        (output / "PAUSE").touch() if args.command == "pause" else (output / "PAUSE").unlink(missing_ok=True)
        print(args.command, output)
        return 0
    verify(output, check_data=True)
    sched = saved["config"]["scheduler"]
    args.workers = args.workers or sched["max_workers"]
    args.memory_gb = args.memory_gb or sched["memory_gb"]
    with (output / ".task7.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return dispatch(output, saved, args)


if __name__ == "__main__":
    sys.exit(main())
