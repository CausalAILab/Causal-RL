"""Measure a SCIL expert's success rate on the fixed (launch-check) environment.

usage: python check_experts.py ENV [--episodes 200] [--stream expert_dev] [--out runs/task7/experts/check]
Writes <out>/<ENV>_<stream>.json with per-episode results.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

ROOT = HERE.parents[1]
FILES = C.EXPERT_FILES
SCIL_SR = {"antmed": 0.876, "antlarge": 0.559, "hummed": 0.338, "humlarge": 0.070}  # SCIL test notebooks, 1000 eps

p = argparse.ArgumentParser()
p.add_argument("env", choices=C.ENV_NAMES)
p.add_argument("--episodes", type=int, default=200)
p.add_argument("--stream", default="expert_dev")
p.add_argument("--legacy", action="store_true", help="original env behavior (no launch-check flags)")
p.add_argument("--only", choices=["seed", "success"], help="enable only one launch-check flag")
p.add_argument("--out", default=str(ROOT / "runs/task7/experts/check"))
a = p.parse_args()
torch.set_num_threads(1)

path = ROOT / "runs/task7/experts/scil" / FILES[a.env]
pi, ck = C.load_expert(path)
seeds = [C.seed_of(a.stream, C.spec(a.env)["index"], 0, i) for i in range(a.episodes)]
flags = None if a.only is None else {"seed_base_reset": a.only == "seed", "success_after_step": a.only == "success"}
env = C.make_env(a.env, "expert", seed=seeds[0], legacy=a.legacy, flags=flags)
H = C.spec(a.env)["H"]
t0 = time.time()
eps = []
with torch.no_grad():
    for i, s in enumerate(seeds):
        eps.append(C.run_episode(env, lambda o, t: pi(o), s, H))
        if (i + 1) % 25 == 0:
            sr = sum(e["success"] for e in eps) / len(eps)
            print(f"{a.env} {i + 1}/{a.episodes} SR={sr:.3f} {(time.time() - t0) / (i + 1):.1f}s/ep", flush=True)
res = C.summarize_episodes(eps, H)
res.update(legacy_env=a.legacy, only_flag=a.only, env=a.env, stream=a.stream, expert_file=FILES[a.env], expert_sha256=C.sha256(path),
           expert_lookback=ck.get("lookback"), scil_reported_sr=SCIL_SR[a.env], seconds=round(time.time() - t0, 1),
           seeds=[seeds[0], seeds[-1]])
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / f"{a.env}_{a.stream}{'_legacy' if a.legacy else ''}{'_only-' + a.only if a.only else ''}_{a.episodes}.json").write_text(json.dumps(res, indent=1))
print(f"{a.env}: SR {res['success_rate']:.3f} (SCIL {SCIL_SR[a.env]:.3f}), return {res['return_mean']:.1f}, "
      f"length {res['length_mean']:.0f}, {res['seconds']:.0f}s", flush=True)
