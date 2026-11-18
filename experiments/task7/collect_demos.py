"""B0 step 3: collect the frozen SCIL expert's demonstrations for one environment.

usage: python collect_demos.py ENV [--episodes 1000] [--train 800] [--out runs/task7/demos]
Writes <out>/<ENV>/{train,val}.npz (compact arrays, see common.save_episodes) and manifest.json.
The split is by episode index, fixed before any encoding: episodes 0..train-1 train, the rest validation.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
FILES = C.EXPERT_FILES

ROOT = HERE.parents[1]

p = argparse.ArgumentParser()
p.add_argument("env", choices=C.ENV_NAMES)
p.add_argument("--episodes", type=int, default=1000)
p.add_argument("--train", type=int, default=800)
p.add_argument("--out", default=str(ROOT / "runs/task7/demos"))
a = p.parse_args()
torch.set_num_threads(1)

expert_path = ROOT / "runs/task7/experts/scil" / FILES[a.env]
pi, ck = C.load_expert(expert_path)
s = C.spec(a.env)
seeds = [C.seed_of("demo", s["index"], 0, i) for i in range(a.episodes)]
env = C.make_env(a.env, "expert", seed=seeds[0])
t0 = time.time()
eps = []
with torch.no_grad():
    for i, sd in enumerate(seeds):
        eps.append(C.run_episode(env, lambda o, t: pi(o), sd, s["H"]))
        if (i + 1) % 100 == 0:
            print(f"{a.env} {i + 1}/{a.episodes} SR={np.mean([e['success'] for e in eps]):.3f} "
                  f"{(time.time() - t0) / (i + 1):.2f}s/ep", flush=True)

out = Path(a.out) / a.env
meta = {"env": a.env, "env_id": s["env_id"], "H": s["H"], "radius": s["radius"], "expert_file": FILES[a.env],
        "expert_sha256": C.sha256(expert_path), "seed_stream": "demo", "first_seed": seeds[0], "last_seed": seeds[-1],
        "role": "expert", "flags": {"seed_base_reset": True, "success_after_step": True, "playback": False}}
manifest = dict(meta, episodes=a.episodes, collect_seconds=round(time.time() - t0, 1))
for part, idx in (("train", range(0, a.train)), ("val", range(a.train, a.episodes))):
    sub = [eps[i] for i in idx]
    sha = C.save_episodes(out / f"{part}.npz", a.env, sub, dict(meta, part=part, episode_index=[idx.start, idx.stop]))
    manifest[part] = {"episodes": len(sub), "transitions": int(sum(e["length"] for e in sub)),
                      "success_rate": float(np.mean([e["success"] for e in sub])), "sha256": sha}
manifest["success_rate"] = float(np.mean([e["success"] for e in eps]))
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps({k: manifest[k] for k in ("env", "success_rate", "train", "val")}), flush=True)
