#!/usr/bin/env bash
# Read-only live view of a Task 8 run: counts, each running job's progress, and a rough ETA.
# Usage: scripts/task8_progress.sh [refresh_seconds] [mode_or_output_dir]   (Ctrl-C quits the viewer only)
set -uo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
every="${1:-30}"
target="${2:-main}"
case "$target" in /*) out="$target" ;; *) out="$root/runs/task8/$target" ;; esac
[[ -f "$out/manifest.json" ]] || { echo "No Task 8 experiment at $out" >&2; exit 1; }
while true; do
  clear
  date '+%Y-%m-%d %H:%M:%S'
  python3 - "$out" <<'PY'
import calendar, json, re, sys, time
from collections import Counter
from pathlib import Path
out = Path(sys.argv[1])
m = json.loads((out / "manifest.json").read_text()); cfg = m["config"]
per_env = sum(len(cfg["variants"]) if c != "default" else 1 for c in cfg["curricula"])
planned = len(cfg["environments"]) * per_env * len(cfg["seeds"])
total = cfg["total_timesteps"]
# printed by PPOInterface.train() after each in-training evaluation (every log_interval updates)
EVAL = re.compile(r"^(\d+)/(\d+) - evaluate return: (\S+)")
counts, running, done = Counter(), [], []
for st in out.glob("jobs/*/*/*/status.json"):
    try:
        s = json.loads(st.read_text())
    except (OSError, ValueError):
        continue
    counts[s["status"]] += 1
    if s["status"] == "complete" and s.get("seconds"):
        done.append((s["env"], s["seconds"]))
    if s["status"] == "running":
        step, ret, log = 0, "", st.parent / "train.log"
        # checkpoints '<update>-<step>.pt' are written every log_interval updates (pruned at the end),
        # so they show progress even when the worker's stdout is still buffered
        for p in st.parent.glob("logs/*/*/*/seed-*/checkpoints/*-*.pt"):
            try:
                step = max(step, int(p.stem.split("-")[1]))
            except (IndexError, ValueError):
                pass
        if log.exists():
            with log.open("rb") as f:
                f.seek(0, 2); f.seek(max(0, f.tell() - 8192))
                for line in f.read().decode(errors="ignore").splitlines():
                    hit = EVAL.match(line.strip())
                    if hit:
                        step, ret = max(step, int(hit.group(1))), f"{float(hit.group(3)):.3f}"
        elapsed = time.time() - calendar.timegm(time.strptime(s["started_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        running.append((s["key"], step, ret, elapsed))
c, f = counts["complete"], counts["failed"]
pending = planned - c - f - len(running)
print(f"[{m['mode']}] complete {c}/{planned}   running {len(running)}   failed {f}   pending {pending}")
if f:
    print("  failed jobs: see jobs/<key>/status.json -> error / traceback")
if (out / "PAUSE").exists():
    print("  PAUSED: running jobs finish, no new jobs start (scripts/task8.sh clear-pause, then run)")
print()
for key, step, ret, elapsed in sorted(running):
    pct = 100 * step / total
    eta = f"~{elapsed * (total - step) / step / 60:4.0f} min left" if step else "  first evaluation pending"
    print(f"  {key:42s} {step:>7,}/{total:,} {pct:5.1f}% {'#' * int(pct / 5):20s} "
          f"{elapsed / 60:5.0f} min {eta}  eval {ret}")
if done:
    by_env = {}
    for env, sec in done:
        by_env.setdefault(env, []).append(sec)
    mean = sum(sec for _, sec in done) / len(done)
    print()
    print("  mean job time: " + ", ".join(f"{e} {sum(v) / len(v) / 3600:.2f} h ({len(v)})" for e, v in sorted(by_env.items())))
    if running:
        left = (pending + len(running) / 2) * mean / len(running)
        print(f"  rough time left for the batch: {left / 3600:.1f} h at {len(running)} parallel jobs")
PY
  free=$(df -h "$out" | awk 'NR==2{print $4}')
  echo; echo "free disk: $free    (refresh every ${every}s, Ctrl-C quits the viewer only)"
  sleep "$every"
done
