#!/usr/bin/env bash
# Read-only live view of a Task 7 run: counts plus each running job's progress.
# Usage: scripts/task7_progress.sh [refresh_seconds] [mode_or_output_dir]   (Ctrl-C quits the viewer only)
set -uo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
every="${1:-30}"
target="${2:-dev}"
case "$target" in /*) out="$target" ;; *) out="$root/runs/task7/$target" ;; esac
[[ -f "$out/manifest.json" ]] || { echo "No Task 7 experiment at $out" >&2; exit 1; }
while true; do
  clear
  date '+%Y-%m-%d %H:%M:%S'
  python3 - "$out" <<'PY'
import json, sys, time, re, calendar
from collections import Counter
from pathlib import Path
out = Path(sys.argv[1])
m = json.loads((out / "manifest.json").read_text()); cfg = m["config"]
planned = len(cfg["envs"]) * len(cfg["algorithms"]) * len(cfg["encodings"]) * len(cfg["seeds"]) \
    + sum(len(c["algorithms"]) for e, c in cfg["controls"].items() if e in cfg["envs"]) * len(cfg["seeds"])
counts, running = Counter(), []
for st in out.glob("jobs/*/*/*/status.json"):
    try: s = json.loads(st.read_text())
    except Exception: continue
    counts[s["status"]] += 1
    if s["status"] == "running":
        line = ""
        log = st.parent / "train.log"
        if log.exists():
            with log.open("rb") as f:
                f.seek(0, 2); f.seek(max(0, f.tell() - 4096))
                for l in f.read().decode(errors="ignore").splitlines():
                    if l.startswith("progress "): line = l
        started = calendar.timegm(time.strptime(s["started_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        running.append((s["key"], line, (time.time() - started) / 3600))
c, f = counts["complete"], counts["failed"]
print(f"[{m['mode']}] complete {c}/{planned}   running {len(running)}   failed {f}   "
      f"pending {planned - c - f - len(running)}")
if f: print("  failed jobs: see jobs/<key>/status.json -> error / traceback")
print()
for key, line, hrs in sorted(running):
    kv = dict(x.split("=", 1) for x in line.split()[1:]) if line else {}
    if kv:
        done, tot = float(kv["done"]), float(kv["total"]); pct = 100 * done / tot
        eta = hrs / done * (tot - done) if done else float("nan")
        print(f"  {key:30s} {int(done):>9,}/{int(tot):<9,} {kv['unit']:12s} {pct:5.1f}% {'#' * int(pct / 5):20s} "
              f"{hrs:5.1f} h, ~{eta:4.1f} h left")
    else:
        print(f"  {key:30s} starting (loading data / filling expert buffer) {hrs:5.1f} h")
PY
  free=$(df -h "$out" | awk 'NR==2{print $4}')
  echo; echo "free disk: $free    (refresh every ${every}s, Ctrl-C to quit)"
  sleep "$every"
done
