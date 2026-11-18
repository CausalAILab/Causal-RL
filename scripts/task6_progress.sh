#!/usr/bin/env bash
# Read-only live view of a Task 6 run: overall counts plus each running job's progress.
# Usage: scripts/task6_progress.sh [refresh_seconds] [output_dir]   (Ctrl-C to quit)
set -uo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
every="${1:-30}"
out="${2:-$root/runs/task6/main}"
cfg="$out/config.json"
[[ -f "$cfg" ]] || { echo "No Task 6 experiment at $out" >&2; exit 1; }

while true; do
  clear
  date '+%Y-%m-%d %H:%M:%S'
  python3 - "$out" <<'EOF'
import json, sys, time
from collections import Counter
from pathlib import Path
out = Path(sys.argv[1]); cfg = json.loads((out / "config.json").read_text())
steps = {e["id"]: cfg["schedules"][e["schedule"]]["steps"] for e in cfg["environments"]}
planned = (len(cfg["seeds"]) * (sum(len(cfg["modes"]) * len(cfg["algorithms"]) for e in cfg["environments"] if e["enabled"])
           + len(cfg["random_hint"]["algorithms"]) * len(cfg["random_hint"]["environments"])
           + len(cfg["references"]) * sum(e["enabled"] for e in cfg["environments"]) + 2))
counts, running, done_seconds = Counter(), [], []
for status in out.glob("jobs/*/*/*/status.json"):
    try:
        s = json.loads(status.read_text())
    except (OSError, ValueError):
        continue
    counts[s["status"]] += 1
    if s["status"] == "complete" and s.get("seconds"):
        done_seconds.append(s["seconds"])
    if s["status"] == "running":
        log, step = status.parent / "train.log", 0
        if log.exists():
            with log.open("rb") as f:
                f.seek(0, 2); f.seek(max(0, f.tell() - 4096))
                for line in f.read().decode(errors="ignore").splitlines():
                    if line.startswith("global_step="):
                        step = int(line.split(",")[0].split("=")[1])
        total = s.get("steps") or steps.get(s["env"], 0)
        running.append((s["key"], step, total))
complete, failed = counts["complete"], counts["failed"]
print(f"complete {complete}/{planned}   running {len(running)}   failed {failed}   "
      f"pending {planned - complete - len(running) - failed}")
if failed:
    print("  failed jobs: see jobs/<key>/status.json -> error / traceback")
print()
for key, step, total in sorted(running):
    pct = 100 * step / total if total else 0
    bar = "#" * int(pct / 5)
    print(f"  {key:44s} {step:>9,}/{total:<9,} {pct:5.1f}% {bar}")
EOF
  free=$(df -h "$out" | awk 'NR==2{print $4}')
  echo; echo "free disk: $free    (refresh every ${every}s, Ctrl-C to quit)"
  sleep "$every"
done
