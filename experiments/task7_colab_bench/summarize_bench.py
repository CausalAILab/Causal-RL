"""Turn results.jsonl into per-run hours and a per-batch (2 seeds) projection for the SQIL/IQ-Learn block.

usage: python summarize_bench.py RESULTS_DIR [--cu-per-hour X] [--ram-gb Y]
Large mazes are assumed to cost the same as Medium (same networks and encodings).
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("results"); p.add_argument("--cu-per-hour", type=float, default=0.0); p.add_argument("--ram-gb", type=float, default=0.0)
a = p.parse_args()
rows = [json.loads(l) for l in (Path(a.results) / "results.jsonl").read_text().splitlines() if l.strip()]
bad = [r for r in rows if "error" in r]
rows = [r for r in rows if "error" not in r]
dev = rows[0]["device"] if rows else "?"
print(f"device {dev}  gpu {rows[0].get('gpu') if rows else None}  ok {len(rows)}  failed {len(bad)}")

h = defaultdict(list)  # (env, algo, parallel) -> hours per 2M run
for r in rows:
    if "hours_2M" in r:
        h[(r["env"], r["algo"], r["parallel"])].append(r["hours_2M"])
mean = {k: sum(v) / len(v) for k, v in h.items()}
print("\nHours per 2M-interaction run (mean over processes):")
for k in sorted(mean):
    print(f"  {k[0]:7s} {k[1]:5s} x{k[2]:<2d} {mean[k]:6.2f} h")
for r in rows:
    if r["algo"] == "gail":
        print(f"  {r['env']:7s} gail  x1  full schedule (worst case) {r['hours_full_schedule_worst']:.2f} h")
    if r["algo"] == "eval":
        print(f"  {r['env']:7s} eval  {r['ms_per_eval_step']:.2f} ms per evaluation step")

# Per-batch off-policy block: 2 seeds x 2 mazes x 2 encodings per (env family, algo) + Ant-Med W-mask controls.
runs = {("ant", "sqil"): 8 + 2, ("ant", "iq"): 8 + 2, ("hum", "sqil"): 8, ("hum", "iq"): 8}
key_env = {"ant": "antmed", "hum": "hummed"}
levels = sorted({k[2] for k in mean})


def hours(fam, algo, n):
    e = key_env[fam]
    if (e, algo, n) in mean: return mean[(e, algo, n)]
    if (e, algo, 1) not in mean: return math.nan
    # slowdown not measured for this algo: borrow the same family's SQIL (or Ant) slowdown ratio
    for ref in ((e, "sqil"), ("antmed", algo), ("antmed", "sqil")):
        if (*ref, n) in mean and (*ref, 1) in mean:
            return mean[(e, algo, 1)] * mean[(*ref, n)] / mean[(*ref, 1)]
    return math.nan


rss = max((r["peak_rss_MB"] for r in rows if r["algo"] in ("sqil", "iq")), default=0) / 1024
replay = {"ant": 8.0, "hum": 11.0}  # GB at full 1M+ replay with the library's list-of-tensors buffer (local measurement)
print("\nPer batch (2 seeds) SQIL + IQ-Learn block, incl. W-mask controls:")
for n in levels:
    runtime_h = sum(cnt * hours(f, al, n) / n for (f, al), cnt in runs.items())
    longest = max(hours(f, al, n) for (f, al) in runs)
    line = f"  {n} runs at once: {runtime_h:6.1f} runtime-hours, longest single run {longest:5.1f} h"
    if a.cu_per_hour: line += f", ~{runtime_h * a.cu_per_hour:.0f} CU"
    print(line)
if a.ram_gb:
    for f in ("ant", "hum"):
        need = rss + replay[f]
        print(f"  RAM: {f} run needs ~{need:.1f} GB at full replay -> at most {int((a.ram_gb - 4) // need)} at once on {a.ram_gb:.0f} GB")
print("\nLocal M2 Pro reference (single process): Ant SQIL 4.8 h, Ant IQ 20 h, Hum SQIL 8.6 h, Hum IQ 10.6 h;"
      " a local batch of this block takes about 6 days at 2-3 runs at once.")
