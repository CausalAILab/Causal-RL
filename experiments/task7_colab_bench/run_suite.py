"""Run the Task 7 speed suite and write results.jsonl plus a summary with projected run times.

usage: python run_suite.py --device cuda --out RESULTS_DIR [--parallel 1,4,8] [--quick]
Each line of results.jsonl is one process's measurement; parallel tests share a start barrier.
"""
import argparse, json, os, platform, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda")
p.add_argument("--out", required=True)
p.add_argument("--parallel", default="1,4,8")
p.add_argument("--quick", action="store_true", help="only the Ant-Medium SQIL/IQ tests")
args = p.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
res_file = out / "results.jsonl"
levels = [int(x) for x in args.parallel.split(",")]

single = [("antmed", "sqil", "causal"), ("antmed", "iq", "causal"), ("antmed", "iq", "naive")]
if not args.quick:
    single += [("antmed", "gail", "causal"), ("hummed", "sqil", "causal"), ("hummed", "iq", "causal"),
               ("hummed", "gail", "causal"), ("antmed", "eval", "causal"), ("hummed", "eval", "causal")]
parallel = [("antmed", "iq", "causal"), ("antmed", "sqil", "causal")] + ([] if args.quick else [("hummed", "sqil", "causal")])


def run(test, n, device):
    env, algo, enc = test
    with tempfile.TemporaryDirectory() as bar:
        envv = dict(os.environ, BENCH_BARRIER_DIR=bar, BENCH_BARRIER_N=str(n), PYTHONWARNINGS="ignore",
                    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        if sys.platform.startswith("linux"): envv.setdefault("MUJOCO_GL", "egl")
        cmd = [sys.executable, str(HERE / "bench_task7.py"), env, algo, enc, "--device", device]
        t = time.time()
        logs = [(open(os.path.join(bar, f"o{i}"), "w+"), open(os.path.join(bar, f"e{i}"), "w+")) for i in range(n)]
        procs = [subprocess.Popen(cmd, stdout=o, stderr=e, text=True, env=envv) for o, e in logs]
        rows = []
        for pr, (o, e) in zip(procs, logs):
            pr.wait()
            o.seek(0); e.seek(0); so, se = o.read(), e.read(); o.close(); e.close()
            line = [l for l in so.splitlines() if l.startswith("{")]
            if line:
                rows.append(json.loads(line[-1]))
            else:
                rows.append({"env": env, "algo": algo, "enc": enc, "device": device, "parallel": n, "error": se[-1500:]})
        for r in rows:
            r["wall_s"] = round(time.time() - t, 1)
        with res_file.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        ok = [r for r in rows if "error" not in r]
        key = "update_ms" if algo in ("sqil", "iq") else ("ms_per_eval_step" if algo == "eval" else "ms_per_interaction")
        vals = sorted(r[key] for r in ok)
        print(f"{env:7s} {algo:5s} {enc:6s} {device:5s} x{n:<2d} {key}={vals}  "
              f"{'hours_2M=' + str(ok[0].get('hours_2M')) if ok and 'hours_2M' in ok[0] else ''}"
              f"{'  FAILED ' + str(len(rows) - len(ok)) if len(ok) < len(rows) else ''}", flush=True)
        if len(ok) < len(rows):
            print(rows[[i for i, r in enumerate(rows) if 'error' in r][0]]["error"][-800:], flush=True)


print(f"python {platform.python_version()}  cpus {os.cpu_count()}  device {args.device}", flush=True)
start = time.time()
for t in single:
    run(t, 1, args.device)
for n in levels:
    if n > 1:
        for t in parallel:
            run(t, n, args.device)
print(f"suite finished in {(time.time() - start) / 60:.1f} min; results in {res_file}", flush=True)
