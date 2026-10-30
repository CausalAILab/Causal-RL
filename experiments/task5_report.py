#!/usr/bin/env python3
"""Task 5: raw seed scores, percentile CIs, fixed-suite stratified summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task5 as training

METHODS = training.METHODS
LABELS = ("Full", "MIS", "POMIS", "All-at-once")
COLORS = ("#eb896d", "#c54b66", "#72356f", "#347f9d")
METRICS = ("cumulative_pseudo_regret", "own_set_regret", "candidate_set_value_gap",
           "simple_regret", "own_set_simple_regret", "optimal_arm_fraction",
           "observed_mean_reward", "normalized_score")


def iqm(values):
    ordered = np.sort(values, axis=-1)
    trim = ordered.shape[-1] // 4
    return ordered[..., trim:ordered.shape[-1]-trim].mean(axis=-1)


def probability_of_improvement(x, y):
    """All cross-run pairs within each environment; ties contribute one half."""
    wins = x[..., :, None] > y[..., None, :]
    ties = x[..., :, None] == y[..., None, :]
    return (wins + .5 * ties).mean(axis=(-3, -2, -1))


def stratified_samples(scores, reps, rng, batch=100):
    # Input [method, environment, run]; the same sampled run IDs for every method.
    # This preserves the common-exogenous-noise pairing, without changing the
    # all-cross-run-pair probability-of-improvement estimand.
    _, environments, runs = scores.shape
    for start in range(0, reps, batch):
        ids = rng.integers(runs, size=(min(batch, reps-start), environments, runs))
        yield scores[:, np.arange(environments)[None, :, None], ids]


def mean_ci(values, reps, rng):
    """Input [method, run, ...]; one run index resamples its complete vector."""
    draws = []
    for start in range(0, reps, 200):
        ids = rng.integers(values.shape[1], size=(min(200, reps-start), values.shape[1]))
        draws.append(values[:, ids].mean(axis=2))
    bounds = np.quantile(np.concatenate(draws, axis=1), [.025, .975], axis=1)
    return values.mean(axis=1), bounds[0], bounds[1]


def suite_metrics(scores):
    flat = scores.reshape(scores.shape[0], -1)
    return np.stack([iqm(flat), np.maximum(1-flat, 0).mean(axis=-1)], axis=-1)


def suite_ci(scores, reps, rng):
    draws = []
    for sample in stratified_samples(scores, reps, rng):
        flat = sample.reshape(sample.shape[0], sample.shape[1], -1)
        draws.append(np.stack([iqm(flat), np.maximum(1-flat, 0).mean(axis=-1)], axis=-1))
    bounds = np.quantile(np.concatenate(draws, axis=1), [.025, .975], axis=1)
    return suite_metrics(scores), bounds[0], bounds[1]


def profile_ci(scores, thresholds, reps, rng):
    point = (scores[..., None] > thresholds).mean(axis=(1, 2))
    draws = []
    for sample in stratified_samples(scores, reps, rng, batch=25):
        draws.append((sample[..., None] > thresholds).mean(axis=(2, 3)))
    bounds = np.quantile(np.concatenate(draws, axis=1), [.025, .975], axis=1)
    return point, bounds[0], bounds[1]


def probability_ci(scores, reps, rng):
    n = len(scores)
    point = np.array([[probability_of_improvement(scores[i], scores[j])
                       for j in range(n)] for i in range(n)])
    draws = []
    for sample in stratified_samples(scores, reps, rng):
        draws.append(np.stack([np.stack([probability_of_improvement(sample[i], sample[j])
                                         for j in range(n)], axis=-1) for i in range(n)], axis=-2))
    bounds = np.quantile(np.concatenate(draws, axis=0), [.025, .975], axis=0)
    return point, bounds[0], bounds[1]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment_label(spec, latex=False):
    if spec["kind"] == "example":
        return "Example 9.5"
    if spec["kind"] == "six_variable":
        return "Six-variable"
    d = f"$d={spec['d']}$" if latex else f"d={spec['d']}"
    return f"{spec['structure'].title()} ({d})"


def read_results(output):
    saved = training.read_json(output / "manifest.json")
    cfg = saved["config"]
    training.source_paths()
    from causal_rl.algo.ctf_do.ctfucb import UCB
    plans, results, missing, rows = {}, {}, [], []
    hashes = training.read_json(output / "preflight.json")["environment_hashes"]
    for spec in cfg["environments"]:
        env_id = spec["id"]
        plan = training.read_json(output / "environments" / (env_id + ".json"))
        if training.digest(plan) != hashes[env_id]:
            raise ValueError(f"Environment metadata changed: {env_id}")
        plans[env_id] = plan
        for method, seed in itertools.product(METHODS, cfg["seeds"]):
            directory = training.job_path(output, env_id, method, seed)
            try:
                status = training.read_json(directory / "status.json")
                if status["status"] != "complete":
                    raise ValueError(status.get("reason", status["status"]))
                identity = training.digest({"experiment": saved["identity"], "environment": hashes[env_id],
                                            "method": method, "seed": seed})
                agent = UCB(plan["candidates"][method], alpha=cfg["alpha"])
                actions, rewards, episode_seeds, recommendations = training.load_checkpoint(
                    directory / "checkpoint.npz", identity, agent, np.random.default_rng(), spec["budget"])
                if len(actions) != spec["budget"] or status["identity"] != identity:
                    raise ValueError("Completed job has inconsistent budget/identity")
                metrics = training.metrics_for_trace(plan, method, np.array(actions), np.array(rewards),
                                                      recommendations, cfg)
                if [m["steps"] for m in metrics] != spec["checkpoints"]:
                    raise ValueError("Missing checkpoint recommendations")
                entry = {"actions": np.array(actions), "metrics": {m["steps"]: m for m in metrics}}
                results[(env_id, method, seed)] = entry
                for metrics_row in metrics:
                    rows.append({"environment": env_id, "method": method, "seed": seed,
                                 "candidate_count": len(plan["candidates"][method]), **metrics_row})
            except (OSError, ValueError, KeyError) as exc:
                missing.append({"environment": env_id, "method": method, "seed": seed, "reason": str(exc)})
    return saved, plans, results, missing, rows


def endpoint_tables(report, cfg, results, rng_seed, reps):
    rows = []
    for env_index, spec in enumerate(cfg["environments"]):
        for step in sorted({cfg["common_steps"], spec["budget"]}):
            values = np.array([[[results[(spec["id"], method, seed)]["metrics"][step][metric]
                                 for metric in METRICS] for seed in cfg["seeds"]] for method in METHODS])
            point, low, high = mean_ci(values, reps, np.random.default_rng(
                np.random.SeedSequence([rng_seed, env_index, step, 1])))
            # Paired raw regret differences versus Full, resampling the same run IDs.
            diff = values[:, :, 0] - values[0, :, 0]
            dp, dl, dh = mean_ci(diff, reps, np.random.default_rng(
                np.random.SeedSequence([rng_seed, env_index, step, 2])))
            for i, method in enumerate(METHODS):
                row = {"environment": spec["id"], "steps": step, "method": method,
                       "completed_runs": len(cfg["seeds"])}
                for k, metric in enumerate(METRICS):
                    row.update({metric: point[i, k], metric+"_ci_low": low[i, k], metric+"_ci_high": high[i, k]})
                row.update({"regret_difference_vs_full": dp[i], "regret_difference_ci_low": dl[i],
                            "regret_difference_ci_high": dh[i]})
                rows.append(row)
    write_csv(report / "endpoints.csv", rows)
    lookup = {(r["environment"], r["method"], r["steps"]): r for r in rows}
    lines = ["% Requires booktabs and graphicx. Mean [95% percentile bootstrap CI].",
             r"\begin{table*}[t]", r"\centering\small",
             r"\caption{Task 5 cumulative pseudo-regret to the full legal optimum. "
             + f"{len(cfg['seeds'])} runs per cell; mean [95\\% CI]." + "}",
             r"\label{tab:task5-regret}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{lrrrr}\toprule", "Environment & " + " & ".join(LABELS) + r"\\\midrule"]
    groups = [(f"Common checkpoint: {cfg['common_steps']:,} interventions", cfg["environments"], cfg["common_steps"])]
    extra = [e for e in cfg["environments"] if e["budget"] != cfg["common_steps"]]
    if extra:
        groups.append(("Extended endpoints", extra, None))
    for title, specs, common in groups:
        lines.append(r"\multicolumn{5}{l}{" + title + r"}\\")
        for spec in specs:
            step = common or spec["budget"]
            cells = []
            for method in METHODS:
                record = lookup[(spec["id"], method, step)]
                cells.append(f"{record['cumulative_pseudo_regret']:.2f} "
                             f"[{record['cumulative_pseudo_regret_ci_low']:.2f}, {record['cumulative_pseudo_regret_ci_high']:.2f}]")
            label = environment_label(spec, latex=True)
            if common is None:
                label += f", {step//1000}k"
            lines.append(label + " & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table*}"]
    (report / "endpoints.tex").write_text("\n".join(lines)+"\n")


def learning_curves(report, cfg, plans, results, reps, seed, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), constrained_layout=True)
    for env_index, spec in enumerate(cfg["environments"]):
        ax = axes.flat[env_index]
        plan = plans[spec["id"]]
        x = np.unique(np.linspace(1, spec["budget"], min(201, spec["budget"]), dtype=int))
        curves = np.array([[np.cumsum(np.maximum(
            plan["optimum"] - np.asarray(plan["means"])[results[(spec["id"], method, s)]["actions"]], 0))[x-1]
            for s in cfg["seeds"]] for method in METHODS])
        point, low, high = mean_ci(curves, reps, np.random.default_rng(
            np.random.SeedSequence([seed, env_index, 3])))
        for i, (label, color) in enumerate(zip(LABELS, COLORS)):
            for curve in curves[i]:
                ax.plot(x, curve, color=color, alpha=.10, lw=.55)
            ax.fill_between(x, low[i], high[i], color=color, alpha=.18, linewidth=0)
            ax.plot(x, point[i], color=color, label=label, lw=1.6)
        ax.set_title(environment_label(spec))
        ax.set_xlabel("Online interventions")
        ax.set_ylabel("Cumulative pseudo-regret")
        ax.grid(alpha=.15)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle(title + ": mean and pointwise 95% bootstrap CI; faint lines are runs")
    for extension in ("pdf", "png"):
        fig.savefig(report / ("learning_curves."+extension), dpi=180)
    plt.close(fig)


def aggregate_report(report, cfg, results, repetitions, title):
    import matplotlib.pyplot as plt
    score = np.array([[[results[(e["id"], method, s)]["metrics"][cfg["common_steps"]]["normalized_score"]
                        for s in cfg["seeds"]] for e in cfg["environments"]] for method in METHODS])
    seed = cfg["bootstrap"]["seed"]
    point, low, high = suite_ci(score, repetitions["aggregate_reps"], np.random.default_rng(seed))
    suite_rows = []
    for i, method in enumerate(METHODS):
        for k, metric in enumerate(("iqm", "optimality_gap")):
            suite_rows.append({"method": method, "metric": metric, "score": point[i, k],
                               "ci_low": low[i, k], "ci_high": high[i, k],
                               "steps": cfg["common_steps"], "environments": score.shape[1],
                               "runs_per_environment": score.shape[2]})
    write_csv(report / "suite.csv", suite_rows)
    lines = [r"\begin{tabular}{lrr}\toprule", r"Method & IQM $\uparrow$ & Gap to 1 $\downarrow$\\\midrule"]
    for i, label in enumerate(LABELS):
        cells = [f"{point[i,k]:.3f} [{low[i,k]:.3f}, {high[i,k]:.3f}]" for k in range(2)]
        lines.append(label + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular}"]
    (report / "suite.tex").write_text("\n".join(lines)+"\n")

    thresholds = np.linspace(0, 1, 101)
    pp, pl, ph = profile_ci(score, thresholds, repetitions["profile_reps"], np.random.default_rng(seed+1))
    write_csv(report / "profiles.csv", [
        {"method": method, "threshold": threshold, "fraction_above": pp[i, j],
         "ci_low": pl[i, j], "ci_high": ph[i, j]}
        for i, method in enumerate(METHODS) for j, threshold in enumerate(thresholds)])
    pi, pil, pih = probability_ci(score, repetitions["probability_reps"], np.random.default_rng(seed+2))
    write_csv(report / "probability_of_improvement.csv", [
        {"method_x": x, "method_y": y, "probability_x_better": pi[i, j],
         "ci_low": pil[i, j], "ci_high": pih[i, j]}
        for i, x in enumerate(METHODS) for j, y in enumerate(METHODS)])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        axes[0].plot(thresholds, pp[i], color=color, label=label)
        axes[0].fill_between(thresholds, pl[i], ph[i], color=color, alpha=.18)
        for k in range(2):
            axes[k+1].hlines(i, low[i, k], high[i, k], color=color, lw=1.5)
            axes[k+1].plot(point[i, k], i, "o", color=color)
    axes[0].set(xlabel="Normalized score threshold", ylabel="Fraction of runs above threshold", ylim=(0, 1.02))
    axes[0].legend(frameon=False, fontsize=8)
    for k, label in enumerate(("IQM (higher is better)", "Gap to 1 (lower is better)"), 1):
        axes[k].set(xlabel=label, yticks=range(4), yticklabels=LABELS)
        axes[k].invert_yaxis()
    fig.suptitle(title + f": eleven fixed environments at {cfg['common_steps']:,} interventions")
    for extension in ("pdf", "png"):
        fig.savefig(report / ("suite."+extension), dpi=180)
    plt.close(fig)


def generate(output, quick=False):
    training.verify_snapshot(output)
    saved, plans, results, missing, rows = read_results(output)
    cfg = saved["config"]
    diagnostic = quick or saved["mode"] != "main"
    report = output / "reports" / ("diagnostic" if diagnostic else "main")
    report.mkdir(parents=True, exist_ok=True)
    write_csv(report / "per_seed.csv", rows)
    expected = len(cfg["environments"])*len(METHODS)*len(cfg["seeds"])
    repetitions = dict(cfg["bootstrap"])
    if diagnostic:
        repetitions.update(aggregate_reps=200, curve_reps=100, profile_reps=100, probability_reps=100)
    metadata = {"experiment_identity": saved["identity"], "mode": saved["mode"],
                "diagnostic_only": diagnostic, "completed_runs": len(results), "planned_runs": expected,
                "missing_runs": missing, "bootstrap": repetitions, "ci": "95% percentile",
                "fixed_suite": True, "paired_run_ids": True,
                "normalization": "1 - cumulative_regret / (T * (full_optimum - full_minimum)); no clipping",
                "probability_of_improvement": "all cross-run pairs per environment; ties=0.5",
                "evaluation_interactions": 0,
                "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    training.write_json(report / "report_manifest.json", metadata)
    if missing:
        write_csv(report / "incomplete_jobs.csv", missing)
        # Avoid exposing a stale full-suite summary after an input became unavailable.
        for filename in ("endpoints.csv", "endpoints.tex", "suite.csv", "suite.tex", "profiles.csv",
                         "probability_of_improvement.csv", "learning_curves.pdf", "learning_curves.png",
                         "suite.pdf", "suite.png"):
            (report / filename).unlink(missing_ok=True)
        print(f"{len(results)}/{expected} complete; full-suite tables/figures withheld. See {report}")
        return 2
    title = "Task 5" if not diagnostic else f"Task 5 DIAGNOSTIC ({saved['mode']}; not main results)"
    endpoint_tables(report, cfg, results, cfg["bootstrap"]["seed"], repetitions["aggregate_reps"])
    learning_curves(report, cfg, plans, results, repetitions["curve_reps"], cfg["bootstrap"]["seed"], title)
    aggregate_report(report, cfg, results, repetitions, title)
    print(f"Report written: {report}; {len(results)}/{expected} complete; diagnostic_only={diagnostic}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    return generate(args.output.resolve(), args.quick)


if __name__ == "__main__":
    sys.exit(main())
