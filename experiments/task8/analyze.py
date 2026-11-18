#!/usr/bin/env python3
"""Task 8 analysis following the Statistical Precipice protocol (same statistics as Task 6).

Reads runs/task8/main (results.csv, curves.csv) and writes tables (CSV/JSON) and figures (PNG)
to --out. Run units are training seeds; the 128 evaluation episodes are averaged inside a run
and never treated as extra samples. The final score of a run is its last in-training
evaluation (update 620, 297,600 steps).

  python experiments/task8/analyze.py --run runs/task8/main --out runs/task8/main/analysis
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEED = 314159
AGG_REPS, CURVE_REPS = 50_000, 2_000
ENVS = ("colored_sokoban", "button_maze")
ENV_LABEL = {"colored_sokoban": "Colored Sokoban", "button_maze": "Button Maze"}
CURRICULA = ("alp_gmm", "plr", "goal_gan", "currot")
CUR_LABEL = {"alp_gmm": "ALP-GMM", "plr": "PLR", "goal_gan": "Goal-GAN", "currot": "CurrOT"}
VAR_LABEL = {"original": "Original", "causal": "Causal", "default": "Target only"}
# Normalization q = (J - l) / (u - l) on the target task (single start state in both envs):
#   u: exact optimal expected return (experiments/task8/optimum.py), by expected-value DP over deep copies of
#      the environment with the hidden colour as a chance node after every step
#      (Button Maze: press the button, then enter the goal; Colored Sokoban: reach the box and
#      push it onto the target on a yellow step). q = 1 is the optimal policy.
#   l: an episode that achieves nothing. Colored Sokoban (20 steps, -0.1 per step): never
#      finishing gives -2.0. Button Maze (30 steps, -0.01 per step, -0.1 for not moving):
#      never moving gives -3.3.
# l is not the absolute minimum (finishing on a blue step, or repeatedly un-pressing the button,
# scores lower), and an evaluation mean can exceed u by chance, so q outside [0, 1] is possible;
# such evaluations are counted, never clipped.
BOUNDS = {"colored_sokoban": (-2.0, 9.103783080726863), "button_maze": (-3.3, 1.31)}

# Reference palette (dataviz skill, light mode), as in Task 6.
C = {"original": "#2a78d6", "causal": "#eb6834", "default": "#8a8984",
     "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}


# ── Data ─────────────────────────────────────────────────────────────────────

def load(run):
    rows = list(csv.DictReader((run / "results.csv").open()))
    for r in rows:
        r["seed"], r["last_eval_step"] = int(r["seed"]), int(r["last_eval_step"])
        r["last_eval_return"] = float(r["last_eval_return"])
    curves = list(csv.DictReader((run / "curves.csv").open()))
    for c in curves:
        c["seed"], c["step"] = int(c["seed"]), int(c["step"])
        for k in ("eval_return", "num_episodes", "misaligned"):
            c[k] = float(c[k]) if c[k] not in ("", None) else None
    return rows, curves


def normalize(x, env):
    lo, hi = BOUNDS[env]
    return (np.asarray(x, dtype=float) - lo) / (hi - lo)


def methods():
    return [("default", "original")] + [(c, v) for c in CURRICULA for v in ("original", "causal")]


def label(cur, var):
    return VAR_LABEL["default"] if cur == "default" else f"{CUR_LABEL[cur]} {VAR_LABEL[var].lower()}"


def final(rows, env, cur, var):
    sel = sorted((r for r in rows if (r["env"], r["curriculum"], r["variant"]) == (env, cur, var)),
                 key=lambda r: r["seed"])
    return np.array([r["seed"] for r in sel]), np.array([r["last_eval_return"] for r in sel])


def eval_matrix(curves, env, cur, var):
    sel = [c for c in curves if (c["env"], c["curriculum"], c["variant"]) == (env, cur, var)
           and c["eval_return"] is not None]
    steps, seeds = sorted({c["step"] for c in sel}), sorted({c["seed"] for c in sel})
    table = {(c["seed"], c["step"]): c["eval_return"] for c in sel}
    return np.array(steps), np.array([[table[(s, t)] for t in steps] for s in seeds])


def misaligned_portion(curves, env, cur, var):
    """Per seed: misaligned training episodes / all training episodes over the whole run."""
    out = {}
    for c in curves:
        if (c["env"], c["curriculum"], c["variant"]) != (env, cur, var) or c["num_episodes"] is None:
            continue
        m, n = out.get(c["seed"], (0.0, 0.0))
        out[c["seed"]] = (m + c["misaligned"], n + c["num_episodes"])
    return np.array([out[s][0] / out[s][1] for s in sorted(out)])


# ── Statistics (identical to experiments/task6/analyze.py) ───────────────────

def ci_mean(x, reps=AGG_REPS, rng=None):
    rng = rng or np.random.default_rng(SEED)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    boot = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def iqm(a, axis=-1):
    a = np.sort(a, axis=axis)
    n = a.shape[axis]
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return np.take(a, range(lo, hi), axis=axis).mean(axis=axis)


def stratified(scores, reps, rng):
    """scores: dict task -> (n_runs,) array. Returns (reps, tasks, n) resampled within tasks."""
    tasks = list(scores)
    n = len(scores[tasks[0]])
    idx = rng.integers(0, n, size=(reps, len(tasks), n))
    mat = np.stack([scores[t] for t in tasks])
    return mat[np.arange(len(tasks))[None, :, None], idx], idx


def aggregate(scores_by_method, reps=AGG_REPS):
    """IQM, mean, optimality gap with stratified percentile bootstrap CIs (same indices for all methods)."""
    rng = np.random.default_rng(SEED)
    tasks = list(next(iter(scores_by_method.values())))
    n = len(next(iter(scores_by_method.values()))[tasks[0]])
    idx = rng.integers(0, n, size=(reps, len(tasks), n))
    out = {}
    for method, scores in scores_by_method.items():
        mat = np.stack([scores[t] for t in tasks])
        boot = mat[np.arange(len(tasks))[None, :, None], idx].reshape(reps, -1)
        flat = mat.reshape(-1)
        res = {}
        for name, fn in (("iqm", lambda a: iqm(a)), ("mean", lambda a: a.mean(-1)),
                         ("optimality_gap", lambda a: np.maximum(0.0, 1.0 - a).mean(-1))):
            b = fn(boot)
            res[name] = (float(fn(flat[None])[0]), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
        out[method] = res
        out[method]["_boot_iqm"] = iqm(boot)
    return out


def prob_improvement(x_scores, y_scores, reps=CURVE_REPS):
    """Average over tasks of P(X > Y) via Mann-Whitney (ties .5); independent stratified bootstrap."""
    rng = np.random.default_rng(SEED)
    tasks = list(x_scores)

    def poi(xs, ys):
        vals = []
        for x, y in zip(xs, ys):
            diff = x[..., :, None] - y[..., None, :]
            vals.append(((diff > 0) + 0.5 * (diff == 0)).mean((-1, -2)))
        return np.mean(vals, axis=0)
    point = float(poi([x_scores[t] for t in tasks], [y_scores[t] for t in tasks]))
    bx, _ = stratified(x_scores, reps, rng)
    by, _ = stratified(y_scores, reps, rng)
    boot = poi([bx[:, i] for i in range(len(tasks))], [by[:, i] for i in range(len(tasks))])
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def profile(scores, taus, reps=CURVE_REPS):
    rng = np.random.default_rng(SEED)
    flat = np.concatenate([scores[t] for t in scores])
    point = (flat[None, :] > taus[:, None]).mean(1)
    boot, _ = stratified(scores, reps, rng)
    boot = boot.reshape(reps, -1)
    b = (boot[:, None, :] > taus[None, :, None]).mean(2)
    return point, np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


def curve_band(mat, reps=CURVE_REPS):
    """mat: (seeds, steps). Pointwise IQM over seeds and its percentile bootstrap band."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, mat.shape[0], size=(reps, mat.shape[0]))
    boot = iqm(mat[idx], axis=1)
    return iqm(mat, axis=0), np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)


def fmt(t, d=3):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


# ── Tables ───────────────────────────────────────────────────────────────────

def cell_table(rows):
    """Per environment and curriculum: final q of original, causal, target only; paired differences."""
    out = []
    for env in ENVS:
        d_seed, d = final(rows, env, "default", "original")
        dq = normalize(d, env)
        for cur in CURRICULA:
            o_seed, o = final(rows, env, cur, "original")
            c_seed, c = final(rows, env, cur, "causal")
            assert (o_seed == c_seed).all() and (o_seed == d_seed).all()
            oq, cq = normalize(o, env), normalize(c, env)
            out.append({"env": env, "curriculum": cur, "n_seeds": len(c),
                        "original_return": ci_mean(o), "causal_return": ci_mean(c), "target_only_return": ci_mean(d),
                        "original_q": ci_mean(oq), "causal_q": ci_mean(cq), "target_only_q": ci_mean(dq),
                        "causal_minus_original_q": ci_mean(cq - oq), "causal_minus_target_only_q": ci_mean(cq - dq),
                        "original_minus_target_only_q": ci_mean(oq - dq),
                        "wins_vs_original": int((c > o).sum()), "wins_vs_target_only": int((c > d).sum())})
    return out


def misaligned_table(curves):
    out = []
    for env in ENVS:
        for cur in ("default",) + CURRICULA:
            for var in (("original",) if cur == "default" else ("original", "causal")):
                p = misaligned_portion(curves, env, cur, var)
                out.append({"env": env, "curriculum": cur, "variant": var, "portion": ci_mean(100 * p),
                            "max_seed": float(100 * p.max())})
    return out


# ── Figures ──────────────────────────────────────────────────────────────────

def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Helvetica Neue", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"],
        "axes.facecolor": C["surface"], "figure.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "text.color": C["ink"],
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2, "savefig.dpi": 200})
    return plt


def fig_aggregate(agg, poi_o, poi_d, path):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), gridspec_kw={"width_ratios": [1.15, 0.9, 0.9]})
    ax = axes[0]
    m, lo, hi = agg[("default", "original")]["iqm"]
    ax.axvspan(lo, hi, color=C["default"], alpha=0.15, lw=0)
    ax.axvline(m, color=C["default"], lw=1.2, ls=(0, (3, 3)), label=VAR_LABEL["default"])
    for i, cur in enumerate(CURRICULA):
        for j, var in enumerate(("original", "causal")):
            m, lo, hi = agg[(cur, var)]["iqm"]
            y = i + (j - 0.5) * 0.32
            ax.plot([lo, hi], [y, y], color=C[var], lw=2, solid_capstyle="round")
            ax.plot(m, y, "o", color=C[var], ms=7, mec=C["surface"], mew=1.5)
    for var in ("original", "causal"):
        ax.plot([], [], "o", color=C[var], label=VAR_LABEL[var])
    ax.set_yticks(range(len(CURRICULA)), [CUR_LABEL[c] for c in CURRICULA])
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("IQM normalized score q")
    ax.set_title("IQM, both environments", loc="left")
    ax.legend(loc="lower left", fontsize=7.5)
    for ax, poi, title in ((axes[1], poi_o, "P(causal > original)"), (axes[2], poi_d, "P(causal > target only)")):
        for i, cur in enumerate(CURRICULA):
            m, lo, hi = poi[cur]
            ax.plot([lo, hi], [i, i], color=C["causal"], lw=2, solid_capstyle="round")
            ax.plot(m, i, "o", color=C["causal"], ms=7, mec=C["surface"], mew=1.5)
        ax.axvline(0.5, color=C["ink2"], lw=1)
        ax.set_yticks(range(len(CURRICULA)), [CUR_LABEL[c] for c in CURRICULA])
        ax.invert_yaxis()
        ax.set_xlim(0.3, 1.02)
        ax.set_xlabel("probability of improvement")
        ax.set_title(title, loc="left")
    fig.text(0.01, 0.005, "Stratified percentile bootstrap over seeds within environments: IQM 50,000 resamples; "
             "probability of improvement 2,000 (independent sampling, Mann–Whitney, ties = .5).",
             fontsize=7.5, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_final_scores(rows, path):
    """Per environment: every seed's final q, with mean and 95% CI; shows the bimodal outcomes."""
    plt = style()
    fig, axes = plt.subplots(1, len(ENVS), figsize=(12.0, 3.8), sharey=True)
    rng = np.random.default_rng(SEED)
    for ax, env in zip(axes, ENVS):
        x = 0
        ticks, labels = [], []
        for cur, var in methods():
            q = normalize(final(rows, env, cur, var)[1], env)
            col = C["default" if cur == "default" else var]
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(q)), q, s=12, color=col, alpha=0.55, lw=0)
            m, lo, hi = ci_mean(q)
            ax.plot([x + 0.25, x + 0.25], [lo, hi], color=col, lw=2, solid_capstyle="round")
            ax.plot(x + 0.25, m, "o", color=col, ms=6, mec=C["surface"], mew=1.2)
            ticks.append(x)
            labels.append("Target\nonly" if cur == "default" else f"{CUR_LABEL[cur]}\n{var}")
            x += 1 if var == "original" and cur != "default" else 1.4
        ax.set_xticks(ticks, labels, fontsize=6.5)
        ax.set_title(ENV_LABEL[env], loc="left")
        ax.axhline(1.0, color=C["ink2"], lw=0.8, ls=(0, (3, 3)))
    axes[0].set_ylabel("final normalized score q")
    fig.text(0.01, 0.005, "Dots: the 10 seeds (final evaluation, 128 episodes each); bars: mean with 95% "
             "percentile bootstrap CI over seeds (50,000).", fontsize=7.5, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_curves(curves, path):
    plt = style()
    fig, axes = plt.subplots(len(ENVS), len(CURRICULA), figsize=(10.5, 5.0), sharex=True, sharey="row")
    for i, env in enumerate(ENVS):
        steps, dmat = eval_matrix(curves, env, "default", "original")
        dm, dlo, dhi = curve_band(normalize(dmat, env))
        for j, cur in enumerate(CURRICULA):
            ax = axes[i, j]
            x = steps / 1e3
            ax.fill_between(x, dlo, dhi, color=C["default"], alpha=0.15, lw=0)
            ax.plot(x, dm, color=C["default"], lw=1.2, ls=(0, (3, 3)), label=VAR_LABEL["default"])
            for var in ("original", "causal"):
                s, mat = eval_matrix(curves, env, cur, var)
                assert (s == steps).all()
                m, lo, hi = curve_band(normalize(mat, env))
                ax.fill_between(x, lo, hi, color=C[var], alpha=0.18, lw=0)
                ax.plot(x, m, color=C[var], lw=1.8, label=VAR_LABEL[var])
            if i == 0:
                ax.set_title(CUR_LABEL[cur], loc="left")
            if j == 0:
                ax.set_ylabel(f"{ENV_LABEL[env]}\nnormalized score q")
            if i == len(ENVS) - 1:
                ax.set_xlabel("environment steps (thousands)")
    axes[1, 0].legend(loc="lower right", fontsize=7.5)
    fig.text(0.01, 0.005, "IQM of 10 seeds (128 evaluation episodes every 4,800 steps); shaded: pointwise 95% "
             "percentile bootstrap bands over seeds (2,000). q = 1: exact optimal policy.",
             fontsize=7.5, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_profiles(scores, path):
    plt = style()
    taus = np.linspace(-0.1, 1.05, 116)
    fig, axes = plt.subplots(1, len(CURRICULA), figsize=(10.2, 3.0), sharey=True)
    for ax, cur in zip(axes, CURRICULA):
        for key, col in (((cur, "original"), "original"), ((cur, "causal"), "causal"), (("default", "original"), "default")):
            p, lo, hi = profile(scores[key], taus)
            ax.fill_between(taus, lo, hi, color=C[col], alpha=0.18, lw=0)
            ax.plot(taus, p, color=C[col], label=VAR_LABEL[col], ls="-" if col != "default" else (0, (3, 3)))
        ax.set_title(CUR_LABEL[cur], loc="left")
        ax.set_xlabel("normalized score threshold τ")
    axes[0].set_ylabel("fraction of runs with q > τ")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.text(0.01, 0.005, "Run-score performance profiles over both environments × 10 seeds; "
             "shaded: pointwise 95% stratified bootstrap bands (2,000).", fontsize=7.5, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path(__file__).resolve().parents[2] / "runs/task8/main")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.run / "analysis"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows, curves = load(args.run)

    # Checks: every planned job, the same final evaluation step, a full evaluation curve.
    assert len(rows) == 180, len(rows)
    assert {r["last_eval_step"] for r in rows} == {297_600}
    outside = {}
    for env in ENVS:
        ev = np.array([c["eval_return"] for c in curves if c["env"] == env and c["eval_return"] is not None])
        fin = np.array([r["last_eval_return"] for r in rows if r["env"] == env])
        q, fq = normalize(ev, env), normalize(fin, env)
        outside[env] = {"evaluations": int(len(ev)), "below_0": int((q < 0).sum()), "above_1": int((q > 1).sum()),
                        "final_below_0": int((fq < 0).sum()), "final_above_1": int((fq > 1).sum()),
                        "min_q": float(q.min()), "max_q": float(q.max())}

    scores = {m: {env: normalize(final(rows, env, *m)[1], env) for env in ENVS} for m in methods()}
    agg = aggregate(scores)
    poi_o = {c: prob_improvement(scores[(c, "causal")], scores[(c, "original")]) for c in CURRICULA}
    poi_d = {c: prob_improvement(scores[(c, "causal")], scores[("default", "original")]) for c in CURRICULA}
    poi_od = {c: prob_improvement(scores[(c, "original")], scores[("default", "original")]) for c in CURRICULA}
    diff = {}
    for c in CURRICULA:
        for name, ref in (("causal_minus_original", (c, "original")), ("causal_minus_target_only", ("default", "original"))):
            b = agg[(c, "causal")]["_boot_iqm"] - agg[ref]["_boot_iqm"]
            point = agg[(c, "causal")]["iqm"][0] - agg[ref]["iqm"][0]
            diff[(c, name)] = (point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    # Pooled: all four causal curricula vs all four original (both environments, 40 runs per env).
    pooled = {v: {env: np.concatenate([scores[(c, v)][env] for c in CURRICULA]) for env in ENVS}
              for v in ("original", "causal")}
    agg_pooled = aggregate(pooled)
    poi_pooled = prob_improvement(pooled["causal"], pooled["original"])
    cells = cell_table(rows)
    mis = misaligned_table(curves)

    fig_aggregate(agg, poi_o, poi_d, out / "figures/fig1_aggregate_iqm_poi.png")
    fig_final_scores(rows, out / "figures/fig2_final_scores.png")
    fig_curves(curves, out / "figures/fig3_learning_curves.png")
    fig_profiles(scores, out / "figures/fig4_performance_profiles.png")

    keys = ["env", "curriculum", "n_seeds", "original_return", "causal_return", "target_only_return", "original_q",
            "causal_q", "target_only_q", "causal_minus_original_q", "causal_minus_target_only_q",
            "original_minus_target_only_q", "wins_vs_original", "wins_vs_target_only"]
    with (out / "cell_effects.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for c in cells:
            w.writerow({k: (fmt(c[k]) if isinstance(c[k], tuple) else c[k]) for k in keys})
    with (out / "misaligned_portion.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env", "curriculum", "variant", "portion_percent", "max_seed_percent"])
        w.writeheader()
        for m in mis:
            w.writerow({"env": m["env"], "curriculum": m["curriculum"], "variant": m["variant"],
                        "portion_percent": fmt(m["portion"], 2), "max_seed_percent": f"{m['max_seed']:.2f}"})
    summary = {
        "bounds": {e: {"l": b[0], "u": b[1]} for e, b in BOUNDS.items()},
        "outside_bounds": outside,
        "aggregate": {f"{c}/{v}": {k: list(x) for k, x in agg[(c, v)].items() if not k.startswith("_")}
                      for c, v in agg},
        "iqm_difference": {f"{c}/{n}": list(v) for (c, n), v in diff.items()},
        "probability_of_improvement": {"causal_vs_original": {c: list(v) for c, v in poi_o.items()},
                                       "causal_vs_target_only": {c: list(v) for c, v in poi_d.items()},
                                       "original_vs_target_only": {c: list(v) for c, v in poi_od.items()},
                                       "pooled_causal_vs_original": list(poi_pooled)},
        "pooled_aggregate": {v: {k: list(x) for k, x in agg_pooled[v].items() if not k.startswith("_")}
                             for v in agg_pooled},
        "cells": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()} for c in cells],
        "misaligned_portion_percent": [{**{k: m[k] for k in ("env", "curriculum", "variant", "max_seed")},
                                        "portion": list(m["portion"])} for m in mis],
        "bootstrap": {"seed": SEED, "aggregate_reps": AGG_REPS, "curve_reps": CURVE_REPS, "method": "percentile"},
        "score": "last in-training evaluation (update 620, 297,600 steps), 128 sampled-action episodes",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("BOUNDS", BOUNDS)
    print("OUTSIDE [l, u]:", outside)
    print("\nAGGREGATE over both environments (q): IQM | mean | optimality gap")
    for (c, v), x in agg.items():
        print(f"  {label(c, v):22s} IQM {fmt(x['iqm'])}  mean {fmt(x['mean'])}  gap {fmt(x['optimality_gap'])}")
    print("\nIQM difference:", {f"{c}/{n}": fmt(v) for (c, n), v in diff.items()})
    print("P(causal > original):", {c: fmt(v) for c, v in poi_o.items()})
    print("P(causal > target only):", {c: fmt(v) for c, v in poi_d.items()})
    print("P(original > target only):", {c: fmt(v) for c, v in poi_od.items()})
    print("POOLED IQM:", {v: fmt(agg_pooled[v]["iqm"]) for v in agg_pooled}, " P(causal > original):", fmt(poi_pooled))
    print("\nPER CELL q: original | causal | target only | causal - original | causal - target only | wins")
    for c in cells:
        print(f"  {c['env']:15s} {c['curriculum']:8s} {fmt(c['original_q'])} | {fmt(c['causal_q'])} | "
              f"{fmt(c['target_only_q'])} | {fmt(c['causal_minus_original_q'])} | "
              f"{fmt(c['causal_minus_target_only_q'])} | {c['wins_vs_original']}/10 {c['wins_vs_target_only']}/10")
    print("\nMISALIGNED PORTION (%) mean [95% CI] over seeds")
    for m in mis:
        print(f"  {m['env']:15s} {m['curriculum']:8s} {m['variant']:8s} {fmt(m['portion'], 2)}")


if __name__ == "__main__":
    main()
