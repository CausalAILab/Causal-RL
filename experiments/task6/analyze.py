#!/usr/bin/env python3
"""Task 6 analysis following the Statistical Precipice protocol.

Reads runs/task6/main (results.csv, checkpoints.csv, per-job XOR traces) and writes
tables (CSV/JSON) and figures (PNG) to --out. Run units are training seeds; evaluation
episodes are averaged inside a run and never treated as extra samples.

  python experiments/task6/analyze.py --run runs/task6/main --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

SEED = 314159
AGG_REPS, CURVE_REPS = 50_000, 2_000
BACKBONES = ("dqn", "c51", "ppo")
BACKBONE_LABEL = {"dqn": "DQN", "c51": "C51", "ppo": "PPO"}
POSITIVE = ("robotwalk", "mdp_balanced", "drone_u2_h4", "drone_b2_h4", "drone_r4_h8", "drone_r6_h10")
DRONES = POSITIVE[2:]
ENV_LABEL = {"robotwalk": "RobotWalk", "mdp_balanced": "Balanced MDP", "drone_u2_h4": "Drone U2-H4",
             "drone_b2_h4": "Drone B2-H4", "drone_r4_h8": "Drone R4-H8", "drone_r6_h10": "Drone R6-H10",
             "cartpole_calm": "CartPole calm"}
# Exact optimal values (protocol, Task 6 calibration). State J* / +Natural J* (analysis only).
ORACLE = {"robotwalk": (5.5, 10.0), "mdp_balanced": (10.0, 18.0), "drone_u2_h4": (0.5668, 0.7685),
          "drone_b2_h4": (0.7853, 0.8379), "drone_r4_h8": (0.7301, 0.8866), "drone_r6_h10": (0.7371, 0.8618)}
ORACLE_SR = {"drone_u2_h4": (0.5904, 0.7872), "drone_b2_h4": (0.8022, 0.8528),
             "drone_r4_h8": (0.7777, 0.9249), "drone_r6_h10": (0.8023, 0.9190)}

# Reference palette (dataviz skill, light mode): slots 1-3 validate all-pairs.
C = {"do": "#2a78d6", "ctf": "#eb6834", "random_hint": "#1baf7a", "ref": "#8a8984",
     "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
MODE_LABEL = {"do": "State", "ctf": "State + Natural", "random_hint": "State + random hint"}


# ── Data ─────────────────────────────────────────────────────────────────────

def load(run):
    rows = list(csv.DictReader((run / "results.csv").open()))
    for r in rows:
        r["seed"] = int(r["seed"])
        for k in ("final_mean_return", "final_success_rate", "final_hint_agreement",
                  "conditional_oracle_regret", "own_information_regret", "final_sampled_mean_return"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None
    curves = list(csv.DictReader((run / "checkpoints.csv").open()))
    for c in curves:
        c["seed"], c["step"] = int(c["seed"]), int(c["step"])
        c["mean_return"] = float(c["mean_return"])
        c["success_rate"] = float(c["success_rate"]) if c["success_rate"] not in ("", "None") else None
    return rows, curves


def runs(rows, env, algo, mode, key="final_mean_return"):
    sel = sorted((r for r in rows if r["env"] == env and r["algorithm"] == algo and r["mode"] == mode),
                 key=lambda r: r["seed"])
    return np.array([r["seed"] for r in sel]), np.array([r[key] for r in sel], dtype=float)


def anchors(rows):
    """Random anchor: the Random reference (10 seeds x 1000 shared episodes); oracle: exact J_I*."""
    out = {}
    for env in POSITIVE:
        _, rnd = runs(rows, env, "random", "ctf")
        out[env] = {"random": float(rnd.mean()), "oracle": ORACLE[env][1]}
    return out


def normalize(x, env, anc):
    a = anc[env]
    return (np.asarray(x, dtype=float) - a["random"]) / (a["oracle"] - a["random"])


# ── Statistics ───────────────────────────────────────────────────────────────

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
    mat = np.stack([scores[t] for t in tasks])                       # (tasks, n)
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
    """mat: (seeds, steps). Pointwise mean and percentile band over seeds."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, mat.shape[0], size=(reps, mat.shape[0]))
    boot = mat[idx].mean(1)
    return mat.mean(0), np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)


# ── Tables ───────────────────────────────────────────────────────────────────

def fmt(t, d=3):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def cell_table(rows, anc):
    out = []
    for env in POSITIVE + ("cartpole_calm",):
        for algo in BACKBONES:
            s_seed, s = runs(rows, env, algo, "do")
            n_seed, n = runs(rows, env, algo, "ctf")
            assert (s_seed == n_seed).all()
            row = {"env": env, "backbone": algo, "n_seeds": len(s),
                   "state_return": ci_mean(s), "natural_return": ci_mean(n), "paired_diff_return": ci_mean(n - s)}
            _, ss = runs(rows, env, algo, "do", "final_success_rate")
            _, ns = runs(rows, env, algo, "ctf", "final_success_rate")
            if not np.isnan(ss).any():
                row.update(state_sr=ci_mean(ss), natural_sr=ci_mean(ns), paired_diff_sr=ci_mean(ns - ss))
            _, agree = runs(rows, env, algo, "ctf", "final_hint_agreement")
            row["natural_hint_agreement"] = ci_mean(agree)
            if env in anc:
                row["state_q"] = ci_mean(normalize(s, env, anc))
                row["natural_q"] = ci_mean(normalize(n, env, anc))
                row["paired_diff_q"] = ci_mean(normalize(n, env, anc) - normalize(s, env, anc))
            row["wins"] = int((n > s).sum())
            out.append(row)
    return out


def reference_table(rows):
    out = []
    for env in POSITIVE + ("cartpole_calm",):
        for algo in ("random", "follow_natural"):
            _, x = runs(rows, env, algo, "ctf")
            _, sr = runs(rows, env, algo, "ctf", "final_success_rate")
            row = {"env": env, "reference": algo, "return": ci_mean(x)}
            if not np.isnan(sr).any():
                row["sr"] = ci_mean(sr)
            out.append(row)
    return out


def hint_table(rows):
    out = []
    for r in sorted({(r["env"], r["algorithm"]) for r in rows if r["mode"] == "random_hint"}):
        env, algo = r
        _, s = runs(rows, env, algo, "do")
        _, h = runs(rows, env, algo, "random_hint")
        _, n = runs(rows, env, algo, "ctf")
        out.append({"env": env, "backbone": algo, "state": ci_mean(s), "random_hint": ci_mean(h),
                    "natural": ci_mean(n), "hint_minus_state": ci_mean(h - s)})
    return out


def xor_table(rows, run):
    res = {}
    for algo in ("ucb", "ctf_ucb"):
        seeds, reg = runs(rows, "casino_xor", algo, "native", "conditional_oracle_regret")
        traces = []
        for seed in seeds:
            path = run / "jobs/casino_xor" / f"{algo}-native" / f"seed_{seed}" / "interactions.csv.gz"
            with gzip.open(path, "rt") as f:
                traces.append([float(r["conditional_oracle_regret"]) for r in csv.DictReader(f)])
        res[algo] = {"final": ci_mean(reg), "trace": np.array(traces)}
    res["diff"] = ci_mean(runs(rows, "casino_xor", "ctf_ucb", "native", "conditional_oracle_regret")[1]
                          - runs(rows, "casino_xor", "ucb", "native", "conditional_oracle_regret")[1])
    return res


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


def fig_effects(cells, path):
    plt = style()
    pos = [c for c in cells if c["env"] in POSITIVE]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    offsets = {"dqn": -0.22, "c51": 0.0, "ppo": 0.22}
    markers = {"dqn": "o", "c51": "s", "ppo": "D"}
    for c in pos:
        y = POSITIVE.index(c["env"]) + offsets[c["backbone"]]
        m, lo, hi = c["paired_diff_q"]
        ax.plot([lo, hi], [y, y], color=C["ctf"], lw=2, solid_capstyle="round")
        ax.plot(m, y, markers[c["backbone"]], color=C["ctf"], ms=6, mec=C["surface"], mew=1.5)
    ax.axvline(0, color=C["ink2"], lw=1)
    ax.set_yticks(range(len(POSITIVE)), [ENV_LABEL[e] for e in POSITIVE])
    ax.invert_yaxis()
    ax.set_xlabel("Paired effect of the natural action on normalized score  Δq = q(+Natural) − q(State)")
    ax.set_title("+Natural improves every backbone in every positive cell", loc="left")
    for algo in BACKBONES:
        ax.plot([], [], markers[algo], color=C["ctf"], label=BACKBONE_LABEL[algo], ms=6)
    ax.legend(loc="lower right", title="Backbone (marker)", title_fontsize=8)
    fig.text(0.01, 0.005, "Mean over 10 seeds; 95% percentile bootstrap CIs of the paired difference (50,000 "
             "resamples) are narrower than the markers.\nq = (J − J_random) / (J*_+Natural − J_random): exact optimum "
             "with intention = 1, Random reference = 0.", fontsize=7.5, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_aggregate(agg_all, agg_drone, poi_all, path):
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, agg, title in ((axes[0], agg_all, "IQM, six positive cells"),
                           (axes[1], agg_drone, "IQM, four Drone cells")):
        for i, algo in enumerate(BACKBONES):
            for j, mode in enumerate(("do", "ctf")):
                m, lo, hi = agg[(algo, mode)]["iqm"]
                y = i + (j - 0.5) * 0.32
                ax.plot([lo, hi], [y, y], color=C[mode], lw=2, solid_capstyle="round")
                ax.plot(m, y, "o", color=C[mode], ms=7, mec=C["surface"], mew=1.5)
        ax.set_yticks(range(3), [BACKBONE_LABEL[a] for a in BACKBONES])
        ax.invert_yaxis()
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("IQM normalized score q")
        ax.set_title(title, loc="left")
    for mode in ("do", "ctf"):
        axes[0].plot([], [], "o", color=C[mode], label=MODE_LABEL[mode])
    axes[0].legend(loc="lower left", fontsize=8)
    ax = axes[2]
    for i, algo in enumerate(BACKBONES):
        m, lo, hi = poi_all[algo]
        ax.plot([lo, hi], [i, i], color=C["ctf"], lw=2, solid_capstyle="round")
        ax.plot(m, i, "o", color=C["ctf"], ms=7, mec=C["surface"], mew=1.5)
    ax.axvline(0.5, color=C["ink2"], lw=1)
    ax.set_yticks(range(3), [BACKBONE_LABEL[a] for a in BACKBONES])
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("P(+Natural > State)")
    ax.set_title("Probability of improvement", loc="left")
    fig.text(0.01, 0.005, "Stratified percentile bootstrap over seeds within cells: IQM 50,000 resamples; "
             "probability of improvement 2,000 (independent sampling, Mann–Whitney, ties = .5).",
             fontsize=7.5, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_profiles(scores, path):
    plt = style()
    taus = np.linspace(-0.2, 1.1, 131)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    for ax, algo in zip(axes, BACKBONES):
        for mode in ("do", "ctf"):
            p, lo, hi = profile(scores[(algo, mode)], taus)
            ax.fill_between(taus, lo, hi, color=C[mode], alpha=0.18, lw=0)
            ax.plot(taus, p, color=C[mode], label=MODE_LABEL[mode])
        ax.set_title(BACKBONE_LABEL[algo], loc="left")
        ax.set_xlabel("normalized score threshold τ")
    axes[0].set_ylabel("fraction of runs with q > τ")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.text(0.01, 0.005, "Run-score performance profiles over the six positive cells × 10 seeds; "
             "shaded: pointwise 95% stratified bootstrap bands (2,000 resamples).", fontsize=7.5, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_curves(curves, anc, path):
    plt = style()
    fig, axes = plt.subplots(len(BACKBONES), len(DRONES), figsize=(10.5, 6.6), sharex=True, sharey=True)
    for i, algo in enumerate(BACKBONES):
        for j, env in enumerate(DRONES):
            ax = axes[i, j]
            for mode in ("do", "ctf"):
                sel = [c for c in curves if c["env"] == env and c["algorithm"] == algo and c["mode"] == mode]
                steps = sorted({c["step"] for c in sel})
                seeds = sorted({c["seed"] for c in sel})
                table = {(c["seed"], c["step"]): c["mean_return"] for c in sel}
                mat = normalize(np.array([[table[(s, t)] for t in steps] for s in seeds]), env, anc)
                m, lo, hi = curve_band(mat)
                x = np.array(steps) / 1e6
                ax.fill_between(x, lo, hi, color=C[mode], alpha=0.18, lw=0)
                ax.plot(x, m, color=C[mode], lw=1.8, label=MODE_LABEL[mode])
            state_oracle = (ORACLE[env][0] - anc[env]["random"]) / (anc[env]["oracle"] - anc[env]["random"])
            ax.axhline(1.0, color=C["ctf"], lw=1, ls=(0, (3, 3)))
            ax.axhline(state_oracle, color=C["do"], lw=1, ls=(0, (3, 3)))
            if i == 0:
                ax.set_title(ENV_LABEL[env], loc="left")
            if j == 0:
                ax.set_ylabel(f"{BACKBONE_LABEL[algo]}\nnormalized score q")
            if i == len(BACKBONES) - 1:
                ax.set_xlabel("training transitions (millions)")
    axes[0, 0].legend(loc="lower right", fontsize=7.5)
    fig.text(0.01, 0.005, "Mean of 10 seeds (100 checkpoint episodes each); shaded: pointwise 95% bootstrap bands "
             "(2,000). Dashed: exact optimum with intention (q = 1, orange) and state-only optimum (blue).",
             fontsize=7.5, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_xor(xor, path):
    plt = style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    label = {"ucb": "UCB (no intention)", "ctf_ucb": "Ctf-UCB (conditions on intention)"}
    color = {"ucb": C["do"], "ctf_ucb": C["ctf"]}
    for algo in ("ucb", "ctf_ucb"):
        m, lo, hi = curve_band(xor[algo]["trace"])
        x = np.arange(1, len(m) + 1)
        ax.fill_between(x, lo, hi, color=color[algo], alpha=0.18, lw=0)
        ax.plot(x, m, color=color[algo], label=label[algo])
        ax.annotate(f"{m[-1]:.0f}", (x[-1], m[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=C["ink2"])
    ax.set_xlabel("pulls")
    ax.set_ylabel("cumulative conditional-oracle regret")
    ax.set_title("Casino XOR: regret against the best intention-aware arm", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    fig.text(0.01, 0.005, "Mean of 10 seeds; shaded: pointwise 95% bootstrap bands (2,000). "
             "Oracle means are analysis-only.", fontsize=7.5, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_drone_sr(rows, path):
    plt = style()
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    width = 0.12
    groups = [("do", "dqn"), ("ctf", "dqn"), ("do", "c51"), ("ctf", "c51"), ("do", "ppo"), ("ctf", "ppo")]
    marker = {"dqn": "o", "c51": "s", "ppo": "D"}
    for j, env in enumerate(DRONES):
        for k, (mode, algo) in enumerate(groups):
            x = j + (k - 2.5) * width
            _, sr = runs(rows, env, algo, mode, "final_success_rate")
            m, lo, hi = ci_mean(sr)
            ax.plot([x, x], [lo, hi], color=C[mode], lw=2, solid_capstyle="round")
            ax.plot(x, m, marker[algo], color=C[mode], ms=5.5, mec=C["surface"], mew=1.2)
        for mode, val in zip(("do", "ctf"), ORACLE_SR[env]):
            ax.plot([j - 0.42, j + 0.42], [val, val], color=C[mode], lw=1, ls=(0, (3, 3)))
        _, fn = runs(rows, env, "follow_natural", "ctf", "final_success_rate")
        ax.plot([j - 0.42, j + 0.42], [fn.mean()] * 2, color=C["ref"], lw=1, ls=(0, (1, 2)))
    ax.set_xticks(range(len(DRONES)), [ENV_LABEL[e] for e in DRONES])
    ax.set_ylabel("rescue success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Drone rescue success: learned policies vs exact optima", loc="left")
    for mode in ("do", "ctf"):
        ax.plot([], [], "-", color=C[mode], label=MODE_LABEL[mode])
    for algo in BACKBONES:
        ax.plot([], [], marker[algo], color=C["ink2"], label=BACKBONE_LABEL[algo], ms=5.5)
    ax.plot([], [], color=C["ref"], ls=(0, (1, 2)), lw=1, label="Follow-natural")
    ax.legend(loc="lower left", ncol=6, fontsize=7.5)
    fig.text(0.01, 0.005, "Points: mean of 10 seeds × 1,000 final episodes with 95% bootstrap CIs over seeds. "
             "Dashed: SR of the exact return-optimal policy per input.", fontsize=7.5, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def jsonable(v):
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, np.ndarray):
        return None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path(__file__).resolve().parents[2] / "runs/task6/main")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rows, curves = load(args.run)
    anc = anchors(rows)
    cells = cell_table(rows, anc)
    refs = reference_table(rows)
    hints = hint_table(rows)
    xor = xor_table(rows, args.run)

    scores = {}
    for algo in BACKBONES:
        for mode in ("do", "ctf"):
            scores[(algo, mode)] = {env: normalize(runs(rows, env, algo, mode)[1], env, anc) for env in POSITIVE}
    agg_all = aggregate(scores)
    agg_drone = aggregate({k: {e: v[e] for e in DRONES} for k, v in scores.items()})
    poi_all = {algo: prob_improvement(scores[(algo, "ctf")], scores[(algo, "do")]) for algo in BACKBONES}
    diff_iqm = {}
    for algo in BACKBONES:
        b = agg_all[(algo, "ctf")]["_boot_iqm"] - agg_all[(algo, "do")]["_boot_iqm"]
        point = agg_all[(algo, "ctf")]["iqm"][0] - agg_all[(algo, "do")]["iqm"][0]
        diff_iqm[algo] = (point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))

    fig_effects(cells, out / "figures/fig1_paired_effects.png")
    fig_aggregate(agg_all, agg_drone, poi_all, out / "figures/fig2_aggregate_iqm_poi.png")
    fig_profiles(scores, out / "figures/fig3_performance_profiles.png")
    fig_curves(curves, anc, out / "figures/fig4_drone_learning_curves.png")
    fig_drone_sr(rows, out / "figures/fig5_drone_success.png")
    fig_xor(xor, out / "figures/fig6_xor_regret.png")

    with (out / "cell_effects.csv").open("w", newline="") as f:
        keys = ["env", "backbone", "n_seeds", "state_return", "natural_return", "paired_diff_return", "state_sr",
                "natural_sr", "paired_diff_sr", "state_q", "natural_q", "paired_diff_q", "natural_hint_agreement", "wins"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for c in cells:
            w.writerow({k: (fmt(c[k]) if isinstance(c.get(k), tuple) else c.get(k, "")) for k in keys})
    summary = {
        "anchors": anc, "cells": [{k: jsonable(v) for k, v in c.items()} for c in cells],
        "references": [{k: jsonable(v) for k, v in r.items()} for r in refs],
        "random_hint": [{k: jsonable(v) for k, v in r.items()} for r in hints],
        "aggregate_six": {f"{a}/{m}": {k: list(v) for k, v in agg_all[(a, m)].items() if not k.startswith("_")}
                          for a, m in agg_all},
        "aggregate_drone": {f"{a}/{m}": {k: list(v) for k, v in agg_drone[(a, m)].items() if not k.startswith("_")}
                            for a, m in agg_drone},
        "iqm_difference_six": {a: list(v) for a, v in diff_iqm.items()},
        "probability_of_improvement_six": {a: list(v) for a, v in poi_all.items()},
        "xor": {"ucb_final_regret": list(xor["ucb"]["final"]), "ctf_ucb_final_regret": list(xor["ctf_ucb"]["final"]),
                "difference": list(xor["diff"])},
        "bootstrap": {"seed": SEED, "aggregate_reps": AGG_REPS, "curve_reps": CURVE_REPS, "method": "percentile"},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("ANCHORS (random empirical, oracle exact):", {e: round(a["random"], 4) for e, a in anc.items()})
    print("\nPER CELL  return: State | +Natural | paired diff   SR diff   hint-agreement   wins/10")
    for c in cells:
        sr = fmt(c["paired_diff_sr"], 3) if "paired_diff_sr" in c else "-"
        print(f"{c['env']:14s} {c['backbone']:4s} {fmt(c['state_return'])} | {fmt(c['natural_return'])} | "
              f"{fmt(c['paired_diff_return'])}  SR {sr}  agree {c['natural_hint_agreement'][0]:.2f}  {c['wins']}/10")
    print("\nREFERENCES")
    for r in refs:
        print(f"{r['env']:14s} {r['reference']:15s} return {fmt(r['return'])}  SR {fmt(r['sr']) if 'sr' in r else '-'}")
    print("\nRANDOM HINT (hint - state return)")
    for h in hints:
        print(f"{h['env']:14s} {h['backbone']:4s} state {fmt(h['state'])} hint {fmt(h['random_hint'])} "
              f"natural {fmt(h['natural'])} diff {fmt(h['hint_minus_state'])}")
    print("\nAGGREGATE six cells (q): IQM | optimality gap")
    for (a, m), v in agg_all.items():
        print(f"  {a:4s} {m:4s} IQM {fmt(v['iqm'])}  mean {fmt(v['mean'])}  gap {fmt(v['optimality_gap'])}")
    print("  IQM difference (+Natural - State):", {a: fmt(v) for a, v in diff_iqm.items()})
    print("  P(+Natural > State):", {a: fmt(v) for a, v in poi_all.items()})
    print("\nAGGREGATE four Drone cells (q) IQM")
    for (a, m), v in agg_drone.items():
        print(f"  {a:4s} {m:4s} IQM {fmt(v['iqm'])}")
    print("\nXOR final regret: UCB", fmt(xor["ucb"]["final"], 1), " Ctf-UCB", fmt(xor["ctf_ucb"]["final"], 1),
          " diff", fmt(xor["diff"], 1))


if __name__ == "__main__":
    main()
