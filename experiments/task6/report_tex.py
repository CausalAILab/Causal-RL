#!/usr/bin/env python3
"""Write the Task 6 report figures (pgfplots) and tables (booktabs) from runs/task6/main.

Style follows the Task 4/5 figures of the CausalGym report: groupplot, left/bottom axes,
dotted y grid, 1pt lines, .18 bootstrap bands, colored-text legend, report colors
ucbcolor (without the natural action), causalcolor (with it), transfercolor (random hint).

  python experiments/task6/report_tex.py --report <tech report dir> [--run runs/task6/main]
"""
from __future__ import annotations

import argparse
import string
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402

MODES = ("do", "ctf")
COLOR = {"do": "ucbcolor", "ctf": "causalcolor", "random_hint": "transfercolor"}
ENV_TEX = {"robotwalk": "RobotWalk", "mdp_balanced": "Balanced MDP", "drone_u2_h4": "Drone U2-H4",
           "drone_b2_h4": "Drone B2-H4", "drone_r4_h8": "Drone R4-H8", "drone_r6_h10": "Drone R6-H10",
           "cartpole_calm": "CartPole (no wind)"}
BB = {"dqn": "DQN", "c51": "C51", "ppo": "PPO"}
BASE = (r"scale only axis=true,axis lines*=left,ymajorgrids=true,grid style={dotted,black!20},"
        r"tick label style={font=\scriptsize},label style={font=\scriptsize},title style={font=\footnotesize},"
        r"scaled x ticks=false,scaled y ticks=false,/pgf/number format/1000 sep={,},clip=true")


def coords(x, y):
    return " ".join(f"({a:.6g},{b:.6g})" for a, b in zip(x, y))


def band(name, x, m, lo, hi, color, style=""):
    return "\n".join([
        rf"\addplot[draw=none,name path={name}l,forget plot] coordinates {{{coords(x, lo)}}};",
        rf"\addplot[draw=none,name path={name}u,forget plot] coordinates {{{coords(x, hi)}}};",
        rf"\addplot[fill={color},fill opacity=.18,draw=none,forget plot] fill between[of={name}l and {name}u];",
        rf"\addplot[color={color},line width=1pt,mark=none,{style}] coordinates {{{coords(x, m)}}};", ""])


def hline(y, color, xmax):
    return rf"\addplot[color={color},line width=.5pt,densely dashed,forget plot] coordinates {{(0,{y:.6g}) ({xmax:.6g},{y:.6g})}};" + "\n"


def drone_curve_matrix(curves, env, algo, mode, anc):
    sel = [c for c in curves if c["env"] == env and c["algorithm"] == algo and c["mode"] == mode]
    steps = sorted({c["step"] for c in sel})
    seeds = sorted({c["seed"] for c in sel})
    table = {(c["seed"], c["step"]): c["mean_return"] for c in sel}
    mat = A.normalize(np.array([[table[(s, t)] for t in steps] for s in seeds]), env, anc)
    return np.array(steps), mat


def state_oracle_q(env, anc):
    a = anc[env]
    return (A.ORACLE[env][0] - a["random"]) / (a["oracle"] - a["random"])


def drone_panel(curves, anc, env, algo, letter, ylabel, xlabel):
    out = [rf"\nextgroupplot[title={{({letter}) {ENV_TEX[env]}}},ymin=-.05,ymax=1.1,ytick={{0,.5,1}},"
           rf"xmin=0,xmax=1.048576,xtick={{0,.5,1}},xticklabels={{0,0.5M,1M}},"
           rf"ylabel={{{ylabel}}},xlabel={{{xlabel}}}]" + "\n"]
    out.append(hline(1.0, COLOR["ctf"], 1.048576))
    out.append(hline(state_oracle_q(env, anc), COLOR["do"], 1.048576))
    for mode in MODES:
        steps, mat = drone_curve_matrix(curves, env, algo, mode, anc)
        m, lo, hi = A.curve_band(mat)
        out.append(band(f"t6{env[6:]}{algo}{mode}", steps / 1e6, m, lo, hi, COLOR[mode]))
    return "".join(out)


def fig_main(rows, curves, anc, run):
    xor = A.xor_table(rows, run)
    legend = (r"{\small\textcolor{ucbcolor}{Without the natural action (UCB; State)}\quad"
              r"\textcolor{causalcolor}{With the natural action (Ctf-UCB; State + Natural)}}\par\vspace{6pt}")
    parts = [r"\begingroup", r"\color{black}", r"\begin{figure}[H]\centering", legend,
             r"\resizebox{\linewidth}{!}{\begin{tikzpicture}\begin{groupplot}[group style={group size=5 by 1,"
             r"horizontal sep=1.3cm},width=2.6cm,height=2.6cm," + BASE + "]"]
    idx = np.r_[0, np.arange(49, 10000, 50)]
    parts.append(r"\nextgroupplot[title={(a) Casino XOR},ymin=-30,ymax=1600,xmin=0,xmax=10000,"
                 r"xtick={0,5000,10000},xticklabels={0,5k,10k},ytick={0,500,1000,1500},"
                 r"ylabel={Cumulative regret},xlabel={Pulls}]")
    for algo, mode in (("ucb", "do"), ("ctf_ucb", "ctf")):
        m, lo, hi = A.curve_band(xor[algo]["trace"])
        x = np.r_[0, idx[1:] + 1]
        parts.append(band(f"t6xor{mode}", x, np.r_[0, m[idx[1:]]], np.r_[0, lo[idx[1:]]], np.r_[0, hi[idx[1:]]],
                          COLOR[mode]))
    for k, env in enumerate(A.DRONES):
        parts.append(drone_panel(curves, anc, env, "dqn", string.ascii_lowercase[k + 1],
                                 "Normalized score" if k == 0 else "", "Transitions"))
    parts += [r"\end{groupplot}\end{tikzpicture}}",
              r"\caption{\textbf{Task~6 learning curves.} (a) Cumulative regret of UCB and Ctf-UCB on Casino XOR "
              r"against the best intention-aware arm. (b)--(e) Normalized score of DQN on the four Drone configurations, "
              r"evaluated every 32,768 transitions on 100 episodes. Lines are means over 10 seeds; bands are pointwise "
              r"95\% bootstrap CIs (2,000 resamples). Dashed lines mark the exact optima with the natural action "
              r"($q=1$) and with the state alone. C51 and PPO curves appear in Appendix~\ref{app:task6}.}",
              r"\label{fig:task6-curves}", r"\end{figure}", r"\endgroup", ""]
    return "\n".join(parts)


def fig_appendix_curves(curves, anc):
    legend = (r"{\small\textcolor{ucbcolor}{State}\quad\textcolor{causalcolor}{State + Natural}}\par\vspace{6pt}")
    parts = [r"\begingroup", r"\color{black}", r"\begin{figure}[H]\centering", legend,
             r"\resizebox{\linewidth}{!}{\begin{tikzpicture}\begin{groupplot}[group style={group size=4 by 2,"
             r"horizontal sep=1.0cm,vertical sep=1.45cm},width=3.0cm,height=2.3cm," + BASE + "]"]
    k = 0
    for r, algo in enumerate(("c51", "ppo")):
        for j, env in enumerate(A.DRONES):
            parts.append(drone_panel(curves, anc, env, algo, string.ascii_lowercase[k],
                                     f"{BB[algo]}: normalized score" if j == 0 else "",
                                     "Transitions" if r == 1 else ""))
            k += 1
    parts += [r"\end{groupplot}\end{tikzpicture}}",
              r"\caption{\textbf{Drone learning curves for C51 and PPO.} Same protocol as "
              r"Figure~\ref{fig:task6-curves}: means over 10 seeds with pointwise 95\% bootstrap CIs; dashed lines "
              r"mark the exact optima with and without the natural action.}",
              r"\label{fig:task6-curves-app}", r"\end{figure}", r"\endgroup", ""]
    return "\n".join(parts)


def fig_profiles(scores):
    taus = np.linspace(0, 1.05, 106)
    legend = (r"{\small\textcolor{ucbcolor}{State}\quad\textcolor{causalcolor}{State + Natural}}\par\vspace{6pt}")
    parts = [r"\begingroup", r"\color{black}", r"\begin{figure}[H]\centering", legend,
             r"\resizebox{\linewidth}{!}{\begin{tikzpicture}\begin{groupplot}[group style={group size=3 by 1,"
             r"horizontal sep=1.1cm},width=3.9cm,height=2.4cm," + BASE +
             r",xmin=0,xmax=1.05,ymin=-.03,ymax=1.06,xtick={0,.5,1},ytick={0,.5,1}]"]
    for k, algo in enumerate(A.BACKBONES):
        parts.append(rf"\nextgroupplot[title={{({string.ascii_lowercase[k]}) {BB[algo]}}},"
                     rf"xlabel={{Score threshold $\tau$}},ylabel={{{'Fraction of runs with $q>\\tau$' if k == 0 else ''}}}]")
        for mode in MODES:
            p, lo, hi = A.profile(scores[(algo, mode)], taus)
            parts.append(band(f"t6prof{algo}{mode}", taus, p, lo, hi, COLOR[mode]))
    parts += [r"\end{groupplot}\end{tikzpicture}}",
              r"\caption{\textbf{Task~6 performance profiles.} Fraction of the 60 runs (six configurations, 10 seeds) "
              r"whose normalized score exceeds $\tau$; bands are pointwise 95\% stratified bootstrap CIs "
              r"(2,000 resamples).}",
              r"\label{fig:task6-profiles}", r"\end{figure}", r"\endgroup", ""]
    return "\n".join(parts)


# ── Tables ───────────────────────────────────────────────────────────────────

def digits(env):
    return 2 if env in ("robotwalk", "mdp_balanced") else (1 if env == "cartpole_calm" else 3)


def num(v, d):
    s = f"{v:,.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def ci(t, d):
    m, lo, hi = t
    if round(lo, d) == round(m, d) == round(hi, d):
        return num(m, d)
    return f"{num(m, d)} [{num(lo, d)}, {num(hi, d)}]"


def tab_main_block(cells):
    by = {(c["env"], c["backbone"]): c for c in cells}
    lines = [r"\vspace{10pt}", r"\begin{tabular}{@{}lcccc@{}}",
             r"\multicolumn{5}{@{}l}{\textbf{(c) Task 6: counterfactual decision-making}} \\", r"\toprule",
             r"\textbf{Configuration} & \textbf{Exact optimum} & \textbf{DQN} & \textbf{C51} & \textbf{PPO} \\",
             r"\midrule"]
    for env in A.POSITIVE:
        d = digits(env)
        o = A.ORACLE[env]
        row = [ENV_TEX[env], f"{num(o[0], d)} / {num(o[1], d)}"]
        for algo in A.BACKBONES:
            c = by[(env, algo)]
            row.append(f"{num(c['state_return'][0], d)} / \\textbf{{{num(c['natural_return'][0], d)}}}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tab_cells(cells):
    lines = [r"\begin{table}[H]", r"\centering", r"\reviewcolor[tab-task6]", r"\scriptsize",
             r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"\textbf{Configuration} & & \textbf{State} & \textbf{State + Natural} & "
             r"\textbf{Paired difference} & \textbf{Success difference} \\", r"\midrule"]
    envs = A.POSITIVE + ("cartpole_calm",)
    for i, env in enumerate(envs):
        d = digits(env)
        for j, algo in enumerate(A.BACKBONES):
            c = next(x for x in cells if x["env"] == env and x["backbone"] == algo)
            srd = ci(c["paired_diff_sr"], 3) if "paired_diff_sr" in c else "--"
            label = ENV_TEX[env] if j == 0 else ""
            lines.append(f"{label} & {BB[algo]} & {ci(c['state_return'], d)} & {ci(c['natural_return'], d)} & "
                         f"{ci(c['paired_diff_return'], d)} & {srd} \\\\")
        if env == "drone_r6_h10":
            lines.append(r"\midrule")
        elif i < len(envs) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Task~6 final native return of each backbone with the state alone and with the natural "
              r"action, over 10 seeds and 1,000 shared evaluation episodes per seed. Entries are means [95\% "
              r"percentile bootstrap CIs over seeds, 50,000 resamples]; the paired difference is State + Natural "
              r"minus State within each seed. Success is the rescue rate for Drone and goal rate for RobotWalk; the "
              r"MDP has no success event. Intervals of zero width are shown as a single value. CartPole without wind "
              r"is a control whose two optima coincide.}",
              r"\label{tab:task6-cells}", r"\end{table}", ""]
    return "\n".join(lines)


def tab_aggregate(agg_all, agg_drone, diff, poi):
    lines = [r"\begin{table}[H]", r"\centering", r"\reviewcolor[tab-task6]", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{\textbf{Six configurations}} & \textbf{Four Drone} \\",
             r"\cmidrule(lr){3-4}\cmidrule(l){5-5}",
             r"\textbf{Backbone} & \textbf{Input} & \textbf{IQM} & \textbf{Optimality gap} & \textbf{IQM} \\",
             r"\midrule"]
    for k, algo in enumerate(A.BACKBONES):
        for mode, label in (("do", "State"), ("ctf", "State + Natural")):
            a, b = agg_all[(algo, mode)], agg_drone[(algo, mode)]
            lines.append(f"{BB[algo] if mode == 'do' else ''} & {label} & {ci(a['iqm'], 3)} & "
                         f"{ci(a['optimality_gap'], 3)} & {ci(b['iqm'], 3)} \\\\")
        lines.append(f" & Difference & {ci(diff[algo], 3)} & & \\\\")
        lines.append(f" & $P$(Natural $>$ State) & {ci(poi[algo], 3)} & & \\\\")
        if k < 2:
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Task~6 aggregate normalized scores $q=(J-J_{\mathrm{rand}})/(J^\star_{X'}-J_{\mathrm{rand}})$ "
              r"over 10 seeds per configuration, with 95\% stratified percentile bootstrap CIs (50,000 resamples; "
              r"2,000 for the probability of improvement). IQM is the interquartile mean of all run scores; the "
              r"optimality gap is the mean shortfall below $q=1$. The difference resamples both inputs with the same "
              r"seed indices. The probability of improvement averages, over configurations, the Mann--Whitney "
              r"probability that a run with the natural action outperforms a run without it.}",
              r"\label{tab:task6-aggregate}", r"\end{table}", ""]
    return "\n".join(lines)


def tab_controls(rows, refs, hints, xor, anc):
    lines = [r"\begin{table}[H]", r"\centering", r"\reviewcolor[tab-task6]", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lrrrr@{}}",
             r"\multicolumn{5}{@{}l}{\textbf{(a) Reference policies on the shared evaluation episodes}} \\",
             r"\toprule",
             r"\textbf{Configuration} & \textbf{Random} & \textbf{Follow-natural} & "
             r"$\boldsymbol{J^\star}$ & $\boldsymbol{J^\star_{X'}}$ \\", r"\midrule"]
    for env in A.POSITIVE + ("cartpole_calm",):
        d = digits(env)
        r = {x["reference"]: x for x in refs if x["env"] == env}
        opt = A.ORACLE.get(env, (200.0, 200.0))
        lines.append(f"{ENV_TEX[env]} & {ci(r['random']['return'], d)} & {ci(r['follow_natural']['return'], d)} & "
                     f"{num(opt[0], d)} & {num(opt[1], d)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{10pt}",
              r"\begin{tabular}{@{}llrrr@{}}",
              r"\multicolumn{5}{@{}l}{\textbf{(b) Uninformative random hint in place of the natural action}} \\",
              r"\toprule",
              r"\textbf{Configuration} & & \textbf{State} & \textbf{State + random hint} & "
              r"\textbf{Paired difference} \\", r"\midrule"]
    order = ("robotwalk", "mdp_balanced", "drone_u2_h4")
    for env in order:
        d = digits(env)
        for j, algo in enumerate(("dqn", "ppo")):
            h = next(x for x in hints if x["env"] == env and x["backbone"] == algo)
            lines.append(f"{ENV_TEX[env] if j == 0 else ''} & {BB[algo]} & {ci(h['state'], d)} & "
                         f"{ci(h['random_hint'], d)} & {ci(h['hint_minus_state'], d)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{10pt}",
              r"\begin{tabular}{@{}lrrr@{}}",
              r"\multicolumn{4}{@{}l}{\textbf{(c) Casino XOR: cumulative regret after 10,000 pulls}} \\",
              r"\toprule", r"\textbf{Configuration} & \textbf{UCB} & \textbf{Ctf-UCB} & \textbf{Paired difference} \\",
              r"\midrule",
              f"Casino XOR & {ci(xor['ucb']['final'], 1)} & \\textbf{{{ci(xor['ctf_ucb']['final'], 1)}}} & "
              f"{ci(xor['diff'], 1)} \\\\",
              r"\bottomrule", r"\end{tabular}",
              r"\caption{Task~6 references and controls, 10 seeds each, with 95\% percentile bootstrap CIs over seeds. "
              r"(a) Random and follow-natural ($A=X'$) policies evaluated on the same 1,000 episodes per seed as the "
              r"learners, with the exact optimal returns without ($J^\star$) and with ($J^\star_{X'}$) the natural action. (b) DQN and PPO with an independently sampled one-hot hint of "
              r"the same width as the natural action. (c) Casino XOR, where both arms have mean $0.30$ without the "
              r"intended action and the best intention-aware arm has mean $0.45$; regret is measured against that arm.}",
              r"\label{tab:task6-controls}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path(__file__).resolve().parents[2] / "runs/task6/main")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    rows, curves = A.load(args.run)
    anc = A.anchors(rows)
    cells = A.cell_table(rows, anc)
    refs = A.reference_table(rows)
    hints = A.hint_table(rows)
    xor = A.xor_table(rows, args.run)
    scores = {(algo, mode): {env: A.normalize(A.runs(rows, env, algo, mode)[1], env, anc) for env in A.POSITIVE}
              for algo in A.BACKBONES for mode in MODES}
    agg_all = A.aggregate(scores)
    agg_drone = A.aggregate({k: {e: v[e] for e in A.DRONES} for k, v in scores.items()})
    poi = {algo: A.prob_improvement(scores[(algo, "ctf")], scores[(algo, "do")]) for algo in A.BACKBONES}
    diff = {}
    for algo in A.BACKBONES:
        b = agg_all[(algo, "ctf")]["_boot_iqm"] - agg_all[(algo, "do")]["_boot_iqm"]
        diff[algo] = (agg_all[(algo, "ctf")]["iqm"][0] - agg_all[(algo, "do")]["iqm"][0],
                      float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    rep = args.report
    (rep / "figures/fig_task6_curves.tex").write_text(fig_main(rows, curves, anc, args.run))
    (rep / "figures/fig_task6_curves_app.tex").write_text(fig_appendix_curves(curves, anc))
    (rep / "figures/fig_task6_profiles.tex").write_text(fig_profiles(scores))
    (rep / "tables/tab_task6_main_block.tex").write_text(tab_main_block(cells) + "\n")
    (rep / "tables/tab_task6_cells.tex").write_text(tab_cells(cells))
    (rep / "tables/tab_task6_aggregate.tex").write_text(tab_aggregate(agg_all, agg_drone, diff, poi))
    (rep / "tables/tab_task6_controls.tex").write_text(tab_controls(rows, refs, hints, xor, anc))
    print("wrote Task 6 figures/tables to", rep)


if __name__ == "__main__":
    main()
