# Task 5: training and reporting

## Fixed main experiment

| Item | Setting |
|---|---|
| Environments | Example 9.5; six-variable; chain d=2/4/6; fork d=3/4/6; collider d=2/4/6 |
| Methods | Full, MIS, POMIS, All-at-once |
| Learner | Existing `causal_rl.algo.ctf_do.ctfucb.UCB`, alpha=1 |
| Budget | d<=4: 10,000; d=6: 25,000 interventions per run |
| Repeats | Seeds 1000–1019; 880 runs; 12.4M training interactions |
| Horizon | H=1; one fresh SCM unit per intervention |
| Checkpoints | 500/1k/2k/5k/10k; d=6 also 15k/20k/25k |
| Evaluation | Exact SCM arm means; zero extra evaluation interactions |
| Stopping | Complete each fixed budget; no convergence stopping or best-checkpoint selection |

Configuration: `configs/task5.json`. Structural equations and noise probabilities come unchanged from the local Causal-Gymnasium implementations. The six-variable environment only permits B/D/E; its reviewed legal reference MIS/POMIS scopes are explicit in the driver. Other scopes come from the existing local WhereDo solver.

Arms have a deterministic order: variables in environment order, assignments in `(None, 0, 1)` product order; candidates retain their full-space IDs. The unmodified local UCB handles exploration and ties. UCB sees only candidate IDs and the executed reward. Oracle means, post-intervention variables, and hidden noise are never learner inputs. The recommended arm maximizes the learner's empirical mean among tried arms; first arm wins ties.

Within an environment/seed, methods deliberately share the sequence of exogenous units through recorded 64-bit episode seeds. Different environment/seed pairs have separate structured RNG streams. Bootstrap resampling preserves these method pairs and each run's complete trajectory.

## Commands

Run from this checkout:

```bash
./scripts/task5.sh plan                         # check counts, legal arms and exact optima
./scripts/task5.sh smoke --workers 4            # 44 development runs, seed 20, 1,024 pulls each
./scripts/task5.sh run --workers 4              # start/resume the full main experiment
./scripts/task5.sh status
./scripts/task5.sh report                      # generate formal statistical outputs after completion
```

The launcher uses an active virtual environment, this checkout's `.venv`, or the existing sibling `Causal-RL/.venv`, in that order. `TASK5_PYTHON=/absolute/path/to/python` overrides this choice. The default environment source is sibling `Causal-Gymnasium`; override with `CAUSAL_GYM_SOURCE=/path/to/repo`. No dependency installation or network access occurs. Four CPU processes is the default; the local UCB implementation is used directly without vectorizing or replacing it.

Main outputs: `runs/benchmark1.0/task5/`. Development outputs: `runs/benchmark1.0/task5_smoke/`. `--output /absolute/path` selects a different bundle. Main and development results cannot share a bundle.

To run selected jobs while retaining the full 880-job reporting plan:

```bash
./scripts/task5.sh run --environments example_9_5 --methods full mis pomis all_at_once --seeds 1000 1001
```

## Recovery and provenance

The first launch freezes local Causal-RL, Causal-Gymnasium source/assets, scripts, configuration, package versions and hashes. Workers execute that saved snapshot. A later launch uses the same snapshot, even if the live repositories changed. Changing the experiment configuration requires a new output directory. Snapshot corruption or changed Python/NumPy/Gymnasium/NetworkX versions blocks resume.

Each job saves an atomic checkpoint every 500 pulls, including local UCB counts/values/total, RNG state, selected arm, reward, episode seed, per-pull global/own-set pseudo-regret, and checkpoint recommendations. Resume verifies identity and recomputes sufficient statistics from the saved trajectory; an interrupted unfinished block is replayed from its last durable checkpoint. Complete jobs retain their trajectories. An output-level lock prevents concurrent launchers from writing the same experiment.

```bash
./scripts/task5.sh pause                       # cooperative pause; no result deletion
./scripts/task5.sh run --clear-pause --workers 4
```

Ctrl-C also requests a pause. Inspect `jobs/<environment>/<method>/<seed>/status.json` and `error.txt` for failures. A crash stays missing evidence, never a zero score. Results are not uploaded or pushed.

## Statistical outputs

`report` writes `reports/main/` only after every planned run completes. With missing runs it exports available per-seed records and an incomplete-job inventory, and withholds full-suite tables/figures. Development/`--quick` outputs go to `reports/diagnostic/` and are explicitly labeled.

- `per_seed.csv`: all configured checkpoints, all independent runs.
- `endpoints.csv` / `endpoints.tex`: native cumulative pseudo-regret at the common 10k checkpoint; d=6 25k endpoints separately. CSV also includes own-set regret, candidate-set value gap, final simple regret, optimal-arm fraction over the last 1,000 pulls, and paired regret differences versus Full.
- `learning_curves.pdf` / `.png`: all eleven environments; mean, faint individual runs, pointwise 95% bootstrap bands.
- `suite.csv` / `suite.tex` / `suite.pdf`: IQM and gap to target 1, using all eleven environments at 10k with equal environment weight.
- `profiles.csv`: run-score performance profiles and pointwise confidence bands; also plotted in `suite.pdf`.
- `probability_of_improvement.csv`: every method pair; all cross-run pairs within each environment, ties=0.5.
- `report_manifest.json`: completed/planned runs, analysis settings and provenance.

Per-environment tables use mean [95% percentile bootstrap CI]. Suite resampling keeps the suite fixed and samples runs within environments, preserving shared run IDs across methods. Main settings: 50,000 replicates for aggregate scores, endpoint means and paired effects; 2,000 for curves, profiles and probability of improvement; RNG seed 314159. Independent RNG substreams separate the outputs. No SE bars, seed-win-rate substitute, or zero imputation.

Suite score is `q = 1 - R_T / [T * (mu_full_best - mu_full_worst)]`, with no clipping. IQM trims the bottom/top quartiles of pooled equally weighted run scores. Gap is `mean(max(1-q, 0))`. Anchors come from the full legal intervention space and are shared by all methods. Statistical definitions follow Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, Sections 3–4 and Appendix A; implementation uses NumPy, with estimator regression tests.

Tests (no main training):

```bash
PYTHONDONTWRITEBYTECODE=1 ../Causal-RL/.venv/bin/python -m unittest discover -s tests -v
```
