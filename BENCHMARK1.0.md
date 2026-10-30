# benchmark1.0 preparation

- Base: Causal-RL `75471c76341c80d921154238bde1ec1a5dcdc22d`.
- Retains all current `causal_rl` implementations, including local BC/GAIL fixes and the merged Causal SQIL / Causal IQ-Learn implementations.
- Retains the local namespace-package discovery fix in `setup.py`.
- Generic CleanRL baselines are preserved unchanged under `third_party/cleanrl/`; upstream revision, hashes, and license are included there. Their runtime dependencies have not been installed in this worktree.
- Excludes benchmark0.1 experiment runners, training launchers, runtime requirement files, results, checkpoints, and the old virtual environment.
- Local environment dependency: sibling `Causal-Gymnasium` working tree, branch `ey/rescue-drone`, base commit `6802571040299f28bf4d91ea8fbd85ca1d4f1788`, including its uncommitted environment updates. The new WhereDo environments currently live there as `wheredo_experiemnt.py` and `wheredo_synthetic.py`; they are not part of this Git repository.
- Task 5 training/reporting entry point: `scripts/task5.sh`; configuration and usage: `configs/task5.json` and `TASK5.md`. Main training is a separate explicit `run` command.
