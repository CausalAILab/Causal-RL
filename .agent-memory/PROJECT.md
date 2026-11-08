# Causal-RL

Causal-RL implements baselines and research experiments for causal decision making. This memory begins on `benchmark1.0`. The linked worktree normally named `Causal-RL-benchmark1.0` is distinct from the older `Causal-RL` checkout used for `benchmark0.1`.

## Code map

| Area | Entry points |
| --- | --- |
| Basic estimators and exploration | `causal_rl/algo/baselines/`, `causal_rl/algo/cool/cool.py` (bandit transfer and bounds) |
| Where to intervene | `causal_rl/algo/where_do/`, `configs/task5.json`, `experiments/task5.py`, `scripts/task5.sh` |
| Counterfactual decisions | `causal_rl/algo/ctf_do/`, `causal_rl/wrappers/ctf.py`, `tests/test_ctf_wrapper.py` |
| Imitation | `causal_rl/algo/imitation/` with BC, GAIL, SQIL and IQ-Learn paths. Commit `668afa1` adds SCIL training options to SQIL and IQ-Learn. |
| Reward shaping | `causal_rl/algo/reward_shaping/` and `examples/reward_shaping/` |
| Baseline provenance | [CleanRL snapshot](../third_party/cleanrl/PROVENANCE.json) pins upstream revision `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`. |

Research task numbering in this project is Task 4 offline-to-online, Task 5 intervention selection, Task 6 counterfactual decision making, and Task 7 causal imitation. See the [experiment index](EXPERIMENTS.md) for executed scope.

## Environment boundary

Use this checkout's independent `.venv` and verify editable imports. Gymnasium, NumPy, NetworkX and greenlet are among the dependencies declared in [setup.py](../setup.py). DeepRL and imitation also need their learner-specific packages. The companion repository Causal-Gymnasium owns the SCMs and native `see`, `do`, and `ctf_do` policy callbacks. Its matching checkout is `ey/rescue-drone` in this workspace. Some local environment functionality comes from pending changes there, so a commit SHA alone may be insufficient to reproduce a run. Use the run's frozen source and manifest for historical reproduction.
