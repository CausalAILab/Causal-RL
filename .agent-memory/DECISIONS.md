# Decisions

## RL-01 Isolate the benchmark worktree and run inputs

**Decision · adopted.** Use an independent virtual environment for `benchmark1.0` (`76cc33c`) and frozen sources for resumable experiments (Task 5 in `aaed578`, also visible in saved Task 4 and Task 6 bundles). Reusing the older environment or resuming from mutable editable imports makes runs depend on worktree and package changes. Isolation requires recording both repository revisions, local source changes when applicable, and run identity. Keep the frozen bundle in approved artifact storage.

## RL-02 Native PCH wrapper for Task 6

**Decision · implemented.** Commit `3649a9e` introduced [CTFWrapper](../causal_rl/wrappers/ctf.py) and its original usage rules (`git show 3649a9e:AGENTS.md`). Use it around native PCH environments for Task 6 DeepRL training and evaluation. Use `mode="ctf"` for state plus natural action and `mode="do"` for the state-only control. Construct one wrapper per environment in each vector or worker factory.

The wrapper suspends the native callback with greenlet so the natural action and the executed action's outcome share the sampled context. The next observation uses the next native callback. Keep experiment code limited to configuration and orchestration. Put new causal interaction support in the native PCH contract and shared wrapper tests, preserving rewards, policies and latent sampling. Per-experiment causal adapters would risk resampling context or changing the task. The wrapper requires a discrete action space and a flattenable observation space.

Pass `horizon=H` only when the task is finite horizon, with the same value in both modes. At that boundary the wrapper terminates, and a learner must mask bootstrapping from padded terminal intentions. A training-budget cutoff is distinct from a task terminal. Native ctf truncation without a next intention is rejected outside the configured horizon boundary. DeepRL still requires auditing learner updates, environment assumptions, and whether the input is Markov. This adapter carries no claim of reproducing textbook Ctf-UCBVI. The later prepare/execute change `578420a` was reverted by `d58d996`.

## RL-03 Constrain Task 5 to legal interventions and matched feedback

**Decision · used in the saved main run.** Hold the UCB learner fixed and change its candidate set. The six-variable reference SCM permits only B/D/E, so the driver encodes reviewed legal MIS/POMIS scopes. The unrestricted solver would otherwise consider illegal targets. Use exact arm means for scoring only. Pair methods within environment and seed using the same exogenous stream, and score them at a common budget. See [the saved run record](experiments/task5.md).
