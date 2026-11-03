# Task 6 DeepRL integration

- Use `causal_rl.wrappers.CTFWrapper` around native PCH environments for all new Task 6 DeepRL training and evaluation. Use `mode="ctf"` for natural-action policies and `mode="do"` for state-only controls.
- Keep experiment files limited to environment parameters, learner configuration and orchestration. Do not duplicate natural-action concatenation, noise caching or SCM stepping there.
- Supply `horizon=H` only when H defines the finite-horizon task. Use the same horizon for do/ctf controls. Do not bootstrap from padded terminal intentions or treat an arbitrary training-budget cutoff as termination.
- Preserve native rewards, policies and latent sampling. Add new environment support through its PCH contract and shared wrapper tests, never an experiment-specific causal adapter.
- The wrapper is an interaction adapter, not a replacement for auditing learner updates or environment assumptions. Do not label DeepRL training as a reproduction of textbook Ctf-UCBVI.
