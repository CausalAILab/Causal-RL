# Task 4 main · confounded offline-to-online bandits

**Verified from saved artifacts.** `runs/task4/main/manifest.json`, identity `c225aff60a66007c08fd6cfb9b683f762fc1c214b12fef03995967dbe2356706`. Frozen config in `config.json` and source in `snapshot/`. Live launcher `scripts/task4.sh` and driver `experiments/task4/run.py` were local at initialization.

- Design `2026-10-31-replacement`: horizon 1, 3 MABCOOL reward-gap and 4 configurable MAB settings, 3 methods (online UCB, direct transfer, COOL). Seeds 2000–2019, 5,000 offline samples per environment/seed, 10,000 online steps per run, bound alpha 0.05. The transfer methods share offline data. All three methods share online unit seeds within an environment/seed.
- Checked 420 complete status files and 420 saved summaries. Per-configuration means from final cumulative pseudo-regret: COOL 0 in every saved run, direct transfer 1,000–3,000, UCB 24.285–55.905. In `mabcool_gap025`, UCB/direct/COOL means are 31.1/2,500/0.
- The saved bootstrap configuration uses 50,000 aggregate and 2,000 curve/profile/probability replicates, RNG seed 314159. The means above were recomputed from `jobs/**/summary.json`; intervals were not recomputed.
- **Interpretation.** The result concerns these bandit mechanisms, offline data sizes and bounds. Zero observed regret is configuration-specific. The exact interventional means are used for scoring.
