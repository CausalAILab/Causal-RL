# Task 6 · counterfactual decision making

## Main

**Verified from saved artifacts.** `runs/task6/main/manifest.json`, design `2026-11-04-main`, identity `53253e77d4947c2d2a74af8c09f050296b9fc04215f17e2508d823367a3bebea`. Saved `config.json`, `snapshot/`, `preflight.json`, `status.json`, `checkpoints.csv` and `results.csv` are the bundle's evidence. The live `experiments/task6/` runner and `scripts/task6*.sh` were local at initialization.

- Verified 640 complete statuses and 640 result rows. Seeds 1000–1009, including XOR. Six positive sequential settings (RobotWalk, balanced MDP, Drone U2-H4/B2-H4/R4-H8/R6-H10) plus calm CartPole. DQN/C51/PPO each use state-only and natural-action input. Random and follow-natural references, with random-hint in a subset. XOR compares native UCB and Ctf-UCB.
- Short schedule 102,400 transitions, long Drone schedule 1,048,576, XOR 10,000 pulls. Sequential evaluations use 100 episodes at intermediate checkpoints and 1,000 final episodes, with a sampled PPO final diagnostic. See the saved config for exact distributions, start distance and horizon.
- Means recomputed across seeds from `results.csv`: XOR UCB → Ctf-UCB mean reward 0.29966 → 0.44483. DQN State → State + Natural mean return 5.5000 → 10.0000 on RobotWalk, 9.9904 → 18.0024 on balanced MDP. DQN final Drone rescue rates: 59.39% → 78.62% (U2), 80.95% → 85.04% (B2), 76.49% → 91.87% (R4), 78.70% → 90.19% (R6). These are descriptive means. Use paired uncertainty for inferential claims.
- The batch excludes disabled Lava S5 and has no Highway jobs. The lab manuscript `overleaf_report/sections/tasks/task6.tex` (workspace, outside Git) reports the six positives separately from CartPole and XOR. Check exact estimands and artifacts before quoting draft aggregate or oracle claims.

## Development

- `runs/task6_pilot_20261103/`: 168 completed jobs and result rows, seeds 20–22; RobotWalk/MDP use 16,384 transitions and Drone 32,768; 256 final evaluation episodes per sequential run. Development evidence.
- `runs/task6/smoke/`: 64 complete statuses and result rows. Keep separate from the main run.
