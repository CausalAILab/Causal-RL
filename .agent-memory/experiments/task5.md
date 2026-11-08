# Task 5 main · where to intervene

**Verified from saved artifacts.** `runs/benchmark1.0/task5/manifest.json`, identity `e2bc4f9116a2a480000208662ddf61e48add9b6cf26078ab3ab6bc21e0d8118b`. See [live config](../../configs/task5.json) and `git show aaed578:TASK5.md` for the original protocol. The original report command and module are deleted in the current worktree.

- Example 9.5, six-variable SCM, chain d=2/4/6, fork d=3/4/6, collider d=2/4/6. Full, MIS, POMIS, All-at-once use the same UCB with alpha 1. Horizon 1. Seeds 1000–1019. Runs use 10,000 interventions at d≤4 and 25,000 at d=6. Compare at 10,000 and show d=6 final results separately.
- Checked 880 complete status files and 880 summaries, totaling 12,400,000 interactions. At 10k, mean cumulative pseudo-regret from `jobs/**/summary.json` for Full/MIS/POMIS is 2,432.899/259.180/64.728 in chain d=6 (729/13/4 candidates) and 7,097.130/7,097.130/687.760 in collider d=6.
- At the actual final budget, all 660 Full/MIS/POMIS runs had zero simple regret (tolerance `1e-10`); 160 of 220 All-at-once runs had nonzero simple regret. The stored bootstrap configuration matches Task 4. The run bundle contains per-seed checkpoints and summaries.
- **Interpretation.** Smaller legal candidate sets can reduce exploration while retaining an optimum. All-at-once can lose the optimum. Equal final recommendations and lower cumulative regret answer different questions.
