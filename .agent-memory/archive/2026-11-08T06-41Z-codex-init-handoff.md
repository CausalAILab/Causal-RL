# Archived handoff · Codex memory initialization

- **Last updated** — 2026-11-08T06:41:59Z
- **Agent** — OpenAI Codex
- **Branch and base** — `benchmark1.0`, HEAD `668afa139449631af0ae77368cd54a317d79c8cd`. The matching Causal-Gymnasium worktree is `ey/rescue-drone` at `6802571` with local changes.
- **Task objective** — Make project context and evidence available for handoffs in both directions between Codex and Claude Code.
- **Completed work** — Initialized project memory and the common startup and Git protocol. Recovered run scope and results from saved artifacts. The Task 6 main bundle supersedes older pilot-only status for the settings it covers. See the [experiment index](../EXPERIMENTS.md).
- **Modified files** — Root `AGENTS.md`, `CLAUDE.md`, and `.agent-memory/`. The prior Task 6-only `AGENTS.md` was already locally deleted. Its guidance now lives in [RL-02](../DECISIONS.md#rl-02-native-pch-wrapper-for-task-6), with the original in Git. This work recreates the root path for the new shared instructions.
- **Tests and verification** — `.venv/bin/python -m unittest discover -s tests -v`: 19 passed, including two 32-step learner smoke checks. Read-only checks verified Task 4–6 saved status counts and recomputed selected metrics. Startup paths and the Claude import were checked against the client rules. The local `codex debug prompt-input` probe hit a sandbox permission error. A live Claude session was outside this check.
- **Outstanding issues** — Existing local work includes Task 4/6 runners, Task 5 driver/test edits and report/doc deletions, and LavaCrossing reward-shaping data and notebook edits. Confirm which changes to carry into a reviewed commit. Run bundles are local and ignored. The Task 7 launch remains unverified, and Task 6 extensions are outside the saved main run.
- **Next actions** — (1) Inspect current Git status in both repos. (2) Confirm the approved durable location for run bundles and whether the Task 5 report deletion is intentional. (3) Confirm the current Task 7 protocol and any Task 6 extension priorities before additional training. Integrate memory through the [Git protocol](../README.md#synchronization) only with authorization.
