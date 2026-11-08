# Repository instructions

Use [.agent-memory/](.agent-memory/README.md) as the canonical project memory.

- Start at the Git root. Read [.agent-memory/PROJECT.md](.agent-memory/PROJECT.md) and [.agent-memory/HANDOFF.md](.agent-memory/HANDOFF.md). Check the branch, HEAD, staged and unstaged changes, and any source or artifacts relevant to the task. Identify active tasks and constraints. Report meaningful discrepancies before relying on the memory.
- Consult [.agent-memory/DECISIONS.md](.agent-memory/DECISIONS.md) before changing established behavior, [.agent-memory/RESEARCH.md](.agent-memory/RESEARCH.md) for research work, and [.agent-memory/EXPERIMENTS.md](.agent-memory/EXPERIMENTS.md) for run provenance. Record significant decisions and keep observations separate from interpretations.
- After meaningful work, update the handoff and any other entry that materially changed. Record exact verification and its limits. Follow the shared [handoff and Git protocol](.agent-memory/README.md).
- Preserve existing work. Keep secrets, private session material and large artifacts out of memory. Get explicit authorization before committing, merging, pushing, or modifying a remote.

For Task 6 DeepRL, apply the native PCH wrapper rules in [.agent-memory/DECISIONS.md](.agent-memory/DECISIONS.md#rl-02-native-pch-wrapper-for-task-6).
