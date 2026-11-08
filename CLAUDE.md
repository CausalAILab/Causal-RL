# Claude Code

@AGENTS.md

Claude-specific notes. Project knowledge lives only in [.agent-memory/](.agent-memory/README.md).

- Launch Claude Code at this repository or worktree root. From a parent folder, Claude Code loads this file only after it touches files here, so startup instructions can be missed.
- The memory files are linked, not imported. At session start, read `.agent-memory/PROJECT.md` and `.agent-memory/HANDOFF.md` with the Read tool.
- Claude Code's auto-memory and conversation transcripts are local conveniences, not project records. Write durable project knowledge to `.agent-memory/`.
- In the handoff, set **Agent** to `Claude Code (<model>)`.
