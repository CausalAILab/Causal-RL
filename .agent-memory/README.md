# Shared project memory

This directory travels with the repository and serves both Codex and Claude Code. Start either client at the repository root. Codex discovers `AGENTS.md`. Claude Code loads `CLAUDE.md`, which imports the same instructions. Keep shared instructions in `AGENTS.md` and project knowledge here.

| Location | Owns |
| --- | --- |
| [PROJECT.md](PROJECT.md) | Scope, architecture, entry points, dependencies and conventions |
| [DECISIONS.md](DECISIONS.md) | Significant choices, alternatives, rationale, consequences and status |
| [RESEARCH.md](RESEARCH.md) | Questions, hypotheses, assumptions and limits |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Run index, coverage, provenance and results pointers |
| [HANDOFF.md](HANDOFF.md) | Latest work, current checkout state and next actions |
| [experiments/](experiments/) | Small records keyed by stable experiment or demonstration ID |
| [archive/](archive/) | Superseded context worth retaining outside the active view |

## Session and evidence rules

1. At startup read `PROJECT.md` and `HANDOFF.md`. Run `git status --short --branch` and `git rev-parse HEAD`. Inspect relevant diffs, sources and artifacts, including the companion repository when a task depends on it. Report meaningful discrepancies with the handoff. Consult the other files as needed.
2. During work, follow established decisions. When a change is justified, record the evidence, alternative and tradeoff in `DECISIONS.md`. Label claims as **Verified** (name the source and revision or artifact), **Decision** (state scope and status), **Hypothesis/interpretation**, or **Unverified**. Use current code and run artifacts for the claims they actually support. Treat previous notes and reports as leads to verify.
3. After meaningful work, update the current handoff. Edit other records only when their contents materially change. Use paths relative to the repository root for code, stable experiment IDs and manifest digests for runs, and commit IDs where available. State when something is local, uncommitted, external, or unavailable. For experiments, record exact configuration or its authoritative path, budgets, seeds, metrics, aggregation, results, and caveats. Treat Git history as the audit trail and link to existing evidence.
4. Keep project memory compact. Store approved pointers to artifacts instead of copying checkpoints, raw datasets, large logs or conversation transcripts. Exclude credentials, private keys, sensitive personal information, restricted datasets and private agent/session storage. Mark partial runs, missing results and test failures explicitly.

## Handoff format

Use the same fields for Codex → Claude Code and Claude Code → Codex. Replace the current entry when handing off, preserving still-active issues and next actions.

- **Last updated** — ISO 8601 timestamp with timezone.
- **Agent** — responsible agent.
- **Branch and base** — branch and current HEAD, with other relevant revisions or uncommitted dependencies.
- **Task objective** — intended outcome.
- **Completed work** — what was delivered and where to find it.
- **Modified files** — paths owned by this handoff, separate from pre-existing edits.
- **Tests and verification** — commands or checks actually run, outcomes, and any untested scope.
- **Outstanding issues** — open questions, blockers, and ownership if known.
- **Next actions** — ordered, concrete actions and any required approvals.

## Synchronization

1. For simultaneous work, use separate branches or worktrees. Begin from a known base. In a linked worktree Git shares commits and refs with the main checkout, while local edits stay in that worktree.
2. Before passing work over, review `git status --short --branch`, `git diff`, and `git diff --cached`. Update the handoff alongside the related code or research change. In the same checkout, the receiver can read the pending files directly. Across checkouts, give the receiver the source worktree or an explicitly approved commit reference. A clean checkout sees only tracked history.
3. With explicit authorization, stage only the intended files, commit them with the related work, and integrate through the agreed branch or review process. Push or change a remote only when authorized. The receiving agent verifies the source/base and reviews incoming changes before integrating them. Never reset, stash, clean, or overwrite another agent's unmerged work to synchronize.
4. If both sides edited memory, reconcile the handoff with the actual integrated tree and keep each side's unresolved work. Reconcile decisions and results against their evidence, using stable IDs and a single record for each run. Escalate genuine conflicts over ownership, scope or evidence to the user.
