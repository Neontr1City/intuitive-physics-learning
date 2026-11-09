---
name: develop-intuitive-physics
description: Advance Intuitive Physics Learning through research planning, experiment design or interpretation, project-relevant paper discussion, implementation, bug fixes, configuration, tests, or documentation. Do not use for behavior-preserving cleanup, unrelated paper summaries, or standalone workload execution.
---

# Advance Intuitive Physics Learning

Produce the smallest coherent implementation, decision, or experiment that advances the project.

## Maintain project memory

- When the invoking developer uses external project memory, read
  [references/project-memory.md](references/project-memory.md) before substantive work.
- Use their canonical external project model. Never hard-code a personal memory path or store
  personal project cognition in this repository.
- Use a light memory pass for a small local change and a full pass for architecture, experiment,
  or project-direction changes.

## Select the work mode

- For research planning, experiment design or interpretation, project-relevant paper reading, or
  technical discussion, read [references/research-workflow.md](references/research-workflow.md).
- For implementation, bug fixes, configuration, tests, or documentation, follow the sections below.
- When research and implementation are coupled, define the claim and required evidence before
  changing code, then keep the implementation scoped to that experiment or decision.

## Understand and implement the change

- Inspect the owning module, callers, configuration, and tests.
- State the behavior to add or fix and the public contract that must remain stable.
- Identify affected tensor shapes, devices, dtypes, and configuration fields.
- Keep the patch focused on the requested change. Use `$refactor-intuitive-physics` when
  simplification becomes the primary task.
- Place new behavior in functions that semantically own the affected data or responsibility.
- Avoid pass-through functions unless they establish a clear abstraction boundary.
- Extract a single-use helper only when it represents a clear, cohesive responsibility.
- Preserve public Python APIs, CLI behavior, configuration, serialized formats, and runtime
  artifacts unless the task explicitly changes them.
- Update tests and user-facing documentation when their contract changes.

## Defensive checks and tests

- Avoid checks whose conditions are provably always true within the function scope.
- Avoid defensive checks for unlikely or unsupported edge cases.
- Retain validation at public, external, destructive, or silent-corruption boundaries and for
  documented invariants.
- Add tests only for meaningful behavior or regressions; skip trivial tests.
- Prefer tests at stable public or behavioral boundaries over private helper choreography.

## Validation and completion

- Scale validation to the change. Start with the smallest relevant checks and tests.
- Before committing code changes, run the repository's configured checks and test suite.
  When configured, use `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest`.
- Inspect the final diff for unrelated changes, stale references, generated artifacts, and
  accidental contract changes.
- For research work, report the decision, hypotheses, experiment design or interpretation,
  evidence limits, and next discriminating action.
- Summarize validation performed, skipped checks, pre-existing failures, and remaining limitations.
