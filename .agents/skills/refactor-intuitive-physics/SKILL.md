---
name: refactor-intuitive-physics
description: Simplify and consolidate existing Intuitive Physics Learning code without changing required behavior. Use for cleanup, de-duplication, dead-code removal, wrapper collapse, module consolidation, or preparing an experimental implementation for review. Do not use when the primary task adds new behavior, changes a public contract, formats code only, or requests diagnosis without a fix.
---

# Refactor Intuitive Physics Learning

Leave one coherent implementation that is easier to understand, test, and change. Preserve required
behavior and refactor from evidence, not aesthetics.

## Protect the contract

- Inspect the code, callers, tests, entry points, configuration, documentation, and nearby style.
- Preserve unrelated work already present in the worktree.
- Preserve observable behavior unless the task explicitly changes it, including public APIs,
  CLI contracts, configuration defaults, serialized formats, runtime artifacts, and tensor semantics.
- Treat tests and runtime output as evidence of behavior, not proof that the current structure must survive.
- Surface unresolved consumer or compatibility decisions that affect the refactor.

## Establish a baseline

- State the refactor scope and the behaviors and artifacts that must survive.
- Run the cheapest useful checks before editing. Record pre-existing failures separately from regressions.
- Identify dead code, duplicate paths, forwarding wrappers, temporary adapters, stale exports,
  and obsolete compatibility layers.
- Distinguish required compatibility from accidental implementation structure.

## Choose the target shape

- Prefer one supported pipeline or entry point for each responsibility.
- Group code by responsibility and lifecycle rather than by the order patches were added.
- Avoid tiny packages, one-function modules, and wrappers unless they create a real boundary or
  improve locality.
- Abstract duplicate code only when it represents the same knowledge and should change together.
- Put behavior in its semantic owner.
- Keep a single-use helper only when it isolates a cohesive concept, invariant, or side effect.

## Implement in verified steps

- Replace superseded paths instead of wrapping them with another transition layer.
- Before deleting a path, check callers, imports, configuration, documentation, tests, and runtime
  discovery. Remove stale references with the implementation.
- Preserve validation at external, destructive, public, or silent-corruption boundaries.
  Remove duplicated checks and checks already guaranteed by an earlier boundary.
- Match repository naming, typing, configuration, logging, and error-handling conventions.
- Use domain names. Avoid `new`, `v2`, `fixed`, `temp`, or `compat` unless they express a durable contract.
- Add comments only for contracts, units, shapes, side effects, algorithms, or non-obvious constraints.
- Run focused checks after meaningful steps.

## Align tests and documentation

- Test stable behavior and public interfaces rather than private helper choreography.
- Add a focused regression test when the refactor exposes a meaningful behavior gap.
- Preserve useful coverage. Rewrite or remove a test only when its boundary disappeared or another
  test covers the same contract more directly.
- Update documentation to describe the final architecture, entry points, and commands.

## Verify and finish

- Scale validation to the risk. Before committing code changes, run the repository's configured
  checks and test suite. When configured, use `uv run ruff check .`,
  `uv run ruff format --check .`, and `uv run pytest`.
- When tensor or data pipelines change, verify provenance, shape, device, dtype, alignment,
  and a semantic round trip where applicable.
- Inspect the final diff for stale names, old entry points, orphan imports, obsolete references,
  generated artifacts, and unrelated changes.
- Compare the before and after structure without optimizing for line count alone.
- If a commit is requested, stage only the intended files and inspect the staged diff.
- Report what was simplified, what behavior was preserved, changed, or retired, which obsolete
  paths were removed, checks and results, and remaining limitations.
