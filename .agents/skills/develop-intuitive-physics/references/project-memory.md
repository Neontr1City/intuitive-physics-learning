# Project memory workflow

Maintain the invoking developer's project understanding as one durable, revisable external model.
Treat it as working memory and a user-alignment object, not a static summary or activity log.

## Locate and read the model

- Use the developer-designated external memory. If its location is unclear, ask where it lives.
- Reuse the existing canonical model. Never create a competing parallel memory or put personal
  project cognition in the source repository.
- If the developer does not use external memory, work from the conversation, repository, and
  relevant artifacts without requiring a new memory file.
- Read the objective, current mechanism, active hypotheses, implementation state, open questions,
  and next discriminating work.
- Check whether the model is stale relative to the code, experiments, or decisions in scope.

## Revise carefully

- Separate project intent, code facts, runtime evidence, external evidence, inference, and hypothesis.
- Do not promote intent, code presence, successful import, or successful execution into empirical success.
- Surface conflicts between memory, code, tests, runtime results, and developer decisions.
- After substantive work, check whether claims, confidence, implementation state, priorities, or
  next actions materially changed. If not, leave the model unchanged.
- Otherwise update the relevant sections in place. Preserve assumptions, evidence limits,
  consequential decisions, and useful negative results.
- Link exact code, notes, or experiment artifacts when they improve auditability.
- Synthesize; do not append chat transcripts, raw logs, or duplicated code documentation.
- Report what changed, the decisive evidence or reasoning, its practical consequence, remaining
  uncertainty, and the memory location updated.
