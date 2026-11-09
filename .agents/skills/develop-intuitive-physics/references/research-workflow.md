# Research planning and experiment workflow

Turn important uncertainty into explicit decisions and discriminating evidence, not a generic list
of ideas.

## Plan and design

- State the project decision or uncertainty this work must resolve and why it matters now.
- Check current capabilities, constraints, evidence, and open questions before proposing new work.
- Express the central claim as a falsifiable hypothesis. State plausible alternatives and what
  observation would distinguish them.
- Prefer the cheapest experiment or analysis that can distinguish the leading hypotheses.
- Sequence work from prerequisite and sanity checks to mechanism isolation, controlled comparison,
  and only then broader evaluation.
- Before an important experiment, specify the variables, baselines, controls, data split, seeds,
  comparison budget, metrics, uncertainty estimates, and success, failure, or inconclusive criteria.
- Record configuration, code version, output artifacts, expected cost, and stop conditions.
- Keep comparison budgets fair. Check data leakage and validate the measurement path before the main run.
- Do not assume a team member's compute environment.
- Stop extending the plan when the next discriminating action is clear enough to execute.

## Interpret results and use papers

- Verify run integrity before interpreting the metric.
- Separate infrastructure failure, implementation failure, unsupported hypothesis, and inconclusive evidence.
- Do not reinterpret success criteria after seeing the result.
- Check variability, alternative explanations, confounds, and metric artifacts.
- State what the result supports, contradicts, and does not establish.
- Read papers to resolve a project question, evaluate a mechanism, or design an experiment.
- Before recommending a method or dataset, establish mechanism fit, assumption fit, a concrete
  insertion point, expected value, and a discriminating test.
- Create separate paper notes only when requested.
- Feed material results, constraints, and belief changes back into the canonical project model,
  when the developer uses one.
