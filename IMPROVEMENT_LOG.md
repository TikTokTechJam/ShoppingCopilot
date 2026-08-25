# Agent Improvement Log

This log records how the shopping agent evolves during development. Each entry explains what changed, why it changed, how it was evaluated, and what the results mean.

## How to read this log

Each meaningful agent change should include:

- the date and commit;
- the behavior or component that changed;
- the reason for the change;
- the expected impact;
- evaluation results before and after the change;
- evidence, observations, and the next step.

## Baseline

The repository starts from the weak BM25 starter agent in [`starter/agent.py`](starter/agent.py). The challenge-provided reference results are:

| Metric | Baseline |
| --- | ---: |
| Hit Rate@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |

These values are the published starter reference from the challenge README and were not rerun as part of this documentation change.

## Improvement entries

Add the newest entry at the top of this section.

### YYYY-MM-DD — Short description

- **Commit:** `commit-sha`
- **Change:** What behavior, data flow, or component changed?
- **Why:** What limitation or observation motivated the change?
- **Expected impact:** Which scenario or metric should improve, and why?
- **Evaluation:** Record the command, dataset, and before/after Hit Rate@10, MRR, and MTTC.
- **Evidence:** Link to relevant code, result files, or analysis.
- **Result and next step:** Explain whether the change helped, regressed, or was inconclusive, and what will be tried next.
