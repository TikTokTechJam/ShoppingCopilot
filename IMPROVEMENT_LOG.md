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

### 2026-08-26 — Fixed Manual400 benchmark source of truth

- **Commit:** this PR (manual400 benchmark)
- **Change:** Treat the already curated Manual400 files under `data/derived/manual400/` as immutable benchmark data. The evaluator reads fixed sessions, labels, audits, debug records, and the report; the full catalog is used only for target-ASIN validation and Agent retrieval.
- **Why:** Judges need a stable, inspectable benchmark whose labels and hidden cards do not change when evaluation runs. Static validation checks cross-file consistency, evidence, scenario counts, and customer-facing text safety.
- **Expected impact:** Make failures actionable without rewarding raw metadata leakage or noisy n-grams. The evaluator preserves the ten-turn limit, strict Agent schema, exact `parent_asin` equality, and the official `HitRate@10`, `MRR`, `MTTC`, `Efficiency`, and `TechnicalScore` metrics.
- **Evaluation:** `python -m scripts.validate_manual400` validates the committed artifacts, `python -m unittest discover -s tests -p "test*.py"` runs the repository tests, and `python -m evaluator.manual400_evaluator --output results_manual400.json` evaluates the starter Agent against all 400 fixed sessions.

| Metric | Final manual400 result |
| --- | ---: |
| Hit Rate@10 | 0.707500 |
| MRR | 0.333712 |
| MTTC | 5.135 |
| Efficiency | 0.586500 |
| TechnicalScore | 0.571164 |

- **Dataset evidence:** The committed report records 400 unique targets, 400/400 passing label audits, and hidden cards of 2/3/4 facts in 123/143/134 sessions. The benchmark artifacts remain separate from the unchanged catalog and public set, and this PR removes the former builder path.
- **Scenario evidence:** Boundary `0.450 / 0.157222222 / 7.150`, Browsing `0.68125 / 0.312539683 / 5.300`, Buying `0.78125 / 0.357368552 / 4.294`, and Intent Override `0.666667 / 0.385919312 / 6.267` for Hit Rate@10 / MRR / MTTC.
- **Result and next step:** The benchmark is now reproducible by reading fixed files rather than rebuilding them. After the validation and evaluation rerun, the next agent iteration should improve stale-constraint replacement and ranking after an override.

### 2026-08-25 — Expected-utility adaptive search

- **Commit:** [`50d010d`](https://github.com/TikTokTechJam/ShoppingCopilot/commit/50d010d7a6c2d5fe07957e55502f920a8c743e62)
- **Change:** Replaced stateless BM25 retrieval with a bounded adaptive-search policy. The agent maintains the active conversation constraints, retrieves an 800-product lexical posterior, estimates expected Top-10 concentration and information gain for each unused attribute, and asks the highest-utility question. It also handles explicit no-preference answers and resets stale constraints after an intent override.
- **Why:** The starter agent returned recommendations without asking questions and lost earlier context after each customer reply. That made it unable to use the evaluator's simulated clarification branches.
- **Expected impact:** Improve Hit Rate@10 and MRR by acquiring material, feature, color, style, size, use-case, budget, or brand evidence while reducing the number of turns needed to identify the target.
- **Evaluation:** `python -m evaluator.local_evaluator` on all 200 public sessions, using the same catalog and dataset as the baseline.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.125 | 0.765 | +0.640 |
| MRR | 0.068034 | 0.495623 | +0.427589 |
| MTTC | 9.810 | 5.145 | -4.665 turns |
| Recommended technical score | 0.106710 | 0.648287 | +0.541577 |

Scenario results after the change:

| Scenario | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: |
| Boundary | 0.700 | 0.633333 | 5.500 |
| Browsing | 0.7875 | 0.507326 | 5.150 |
| Buying | 0.825 | 0.498710 | 4.325 |
| Intent override | 0.566667 | 0.410278 | 7.200 |

- **Evidence:** [`starter/agent.py`](starter/agent.py) contains the candidate-pool posterior approximation and query-utility calculation. The policy is a one-step bounded approximation, not an exact dynamic program over all 50,000-product subsets.
- **Result and next step:** The change improved all three overall metrics substantially. Intent override remains the weakest scenario, so the next iteration should focus on preserving useful category evidence while replacing stale preference terms more selectively.

Add the newest entry at the top of this section.

### YYYY-MM-DD — Short description

- **Commit:** `commit-sha`
- **Change:** What behavior, data flow, or component changed?
- **Why:** What limitation or observation motivated the change?
- **Expected impact:** Which scenario or metric should improve, and why?
- **Evaluation:** Record the command, dataset, and before/after Hit Rate@10, MRR, and MTTC.
- **Evidence:** Link to relevant code, result files, or analysis.
- **Result and next step:** Explain whether the change helped, regressed, or was inconclusive, and what will be tried next.
