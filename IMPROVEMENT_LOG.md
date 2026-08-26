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

### 2026-08-26 — Hard evaluator for expected-utility adaptive search

- **Commit:** this documentation PR; evaluator implementation is in [`evaluator/hard_evaluator.py`](evaluator/hard_evaluator.py)
- **Change:** Added a fixed GPTAnnotation hard benchmark with 400 sessions across Buying (160), Browsing (160), Intent Override (60), and Boundary (20). The evaluator simulates replies from committed hidden facts, validates Agent responses strictly, and scores exact catalog `parent_asin` recommendations under the ten-turn protocol.
- **Why:** The public-set score alone does not show whether an adaptive question policy can acquire useful evidence, handle no-preference branches, or replace stale constraints after an intent override. A fixed hard benchmark makes those behaviors reproducible and inspectable.
- **Expected impact:** Measure whether expected-utility query selection improves Top-10 identification and time-to-correctness across different conversational conditions without rebuilding or semantically relabeling the benchmark at evaluation time.
- **Evaluation:** `python -m evaluator.hard_evaluator` on all 400 fixed GPTAnnotation sessions, using `data/catalog.jsonl` as the Agent retrieval universe and exact target-ASIN validation source. The evaluator runs the Agent for at most 10 turns and reports Hit Rate@10, MRR, MTTC, Efficiency, TechnicalScore, and per-scenario metrics.

| Metric | Full hard-benchmark result |
| --- | ---: |
| Sessions | 400 |
| Hit Rate@10 | 0.507500 |
| MRR | 0.163393 |
| MTTC | 7.5425 |
| Efficiency | 0.345750 |
| TechnicalScore | 0.371918 |

Scenario results from the full hard-evaluator run:

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Buying | 160 | 0.568750 | 0.179975 | 6.7000 | 0.430000 | 0.424368 |
| Browsing | 160 | 0.487500 | 0.163177 | 8.08125 | 0.291875 | 0.351078 |
| Intent Override | 60 | 0.416667 | 0.129213 | 8.166667 | 0.283333 | 0.303764 |
| Boundary | 20 | 0.450000 | 0.135000 | 8.1000 | 0.290000 | 0.323500 |
- **Evidence:** The benchmark has 400 unique catalog targets and the required scenario counts. The evaluator uses attribute-scoped fact IDs, fixed evidence fields, exact ASIN equality, and process-local simulation state; it does not reconstruct ontology facts from the catalog.
- **Result and next step:** The full benchmark confirms that Buying is the strongest scenario, while Intent Override is currently weakest by Hit Rate@10, MRR, MTTC, and TechnicalScore. The next Agent iteration should focus on stale-constraint replacement and boundary/no-preference handling, then rerun all 400 sessions.

### 2026-08-25 — Expected-utility adaptive search

- **Commit:** [`50d010d`](https://github.com/TikTokTechJam/ShoppingCopilot/commit/50d010d7a6c2d5fe07957e55502f920a8c743e62)
- **Change:** Replaced stateless BM25 retrieval with a bounded adaptive-search policy. The agent maintains the active conversation constraints, retrieves an 800-product lexical posterior, estimates expected Top-10 concentration and information gain for each unused attribute, and asks the highest-utility question. It also handles explicit no-preference answers and resets stale constraints after an intent override.
- **Why:** The starter agent returned recommendations without asking questions and lost earlier context after each customer reply. That made it unable to use the evaluator's simulated clarification branches.
- **Expected impact:** Improve Hit Rate@10 and MRR by acquiring material, feature, color, style, size, use-case, budget, or brand evidence while reducing the number of turns needed to identify the target.
- **Historical evaluation:** `python -m evaluator.local_evaluator` on all 200 public sessions, using the same catalog and dataset as the baseline. These public-set numbers are retained as historical context; current benchmark reporting uses `python -m evaluator.hard_evaluator` and the fixed GPTAnnotation sessions above.

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
