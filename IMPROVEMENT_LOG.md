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

### 2026-08-26 — Generalized hybrid retrieval and independent benchmarks

- **Commit:** This PR (`codex/generalized-shopping-copilot`).
- **Change:** Replaced the evaluator-oriented adaptive policy with a generalized
  category-alias and weighted-FTS retrieval path, IDF/phrase evidence ranking,
  candidate-driven concrete clarification fields, intent-override state reset,
  and failed-result rotation. Added an interactive terminal demo, a fixed
  human-authored 20-product benchmark, a seeded 10-product raw-metadata sanity
  check, their complete transcripts, and focused unit tests.
- **Why:** The previous agent improved the public score but still depended too
  heavily on the evaluator's conversational shape. The new implementation avoids
  `other`, does not reconstruct intent-card constraints, and exposes performance
  on independent inputs as well as the released labels.
- **Expected impact:** Improve public retrieval and conversion speed while making
  behavior reproducible on catalog-derived and manually paraphrased requests.
- **Evaluation:** `python -m evaluator.local_evaluator --output results.json` on
  all 200 public sessions, plus `python -m evaluator.hard_evaluator
  --sample-count 200 --seed 20260826` and the committed independent benchmarks.

| Public metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.765 | 1.000 | +0.235 |
| MRR | 0.495623 | 0.687687 | +0.192064 |
| MTTC | 5.145 | 1.610 | -3.535 turns |
| Technical score | 0.648287 | 0.894106 | +0.245819 |

Additional generalization measurements:

| Benchmark | Samples | Hit Rate@10 | MRR | MTTC | Technical score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Catalog-driven hard evaluator | 200 | 0.890 | 0.457266 | 4.740 | 0.707380 |
| Human-authored fixed queries | 20 | 0.750 | 0.424008 | 4.650 | 0.629202 |
| Automated raw-metadata sanity check | 10 | 1.000 | 0.883333 | 1.900 | 0.947000 |

- **Evidence:** See [`docs/solution_results.json`](docs/solution_results.json),
  [`docs/hard_benchmark_results.json`](docs/hard_benchmark_results.json),
  [`docs/human_benchmark_20_results.json`](docs/human_benchmark_20_results.json),
  and [`docs/random_benchmark_10.json`](docs/random_benchmark_10.json).
- **Result and next step:** The generalized implementation materially improves
  every public metric and reaches 0.89 Hit Rate on the harder 200-product run.
  The hand-authored benchmark remains lower and exposes failures in sparse
  metadata, category inference, and intent overrides. The next improvement should
  add compact offline semantic retrieval and relevance-qualified diversity without
  tuning against individual public targets.

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
