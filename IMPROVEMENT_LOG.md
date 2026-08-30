# Agent improvement log

This file records benchmark evidence, not just implementation history. Add a
new entry only when the run can be tied to code, data, artifacts, configuration,
and an evaluator command.

## Evidence standard

Every future result should record:

- date, commit, and dirty-tree status;
- the exact evaluator command;
- dataset and catalog checksums;
- facts, dictionary, and embedding manifest checksums;
- relevant feature switches and scoring constants;
- before/after overall and scenario metrics;
- latency, memory, and token totals; and
- fixed-session wins, losses, and the next hypothesis.

Older checked-in result JSON files predate that standard. They are useful
historical snapshots, but their filenames are not sufficient proof that two
runs differ in only one component.

## Published challenge baseline

The challenge's original weak BM25 starter reported:

| Metric | Baseline |
| --- | ---: |
| Hit Rate@10 | 0.125000 |
| MRR | 0.068034 |
| MTTC | 9.810000 |

The current `starter/agent.py` is no longer that baseline. The values above are
copied from `docs/baseline_results.json`; they were not reproduced during the
documentation rewrite.

## Checked-in result inventory

| File | Last commit | Sessions | Hit Rate@10 | MRR | MTTC | Score field |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `results_gptannotation.json` | `28633b3` | 400 | 0.6900 | 0.255250 | 6.3900 | 0.513775 |
| `results_layer1_layer2_jina.json` | `8ff04d4` | 400 | 0.5550 | 0.212783 | 7.0000 | 0.421335 |
| `results_layer1.json` | `8ff04d4` | 400 | 0.3675 | 0.134340 | 8.5150 | 0.273752 |
| `results_public_layer1.json` | `661b5dd` | 200 | 0.7700 | 0.300103 | 4.7700 | 0.599631 |
| `results_sessions_layer1.json` | `661b5dd` | 400 | 0.4300 | 0.150412 | 8.1625 | 0.316874 |

The table reports the values stored in each file. It does not assert that the
named approaches are directly comparable or still match the code on `main`.

## Current directional finding

The newest checked-in hard snapshot is `results_gptannotation.json`:

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Buying | 160 | 0.756250 | 0.298033 | 5.218750 | 0.583160 |
| Browsing | 160 | 0.743750 | 0.263628 | 6.425000 | 0.542464 |
| Intent Override | 60 | 0.316667 | 0.069993 | 9.566667 | 0.207998 |
| Boundary | 20 | 0.850000 | 0.401726 | 5.950000 | 0.646518 |

Intent Override is the clear weak slice in this snapshot. The next diagnostic
should classify those misses into override detection, constraint extraction,
stale-state pruning, candidate recall, and final ranking before changing more
weights.

## Historical entries

### 2026-08-29 — New strategy snapshot

- **Commit:** `28633b36fc2c2481d1b90ccabcdd77af17e40bd1`
- **Artifact:** `results_gptannotation.json`
- **Observed result:** Hit Rate@10 `0.69`, MRR `0.25525`, MTTC `6.39`,
  TechnicalScore `0.513775` on 400 hard sessions.
- **Interpretation:** Buying, Browsing, and Boundary are substantially stronger
  than Intent Override. Because the result lacks a complete run manifest, do
  not attribute the movement to a single code change.
- **Next step:** reproduce from the current commit with complete artifact
  provenance and generate an override failure report.

### 2026-08-26 — Fixed 400-session hard evaluator

- **Change:** added a deterministic development benchmark with 160 Buying, 160
  Browsing, 60 Intent Override, and 20 Boundary sessions.
- **Why:** the public evaluator alone did not expose enough turn-level behavior
  for question selection, no-preference branches, and intent overrides.
- **Historical result at introduction:** Hit Rate@10 `0.5075`, MRR `0.163393`,
  MTTC `7.5425`, TechnicalScore `0.371918`.
- **Caution:** this is an older code state and should not be presented as the
  current result.

### 2026-08-25 — Expected-utility adaptive search

- **Commit:** `50d010d7a6c2d5fe07957e55502f920a8c743e62`
- **Change:** replaced the original stateless recommendation behavior with
  conversation constraints, candidate-pool question utility, no-preference
  handling, and intent-override reset behavior.
- **Historical public result:** Hit Rate@10 `0.765`, MRR `0.495623`, MTTC
  `5.145`, reported TechnicalScore `0.648287`.
- **Caution:** the present code and the newer `results_public_layer1.json` do
  not reproduce that exact MRR. Preserve this as historical evidence, not a
  current benchmark claim.

## Entry template

### YYYY-MM-DD — Short experiment name

- **Commit and tree:** SHA; clean/dirty
- **Hypothesis:** one falsifiable claim
- **Change:** the smallest behavior changed
- **Artifacts:** dataset/catalog/facts/dictionary/model manifests
- **Command:** exact evaluator invocation
- **Before:** overall and scenario metrics
- **After:** overall and scenario metrics
- **Diagnostics:** candidate recall, rank movement, latency, wins/losses
- **Decision:** keep, revert, or investigate
- **Next step:** one follow-up hypothesis
