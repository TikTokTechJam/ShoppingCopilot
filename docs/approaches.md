# Solution approaches and experiment plan

This document turns the repository exploration into a decision guide. It
separates the implemented system from reasonable alternatives and recommends
an order for experiments. No proposal here changes the competition contract.

## 1. What the evidence says

The newest checked-in hard-result snapshot is
`results_gptannotation.json`, last changed in commit `28633b3` on 2026-08-29:

| Slice | Sessions | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 400 | 0.6900 | 0.255250 | 6.3900 | 0.513775 |
| Buying | 160 | 0.7563 | 0.298033 | 5.2188 | 0.583160 |
| Browsing | 160 | 0.7438 | 0.263628 | 6.4250 | 0.542464 |
| Intent Override | 60 | 0.3167 | 0.069993 | 9.5667 | 0.207998 |
| Boundary | 20 | 0.8500 | 0.401726 | 5.9500 | 0.646518 |

This is useful directional evidence, especially the override gap, but it is not
a clean “current main” benchmark. The result JSON does not record every runtime
artifact and switch needed to reproduce it. Treat it as a snapshot until the
same command is rerun from a manifest-pinned setup.

Code inspection adds five concrete observations:

1. The active evaluator path is canonical attribute retrieval, not
   whole-product vector retrieval.
2. Brand has a much larger structured weight than every other field.
3. Buying and Browsing share the same ranker weights despite separate routing.
4. BM25 only enters the final score when a semantic/dense score mapping exists.
5. Clarification is a candidate-split heuristic; it does not estimate a target
   posterior directly.

These observations suggest improving measurement and recall before replacing
the architecture wholesale.

## 2. Approach comparison

| Approach | Main idea | Advantages | Risks | Recommendation |
| --- | --- | --- | --- | --- |
| Canonical hybrid, stabilized | Keep current dictionary/state design; repair lexical fusion and semantic parsing | Smallest change, interpretable, deterministic, uses existing tests/debugger | Still depends on annotation quality and hand-tuned weights | **Do first** |
| Multi-source candidate fusion | Generate candidates independently from BM25, attribute postings, and semantic evidence; fuse before reranking | Better recall on vague and mixed queries; avoids one source gating another | Requires score calibration or rank fusion | **Do second** |
| Learned lightweight reranker | Train/calibrate a local model over candidate features | Can learn field and mode interactions | Leakage risk, needs reliable train/validation split and feature manifests | Do after recall is stable |
| Whole-product dense retrieval | Embed title/features/description and search the catalog | Good paraphrase recall; handles vocabulary outside canonical tags | Large artifacts, model/version coupling, false semantic matches | Run as an ablation, not the default yet |
| Probabilistic adaptive search | Maintain candidate probabilities and ask by expected value of information | Principled question selection and stopping | Simulator mismatch, sparse fact coverage, higher complexity | Explore after ranking is calibrated |
| Runtime LLM planner/reranker | Ask a model to interpret turns or rerank candidates | Flexible language understanding | Latency, cost, nondeterminism, prompt/version drift | Use only on narrow low-confidence cases |

## 3. Recommended near-term architecture

The strongest next version keeps the existing session and canonical-fact
layers, but gives each retrieval source an independent path:

```text
message and session state
        |
        +-> exact structured constraints -> posting candidates
        +-> canonical semantic evidence -> semantic-fact candidates
        +-> accumulated lexical query    -> BM25 candidates
        +-> optional product embedding   -> dense candidates

candidate union
        -> deterministic rank fusion
        -> feature-based reranking
        -> budget/exclusion policy
        -> Top K
        -> clarification utility over the same candidate distribution
```

The key change is candidate union before scoring. No source should disappear
because another source returned an empty mapping. Reciprocal-rank fusion is a
good first implementation because it avoids pretending BM25, cosine, and
weighted attribute points share a calibrated numeric scale:

```text
RRF(x) = sum_source weight(source) / (k + rank_source(x))
```

Once this is benchmarked, a small deterministic reranker can use source ranks,
source scores, matched fields, contradiction flags, rating, price distance,
mode, and turn as explicit features.

## 4. Workstreams

### 4.1 Reproducibility first

**Problem:** benchmark files do not fully describe the run that produced them.

**Change:** write a run manifest next to every result with:

- Git commit and dirty-tree flag;
- evaluator command and dataset checksum;
- catalog checksum;
- facts and dictionary manifest checksums;
- embedding model/path identity and matrix checksum;
- all ranking and clarification constants;
- profile switch; and
- wall-clock and token totals.

**Acceptance:** two runs with the same manifest produce identical session
recommendations and aggregate metrics.

### 4.2 Fix semantic query candidates

**Problem:** the normalization audit demonstrates contraction artifacts,
negation loss, unsafe short tokens, and cross-field n-gram matches.

**Change:** preserve raw text and introduce a matcher-only candidate view that:

- handles ASCII and Unicode contractions without turning `I'd` into `id`;
- keeps polarity terms until negative intent is identified;
- protects known short values by field;
- prefers deliberate attribute candidates over every possible n-gram; and
- deduplicates exact and semantic evidence by canonical ID.

**Acceptance:** add the regression cases listed in
[`query_text_normalization_audit.md`](query_text_normalization_audit.md) and
measure precision/recall on a hand-audited utterance set before rerunning the
shopping benchmark.

### 4.3 Remove BM25's dependency on dense evidence

**Problem:** vague queries can fall back to rating/catalog order even though a
BM25 score was computed.

**Change:** make lexical retrieval an independent candidate source. Start with
RRF or normalized per-query ranks rather than adding raw BM25 scores to cosine
and attribute points.

**Ablation:** current branch behavior versus BM25-always-on versus RRF.

**Acceptance:** improve first-turn target recall and Browsing Hit Rate@10
without a material Intent Override regression.

### 4.4 Represent negative constraints

**Problem:** “not red”, “anything but leather”, and “no Nike” cannot be stored
as negative preferences. Stopword cleanup can even invert their meaning.

**Change:** add explicit positive and negative values per field. Apply a hard
negative only when extraction confidence is exact and field-scoped; use a soft
penalty for semantic negatives until precision is demonstrated.

**Acceptance:** focused tests for negation, correction, double negation, and
override interaction; no negative value may become a positive constraint.

### 4.5 Make override recovery observable

**Problem:** Intent Override is the weakest result slice, and aggregate metrics
do not reveal whether failures come from detection, stale state, exclusions,
or reranking.

**Change:** report per override session:

- override turn and detected kind;
- fields removed, retained, and inferred descendants pruned;
- prior recommendations released or retained;
- target rank immediately before and after the override; and
- first turn the new constraint affected ranking.

**Acceptance:** classify every miss into detection, extraction, state, recall,
or ranking. Improve the largest bucket, not the average symptom.

### 4.6 Calibrate Buying and Browsing separately

**Problem:** routing currently changes the question prior but not the ranker.

**Change:** test mode-specific source weights and candidate depths. Buying may
benefit from stronger exact/constraint evidence; Browsing may benefit from
broader lexical/dense recall and more diverse Top K.

**Acceptance:** report a two-by-two ablation: shared versus mode-specific
ranking, with routing enabled versus oracle scenario labels in evaluator-only
analysis. Oracle labels must never enter the agent.

### 4.7 Improve clarification after retrieval stabilizes

**Problem:** split quality treats the current candidate pool as uniform and
does not use calibrated target likelihood.

**Change:** turn normalized candidate scores into a conservative posterior and
estimate, for each question:

```text
expected_gain(attribute)
  = answer_probability
  * sum_answer P(answer) * best_future_score(answer)
  - current_recommendation_score
```

Include a no-preference branch and a remaining-turn budget. Start with one-step
lookahead; do not implement a full dynamic program until calibration is proven.

**Acceptance:** compare current Gini utility, entropy reduction, and direct
expected TechnicalScore on identical cached candidate traces.

## 5. Experiment sequence

Run one meaningful change at a time:

1. Add run manifests and reproduce the current hard result.
2. Build a failure report for the 400 hard sessions.
3. Fix semantic candidate normalization with focused precision tests.
4. Make BM25 independent and compare current/additive/RRF fusion.
5. Add negative constraints.
6. Diagnose and fix the largest Intent Override failure bucket.
7. Test mode-specific ranking.
8. Only then compare optional whole-product dense retrieval.
9. Calibrate a posterior and revisit clarification value.

For each experiment, preserve:

- the baseline and candidate commit;
- complete artifact manifests;
- overall and per-scenario metrics;
- first-turn candidate recall at 10/50/100;
- target-rank movement by turn;
- latency and memory; and
- a fixed list of newly fixed and newly regressed sessions.

## 6. What not to combine in one experiment

Avoid changing these together, because the benchmark will not identify the
cause of movement:

- semantic normalization and ranking weights;
- override detection and exclusion behavior;
- candidate generation and clarification policy;
- product embeddings and mode routing; or
- annotation vocabulary and evaluator session data.

The repository already has enough interacting layers. Clean ablations will
produce more useful progress than another broad “new strategy” result file.

## 7. Suggested success criteria

A change is ready to keep when it:

1. passes focused unit tests and the full suite with the required artifacts;
2. improves or preserves overall TechnicalScore on the hard benchmark;
3. does not hide a large scenario regression behind the overall average;
4. improves the metric it was designed to affect;
5. has a self-contained result manifest; and
6. remains deterministic when no optional model is configured.

For risky changes, use a stricter rule: require improvement on both the public
development evaluator and the fixed hard benchmark, or explain why their
simulators reward different behavior.
