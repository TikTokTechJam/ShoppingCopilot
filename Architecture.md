# ShoppingCopilot architecture

This document describes the implementation in the current repository, not an
aspirational design. For alternatives and proposed experiments, see
[`docs/approaches.md`](docs/approaches.md). For the challenge contract, see
[`docs/competition_specification.md`](docs/competition_specification.md).

## 1. Design goals and constraints

The agent searches a frozen 50,000-product catalog under a strict conversational
protocol:

- at most ten turns;
- at most one structured clarification attribute per turn;
- at most ten scored recommendations per turn;
- exact `parent_asin` equality for a hit;
- no access to the target or hidden intent card; and
- deterministic local behavior in the active runtime path.

The implementation therefore optimizes three related outcomes: whether the
target appears, how high it ranks, and how early it appears. A clarification is
useful only if its expected future gain is larger than the score that can be
earned by recommending now.

## 2. System map

```text
offline preparation

catalog.jsonl ------------------------------+
      |                                     |
      +-> V5 attribute annotation           |
              |                             |
              +-> annotations.jsonl         |
                      |                     |
                      +-> canonical dictionary
                      |       |
                      |       +-> exact lookup
                      |       +-> BGE attribute matrices
                      |
                      +-> product fact indexes

runtime

reset(session_id, user_profile)
      |
user message
      |
      +-> override/no-preference detection
      +-> canonical constraint extraction
      +-> initial Buying/Browsing routing
      +-> session-state merge
      +-> retrieve and rank catalog candidates
      +-> choose next clarification
      +-> return message + attribute + Top K
```

The evaluator constructs the agent through
[`evaluator/agent_factory.py`](evaluator/agent_factory.py). That factory passes
the catalog path and profile switch only. It does not configure whole-product
embeddings or an intent reranker.

## 3. Runtime turn

[`starter/agent.py`](starter/agent.py) owns the evaluator-facing `Agent` and
coordinates all runtime components.

### 3.1 Reset

`Agent.reset(session_id, user_profile)`:

1. creates a fresh in-memory `SessionState`;
2. stores a safe copy of the profile;
3. clears previous recommendation and clarification history; and
4. optionally builds a `ProfileAffinity` prior from `preference_tags`.

Catalog, fact, dictionary, BM25, and embedding artifacts are process-level
resources loaded outside the per-session state.

### 3.2 Respond

`Agent.respond(...)` follows this order:

1. Read the previous `last_asked` value.
2. Detect no-preference and generic evaluator replies. These messages are
   conversation control, so they are excluded from constraint extraction and
   query history.
3. Extract constraints from the new message.
4. Detect a full-goal or preference-level override against the pre-update state.
5. If the message answers a prior question, optionally repeat extraction scoped
   to that attribute.
6. Reset or selectively prune state for an override; otherwise mark the
   previous recommendations as excluded.
7. Route the first active goal to Buying or Browsing.
8. Merge structured and semantic constraints with provenance.
9. Retrieve and rank up to 100 candidates.
10. Fill the requested Top K, using a relaxed backfill if needed.
11. Select one clarification attribute, or use `other` as the end-of-cycle
    boundary.
12. Store the question and recommendations for the next turn.

The response always uses the required contract:

```python
{
    "message": "Which material should I prioritize?",
    "ask_attribute": "material",
    "recommendations": [{"parent_asin": "B000..."}],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
}
```

The active runtime makes no hosted-model call, so reported token usage is zero.

## 4. Intent routing

[`starter/routing/intent_router.py`](starter/routing/intent_router.py) provides
several routers; `Agent` instantiates `TwoPhaseIntentRouter` directly.

The active router works in two phases:

1. **Constraint-count phase.** If the message fills enough distinct canonical
   fields, it is treated as Buying unless a strong lexical Browsing veto fires.
2. **Signal-ledger phase.** Weighted lexical signals classify the remaining
   messages. Low-confidence unresolved cases default to Browsing.

Intent is sticky for an active goal. The agent does not reclassify every
clarification reply because answers such as “black” or “under $50” do not
express the session's overall mode. Explicit goal changes are handled by the
session override logic instead.

The repository contains an optional local reranker backend and a cascading
router, but the evaluator factory does not enable them.

## 5. Constraint representation

The public constraint vocabulary is:

```text
category, brand, color, material, size, style, feature, use_case,
price_min, price_max
```

Extraction is implemented in
[`starter/routing/constraints.py`](starter/routing/constraints.py).

### 5.1 Structured track

The state passed to the structured product scorer retains:

- exact canonical `brand`;
- numeric or explicitly stated `size`; and
- numeric price bounds.

Price parsing recognizes maxima, minima, ranges, approximate prices, and
explicit currency expressions. Size parsing is intentionally anchored on size
language so an arbitrary number is not treated as a size.

### 5.2 Canonical semantic track

Six descriptive fields use independent semantic state:

```text
category, color, material, style, feature, use_case
```

The generated dictionary first performs longest-first, token-boundary exact
lookup. Ambiguous surfaces are resolved only by directly attached field context
or strong catalog-frequency dominance. Remaining normalized text is split into
one-, two-, and three-token candidates and matched against per-attribute
canonical BGE matrices. Accepted matches retain their cosine similarity as
evidence.

`brand` is exact-only. It has no semantic matrix. `size` and price are outside
the dictionary semantic contract.

An important implementation detail is that exact matches for the six
descriptive fields do not become a second structured claim. They must appear in
the independent semantic state to influence the current scorer. This avoids
double counting, but it also means an exact-only dictionary without semantic
matrices provides a reduced brand/size/price agent.

### 5.3 Text views

The system deliberately maintains different views of a message:

- raw text for routing, overrides, corrections, and conversation history;
- normalized token surfaces for exact dictionary matching;
- stopword-filtered n-grams for semantic attribute matching; and
- accumulated query history for optional lexical or product-level retrieval.

These views must not be collapsed into one global normalization pass. The
known contraction and negation risks are documented in
[`docs/query_text_normalization_audit.md`](docs/query_text_normalization_audit.md).

## 6. Session state and overrides

[`starter/session.py`](starter/session.py) stores process-local state per
`session_id`:

- active mode and turn;
- raw query messages;
- structured and semantic constraints;
- provenance for every retained value;
- asked-attribute counts and no-preference fields;
- last and excluded recommendations; and
- override diagnostics.

There are two override scopes:

- **Full goal:** replace the shopping goal and reset goal-dependent state.
- **Preference:** replace selected fields and prune only inferred descendants.

The dependency graph is intentionally small. Category may parent use case,
size, and style; use case may parent feature, material, and style. Independent
fields such as brand, color, and price are not deleted merely because another
preference changed.

On an ordinary non-hit turn, the previous recommendations are added to the
session exclusion set. An override does not blindly carry all exclusions into
the new goal.

The `other` attribute is a clarification-cycle boundary, not a product field.
A useful answer starts a new cycle; an empty answer stops further questioning.

## 7. Product data and indexes

[`starter/retrieval.py`](starter/retrieval.py) loads the catalog into memory and
builds:

- `product_by_asin` and stable catalog order;
- per-field inverted indexes over canonical facts;
- price and rating lookups; and
- an in-memory SQLite FTS5 BM25 index over product text.

When V5 facts are present, they are merged with safe catalog-derived category
facts. An unannotated product contributes only those safe catalog category
facts. The raw catalog remains the authority for valid ASINs, price fallback,
rating, and catalog order.

The retriever contains loaders for direct product and multi-view embeddings.
Those paths require an explicitly compatible query encoder and are not
configured by `build_evaluator_agent`.

## 8. Ranking

### 8.1 Eligibility

Budget is the only product constraint used as a hard eligibility filter in the
normal retrieval pass. Previously shown recommendations are also hard-excluded.
Other fields are soft positive evidence; their absence does not remove a
product.

If the primary list is short, backfill relaxes budget and previous-result
exclusions before returning fewer than the requested Top K.

### 8.2 Contributions

The configured structured points are:

| Field | Weight |
| --- | ---: |
| brand | 7.00 |
| price | 1.50 |
| material | 1.20 |
| color | 1.00 |
| size | 0.80 |
| category | 0.70 |
| style | 0.50 |
| feature | 0.50 |
| use_case | 0.50 |

Under the active canonical flow, only brand, price, and size normally enter
this structured total. Semantic evidence contributes the retained similarity
for each canonical product fact that matches an accepted semantic constraint.

The base score is:

```text
base(x) = structured(x) + semantic_or_dense(x) + 0.20 * BM25(x)
rank(x) = base(x) + rating_weight(user) * normalized_rating(x)
```

Both modes currently use the same coefficients. The rating weight is `0.15`
for a shopper whose `average_prior_rating` is below `3.5`, and `0.02`
otherwise. Missing product ratings are neutral (`0.5`) rather than zero.

### 8.3 Current BM25 gating behavior

BM25 is built and queried on each retrieval, but the current branch structure
combines it into the final score only when a semantic/dense score mapping is
present. With structured constraints but no semantic/dense mapping, candidates
rank by structured points and rating. With neither constraints nor dense
evidence, they rank by rating and catalog order.

This is current behavior, not a stated design objective. It is a high-value
ablation target because vague first turns are exactly where independent lexical
recall may help.

### 8.4 Determinism

Ties end in stable catalog order. The implementation ranks the whole eligible
pool before slicing to `limit`; it does not take an arbitrary catalog prefix
and then rerank it.

## 9. Clarification policy

[`starter/clarification.py`](starter/clarification.py) scores each unasked,
unknown attribute over the current candidate pool.

For an attribute `a`, the policy approximates:

```text
question_value(a)
  = split_quality(a)
  * probability_of_useful_answer(a, mode, profile)
  * remaining_score_horizon(turn)
```

`split_quality` combines:

- coverage: how many candidates have the attribute;
- Gini impurity: whether values create a meaningful partition; and
- a small capped diversity factor.

Coverage below `0.20` and Gini below `0.10` are rejected. The mode prior models
how likely a shopper is to answer each attribute. `ProfileAffinity` can reorder
near-ties using `preference_tags`, while an explicit no-preference reply vetoes
the declined attribute. The score horizon falls with the turn and becomes zero
on turn ten, so the policy naturally stops asking on the final turn.

This is a one-step expected-utility heuristic. It is not an exact Bayesian
posterior or a dynamic program over all future conversations.

## 10. Artifact pipeline

The raw catalog is never modified. Generated product knowledge follows this
path:

```text
data/catalog.jsonl
  -> six V5 attribute annotation jobs
  -> category/brand/color/material/feature/use_case JSONL files
  -> scripts.aggregate_v5_annotations
  -> data/derived/annotations/v5/annotations.jsonl
       |                                  |
       |                                  +-> ProductRetriever fact indexes
       +-> scripts.build_attribute_dictionary --no-embeddings
             -> canonical_values.json
             -> normalized_lookup.json
             -> manifest.json
             -> scripts.build_v5_attribute_embeddings
                   -> attribute_embeddings/*.npy
                   -> attribute_embeddings/metadata.json
```

The annotation runners are resumable, record failures separately, validate the
model's schema, and write manifests. The deterministic aggregate and dictionary
builders should be rerunnable without a hosted model call.

See [`docs/attribute_dictionary.md`](docs/attribute_dictionary.md) for the file
contract and build commands.

## 11. Evaluation and debugging

There are two main evaluator entry points:

- `python -m evaluator.local_evaluator` runs the released 200-session public
  development set.
- `python -m evaluator.hard_evaluator` runs the tracked 400-session development
  benchmark.

Both construct the agent through the same factory and validate exact catalog
ASINs. The hard evaluator also supports scenario filtering, override-only runs,
turn-level diagnostics, and the browser debugger.

The hard benchmark simulator owns target facts and customer replies. Those
values are evaluator-side only and are never passed to `Agent.respond()`.

`results_*.json` files are historical snapshots. A valid comparison must pin
the code commit, dataset, catalog, facts manifest, dictionary manifest, model,
profile switch, evaluator command, and output file.

## 12. Active, optional, and retired paths

| Capability | Status in evaluator factory |
| --- | --- |
| Generated canonical dictionary | Required |
| Exact brand, structured size, numeric price | Active |
| Per-attribute BGE canonical matching | Active when local artifacts exist |
| BM25 index | Built; contribution is branch-gated |
| Rating-aware tie-breaking | Active |
| Profile-conditioned question prior | Active; can be disabled for ablation |
| Whole-product embeddings | Available in code, not wired |
| Multi-view product embeddings | Available in code, not wired |
| Local intent reranker | Available in code, not wired |
| Hosted LLM during a session | Not used |

## 13. Known limitations

The most important current limitations are:

1. Generated dictionary artifacts are required but not distributed in Git, so
   a fresh checkout cannot import the full agent until they are supplied.
2. Semantic n-grams still have known contraction, negation, short-token, and
   cross-field false-positive risks.
3. Negative preferences are not represented as first-class constraints.
4. Buying and Browsing use identical ranking coefficients.
5. BM25 does not independently drive vague or structured-only rankings.
6. Structured facts are soft except for budget; strong contradictions are not
   explicitly penalized.
7. The clarification policy estimates one-step split value, not target
   probability or multi-step value of information.
8. Intent Override is the weakest scenario in the newest checked-in hard-result
   snapshot.
9. Historical result files are not self-describing enough for strict
   reproducibility.

These limitations are prioritized and turned into testable experiments in
[`docs/approaches.md`](docs/approaches.md).

## 14. Source guide

| Concern | Primary source |
| --- | --- |
| Evaluator-facing orchestration | `starter/agent.py` |
| Conversation state and overrides | `starter/session.py` |
| Constraint extraction | `starter/routing/constraints.py` |
| Intent routing | `starter/routing/intent_router.py` |
| Product retrieval and ranking | `starter/retrieval.py` |
| BM25 | `starter/bm25.py` |
| Clarification utility | `starter/clarification.py` |
| Profile prior | `starter/profile_affinity.py` |
| Dictionary loader | `dictionary/registry.py` |
| Local BGE loader | `dictionary/semantic.py` |
| Evaluator construction | `evaluator/agent_factory.py` |
| Hard benchmark | `evaluator/hard_evaluator.py` |
| Interactive debugger | `evaluator/debug_web.py` |
| Offline annotation | `annotation/` and `scripts/annotate_*.py` |
