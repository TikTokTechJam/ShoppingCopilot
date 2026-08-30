# Shopping Copilot architecture

Status: current implementation reference.

This document describes what the repository implements today. It is intentionally descriptive, not aspirational: planned retrieval changes, retired experiments, and future ideas are not part of the active architecture unless explicitly marked as inactive/legacy.

## 1. Runtime objective and constraints

Shopping Copilot runs over a frozen catalog of roughly 50,000 products and serves multi-turn conversational product search in one process.

The runtime is optimized for the competition contract:

- return ordered `parent_asin` recommendations every scoreable turn;
- maximize HitRate@10 and MRR while finding the target as early as possible;
- maintain conversational constraints across turns;
- support Buying, Browsing, Intent Override, and Boundary scenarios;
- ask at most one clarification attribute per turn;
- keep evaluator-only target information outside Agent logic;
- use precomputed artifacts and in-process indexes rather than external databases.

The active evaluator path uses:

- deterministic structured parsing;
- V5 exact canonical matching;
- BGE semantic matching against canonical attribute values;
- per-attribute product posting lists;
- SQLite FTS5 BM25 over raw catalog text;
- hybrid product ranking;
- process-local conversational state with selective override invalidation.

Whole-product Jina retrieval still exists in the codebase, but it is not enabled by the standard evaluator factory.

## 2. End-to-end runtime flow

```text
Agent.reset(session_id, user_profile)
        |
        v
create SessionState + profile affinity
        |
        v
Agent.respond(user_message, turn, top_k)
        |
        +--> skip extraction for generic/no-preference replies when applicable
        |
        v
extract_constraints(user_message)
        |
        +--> structured price parsing
        +--> structured size parsing
        +--> V5 exact canonical matching
        `--> BGE semantic canonical matching
                |
                +--> stopword removal
                +--> 1/2/3-gram generation
                `--> similarity thresholding
        |
        v
detect correction / override
        |
        +--> normal turn
        +--> selective preference reset
        `--> full-goal reset
        |
        v
update SessionState
        |
        +--> structured constraints
        +--> semantic constraints
        +--> provenance/dependency metadata
        +--> message history
        +--> clarification state
        `--> recommendation exclusions
        |
        v
ProductRetriever.retrieve()
        |
        +--> budget eligibility
        +--> recommendation exclusions
        +--> structured posting-list score
        +--> semantic canonical posting-list score
        +--> SQLite FTS5 BM25 score
        `--> optional direct dense product score when manually configured
        |
        v
hybrid score + rating tie-break
        |
        v
rank candidates
        |
        v
Top K / relaxed backfill when required
        |
        v
ClarificationPolicy
        |
        v
Agent response
```

## 3. Main components

| Responsibility | Active implementation |
| --- | --- |
| Agent lifecycle | `starter/agent.py` |
| Session state | `starter/session.py` |
| Constraint extraction | `starter/routing/constraints.py` |
| Intent routing | `starter/routing/intent_router.py` |
| Canonical registry | `dictionary/registry.py` |
| BGE semantic matcher | `dictionary/semantic.py` |
| BM25 index | `starter/bm25.py` |
| Retrieval/ranking | `starter/retrieval.py` |
| Clarification | `starter/clarification.py` |
| Evaluator Agent construction | `evaluator/agent_factory.py` |
| Hard evaluator | `evaluator/hard_evaluator.py` |
| Debug UI | `evaluator/debug_web.py` |

## 4. Agent lifecycle

### 4.1 Construction

`Agent` creates and owns the long-lived retrieval components used across sessions. The retriever loads catalog facts, product metadata, posting lists, and BM25 state during construction.

The standard evaluator factory constructs the normal Agent without explicitly enabling the legacy direct product-embedding path.

### 4.2 `reset()`

`reset(session_id, user_profile)` creates a fresh process-local `SessionState` and stores the user profile for that session.

It also builds profile affinity used by clarification/ranking support where configured.

### 4.3 `respond()`

For each turn, `respond()` currently performs the following sequence:

1. Load session state.
2. Detect generic/no-preference replies that should not contribute normal extraction/query text.
3. Extract current-turn structured and semantic constraints.
4. Detect correction/override behavior.
5. If the message answers the previously asked attribute, optionally re-run extraction scoped to that attribute.
6. Split new evidence into structured and semantic deltas.
7. Identify corrected/replaced fields.
8. Apply either normal-turn state handling, selective preference reset, or full-goal reset.
9. Route the shopping mode when the session has no active mode.
10. Update structured constraints, semantic constraints, provenance, and message history.
11. Retrieve and rank products.
12. Backfill if fewer than `top_k` products are available.
13. Select the next clarification attribute.
14. Store current recommendations and return the response.

## 5. Session state

`SessionState` tracks the active conversational search state in memory.

Important state includes:

```text
mode
structured constraints
semantic constraints
messages
last_user_message
last_asked
asked_attributes
no_preference_attributes
clarification-cycle state
last_recommendations
excluded_recommendations
structured provenance
semantic provenance
last override information
```

### 5.1 Message/query history

The active query text is constructed from retained session messages:

```text
state.query_text = " ".join(state.messages)
```

Generic/no-preference filler replies may be omitted from this history.

A selective preference reset does not currently rewrite historical messages. Therefore stale text from an overridden preference can remain in `state.query_text` and still influence BM25. This is a known current limitation.

### 5.2 Recommendation exclusions

On an ordinary turn, the previous recommendation set is promoted into `excluded_recommendations`.

Normal retrieval excludes these products so later turns explore new candidates.

- preference-level reset preserves the exclusion set;
- full-goal reset clears the exclusion set.

### 5.3 Override types

The runtime distinguishes:

```text
NONE
PREFERENCE
FULL_GOAL
```

Preference-style markers include language such as:

```text
actually
instead
rather
changed my mind
priority changed
```

Full-goal markers include stronger reset language such as:

```text
scratch that
forget that
start over
new search
ignore the earlier ...
```

A full-goal reset clears stale goal-specific state and allows the new goal to be routed again.

A preference override performs selective invalidation instead of clearing the complete session.

## 6. Constraint provenance and dependency-aware override handling

The current session model stores provenance for constraints so selective invalidation can preserve independent explicit information.

Provenance records track conceptually:

```text
attribute
value
source: explicit | inferred
optional parent/dependency constraint
```

The active dependency graph is:

```text
category
├── use_case
│   ├── feature
│   ├── material
│   └── style
├── size
└── style

mostly independent:
brand
color
price
```

When a preference is replaced, the runtime removes the old conflicting value and traverses dependent fields. Inferred descendants that depend on the replaced constraint can be invalidated, while independently explicit descendants are preserved.

Example:

```text
before:
category = shirt                  explicit
use_case = sunny weather         explicit
feature = uv protection          inferred from use_case
color = black                    explicit

user:
"Actually I need it for rainy weather."

after selective invalidation:
category = shirt                  keep
color = black                     keep
use_case = rainy weather         replace
feature = uv protection          remove if dependent/inferred
```

This is different from a full-goal reset.

## 7. Product facts and V5 annotations

The primary runtime facts artifact is:

```text
data/derived/annotations/v5/annotations.jsonl
```

Current aggregate schema:

```json
{
  "parent_asin": "B123",
  "price": 22.99,
  "facts": {
    "category": [],
    "brand": [],
    "color": [],
    "material": [],
    "feature": [],
    "use_case": []
  }
}
```

The current V5 aggregate covers the catalog and intentionally does not yet contain `style` or `size` facts.

The runtime also retains fallback compatibility with older fact artifacts where implemented.

### 7.1 Canonical dictionary

The generated V5 canonical dictionary is under:

```text
data/derived/annotations/v5/dictionary
```

At the inspected snapshot, the aggregate dictionary reports approximately:

```text
50,000 product records
47,187 canonical values
3,501 category values
26,882 brand values
1,417 color values
701 material values
0 style values
8,219 feature values
6,467 use_case values
659 ambiguous normalized surfaces
21 normalized collisions
```

The non-brand BGE semantic artifact contains the semantic canonical values used by the active attribute matcher.

## 8. Constraint extraction

Constraint extraction combines deterministic parsing, exact canonical matching, and BGE semantic matching.

### 8.1 Structured parsing

Price parsing currently supports common forms such as:

```text
under 100
below 100
over 50
above 50
between 30 and 60
$100
100 USD
budget / price wording
```

The parser contains guards against confusing likely years, sizes, and measurements with price.

Size remains a structured runtime constraint even though it is not part of the active V5 semantic dictionary.

### 8.2 Exact canonical matching

Exact canonical matching uses the generated V5 registry.

Important behavior:

- deterministic normalization;
- token-boundary matching;
- longest/specific phrase first;
- non-overlapping selected spans;
- ambiguity preserved unless local context or frequency evidence resolves it.

For ambiguous surfaces without explicit attribute context, frequency dominance may resolve the value only when both conditions hold:

```text
top candidate share >= 0.75
top frequency / second frequency >= 3.0
```

Common-word brand collisions are guarded for terms such as `find`, `it`, `make`, and `on`.

### 8.3 Structured session view

The current `structured_only()` path intentionally retains only:

```text
brand
size
price
```

Therefore exact matches for category, color, material, feature, use_case, or style may exist in raw extraction output without entering Layer-1 structured session state through this view.

Those fields are primarily represented in the active semantic constraint state.

## 9. BGE semantic canonical matching

The active semantic model is:

```text
BAAI/bge-small-en-v1.5
embedding dimension: 384
L2-normalized vectors
```

Semantic matrices exist for:

```text
category
color
material
style
feature
use_case
```

Brand has no BGE matrix and remains exact-only.

### 9.1 Runtime semantic flow

```text
user message
    |
    v
semantic tokenization + stopword removal
    |
    v
1-word, 2-word, 3-word phrases
    |
    v
BGE encode phrases
    |
    v
search canonical matrices
    |
    v
keep similarities >= 0.80
    |
    v
deduplicate by canonical ID using strongest score
    |
    v
store semantic constraints + evidence
```

The current n-gram path does not impose a small semantic top-k after thresholding. Every canonical value that passes the configured similarity threshold can be retained.

This can create large semantic state when many values exceed `0.80`.

Semantic values accumulate across turns until explicitly replaced/reset.

## 10. In-memory product posting lists

Product canonical facts are indexed in memory using per-field posting lists:

```text
field -> canonical value -> set(parent_asin)
```

Conceptually:

```text
category["running shoes"] -> { ...ASINs... }
brand["nike"]             -> { ...ASINs... }
color["black"]            -> { ...ASINs... }
material["leather"]       -> { ...ASINs... }
feature["waterproof"]     -> { ...ASINs... }
use_case["hiking"]        -> { ...ASINs... }
```

This prevents semantic canonical scoring from scanning product facts value-by-value across the entire catalog.

## 11. Structured product scoring

The retriever defines the following structured field weights:

| Field | Weight |
| --- | ---: |
| category | 0.70 |
| price | 1.50 |
| brand | 7.00 |
| size | 0.80 |
| color | 1.00 |
| material | 1.20 |
| style | 0.50 |
| feature | 0.50 |
| use_case | 0.50 |

In the normal Agent path, only `brand`, `size`, and `price` are supplied through the structured-only session view.

Price bounds affect candidate eligibility rather than acting only as a soft score.

When a budget is active:

- known prices inside the bounds are eligible;
- known prices outside the bounds are excluded;
- products with missing price are excluded.

## 12. Semantic canonical product scoring

BGE resolves user phrases to canonical values. Retrieval then uses product posting lists for those values.

For each accepted semantic canonical value, products containing that value receive the retained BGE similarity as semantic evidence.

Conceptually:

```text
"something for rainy weather"
        |
        v
BGE canonical evidence
use_case:rain          0.91
use_case:wet weather   0.84
        |
        v
posting-list lookup
        |
        v
products annotated with those canonical values receive semantic score
```

This is canonical semantic matching followed by inverted lookup. It is not direct query-to-product vector similarity in the standard runtime.

## 13. SQLite FTS5 BM25

Raw catalog text is actively searched using SQLite FTS5 in `starter/bm25.py`.

The in-memory FTS table contains:

```text
parent_asin  UNINDEXED
title
categories
features
details
store
description
```

Tokenizer:

```text
unicode61 remove_diacritics 2
```

The tokenizer does not currently use Porter stemming.

BM25 field weights are:

| Field | Weight |
| --- | ---: |
| parent_asin | 0.0 |
| title | 6.0 |
| categories | 4.0 |
| features | 2.5 |
| details | 2.5 |
| store | 1.5 |
| description | 1.0 |

The query is currently built from session query text using the semantic tokenizer/stopword list:

1. tokenize retained session text;
2. remove stopwords;
3. deduplicate terms;
4. keep at most 40 terms;
5. build one quoted `OR` expression.

Example:

```text
"coat" OR "rain" OR "waterproof"
```

There is currently no BGE-to-BM25 synonym expansion in the active implementation and no separate BM25 search per semantic constraint.

Because preference overrides preserve message history, BM25 can still receive lexical terms from an overridden earlier preference. This is one of the main current state/query inconsistencies.

## 14. Retrieval and hybrid ranking

`ProductRetriever.retrieve()` starts from products that remain eligible after budget filtering and recommendation exclusions.

It then combines available score sources.

The active mode weights are currently identical for Buying and Browsing:

```text
BUYING
structured = 1.00
dense      = 1.00
bm25       = 0.20

BROWSING
structured = 1.00
dense      = 1.00
bm25       = 0.20
```

In the standard V5 evaluator path, `dense` normally means the BGE canonical posting-list score, not Jina whole-product similarity.

Base hybrid scoring is conceptually:

```text
base_score =
    1.00 * structured_score
  + 1.00 * semantic/dense_score
  + 0.20 * bm25_score
```

A normalized product-rating bonus participates in actual ordering as a tie-break/supporting signal.

`Candidate.score` represents the base hybrid score, while `Candidate.ranking_score` includes the rating adjustment used by sorting.

### 14.1 Structured-only fallback behavior

There is a current branch where, when semantic/dense scores are absent and constraints exist, ranking can fall back to structured scoring. BM25 may already have been computed but not affect ordering in that branch.

### 14.2 Candidate pool

Candidate generation begins from all budget/exclusion-eligible catalog products rather than from a small lexical or semantic top-k union.

`minimum_candidates` is accepted by the retrieval API but is not currently used by `retrieve()` to alter candidate generation.

## 15. Backfill and relaxation

The Agent attempts to return the requested number of recommendations.

If the normal constrained retrieval returns too few products, a relaxed backfill path can be used.

Current behavior of that path is important:

```text
apply_budget = False
recommendation exclusions are not supplied
```

Therefore backfill can reintroduce:

- products outside the active budget;
- products deliberately excluded because they were recommended previously.

This is a known correctness tradeoff/limitation of the current implementation.

## 16. Intent routing

The default Agent uses `TwoPhaseIntentRouter`.

`SessionIntentTracker` exists in the repository but is not used by the default Agent path.

### 16.1 Phase 1

The router extracts constraints and counts populated non-category fields.

A session can be routed to Buying when at least two such fields are populated unless sufficiently strong Browsing evidence vetoes that decision.

Important thresholds include:

```text
weak confidence        0.70
decision confidence    0.70
Buying tag threshold   2
Browsing veto          0.70
reranker acceptance    0.80
session flip margin    0.80
```

### 16.2 Lexical routing signals

Current signal weights include:

| Signal | Weight |
| --- | ---: |
| budget | +1.50 |
| brand | +1.20 |
| size | +1.00 |
| color | +1.00 |
| material | +1.00 |
| feature | +1.00 |
| use_case | +0.60 |
| style | +0.70 |
| request verb | +0.45 |
| undecided | -1.40 |
| explore verb | -1.10 |
| option seeking | -0.90 |

Once selected, the mode remains active through normal turns and preference overrides. A full-goal reset clears the mode so the new goal can be routed again.

## 17. Clarification policy

Clarification is deterministic and runs after retrieval.

The supported attributes are:

```text
category
material
color
size
style
brand
budget
feature
use_case
```

The policy evaluates the current candidate pool using:

- attribute coverage;
- value diversity;
- Gini split quality;
- answer probability;
- remaining-turn horizon;
- profile affinity where available;
- whether the attribute is already known/asked/declined.

Important constants include:

```text
minimum useful coverage   0.20
minimum split Gini        0.10
approx. utility floor     0.0315
```

At most one normal question per attribute is asked within a clarification cycle.

The final turn naturally has zero future clarification value.

If no useful specific attribute remains, the Agent can ask `other`.

Recommendations are still returned on the same turn as a clarification question.

## 18. User profile usage

`reset()` receives an anonymized `user_profile`.

The active code can build profile affinity and use it as a supporting signal in clarification/ranking behavior.

The evaluator can disable this behavior with its user-profile option for ablation/testing.

## 19. Evaluator boundary

The hard evaluator is outside the Agent boundary.

Standard runtime contract:

```text
reset(session_id, user_profile)
respond(session_id, user_message, turn, top_k)
```

Agent response shape:

```json
{
  "message": "Do you have a material preference?",
  "ask_attribute": "material",
  "recommendations": [
    {"parent_asin": "B000..."}
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0
  }
}
```

Requirements enforced by the evaluator include valid catalog IDs, ordering, uniqueness, and supported `ask_attribute` values.

Hidden target ASINs and hidden simulator facts remain evaluator-side and are never passed into Agent retrieval/ranking logic.

### 19.1 Hard evaluator behavior

The hard evaluator uses up to 10 turns and scores Top 10 recommendations.

Its fixed development session mix is:

```text
Buying          160
Browsing        160
Intent Override  60
Boundary         20
```

For intent-override sessions, a target appearing before the configured override does not count as the successful hit. The session becomes scoreable after the override has occurred.

Core metrics are:

```text
HitRate@10
MRR
MTTC
Efficiency
TechnicalScore
```

### 19.2 Local evaluator

`evaluator/local_evaluator.py` is an older, separate simulator/evaluator path.

It does not exactly reproduce hard-evaluator behavior and should not be treated as the authoritative benchmark implementation.

### 19.3 Debug UI

`evaluator/debug_web.py` uses the same Agent/retrieval path while exposing diagnostics such as:

```text
structured rank
semantic/dense rank
BM25 rank
hybrid rank
score components
matched constraints
session/override state
```

## 20. Startup and runtime data loading

At Agent/retriever construction, the implementation can load/build:

```text
V5 product annotations
catalog products
product_by_asin lookup
canonical fact posting lists
SQLite in-memory FTS5 index
canonical dictionary
BGE attribute matrices/model
optional legacy product-embedding artifacts
```

These objects are reused across sessions in the same Agent instance.

Important default paths include:

```text
data/catalog.jsonl
data/derived/annotations/v5/annotations.jsonl
data/derived/annotations/v5/dictionary
data/derived/gptannotation/sessions.jsonl
```

Important semantic model configuration:

```text
BAAI/bge-small-en-v1.5
SHOPPING_ATTRIBUTE_EMBEDDING_MODEL
```

Other optional configuration includes:

```text
SHOPPING_COPILOT_RERANKER_DIR
```

BM25 and hybrid weights are currently code constants rather than environment-configured values.

## 21. Performance-sensitive paths

Current startup/runtime costs include:

1. Reading 50,000 catalog products.
2. Building product fact posting lists.
3. Building the full in-memory SQLite FTS5 catalog index for every new Agent instance.
4. Loading the BGE model and canonical matrices.
5. Encoding multiple 1/2/3-gram phrases per user turn.
6. Comparing phrases against canonical matrices.
7. Retaining potentially many semantic values above the `0.80` threshold.
8. Iterating over the eligible product set for final score assembly/sorting.
9. Filtering BM25 results against allowed ASINs in Python.
10. Running additional ranking snapshots in debug diagnostics.

The BGE n-gram path is especially sensitive to the number of threshold-passing canonical values because semantic constraints accumulate across turns.

## 22. Direct Jina product retrieval: present but inactive by default

The repository still contains a legacy/direct Layer-2 product embedding path and Jina artifact support.

Known artifact example:

```text
data/derived/product_embeddings_jina/
model: jinaai/jina-embeddings-v5-text-nano
dimension: 768
views: categories, title, features, description
```

The standard evaluator factory does not pass a Layer-2 artifact directory or compatible query encoder into the Agent.

The default product-embedding artifact search paths also do not include the Jina artifact directory above.

Therefore the standard evaluator does not normally execute direct Jina product-vector retrieval.

This code is retained for compatibility/manual experimentation but is not part of the default active retrieval path.

## 23. Other present but non-default components

The codebase also contains components that are not normally active in the default evaluator path, including:

```text
SessionIntentTracker
optional Qwen intent reranker
build_default_router()
HashEmbeddingModel for smoke/build scenarios
legacy product embedding loaders
V4 fact fallbacks
local evaluator simulator
```

Their presence in the repository should not be interpreted as evidence that they participate in the standard runtime.

## 24. Known current limitations and inconsistencies

The following behaviors are part of the current implementation snapshot and should be kept visible when interpreting benchmarks:

1. Preference overrides selectively invalidate constraint state but do not remove stale text from `state.query_text`, so BM25 can still see overridden words.
2. Exact non-brand canonical matches can be extracted but excluded from the structured-only session view.
3. Semantic thresholding can retain many canonical values because there is no small top-k restriction after the `0.80` threshold.
4. Semantic state can accumulate substantially across turns.
5. BM25 is computed on every normal retrieval call, including branches where its score may later be ignored.
6. The structured-only fallback can rank without BM25 even after BM25 was computed.
7. `Candidate.score` does not always equal the final sorting score because rating is applied separately.
8. Relaxed backfill can bypass active budget bounds and recommendation exclusions.
9. The normal retriever starts from all eligible products rather than a small candidate union.
10. `minimum_candidates` is accepted but not currently used by `retrieve()`.
11. Direct Jina product retrieval remains in code but is not enabled by the standard evaluator factory.
12. Debug Layer-2 terminology can refer to BGE canonical semantics rather than direct product embeddings.
13. Artifact manifests in the V5 tree may not all describe the same aggregation stage and should not be assumed interchangeable.

## 25. Active architecture summary

The current standard evaluator path can be summarized as:

```text
                         USER TURN
                             |
                             v
                     SessionState
                             |
                             v
                  Constraint Extraction
             +---------------+---------------+
             |               |               |
             v               v               v
       structured parse   exact V5       BGE canonical
       price / size       dictionary      semantic match
             |               |               |
             +---------------+---------------+
                             |
                             v
                  override/state update
                             |
             +---------------+---------------+
             |               |               |
             v               v               v
         structured       semantic         raw session
          state            state             text
             |               |               |
             v               v               v
        fact posting     fact posting      SQLite FTS5
        list scoring     list scoring         BM25
             |               |               |
             +---------------+---------------+
                             |
                             v
                      Hybrid Ranking
                   + rating tie-break
                             |
                             v
                           Top K
                             |
                             v
                    ClarificationPolicy
                             |
                             v
                        Agent response
```

This diagram is the authoritative high-level description of the active implementation represented by this document.