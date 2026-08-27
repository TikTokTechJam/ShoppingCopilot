# Shopping Copilot architecture

Status: MVP architecture and component contract for Issues #5–#17.

This document is the source of truth for the data-to-search boundaries. It
preserves the Issue #5–#9 contracts below and extends them through the first
end-to-end Agent boundary in Issues #12–#17. Existing implementation notes are
identified as runtime behavior; proposed integration notes are contracts for
the next MVP slices. A documented artifact, integration, or diagnostic is not
evidence that it has been generated, deployed, or benchmarked.

## End-to-end boundary

```text
OFFLINE / BUILD TIME (deterministic, never per session)

catalog.jsonl + Issue #5 facts
        |
        +--> Issue #8 attribute registry
        |       `-- optional attribute-value embedding artifacts
        |
        `--> Issue #12 product text -> product embedding artifacts
                (optional/generated .npy + row metadata + manifest)

RUNTIME / ONE AGENT PROCESS

facts + registry + valid #12 artifact (when available)
        |
        v
Issue #13 in-memory product/fact indexes and product retriever
        |
        +--> reset(session_id, user_profile)
        |       `-- Issue #14 process-local session state
        |
user turn -> Issue #15 respond()
        |
        +--> first turn only: Issue #6 BUYING/BROWSING mode
        |
        +--> every turn: Issue #7/#8 canonical constraints
        |       `-- merge into session state; stale override facts are removed
        |
        +--> Issue #13 retrieval
        |       +-- BUYING: structured intersection -> controlled relaxation
        |       |              -> dense full-catalog fallback if necessary
        |       `-- BROWSING: dense full-catalog retrieval + preference boosts
        |
        +--> Issue #16 deterministic optional clarification (at most one)
        |
        `--> response contract: message + ask_attribute + current Top-K
                |
                v
Issue #17 evaluator-side fixed Manual400 metrics and diagnostics
```

The boundary is intentional: Issues #7/#8 resolve user language into canonical
values, Issue #13 resolves canonical state and semantic context into products,
and Issue #15 only assembles the end-to-end response. Product retrieval must
not repeat user-language matching. Issue #17 may inspect evaluator-visible
outputs and timing, but hidden targets and simulator facts never enter the
Agent path.

## Runtime versus optional/generated artifacts

The repository has two deliberately different paths:

- The current starter runtime loads the in-memory `ProductRetriever` from
  `starter/retrieval.py` at Agent construction. It builds catalog, canonical
  fact, categorical-inverted-index, and price-lookup structures; valid product
  vectors are used when available. If optional vectors are absent or invalid,
  it ranks the in-memory catalog/fact pool deterministically using available
  constraint scores and catalog order.
- The generated Issue #8 dictionary under `data/derived/dictionary/` and the
  Issue #12 product artifacts under `data/derived/product_embeddings/` are
  build outputs. They are loaded once when present and valid, not regenerated
  for every session. Attribute embeddings remain optional for #8 semantic
  fallback; product embeddings are required for the dense #13 path but are not
  required by the current compatibility runtime.
- A missing optional artifact must select a documented fallback or make the
  corresponding advanced path unavailable; it must not create pseudo-vectors,
  silently change row-to-ASIN mapping, or turn a normal retrieval miss into an
  invalid Agent response.

The current checkout contains the existing deterministic routing,
canonicalization, SQLite/lexical starter, process-local session container, and
one-step question utility in different maturity levels. The #12–#17 sections
below define the intended shared contracts; they do not claim that every
planned retriever, loader, integration, or diagnostic report already exists.

## Issue #5 input contract

The registry consumes deterministic canonical-facts JSONL. The preferred
flattened record is:

```json
{
  "parent_asin": "B123",
  "category": ["hiking_boots"],
  "brand": "example_brand",
  "price": 79.99,
  "color": ["black"],
  "material": ["leather"],
  "size": [],
  "style": ["casual"],
  "feature": ["waterproof"],
  "use_case": ["hiking"]
}
```

The builder consumes the flattened records emitted by Issue #5. If starting from
successful annotation JSONL with facts nested under `facts`, run the Issue #5
facts builder first. `price` remains numeric and is not a categorical registry field.

## Issue #8: canonical attribute registry

`dictionary/registry.py` owns the shared registry. The builder is
`scripts/build_attribute_dictionary.py`.

The builder aggregates the eight categorical fields:

```text
category, brand, color, material, size, style, feature, use_case
```

Each unique value receives a stable ID:

```text
<attribute>:<canonical_value>
```

For example, `feature:waterproof`. Counts are the number of distinct product
records containing the value; duplicate values within one product count once.
Null and empty values are excluded, and values are emitted in deterministic
order without semantic expansion.

The generated directory is:

```text
data/derived/dictionary/
├── canonical_values.json       # ID, attribute, value, normalized surface, count
├── normalized_lookup.json      # attribute-scoped exact lookup; ambiguity is preserved
├── embedding_metadata.json     # vector row -> canonical ID metadata
├── attribute_embeddings.npy    # optional shared semantic matrix
└── manifest.json               # source hash, normalization, model, dimensions, counts
```

The normalized lookup performs only lexical normalization: Unicode NFKC,
case-folding, separator normalization, safe punctuation handling, and
whitespace collapse. It does not create aliases such as `dark blue -> navy`.

Semantic lookup uses one shared matrix with metadata-backed attribute row
views. The default semantic set is `category`, `color`, `material`, `style`,
`feature`, and `use_case`; `brand` and `size` remain exact/structured, and
`price` remains numeric. Semantic results expose a similarity score and are
accepted only when they pass the configured threshold and optional Top-1 vs
Top-2 margin.

Build the exact registry without model dependencies:

```powershell
python -m scripts.build_attribute_dictionary --no-embeddings
python -m scripts.validate_attribute_dictionary
```

Embedding generation is optional and requires the dependencies in
`requirements-embeddings.txt` plus a compatible local model. The embedding
model is not selected or invoked by the starter automatically.

## Issue #7: utterance canonicalization

`starter/routing/constraints.py` exposes the existing `extract_constraints`
entry point. When a generated Issue #8 dictionary is present, its flow is:

```text
one user utterance
    |
    v
1. structured price and numeric-size parsing
    |
    v
2. longest-first exact normalized dictionary matches
    |
    v
3. mark matched spans and retain unresolved meaningful text
    |
    v
4. optional semantic matcher over the residual phrase
    |
    v
canonical fields + internal provenance
```

Exact matches outrank semantic matches. An ambiguous surface is left
unresolved rather than selected arbitrarily. A semantic result must refer to a
known registry `canonical_id` and meet the confidence threshold; otherwise it
is recorded as unmapped. Provenance records the canonical ID, raw phrase,
resolution method (`structured`, `exact`, or `semantic`), and confidence while
the public `ShoppingConstraints.as_dict()` shape remains unchanged.

Until Issue #5 facts have been built locally, the starter falls back to its
pre-existing offline vocabulary so the competition agent remains runnable. A
generated dictionary automatically becomes the runtime source when placed at
`data/derived/dictionary/`.

## Issue #9: product retrieval boundary

Issue #9 is architecture-only in this change. It starts after Issues #6–#8:

```text
intent mode + canonical constraints + semantic session context
    |
    +-- BUYING: structured set intersection and price filtering
    |            -> constraint-aware scoring
    |            -> optional dense scoring among survivors
    |
    +-- BROWSING: dense retrieval over product embeddings
                 -> canonical-preference boosts
```

Buying is precision-first. Explicit requirements dominate dense similarity.
If the strict pool is empty or too small, relax the lowest-confidence semantic
or soft preference first and preserve explicit exact constraints and numeric
budget limits. Browsing is recall-first: vague preferences are boosts rather
than hard intersections.

Both tracks return the same downstream candidate shape:

```json
{
  "parent_asin": "B123",
  "retrieval_mode": "BUYING",
  "dense_score": 0.82,
  "constraint_score": 1.0,
  "matched_constraints": ["category:hiking_boots", "color:black"],
  "violated_constraints": [],
  "relaxed_constraints": []
}
```

The frozen 50,000-product catalog is small enough for in-memory product facts,
inverted sets, price lookup, and exact dense search over a versioned product
embedding matrix. Product embeddings are distinct from Issue #8 attribute
embeddings and should use stable text combining title, canonical facts, and
selected source fields.

The first retrieval MVP deliberately excludes a parallel BM25 product branch,
result-list fusion, hosted LLM reranking, external vector databases, and DP
logic. Those require benchmark evidence and separate architecture updates.

## Issue #12: offline deterministic product embedding artifacts and loader

Status: planned/generated artifact boundary. Product embeddings are not
created by `Agent.respond()` and are not required by the current compatibility
runtime. The artifact, when generated, is separate from the Issue #8
attribute-value matrix.

The offline builder consumes Issue #5 canonical facts plus stable catalog text
where it adds useful semantics. Product text is deterministic and excludes
raw JSON noise, annotation provenance, and hidden evaluator fields. A practical
representation is:

```text
Title: <title>
Category: <category values>
Brand: <brand>
Material: <material values>
Color: <color values>
Style: <style values>
Features: <feature values>
Use cases: <use_case values>
Description: <short useful source description>
```

The build flow is:

```text
catalog facts + selected source fields
        -> deterministic product text
        -> one configured embedding model
        -> L2-normalized float32 vectors
        -> matrix + row metadata + manifest
```

The expected output is:

```text
data/derived/product_embeddings/
├── product_embeddings.npy
├── product_embedding_metadata.json
└── manifest.json
```

`product_embeddings.npy` is a dense matrix with one row per catalog product,
approximately `(50000, embedding_dim)`. Metadata preserves the exact row order:

```json
[
  {"row": 0, "parent_asin": "B001..."},
  {"row": 1, "parent_asin": "B002..."}
]
```

The manifest records the model and configuration identity, dimension,
normalization policy, catalog/facts version, and build metadata. For identical
catalog, facts, model, and configuration, vector content and row order must be
reproducible; a generation timestamp is descriptive metadata only.

The lightweight loader validates the manifest, matrix dimensionality and
finite values, row count, contiguous row numbers, unique `parent_asin` values,
catalog/facts compatibility, and the row-to-ASIN mapping before exposing the
matrix. With normalized vectors, exact NumPy inner product or FAISS
`IndexFlatIP` is sufficient for 50,000 products:

```text
query text -> same configured encoder -> normalize
           -> exact inner-product search
           -> row numbers -> parent_asin metadata
```

The loader does not call the model once per product at runtime. If the artifact
is absent or invalid, the runtime must use the documented compatibility
fallback rather than synthesize pseudo-embeddings or silently use a mismatched
row mapping.

## Issue #13: Buying/Browsing in-memory retrieval and fallback boundary

Status: integrated MVP runtime for Issues #13–#16. The current Agent uses
the in-memory canonical-facts retriever, process-local session state, and
deterministic clarification policy. When optional dense artifacts are absent or
invalid, retrieval uses the documented compatibility fallback; this is not a
second product-lexical branch.

Load once at Agent construction:

```text
canonical product facts
product_by_asin
product embedding matrix and row metadata, when valid
in-memory categorical inverted sets
numeric price lookup
```

The structured indexes have the shape:

```text
category[value] -> set(parent_asin)
brand[value]    -> set(parent_asin)
color[value]    -> set(parent_asin)
material[value] -> set(parent_asin)
size[value]    -> set(parent_asin)
style[value]   -> set(parent_asin)
feature[value] -> set(parent_asin)
use_case[value] -> set(parent_asin)
```

Price remains numeric and is checked against the product facts. The retriever
accepts the selected mode, canonical constraints, and current session/query
context; it does not parse raw user language.

### Buying

Buying is structure-first:

```text
50k facts -> category/attribute set intersection -> price check
          -> constraint-aware scoring -> dense rank among survivors
```

Explicit canonical requirements are stronger than semantic similarity. A
nearby vector cannot rescue a product that violates a strong explicit
category, attribute, or budget requirement.

If the strict pool is empty or too small, relax the weakest non-essential
categorical or semantic preference one at a time, using #7/#8 provenance. Keep
explicit category and numeric budget requirements as long as possible. A
simple MVP floor such as roughly 50 candidates is a tuning target, not a
correctness rule. If controlled relaxation still cannot produce a useful pool,
fall back to dense retrieval over the full catalog using the current known
context and record that fallback in candidate provenance.

### Browsing

Browsing is recall-first:

```text
current session context -> exact dense search over the full catalog
                         -> broad candidate pool
                         -> canonical-preference boosts
```

Vague preferences are not hard intersections. A practical MVP can retain about
100 dense candidates for downstream work; pool sizes are tuning targets rather
than fixed correctness requirements.

### Shared candidate and fallback contract

Both modes return the same candidate representation so ranking,
clarification, and future posterior logic do not depend on the retrieval route:

```json
{
  "parent_asin": "B123",
  "retrieval_mode": "BUYING",
  "dense_score": 0.83,
  "constraint_score": 1.0,
  "matched_constraints": ["color:black", "feature:waterproof"],
  "violated_constraints": [],
  "relaxed_constraints": []
}
```

An unresolved phrase from #7/#8 leaves known constraints usable; it does not
cause retrieval to parse the phrase a second time. Missing dense artifacts
leave dense scores unavailable; retrieval ranks the in-memory catalog/fact pool
deterministically using available constraint scores and catalog order. An empty
strict Buying pool selects controlled relaxation before broad fallback. In every
case the Agent still returns the best available valid candidates, rather than an
exception or an ask-only response.

## Issue #14: process-local session state

Status: target session contract. The current starter already keeps session
data in the Agent process, but the complete mode/constraint state below is the
boundary required for #15 integration.

`reset(session_id, user_profile)` replaces the entry for that session with a
fresh, isolated state. Conceptually:

```json
{
  "session_id": "...",
  "mode": null,
  "constraints": {
    "category": [],
    "brand": [],
    "price_min": null,
    "price_max": null,
    "color": [],
    "material": [],
    "size": [],
    "style": [],
    "feature": [],
    "use_case": []
  },
  "asked_attributes": [],
  "last_recommendations": [],
  "last_user_message": null,
  "turn": 0
}
```

The first user turn runs the #6 Buying/Browsing router and stores the mode.
Later clarification replies are treated primarily as new constraints; the
router is not rerun on every answer. List values merge without duplication,
price bounds refine or replace earlier bounds, and an explicit correction wins
over the stale value (for example, `black` becomes `brown`, not both).

An explicit intent override clears stale shopping constraints, reinitializes
the new goal, and applies the new message. This prevents the old goal from
contaminating #13 retrieval.

State is deliberately process-local for the MVP: no database, Redis, or other
external persistence is required. Separate worker processes must have separate
Agent instances and session maps; session state is not shared across workers.
The evaluator's fixed sessions and hidden facts are not session state available
to the Agent.

## Issue #15: Agent integration and response contract

Status: integrated end-to-end runtime. The response shape remains
evaluator-compatible, while #13 retrieval, #14 state, and #16 clarification are
wired through Agent. Missing optional dense artifacts use the documented
compatibility fallbacks.

Expensive data is loaded once at construction, not once per session:

```text
catalog facts + #8 registry + valid #12 product artifact + #13 indexes
```

The intended control flow is:

```text
respond(session_id, user_message, turn, top_k)
        -> load process-local #14 state
        -> first turn only: route with #6 and store mode
        -> canonicalize this message with #7/#8
        -> merge constraints and override state
        -> retrieve with #13
        -> choose optional #16 clarification
        -> store turn and current recommendations
        -> return the strict response contract
```

The contract is:

```json
{
  "message": "Do you have a material preference?",
  "ask_attribute": "material",
  "recommendations": [
    {"parent_asin": "B000..."}
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0}
}
```

`message` is customer-facing text. `ask_attribute` is one supported enum value
or `null`. Recommendations are ordered best-first, use valid unique catalog
`parent_asin` values, and are limited to the requested `top_k` by the Agent;
the evaluator keeps the first 10 valid unique IDs. An optional finite numeric
`score` is allowed. `usage` is optional and, when present, contains only
non-negative `prompt_tokens` and `completion_tokens`. Do not add hidden or
target-specific fields.

Every scoreable turn returns the current best recommendations, including a
turn that also asks a clarification. If canonicalization is partial or strict
retrieval fails, retain known constraints, select the documented fallback, and
return a valid response. The Agent never reads Manual400 targets, hidden facts,
or evaluator-only labels.

## Issue #16: always-recommend deterministic clarification

Status: deterministic one-step MVP policy. The current starter already follows
this pattern over its candidate pool; the #16 contract keeps it when candidates
come from #13.

After retrieval, every turn follows one rule:

```text
current candidate pool -> current best Top-K recommendations
                        + optional ONE clarification attribute
```

Recommendations are never replaced by an ask-only turn. Candidate attributes
are the supported fields `category`, `material`, `color`, `size`, `style`,
`brand`, `budget`, `feature`, and `use_case`; `other` is not the default MVP
path. Do not ask an attribute that is already known, already asked, declined,
or unlikely to change the candidate set.

The deterministic utility estimates, for each remaining attribute:

1. coverage of usable values in the current candidates;
2. diversity or split quality of those values;
3. expected concentration or remaining-candidate benefit; and
4. a mode prior, with penalties for missing, constant, known, or asked fields.

An entropy/Gini or equivalent one-step calculation is sufficient. Return
`ask_attribute = null` when the pool is already concentrated, coverage is too
poor, or no remaining question is useful. The human message should correspond
to the structured enum value, which is what the simulator consumes. This MVP
does not add recursive DP/lookahead or an LLM question policy; those can replace
the utility behind the same `respond()` contract later.

## Issue #17: evaluator-side fixed Manual400 diagnostics

Status: evaluator-side diagnostic contract, not a benchmark result. The fixed
GPTAnnotation/Manual400 source contains 400 sessions: 160 Buying, 160
Browsing, 60 Intent Override, and 20 Boundary. The evaluator validates the
fixed shape, simulates replies, validates the response contract, and scores
exact catalog `parent_asin` matches. Hidden target data remains on the
evaluator side.

The current full-run command is:

```powershell
python -m evaluator.hard_evaluator
```

It uses `data/derived/gptannotation/sessions.jsonl` by default and writes
`results_gptannotation.json`; the current CLI has no sampling flag. This
document does not claim that the command has been run or that any metric has
been produced.

When a run is authorized, report the overall and per-scenario:

```text
HitRate@10, MRR, MTTC, Efficiency, TechnicalScore
```

Evaluator-side diagnostics should include:

- cumulative hit rate by turn and first-hit-turn distribution;
- first-hit target-rank bins: 1, 2–3, 4–5, 6–10, and miss;
- deterministic dictionary matches, semantic-fallback matches, unresolved
  phrases, and canonical constraints per turn;
- mode, Buying candidate counts before/after filtering, controlled-relaxation
  use, dense full-catalog fallback use, and Top-K retrieval latency;
- `ask_attribute` frequency by attribute, non-null-question rate, and whether
  the next reply shrinks the candidate pool; and
- Agent startup time, mean/p50/p95 response latency, and total evaluator time.

These counters are diagnostics only. They must not use hidden target facts in
Agent logic or alter the fixed targets, hidden facts, customer replies, scoring
rules, or session order. A reproducible report may record the Git SHA,
artifact/model versions, aggregate metrics, per-session results, and observed
failure categories, but its existence must not be implied until a run actually
produces it.

## Follow-up implementation slices

The #13–#16 runtime path is implemented in the current worktree. Remaining
work is bounded to artifacts, measurement, and evidence-led quality updates:

1. Generate optional #12 product embedding artifacts from versioned product text,
   then validate the manifest, matrix shape/dtype/normalization, contiguous
   metadata rows, catalog ASIN coverage, and deterministic row mapping.
2. Keep artifact loading optional and preserve the deterministic in-memory
   catalog/fact compatibility fallback when vectors are absent or invalid.
3. When benchmark execution is authorized, measure the fixed Manual400
   end-to-end path and report core/scenario metrics plus turn, rank,
   canonicalization, retrieval, clarification, and latency diagnostics. Do
   not change benchmark inputs/scoring or imply a result before a run
   produces it.
4. Use those measurements to prioritize improvements to retrieval, ranking,
   canonicalization, and clarification. Change one boundary at a time and
   compare against public200 or another unseen validation set; do not add a
   second BM25/lexical route without evidence and an intentional architecture
   update.

These follow-ups should preserve exact ASIN validation, deterministic ordering,
the shared Candidate/Agent response contracts, and the current no-vector
fallback. Any generated artifact or benchmark report must be labeled as
generated output rather than runtime source-of-truth.
