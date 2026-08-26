# Shopping Copilot architecture

Status: MVP architecture and component contract for Issues #5–#9.

This document is the source of truth for the data-to-search boundaries. It
describes the implemented Issue #7/#8 data components and the proposed Issue
#9 retrieval boundary. Product retrieval, reranking, candidate-posterior, and
clarification policy remain follow-up work.

## End-to-end boundary

```text
catalog.jsonl
    |
    v
Issue #5: canonical product facts
    |
    v
Issue #8: canonical attribute registry
    |                         \
    | exact normalized lookup  \ one shared attribute embedding matrix
    v                           \
Issue #7: structured constraints + semantic fallback
    |
    v
mode from Issue #6 + canonical constraints + semantic session context
    |
    v
Issue #9: product retrieval architecture
    |
    v
shared candidate contract -> later ranking / DP / Top-10 response
```

The boundary is intentional: Issue #7 resolves user language into canonical
values, while Issue #9 resolves canonical state and semantic context into
actual products. Product retrieval must not repeat user-language matching.

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

## Follow-up implementation slices

1. Load Issue #5 facts and build product inverted indexes and price lookup.
2. Define versioned product embedding text and generate product artifacts.
3. Add exact in-memory product vector search with ASIN metadata validation.
4. Implement Buying filtering, scoring, and provenance-based relaxation.
5. Implement Browsing dense retrieval and preference boosts.
6. Adapt both modes to the shared candidate contract and later ranking/DP.

Each slice should measure recall, Top-10 accuracy, latency, and fallback
behavior independently before adding another retrieval route or reranker.
