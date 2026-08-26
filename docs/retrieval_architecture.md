# Product retrieval architecture (Issue #9)

Status: MVP proposal. This document is the Issue #9 architecture deliverable;
it does not implement retrieval; the authoritative cross-issue document is `architecture.md`.

## Boundary

Product retrieval begins only after Issue #6 has selected the session mode and
Issues #7/#8 have converted the utterance into canonical constraints and
semantic session context:

```text
utterance
  -> #7/#8 structured parsing and canonicalization
  -> mode + canonical constraints + semantic context
  -> #9 product retrieval
  -> shared candidate pool
  -> later ranking / posterior / clarification policy
```

Issue #9 must not repeat user-language matching. Exact surface matching belongs
to the Issue-8 registry; unresolved meaning is handled by the Issue-8 semantic
fallback.

## Shared runtime data

The MVP keeps the 50,000-product runtime in memory. It loads or builds:

- `product_by_asin` from the frozen catalog;
- canonical facts from Issue #5;
- structured inverted sets such as
  `feature["waterproof"] -> set(parent_asin)`;
- a price lookup;
- a product embedding matrix and row metadata.

No Postgres, external vector database, or hosted LLM is required. Issue-8
attribute vectors and Issue-9 product vectors are different artifacts:

```text
attribute vector: "handles heavy rain" -> feature:waterproof
product vector:   "black hiking boots for rainy walks" -> product ASINs
```

Product embeddings should be generated from a deterministic, versioned text
representation combining title, canonical facts, and selected useful source
fields. A title-only representation is insufficient. The intended artifacts
are `product_embeddings.npy` and `product_embedding_metadata.json`, where each
metadata row maps to one `parent_asin`. Exact in-memory cosine/inner-product
search is sufficient for 50,000 products.

## Buying flow

Buying is precision-first. Explicit constraints are hard requirements unless a
controlled relaxation is required:

```text
canonical constraints
  -> structured set intersection and price filtering
  -> constraint-aware scoring
  -> optional dense scoring among survivors
  -> shared candidate pool
```

For example, category, color, and feature sets are intersected before ranking.
Dense similarity may rank compliant candidates, but it must not rescue a
product that violates a strong explicit requirement.

If the strict pool is empty or too small, relax one constraint at a time using
provenance from #7/#8. Relax the lowest-confidence semantic or soft preference
first; retain exact explicit constraints and numeric budget limits longer.
The policy is deterministic and records every relaxed constraint.

## Browsing flow

Browsing is recall-first. It should not turn vague preferences into an
intersection of hard filters:

```text
full semantic session context
  -> dense retrieval over the 50k product matrix
  -> broad candidate pool
  -> canonical-preference boosts
  -> shared candidate pool
```

Known canonical preferences are ranking boosts or provenance, while dense
semantic relevance keeps exploratory products in consideration. A practical
MVP target is roughly 200 dense candidates before preference scoring and about
100 candidates handed downstream; those counts are tuning targets, not fixed
correctness rules.

## Shared candidate contract

Both tracks hand downstream components the same representation:

```json
{
  "parent_asin": "B123",
  "retrieval_mode": "BUYING",
  "dense_score": 0.82,
  "constraint_score": 1.0,
  "matched_constraints": [
    "category:hiking_boots",
    "color:black",
    "feature:waterproof"
  ],
  "violated_constraints": [],
  "relaxed_constraints": []
}
```

For Browsing, the same fields carry preference matches instead of strict
filter provenance. Later ranking, candidate-posterior, DP, and clarification
logic should consume this contract rather than depend on the retrieval route.

## Deliberate exclusions and fallbacks

The first MVP does not add:

- a parallel user-utterance-to-BM25 product branch;
- result-list fusion across independent lexical and dense routes;
- an LLM or hosted cross-encoder reranker;
- an external vector service;
- candidate-posterior or DP logic in the retriever.

If dense retrieval misses exact model names, rare brands, or unusual phrases,
that should be demonstrated by benchmark evidence before proposing a separate
lexical route. If structured Buying filters return no products, controlled
provenance-based relaxation is the first fallback. Browsing falls back to the
broad dense pool and preference boosts rather than strict filtering.

## Implementation breakdown

Follow-up implementation tickets can remain isolated:

1. Build a catalog/facts loader and structured inverted indexes.
2. Define and version product embedding text and generate the product matrix.
3. Add exact in-memory product vector search with ASIN metadata validation.
4. Implement Buying intersection, price handling, and provenance-based
   relaxation.
5. Implement Browsing dense retrieval and canonical preference boosts.
6. Adapt both flows to the shared candidate contract and integrate later
   ranking/clarification components.

These tickets should benchmark recall, Top-10 accuracy, latency, and failure
fallbacks independently before any route or reranker is expanded.
