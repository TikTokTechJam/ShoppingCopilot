# Retrieval architecture

This document describes the active runtime path. Whole-product embedding
retrieval is retired; the runtime uses the generated V5 canonical dictionary
and its BGE semantic attribute matrices.

## Runtime flow

```text
user message
  -> intent routing
  -> structured + exact canonical extraction
  -> residual stopword-filtered 1/2/3-gram extraction
  -> BGE-small semantic attribute matching
  -> session constraints and evidence
  -> structured scorer / reranker
  -> hard filters and recommendation exclusions
  -> deterministic Top10
```

The active semantic model is `BAAI/bge-small-en-v1.5` with 384-dimensional
L2-normalized vectors. Each attribute has its own canonical-value matrix:

```text
category, color, material, style, feature, use_case
```

Brand remains exact-only and has no semantic matrix. The BGE model is loaded
locally through `SHOPPING_ATTRIBUTE_EMBEDDING_MODEL` or the default local model
directory. Runtime loading never downloads a model.

## Matching and scoring

Layer 1 still handles numeric and structured values, exact canonical matches,
price bounds, size, and intent state. The semantic matcher removes configured
stopwords, creates one-, two-, and three-token phrases, and searches only the
selected attribute matrices. Matches at or above the configured threshold are
stored as evidence with their cosine similarity.

The existing product scorer uses those evidence similarities when calculating
candidate points. It does not load or compare whole-product vectors. Missing
semantic evidence is not treated as a hard contradiction.

Price eligibility and recommendation exclusions remain hard. The same
`ProductRetriever` and `Agent` serve Buying and Browsing; Browsing changes
clarification and soft preference behavior, not the embedding model.

## Artifacts

The active artifacts are:

```text
data/derived/annotations/v5/dictionary/
├── canonical_values.json
├── normalized_lookup.json
├── manifest.json
└── attribute_embeddings/
    ├── category_embeddings.npy
    ├── color_embeddings.npy
    ├── material_embeddings.npy
    ├── style_embeddings.npy
    ├── feature_embeddings.npy
    ├── use_case_embeddings.npy
    └── metadata.json
```

The retired `data/derived/product_embeddings_jina` artifact is not discovered
or loaded by the evaluator. It may remain locally ignored for cleanup or
historical comparison, but it is outside the active runtime.

## Commands

Run the interactive canonical semantic tool:

```bash
python -m scripts.console_semantic_attribute_test
```

Run the evaluator without any product-embedding model configuration:

```bash
python -m evaluator.hard_evaluator
```
