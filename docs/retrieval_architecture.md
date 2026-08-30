# Retrieval architecture

This is the retrieval-specific companion to
[`../Architecture.md`](../Architecture.md). It describes the evaluator factory's
active path and the optional paths that remain available for experiments.

## Active path

```text
session query + constraints
        |
        +-> budget eligibility
        +-> prior-recommendation exclusion
        +-> exact brand/size/price points
        +-> canonical semantic attribute points
        +-> BM25 product-text scores
        +-> catalog rating adjustment
        -> deterministic Top K
```

`ProductRetriever` loads the full catalog, optional V5 facts, inverted
attribute indexes, price/rating lookups, and an in-memory SQLite FTS5 BM25
index. The raw catalog defines valid ASINs and stable tie order.

The canonical semantic track covers category, color, material, style, feature,
and use case. It matches user phrases to canonical BGE attribute values and
then scores products through fact posting lists. It does not compare the query
with whole-product vectors.

## Eligibility and soft evidence

Budget and prior recommendation exclusions are the normal hard filters. Other
attributes are positive soft evidence; a missing field does not remove a
product. Backfill can relax budget and exclusion state rather than return fewer
than the requested Top K.

The base score configured in `starter/retrieval.py` is:

```text
structured + semantic_or_dense + 0.20 * BM25
```

A rating term then breaks or adjusts close scores. Buying and Browsing use the
same ranking weights in the current code.

## Important branch behavior

BM25 is computed independently but currently enters the final score only when
a semantic/dense score mapping is present. A structured-only query ranks by
structured points and rating; a query with neither constraints nor dense
evidence ranks by rating and catalog order. This is an implementation behavior
worth ablating, not a requirement of the architecture.

## Optional product embedding paths

`product_embeddings/` and `ProductRetriever` still support:

- one whole-product embedding matrix; and
- four-view Layer 2 matrices over categories, title, features, and description.

These paths need matching artifacts and an explicitly compatible query encoder.
`evaluator.agent_factory.build_evaluator_agent` supplies neither, so they are
not active in normal local or hard evaluation.

## Runtime artifacts

The active semantic artifact is:

```text
data/derived/annotations/v5/dictionary/attribute_embeddings/
|-- category_embeddings.npy
|-- color_embeddings.npy
|-- material_embeddings.npy
|-- style_embeddings.npy
|-- feature_embeddings.npy
|-- use_case_embeddings.npy
`-- metadata.json
```

The matching local model is `BAAI/bge-small-en-v1.5`. Runtime loading is local
only. Brand remains exact-only.

## Proposed next step

Make attribute postings, BM25, and optional dense retrieval independent
candidate generators, union their candidates, and fuse ranks before final
reranking. The rationale and experiment design are in
[`approaches.md`](approaches.md).
