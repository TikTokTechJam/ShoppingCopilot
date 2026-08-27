# Layer 2 product-field embedding artifacts

Layer 2 is the independent dense-retrieval path described in
`Architecture.md`. It reads only the immutable `data/catalog.jsonl`; it does
not require Tier 1, Tier 2, or Tier 3 facts and never rewrites the catalog.

## Views

Each catalog product receives four independent semantic views:

```text
categories   -> ordered product category path
title        -> original title
features     -> all feature bullets joined in source order
description  -> all description strings joined in source order
```

The optional selected-details view is intentionally not implemented in this
MVP. Details should not be embedded blindly because many keys are bookkeeping
or numeric metadata already handled by structured retrieval.

Empty fields are represented by a zero row and a `has_<view>` presence mask in
`product_embedding_metadata.json`. Missing text therefore contributes no
score and is never treated as negative evidence.

## Build

Use a local or injected embedding model for benchmark artifacts:

```bash
python -m scripts.build_layer2_embeddings \
  --catalog data/catalog.jsonl \
  --output-dir data/derived/product_embeddings \
  --model path/to/local/sentence-transformer
```

The builder loads local SentenceTransformers models with
`local_files_only=True`; it will not download a model or call an API. For a
dependency-light pipeline smoke test, omit `--model` and use the deterministic
hashing fallback, or provide `--embedder package.module:object`.

Pass `--generated-at-utc` when a byte-stable manifest is useful. The catalog
row order is preserved in every matrix and in the metadata mapping.

## Artifacts

```text
data/derived/product_embeddings/
├── category_embeddings.npy
├── title_embeddings.npy
├── features_embeddings.npy
├── description_embeddings.npy
├── product_embedding_metadata.json
└── manifest.json
```

All four matrices have shape `(product_count, embedding_dimension)`, use
`float32`, and have L2-normalized rows for present fields. The manifest records
the source catalog hash/version, model, dimension, normalization policy, view
names, row order, and generation configuration.

## Runtime search

```python
from product_embeddings import load_layer2_embedding_index

index = load_layer2_embedding_index(
    "data/derived/product_embeddings",
    expected_asins=["B07K34RX5J", "B07KCFS4VC"],
)
matches = index.search(query_embedding, top_k=10)
```

The loader validates all matrix shapes, dtypes, finite/L2-normalized present
rows, zero missing rows, metadata row numbers, unique ASINs, manifest counts,
and catalog order. Search normalizes one query vector and computes exact
inner products against all four views. The default score is a
presence-aware weighted average; callers can supply benchmark-tuned view
weights.

`ProductRetriever` automatically looks for a valid Layer 2 artifact under
`data/derived/product_embeddings` and uses it when a compatible query encoder
is supplied. Layer 1 constraints still run independently and are combined only
at candidate scoring. A compatibility loader can still read previously
generated single-matrix `product_embeddings.npy` artifacts.
