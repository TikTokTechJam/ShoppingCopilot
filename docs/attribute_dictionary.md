# V5 canonical attribute dictionary

The generated dictionary is the runtime vocabulary for categorical product
facts. It is built deterministically from the aggregated V5 annotation JSONL;
it is not a hand-maintained synonym list.

```text
V5 annotations.jsonl
        |
        +-> exact canonical registry
        |       +-> canonical_values.json
        |       +-> normalized_lookup.json
        |       +-> manifest.json
        |
        +-> optional per-attribute BGE matrices
                +-> attribute_embeddings/*.npy
                +-> attribute_embeddings/metadata.json
```

## Field contract

The dictionary has seven fields:

```text
category, brand, color, material, style, feature, use_case
```

Price remains numeric and size remains a structured runtime value. Neither is
a dictionary attribute.

The active semantic subset is:

```text
category, color, material, style, feature, use_case
```

Brand is exact-only and has no semantic matrix.

## Canonical values

Values are lowercase natural text with spaces. Canonical IDs are
attribute-scoped machine identifiers:

```text
value:         "moisture wicking"
canonical_id:  "feature:moisture_wicking"
normalized:    "moisture wicking"
```

Lexical normalization applies Unicode NFKC and case folding, converts
separators to spaces, removes apostrophes inside words, collapses whitespace,
and trims:

```text
New_Balance  -> new balance
New-Balance  -> new balance
V_Neck       -> v neck
Levi's       -> levis
O'Neill      -> oneill
```

The display value can still preserve a readable apostrophe. Normalization does
not stem, singularize, or invent synonyms.

## Exact lookup and ambiguity

Runtime exact matching is token-based, longest-first, and non-overlapping. It
does not use arbitrary substring matching.

A normalized surface may exist in more than one attribute. The runtime resolves
an ambiguous exact match only when:

1. directly attached directional context identifies the field; or
2. the leading catalog count owns at least 75% of the candidate count and is at
   least three times the runner-up.

Otherwise the surface remains unresolved. Supported direct contexts include
forms such as `brand VALUE`, `VALUE brand`, `made by VALUE`, `color VALUE`,
`VALUE material`, `style VALUE`, `VALUE feature`, and `for VALUE`.

A small guard suppresses confirmed single-word brand/query collisions such as
`find` unless explicit brand context is present. Multi-word brands and
non-brand fields are unaffected.

## Semantic matching

The active semantic model is `BAAI/bge-small-en-v1.5` with 384-dimensional,
L2-normalized vectors. Canonical phrases are encoded without a query or
document prefix. Each semantic attribute has its own matrix so an
attribute-scoped clarification answer searches only the relevant space.

The model must be available locally. Runtime loading never downloads it and
rejects an incompatible model family or vector dimension.

Semantic matches retain their cosine similarity and evidence phrase. The
default minimum similarity is `0.80`; rare canonical values can be excluded
from semantic matching by the registry's count guard. Exact registry loading
does not require embedding files.

## Artifact layout

The current generated layout is:

```text
data/derived/annotations/v5/dictionary/
|-- canonical_values.json
|-- normalized_lookup.json
|-- manifest.json
`-- attribute_embeddings/
    |-- category_embeddings.npy
    |-- color_embeddings.npy
    |-- material_embeddings.npy
    |-- style_embeddings.npy
    |-- feature_embeddings.npy
    |-- use_case_embeddings.npy
    `-- metadata.json
```

`canonical_values.json` records canonical IDs, attributes, natural values,
normalized surfaces, and distinct-product counts. `normalized_lookup.json`
preserves one-to-many surfaces. The manifests record source provenance,
normalization, field counts, model identity, dimension, and normalization.

An older builder mode can create a single `attribute_embeddings.npy` plus
`embedding_metadata.json` in the dictionary root. The loader retains
compatibility with that format, but new active builds should use the separate
per-attribute matrices produced by `scripts.build_v5_attribute_embeddings`.

## Build

The default input is
`data/derived/annotations/v5/annotations.jsonl`. Build the exact registry first:

```powershell
python -m scripts.build_attribute_dictionary `
  --input data/derived/annotations/v5/annotations.jsonl `
  --input-format v5 `
  --output-dir data/derived/annotations/v5/dictionary `
  --no-embeddings
```

Then optionally download the pinned model and build the separate semantic
matrices:

```powershell
python -m pip install -r requirements-embeddings.txt
python -m scripts.setup_bge_attribute_model
python -m scripts.build_v5_attribute_embeddings `
  --dictionary-dir data/derived/annotations/v5/dictionary `
  --output-dir data/derived/annotations/v5/dictionary/attribute_embeddings `
  --model models/bge-small-en-v1.5
```

Validate after either an exact-only or semantic build:

```powershell
python -m scripts.validate_attribute_dictionary `
  --directory data/derived/annotations/v5/dictionary
```

The build must be rerun whenever the annotation aggregate, normalization
policy, semantic-field set, or embedding model changes. Keep the aggregate and
dictionary manifests with every benchmark result even when the large matrices
remain outside Git.

## Runtime loading

`starter.routing.constraints` looks only at:

```text
data/derived/annotations/v5/dictionary
```

The exact files are required. Missing or invalid files are a configuration
error because the agent intentionally has no hidden fallback vocabulary. If
semantic files or the local BGE model are unavailable, exact lookup remains
available but semantic matching is disabled.

Under the current agent split, only exact brand plus structured size and price
enter the structured scorer. The other six descriptive fields need semantic
state to affect ranking. This behavior is documented in
[`../Architecture.md`](../Architecture.md), not implied by the dictionary file
format itself.
