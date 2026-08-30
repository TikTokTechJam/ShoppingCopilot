# Data and generated artifacts

The repository tracks only small development datasets. The frozen catalog,
public release set, model outputs, dictionaries, and embedding matrices are
excluded from Git unless explicitly listed below.

## Source data

### `catalog.jsonl`

The frozen 50,000-product `Clothing_Shoes_and_Jewelry` catalog. Place the
released file at this exact path. It is required by every evaluator and defines
the valid `parent_asin` universe.

Expected product fields include `parent_asin`, title, features, description,
price, categories, details, average rating, rating count, and store. The agent
must not modify this file.

### `public_set.jsonl`

The released 200-session development set: 80 Buying, 80 Browsing, 30 Intent
Override, and 10 Boundary sessions. It contains safe aggregate profiles and
public labels for local evaluation.

This file is not tracked in Git.

## Tracked development data

### `derived/gptannotation/sessions.jsonl`

A fixed 400-session hard development benchmark used by
`python -m evaluator.hard_evaluator`. It contains 160 Buying, 160 Browsing, 60
Intent Override, and 20 Boundary sessions. It is not the organizer's private
holdout.

### `derived/intent/dev_set.jsonl`

A small intent-routing development set used by the routing evaluation tools.

## Generated local data

The normal generated layout is:

```text
derived/
|-- annotations/v5/
|   |-- category.jsonl
|   |-- brand.jsonl
|   |-- color.jsonl
|   |-- material.jsonl
|   |-- feature.jsonl
|   |-- use_case.jsonl
|   |-- annotations.jsonl
|   `-- dictionary/
|       |-- canonical_values.json
|       |-- normalized_lookup.json
|       |-- manifest.json
|       `-- attribute_embeddings/
|           |-- *_embeddings.npy
|           `-- metadata.json
`-- ... optional experiment outputs
```

The aggregate `annotations.jsonl` joins the six V5 attribute files in catalog
order. The dictionary is built from that aggregate and is required by the
runtime agent. All of these generated paths are ignored by Git.

## Safety and provenance

- Never place API keys, raw user identifiers, private evaluation data, or
  organizer-only files in this directory.
- Preserve catalog, annotation, dictionary, and model checksums with benchmark
  results.
- Do not compare result files built from different generated artifacts unless
  the comparison explicitly studies that artifact change.
- Follow [`../DATA_ATTRIBUTION.md`](../DATA_ATTRIBUTION.md) before using or
  redistributing source-derived data.
