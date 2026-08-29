# V4 canonical attribute dictionary

The dictionary is a deterministic exact-lookup layer built from successful V4
annotation records:

```text
V4 annotations
      ↓
exact canonical dictionary
      ↓
runtime registry
      ↓
longest-first exact utterance matching
```

The input is the V4 annotation JSONL. Each record contains nested `facts` and
an annotation status. Only records whose status is `success` contribute values;
failed records are reported and do not block the build.

The dictionary has exactly seven fields:

```text
category, brand, color, material, style, feature, use_case
```

`price` remains numeric/structured and `size` remains a structured runtime
constraint. Neither is included in this dictionary. V4 `brand` values are
lists, and each brand value becomes an independent dictionary entry.

Canonical values remain lowercase natural text with spaces:

```text
value:         "moisture wicking"
canonical_id:  "feature:moisture_wicking"
normalized:    "moisture wicking"
```

Machine IDs are attribute-scoped and derived from the normalized surface. A
normalized surface can map to more than one attribute. The runtime resolves
those exact matches conservatively: directly attached directional context is preferred,
then catalog-count dominance is accepted only when the leading attribute owns at
least 75% of the candidate count and is at least 3 times the runner-up. Otherwise
the surface remains unresolved rather than being assigned arbitrarily.

Attribute context is directional and directly attached to the matched value.
Supported forms include `brand VALUE`, `VALUE brand`, `from VALUE`, `made by
VALUE`, `by VALUE`, `color/colour VALUE`, `VALUE color/colour`, `made of/from/
with VALUE`, `VALUE material/fabric`, `style/fit VALUE`, `VALUE style/fit`,
`feature(s) VALUE`, `VALUE feature`, and `for`, `good for`, `use for`, or `for
use VALUE`. Proximity alone does not resolve ambiguity; the cue words do not
consume or reserve other dictionary matches.

A small brand-only collision guard suppresses confirmed query-language terms
(currently `find`) when they appear as single-word brands without explicit
brand context. Cues such as `brand`, `from`, `made by`, and `by` restore an
intentional match. Multi-word brands and non-brand attributes are unaffected.

Normalization is lexical only. It applies Unicode NFKC and case folding,
converts separators to spaces, removes apostrophes inside words, collapses
whitespace, and trims. For example:

```text
New_Balance  → new balance
New-Balance  → new balance
V_Neck       → v neck
Levi's       → levis
O'Neill      → oneill
```

The readable canonical value may still be `levi's` or `o'neill`. The exact
path does not add stemming, synonym aliases, or plural conversion.

Optional semantic attribute matching uses a separate local-only
`BAAI/bge-small-en-v1.5` encoder for short canonical phrases. It generates
384-dimensional L2-normalized vectors for the six descriptive attributes
(`category`, `color`, `material`, `style`, `feature`, and `use_case`). Phrases are
encoded directly without a retrieval instruction or prefix. This BGE artifact
is the active semantic layer. Brand remains exact-only; whole-product vector
retrieval is retired from the Agent path.

## Artifacts

An exact-only build writes:

```text
data/derived/annotations/v5/dictionary/
├── canonical_values.json
├── normalized_lookup.json
└── manifest.json
```

`canonical_values.json` stores deterministic entries with the canonical ID,
attribute, natural value, normalized surface, and distinct-product count.
`normalized_lookup.json` stores attribute-scoped lookup lists and preserves
one-to-many ambiguity. `manifest.json` records source provenance, V4 coverage,
normalization version, counts, and whether BGE attribute embeddings were
generated. When embeddings are enabled, the manifest records the exact BGE
model, dimension, L2 normalization, and the absence of a query prefix. The
runtime rejects other attribute model families or dimensions rather than
mixing them with the BGE matrix.

No embedding files are required for the exact-only flow. The optional semantic
artifact consists of `attribute_embeddings.npy` and
`embedding_metadata.json`; it contains canonical values only and never product
embeddings.

## Build and validate

Use the actual V4 annotation output directly:

```powershell
python -m scripts.build_attribute_dictionary `
  --input data/derived/annotations/v5/annotations.jsonl `
  --output-dir data/derived/annotations/v5/dictionary `
  --no-embeddings

python -m scripts.validate_attribute_dictionary `
  --directory data/derived/annotations/v5/dictionary
```

The build command defaults to `data/derived/annotations/v5/annotations.jsonl`,
the current annotation output in this repository. Use `--input` to point it
at another annotation JSONL location, such as a generated release under
`data/derived/annotations/v5/annotations.jsonl`. The optional embedding command
requires the dependencies in `requirements-embeddings.txt`; it is intentionally
not run as part of normal repository checks.

For the BGE semantic attribute artifact, download the model once and build only
the canonical attribute vectors:

```powershell
python -m scripts.setup_bge_attribute_model

python -m scripts.build_attribute_dictionary `
  --input data/derived/annotations/v5/annotations.jsonl `
  --output-dir data/derived/annotations/v5/dictionary `
  --embedding-model models/bge-small-en-v1.5

python -m scripts.validate_attribute_dictionary `
  --directory data/derived/annotations/v5/dictionary
```

The runtime requires the generated registry. Missing or incomplete dictionary
artifacts are configuration errors; categorical extraction cannot proceed
without these artifacts.
