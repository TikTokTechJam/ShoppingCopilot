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
normalized surface can map to more than one attribute; ambiguous matches are
left unresolved by the runtime rather than assigned arbitrarily.

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

The readable canonical value may still be `levi's` or `o'neill`. The system
does not add stemming, synonym aliases, plural conversion, or semantic lookup.

## Artifacts

An exact-only build writes:

```text
data/derived/dictionary/
├── canonical_values.json
├── normalized_lookup.json
└── manifest.json
```

`canonical_values.json` stores deterministic entries with the canonical ID,
attribute, natural value, normalized surface, and distinct-product count.
`normalized_lookup.json` stores attribute-scoped lookup lists and preserves
one-to-many ambiguity. `manifest.json` records source provenance, V4 coverage,
normalization version, counts, and whether embeddings were generated.

No embedding files are required for the exact-only flow. Optional embedding
support remains available for a later stage, but is not part of the first
baseline.

## Build and validate

Use the actual V4 annotation output directly:

```powershell
python -m scripts.build_attribute_dictionary `
  --input data/derived/annotations/v4/annotations.jsonl `
  --output-dir data/derived/dictionary `
  --no-embeddings

python -m scripts.validate_attribute_dictionary `
  --directory data/derived/dictionary
```

The runtime loads the generated registry when both exact artifact files are
present. If they are absent, the existing legacy vocabulary fallback remains
available so the starter Agent can still run.
