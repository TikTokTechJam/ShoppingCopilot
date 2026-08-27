# Canonical attribute dictionary (Issue #8)

Issue #8 turns the canonical product facts from Issue #5 into one stable
attribute registry. It does not extract facts, invent aliases, parse user
utterances, or retrieve products.

The registry is written to `data/derived/dictionary/`:

- `canonical_values.json` stores the canonical ID, source attribute, exact
  Issue-5 value, normalized surface, and product frequency for every value.
- `normalized_lookup.json` stores attribute-scoped exact lookup lists. Lists
  preserve ambiguity when more than one canonical value has the same surface.
- `embedding_metadata.json` maps rows in the one shared semantic matrix back
  to canonical IDs.
- `attribute_embeddings.npy` stores normalized vectors for the configured
  semantic attributes when embeddings are generated.
- `manifest.json` records the source facts hash, normalization policy, model,
  dimensions, and counts.

The JSON files are deterministic registry metadata; they are not a database or
a separate dictionary of invented synonyms. The semantic lookup representation
is one in-memory matrix, not eight vector stores. Brand and size are excluded
from the default semantic set; price remains numeric.

Build exact lookup artifacts without a model:

```powershell
python -m scripts.build_attribute_dictionary --no-embeddings
python -m scripts.validate_attribute_dictionary
```

Build the shared semantic matrix with a local SentenceTransformers model:

```powershell
python -m scripts.build_attribute_dictionary `
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

The facts path defaults to
`data/annotations.jsonl` is accepted directly as the preferred V4 annotation
input; the flattened `data/derived/catalog_facts/catalog_facts.jsonl` from
Issue #5 is also supported. The embedding command requires the optional dependencies in
`requirements-embeddings.txt`; it is intentionally not run as part of normal
repository checks.

Runtime consumers should load `AttributeDictionary` and use this order:

```text
exact_match(raw_text, allowed_attribute)
        ↓ if exactly one result
canonical_id
        ↓ if unresolved
semantic_match(raw_text, allowed_attribute, min_similarity, min_margin)
        ↓ if confident
canonical_id + similarity
```

An exact ambiguity or a weak semantic result is unresolved. No LLM-generated
alias table is included in this MVP. Issue #7 can use the returned
`canonical_id`, `match_method`, and `similarity` as its canonicalization
contract; Issue #9 starts after that contract is produced.
