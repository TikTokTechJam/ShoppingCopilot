# Retrieval architecture

This document describes the active retrieval implementation. `Agent` owns the
conversation and clarification policy; `ProductRetriever` owns the shared
catalog indexes and mode-specific ranking.

## Runtime flow

```text
user turn
  -> intent routing and deterministic constraint extraction
  -> session state (structured values, BGE evidence, active-goal query text)
  -> budget eligibility and recommendation exclusions
  ->
     BUYING: field-aware BM25 with accepted canonical expansions
     BROWSING: Qwen product-card Top100 + slot BM25 Top100
  ->
     BUYING: BM25 score
     BROWSING: reciprocal-rank fusion, then product-card MMR
  -> Top10
```

The same `Agent` and `ProductRetriever` are used by the evaluators and the
debug web page. The debug page only exposes the intermediate signals; it does
not create a second ranking path.

## Two semantic roles

### BGE canonical expansion

The local `BAAI/bge-small-en-v1.5` registry searches short canonical attribute
phrases. It turns unresolved user spans into accepted canonical values and
stores their cosine confidence in session evidence. Matching is attribute
scoped and uses the existing stopword-filtered one-, two-, and three-gram
matcher. Brand remains exact-only.

These canonical matches are sparse posting-list evidence. They are not
product-vector dense retrieval, and they are not a ranking term. Their role is
to supply the bounded expansions the BM25 concept groups are compiled from;
the resulting product score is retained for diagnostics only.

### V5 product-card vectors

Browsing uses a separate product-level artifact built from the title and V5
facts (`category`, `brand`, `color`, `material`, `style`, `feature`, and
`use_case`):

```text
V5 annotations + catalog title
  -> deterministic product card
  -> Qwen/Qwen3-Embedding-0.6B
  -> one L2-normalized vector per catalog product
```

The artifact is loaded from `data/derived/product_embeddings_v5`. Its query
encoder must be the same local model named in the manifest and is configured
with `SHOPPING_PRODUCT_EMBEDDING_MODEL`. Runtime loading is local-only; a
missing or incompatible model disables product-vector search without falling
back to a hash encoder.

The query compiler serializes only the active slots: `brand` from the exact
structured state, and `category`, `color`, `material`, `style`, `feature`, and
`use_case` from semantic state. It excludes transcript history, stale goal
segments, price, and size. The local Qwen adapter formats the query as:

```text
Instruct: Retrieve products that best match the shopper's product type, intended use, desired features, and preferences.
Query: category: jumpsuit
feature: lightweight
use_case: cosplay
```

The instruction is query-side only. Product cards remain unprefixed documents;
the document embedding build path does not use this instruction.

## Mode-specific ranking

Buying applies budget eligibility first. It then ranks all eligible,
non-excluded products with:

```text
1.00 * expanded_bm25_score
+ rating tie-break
```

Buying is field-aware BM25 alone. Every constraint reaches the ranking through
one route: the per-slot concept groups, which carry each active slot's value
and its bounded BGE expansions. Price is applied before scoring, as numeric
eligibility. Three terms that scored the same constraints a second time have
been removed:

- `0.20 * canonical_expansion_score`, which scored the canonical postings
  again after they had already been compiled into the concept groups;
- the per-field cosine that scaled `structured_score`; and
- `1.00 * structured_score` itself, which scored by posting list what the
  concept groups already score lexically.

`structured_score` and the canonical score are still computed and surfaced as
`Candidate.constraint_score` and `Candidate.semantic_score` for diagnostics,
and the matched/violated constraint labels still come from the structured
match. Neither is a ranking term.

Note that `size` has no posting-list route left and is not a BM25 query field,
so it currently contributes nothing to Buying rank. Price still filters.

For Buying, the expanded BM25 signal contains only active per-slot values plus
bounded BGE canonical/user-surface expansions. The raw current-goal query is
not a Buying scoring group. The BM25 score is normalized per query group before
the groups are averaged, so a slot cannot dominate merely by having more
synonym text.

Browsing searches the full eligible catalog through both independent signals:

```text
dense_top100 = Qwen product-card cosine ranking
bm25_top100  = field-routed slot BM25 ranking
rrf(p) = 1 / (60 + dense_rank) + 1 / (60 + bm25_rank)
```

The union is sorted by RRF and truncated to a fused Top-50 pool. When the
V5 product index and query encoder are available, Browsing then applies MMR
using product-card cosine similarity (`lambda = 0.80`) to reduce near-duplicate
results before the requested recommendation limit. Buying does not use MMR. If
product vectors are unavailable, Browsing retains its field-routed slot BM25
RRF arm and skips product-vector MMR.

`Candidate.score` is the score used by the final ordering, including the
rating tie-break. `Candidate.dense_score` means the V5 product-vector score;
`Candidate.semantic_score` means BGE canonical posting evidence;
`Candidate.fusion_score` is the Browsing RRF score; and `Candidate.mmr_score`
is the diversity score when Browsing MMR is active. The debug evaluator shows
the FTS expression and target rank for each active BM25 slot query; raw
current-goal BM25 is not part of the runtime flow.

## State and hard filters

`SessionState.retrieval_query_text` contains only the active goal segment.
Preference overrides preserve independent constraints but clear the replaced
lexical segment, so obsolete wording is not sent to BM25. They also restart
recommendation history, allowing products shown under the previous preference
to re-enter the new ranking context. The visible transcript remains available
for debugging.

Known prices outside an active budget are ineligible, and unknown prices are
also ineligible when a budget is active. Previously shown/rejected
recommendations remain excluded before ranking within the active preference
context. An override starts a fresh recommendation context; no semantic or
lexical signal can bypass either rule within that context.

## Artifacts and commands

The BGE canonical artifacts are under:

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

The V5 product-card artifact contains `product_embeddings.npy`,
`product_embedding_metadata.json`, `product_cards.jsonl`, and `manifest.json`
under `data/derived/product_embeddings_v5/`.

The standalone `style.jsonl` annotations are included in the aggregated V5
facts before canonical dictionary generation. The per-attribute BGE build then
writes the style vectors to `attribute_embeddings/style_embeddings.npy`.
Rebuild the dictionary first, then build only the style matrix (existing
compatible attribute matrices are preserved):

```bash
python -m scripts.build_attribute_dictionary \
  --input data/derived/annotations/v5/annotations.jsonl \
  --input-format v5 \
  --output-dir data/derived/annotations/v5/dictionary \
  --no-embeddings
python -m scripts.build_v5_attribute_embeddings \
  --dictionary-dir data/derived/annotations/v5/dictionary \
  --output-dir data/derived/annotations/v5/dictionary/attribute_embeddings \
  --model models/bge-small-en-v1.5 \
  --attributes style
```

Build the V5 product-card artifact after placing a compatible local Qwen model
on disk:

```bash
python -m scripts.setup_qwen_product_model
python -m scripts.build_v5_product_embeddings \
  --catalog data/catalog.jsonl \
  --annotations data/derived/annotations/v5/annotations.jsonl \
  --output-dir data/derived/product_embeddings_v5 \
  --model models/qwen3-embedding-0.6b \
  --progress
```

Configure the same model for runtime queries:

```powershell
$env:SHOPPING_PRODUCT_EMBEDDING_MODEL="models/qwen3-embedding-0.6b"
python -m evaluator.hard_evaluator
```

The product artifact is generated offline and is ignored by Git. The existing
Jina four-view artifact is not part of this V5 product-card path and is not
auto-discovered. The older direct-catalog Layer 2 interface remains available
only to callers that explicitly pass `layer2_artifact_dir` for compatibility.
