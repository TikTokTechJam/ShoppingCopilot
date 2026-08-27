# Product embedding artifacts

Issue #12 adds a pure offline artifact boundary for whole-product semantic retrieval. It keeps `Agent.respond` and model invocation unchanged; the #13 retriever consumes the optional valid artifact when present while retaining compatibility fallback behavior.

## Inputs

The builder joins two JSONL sources by `parent_asin`:

- `data/catalog.jsonl`: source catalog rows. The current catalog uses `title` as a string and `description` as an array of strings.
- `data/derived/catalog_facts/catalog_facts.jsonl`: one canonical facts record per catalog product, using `category`, `brand`, `price`, `color`, `material`, `size`, `style`, `feature`, and `use_case`.

The Issue #5 annotation wrapper (`{"parent_asin": ..., "facts": {...}, ...}`) is accepted too. The builder still requires every catalog ASIN exactly once in both inputs, and writes rows in catalog order.

## Stable product text

Each product is represented by this fixed, non-JSON document shape:

```text
Title: <source title>
Category: <canonical values>
Brand: <canonical brand>
Material: <canonical values>
Color: <canonical values>
Style: <canonical values>
Features: <canonical feature values>
Use cases: <canonical values>
Description: <short normalized source description>
```

Canonical array values are de-duplicated and sorted. Source descriptions are whitespace-normalized and capped at 1,000 characters by default. Price, size, raw source `features`, annotation metadata, and arbitrary JSON are not embedded: price and size remain available to structured retrieval, while the embedding representation follows the semantic product fields in the Issue #12 solution.

## Build

```bash
python scripts/build_product_embeddings.py \
  --catalog data/catalog.jsonl \
  --facts data/derived/catalog_facts/catalog_facts.jsonl \
  --output-dir data/derived/product_embeddings \
  --model path/to/local/sentence-transformer
```

`--model` is loaded with `local_files_only=True`; it cannot download a model or call an API. `sentence-transformers` is optional and is imported only when `--model` is used. A local injected object/factory can be supplied with `--embedder package.module:object`. The default built-in hashing embedder needs only NumPy and is intended as a deterministic pipeline fallback, not as a semantic-quality benchmark model.

The builder accepts an embedder exposing `embed_documents(texts)`, `encode(texts, ...)`, `embed(texts)`, or a callable receiving a list of texts. All returned rows are converted to finite `float32`, L2-normalized, and rejected if they are zero vectors.

## Artifacts

```text
data/derived/product_embeddings/
├── product_embeddings.npy
├── product_embedding_metadata.json
└── manifest.json
```

The matrix is shaped `(product_count, embedding_dimension)` and has dtype `float32`. Metadata is a JSON array with the exact row-to-ASIN mapping. The manifest records the model identity, dimension, `l2` normalization policy, text/builder versions, source paths/versions, row order, and generation timestamp. Pass `--generated-at-utc` when a byte-stable manifest is required across rebuilds; vectors and row metadata are deterministic for the same inputs/model/config.

## Exact loading

```python
from product_embeddings import load_product_embedding_index

index = load_product_embedding_index(
    "data/derived/product_embeddings",
    expected_asins=["B07K34RX5J", "B07KCFS4VC"],
)
matches = index.search(query_embedding, top_k=10)
```

The loader validates matrix dtype/shape, manifest dimension/count, finite L2-normalized rows, unique ASINs, and contiguous metadata row numbers. Query vectors are normalized before exact matrix multiplication, so inner product is cosine similarity. Ties retain deterministic row order. No FAISS, vector database, network service, or Agent integration is required.
