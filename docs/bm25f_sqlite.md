# SQLite BM25F lexical path

The lexical path has a readable `PythonBM25FIndex` reference/fallback and a
native SQLite FTS5 implementation. The native implementation uses the same
six catalog fields, weights, tokenizer, and explicit field-length BM25F
formula. It is loaded only when its artifact and compiled extension are
available; otherwise retrieval reports and uses the Python reference.

Build the Apple Silicon/macOS extension and catalog artifact from the repo
root:

```bash
python -m scripts.build_bm25f_sqlite_extension
python -m scripts.build_bm25f_sqlite_index \
  --catalog data/catalog.jsonl \
  --output-dir data/derived/bm25f_sqlite \
  --force
```

The runtime looks for `native/build/bm25f.dylib` and
`data/derived/bm25f_sqlite/bm25f.db` by default. Override these locations with
`SHOPPING_BM25F_EXTENSION` or the `ProductRetriever` constructor options.

For direct score/ranking parity and lexical timing diagnostics:

```bash
python -m scripts.compare_bm25f
```

For a controlled Python-reference benchmark arm, set:

```bash
SHOPPING_BM25F_BACKEND=python python -m evaluator.hard_evaluator \
  --catalog data/catalog.jsonl \
  --sessions data/derived/gptannotation/sessions.jsonl
```

Without that environment variable, the evaluator uses the native artifact
when it is valid. Missing or incompatible native artifacts fall back to the
Python reference and print the reason.
