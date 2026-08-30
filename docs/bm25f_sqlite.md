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

The diagnostic compares the current native unigram baseline with the
overlapping contiguous 1-to-n-gram experiment. The baseline remains the
available as an explicit compatibility mode:

```bash
python -m scripts.compare_bm25f --mode baseline
python -m scripts.compare_bm25f --mode ngrams
```

The evaluator/retriever runtime now uses 1-to-n-grams by default. Use this
environment variable only when you want the old unigram baseline:

```bash
SHOPPING_BM25F_NGRAMS=0 python -m evaluator.hard_evaluator
```

The comparison script also accepts `--mode baseline`, `--mode ngrams`, or
`--mode both`. In n-gram mode, a cleaned query with `n` tokens produces every
contiguous window for `k=1..n`; all phrases at one level are OR alternatives,
and every matching phrase contribution is summed. The final score is
`S1 + S2 + ... + Sn`, with level weights of `1.0`. Phrase matches require
ordered adjacency within one FTS5 column. The native scorer continues to use
the existing BM25F weights, `k1`, `b`, IDF, field lengths, and candidate
semantics.

For a smaller diagnostic run with per-product `S_k` breakdowns:

```bash
python -m scripts.compare_bm25f --mode both --repeats 5 --breakdown-top 3 \
  "waterproof hiking shoes" "black leather hiking shoes"
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
