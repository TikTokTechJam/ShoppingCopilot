# ShoppingCopilot

## **[→ READ THE FULL ARCHITECTURE](Architecture.md)**

**For the complete system design, Buying/Browsing retrieval flows, dialogue-state handling, preference overrides, and clarification strategy, see [Architecture.md](Architecture.md).**


ShoppingCopilot is a conversational product-search agent designed to help users move from broad product exploration to more precise purchase decisions through multi-turn conversation.

The system maintains shopping preferences across turns, handles preference changes and overrides, supports both exploratory and constraint-driven search, and asks clarification questions when the user's request is still too broad.

All language and embedding models used by the project are self-hosted. A self-hosted **Qwen3-27B** model is used for offline product annotation and runtime turn interpretation, while **BGE-small-en-v1.5**, **Qwen3-Embedding-0.6B**, and **SQLite FTS5 BM25** support semantic and lexical product retrieval.

Approximately **50k products** are annotated into structured attributes such as category, brand, color, material, features, use cases, style, and price before retrieval.

## Benchmark results

Current evaluation results:

- **Hit@10:** 0.915
- **Technical score:** 0.7441
- **Average latency:** ~4 seconds per conversation turn

## Key features

- Multi-turn conversational product search
- Preference accumulation and override handling
- Adaptive Buying and Browsing behavior
- Hybrid lexical and semantic retrieval
- Candidate-aware clarification
- Self-hosted LLM and embedding models
- Local evaluation and debugging tools

## Repository layout

```text
starter/                         Agent, session, routing, clarification, retrieval
starter/browsing/                Qwen query compiler, product cards, dense retrieval
dictionary/                      Canonical values and BGE semantic matching
product_embeddings/             Product-vector artifact loaders/builders
annotation/                      Annotation clients, schemas, and runners
evaluator/                       Local evaluator, Manual400 evaluator, debug UI
scripts/                         Annotation, dictionary, and embedding commands
data/catalog.jsonl               Frozen catalog input (provided locally)
data/derived/annotations/v5/     V5 facts and canonical artifacts
data/derived/product_embeddings_v5/
                                 Qwen V5 product-card artifact
models/                          Local model weights / model configuration
```

## Setup

Use Python 3.11 or a compatible environment.

Install the embedding and Hugging Face dependencies:

```bash
python -m pip install -r requirements-embeddings.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

Configure the required model paths and the self-hosted Qwen3-27B endpoint in `.env`.

The Qwen3-27B endpoint is self-hosted but uses API-key authentication. Credentials must remain local and must not be committed to Git.

## Product annotations and embeddings

The active V5 annotation source is normally:

```text
data/derived/annotations/v5/annotations.jsonl
```

Build or rebuild the canonical dictionary and BGE attribute embeddings when the V5 annotations change:

```bash
python -m scripts.build_attribute_dictionary \
  --input data/derived/annotations/v5/annotations.jsonl \
  --input-format v5 \
  --output-dir data/derived/annotations/v5/dictionary \
  --no-embeddings

python -m scripts.build_v5_attribute_embeddings \
  --dictionary-dir data/derived/annotations/v5/dictionary \
  --output-dir data/derived/annotations/v5/dictionary/attribute_embeddings \
  --model "$PWD/model/bge-small-en-v1.5"
```

Set up the local Qwen embedding model:

```bash
python -m scripts.setup_qwen_product_model \
  --model-dir "$PWD/model/Qwen3-Embedding-0.6B"
```

Build the V5 product embeddings:

```bash
python -m scripts.build_v5_product_embeddings \
  --catalog data/catalog.jsonl \
  --annotations data/derived/annotations/v5/annotations.jsonl \
  --output-dir data/derived/product_embeddings_v5 \
  --model "$PWD/model/Qwen3-Embedding-0.6B" \
  --progress
```

## Run the evaluators

Local public-set evaluation:

```bash
python -m evaluator.local_evaluator
```

Manual400 evaluation:

```bash
python -m evaluator.hard_evaluator
```

Useful Manual400 options:

```bash
python -m evaluator.hard_evaluator --no-progress
python -m evaluator.hard_evaluator --override-only
python -m evaluator.hard_evaluator --debug --seed 1 --debug-sessions 10
```

## Run the debug UI

Start the default Manual400 viewer:

```bash
python -m evaluator.debug_web
```

Open:

```text
http://127.0.0.1:8765
```

Other modes:

```bash
python -m evaluator.debug_web --evaluator local \
  --dataset data/public_set.jsonl

python -m evaluator.debug_web --interactive

python -m evaluator.debug_web --browsing-retrieval qwen_dense
```

The debug UI exposes session state, active constraints, clarification signals, target positions, BM25 results, dense retrieval results, RRF, and MMR diagnostics where available.

## Data and Git policy

The catalog, derived annotations, embedding matrices, model weights, `.env`, credentials, and evaluation outputs are local artifacts and are intentionally excluded from Git where appropriate.

A fresh clone therefore needs the required catalog data, generated artifacts, model weights or model-server access, and local credentials before the complete runtime is available.