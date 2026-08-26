# ShoppingCopilot Architecture

This file is the **main source of truth for the implementation architecture** of this repository.

All production code, data-processing scripts, retrieval components, Agent orchestration, and future architecture work should follow this document. If an implementation decision changes, update this file in the same PR/change that introduces the new architecture.

## Authority order

When sources disagree, use this order:

1. Official competition specification / evaluator contract.
2. `Architecture.md` for repository architecture.
3. Accepted ADR/design documents.
4. GitHub issues and comments as proposals/work tracking.
5. Existing code if it has drifted from the documented architecture.

If code conflicts with this file, either fix the code or intentionally update this file with the architectural change. Do not allow silent architecture drift.

---

## 1. Current MVP architecture

```text
                           OFFLINE / PRECOMPUTE

                        raw 50k catalog.jsonl
                                 │
                                 ▼
                     #5 AI catalog annotation
                     row -> prompt -> hosted LLM
                                 │
                                 ▼
                        annotations.jsonl
                                 │
                     deterministic normalization
                                 │
                                 ▼
                       canonical catalog facts
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
            #8 canonical registry       product embeddings
            + normalized lookup         for product retrieval
            + attribute embeddings


                              RUNTIME

                           user utterance
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
            #6 first-turn intent      #7 constraint extraction
            BUYING / BROWSING                │
                                              ▼
                                  deterministic normalization
                                              │
                                  exact canonical matching
                                              │
                                      mark matched spans
                                              │
                                  semantic fallback only for
                                  remaining meaningful phrases
                                              │
                                              ▼
                                   canonical constraints
                                              │
                           ┌──────────────────┴──────────────────┐
                           │                                     │
                           ▼                                     ▼
                        BUYING                                BROWSING
                  structured narrowing                    broad dense retrieval
                  + constraint scoring                    + preference boosting
                           │                                     │
                           └──────────────────┬──────────────────┘
                                              ▼
                                      candidate products
                                              │
                                     later ranking / DP
                                              │
                                              ▼
                                      Top-10 + optional ask
```

The key separation is:

- **#7/#8 understand the user and produce canonical constraints.**
- **#9 begins after canonicalization and retrieves actual products.**
- The MVP does **not** add a second BM25/FTS lexical product-retrieval branch unless later evidence shows it is needed.

---

## 2. Runtime / deployment constraints

The challenge catalog is frozen and small enough to keep runtime structures in memory.

Current MVP decision:

- no Postgres for catalog retrieval;
- no Pinecone, Milvus, Weaviate, or external industrial vector DB;
- no network database dependency for the frozen 50k catalog;
- precomputed artifacts may live on disk in the repository/artifact bundle;
- load required data and indexes into process memory at Agent startup.

Typical runtime structures:

```text
product_by_asin
canonical facts
attribute value registry
normalized-value lookup maps
attribute embedding matrix
product embedding matrix
structured inverted indexes / sets
```

A lightweight in-process library such as NumPy/FAISS may be used for vector similarity.

---

## 3. Issue #5: canonical product facts

### 3.1 Source catalog

`data/catalog.jsonl` remains immutable.

### 3.2 Annotation flow

Each raw catalog row is converted into a **prompt string** and sent to the hosted annotation LLM.

```text
catalog row
    ↓
build versioned annotation prompt
    ↓
hosted LLM endpoint
    ↓
strict structured JSON response
    ↓
validate
    ↓
append annotation artifact
```

The model is not the owner of immutable identifiers/numeric source fields. `parent_asin` and authoritative source values such as price should be copied from the catalog by code where possible.

### 3.3 Annotation storage

Recommended layout:

```text
data/
├── catalog.jsonl
└── derived/
    ├── annotations/
    │   └── v1/
    │       ├── annotations.jsonl
    │       ├── failures.jsonl
    │       └── manifest.json
    └── catalog_facts/
        ├── catalog_facts.jsonl
        └── catalog_facts.parquet   # optional derived representation
```

`annotations.jsonl` is append/resume-safe and acts as the annotation audit/source-of-truth layer. `catalog_facts` is the clean Agent-facing layer.

### 3.4 Canonical facts schema

One record per product, conceptually:

```json
{
  "parent_asin": "B123",
  "category": ["boots", "hiking_boots"],
  "brand": "abc_shoes",
  "price": 79.99,
  "color": ["black"],
  "material": ["leather"],
  "size": [],
  "style": [],
  "feature": ["waterproof", "lightweight", "rubber_sole"],
  "use_case": ["hiking", "outdoor"]
}
```

Accuracy is more important than filling every field. Unsupported semantic claims must not be invented.

---

## 4. Issue #8: canonical attribute registry and matching artifacts

Issue #8 derives the global attribute vocabulary from the canonical facts produced by #5.

### 4.1 One canonical registry

Do not build unrelated dictionaries that can drift apart. Build one canonical registry and multiple lookup representations over it.

Each canonical value should have a stable ID, for example:

```text
category:hiking_boots
color:black
material:leather
feature:waterproof
use_case:hiking
```

Conceptual record:

```json
{
  "canonical_id": "feature:waterproof",
  "attribute": "feature",
  "value": "waterproof",
  "normalized": "waterproof",
  "count": 615
}
```

The registry covers:

```text
category
brand
color
material
size
style
feature
use_case
```

`price` remains numeric and is not a categorical dictionary.

### 4.2 Deterministic normalized lookup

For the MVP there is **no LLM-generated synonym/alias dictionary**.

Use deterministic normalization such as:

- lowercase;
- trim leading/trailing whitespace;
- collapse repeated whitespace;
- normalize `_` / `-` separators when appropriate;
- remove safe punctuation where appropriate;
- conservative singular/plural handling only if deterministic and unambiguous.

Examples:

```text
"Hiking Boots"  -> "hiking boots"
"hiking_boots"  -> "hiking boots"
"hiking-boots"  -> "hiking boots"
" Waterproof "  -> "waterproof"
```

Exact/normalized matching is preferred over semantic matching because it is deterministic.

### 4.3 Attribute embeddings

Generate embeddings for canonical values that benefit from semantic matching. Use **one shared embedding matrix with metadata**, not eight separate vector databases.

Suggested artifacts:

```text
data/derived/dictionary/
├── canonical_values.json
├── embedding_metadata.json
├── attribute_embeddings.npy
└── manifest.json
```

`embedding_metadata.json` maps each vector row to its canonical ID and attribute.

Conceptually:

```json
[
  {"row": 0, "canonical_id": "category:hiking_boots", "attribute": "category", "value": "hiking_boots"},
  {"row": 1, "canonical_id": "feature:waterproof", "attribute": "feature", "value": "waterproof"},
  {"row": 2, "canonical_id": "use_case:hiking", "attribute": "use_case", "value": "hiking"}
]
```

Not every attribute requires semantic lookup equally:

```text
category    semantic useful
brand       primarily exact/normalized
color       exact first; semantic optional
material    exact first; semantic optional
size        structured/exact; semantic unnecessary
style       semantic useful
feature     semantic useful
use_case    semantic useful
price       numeric parser only
```

Semantic results must respect a similarity/confidence threshold. Low-confidence input remains unresolved rather than being forced to an incorrect canonical value.

---

## 5. Issue #7 + #8: utterance -> canonical constraints

This is the agreed MVP canonicalization flow.

```text
user utterance
      ↓
1. parse obvious structured values
   price / size / numeric bounds
      ↓
2. normalize text
      ↓
3. scan for exact/normalized canonical dictionary matches
      ↓
4. mark/remove matched spans from semantic fallback input
      ↓
5. identify remaining meaningful unmatched phrase(s)
      ↓
6. embed unmatched phrase(s)
      ↓
7. search canonical attribute embeddings
      ↓
8. accept only sufficiently confident semantic matches
      ↓
structured canonical constraints
```

Example:

```text
"I want black waterproof shoes for long mountain walks under $100"
```

Deterministic matches:

```text
black       -> color:black
waterproof  -> feature:waterproof
shoes       -> category:shoes
under $100  -> price_max=100
```

Remaining meaningful phrase:

```text
"long mountain walks"
```

Semantic fallback may resolve:

```text
"long mountain walks" -> use_case:hiking
```

Final canonical constraints may become:

```json
{
  "category": ["shoes"],
  "color": ["black"],
  "feature": ["waterproof"],
  "use_case": ["hiking"],
  "price_max": 100
}
```

### 5.1 Matching priority

```text
structured parsing > exact/normalized dictionary match > semantic fallback > unresolved
```

Do not run semantic matching for values already resolved exactly.

### 5.2 Future enhancement

An explicit synonym/alias layer such as:

```text
"dark blue"      -> color:navy
"rain proof"     -> feature:waterproof
"trainers"       -> category:sneakers
```

may be added later if benchmark evidence justifies it. It is **not part of the current MVP architecture**.

---

## 6. Issue #6: Buying vs Browsing routing

For the MVP, the Buying/Browsing router runs on the **first user utterance only**.

```text
turn 1
user message -> BUYING / BROWSING
```

Store this mode in session state. Later clarification replies normally update constraints/preferences rather than rerunning the router every turn.

Intent Override is a separate state transition and should not be conflated with the initial router.

---

## 7. Issue #9: product retrieval architecture

Issue #9 starts with the canonicalized state produced by #7/#8.

The MVP does not need an additional BM25/FTS branch merely to rediscover text already converted into structured canonical constraints.

### 7.1 Shared inputs

```text
session mode
canonical constraints/preferences
full current user/session text for semantic context
canonical product facts
product embeddings
```

### 7.2 Buying retrieval

Buying intent is high precision.

```text
canonical constraints
        ↓
structured candidate narrowing
        ↓
controlled fallback if candidate set is too small
        ↓
dense/product relevance scoring where useful
        ↓
ranked candidate pool
```

Examples of structured runtime indexes:

```text
category["hiking_boots"] -> ASIN set
color["black"]           -> ASIN set
feature["waterproof"]    -> ASIN set
brand["nike"]            -> ASIN set
material["leather"]      -> ASIN set
```

Price is handled numerically.

Explicit requirements should dominate ranking. Do not let dense similarity rescue products that clearly violate hard user requirements.

If strict intersections return too few products, relax only lower-confidence/soft constraints before explicit hard constraints.

### 7.3 Browsing retrieval

Browsing intent prioritizes recall and exploration.

```text
full user/session context
        ↓
dense search over product embeddings
        ↓
broad candidate pool
        ↓
boost candidates matching known canonical preferences
        ↓
ranked candidate pool
```

Do not aggressively hard-filter vague browsing preferences early.

### 7.4 Product embeddings

Precompute one embedding per product from a stable semantic representation containing the useful shopping content, for example:

```text
Title: ...
Category: ...
Brand: ...
Material: ...
Features: ...
Use cases: ...
Description: ...
```

Store artifacts such as:

```text
product_embeddings.npy
product_embedding_metadata.json
```

Load them into RAM and use an in-process exact or lightweight vector index. With 50k products, exact normalized inner-product/cosine search is acceptable as an MVP.

### 7.5 Candidate contract

Retrieval should return a shared candidate representation so downstream ranking/DP does not care which track produced it.

Conceptually:

```json
{
  "parent_asin": "B123",
  "retrieval_mode": "BUYING",
  "dense_score": 0.82,
  "constraint_score": 1.0,
  "matched_constraints": ["category:hiking_boots", "color:black", "feature:waterproof"],
  "violated_constraints": []
}
```

The exact schema can evolve, but candidate provenance and constraint match information should remain available downstream.

### 7.6 MVP exclusions

Not required initially:

- external vector DB;
- Postgres-backed retrieval;
- LLM reranking over every candidate;
- a separate BM25/FTS product-retrieval branch;
- complex ANN infrastructure for 50k products.

These are benchmark-driven future enhancements, not default architecture.

---

## 8. Recommendation / clarification behavior

The Agent contract allows returning recommendations and an `ask_attribute` in the same turn.

Current design principle:

> Return the current best Top-10 on every scoreable turn and optionally ask one useful clarification question simultaneously.

This preserves the chance of an early target hit while still gathering information for future turns.

The clarification/DP subsystem will consume the common candidate representation and canonical product facts. Detailed DP design remains a later architecture decision.

---

## 9. Data flow summary

```text
catalog.jsonl
    ↓
#5 annotation prompt pipeline
    ↓
annotations.jsonl
    ↓
catalog_facts.jsonl / parquet
    ↓
#8 canonical registry + attribute embeddings
    │
    ├───────────────┐
    │               │
    ▼               ▼
#7 user        product embedding
canonicalizer  preprocessing
    │               │
    └───────┬───────┘
            ▼
       #9 retrieval
            ▼
      candidate pool
            ▼
      ranking / DP
            ▼
 Top-10 + optional ask
```

---

## 10. Architecture change policy

Any change that modifies one of the following must update this file in the same PR/change:

- data schemas or canonical fields;
- annotation pipeline responsibilities;
- dictionary/canonicalization strategy;
- embedding representation or storage;
- Buying/Browsing routing behavior;
- retrieval stages;
- database/service dependencies;
- session-state ownership;
- ranking/DP component boundaries;
- Agent contract assumptions.

Implementation tickets may refine internal details without changing this file when they stay within the boundaries above.

See `AGENTS.md` for the repository policy that automated coding agents must follow.