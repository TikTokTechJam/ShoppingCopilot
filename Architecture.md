# ShoppingCopilot Architecture

This document is the concise source of truth for our target architecture.

## 1. End-to-end flow

```text
USER TURN
    ↓
LLM TURN INTERPRETER
current-turn slot delta + override
    ↓
DETERMINISTIC VALIDATION
price / budget + exact brand
    ↓
DEPENDENCY-AWARE SESSION STATE
active constraints + provenance + profile
    ↓
DETERMINISTIC INTENT ROUTER
BUYING ↔ BROWSING
    ↓
┌──────────────────────────┴──────────────────────────┐
↓                                                     ↓
BUYING                                             BROWSING
precision-oriented                                discovery-oriented
↓                                                     ↓
price + exact brand                              active semantic state
↓                                                     ↓
BGE canonical expansion                    Qwen3 dense retrieval
↓                                                     +
slot / concept groups                      raw active-goal BM25
↓                                                     ↓
field-routed BM25                                 RRF
└──────────────────────────┬──────────────────────────┘
                           ↓
                  BROAD CANDIDATE POOL
                           ↓
                  CANDIDATE POOL ANALYZER
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
        OVER-GENERAL                  READY
              ↓                         ↓
      clarification              final mode ranking
      + cheap results            Browsing: MMR
                                        ↓
                                       Top-K
```

We keep Buying and Browsing as different retrieval paths:

- **Buying:** explicit constraints and lexical precision.
- **Browsing:** semantic recall, discovery, and diversity.

## 2. Offline retrieval assets

```text
Amazon catalog (~50k)
        ↓
V5 semantic annotation
        ↓
┌─────────────────┬──────────────────┬─────────────────┐
↓                 ↓                  ↓
Canonical facts   Product cards      Raw product text
↓                 ↓                  ↓
BGE attribute     Qwen3 product      SQLite FTS5
matrices           matrix             BM25 index
```

### 2.1 V5 semantic facts

We normalize:

```text
category
brand
color
material
style
feature
use_case
price
```

Broad and specific category labels may coexist for discovery.

### 2.2 BGE canonical expansion

We use:

```text
BAAI/bge-small-en-v1.5
```

We use BGE to match extracted semantic slots against canonical values for:

```text
category
color
material
style
feature
use_case
```

We use BGE for Buying query expansion, not direct product-level ranking. [R1]

### 2.3 SQLite FTS5 BM25

We use field-weighted SQLite FTS5 BM25 over:

```text
title
categories
features
details
store
description
```

We give stronger weights to high-value fields such as title and category. [R2]

BM25 serves two roles:

```text
Buying   → BGE-expanded slot/concept groups
Browsing → raw/current-goal lexical retrieval
```

### 2.4 Qwen dense product matrix

We serialize V5 product facts into compact product cards and embed them with:

```text
Qwen/Qwen3-Embedding-0.6B
dimension: 1024
normalization: L2
```

Browsing queries use a query-only instruction:

```text
Instruct: Retrieve products that best match the shopper's product type, intended use, desired features, and preferences.
Query: <active semantic state>
```

We keep product documents unprefixed. With normalized vectors, runtime cosine similarity becomes:

```text
product_matrix @ query_vector
```

This follows the intent-aware dense-retrieval direction used in recent e-commerce retrieval work. [R3] [R4]

## 3. Turn interpretation and state

### 3.1 LLM turn interpreter

We use a small local/self-hosted LLM only to interpret the current turn. [R5] [R6] [R7]

```json
{
  "updates": {
    "category": [],
    "brand": [],
    "color": [],
    "material": [],
    "feature": [],
    "use_case": [],
    "style": [],
    "price_min": null,
    "price_max": null
  },
  "override": {
    "type": "none | preference_override | full_goal_override",
    "fields": []
  }
}
```

The LLM does **not** choose Buying/Browsing mode.

We validate price/budget and exact brand deterministically before updating state.

### 3.2 Dependency-aware session state

We apply each turn delta to the existing state instead of regenerating the full state. [R8]

For each active constraint we retain:

```text
attribute
value
source: explicit | inferred
parent_constraint: optional
```

On overrides, we preserve independent explicit constraints and remove only invalidated dependent state. [R9]

Retrieval always uses the **active state**, not stale full conversation history.

### 3.3 Intent routing

```text
LLM delta
    ↓
SessionManager
    ↓
updated active state
    ↓
deterministic router
    ↓
BUYING or BROWSING
```

We route using specificity, explicit constraints, brand/price signals, previous mode, clarification context, and browsing-language cues.

We use hysteresis to avoid unstable mode switching.

## 4. Buying retrieval

```text
Active Buying State
        ↓
price eligibility + exact brand evidence
        ↓
semantic constraints
        ↓
BGE canonical expansion
        ↓
slot / concept groups
        ↓
field-routed weighted BM25
        ↓
constraint coverage + grouped evidence
        ↓
Buying candidates
```

### 4.1 Structured logic

```text
price_min / price_max → numeric eligibility
brand                 → exact evidence
```

For an active budget:

```text
satisfying price → eligible
violating price  → excluded
null price       → excluded
```

### 4.2 Concept-group retrieval

We group BGE expansions by active slot instead of treating every synonym as a separate requirement. [R10] [R11]

```text
C_category
C_color
C_material
C_style
C_feature
C_use_case
```

Each group represents one user constraint with multiple lexical alternatives.

We do not generate Cartesian synonym combinations.

### 4.3 Buying ranking

We combine:

```text
price eligibility
+ exact brand evidence
+ semantic constraint coverage
+ grouped BM25 evidence
+ soft profile tie-break
```

BGE only improves the BM25 query. We do not add a second direct BGE product score.

## 5. Browsing retrieval

```text
Active Browsing State
        │
        ├────────→ raw/current-goal BM25 → Sparse Top-N
        │
        ↓
semantic query serializer
        ↓
Qwen3-Embedding-0.6B
        ↓
Dense Top-N
        │
        └──────────────┬──────────────┘
                       ↓
                      RRF
                       ↓
                fused candidates
```

### 5.1 Dense query

We build the dense query from active semantic state:

```text
category: running shoes
color: black
feature: slip resistant, lightweight
use_case: hiking, wet weather
```

We exclude stale or overridden preferences.

Price stays outside the dense query. Brand may appear as context, while exact brand evidence is handled separately.

### 5.2 Dense retrieval

We initially use exact search over the ~50k normalized Qwen product vectors.

We retrieve a broad candidate set such as:

```text
Dense Top-100
```

### 5.3 BM25 complement

We run BM25 over the **active current-goal text**.

Text invalidated by overrides is excluded.

This route preserves exact lexical evidence such as brand names, model names, and rare product terms.

### 5.4 RRF fusion

We combine dense and BM25 ranked lists using **Reciprocal Rank Fusion (RRF)** rather than adding incompatible raw scores. [R12] [R13]

```text
RRF(product)
= contribution from dense rank
+ contribution from BM25 rank
```

### 5.5 MMR diversity

For ready Browsing pools, we apply **Maximal Marginal Relevance (MMR)** to reduce redundant recommendations. [R14]

We reuse Qwen product vectors for product-to-product redundancy similarity.

## 6. Candidate-aware clarification

We analyze a broad candidate pool before expensive final ranking. [R15] [R16]

For unresolved attributes, we consider:

```text
coverage
expected candidate reduction
value diversity
answerability
remaining-turn value
```

```text
broad candidates
        ↓
Candidate Pool Analyzer
        ↓
┌───────────────┴───────────────┐
↓                               ↓
OVER-GENERAL                   READY
↓                               ↓
strategic clarification      final mode ranking
+ cheap recommendations           ↓
                                  Top-K
```

We skip attributes that are already resolved, previously asked, marked as no preference, or poorly represented in the candidate pool.

## 7. Personalization

We maintain:

```text
short-term active state
→ current goal and explicit constraints

soft long-term profile
→ stable repeated preferences used as weak priors
```

Profile updates come from repeated successful interactions or explicit feedback.

Explicit current requirements always override profile priors.

## 8. Evaluation

Core metrics:

```text
Hit@10
MRR
MTTC
latency
```

Browsing retrieval additionally tracks:

```text
Hit@50
Hit@100
dense-only recovery@100
```

`dense-only recovery@100` measures relevant products recovered by dense retrieval when BM25 misses them in its Top-100.

## 9. Architectural invariants

1. We use distinct retrieval strategies for **Buying** and **Browsing**, selected from the active dialogue state.

2. We derive retrieval only from the **active state**; stale or overridden constraints must not affect search.

3. For **Buying**, we prioritize precision using structured price/brand evidence and BGE-expanded grouped BM25. We do not use BGE as a direct product-ranking signal.

4. For **Browsing**, we prioritize recall and diversity using Qwen dense retrieval plus active-goal BM25, fused with RRF and diversified with MMR.

5. We preserve independent explicit constraints across preference overrides and invalidate only dependent state.

6. We use candidate-aware clarification when the candidate space remains over-general, avoiding unnecessary expensive ranking.

7. We use long-term profile information only as a soft prior; explicit current requirements always take precedence.

## References

[R1] Xiao et al., *C-Pack: Packaged Resources To Advance General Chinese Embedding*, 2023. https://arxiv.org/abs/2309.07597

[R2] Robertson, Zaragoza & Taylor, *Simple BM25 Extension to Multiple Weighted Fields*, CIKM 2004.

[R3] Qwen3 Embedding, 2025. https://arxiv.org/abs/2506.05176

[R4] *INSPIRE: Intent-aware Neural Sponsored Product Retrieval for E-commerce*, 2026. https://arxiv.org/abs/2606.23889

[R5] Lee et al., *Dialogue State Tracking with a Language Model using Schema-Driven Prompting*, EMNLP 2021.

[R6] Li et al., *Large Language Models as Zero-shot Dialogue State Tracker through Function Calling*, ACL 2024.

[R7] Gupta et al., *Show, Don't Tell: Demonstrations Outperform Descriptions for Schema-Guided Task-Oriented Dialogue*, NAACL 2022.

[R8] Kim et al., *Efficient Dialogue State Tracking by Selectively Overwriting Memory*, ACL 2020.

[R9] Doyle, *A Truth Maintenance System*, Artificial Intelligence, 1979.

[R10] Crimp & Trotman, *Automatic Term Reweighting for Query Expansion*, 2017.

[R11] Dai et al., *End-to-End Query Term Weighting (TW-BERT)*, 2023.

[R12] Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR 2009.

[R13] Lee et al., *On Complementarity Objectives for Hybrid Retrieval*, ACL 2023.

[R14] Carbonell & Goldstein, *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*, SIGIR 1998.

[R15] *ProductAgent: Benchmarking Conversational Product Search Agent with Asking Clarification Questions*, EMNLP Industry 2025.

[R16] *Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision-Tree Search*, ACL 2025.
