# ShoppingCopilot Architecture

This file is the **single source of truth for the target architecture**.

Implementation work should conform to this document. If existing runtime code, old comments, legacy artifacts, or previous experiments conflict with this file, treat those as stale unless this document explicitly marks them as compatibility behavior.

The architecture is intentionally split into explicit components so each stage can be benchmarked independently.

## 1. End-to-end architecture

```text
                                USER TURN
                                    ↓
                         LLM TURN INTERPRETER
                    intent + slots + override delta
                                    ↓
                     DETERMINISTIC VALIDATION
                     price / budget + exact brand
                                    ↓
                 DEPENDENCY-AWARE SESSION STATE
               active constraints + provenance + profile
                                    ↓
                    ADAPTIVE INTENT ORCHESTRATOR
                         BUYING ↔ BROWSING
                                    ↓
              ┌─────────────────────┴─────────────────────┐
              ↓                                           ↓
           BUYING                                      BROWSING
      precision-oriented                           discovery-oriented
              ↓                                           ↓
               price eligibility                            active semantic intent
              ↓                                           ↓
     BGE canonical expansion              Qwen3-Embedding-0.6B
              ↓                                   query embedding
      slot / concept groups                            ↓
              ↓                              V5 product-card dense index
      grouped weighted BM25                           +
              ↓                                 raw BM25 complement
      Buying candidate rank                           ↓
              │                                      RRF
              │                                       ↓
              │                                      MMR
              └─────────────────────┬─────────────────┘
                                    ↓
                         CANDIDATE POOL ANALYZER
              coverage + expected reduction + answerability
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
              OVER-GENERAL                        READY
                    ↓                               ↓
          strategic clarification                 Top-K
                    │                               │
                    └───────────────┬───────────────┘
                                    ↓
                               NEXT TURN
```

The key architectural rule is that **Buying and Browsing are genuinely different retrieval paths**, not the same retriever with different labels.

- **Buying** prioritizes explicit constraints and semantically expanded lexical precision.
- **Browsing** prioritizes semantic recall, cross-category discovery, and diversity.

---

## 2. Offline preprocessing and indexes

The online Agent should not repeatedly reconstruct expensive catalog representations. Catalog understanding and dense product indexing are prepared offline.

```text
Amazon catalog (~50k products)
          ↓
   V5 semantic annotation
          ↓
 ┌────────┼───────────────────────────────┐
 ↓        ↓                               ↓
Canonical facts                     Product cards
 ↓                                      ↓
Canonical dictionary             Qwen3-Embedding-0.6B
 ↓                                      ↓
BGE attribute matrices           normalized dense matrix
 │                                      │
 └──────────────┐              ┌────────┘
                │              │
Raw catalog ───→ SQLite FTS5 BM25 index
```

### 2.1 V5 semantic facts

Each product is normalized into retrieval-oriented facts such as:

```text
category
brand
color
material
feature
use_case
style
price
```

Example:

```text
parent_asin: B...
category:
  - footwear
  - running shoes
  - trail running shoes
brand:
  - salomon
color:
  - black
material:
  - mesh
  - rubber
feature:
  - slip resistant
  - lightweight
  - breathable
use_case:
  - hiking
  - trail running
  - wet weather
price:
  99.00
```

Broad and specific category identities may coexist when useful for Browsing, e.g. `footwear`, `running shoes`, `trail running shoes`.

### 2.2 BGE canonical attribute index

BGE is used for **semantic query understanding / canonical expansion**.

Target model:

```text
BAAI/bge-small-en-v1.5
```

BGE searches canonical values for the semantic product attributes that are not
handled as structured constraints:

```text
category
color
material
style
feature
use_case
```

These six categorical attributes are semantic constraints in the current architecture. Brand is handled as an exact structured constraint, and price/budget is the numeric structured constraint.

BGE is **not** the product-level dense retrieval path.

### 2.3 BM25 index

The lexical route uses a field-weighted SQLite FTS5 BM25 index over raw product text, including:

```text
title
categories
features
details
store
description
```

The index should retain stronger weights for high-value fields such as title/category than generic description text.

This is BM25F-inspired field weighting, not a requirement to implement a textbook BM25F engine.

Reference:
- Robertson, Zaragoza & Taylor, *Simple BM25 Extension to Multiple Weighted Fields*, CIKM 2004.

### 2.4 V5 Browsing product-card index

Browsing uses a separate product-level dense representation built from clean V5 semantic facts rather than long noisy raw descriptions.

Example product card:

```text
Salomon Speedcross 5
category: footwear, running shoes, trail running shoes
brand: salomon
color: black
material: mesh, rubber
feature: slip resistant, lightweight, breathable
use_case: hiking, trail running, wet weather
```

Target embedding model:

```text
Qwen/Qwen3-Embedding-0.6B
```

Target full embedding dimension:

```text
1024
```

Use L2-normalized vectors so cosine similarity can be implemented as a matrix dot product.

For ~50k products, 1024-dimensional float32 embeddings are small enough to keep as the quality baseline. Smaller Matryoshka dimensions may be benchmarked later, but 1024 is the reference configuration.

This representation follows the intent-aware retrieval direction in recent e-commerce work, where semantic query/product intent attributes are embedded instead of relying only on raw product text. Price remains a separate numeric eligibility field, while brand remains an exact structured field.

The Browsing query side uses the same Qwen model with this query-only
instruction:

```text
Instruct: Retrieve products that best match the shopper's product type, intended use, desired features, and preferences.
```

The query body is compiled from the active session slots rather than the raw
conversation transcript. Brand is taken from the exact structured state;
category, color, material, style, feature, and use_case are taken from the
active semantic state. Price and size remain outside the embedding query.
For example:

```text
Instruct: Retrieve products that best match the shopper's product type, intended use, desired features, and preferences.
Query: category: jumpsuit
feature: lightweight
use_case: cosplay
```

The instruction is applied only to Qwen queries. Product-card documents remain
unprefixed before their offline embedding is generated.

References:
- Qwen3 Embedding, 2025 — https://arxiv.org/abs/2506.05176
- INSPIRE: Intent-aware Neural Sponsored Product Retrieval for E-commerce, 2026 — https://arxiv.org/abs/2606.23889

---

## 3. LLM turn interpreter

The runtime uses a small local/self-hosted LLM as a **schema-guided turn interpreter**.

Its job is language understanding only:

```text
User utterance
      ↓
LLM turn interpreter
      ↓
       current-turn semantic delta + price delta
      ↓
deterministic validation
      ↓
SessionManager
```

The LLM does not retrieve products and does not directly mutate persistent state.

### 3.1 Output schema

Conceptually:

```json
{
  "intent": "buying | browsing | null",
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

Category, color, material, style, feature, and use_case in `updates` are semantic
constraints. Brand is an exact structured constraint. `price_min` and
`price_max` are the numeric structured constraint and remain a price eligibility
rule. Size is not part of the current structured constraint contract.

Example:

```text
"I'm exploring sweatshirts and would like to compare some options."

→ intent: browsing
→ category: sweatshirt
→ use_case: none
```

This prevents dialogue framing words such as `exploring` or `compare` from becoming false product attributes.

Contrastive examples should distinguish cases such as:

```text
"I'm exploring sweatshirts."
→ browsing intent
→ category: sweatshirt

"I need boots for exploring caves."
→ category: boots
→ use_case: exploring caves
```

References:
- Lee et al., *Dialogue State Tracking with a Language Model using Schema-Driven Prompting*, EMNLP 2021.
- Goo et al., *Slot-Gated Modeling for Joint Slot Filling and Intent Prediction*, NAACL 2018.
- Li et al., *Large Language Models as Zero-shot Dialogue State Tracker through Function Calling*, ACL 2024.
- Gupta et al., *Show, Don't Tell: Demonstrations Outperform Descriptions for Schema-Guided Task-Oriented Dialogue*, NAACL 2022.

---

## 4. Dependency-aware selective dialogue state

The Agent preserves valid accumulated preferences instead of rebuilding the entire state each turn.

Example:

```text
Current:
category = shirt
use_case = sunny weather
color = black

User:
"Actually I need it for rainy weather."

Operations:
category → CARRYOVER
color    → CARRYOVER
use_case → UPDATE sunny → rainy
```

This follows the selective update principle used by **SOM-DST**.

Reference:
- Kim et al., *Efficient Dialogue State Tracking by Selectively Overwriting Memory*, ACL 2020.

### 4.1 Provenance and dependencies

Every constraint should retain provenance:

```text
attribute
value
source: explicit | inferred
parent_constraint: optional
```

Example:

```text
use_case: sunny weather [explicit]
    └── feature: UV protection [inferred]

category: shirt [explicit]
color: black [explicit]
```

If `sunny weather` is overridden, the dependent inferred `UV protection` may be removed while `shirt` and `black` remain.

This is a lightweight truth-maintenance / dependency-aware belief-revision mechanism.

Reference:
- Doyle, *A Truth Maintenance System*, Artificial Intelligence, 1979.

### 4.2 Dependency graph

Dependencies represent **derivation**, not ranking importance.

Example:

```text
category
├─ use_case
│  ├─ feature
│  ├─ material
│  └─ style
└─ style
```

Brand, color, and price/budget are generally independent unless explicit provenance says otherwise. Brand is exact structured evidence; color is semantic evidence. The graph describes semantic derivation and does not turn semantic attributes into structured filters.

### 4.3 Retrieval context must follow active state

Visible conversation history and retrieval state are different concepts.

After an override, obsolete text must not continue to influence retrieval simply because it remains in the transcript.

```text
visible transcript
≠
active retrieval state
```

Retrieval should be generated from active constraints / active goal context.

---

## 5. Adaptive intent orchestration

Intent is not permanently fixed after the first turn.

The system maintains session intent and allows controlled transitions:

```text
"I'm exploring things for a beach holiday"
        ↓
BROWSING

"Make it a black dress under $70"
        ↓
BUYING
```

The router should consider:

```text
current turn intent
+ active constraint specificity
+ previous mode
+ whether the turn is a clarification answer
```

Use hysteresis so weak signals do not cause random mode flipping.

A normal answer to the Agent's own clarification question should generally update state without being treated as an unrelated intent switch.

---

## 6. Buying retrieval path

Buying is precision-oriented.

```text
Active Buying State
        ↓
price eligibility
exact brand
semantic evidence
        ↓
BGE semantic expansion
        ↓
slot / concept groups
        ↓
grouped field-weighted BM25
        ↓
per-group normalization / coverage
        ↓
Buying rank
        ↓
Candidate Pool
```

### 6.1 Structured constraints

Structured logic currently owns two constraints:

```text
price_min
price_max
brand (exact)
```

Known satisfying prices are eligible; known violating prices and null prices are excluded when a budget is active. Exact brand is preserved as structured evidence. No category, color, material, style, feature, use-case, or size value is currently a structured constraint.

Non-brand categorical values remain semantic evidence for BGE expansion, product-card retrieval, and BM25 ranking.

### 6.2 Slot-guided BM25 query compilation

BM25 queries come from **active slots**, not blindly concatenated full conversation history.

Example:

```text
feature:
"won't slip"
```

BGE may expand this to:

```text
slip resistant
non slip
traction
```

These terms form one semantic concept group:

```text
C_feature = {
  won't slip,
  slip resistant,
  non slip,
  traction
}
```

Likewise:

```text
C_use_case = {
  rain,
  rainy weather,
  wet weather
}
```

The system must **not** treat every synonym as an independent user requirement.

### 6.3 No Cartesian synonym enumeration

Do not evaluate combinations such as:

```text
20 × 20 × 20 × 20 × 20
```

Maintain one small lexical group per active slot/concept.

### 6.4 Conservative expansion

Use only a small high-confidence BGE expansion set, e.g. top candidates satisfying a similarity threshold and margin from the best match.

The explicit user value remains the strongest evidence.

Uncontrolled expansion can cause query drift.

References:
- Crimp & Trotman, *Automatic Term Reweighting for Query Expansion*, 2017.
- Dai et al., *End-to-End Query Term Weighting (TW-BERT)*, 2023.

### 6.5 Buying ranking invariant

The target Buying flow is:

```text
price eligibility
exact brand
semantic evidence
          +
BGE-expanded grouped BM25 evidence
          +
profile-aware rating tie-break
```

**BGE does not contribute a second direct product posting-list score.**

There is no architectural term such as:

```text
+ 0.20 × canonical BGE posting-list score
```

BGE's production role in Buying is to improve the lexical query that BM25 searches.

Conceptually:

```text
BGE
 ↓
canonical alternatives
 ↓
BM25 query groups
 ↓
product ranking
```

not:

```text
BGE posting score ─┐
                   ├─ add together
BGE-expanded BM25 ─┘
```

A product covering multiple independent constraints should beat one matching many synonyms from only one constraint.

Example:

```text
Product A
category   ✓
color      ✓
feature    ✓
use_case   ✓

Product B
category   ✓
color      ✗
feature    ✗
use_case   ✓✓✓✓✓
```

Product A should normally receive stronger overall evidence.

References:
- Voskarides et al., *Query Resolution for Conversational Search with Limited Supervision (QuReTeC)*, SIGIR 2020.
- Robertson, Zaragoza & Taylor, *Simple BM25 Extension to Multiple Weighted Fields*, CIKM 2004.

---

## 7. Browsing retrieval path

Browsing is discovery-oriented and is intentionally different from Buying.

```text
Active Browsing State
        ↓
semantic query serializer
        ↓
Qwen3-Embedding-0.6B
        ↓
Dense Top-N from V5 product matrix
        │
        │            Raw active-goal BM25 Top-N
        │                      │
        └──────────┬───────────┘
                   ↓
                  RRF
                   ↓
             fused candidates
                   ↓
                  MMR
                   ↓
          diverse Browsing rank
```

### 7.1 Dense query representation

The dense query should be constructed from the **active semantic state**, not stale full transcript text.

Recommended serialization:

```text
running shoes
; color: black
; feature: slip resistant
; feature: lightweight
; use_case: hiking
; use_case: wet weather
```

With Qwen3, use an instruction oriented toward product suitability, for example:

```text
Retrieve products that best satisfy the shopper's intended scenario,
product type, desired features, use case and preferences.
Prioritize semantic suitability even when wording differs.
```

The product side uses the V5 semantic product card built offline. Brand can
also contribute exact structured evidence independently of the semantic card.

### 7.2 Dense retrieval

Use L2-normalized 1024-dimensional Qwen3 embeddings.

Runtime dense retrieval can initially be exact matrix-vector cosine search over ~50k products:

```text
product_matrix @ query_vector
```

Retrieve a broad candidate set such as Top-100 before fusion.

### 7.3 BM25 lexical complement

Browsing also runs a lexical route over the active goal text.

This is an independent complement to dense retrieval, useful for exact names, rare terms, brands, model names, and lexical evidence that dense retrieval may blur.

### 7.4 Reciprocal Rank Fusion

Do not directly add raw cosine and SQLite BM25 scores because they have unrelated scales.

Fuse ranked lists with **Reciprocal Rank Fusion (RRF)**.

Conceptually:

```text
RRF(product)
= contribution from dense rank
+ contribution from BM25 rank
```

Reference:
- Lee et al., *On Complementarity Objectives for Hybrid Retrieval*, ACL 2023.

### 7.5 MMR diversity

Browsing should avoid returning ten near-identical products.

After fusion, apply **Maximal Marginal Relevance (MMR)** over a manageable fused candidate pool.

```text
MMR
= relevance
- redundancy with already-selected products
```

The dense product vectors can be reused for redundancy similarity.

Reference:
- Carbonell & Goldstein, *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*, 1998.

### 7.6 Browsing benchmark focus

Dense retrieval should justify itself by recovering useful products missed by the strengthened lexical path.

Track at least:

```text
Browsing Hit@10
Browsing Hit@50
Browsing Hit@100
MRR
latency
dense-only recovery@100 when BM25 misses
```

---

## 8. Candidate-aware clarification

Clarification is selected from the **retrieved candidate distribution**, not simply from whichever state field is empty.

```text
Broad candidate retrieval
        ↓
Candidate Pool Analyzer
        ↓
facet distributions
        ↓
question utility
        ↓
clarify or recommend
```

For each unresolved attribute, calculate:

```text
coverage
expected candidate reduction
value diversity
answerability
remaining-turn value
```

Expected reduction can use:

```text
Reduction(A) = 1 - Σ p(v)^2
```

A simple conceptual utility is:

```text
QuestionValue(A)
≈ coverage(A)
× expected_reduction(A)
× answerability(A)
× remaining_turn_value
```

Example:

```text
8,000 candidates

color:
black 29%
white 26%
blue  24%
other 21%

brand:
other  96%
Nike    2%
Adidas  1%
...
```

`color` partitions the candidate set much more strongly than `brand`, so it is usually the better question.

Question text may expose the dominant candidate values:

```text
"Do you prefer black, white, blue, or another color?"
```

Skip attributes that are already resolved, previously asked, marked `no preference`, or have insufficient candidate coverage.

References:
- *ProductAgent: Benchmarking Conversational Product Search Agent with Asking Clarification Questions*, EMNLP Industry 2025.
- *Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision-Tree Search*, ACL 2025.

---

## 9. Over-generality cutoff

An over-general request should control **what the Agent asks**, and ideally
also **what it spends**. The first is implemented; the second is not.

Example:

```text
"I need some clothes"
```

### 9.1 Implemented: breadth decides whether to ask

Every turn analyses the candidate pool by facet, and pool size gates the
clarification question:

```text
                        retrieval
                            ↓
                 candidate facet analysis
                            ↓
              is the pool still broad (over 50)?
                            │
                 ┌──────────┴──────────┐
                 ↓                     ↓
                no                    yes
          recommend only    score every askable attribute
                                       ↓
                          does the best clear the abstain floor?
                                       │
                            ┌──────────┴──────────┐
                            ↓                     ↓
                           no                    yes
                     recommend only     ask that attribute
```

Below the breadth threshold, another question is worth less than the turn it
costs, so the Agent recommends instead. Above it, the attribute is chosen by
the Section 8 utility and must still clear the abstain floor.

Recommendations are returned on every turn regardless, alongside any
question, because the evaluator scores a list each turn.

### 9.2 Not implemented: breadth does not yet control computation

The current order of work is:

```text
                        active session state
                                 │
             ┌───────────────────┴───────────────────┐
             ↓                                       ↓
      recommendation ranking                  clarification pool
      top-K, diversified                      broad, undiversified
             │                                       ↓
             │                              candidate facet analysis
             │                                       ↓
             │                                  breadth test
             │                                       │
             │                            ┌──────────┴──────────┐
             │                            ↓                     ↓
             │                       ask nothing        ask an attribute
             │                            │                     │
             └───────────────┬────────────┴─────────────────────┘
                             ↓
                recommendations + optional question
```

Both branches run on every turn, and neither is conditional on the other. The
breadth test reads only the right-hand branch.

The second retrieval is deliberate and correct in itself. Clarification needs
a broad distribution of product facts, which the small diversified Browsing
recommendation pool would truncate, and the broad pool is already the cheaper
of the two because it skips diversification.

So the breadth threshold currently gates *conversation*, not *computation*.

### 9.3 What closing the gap would require

Reordering, not new components. The broad clarification pool is already the
cheap retrieval the design asks for, so running it first would let the
breadth test decide whether the full ranking is worth doing at all:

```text
                     cheap broad retrieval
                             ↓
                  candidate facet analysis
                             ↓
                        breadth test
                             │
                  ┌──────────┴──────────┐
                  ↓                     ↓
                 yes                   no
        cheap top-K from the      full ranking
        pool + clarification      as today
```

This removes the duplicate retrieval rather than adding a stage. It is
unmeasured: the saving is proportional to the share of turns that are
actually broad, which has not been instrumented, and any change to how the
recommendation list is produced on broad turns affects the scored metric
directly.

---

## 10. Personalization and dynamic context

The Agent maintains two conceptually different memory layers:

```text
short-term active state
= current goal and explicit conversation constraints

soft long-term profile
= stable preferences / tendencies used as weak priors
```

Long-term profile signals must never override explicit current-turn requirements.

Good uses of profile priors include:

```text
clarification answerability
rating tie-break strength
soft preference ordering
```

Explicit current intent always wins.

---

## 11. Runtime scenario examples

### 11.1 Buying

```text
User:
"black running shoes for wet weather under $100"

Turn interpreter:
intent = BUYING
category = running shoes
color = black
use_case = wet weather
price_max = 100

            ↓
price budget eligibility
            ↓
BGE expansion
  running shoes → trail running shoes ...
  wet weather   → rain / rainy weather ...
            ↓
slot-group BM25
            ↓
Buying rank
```

### 11.2 Browsing → Buying

```text
Turn 1:
"I'm exploring things for a beach holiday"

intent = BROWSING
use_case = beach holiday
        ↓
Qwen3 dense + raw BM25
        ↓
       RRF
        ↓
       MMR
        ↓
diverse cross-category ideas

Turn 2:
"Make it a black dress under $70"

state gains:
category = dress
color = black
price_max = 70

BROWSING → BUYING
        ↓
BGE-expanded grouped BM25
```

### 11.3 Buying → preference override → Buying

```text
Current:
category = jacket            [explicit]
use_case = sunny weather     [explicit]
feature = UV protection      [inferred from use_case]
color = black                [explicit]
brand = Columbia             [explicit]

User:
"Actually I'll mostly use it in rainy weather"
        ↓
UPDATE:
use_case sunny → rainy

DELETE IF DEPENDENT:
feature UV protection

CARRYOVER:
category jacket
color black
brand Columbia

        ↓
regenerate retrieval from active state
        ↓
Buying path
```

### 11.4 Over-general Browsing

```text
User:
"I need some clothes"

cheap broad retrieval
        ↓
8,000 candidates
        ↓
Candidate Pool Analyzer
        ↓
use_case has highest question utility
        ↓
"Is this mainly for casual wear, work, sports, or outdoor use?"
```

---

## 12. Component responsibilities

```text
LLM Turn Interpreter
→ understand current user language
→ slots + override delta

Structured validators
→ parse / validate price/budget
→ validate exact brand

Semantic matcher
→ resolve category, color, material, style, feature, and use_case

SessionManager
→ own state and provenance
→ apply selective dependency-aware updates

Adaptive Intent Orchestrator
→ choose Buying vs Browsing runtime path
→ allow controlled mode transitions
→ stop unnecessary computation for over-general state

BGE
→ canonical semantic expansion for lexical retrieval
→ NOT an independent direct product score

BM25
→ efficient lexical product retrieval
→ grouped expanded lexical signal for both modes

Qwen3-Embedding-0.6B
→ product-level semantic retrieval for Browsing

RRF
→ fuse Browsing dense + BM25 ranks without mixing incompatible score scales

MMR
→ diversify Browsing results

Candidate Pool Analyzer
→ compute facet statistics
→ decide whether another question is worth a turn
```

---

## 13. Architectural invariants

These rules should be preserved during implementation unless this document is explicitly revised:

1. **Buying and Browsing use different retrieval strategies.**
2. **BGE is semantic/canonical query expansion for Buying, not a separate direct product-ranking score.**
3. **Buying uses price eligibility, exact brand evidence, and BGE-expanded slot/concept-group BM25 semantic evidence.**
4. **Browsing uses Qwen3 product-level dense retrieval plus an independent raw BM25 complement.**
5. **Browsing sparse/dense fusion uses rank fusion such as RRF, not raw cosine + raw BM25 addition.**
6. **MMR/diversity is primarily a Browsing concern.**
7. **Retrieval is derived from active state, not blindly from stale full conversation history.**
8. **Preference overrides preserve independent explicit constraints and remove only invalidated dependent state.**
9. **Clarification is candidate-aware and should maximize useful search-space reduction.**
10. **Over-generality can stop expensive computation and trigger clarification.**
11. **Qwen3-Embedding-0.6B uses 1024 dimensions as the reference dense-product configuration.**
12. **Long-term/profile context is a soft prior and never overrides explicit current intent.**
13. **Product-level LLM reranking is optional, not part of the required core path unless benchmark evidence justifies it.**

---

## 14. Implementation priority

Developers should implement toward this architecture in this order when gaps exist:

```text
1. Correct active-state / override semantics
2. Buying = price eligibility + exact brand + BGE-expanded grouped BM25
3. Browsing V5 Qwen3 dense index/query path
4. Browsing BM25 complement + RRF
5. Browsing MMR diversity
6. Adaptive BUYING ↔ BROWSING transitions
7. Candidate-aware clarification / over-generality cutoff
8. Profile priors and optional reranking only after core retrieval is benchmarked
```

Do not revive legacy retrieval stages merely because old code or artifacts still exist. If implementation and this document disagree, update the implementation to this architecture or deliberately revise this file first.
