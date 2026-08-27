# Shopping Copilot architecture

Status: MVP architecture source of truth.

This document describes the overall data, retrieval, state, and Agent architecture for Shopping Copilot. It is intentionally organized by system responsibility rather than GitHub issue number. Issues and pull requests may implement parts of this design, but they do not define the architecture.

## 1. Goals and operating constraints

Shopping Copilot runs over a frozen catalog of roughly 50,000 products. The runtime must support multi-turn conversational shopping while remaining simple enough to run in one process with precomputed artifacts loaded into memory.

The design optimizes for the competition objective:

- return the best current Top-K recommendations on every scoreable turn;
- find the exact target `parent_asin` as early and as high in the ranking as possible;
- ask at most one useful structured clarification per turn;
- preserve good candidates while the user is still clarifying intent;
- treat Buying as precision-first and Browsing as recall-first;
- keep hidden evaluator information completely outside Agent logic.

The first MVP deliberately favors deterministic preprocessing, in-memory indexes, exact dense search, and benchmark-driven iteration over infrastructure complexity.

## 2. System-level architecture

```text
                              catalog.jsonl
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
        LAYER 1 — EXISTING FLOW          LAYER 2 — EMBEDDING FLOW
              (UNCHANGED)                 (DIRECT FROM CATALOG)
                    |                             |
      +-------------+-------------+       +------+-------+-------+-------+---------+
      |             |             |       |              |       |       |         |
      v             v             v       v              v       v       v         v
 Tier 1         Tier 2        Tier 3   categories       title  features description details
 structured     trusted       descriptive                |       |       |         |
 extraction     annotation    annotation                 |       |       |         |
      |             |             |       |              |       |       |         |
      | category    | brand       | style |              |       |       |         |
      | price       | color       | feature              |       |       |         |
      | size labels | material    | use_case             |       |       |         |
      | measurements|             |                      |       |       |         |
      | package dims|             |                      |       |       |         |
      | product dims|             |                      |       |       |         |
      | item weight |             |                      |       |       |         |
      +-------------+-------------+                      |       |       |         |
                    |                                    |       |       |         |
                    v                                    v       v       v         v
        validation + normalization                 category   title   features description selected
                    |                              embedding embedding embedding embedding details
                    v                                                       |                   embedding
          canonical product facts                                           |
                    |                                                       |
          +---------+---------+                                             |
          |                   |                                             |
          v                   v                                             |
 exact / numeric       canonical registries                                 |
      indexes                                                               |
          |                   |                                             |
          +---------+---------+                                             |
                    |                                                       |
                    +----------------------+--------------------------------+
                                           |
                                           v
                                  RETRIEVAL / RANKING
                                           |
                              +------------+------------+
                              |                         |
                              v                         v
                    Layer 1 evidence             Layer 2 scores
                    exact / structured           category similarity
                    canonical matches            title similarity
                                                 features similarity
                                                 description similarity
                                                 details similarity
                              |                         |
                              +------------+------------+
                                           |
                                           v
                                        Top-K
```

The central principle remains unchanged for Layer 1: not all facts have the same reliability or retrieval role.

Layer 2 is an independent semantic retrieval path built directly from `catalog.jsonl`. It does not consume Tier 1, Tier 2, or Tier 3 outputs. Layer 1 and Layer 2 meet only at retrieval/ranking.

Layer 2 creates five separate product views:

```text
categories
title
features
description
selected details
```

Each view is embedded independently so noisy text in one catalog field does not contaminate the semantic representation of another field.

## 3. Product knowledge model

### Tier 1 — exact structured facts

Tier 1 contains facts that can be parsed or copied into a typed representation with high reliability:

```text
category
price
size labels
numeric measurements
package dimensions
product dimensions
item weight
```

These are not free-form semantic keywords. They should be represented with explicit types and units whenever possible.

Examples:

```json
{
  "size_labels": ["s", "m", "l", "xl"],
  "measurements": [
    {
      "type": "inseam",
      "values": [4, 6, 8],
      "unit": "inch"
    }
  ]
}
```

```json
{
  "measurements": [
    {
      "type": "case_diameter",
      "values": [44],
      "unit": "mm"
    }
  ]
}
```

```json
{
  "package_dimensions": {
    "values": [10, 8, 3],
    "unit": "inch"
  }
}
```

The structured extractor should be conservative. If a measurement cannot be typed confidently, leave it unstructured and allow raw-text retrieval to preserve recall.

Package dimensions and product dimensions remain distinct from shopper size. They are preserved because a user may explicitly request packaging, shipping, or physical-dimension constraints.

### Tier 2 — trusted semantic facts

Tier 2 contains semantic fields that are useful enough to receive strong exact-match weight, so annotation precision is more important than coverage:

```text
brand
color
material
```

The annotation policy is sparse and conservative:

- emit only facts clearly supported by the product record;
- prefer omission over a speculative value;
- normalize to reusable canonical forms;
- do not confuse patterns with colors;
- do not transfer accessory or packaging materials to the main product.

Brand should use deterministic structured metadata when it is clearly reliable, but catalog metadata can be noisy. The trusted brand pipeline may normalize, verify, or recover the brand from title/store/manufacturer evidence when necessary. A suspicious manufacturer value must not automatically become the final brand.

Tier 2 exact matches can receive very high ranking weight. Missing Tier 2 annotation is not automatically proof that the product violates the request, because the annotation policy intentionally favors precision over recall.

### Tier 3 — descriptive semantic facts

Tier 3 contains broader semantic descriptors:

```text
style
feature
use_case
```

These fields are primarily for:

- semantic matching;
- product embedding enrichment;
- soft ranking boosts;
- candidate interpretation;
- clarification planning;
- later learned ranking or posterior logic.

The extraction policy is higher recall than Tier 2. Values still need source support, but some noise is acceptable because Tier 3 is not a hard-filter layer by default.

Examples:

```text
style:
  bohemian
  relaxed_fit
  wide_leg
  platform
  wrap
  minimalist

feature:
  breathable
  lightweight
  arch_support
  waterproof
  adjustable_straps
  cushioned

use_case:
  running
  hiking
  camping
  fishing
  wedding_guest
  meditation
```

Generic descriptors that add little retrieval value should still be avoided when possible, for example `lifestyle`, `all_occasions`, or `daily_life`.

`use_case` belongs in Tier 3. It is intentionally broader and fuzzier than brand, color, or material. A use-case mismatch must not eliminate a candidate.

### Tier 4 — raw product text

Tier 4 is the recall safety net:

```text
title
features
description
details
```

The original catalog remains immutable. Raw text is preserved even when a fact was not successfully structured or annotated.

Tier 4 feeds the whole-product embedding representation and can recover unusual language, rare attributes, model names, product-specific phrases, and facts not covered by the structured ontology.

The architecture must not require the annotation schema to represent every possible user request.

## 4. Canonical product-facts representation

A derived product record should conceptually separate the trust tiers:

```json
{
  "parent_asin": "B123",
  "structured": {
    "category": ["women", "clothing", "active", "active_shorts"],
    "price": 20.99,
    "size_labels": ["s", "m", "l", "xl"],
    "measurements": [
      {
        "type": "inseam",
        "values": [4, 6, 8],
        "unit": "inch"
      }
    ],
    "package_dimensions": null,
    "product_dimensions": null,
    "item_weight": null
  },
  "trusted_semantic": {
    "brand": "example_brand",
    "color": ["black"],
    "material": ["nylon", "spandex"]
  },
  "descriptive_semantic": {
    "style": ["high_waisted", "biker_shorts"],
    "feature": ["compression", "hidden_pockets", "non_see_through"],
    "use_case": ["yoga", "training"]
  }
}
```

The physical storage may remain flattened for compatibility during migration, but retrieval code should preserve the conceptual tier distinction and must not assign the same semantics or weight to every field.

Derived facts should retain enough provenance internally to distinguish deterministic extraction, exact normalization, LLM annotation, and later semantic resolution where needed.

## 5. Offline processing pipeline

### 5.1 Preserve the source catalog

`catalog.jsonl` is the immutable source record. Derived processing must not rewrite or remove source text.

### 5.2 Structured extraction

A deterministic extractor should derive:

```text
category path
price
size labels
numeric measurements
package dimensions
product dimensions
item weight
```

Typical supported measurements include:

```text
inseam
waist
length
width
height
diameter
case_diameter
heel_height
shaft_height
canopy_size
necklace_length
shoe/sock size ranges
```

Units should be normalized into a consistent internal representation while retaining the original semantic type.

The parser should distinguish shopper-facing measurements from generic catalog metadata. For example, `31 inch inseam` is a typed shopper measurement, while `Package Dimensions: 10 x 8 x 3 inches` belongs specifically to package dimensions rather than generic size.

### 5.3 Semantic annotation

The LLM annotator should output the semantic fields only:

```text
brand
color
material
style
feature
use_case
```

The prompt must explicitly use two policies:

```text
brand/color/material
    -> sparse, high precision

style/feature/use_case
    -> broader, source-supported semantic coverage
```

The LLM should not be responsible for category hierarchy, price, numeric size parsing, package dimensions, or typed measurements.

### 5.4 Validation and normalization

LLM output must pass deterministic cleanup before becoming canonical product facts.

Typical normalization responsibilities:

```text
crewneck          -> crew_neck
vneck             -> v_neck
quick_dry         -> quick_drying
machine_wash      -> machine_washable
4_way_stretch     -> four_way_stretch
```

Validation should also reject or remap known field mistakes where rules are reliable, such as pattern terms incorrectly emitted as colors.

The validator should remain conservative. It should not become a second speculative semantic model.

## 6. Canonical registries and attribute embeddings

Canonical semantic values receive stable IDs:

```text
brand:nike
color:black
material:leather
style:relaxed_fit
feature:waterproof
use_case:hiking
```

The registry owns deterministic normalization and exact lookup.

Exact phrase matching should be longest/specific-first. A normalized surface may map to multiple canonical IDs; ambiguity should be preserved rather than arbitrarily resolved.

Semantic fallback is attribute-scoped:

- `brand`: exact/normalized matching by default;
- `color`: exact first, optional high-threshold semantic fallback;
- `material`: exact first, optional high-threshold semantic fallback;
- `style`: exact plus semantic fallback;
- `feature`: exact plus semantic fallback;
- `use_case`: exact plus semantic fallback;
- `size`: structured/exact only by default;
- `price` and measurements: numeric only.

Attribute-value embeddings are separate from whole-product embeddings. They resolve user phrases to canonical values; they are not a product retrieval index.

## 7. Layer 2 direct field embeddings

Layer 1 remains unchanged.

Layer 2 is an independent embedding path that reads directly from `catalog.jsonl`.

For every product, Layer 2 creates five separate semantic views:

```text
categories
title
features
description
selected details
```

The views are embedded separately. The architecture does not create one giant embedding from the entire product record.

### 7.1 Categories embedding

Build one embedding from the product's ordered category path.

Example source:

```json
{
  "categories": [
    "Clothing, Shoes & Jewelry",
    "Women",
    "Jewelry",
    "Rings"
  ]
}
```

Embedding text:

```text
clothing shoes jewelry women jewelry rings
```

The category view provides a strong semantic product-type signal and complements the existing Layer 1 category logic.

### 7.2 Title embedding

Build one embedding from the original product title.

Example:

```text
DALEGEM Genuine Yellow Tiger Eye Stone Ring for Men Women,
Retro Vintage Quartz Crystal Gemstone Turkish Ring Jewelry Gift
```

The title is expected to be one of the strongest semantic views because it is concise and often contains product type, identity, material, style, and use-case information.

### 7.3 Features embedding

Build one embedding from all feature bullets for the product.

For the MVP, concatenate the feature strings into one input:

```python
feature_text = " ".join(product["features"])
```

Do not create one vector per individual feature bullet.

Feature text is often information-rich, but it can also contain seller marketing, guarantees, package contents, SEO wording, or unsupported claims. Keeping features in their own embedding view prevents that noise from contaminating title or category similarity.

### 7.4 Description embedding

Build one embedding from the product description when description content exists.

Description can recover useful long-tail information that does not appear in the title or feature bullets, but it can also contain repetitive or marketing-heavy language.

Description therefore remains a separate view with an independently tunable weight.

If a product has no description, description contributes no score rather than negative evidence.

### 7.5 Selected details embedding

Do not embed the full `details` dictionary blindly.

Build the details embedding only from shopper-relevant keys.

Useful examples may include:

```text
Fabric Type
Outer Material
Sole Material
Closure Type
Water Resistance Level
Fit
Compatibility
Style Name
Lining
```

Avoid embedding metadata that usually has little semantic shopping value:

```text
Date First Available
ASIN
Best Sellers Rank
internal identifiers
seller/catalog bookkeeping
```

Numeric details that already have strong structured meaning should continue to be handled by Layer 1 rather than semantic similarity.

Examples include:

```text
Package Dimensions
Item Weight
Stone Width
Stone Length
case diameter
```

### 7.6 Embedding artifacts

A practical artifact layout is:

```text
data/derived/product_embeddings/
├── category_embeddings.npy
├── title_embeddings.npy
├── features_embeddings.npy
├── description_embeddings.npy
├── details_embeddings.npy
├── product_embedding_metadata.json
└── manifest.json
```

All five matrices must use exactly the same product row order.

Metadata must preserve the exact row-to-`parent_asin` mapping.

The manifest records:

```text
embedding model
embedding dimension
normalization
source catalog version
product count
field/view names
generation configuration
```

### 7.7 Query embedding

For the first MVP baseline, create one semantic embedding from the current user/session query and compare the same query vector against all five product views.

Example:

```python
query_embedding = embed(user_query)

category_scores = category_embeddings @ query_embedding
title_scores = title_embeddings @ query_embedding
features_scores = features_embeddings @ query_embedding
description_scores = description_embeddings @ query_embedding
details_scores = details_embeddings @ query_embedding
```

This keeps the baseline simple and avoids adding query-to-field routing before benchmark evidence justifies it.

### 7.8 Multi-view Layer 2 score

A conceptual Layer 2 score is:

```text
layer2_score(product) =
    w_category    * category_similarity
  + w_title       * title_similarity
  + w_features    * features_similarity
  + w_description * description_similarity
  + w_details     * details_similarity
```

Actual weights are benchmark-tuned and should not be fixed in this document.

A reasonable starting trust order is:

```text
title          very high
categories     very high
features       high
description    medium
details        medium / supporting
```

### 7.9 Missing fields

Products may have missing or empty catalog fields.

The runtime should keep presence masks such as:

```text
has_categories
has_title
has_features
has_description
has_details
```

A missing field contributes no Layer 2 score and must not be treated as negative evidence.

### 7.10 Vector normalization and search

Vectors should be L2-normalized float32.

For normalized vectors:

```python
scores = embeddings @ query_embedding
```

is cosine similarity.

For roughly 50,000 products, exact in-process search is sufficient. Normalized NumPy inner product or FAISS `IndexFlatIP` is acceptable.

An external vector database is not required.

### 7.11 Layer 1 and Layer 2 relationship

Layer 1 remains the existing structured/canonical pipeline.

Layer 2 does not replace Layer 1 and does not require Layer 1 output.

The two paths run independently:

```text
catalog.jsonl
    |
    +--> Layer 1 existing structured / annotation / canonical path
    |
    `--> Layer 2 direct field embedding path
```

They meet only during retrieval/ranking.

Layer 1 contributes:

```text
structured constraints
exact / numeric evidence
trusted canonical matches
descriptive canonical matches
```

Layer 2 contributes:

```text
category similarity
title similarity
features similarity
description similarity
details similarity
```

The ranking layer combines both sources of evidence.

## 8. User-utterance processing

User processing mirrors the product trust tiers.

Example:

```text
"I need black waterproof running shorts,
 6 inch inseam, under $30, for hot weather"
```

should be decomposed approximately into:

```json
{
  "structured": {
    "price_max": 30,
    "measurements": [
      {
        "type": "inseam",
        "value": 6,
        "unit": "inch"
      }
    ]
  },
  "trusted_semantic": {
    "color": ["black"]
  },
  "descriptive_semantic": {
    "feature": ["waterproof"],
    "use_case": ["running"]
  },
  "residual_text": "for hot weather"
}
```

The runtime flow is:

```text
user utterance
    |
    v
1. deterministic structured parser
    |   price, quantities, size labels, measurements, dimensions
    |
    v
2. exact/normalized canonical matching
    |   category, brand, color, material, known descriptors
    |
    v
3. mark matched spans
    |
    v
4. semantic fallback on meaningful residual phrases
    |   primarily style, feature, use_case
    |
    v
5. retain unresolved/full semantic context for dense retrieval
```

Priority is:

```text
structured parse
> exact/normalized canonical match
> high-confidence semantic fallback
> unresolved/raw semantic context
```

An exact match must not be remapped semantically.

Size and typed measurements should not rely on embedding similarity for final enforcement. Natural-language aliases may be normalized, but `xl` must not become `l` or `xxl` because those strings are semantically similar.

## 9. Session state

Session state is process-local for the MVP.

Conceptually:

```json
{
  "session_id": "...",
  "mode": "BUYING",
  "structured": {
    "category": [],
    "price_min": null,
    "price_max": null,
    "size_labels": [],
    "measurements": []
  },
  "trusted_semantic": {
    "brand": [],
    "color": [],
    "material": []
  },
  "descriptive_semantic": {
    "style": [],
    "feature": [],
    "use_case": []
  },
  "asked_attributes": [],
  "last_recommendations": [],
  "last_user_message": null,
  "turn": 0
}
```

The Buying/Browsing router runs on the first shopping utterance and stores the mode. Later clarification replies update constraints rather than rerunning the router automatically.

New explicit information overrides stale conflicting information. For example, `actually brown` replaces `black` rather than appending both.

An intent override that clearly changes the shopping goal clears stale goal-specific constraints before applying the new request.

No database or Redis is required for the MVP.

## 10. Retrieval architecture

The retriever consumes parsed session state. It does not re-parse raw user language.

### 10.1 Trust-aware evidence

Retrieval must distinguish evidence strength:

```text
Tier 1 explicit structured match
    -> strongest / often hard constraint in Buying

Tier 2 trusted semantic exact match
    -> very strong ranking evidence

Tier 3 descriptive semantic match
    -> soft ranking evidence

Layer 2 direct-field similarity
    -> recall and broad semantic ranking
```

A useful mental model is:

```text
category exact / numeric budget / explicit size or measurement    VERY HIGH
brand exact                                                       VERY HIGH
color/material exact                                              HIGH
style/feature/use_case                                            MEDIUM
Layer 2 multi-view similarity                                      MEDIUM
```

Actual weights are benchmark-tuned and should not be hard-coded into this document.

Absence of a sparse Tier 2 annotation should not automatically equal contradiction. Hard elimination should be reserved for fields whose semantics and coverage make it safe, especially explicit Tier 1 constraints.

### 10.2 Buying mode

Buying is precision-first.

```text
all products
    |
    v
apply explicit Tier 1 constraints
    |
    v
apply strong trusted-semantic evidence
    |
    v
score descriptive semantic matches
    |
    v
use Layer 2 multi-view similarity among viable candidates
    |   categories / title / features / description / selected details
    v
rank candidate pool
```

Examples of constraints that may be enforced strongly:

```text
category
price bounds
explicit size
explicit typed measurements
package-dimension limits when requested
```

Tier 2 matches such as exact brand, color, or material receive high weight. Whether a Tier 2 mismatch becomes a hard filter is a benchmark-driven decision and must account for annotation coverage.

Tier 3 fields are soft by default.

If the strict pool becomes too small or empty, controlled relaxation should remove the weakest soft semantic evidence first while preserving explicit numeric and exact structured requirements as long as possible.

If controlled relaxation still cannot produce a useful pool, dense retrieval over the broader catalog can recover candidates while recording the fallback in provenance.

### 10.3 Browsing mode

Browsing is recall-first.

```text
full current/session semantic context
        |
        v
Layer 2 multi-view dense retrieval over 50k
        |
        v
broad candidate pool
        |
        v
Tier 2 / Tier 3 preference boosts
        |
        v
ranked candidates
```

Vague preferences should not be aggressive hard filters.

### 10.4 Shared candidate contract

Both modes should produce one downstream candidate representation:

```json
{
  "parent_asin": "B123",
  "retrieval_mode": "BUYING",
  "dense_score": 0.82,
  "structured_score": 1.0,
  "trusted_semantic_score": 0.9,
  "descriptive_semantic_score": 0.55,
  "matched_constraints": ["brand:nike", "color:black"],
  "violated_constraints": [],
  "relaxed_constraints": []
}
```

The exact internal score fields may evolve, but downstream ranking and clarification should not depend on a separate candidate type for each retrieval route.

## 11. In-memory runtime indexes

At Agent construction, load reusable data once:

```text
canonical product facts
product_by_asin
structured numeric arrays/lookups
Tier 2 canonical inverted indexes
Tier 3 canonical inverted indexes where useful
canonical registries
Layer 2 category/title/features/description/details embedding matrices + shared row metadata when valid
```

Typical exact indexes:

```text
category[value] -> set(parent_asin)
brand[value]    -> set(parent_asin)
color[value]    -> set(parent_asin)
material[value] -> set(parent_asin)
style[value]    -> set(parent_asin)
feature[value]  -> set(parent_asin)
use_case[value] -> set(parent_asin)
```

Size and measurements should use typed structured indexes rather than pretending every numeric value is one flat categorical vocabulary.

No Postgres, external vector database, or distributed serving layer is required for the first MVP.

## 12. Clarification policy

Every scoreable turn returns recommendations even when asking a question.

```text
current candidate pool
      |
      +--> current best Top-K
      |
      `--> optional ONE clarification attribute
```

A deterministic one-step policy is sufficient initially. It can estimate question value using candidate coverage, diversity/split quality, expected remaining-pool size, whether the attribute is already known/asked, and Buying/Browsing mode.

Do not ask an attribute if it is already known, already asked, has poor candidate coverage, or is unlikely to materially change ranking.

The supported public `ask_attribute` values remain those required by the competition contract. Internal typed measurements can still map to the closest supported public question category when needed.

Recursive DP, learned question policies, or posterior planning can replace the utility later without changing the Agent response contract.

## 13. Agent contract

The runtime must preserve the official interface:

```text
reset(session_id, user_profile)
respond(session_id, user_message, turn, top_k)
```

A response has the evaluator-compatible shape:

```json
{
  "message": "Do you have a material preference?",
  "ask_attribute": "material",
  "recommendations": [
    {"parent_asin": "B000..."}
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0
  }
}
```

Requirements:

- recommendation IDs must be valid catalog `parent_asin` values;
- recommendations are ordered best-first and unique;
- return at most the requested `top_k`;
- every scoreable turn returns the current best recommendations;
- asking a clarification does not replace recommendations;
- `ask_attribute` is one supported enum value or `null`;
- optional usage values are non-negative;
- hidden target or simulator-only information is never exposed to Agent logic.

Partial parsing or missing artifacts must degrade to a valid best-effort response rather than an exception or ask-only turn.

## 14. Artifact and runtime boundaries

Offline artifacts are versioned build outputs, not runtime source-of-truth claims.

Expected derived areas may include:

```text
data/derived/
├── annotations/
├── catalog_facts/
├── dictionary/
└── product_embeddings/
    ├── category_embeddings.npy
    ├── title_embeddings.npy
    ├── features_embeddings.npy
    ├── description_embeddings.npy
    └── details_embeddings.npy
```

Every generated artifact should record enough information to detect mismatches, including source/facts version, model or normalization configuration, dimensions/counts, and row mappings where relevant.

Artifacts are generated offline and loaded once into the Agent process. The runtime must not regenerate all product annotations or embeddings per session.

## 15. Evaluation architecture

The evaluator is outside the Agent boundary.

Core competition metrics are:

```text
HitRate@10
MRR
MTTC
Efficiency
TechnicalScore
```

The fixed Manual400 benchmark is a development diagnostic set and must remain unchanged while comparing architecture iterations.

Useful diagnostics include:

```text
cumulative hit rate by turn
first-hit turn distribution
target rank buckets
structured parse success/failure
Tier 2 exact matches and unresolved phrases
Tier 3 semantic fallback rates
candidate-pool sizes
controlled relaxation frequency
dense fallback frequency
clarification frequency and value
startup latency
mean/p50/p95 response latency
```

Hidden targets, hidden simulator facts, and benchmark labels stay evaluator-side. They must never influence Agent preprocessing, retrieval, ranking, state, or clarification.

After repeated optimization on Manual400, treat it as a dev set and validate changes on public200 or another unseen sample before drawing strong conclusions.

## 16. Design principles for future changes

1. **Precision and recall come from different layers.**
   Structured and trusted facts provide precision; descriptive semantics and dense retrieval recover recall.

2. **Do not make embeddings enforce discrete constraints.**
   Semantic similarity can help understand wording, but final size, numeric, and measurement constraints are structured.

3. **Do not require perfect annotation coverage.**
   Raw text and product embeddings exist specifically to recover facts the canonical layer misses.

4. **Weight evidence according to trust.**
   A `size=10` exact match is not the same type of evidence as `use_case=hiking`.

5. **Always recommend.**
   Early Top-K hits directly improve the competition objective. Clarification is supplementary.

6. **Keep runtime in memory.**
   The frozen catalog is small enough that external databases and vector services are unnecessary for the MVP.

7. **Add complexity only after measurement.**
   BM25/lexical product branches, cross-encoder reranking, hosted LLM rerankers, recursive DP, ANN indexes, and other advanced paths should be justified by benchmark evidence rather than added preemptively.

8. **Keep component boundaries explicit.**
   Product preprocessing, user parsing, retrieval, ranking, session state, clarification, and evaluation are separate responsibilities.

## 17. Current MVP non-goals

The architecture does not currently require:

```text
Postgres
Redis
Pinecone / Milvus / Weaviate
complex ANN indexing
parallel BM25 product retrieval
cross-encoder reranking
hosted LLM reranking
recursive depth-2+ clarification DP
learned posterior model
```

These may be added later only when measured failures justify them and the architecture is intentionally revised.