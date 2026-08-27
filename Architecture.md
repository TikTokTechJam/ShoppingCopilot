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
                          OFFLINE / BUILD TIME

                               catalog.jsonl
                                    |
              +---------------------+----------------------+
              |                     |                      |
              v                     v                      v
       Tier 1 structured      Tier 2 trusted         Tier 3 descriptive
          extraction           annotation              annotation
              |                     |                      |
              | category            | brand                | style
              | price               | color                | feature
              | size labels         | material             | use_case
              | measurements        |                      |
              | package dims        |                      |
              | product dims        |                      |
              | item weight         |                      |
              +---------------------+-----------+----------+
                                                |
                                                v
                                  validation + normalization
                                                |
                                                v
                                      canonical product facts
                                                |
                    +---------------------------+---------------------------+
                    |                                                       |
                    v                                                       v
              LAYER 1 — EXACT                                        LAYER 2 — EMBEDDING
        exact / numeric / canonical                                field-aware product embeddings
                 indexes                                                     |
                    |                                      +------------------+------------------+
                    |                                      |                  |                  |
                    |                                      v                  v                  v
                    |                                 title embedding    style embedding    feature embedding
                    |                                                                           |
                    |                                                                           v
                    |                                                                    use_case embedding
                    |                                                                           |
                    |                                                                           v
                    |                                                              optional cleaned-raw embedding
                    |                                                              (fallback only, low weight)
                    +-----------------------------------+---------------------------------------+
                                                        |
                                                        v
                                           in-memory Agent runtime


                              RUNTIME / USER TURN

 user utterance
      |
      +--> structured parser
      |      price, size, numeric measurements, dimensions
      |
      +--> trusted canonical matcher
      |      category, brand, color, material
      |
      +--> descriptive semantic parser
      |      style, feature, use_case
      |
      `--> residual semantic context
                    |
                    v
             session state merge
                    |
                    v
          Buying / Browsing retrieval
                    |
          +---------+----------------------------------+
          |                                            |
          v                                            v
   Layer 1 exact filtering                    Layer 2 field-aware scoring
   / strong exact evidence                    title / style / feature / use_case
          |                                            |
          +----------------------+---------------------+
                                 |
                                 v
                          candidate ranking
                                 |
                      +----------+----------+
                      |                     |
                      v                     v
                  Top-K recs        optional clarification
                      |                     |
                      +----------+----------+
                                 |
                                 v
                         Agent response contract
```

The central principle is that product knowledge has different reliability levels, while retrieval uses two execution layers:

```text
Layer 1 — exact / structured retrieval
    category, price, size, measurements, brand, color, material

Layer 2 — field-aware semantic retrieval
    title, style, feature, use_case
    + optional cleaned raw text as a low-weight fallback
```

Tier 1, Tier 2, and Tier 3 remain the product-knowledge trust model. The embedding layer is a retrieval mechanism, not a new knowledge tier.

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

The structured extractor should be conservative. If a measurement cannot be typed confidently, leave it unstructured rather than forcing an incorrect structured value. Semantic recall is handled by the field-aware embedding layer where applicable.

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
crewneck          -> crew neck
vneck             -> v neck
quick_dry         -> quick drying
machine_wash      -> machine washable
4_way_stretch     -> four way stretch
```

Canonical annotation values are stored as lowercase natural text with spaces. Stable internal IDs may still use machine-oriented formatting such as `feature:quick_drying`.

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

Attribute-value embeddings are separate from field-aware embeddings. They resolve user phrases to canonical values; they are not a product retrieval index.

## 7. Field-aware product embeddings

The MVP does not create one field-aware embedding from all metadata.

Raw Amazon listings contain noisy seller text, package contents, guarantees, unrelated accessories, SEO phrases, and other language that can distort semantic similarity. Instead, each product receives separate embeddings for the semantic fields where fuzzy matching is useful.

The default product embedding fields are:

```text
title
style
feature
use_case
```

An optional fifth embedding may be built from cleaned raw product text:

```text
cleaned_raw
```

`cleaned_raw` is a recall fallback only and should receive lower ranking weight than curated field embeddings.

### 7.1 Fields that should not receive product embeddings by default

The following fields already have stronger deterministic comparison semantics:

```text
category
price
brand
color
material
size
measurements
dimensions
weight
```

These fields should use exact, canonical, or numeric matching instead of product-level embedding similarity.

Examples:

```text
brand:nike              -> exact canonical match
color:black             -> exact canonical match
material:faux leather   -> exact canonical match
price <= 50             -> numeric comparison
size = xl               -> exact structured comparison
case diameter = 44 mm   -> typed numeric comparison
```

Embedding similarity must not blur important distinctions such as:

```text
nike != adidas
black != navy
leather != faux leather
xl != xxl
44 mm != 46 mm
```

### 7.2 Title embedding

Every product receives a title embedding generated from the normalized product title.

Example input:

```text
dalegem genuine yellow tiger eye stone ring for men women retro vintage quartz crystal gemstone turkish ring jewelry gift
```

The title embedding provides broad product-identity and semantic recall, especially when:

- the user uses product-specific wording;
- a model/product-line phrase is present mainly in the title;
- the query is open-ended or Browsing-oriented;
- a useful concept was not captured by the curated semantic annotations.

Title similarity is a soft signal. It must not override explicit structured or trusted exact constraints.

### 7.3 Style embedding

The style embedding is generated only from the product's normalized `style` values.

Example product facts:

```json
"style": ["retro", "vintage", "handmade"]
```

Embedding input:

```text
retro vintage handmade
```

Example semantic match:

```text
user phrase:     "something old fashioned"
product style:   "retro vintage"
```

The purpose of the style embedding is to match fuzzy aesthetic, fit, cut, silhouette, form, and pattern language without mixing it with unrelated feature or use-case information.

### 7.4 Feature embedding

The feature embedding is generated only from normalized `feature` values.

Example product facts:

```json
"feature": [
  "hypoallergenic",
  "lead free",
  "nickel free",
  "anti tarnish",
  "high polished"
]
```

Embedding input:

```text
hypoallergenic lead free nickel free anti tarnish high polished
```

Example semantic match:

```text
user phrase:       "something that won't irritate sensitive skin"
product features:  "hypoallergenic nickel free"
```

Feature embeddings are intended for fuzzy functional, performance, care, closure, and construction requirements.

### 7.5 Use-case embedding

The use-case embedding is generated only from normalized `use_case` values.

Example product facts:

```json
"use_case": ["running", "gym", "hiking"]
```

Embedding input:

```text
running gym hiking
```

Example semantic match:

```text
user phrase:       "something for trekking"
product use_case:  "hiking"
```

Use-case similarity is always soft. A mismatch must not eliminate a candidate.

### 7.6 Optional cleaned-raw embedding

A fifth embedding may be generated from cleaned source text when benchmark results show that the curated fields miss useful recall.

Possible input sources:

```text
title
selected product-focused feature text
selected product-focused description text
```

The cleaning step should remove obvious retrieval noise when practical, such as:

```text
shipping information
return or money-back guarantees
seller boilerplate
package contents unrelated to the main product
unrelated bundled accessories
duplicate SEO text
comparison-product language
```

The MVP should not require another expensive LLM annotation pass only to produce `cleaned_raw`.

This embedding is optional and should normally receive lower weight than `title`, `style`, `feature`, and `use_case`.

### 7.7 Embedding artifacts

A practical artifact layout is:

```text
data/derived/product_embeddings/
├── title_embeddings.npy
├── style_embeddings.npy
├── feature_embeddings.npy
├── use_case_embeddings.npy
├── cleaned_raw_embeddings.npy        # optional
├── product_embedding_metadata.json
└── manifest.json
```

All matrices must use exactly the same row order.

Example:

```text
row 0 -> B001...
row 1 -> B002...
row 2 -> B003...
```

Metadata must preserve exact row-to-`parent_asin` mapping.

The manifest should record:

```text
embedding model
embedding dimension
normalization
source catalog version
catalog-facts / annotation version
field names
product count
generation configuration
```

### 7.8 Vector normalization

Vectors should be L2-normalized.

For a normalized product-field matrix:

```python
scores = field_embeddings @ query_field_embedding
```

is cosine similarity.

For approximately 50,000 products, exact NumPy matrix multiplication or FAISS `IndexFlatIP` is sufficient. An external vector database is not required.

### 7.9 Empty fields

Products may have empty semantic fields.

Example:

```json
"use_case": []
```

An empty semantic field should contribute no score.

The runtime should keep a presence mask for each field, conceptually:

```text
has_style
has_feature
has_use_case
```

Missing annotation is not negative evidence and must not be treated as a contradiction.

### 7.10 Field-aware query scoring

Semantic query intent should be compared only with the corresponding product field when possible.

Example:

```text
"I want vintage gold earrings under $50 that are lightweight for a wedding"
```

Exact / structured intent:

```text
category = earrings
gold
price <= 50
```

Semantic intent:

```text
style query    = vintage
feature query  = lightweight
use_case query = wedding
```

The runtime generates:

```text
style_query_embedding
feature_query_embedding
use_case_query_embedding
```

and scores:

```text
style_query_embedding   <-> product style embedding
feature_query_embedding <-> product feature embedding
use_case_query_embedding <-> product use_case embedding
```

The title embedding may provide a general supporting score.

This avoids combining all product concepts into one vector and allows each semantic field to be weighted according to the actual user query.

### 7.11 Residual semantic intent

Information already enforced by Layer 1 should not dominate Layer 2 again.

For:

```text
"gold earrings under $50 for a wedding"
```

Layer 1 handles:

```text
earrings
gold
price <= 50
```

The useful remaining semantic intent is primarily:

```text
wedding
```

Layer 2 should therefore focus on ranking the already-valid candidates by the unresolved semantic preference rather than repeatedly rewarding all candidates for facts already used in filtering.

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
5. retain unresolved semantic context for field-aware dense scoring
```

Priority is:

```text
structured parse
> exact/normalized canonical match
> high-confidence semantic fallback
> unresolved field-aware semantic context
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

Tier 3 descriptive exact/semantic evidence
    -> soft ranking evidence

Layer 2 field-aware embedding similarity
    -> soft semantic ranking and recall
```

A useful mental model is:

```text
category exact / numeric budget / explicit size or measurement    VERY HIGH
brand exact                                                       VERY HIGH
color/material exact                                              HIGH
style/feature/use_case exact or field similarity                  MEDIUM
title semantic similarity                                           MEDIUM / SUPPORTING
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
score descriptive exact matches
    |
    v
apply field-aware semantic scores among viable candidates
    |   title / style / feature / use_case
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

If controlled relaxation still cannot produce a useful pool, broader field-aware semantic retrieval can recover candidates while recording the fallback in provenance.

### 10.3 Browsing mode

Browsing is recall-first.

```text
current/session semantic context
        |
        +--> title query embedding
        |
        +--> style query embedding
        |
        +--> feature query embedding
        |
        `--> use_case query embedding
                |
                v
field-aware exact dense scoring over 50k
                |
                v
broad candidate pool / candidate union
                |
                v
Tier 2 exact boosts + Tier 3 field-aware scores
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
  "title_similarity": 0.42,
  "style_similarity": 0.31,
  "feature_similarity": 0.91,
  "use_case_similarity": 0.84,
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
field embedding matrices + shared row metadata when valid
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
field-aware semantic fallback frequency
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
   Field-aware semantic embeddings provide recall when exact canonical matching is insufficient. Optional cleaned-raw embeddings may be used only as a low-weight fallback when benchmark evidence justifies them.

4. **Weight evidence according to trust.**
   A `size=10` exact match is not the same type of evidence as `use_case=hiking`.

5. **Keep semantic embeddings field-aware.**
   Product-level semantic similarity is computed separately for `title`, `style`, `feature`, and `use_case`. Do not collapse the full Amazon record into one primary embedding, because noisy seller text and unrelated metadata can distort retrieval.

6. **Always recommend.**
   Early Top-K hits directly improve the competition objective. Clarification is supplementary.

7. **Keep runtime in memory.**
   The frozen catalog is small enough that external databases and vector services are unnecessary for the MVP.

8. **Add complexity only after measurement.**
   BM25/lexical product branches, cross-encoder reranking, hosted LLM rerankers, recursive DP, ANN indexes, and other advanced paths should be justified by benchmark evidence rather than added preemptively.

9. **Keep component boundaries explicit.**
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