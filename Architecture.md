# Shopping Copilot architecture

This document describes the implementation on the current main branch. It is
an implementation snapshot, not a roadmap. The Python source is authoritative
when behavior and this document disagree.

## 1. Preprocessing and runtime flows

The system has an offline preparation flow and a per-turn runtime flow. The
offline flow creates the artifacts that the runtime reads; it is not performed
inside Agent.respond.

### Offline preprocessing flow

    data/catalog.jsonl
            |
            v
    V5 single-attribute annotation jobs
    annotation runner + attribute-specific prompts
            |
            +--> annotations/v5/category.jsonl
            +--> annotations/v5/brand.jsonl
            +--> annotations/v5/color.jsonl
            +--> annotations/v5/material.jsonl
            +--> annotations/v5/feature.jsonl
            +--> annotations/v5/use_case.jsonl
            |
            v
    scripts.aggregate_v5_annotations
    catalog order + catalog price + six V5 fields
            |
            v
    data/derived/annotations/v5/annotations.jsonl
            |
            +------------------------------+
            |                              |
            v                              v
    optional catalog-facts build       scripts.build_attribute_dictionary
    scripts.build_catalog_facts        exact canonical registry
            |                              |
            v                              v
    catalog_facts/catalog_facts.jsonl  dictionary/
                                         canonical_values.json
                                         normalized_lookup.json
                                         manifest.json
                                               |
                                               v
                                  scripts.build_v5_attribute_embeddings
                                  local BGE-small, normalized vectors
                                               |
                                               v
                                  dictionary/attribute_embeddings/
                                  category, color, material, style,
                                  feature, and use_case matrices

The current runtime normally reads the aggregated V5 annotations and generated
dictionary. The optional catalog-facts artifact is a compatible downstream
representation. The semantic embedding build is separate from annotation and
does not generate product vectors.

### Per-turn runtime flow

    USER TURN
        |
        v
    Agent.respond
        |
        v
    existing utterance processing
        |
        +-----------------------------+
        |                             |
        v                             v
    Layer 1 parsed state          Layer 2 semantic query
    structured / canonical        independent user phrases
        |                             |
        |                             v
        |                         remove shared stopwords
        |                             |
        |                             v
        |                         1/2/3-gram phrases
        |                             |
        |          +------------------+------------------+
        |          |                  |                  |
        |          v                  v                  v
        |      category matrix   color matrix       material matrix
        |          |                  |                  |
        |          +------------------+------------------+
        |                             |
        |                             v
        |                       style / feature /
        |                       use_case matrices
        |                             |
        |                             v
        |                  BGE cosine attribute matches
        |                             |
        |                             v
        |                       threshold >= 0.80
        |                             |
        |                             v
        |                       Layer 2 evidence
        |                    value + similarity score
        |                             |
        +-----------------------------+
                                      |
                                      v
                         SessionManager keeps both states
                                      |
                                      v
                         ProductRetriever.retrieve
                                      |
             +------------------------+------------------------+
             |                        |                        |
             v                        v                        v
       Layer 1 posting          Layer 2 value-posting       BM25 FTS5
       list score               list score                  text score
             |                        |                        |
             +------------------------+------------------------+
                                      |
                                      v
                         hard price/exclusion filters
                                      |
                                      v
                         shared hybrid score + rating tie-break
                                      |
                                      v
                         rank candidates and return Top 10
                                      |
                                      +--> ClarificationPolicy.choose
                                      +--> evaluator / debug UI

Layer 1 and Layer 2 are independent: exact matches are not removed from the
semantic phrase stream. Layer 2 is the BGE canonical-attribute path; it finds
canonical values first and then uses their per-attribute product posting lists.
It is not a separate Agent or a second asking strategy.

## 2. Runtime ownership by file

| Stage | Implementation |
| --- | --- |
| Public Agent contract | starter/agent.py: Agent.reset and Agent.respond |
| Intent routing | starter/routing/intent_router.py: TwoPhaseIntentRouter, CascadingIntentRouter, LexicalIntentRouter |
| Routing signal definitions | starter/routing/lexicon.py: SIGNALS and thresholds |
| Constraint extraction | starter/routing/constraints.py: extract_constraints and _extract_dictionary_constraints |
| Exact dictionary | dictionary/registry.py: AttributeDictionary, exact_match, normalize_text |
| Attribute semantic matching | dictionary/registry.py: semantic_match_ngrams; dictionary/semantic.py: BGE loader |
| Mutable session state | starter/session.py: SessionState, SessionManager, merge_constraints, merge_semantic_constraints |
| Question policy | starter/clarification.py: ClarificationPolicy.choose |
| Turn horizon/promotion helpers | starter/followup.py: MAX_TURNS, utility, fill_to_top_k |
| Product catalog/fact indexes | starter/retrieval.py: ProductRetriever._load_catalog |
| Structured and semantic retrieval | starter/retrieval.py: ProductRetriever.retrieve and _semantic_scores |
| Optional product-vector retrieval | starter/retrieval.py: _load_layer2, _dense_scores, _query_embedding |
| BM25 retrieval | starter/bm25.py: BM25Index |
| Hard benchmark | evaluator/hard_evaluator.py: validate_sessions, Manual400SessionRunner, evaluate |
| Public evaluator | evaluator/local_evaluator.py |
| Interactive evaluator UI | evaluator/debug_web.py and evaluator/debug_web_ui/ |
| Evaluator Agent construction | evaluator/agent_factory.py: build_evaluator_agent |

## 3. Product data and startup artifacts

The raw product universe is data/catalog.jsonl. Product order and parent ASINs
come from that file.

The default fact search order in ProductRetriever is:

    data/derived/annotations/v5/annotations.jsonl
    data/derived/annotations/v4/annotations.jsonl
    data/derived/catalog_facts/catalog_facts.jsonl
    data/derived/annotations/v2/annotations.jsonl
    data/derived/annotations/v1/annotations.jsonl
    data/derived/facts/facts.jsonl
    data/facts.jsonl

The first existing file is selected unless facts_path is explicitly supplied.
Successful annotation rows are loaded. Raw catalog taxonomy supplies safe
category facts, and annotation facts are merged with them. An annotated usable
price takes precedence over the raw catalog price.

The generated V5 dictionary is required for categorical extraction:

    data/derived/annotations/v5/dictionary/
        canonical_values.json
        normalized_lookup.json
        manifest.json

The dictionary contains these seven fields:

    category, brand, color, material, style, feature, use_case

Price and size remain structured runtime fields. Brand is exact-only in the
semantic matcher. The dictionary loader fails closed to an extraction error if
the generated V5 dictionary is absent; there is no active legacy
CANONICAL_VOCAB fallback.

## 4. Intent routing

Agent.respond uses TwoPhaseIntentRouter when no custom router is injected.
The router first extracts constraints and counts populated fields, excluding
category from the tag count. Two or more populated fields normally classify the
message as BUYING. A lexical browsing veto at confidence 0.70 can prevent that
decision.

Messages not settled by the tag phase use the lexical signal ledger. The
default terminal decision is BROWSING when confidence remains below 0.70.
The optional cascading backend can be used only for uncertain lexical results;
it is not configured by the default evaluator Agent.

IntentResult carries the selected intent, confidence, margin, tier, tags,
signals, and extracted constraints. Agent stores only BUYING or BROWSING as its
session mode. SessionIntentTracker keeps the initial mode sticky and permits a
later flip only when the margin reaches the configured flip margin of 0.80.
A BUYING-to-BROWSING flip also needs an explicit exploring signal.

The lexical ledger has the following weighted signals:

| Signal | Weight | Direction |
| --- | ---: | --- |
| budget | 1.50 | buying |
| brand | 1.20 | buying |
| size | 1.00 | buying |
| color | 1.00 | buying |
| material | 1.00 | buying |
| feature | 1.00 | buying |
| use_case | 0.60 | buying |
| style | 0.70 | buying |
| request verb | 0.45 | buying evidence required |
| undecided | 1.40 | browsing |
| explore verb | 1.10 | browsing |
| option seeking | 0.90 | browsing |
| vague head | 1.00 | browsing |
| vague quality | 0.50 | browsing |
| hedged language | 0.60 | browsing |

The main lexical constants are bias 0.35, buying scale 1.25, browsing scale
1.15, logistic k 1.60, buying tag threshold 2, browsing-veto confidence 0.70,
decision confidence 0.70, reranker acceptance confidence 0.80, and flip margin
0.80. Vague signals are suppressed when hard evidence is present. Request verbs
need buying evidence.

## 5. Constraint extraction

The single public extraction entry point is
starter.routing.constraints.extract_constraints.

It returns CanonicalShoppingConstraints with two deliberately separate
representations:

1. structured constraints: price, size, and exact dictionary claims;
2. semantic_constraints: accepted BGE canonical values plus evidence and
   similarity scores.

The semantic pass is independent of the exact pass. Exact matches do not remove
text from semantic processing. The semantic query uses the shared stopword
policy, creates deterministic one-, two-, and three-token phrases, and keeps
every canonical match at or above 0.80. It searches category, color, material,
style, feature, and use_case matrices. Brand has no attribute embedding and
remains exact-only.

Exact matching normalizes Unicode NFKC, case-folds, changes underscores and
hyphens to spaces, removes apostrophes inside words, and collapses whitespace.
Dictionary phrases are matched on token boundaries longest-first. A consumed
span is not reused by a shorter overlapping phrase.

Explicit local context restricts an exact match to the requested attribute:

| Context | Attribute |
| --- | --- |
| brand, from, made by | brand |
| color, colour | color |
| made of, made from, fabric | material |
| style, fit | style |
| feature, features | feature |
| for, use for, good for | use_case |

If the requested contextual attribute has no candidate, the value is left
unresolved rather than assigned to another attribute by frequency. Without
explicit context, an ambiguous surface is accepted only when one candidate
passes the frequency share and ratio thresholds: top share 0.75 and count ratio
3.0. Common single-token brand collisions for find, it, make, and on are
suppressed unless explicitly scoped.

Price parsing supports upper bounds, lower bounds, ranges, around/about bands,
currency words, and shopping numbers without a currency symbol. Numeric size
phrases, measurement units, and likely years from 1900 through 2099 are guarded
so they do not become price claims.

The semantic stopword policy is owned by
dictionary.registry.SEMANTIC_QUERY_STOPWORDS and is reused by BM25. It
intentionally keeps product-relevant words such as wear, work, fit, dry, and
id.

## 6. Session state, corrections, and overrides

SessionState is held in memory by SessionManager. It includes:

- mode and anonymized profile;
- structured constraints and evidence;
- semantic constraints and semantic evidence;
- last asked attribute and per-cycle ask counts;
- no-preference attributes and clarification state;
- last recommendations and excluded recommendations;
- chronological messages/query text;
- turn and last override metadata.

Categorical fields accumulate by default. Price minimums refine upward and
maximums refine downward. Explicit correction markers can replace populated
fields. The structured and semantic stores are merged independently.

detect_override_kind recognizes:

- FULL_GOAL: an explicit new goal or strong reset;
- PREFERENCE: an explicit preference correction with extracted facts;
- NONE: ordinary clarification, marker-only text, or no usable fact.

FULL_GOAL clears the goal, mode, constraints, semantic state, exclusions, and
messages while retaining the profile. PREFERENCE keeps the category and
explicit budget but resets the replaced preference state. Before an ordinary
turn, the previous recommendations are promoted to the current-goal exclusion
set. A valid override resets the relevant exclusion behavior.

Answers to an asked attribute can be re-read with that attribute as scope.
No-preference replies and the evaluator's generic clarification filler are
treated as conversation metadata and are not passed as normal product claims.

## 7. Clarification strategy

ClarificationPolicy.choose receives the current candidate list and structured
constraints. It does not build a static decision tree or run an exhaustive DP.
It scores each not-yet-used attribute using candidate coverage, value
distribution/Gini split, mode prior, optional profile affinity, and turn
horizon.

The supported question attributes are category, material, color, size, style,
brand, budget, feature, use_case, and the cycle marker other. The normal
question order is not a fixed sequence; the policy chooses the highest utility
that clears its floor.

Mode priors are:

| Attribute | BUYING | BROWSING |
| --- | ---: | ---: |
| material | 1.00 | 0.84 |
| feature | 0.98 | 0.96 |
| color | 0.92 | 0.92 |
| size | 0.90 | 0.76 |
| style | 0.78 | 1.00 |
| use_case | 0.74 | 0.98 |
| brand | 0.68 | 0.64 |
| budget | 0.66 | 0.62 |
| category | 0.60 | 0.90 |

The prior ceiling is 0.90, the historical split floor is 0.035, and the
utility floor is derived from that floor and follow-up utility. Coverage below
20 percent, fewer than two values, or Gini below 0.10 makes an attribute
unaskable. The horizon is zero on turn 10. If no normal field remains, Agent
asks other; a non-answer to other stops clarification.

## 8. Retrieval and ranking

ProductRetriever builds one in-memory ProductRecord per catalog ASIN and
per-attribute posting lists:

    inverted_index[field][normalized_value] -> set[parent_asin]

The candidate universe is the full catalog unless a hard budget is active.
Recommendation exclusions are applied after budget eligibility and before
ranking.

### Structured Layer 1

The structured score is an accumulated weighted sum for matched product facts:

| Field | Weight |
| --- | ---: |
| category | 0.70 |
| price | 1.50 |
| brand | 3.00 |
| size | 0.80 |
| color | 1.00 |
| material | 1.20 |
| style | 0.50 |
| feature | 0.50 |
| use_case | 0.50 |

An exact claim contributes its field weight. A semantic attribute claim can
contribute the field weight multiplied by its retained similarity when the
matching product posting list contains that value. Multiple matched fields add;
there is no normalization or cap at 1.0.

### Semantic attribute path

The active semantic path uses BAAI/bge-small-en-v1.5, dimension 384, for the
short canonical attribute matrices under:

    data/derived/annotations/v5/dictionary/attribute_embeddings/

The matrices cover category, color, material, style, feature, and use_case.
Brand is not embedded. The loader is local-only and keeps exact matching
available if the local BGE model cannot be loaded.

Each accepted semantic canonical ID is looked up through the corresponding
product posting list. Its similarity is retained in semantic evidence and
added to the product's semantic score. Therefore this path searches the
products containing accepted canonical values; it is not a full-product
sentence-vector search.

### Optional product-vector compatibility path

starter/retrieval.py still contains compatibility support for a direct
Layer2EmbeddingIndex and older product embedding matrices. If a compatible
query_encoder is explicitly injected, it can load an artifact from:

    data/derived/product_embeddings
    data/derived/layer2_embeddings
    data/layer2_embeddings
    layer2_embeddings

The direct path validates model compatibility and vector dimension, then scores
the eligible catalog with _dense_scores. The default evaluator factory does
not inject a product query encoder and does not discover or enable the retired
Jina product path. No Jina model is part of the default Agent retrieval flow.

### BM25 product-text path

BM25Index builds an in-memory SQLite FTS5 table over title, categories,
features, details, store, and description. Parent ASIN is unindexed. It reuses
semantic_query_tokens and up to 40 unique terms. FTS fields have weights:

    ASIN 0.0, title 6.0, categories 4.0, features 2.5,
    details 2.5, store 1.5, description 1.0

SQLite's lower-is-better negative rank is converted to a non-negative,
higher-is-better BM25 score. BM25 is built eagerly at Agent startup and logs
preprocessing progress.

### Shared final score and ordering

The current mode table is identical for both modes:

    BUYING  = 1.00 * structured + 1.00 * dense + 0.20 * bm25
    BROWSING = 1.00 * structured + 1.00 * dense + 0.20 * bm25

The implementation uses _final_score as the shared base scorer. Here, dense
means the semantic attribute score when semantic constraints exist; it can mean
the optional direct product-vector score when that compatibility path is
active.

The ordering key adds a secondary catalog-rating bonus:

    base final score + rating_weight * normalized_rating

with rating weight 0.02 by default and 0.15 for a critical user whose prior
rating is below 3.5. Unusable ratings are neutral at 0.5 on a five-point
scale. Catalog order is the deterministic last tie-break. Candidate.score
stores the base final score; Candidate.ranking_score stores the score including
the rating bonus.

One current branch detail matters: when no dense/semantic score exists but
structured constraints do exist, the retrieval code ranks positive structured
matches and pads them by rating. BM25 is still computed but is not included in
that branch's ordering. When a dense/semantic score exists, the full eligible
pool is ranked by the shared hybrid score.

## 9. Hard eligibility and recommendations

If either price bound is active, a product is eligible only when its known
price satisfies all active bounds. A null price is excluded. Without a budget,
all catalog products are eligible.

Previously recommended products are excluded from the next ordinary goal turn.
The exclusion is applied before hybrid ranking. If the normal result is shorter
than requested Top K, Agent calls a relaxed backfill retrieval that can disable
the budget and omit the previous-recommendation exclusion. Backfill exists to
return valid catalog ASINs rather than leave a short response.

Agent retrieves up to 100 candidates and then uses fill_to_top_k to return the
requested number of unique, valid parent ASINs. Its public response contains
only message, ask_attribute, recommendations, and usage. Recommendation scores
are not required by the public Agent contract.

## 10. Evaluators and debug UI

### Hard evaluator

evaluator/hard_evaluator.py defaults to:

    catalog:  data/catalog.jsonl
    sessions: data/derived/gptannotation/sessions.jsonl
    output:   results_gptannotation.json

The fixed benchmark contains exactly 400 sessions:

    buying 160
    browsing 160
    intent_override 60
    boundary 20

validate_sessions checks unique samples and targets, catalog membership,
two-to-four hidden facts, evidence fields, initial-fact rules, and override
turn/fact rules.

Manual400SessionRunner executes one session turn at a time for at most 10
turns. It simulates customer replies from the committed hidden facts and
override message. The target ASIN is kept only in evaluator-side state and is
never given to Agent.respond. A target in Top 10 before an intent override is
recorded as pre_override_hit but is not scoreable. A scoreable hit ends the
session; an unhit session ends at turn 10.

Agent responses are strictly validated. The evaluator keeps only unique,
known catalog parent ASINs from the first 10 recommendations. HitRate@10 is
the fraction of scoreable hits; MRR is the mean reciprocal best rank; MTTC is
the mean first hit turn, using 11 for a miss. Efficiency is:

    clamp((11 - MTTC) / 10, 0, 1)

TechnicalScore is:

    0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency

The batch evaluator can print progress and writes detailed result JSON. Its
debug mode shuffles sessions with an optional seed and can inspect a limited
number of sessions.

### Public evaluator

evaluator/local_evaluator.py uses data/public_set.jsonl, also has a ten-turn
limit and Top 10 contract, simulates public customer behavior, and reports
similar hit, MRR, MTTC, efficiency, and technical-score metrics. It is a
separate public-set harness from the fixed 400-session hard evaluator.

### Local web debugger

evaluator/debug_web.py reuses Manual400SessionRunner and the hard evaluator
ranking/debug helpers. It exposes local HTTP endpoints for loading a random or
named session, advancing one turn, running to the end, and reading state.
The browser UI displays the same structured, semantic, BM25, hybrid, target,
and hard-score diagnostics; it does not change Agent or evaluator behavior.

## 11. Configurable versus hardcoded

### Configurable through constructors or CLI

- Agent catalog_path, facts_path, optional product embedding paths, metadata,
  query encoder, profile use, optional Layer2 artifact directory, optional
  Layer2 weights, retriever, and router;
- ProductRetriever fact/embedding paths, injected query encoder, direct Layer2
  directory, and direct Layer2 view weights;
- evaluator catalog, sessions, output, profile ablation, strictness, progress,
  debug seed, and debug-session count;
- local debug web catalog, sessions, port, and seed;
- ClarificationPolicy candidate/provider/profile callback inputs.

### Hardcoded defaults and policies

- V5 dictionary location and its seven-field contract;
- BGE model identity/path convention and 384 dimension;
- semantic threshold 0.80 and one-/two-/three-gram search;
- all routing signal weights and thresholds;
- all structured field weights;
- identical BUYING/BROWSING hybrid weights;
- BM25 fields, token limit, and field weights;
- price eligibility, exclusion, backfill, rating, and tie-breaking rules;
- clarification questions, priors, floors, and ten-turn horizon;
- hard evaluator scenario counts, score formulas, and default paths.

## 12. Known limitations and code-level risks

These are observations about the current code, not proposed changes:

1. Mode-specific hybrid ranking is not currently mode-specific: BUYING and
   BROWSING use the same structured, dense, and BM25 weights.
2. If semantic/product dense scores are absent while structured constraints are
   present, BM25 does not affect the ordering even though it was calculated.
3. Candidate.score excludes the rating bonus that participates in ordering, so
   a score shown to a caller may not fully explain a rating-based tie decision.
4. ProductRetriever contains optional direct product-vector compatibility code,
   but the default evaluator Agent does not configure it. This can make
   “Layer 2” terminology ambiguous: active BGE semantic attributes and
   optional direct product vectors are different paths.
5. The semantic attribute artifact currently has no useful style rows when its
   source dictionary has no style values, so style semantic matching may have
   no matches.
6. BGE loading is optional at extraction startup. If the local model is absent,
   exact dictionary extraction can still work, but semantic attribute scores
   are unavailable.
7. BM25 construction is eager and scans the entire catalog on every Agent
   construction, so startup cost is visible before the first turn.
8. The initial turn can cause extraction in Agent and again inside the
   TwoPhaseIntentRouter because routing uses its own constraint extraction.

## 13. Intentional non-goals of the current implementation

The current runtime does not include BM25 as a replacement for the structured
or semantic paths, product reranking by a cross-encoder, an LLM query-rewrite
stage, semantic aliases/synonyms, stemming, a vector database, or a network
model download during ordinary evaluator startup. Product retrieval does not
receive hidden evaluator targets or hidden target facts.
