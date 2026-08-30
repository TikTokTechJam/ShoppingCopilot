# Query Text Normalization Audit

> **Point-in-time audit:** repository paths, line numbers, commit state, and the
> statement that `Architecture.md` was unchanged describe the original audit at
> commit `b25dce3`. The current implementation guide is
> [`../Architecture.md`](../Architecture.md). The normalization findings and
> proposed regression cases below remain open unless a later change explicitly
> closes them.

## Repository State

- Repository: `/Users/vietanh/Desktop/TiktokJam/ShoppingCopilot`
- Branch: `main`.
- HEAD: `b25dce3cc6d243f4a6ddd3cf18a0a6e926c177b2`, the merge of PR #69 (fix-negative-respond-user).
- `origin/main` relationship: identical to HEAD (0 commits behind, 0 ahead).
- Working tree at audit start: clean; status was `## main...origin/main`.
- `AGENTS.md` is absent in this checkout. It was removed by commit `41f6c26`.
- `Architecture.md` was read completely (1,184 lines) and was not modified.
- No runtime/source files, ranking weights, embedding artifacts, or architecture documents were changed.

Ignored/local generated material includes Python/pytest caches, the local model, and derived annotation/dictionary artifacts. No untracked changes were reported at audit start. No `state.json` exists in the current repository.

### Recent relevant commits

| Commit | Files/modules affected | Architectural behavior changed |
|---|---|---|
| `b25dce3` | Merge PR #69 | Current HEAD; includes the negative/no-preference response change. |
| `4b9a1f7` | `starter/agent.py`, routing constraints/lexicon, tests | Pending clarification replies can be treated as no preference, skip extraction, and stay out of the semantic transcript. |
| `3fd8a4d` | Merge PR #68 | Merges current debug/evaluator parity work. |
| `d7e4c43` | constraints, retrieval, evaluator/debug paths | Separates structured claims from semantic claims; Layer 2 gets independent semantic text and exact claims are not removed from it. |
| `e8797a9` | session, retrieval, evaluator/debug paths | Separates structured session state and semantic session state; retrieval consumes semantic evidence separately. |
| `226ffbc` | debug web, hard evaluator | Aligns debug UI with hard evaluator flow. |
| `5c8a1cc` | dictionary, V5 annotations, BGE attribute retrieval | Makes BGE canonical attribute retrieval active on the V5 annotation/dictionary path. |
| `41f6c26` | agent factory, evaluator, old Jina scripts/artifacts/docs/tests | Retires whole-product Jina retrieval and removes old Jina/product-embedding utilities, including `AGENTS.md`. |
| `5124209` | V5 metadata loading | Fixes V5 attribute metadata loading. |
| `d519a18` | `dictionary/registry.py`, `dictionary/semantic.py`, constraints, retrieval | Adds Layer 2 attribute n-gram matching against canonical attribute embedding matrices. |
| `a065b33` | V5 dictionary/embedding build | Excludes brand from V5 semantic matrices; brand remains exact-oriented. |
| `38f302a` | V5 per-attribute embedding builder | Adds one matrix per canonical semantic attribute field. |
| `691354a`, `0624cd1`, `3a81cb3`, `e21430f` | Layer 1/V5 dictionary and annotation merges | Routes V5 facts through Layer 1 and merges the BGE attribute-semantic work. |

The important refactor sequence is d519a18 → 41f6c26 → e8797a9 → d7e4c43 → 4b9a1f7. The current false match is in canonical attribute n-gram semantics, not in the retired whole-product Jina path or an old separate product reranker.

## Executive Summary

The issue is real, but it is not only missing stopwords. The shared normalizer in `dictionary/registry.py:39-73` applies NFKC and casefolding, then deletes internal apostrophes. Thus both ASCII and curly `I'd` become `id`. The semantic constraint path then creates the one-gram `id` from the full normalized message and searches every semantic attribute matrix with a default threshold of 0.80.

There is no exact one-token canonical value `id`. There are, however, valid longer values containing it, including `storing id`, `id case`, and `id window`. Their BGE similarities to the artifact gram are above threshold, so they become false semantic constraints.

~~~text
I'd
  -> shared NFKC/casefold + apostrophe deletion
  -> id
  -> semantic text contains one-gram id
  -> BGE searches all attribute fields
  -> longer canonical phrases score >= 0.80
  -> false storing_id / id_case / id_window claims
~~~

The active exact matcher is token/phrase bounded and does not use arbitrary substring matching. The broader failure is the combination of destructive contraction normalization, an incomplete residual stopword list, all-text overlapping 1/2/3-gram generation, cross-field semantic search, and no negative-polarity representation.

The safest direction is separate text views: preserve raw text for routing, overrides, negation, and session history; use a controlled exact-matching view; and feed semantic attribute matching deliberate candidate phrases rather than every arbitrary n-gram. This audit does not implement that direction.

## Current User Query Pipeline

The active public path is `starter/agent.py:78-195`.

~~~text
raw user message
    |
    v
Agent.respond(user_message, session_id)
    |
    +--> SessionManager.get()                         session.py:303-311
    +--> pending no-preference check on raw text       agent.py:91-100
    |       (conditional: may skip extraction)
    +--> extract_constraints(raw message)              agent.py:101-106
    |       + known phrase rewrites                    constraints.py:251-255
    |       + price/size structured parsing             constraints.py:839-905
    |       + exact token phrase matching              constraints.py:533-605
    |       + normalized semantic text                 constraints.py:769-777
    |       + BGE 1/2/3-gram matching                  registry.py:483-565
    +--> structured_delta / semantic_delta split       agent.py:108-110
    +--> raw correction/override checks                session.py:209-293
    +--> sticky mode classification if unset           agent.py:116-121
    |       TwoPhaseIntentRouter.classify; extraction runs again
    +--> reset goal/preference or promote old ASINs    agent.py:123-145
    +--> merge independent session state               session.py:446-503
    +--> record raw message                            session.py:431-444
    |       state.query_text = raw messages joined with newlines
    +--> ProductRetriever.retrieve(...)                agent.py:155-163
            + canonical semantic product-fact score    retrieval.py:539-594
            + structured exact fact score              retrieval.py:751-915
            + dormant direct dense path if artifacts   retrieval.py:301-329
    +--> exclusions, top-k, clarification              agent.py:165-189
    v
response payload: message, ask_attribute, recommendations
~~~

| Step | File/function | Input | Output/state mutation | Conditional? | Original preserved? |
|---|---|---|---|---|---|
| Entry/session | `agent.py:78-91`, `session.py:303-311` | Message, session ID | Retrieves/creates `SessionState` | Every turn | Yes |
| No preference | `agent.py:91-100` | Raw text, pending clarification | Empty delta | Pending clarification only | Yes |
| Extraction | `constraints.py:969-996` | Raw text | Canonical constraints and evidence | Every non-skipped turn | Raw evidence retained |
| Structured exact | `constraints.py:839-966` | Raw text | Layer 1 values/evidence, price/size | Inside extraction | Match evidence retained |
| Semantic | `constraints.py:769-792, 906-953` | Normalized full message | Layer 2 canonical values/evidence | Registry available | Raw remains outside |
| Routing/override | `intent_router.py:440-490`, `session.py:209-293` | Raw message, delta, state | Mode/correction/reset/exclusion decisions | Varies | Yes |
| Merge/history | `session.py:446-503`, `431-444` | Deltas/raw message | In-memory state and raw transcript | Every recorded turn | Yes |
| Retrieval | `retrieval.py:751-915` | Catalog facts, constraints, semantic evidence | Candidate scores | Every retrieval | Raw only for dormant dense |
| Response state | `agent.py:165-189` | Candidates | Top-k, asked attributes, recommendation history | Every response | Not normalized |

## Text Normalization Functions

| File | Function | Purpose | Transformation |
|---|---|---|---|
| `dictionary/registry.py:39-73` | `normalize_text` | Shared query/canonical normalization | NFKC, casefold, alphanumeric retention, internal apostrophe deletion, punctuation/hyphen/underscore/whitespace to spaces, compact/strip. |
| `dictionary/registry.py:76-86` | `canonical_id` | Stable canonical IDs | Normalizes then replaces spaces with underscores. |
| `constraints.py:251-255` | `_normalise_known_phrases` | Targeted phrase rewrites | Examples: machine wash → feature machine washable; non-slip design → feature non slip; no general contraction expansion. |
| `constraints.py:608-642` | `_utterance_tokens` | Exact/context tokenization | Allows internal apostrophes, then normalizes each token; `I'd` becomes one token `id`. |
| `constraints.py:533-605` | `_dictionary_surface_matches` | Exact phrase matcher | Normalized token sequence, longest-first, non-overlapping. |
| `constraints.py:769-777` | `_semantic_text` | Layer 2 input text | Normalizes complete message, removes only residual stopwords, rejoins. Exact spans are not removed. |
| `constraints.py:780-792` | `_semantic_ngrams` | Injected semantic input | Deduplicated contiguous 1/2/3-grams after filtering. |
| `registry.py:483-565` | `semantic_match_ngrams` | Active Layer 2 matching | Re-normalizes, filters stopwords, embeds each n-gram, searches each matrix. |
| `registry.py:601-618` | `_prepare_query` | Vector preparation | L2-normalizes vector; no text change. |
| `semantic.py:63-92` | BGE setup | Query/document encoding | Local bge-small-en-v1.5, 384 dimensions, no task or prompt names. |
| `annotation/schema.py:175-201` | Annotation normalization/canonicalization | Product-side values | Separate NFKC/casefold/punctuation policy; not user query cleanup. |
| `intent_router.py:172-179` | `LexicalIntentRouter.is_contentless` | Filler/contentless routing | Lowercase + ASCII-only `[a-z0-9']+` regex; separate from semantic stopwords. |
| `session.py:209-293` | Correction/override checks | Conversation control | Raw regex/lexicon checks for markers such as actually, instead, rather, not, changed, switch. |

No stemming, lemmatization, or singular/plural normalization was found.

## Exact Reproduction of the I'd -> id Problem

The production path gives:

~~~text
raw ASCII:       "I'd"
normalize_text:   "id"
utterance token:  ["id"]
semantic text:    "id"
semantic ngrams:  ["id"]
~~~

The curly form gives the same dictionary-path result:

~~~text
raw Unicode:      "I’d"
normalize_text:   "id"
utterance token:  ["id"]
semantic text:    "id"
semantic ngrams:  ["id"]
~~~

This is not production tokenization into `['i', 'd']`. Apostrophes are allowed inside the token and then deleted by the shared normalizer.

A read-only call to the real extractor for `"I'd"` produced no exact one-token `id` match, but did produce:

~~~text
use_case: storing id   cosine 0.853645
category:  id case     cosine 0.843887
feature:   id window   cosine 0.839638
~~~

For `"I'd like something for hiking"`, semantic text is `id hiking` and grams are `id`, `hiking`, and `id hiking`. Exact `use_case:hiking` is accompanied by the same id-driven false claims.

## Why `id` Matches

The current dictionary has 47,187 canonical values. Exact lookup for one-token `id` returned no value. A scan found 43 canonical values containing the token, including:

| Field | Canonical ID | Natural value | Source count |
|---|---|---|---:|
| use_case | `storing_id` | storing id | 1 |
| category | `id_case` | id case | 2 |
| feature | `id_window` | id window | 48 |
| other fields | various | id bracelet, id holder, personal identification, etc. | remaining |

`semantic_match_ngrams` does not require query/candidate token counts to match. It embeds `id`, searches all semantic matrices, takes up to one top result per attribute for a query gram, and accepts scores at or above 0.80. Thus a one-token contraction artifact retrieves longer canonical phrases.

This is semantic embedding false positive behavior, not exact substring matching. Exact checks did not match `id` inside `paid`, `idea`, or `identity`.

## Current Tokenization Behavior

`_utterance_tokens` in `starter/routing/constraints.py:608-642` allows apostrophes within a token and normalizes afterward:

| Raw | Utterance token(s) | Semantic text |
|---|---|---|
| `I'd` | [`id`] | `id` |
| `I’d` | [`id`] | `id` |
| `I'm` | [`im`] | `im` |
| `I've` | [`ive`] | `ive` |
| `don't` | [`dont`] | `dont` |
| `machine-washable` | [machine, washable] | `machine washable` |
| `non-slip` | [non, slip] | `non slip` |
| `UV protection` | [uv, protection] | `uv protection` |

The semantic path emits every contiguous 1/2/3-gram after filtering. The lexical intent tokenizer is different: `re.findall(r"[a-z0-9']+", text.lower())` recognizes ASCII apostrophes but not curly apostrophes. ASCII `I'm` can be compared as `im`; curly `I’m` may split into `I` and `m`. ASCII `I'd` becomes `id` for filler comparison, while curly `I’d` may split into `I` and `d`.

## Current Stopword Handling

The semantic residual list in `starter/routing/constraints.py:447-455` is:

~~~text
a an and are as at be but by do for from i in is it me my of on or please
some that the this to want with would you looking need like something show find
under below less than more over between around about within
~~~

It is used by `_semantic_text`, `_semantic_ngrams`, and the registry n-gram matcher. It runs after shared normalization and before semantic n-gram generation. It does not affect exact matching, raw routing, session history, product annotations, or the dormant direct product dense path.

It does not include `id`, `im`, `ive`, `dont`, `doesnt`, `cant`, `wont`, `isnt`, `thats`, `youre`, `rather`, `prefer`, `actually`, `can`, `without`, `not`, `no`, `avoid`, or `except`. The separate `CONVERSATIONAL_FILLER` list in `routing/lexicon.py:253-260` is only for contentless intent detection.

## Contraction Handling

The dictionary normalizer deletes apostrophes instead of expanding contractions:

| Raw | Normalized | Tokens |
|---|---|---|
| `I'd` | `id` | [id] |
| `I’d` | `id` | [id] |
| `I'll` | `ill` | [ill] |
| `I'm` | `im` | [im] |
| `I've` | `ive` | [ive] |
| `you're` | `youre` | [youre] |
| `you'd` | `youd` | [youd] |
| `don't` | `dont` | [dont] |
| `doesn't` | `doesnt` | [doesnt] |
| `can't` | `cant` | [cant] |
| `won't` | `wont` | [wont] |
| `isn't` | `isnt` | [isnt] |
| `it's` | `its` | [its] |
| `that's` | `thats` | [thats] |
| `we're` | `were` | [were] |

This policy may be useful for some names/possessives such as `Levi’s` and `O’Neill` only if concatenation is intended, but it is unsafe for contractions. No module distinguishes those cases.

## Punctuation and Unicode Apostrophes

ASCII and common Unicode apostrophes (’ ʼ ＇) are deleted when internal. Hyphens, underscores, punctuation, and symbols become spaces.

| Pair | Current dictionary result |
|---|---|
| `I'd` / `I’d` | `id` |
| `machine-washable` / `machine washable` | `machine washable` |
| `UV-protection` / `UV protection` | `uv protection` |
| `non-slip` / `non slip` | `non slip` |
| `water-resistant` / `water resistant` | `water resistant` |

Hyphen/space handling is consistent; contraction handling is destructive. Intent regexes are less Unicode-aware than dictionary normalization.

## Matching Mechanisms

| Matcher | File/function | Match type | Normalized text? | Risk |
|---|---|---|---|---|
| Canonical exact lookup | `registry.py:392-424` | Full normalized string equality | Yes | No arbitrary substring; apostrophe deletion can create accidental exact values. |
| Surface phrase extraction | `constraints.py:533-605` | Exact normalized token sequence, longest-first/non-overlap | Yes | Contraction token can become a candidate. |
| Context/ambiguity | `constraints.py:645-766` | Context and canonical disambiguation | Partly | Does not protect semantic path. |
| Alias/intent regex | `constraints.py:501-530` and intent router | Regex with boundaries | Depends on caller | ASCII/Unicode behavior differs. |
| Full semantic match | `registry.py:426-481` | Embedding nearest-neighbor over fields | Yes | Full text can retrieve related values. |
| N-gram semantic match | `registry.py:483-565` | Embedding each 1/2/3-gram | Yes | Current source of `id` false claims. |
| Product fact score | `retrieval.py:539-594` | Canonical fact membership weighted by evidence confidence | Yes | Soft semantic signal, no product embedding at this stage. |
| Intent routing | `intent_router.py:440-490` | Lexical/heuristic signals plus extraction tags | Raw plus normalized extraction | Repeated extraction can inflate intent tags. |

The active exact matcher is not `phrase in normalized_query`. No exact `id` match was observed in `paid`, `idea`, or `identity`. The problematic match is semantic.

Semantic details: BGE-small-en-v1.5, 384 dimensions; query/document vectors L2-normalized; dot product is cosine; default threshold 0.80; actual cosine retained; no buckets; below threshold rejected; each n-gram searched independently; top-one per attribute per gram and canonical-ID max deduplication. No planned product-level MAX-plus-SUM is active.

## Query Embedding Input

### Active Layer 2 query embedding

Input starts from the full current user message, becomes normalized semantic text after residual stopword filtering, and is split into independent contiguous 1/2/3-grams. It is not the full session transcript and exact spans are not removed.

~~~text
raw:             "I'd like something for hiking"
semantic_text:   "id hiking"
semantic_ngrams: "id", "hiking", "id hiking"
~~~

`dictionary/semantic.py:63-92` configures local BGE with `task=None` and no document/query prompt names. The attribute matrices contain canonical normalized values embedded with the same model.

### Dormant direct product Layer 2

`session.py:63-66` builds raw chronological `state.query_text`. `retrieval.py:301-329` can pass that transcript to old product embedding views if an index and encoder are explicitly supplied. The evaluator factory at `evaluator/agent_factory.py:10-18` does not supply them; current product Jina artifacts are absent. This path is dormant.

### Routing and reranker

Default intent routing receives the raw current message. Optional Qwen intent receives raw text plus its own intent prompt. There is no separate product reranker; retrieval combines structured and canonical semantic product scores.

## Phrase-Level Attribute Matching

Active canonical semantic fields are category, color, material, style, feature, and use_case. Brand is exact-oriented and excluded from V5 semantic matrices; size and price are structured.

| Field | Query representation | Exact path | Semantic path | Runtime status |
|---|---|---|---|---|
| use_case | Full message → normalized text → 1/2/3 grams | Exact token phrase | BGE use_case matrix | Active |
| feature | Same | Exact token phrase | BGE feature matrix | Active |
| category | Same | Exact token phrase | BGE category matrix | Active |
| material | Same | Exact token phrase | BGE material matrix | Active |
| color | Same | Exact token phrase | BGE color matrix | Active |
| style | Same | Exact token phrase | Empty style matrix | Effectively inactive semantically |

Layer 1 and Layer 2 objects are separate, but independent Layer 2 text can reproduce an exact value. For example, `hiking` may be both exact Layer 1 and semantic evidence at 1.0.

At ranking, `retrieval.py:539-594` receives canonical semantic evidence such as `use_case:hiking@1.0`, checks normalized product-fact membership, and averages matched evidence confidence. It does not compare query n-grams directly against product phrase vectors. There is no phrase-to-product semantic inverted index, no product phrase similarity matrix, and no semantic score `state.json`.

### Comparison with the previous semantic phrase-scoring plan

| Previous plan | Current implementation | Compatible? | Required interpretation |
|---|---|---|---|
| Unigrams + adjacent bigrams | 1/2/3-grams, overlapping | Partial | Third grams/all-text candidates broaden noise. |
| Compare n-grams with stored phrases | Compare with unique canonical values per field | In spirit | Stored unit is canonical value, not product phrase row. |
| Exact = 1.0 | Layer 1 exact = 1.0; semantic can also return exact 1.0 | Partial | Deduplication is a separate decision. |
| <.80 zero, otherwise actual cosine | Current threshold/cosine behavior | Yes | Confirmed. |
| MAX over a product’s phrases per query gram | Top-one per attribute and canonical-ID max; product fact membership later | No | Old product phrase aggregation is not active. |
| SUM across query grams | Not implemented; retrieval averages matched evidence | No | No current consumer for old sum. |
| Independent fields | Field matrices independent, but every gram searches every field | Partial | Candidate extraction is not field-specific. |
| Product field scores in semantic `state.json` | No file; session stores semantic canonical evidence in memory | No | Old score-state concept is obsolete. |
| Reranker reads phrase scores | No separate product reranker; retriever calculates final score | No | Boundary changed. |
| Layer 2 separate | Session state is separate but retrieval combines scores; old product dense Layer 2 dormant | Partial | Layer 2 now means canonical attribute semantics. |
| No synonym groups/buckets/multiplication | No groups, no buckets, no multiplication | Yes | Confirmed. |
| Keep overlapping grams | Current implementation keeps them | Yes | Still a risk. |
| Do not sum multiple synonyms per gram | Top-one/max behavior limits accumulation | Partial | Not the old product-level algorithm. |

## Short / Dangerous Phrase Analysis

| Path | Rows/format | Producer/consumer | Runtime status |
|---|---|---|---|
| `data/catalog.jsonl` | 50,000 JSONL | Catalog loader/retriever | Active |
| `data/derived/annotations/v4/annotations.jsonl` | 49,999 JSONL | Older annotation pipeline | Present, not default |
| `data/derived/annotations/v5/annotations.jsonl` | 50,000 JSONL | V5 aggregation / retriever | Active default facts |
| `data/derived/annotations/v5/{category,brand,color,material,feature,use_case}.jsonl` | 50,000 each | V5 per-field producers/inspection | Present |
| `data/derived/dictionary/canonical_values.json` | 47,187 JSON values | Dictionary builder / registry | Active |
| `data/derived/dictionary/attribute_embeddings/*.npy` | 384-d float32 matrices | BGE builder / registry | Active |
| `data/derived/dictionary/attribute_embeddings/*/metadata.json` | JSON row mapping | Embedding loader | Active |
| `data/derived/dictionary/manifest.json` | JSON metadata | Dictionary loader | Active |
| `data/derived/gptannotation/sessions.jsonl` | 400 JSONL sessions | evaluator/debug web | Active for evaluation, not phrase matching |

Absent current product artifact directories include `data/derived/product_embeddings_jina`, `data/derived/product_embeddings`, `data/derived/layer2_embeddings`, and the old V5 dictionary subdirectory. `product_embeddings/layer2.py` remains as dormant compatibility code for four old product views (categories, title, features, description).

Canonical dictionary short-value counts: one 1-character value (brand:e), 67 2-character values, 573 3-character values. One-token values of length ≤3 by field: brand 514, use_case 26, color 21, material 21, category 17, feature 8. Examples include use-case bag/bbq/bjj/bmx/gym/spa, feature dry/dwr/esd/gps/led, category bra/tee/wig, and materials eva/mdf/pvc/tpu/wax.

`id` is not a one-token exact value but appears in 43 longer values. `UV` is meaningful in `uv protection` and `uv shirts`; `XL` appears in brand phrases; `3D` is an exact brand value; `5G` was not found. Exact short brands include `on` and `run`. Minimum token length alone is unsafe.

## Negation Safety

There is no general negative constraint representation:

| Input | Current behavior |
|---|---|
| `I don't want leather` | Positive material leather and related semantic claims; no exclusion. |
| `I want shoes without laces` | Positive shoes and lace/no-laces claims; no negative predicate. |
| `not red` | Positive red; `not` remains in semantic text. |
| `anything except cotton` | Positive cotton; no exclusion. |
| `no leather` | Positive leather plus semantic noise. |
| `avoid leather` | Positive leather; no negative intent. |

The latest negative-response change is for no-preference clarification replies, not general negation. Removing `not`, `no`, `without`, `avoid`, `except`, or contraction forms as generic stopwords could erase meaning. Polarity must be identified before filtering.

## Intent and Override Safety

`TwoPhaseIntentRouter.classify` (intent_router.py:440-490) receives raw text, calls extraction again, and can use populated semantic fields as intent tags. Session correction/override logic (session.py:209-293) uses raw markers such as actually, instead, rather, change, changed, switch, and not. Agent reset/promotion logic (agent.py:116-145) also depends on raw message plus delta. No-preference handling intentionally skips extraction and transcript inclusion.

Global cleanup before these decisions could change buying/browsing routing, preference updates, full-goal resets, clarification handling, or exclusions. ASCII and curly apostrophe handling is already inconsistent between routing regexes and dictionary normalization.

## Diagnostic Query Results

These are current production extractor/BGE diagnostics, not ranking benchmarks.

| Raw query | Semantic text | Key exact results | Unexpected/semantic results |
|---|---|---|---|
| `I'd like something for hiking` | `id hiking` | use_case:hiking | storing id .8536, id case .8439, id window .8396 |
| `I'd prefer polarized sunglasses` | `id polarized sunglasses` | no reliable id exact value | id gram remains eligible for false claims |
| `I'm looking for something lightweight` | `im lightweight` | feature:lightweight | related lightweight jacket/outdoor-use results; no observed im id-like claim |
| `I want something machine washable` | `machine washable` | feature:machine_washable | washable/washing/machine-related results |
| `Can you find me something for everyday wear?` | `can everyday wear` | bad brand:you; use_case:everyday wear | everyday/wearing/clothing relations |
| `I don't want leather` | `dont leather` | material:leather | positive leather despite negation; no exclusion |
| `I'd rather have cotton` | `id rather have cotton` | material:cotton | id claims plus fabric/cotton-lined/linen relations |
| `Actually, I'd prefer something waterproof` | `actually id prefer waterproof` | feature:waterproof | id claims plus waterproof relations |
| `Something with UV protection` | `uv protection` | feature:uv_protection | protection/anti-UV/ultraviolet cross-field results |
| `Please show me non-slip shoes` | `non slip shoes` | feature:non_slip, category:shoes; bad use_case:show | slip/slip-on/shoe-compartment relations |
| `Something easy to clean` | `easy clean` | feature:easy_to_clean | easy-clean/wipe-clean/cleaning and clear-color relations |
| `not red` | `not red` | color:red | positive red; no negative representation |
| `I want shoes without laces` | `shoes without laces` | category:shoes | positive shoes and lace-related claims |
| `anything except cotton` | `anything except cotton` | material:cotton | positive cotton and related claims |

Normalization probes:

~~~text
I'd -> id -> [id]       I’d -> id -> [id]
I'll -> ill             I'm -> im             I've -> ive
you're -> youre         don't -> dont
machine-washable -> machine washable
UV-protection -> uv protection
non-slip -> non slip
water-resistant -> water resistant
~~~

## Existing Tests

Read-only command `PYTHONDONTWRITEBYTECODE=1 pytest -q` produced:

~~~text
27 failed, 168 passed, 5 skipped
~~~

Relevant tests:

| Test file | Coverage |
|---|---|
| `tests/test_attribute_dictionary.py` | Canonical normalization, punctuation/apostrophe variants, longest-first phrase matching, ambiguity. |
| `tests/test_ambiguity_resolution.py` | Exact token/context resolution and injected semantic matcher. |
| `tests/test_intent_router.py` | Intent, constraints, single-letter-size safeguards, session/override examples. |
| `tests/test_override_handling.py` | No-preference extraction skip, transcript exclusion, overrides. |
| `tests/test_bge_attribute_semantic.py` | BGE model/dimension/artifact loading and console behavior. |
| `tests/test_console_canonical_test.py` | Console canonical semantic expectations. |
| `tests/test_retrieval.py` | Structured/dense score formula expectations; some still expect old dense weighting. |
| `tests/test_debug_web.py` | Debug web/evaluator endpoints. |
| `tests/test_hard_evaluator_debug.py` | Evaluator state/ranking snapshot flow. |

No focused regression test covers contraction normalization through semantic n-grams, the exact-vs-semantic `id` failure, substring safety for paid/idea/identity, negation polarity, curly apostrophes, valid short values, exact/semantic deduplication, or cross-field n-gram over-generation. Several current failures are stale expectations after semantic/session/evaluator refactors, including old dense weighting, console outcomes, labels, and intent decisions.

## Root Cause

Evidence supports this combination:

1. Shared apostrophe deletion converts `I'd`/`I’d` to `id`.
2. Residual stopwords do not filter the resulting artifact.
3. Full-message semantic processing emits arbitrary overlapping 1/2/3-grams.
4. Every gram searches every semantic attribute field.
5. A one-token gram may retrieve a longer canonical value if cosine ≥0.80.
6. Negative language is not represented as negative constraints.

Evidence does not support arbitrary exact substring matching as the primary cause. The current active BGE path has no query/document prefix, so prompt changes are not the first remedy for this bug.

## Where Cleanup Should Happen

### A. Globally at the beginning

Not recommended. It can change routing, override/correction detection, no-preference handling, negation, and transcript semantics.

### B. Only for lexical phrase matching

Useful if raw text and a separate normalized view remain available, but insufficient unless semantic matching also consumes a controlled candidate view.

### C. Only after intent/constraint extraction

Too late for `id` because the current semantic extraction creates the false claim.

### D. Only when constructing semantic phrase queries

Necessary at the immediate bug entry point, but should be paired with candidate extraction and polarity handling rather than only a broad list.

### E. Separate normalized views

Best fit. The code already has raw message, normalized dictionary tokens, semantic text, extracted structured/semantic constraints, and raw session query text. Make their roles explicit:

~~~text
raw_user_text -> routing / override / correction / polarity
              -> exact_matching_view
              -> semantic_attribute_candidate_view
              -> raw session.query_text
~~~

## Design Options

### Option A — Global Stopwords

Highest risk. It can alter routing, overrides, negation, clarification, and transcript semantics, and would not solve contractions unless artifacts are separately handled.

### Option B — Matcher-Only Stopwords

Keeps raw text unchanged and limits filtering to attribute matching. Compatible with current session design, but requires a contraction policy and protected short values.

### Option C — Candidate Phrase Extraction

Uses known canonical phrases, field context, or explicit extraction instead of every arbitrary n-gram. Directly addresses one-token `id` retrieving unrelated longer phrases and reduces cross-field false positives.

### Option D — Contraction Expansion

For a matching view, expand `I'd` to `I would` and `don't` to `do not`, then selectively filter function words. Better preserves meaning, but needs Unicode-aware rules and a policy for names/possessives.

### Option E — Minimum Token Length

Unsafe alone. Current data contains meaningful UV/XL/3D-style values, sizes, abbreviations, short brands, materials, and short categories/use cases. Use field-aware protection if considered.

### Option F — Known-Phrase Whitelist

Useful for complete candidate phrases or protected short values. Component membership alone is insufficient because `id` occurs inside valid longer phrases.

### Option G — Hybrid

Best fit: preserve raw text; use a Unicode-aware contraction/possessive policy; build a matcher-only cleaned view; protect valid short vocabulary; extract known attribute candidates; retain negation/control evidence; and send only deliberate candidates to semantic matching.

## Risks

- Adding only `id` to stopwords leaves the same issue for `im`, `ive`, `dont`, `cant`, `wont`, and `youre`.
- Removing `not`, `no`, `without`, `avoid`, or `except` can invert intent.
- Expanding all apostrophes can corrupt names/possessives such as `Levi’s` and `O’Neill`.
- Minimum length can remove valid short attribute values, sizes, identifiers, brands, and materials.
- Filtering before routing can change actually/instead/rather/correction behavior.
- Extraction runs twice on a normal turn: constraints and intent classification.
- Exact Layer 1 and independent Layer 2 evidence can duplicate canonical values.
- Stopwords alone will not eliminate semantic cross-field results such as UV → ultraviolet color or easy clean → clear color.

## Recommended Next Implementation Scope

### MUST FIX

- Preserve raw text for routing, overrides, correction, negation, and session history.
- Introduce an explicit matcher-only representation.
- Prevent contraction artifacts such as `id` from becoming standalone semantic candidates.
- Preserve polarity/control words until polarity is identified.
- Add regression tests for ASCII/curly contractions, exact-vs-semantic `id`, punctuation, substring safety, and negation.

### SHOULD CONSIDER

- Field-aware/known-phrase candidate extraction instead of all normalized message n-grams.
- A protected short-value policy for UV, XL, 3D, sizes, and field-specific abbreviations.
- Explicit exact/semantic evidence deduplication semantics.
- Consistent Unicode apostrophe handling in dictionary and intent paths.
- Diagnostics displaying raw text, matching text, semantic candidates, accepted canonical values, and polarity separately.

### DO NOT CHANGE YET

- Jina configuration or embedding prompts;
- BGE model or embedding matrices;
- ranking weights;
- FAISS/BM25/product retrieval;
- intent routing/override behavior;
- phrase embedding generation;
- unrelated runtime architecture solely to mask a normalization regression.

## Questions Requiring Design Decision

1. Should `I'd`/`I’d` expand to `I would`, be excluded as a contraction span, or remain raw but be omitted from attribute candidates?
2. Should names/possessives such as `Levi’s` continue apostrophe deletion?
3. Which short values are explicitly valid per field (UV, XL, 3D, 5G, sizes, material abbreviations)?
4. Should Layer 2 receive only attribute candidates or a full semantic sentence for some fields?
5. How should negative constraints be represented before filtering?
6. Should exact Layer 1 and semantic evidence deduplicate by canonical ID?
7. May a one-token query retrieve a longer canonical phrase, and under what evidence?
8. Should current 3-grams remain, or return to unigrams and adjacent bigrams?
9. Should intent classification continue to call the full extractor after cleanup?
