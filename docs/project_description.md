# Devpost Project Description — Compass

## Inspiration and Problem

Keyword-only commerce search treats “I need something for a winter trip” and “I
need a black leather boot under $80” as variations of the same lookup. They are
not. The first customer needs guided exploration; the second expects hard
constraints to be respected immediately. Real conversations also change: a new
requirement can invalidate everything collected earlier.

Compass is an offline conversational shopping copilot that handles those three
states—browsing, buying, and intent override—against the frozen 50,000-product
Amazon catalog.

## What It Does

On every turn Compass returns up to ten ranked products and, when useful, one
focused clarification question. It accumulates grounded requirements, narrows an
overly broad catalog, removes stale preferences after an override, and avoids
showing failed options repeatedly. It follows the organizer's `Agent` contract
and completes every released public session within the ten-turn cap.

## How It Was Built

Compass constructs general in-memory indexes from the read-only catalog:

1. A category-alias index provides a broad structured candidate pool without
   reconstructing hidden intent cards.
2. A weighted SQLite FTS5 index retrieves across title, category, features,
details, brand/store, description, and price.
3. A generic facet index estimates the coverage and diversity of material,
   feature, style, use case, size, color, brand, and budget values so the agent
   can select one concrete clarification attribute.

The dialog router classifies the message form, updates per-session state, and
performs an atomic slot reset for explicit overrides. The ranker combines phrase
evidence, IDF-weighted token coverage, BM25 relevance, and a small quality prior.
Aggregate profile tags
are reserved for a no-query cold start. Previously
recommended products rotate out after a failed turn, improving catalog coverage
without wasting customer effort.

The implementation uses Python 3.13.5 (compatible with 3.10+), SQLite FTS5, Git,
GitHub CLI, and PowerShell. It uses no external API, hosted LLM, paid service,
third-party Python package, or secret. Reported model token use and model cost are
both zero.

For demonstration, `interactive_demo.py` provides a terminal chat over the full
catalog with free-form multi-turn refinements, visible product details, profile
controls, reset/help commands, and natural-language intent switching. The
separate `demo.py` reproduces a labeled public evaluator scenario for a video or
technical walkthrough.

## Results

On the untouched 200-session public evaluator:

- Hit Rate@10: **1.000**
- MRR: **0.687687**
- MTTC: **1.610**
- Efficiency: **0.939**
- Technical Score: **0.894106**

The published weak BM25 baseline scores 0.106710. Generalized Compass improves
the public Technical Score by 0.787396 while remaining fully offline and
reproducible. It never requests the catch-all clarification field and does not
reconstruct the evaluator's catalog-constraint ordering.

The repository's deterministic 200-session catalog-driven hard evaluator gives
Hit Rate@10 0.890, MRR 0.457266, MTTC 4.74, and Technical Score 0.707380. It is
independent of the public labeled conversations and preserves the materially
lower result as evidence of the remaining generalization gap. Intent override
is again the hardest scenario, with 0.766667 Hit Rate and MTTC 7.5. The exact
measurement is stored in `docs/hard_benchmark_results.json`.

As a separate sanity check, a fixed-seed uniform sample of 10 non-public catalog
targets used independently written free-form conversations. Compass found 10/10,
with MRR 0.883333, MTTC 1.9, and Technical Score 0.947000. Raw metadata clues are
seed-shuffled independently of evaluator ordering. The full reproducible
selection and transcripts are stored in `docs/random_benchmark_10.json`. Because
10 samples are small and catalog-derived clues are still used, this result is not
presented as a private-set estimate.

A stricter benchmark uniformly sampled 20 additional non-public products, after
which every conversation was written manually from individually inspected raw
metadata and frozen before execution. Compass found 15/20: Hit Rate 0.750, MRR
0.424008, MTTC 4.65, and Technical Score 0.629202. The five preserved failures
show that human paraphrases, sparse listings, and ambiguous category language are
substantially harder than catalog-generated prompts. The locked inputs and full
transcripts are in `benchmarks/human_queries_20.json` and
`docs/human_benchmark_20_results.json`. This is the repository's most credible
generalization check, though 20 products remain too small for a private-set
estimate.

## Dataset and Assets

The project uses only the organizer's frozen catalog and public development
sessions, derived from Amazon Reviews 2023 Clothing, Shoes and Jewelry. Catalog
integrity is checked against the published SHA-256 digest. No catalog record is
modified and no ASIN is injected.

## Challenges and Lessons

The core challenge was balancing precision and coverage. Hard-filtering every
parsed phrase is brittle, while broad BM25 alone repeats plausible but incorrect
products. Compass unions broad category and lexical routes, ranks their shared
evidence, and rotates confirmed misses. Clarification fields are selected using
candidate coverage and entropy rather than a simulator-specific catch-all.
Intent override required another subtle rule: pre-override recommendations cannot
be treated as confirmed misses, so their suppression history must also reset.

Aggregate profile tags were useful only as weak evidence. Giving them a large
weight hurt evaluation because a current requirement should beat historical
taste. They are therefore restricted to a no-query cold-start fallback.

## Limitations and Future Work

Fingerprint precision is strongest when customer language is grounded in catalog
metadata. A compact, quantized local sentence encoder would improve unseen
paraphrases. Index serialization would reduce cold-start time. A larger development
set and paraphrase stress suite would support calibrated personalization and
question-value learning without overfitting the public targets.

## Contributions

Compass currently has one integrated code path covering architecture, retrieval,
ranking, conversation state, tests, evaluation, demo, and documentation. Team
member names and contribution ownership should be added before Devpost submission
if this is entered by a team.
