# ADR 0001 — Buying vs Browsing intent router

- **Status:** Proposed
- **Issue:** #6
- **Date:** 2026-08-26

## Context

The agent architecture routes into two retrieval strategies depending on
whether the customer is a targeted buyer or an exploratory browser. Issue #6
asks for that classification layer alone: one message in, one of `BUYING` or
`BROWSING` out, with a confidence in `[0, 1]`.

Two constraints shaped the design more than anything in the ticket text.

**Almost no labelled data.** The project has thirteen labelled examples for
this task — the ones written into the issue. `data/derived/gptannotation/sessions.jsonl`
was excluded from this work by instruction and was not read. Nothing can be
trained or fitted against thirteen examples, so any learned component has to
work zero-shot, and a dev set had to be written as part of the ticket.

**The submission may run offline.** `docs/submission_rules.md` warns that
organizer policy may disable network access for final scoring and forbids
depending on undeclared external services. Whatever ships must work with no
credentials and no network.

## Decision

A two-tier cascade behind one `IntentRouter` protocol.

**Tier 1 — `LexicalIntentRouter`.** A weighted signal ledger scoring evidence
of commitment (budget, brand, size, colour, material, feature; more weakly use
case and style) against evidence of hesitation (undecidedness, exploration
verbs, option-seeking, vague head nouns). Category keywords carry **zero**
weight by design, which is what satisfies the acceptance criterion that
exploratory language must not be overridden by a category term. Standard
library only, ~0.04 ms per message, fully auditable — every result carries the
spans that produced it.

Two conditional rules carry the ambiguity cases the issue names:

- vagueness is **suppressed by hard evidence**, so "I need something for
  hiking, but it must be waterproof and under $80" resolves to Buying;
- a request verb **requires an actual request**, so "Can you show me some gift
  ideas?" stays Browsing.

**Tier 2 — `QwenRerankerBackend`.** Qwen3-Reranker-0.6B, ONNX int8, run
locally on CPU, entered **only** when Tier 1 reports confidence below 0.70.
The message is scored against two written label descriptions; normalising the
two relevance scores under a uniform prior makes the result a posterior, which
is what the issue's `confidence` field is supposed to mean.

**Tier 3 does not exist.** No API path is shipped. Network access is used once,
by `tools/fetch_reranker.py`, to download weights; the request path never
touches it.

### Session behaviour

The router is seeded on turn one and re-run only on *unsolicited* turns — an
override marker fires, or the message does not answer the attribute the agent
asked for and is not a no-preference reply. Flips need a larger margin than the
seed did, and Buying → Browsing additionally requires an explicit exploratory
phrase.

This answers the question raised on the ticket ("dynamically identify intent
per round, or only from the first utterance"). Neither, as stated:

- **Per round, statelessly, is unusable.** `customer_sentence()` in
  `evaluator/hard_evaluator.py` renders every later-turn reply as a constraint
  statement — "I'd prefer something made from {value}.", "For budget, I'd like
  {value}." — because it is an answer to the attribute the agent just asked
  for. Classified in isolation those read as a decided buyer regardless of the
  session, so every Browsing session would convert by turn two. There is a
  test asserting exactly this, so the reasoning cannot silently rot.
- **First utterance only** is stable but deaf to the intent override that
  `docs/competition_specification.md` places on turn 3 or 4 of 15% of sessions.
- **Seed then gate** degrades to first-utterance-only when the gate never
  opens, so it is a strict superset with a bounded downside.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Generative small LLM (Qwen3-0.6B, Gemma 3 270M) | Generation is the wrong shape for a binary label: the output needs constrained decoding to be a label at all, and confidence must be reconstructed from token probabilities. A reranker emits a relevance score natively. |
| NLI zero-shot (`deberta-v3-base-zeroshot-v2.0`) | Viable and a third the size; kept as the documented fallback. The 2026 BTZSC benchmark reports 0.6B rerankers surpassing every NLI cross-encoder, so the reranker was tried first. |
| Fine-tuned encoder or a trained head | Needs training data. Thirteen examples is not a training set. |
| Model on every message | The rules tier answers ~81% of messages at 0.04 ms with no observed confident errors. Paying ~310 ms for those is waste, and it would put a 1.2 GB dependency on the critical path of every turn. |
| API model in the request path | Forbidden in spirit by the submission rules for final scoring, and unnecessary. |

## Measurements

Two labelled sets, neither derived from generated sessions:

- **Golden** (`tests/data/intent_golden.jsonl`) — the thirteen issue examples.
  The spec. Never tuned against.
- **Dev** (`data/derived/intent/dev_set.jsonl`) — 96 hand-written examples
  across seven strata (explicit, exploratory, category-plus-exploratory,
  vague-plus-constraints, bare category, negation, degenerate).

| Configuration | Golden | Dev | Dev macro-F1 |
| --- | ---: | ---: | ---: |
| Rules only | 13/13 | 88/96 | 0.9161 |
| Cascade (rules → reranker) | 13/13 | 93/96 | 0.9687 |

Escalation rate 19.3% of messages. Rules tier ~0.04 ms; an escalated message
~310 ms on CPU with the int8 graph. Reproduce with:

```bash
python -m tools.eval_intent_router            # rules only
python -m tools.eval_intent_router --model    # cascade
```

Two findings worth recording:

**Every rules-tier error landed inside the escalation band.** That is the
property the cascade depends on, and `test_misses_are_flagged_weak` asserts it
rather than leaving it to luck.

**The label descriptions matter more than the model.** An attribute-checklist
phrasing ("...has stated a budget, brand, colour, material, size or feature")
scored 12/21 on the escalating slice, because a bare category request such as
"I'm looking for cardigans" lists no attribute and so read as undecided. The
short, purely semantic pair now in `local_model.LABEL_QUERIES` scores 19/21.
Decidedness is what is being classified, so the labels say that and no more.
Treat edits to those strings as a behaviour change.

## Consequences

- The agent runs with **no network access and no credentials**. Reported token
  usage for routing is zero: the reranker is a scorer, not a generator.
- `onnxruntime` and `tokenizers` are **optional**, listed in
  `requirements-reranker.txt` rather than as repo dependencies. Weights are
  ~1.2 GB, git-ignored, fetched by `python -m tools.fetch_reranker`.
- Without the model the router runs rules-only and still satisfies every
  acceptance criterion, at 88/96 on the dev set instead of 93/96. Every failure
  path — missing package, missing weights, load error, exception, refusal to
  answer — returns the rules verdict with `tier="rules_fallback"`.
- `Agent.respond()` is untouched, so benchmark metrics must be unchanged. Any
  movement means scope leaked.

## Limitations

- The dev set was written by us. It is a fair test of generalisation beyond
  the thirteen examples, but it encodes our own idea of the boundary. The
  label conventions worth arguing about are **bare category → BUYING** and
  **contentless greeting → BROWSING**; both are judgment calls, not facts.
- The label descriptions were selected on the dev set, so dev numbers are
  mildly optimistic. The golden thirteen were never tuned against and held at
  13/13 throughout.
- Confidence is monotone but not statistically calibrated; there is not enough
  labelled data to calibrate it. It is fit for ranking and thresholding, not
  for arithmetic.
- The attribute vocabularies in `lexicon.py` are a small curated starting
  point and should migrate to issues #5 and #8 rather than growing here. Two
  bugs found while building it show why: an unanchored `\bm\b` size pattern
  matched the "m" in "I'm", and `down` as a material matched "down jacket",
  which is a category. Both invented a hard constraint and flipped the label
  confidently.
- Intermittent, harmless: onnxruntime has been seen to raise from its thread
  pool destructor at interpreter shutdown on Python 3.14, after all results
  are produced. `QwenRerankerBackend.close()` releases the session explicitly.

## Follow-ups

- Issue #7 owns constraint extraction and override handling; the router only
  notices that a turn is unsolicited.
- Issue #9 consumes `confidence` and the `weak` flag to decide whether to fork
  hard between retrieval tracks or blend them. The 0.70 threshold should be
  revisited once that consumer exists.
