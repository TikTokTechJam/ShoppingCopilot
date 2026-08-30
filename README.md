# ShoppingCopilot

ShoppingCopilot is a deterministic, multi-turn product-search agent for the
TechJam Conversational E-Commerce Search Challenge. The agent must identify a
hidden catalog product as early as possible, rank it in the Top 10, and decide
which clarification question is worth asking next.

This repository contains both the challenge contract and one evolving solution.
Keep those two ideas separate when changing the code:

- [`docs/competition_specification.md`](docs/competition_specification.md) is
  the problem and evaluation contract.
- [`Architecture.md`](Architecture.md) describes what the current code does.
- [`docs/approaches.md`](docs/approaches.md) compares the current solution with
  practical alternatives and proposes an experiment order.
- [`IMPROVEMENT_LOG.md`](IMPROVEMENT_LOG.md) records benchmark evidence and its
  provenance.

## Problem in one minute

The evaluator starts a session with an anonymized preference profile and a
short customer message. On each of at most ten turns, the agent may return:

- one natural-language clarification question;
- one structured `ask_attribute` value; and
- up to ten ranked catalog `parent_asin` values.

A session succeeds when the hidden target appears in the valid Top 10. Exact
ASIN equality is the only hit condition. The test mix includes Buying,
Browsing, Intent Override, and Boundary sessions.

The core score is:

```text
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
```

See the [competition specification](docs/competition_specification.md) for the
complete protocol and [`docs/agent_api_contract.json`](docs/agent_api_contract.json)
for the machine-readable response schema.

## Current solution

The evaluator-facing implementation is [`starter/agent.py`](starter/agent.py).
It combines:

1. a two-phase Buying/Browsing router;
2. canonical constraint extraction from a generated attribute dictionary;
3. session state with correction and intent-override handling;
4. catalog fact indexes, price eligibility, semantic attribute evidence,
   BM25, and rating-aware ordering;
5. recommendation exclusion after a non-hit turn; and
6. a turn-aware clarification policy based on candidate-pool split quality,
   answer probability, and remaining score horizon.

The official evaluator factory currently uses the canonical attribute path.
Whole-product embedding modules remain in the repository for experiments, but
the factory does not configure them. Buying and Browsing currently use the same
ranking weights; their main runtime difference is the clarification prior.

For the complete data flow, scoring formula, fallbacks, and known limitations,
read [`Architecture.md`](Architecture.md).

## Required local artifacts

Large source and generated artifacts are intentionally excluded from Git. A
fresh clone is therefore source-complete but not benchmark-ready.

| Path | Required for | In Git? |
| --- | --- | --- |
| `data/catalog.jsonl` | Agent construction and all evaluators | No |
| `data/public_set.jsonl` | Public 200-session evaluator | No |
| `data/derived/gptannotation/sessions.jsonl` | Hard 400-session evaluator | Yes |
| `data/derived/annotations/v5/annotations.jsonl` | Product facts and dictionary build | No |
| `data/derived/annotations/v5/dictionary/` | Agent import and categorical extraction | No |
| `models/bge-small-en-v1.5/` | Semantic attribute matching | No |

The generated dictionary is required even for exact-only execution. If its BGE
matrices or local model are absent, exact dictionary loading can still work,
but semantic attribute matching is unavailable. The current checkout does not
silently replace a missing dictionary with a hand-written vocabulary.

## Setup

Use Python 3.10 or newer from the repository root.

1. Place the frozen 50,000-product catalog at `data/catalog.jsonl` and, for the
   public evaluator, place the released session set at `data/public_set.jsonl`.
2. Obtain the team's V5 annotation aggregate, or generate it with the offline
   annotation pipeline described below.
3. Build the required exact dictionary:

   ```powershell
   python -m scripts.build_attribute_dictionary `
     --input data/derived/annotations/v5/annotations.jsonl `
     --output-dir data/derived/annotations/v5/dictionary `
     --no-embeddings

   python -m scripts.validate_attribute_dictionary `
     --directory data/derived/annotations/v5/dictionary
   ```

4. Optional: install the embedding dependencies and build the active semantic
   attribute matrices:

   ```powershell
   python -m pip install -r requirements-embeddings.txt
   python -m scripts.setup_bge_attribute_model
   python -m scripts.build_v5_attribute_embeddings
   python -m scripts.validate_attribute_dictionary `
     --directory data/derived/annotations/v5/dictionary
   ```

5. Run the checks and evaluators:

   ```powershell
   python -m pip install pytest
   python -m pytest -q
   python -m evaluator.local_evaluator
   python -m evaluator.hard_evaluator --output results.json
   ```

The local evaluator defaults to the 200-session public set. The hard evaluator
uses the tracked 400-session development benchmark. Neither is the organizer's
private final evaluation.

## Debug a conversation

Run the local browser-based debugger:

```powershell
python -m evaluator.debug_web
```

Then open <http://127.0.0.1:8765>. Use `--interactive` to choose a catalog
target and enter the customer replies yourself. The page exposes constraint,
candidate, score, and target-rank diagnostics without sending evaluator-only
facts to the agent. See [`docs/debug_web.md`](docs/debug_web.md).

For narrower checks, the repository also includes console tools for canonical
matching and semantic attribute search:

```powershell
python -m scripts.console_canonical_test
python -m scripts.console_semantic_attribute_test
python -m scripts.console_canonical_attribute_semantic_test
```

## Rebuild V5 product facts

Annotation is an offline preparation step, not part of `Agent.respond()`. It
uses an OpenAI-compatible endpoint configured outside version control. Start
with a dry run and a single product before scheduling a larger batch:

```powershell
Copy-Item .env.example .env
python -m scripts.annotate_catalog --dry-run --limit 10
python -m scripts.annotate_catalog `
  --env-file .env `
  --limit 1 `
  --concurrency 1 `
  --timeout 180
```

The runner is resumable and flushes completed successes and failures to JSONL.
The six single-attribute V5 outputs can be joined without another model call:

```powershell
python -m scripts.aggregate_v5_annotations `
  --catalog data/catalog.jsonl `
  --input-dir data/derived/annotations/v5 `
  --output data/derived/annotations/v5/annotations.jsonl
```

Do not start a full-catalog annotation run until a small sample has been
reviewed for precision, latency, failure rate, and cost. Never commit `.env`,
API keys, downloaded models, or generated data artifacts.

## Repository map

| Path | Responsibility |
| --- | --- |
| `starter/` | Agent, routing, state, retrieval, ranking, and clarification |
| `dictionary/` | Canonical registry loading and local semantic encoder support |
| `annotation/` | Hosted-model prompts, schemas, validation, and resumable runners |
| `product_embeddings/` | Optional product-level embedding experiments and loaders |
| `evaluator/` | Public, hard, follow-up, and interactive evaluation tools |
| `scripts/` | Artifact generation, aggregation, validation, and console probes |
| `tests/` | Unit and behavior tests |
| `docs/` | Problem contract, design notes, audits, and operating guides |
| `data/` | Tracked small datasets plus ignored catalog/generated artifacts |

## Benchmark evidence

Checked-in `results_*.json` files are development snapshots, not a single
official leaderboard. Their names reflect historical experiments, and most do
not contain a full configuration manifest. Use them for investigation, but
only compare runs produced from the same code, catalog, facts, model artifacts,
and evaluator command. [`IMPROVEMENT_LOG.md`](IMPROVEMENT_LOG.md) records the
known provenance and the current reporting standard.

## Data and submission rules

- [`docs/submission_rules.md`](docs/submission_rules.md) — submission contents
  and model policy
- [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) — Amazon Reviews 2023 source and
  permitted-use reminder
- [`data/README.md`](data/README.md) — data inventory and generated-artifact
  layout
