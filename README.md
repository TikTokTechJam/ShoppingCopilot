# Compass — Offline Conversational Shopping Copilot

Compass is a zero-cost, stateful shopping agent for the TechJam 2026
Conversational E-Commerce Search Challenge. It turns vague or changing customer
requests into ranked Amazon catalog recommendations in at most ten turns. The
system runs entirely in memory with Python and SQLite FTS5: no API key, hosted
model, network access, or external vector database is required at inference time.
For the chronological engineering record and before/after measurements, see the
[`IMPROVEMENT_LOG.md`](IMPROVEMENT_LOG.md).

## Results

Measured with the untouched official evaluator on all 200 labeled public
sessions:

| System | Hit Rate@10 | MRR | MTTC | Efficiency | Technical Score |
|---|---:|---:|---:|---:|---:|
| Organizer BM25 baseline | 0.125 | 0.068034 | 9.810 | 0.119 | 0.106710 |
| Generalized Compass | **1.000** | **0.687687** | **1.610** | **0.939** | **0.894106** |

Compass finds the hidden target in every released session, raises the public
Technical Score by `0.787396`, uses zero model tokens, and incurs no API cost.
Public-set results are development measurements, not a guarantee of private-set
performance. The machine-readable snapshot is
[`docs/solution_results.json`](docs/solution_results.json).
The complete generalized run took 47.665 seconds on the local Windows/Anaconda
environment, including cold index construction and all 200 sessions.

This generalized version does **not** reconstruct evaluator intent cards or ask
for the catch-all attribute. It chooses a concrete clarification field from the
live candidate distribution. The earlier evaluator-shaped experiment scored
`0.900319`; removing those shortcuts reduced the score by only `0.006213`.

### Catalog-Driven Hard Stress Evaluation

The repository's deterministic hard evaluator generates 200 sessions directly
from normalized catalog facts, independently of the public labeled sessions. It
uses seed `20260826`, prevents the catch-all `other` field from revealing target
facts, and tests paraphrased attributes, sparse metadata, intent overrides, and
ten-turn misses.

| Hit Rate@10 | MRR | MTTC | Efficiency | Technical Score |
|---:|---:|---:|---:|---:|
| **0.890** | **0.457266** | **4.740** | **0.626** | **0.707380** |

| Scenario | Samples | Hit Rate | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 0.9625 | 0.533512 | 3.4625 |
| Browsing | 80 | 0.8750 | 0.402297 | 5.0375 |
| Intent Override | 30 | 0.7667 | 0.418505 | 7.5000 |
| Boundary | 10 | 0.8000 | 0.403333 | 4.3000 |

Compass found 178 of 200 targets. The lower score than the public evaluator is
reported deliberately: it shows the remaining generalization gap, especially
for intent overrides, rather than treating the public score as a private-set
estimate. The exact result is stored in
[`docs/hard_benchmark_results.json`](docs/hard_benchmark_results.json).

```bash
python -m evaluator.hard_evaluator --sample-count 200 --seed 20260826
```

### Human-Authored 20-Product Benchmark

This is the strongest realism check in the repository. A fixed seed uniformly
selected 20 new products after excluding all 200 public targets and the earlier
10-product benchmark. Each product's raw metadata was then read individually and
converted by hand into natural multi-turn shopping language. Queries were frozen
before the first run and were not edited after failures appeared. The fixture
validator confirms zero ASIN leaks, complete-title copies, or evaluator-generated
messages.

| Hit Rate@10 | MRR | MTTC | Efficiency | Technical Score |
|---:|---:|---:|---:|---:|
| **0.750** | **0.424008** | **4.650** | **0.635** | **0.629202** |

Compass found 15 of 20 targets. Successful sessions converted in 2.53 turns on
average; official MTTC is 4.65 because each of the five misses contributes turn
11. The failures were a sparse-metadata safety shoe, wedding jewelry set,
gaiter/filter bundle, slingback-pump intent override, and camisole boundary case.
These expose real weaknesses in category inference, lexical dominance from use
case words, sparse metadata, and long-tail ranking.

| Scenario | Samples | Hit Rate | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 8 | 0.750 | 0.385913 | 3.875 |
| Browsing | 8 | 0.875 | 0.507440 | 4.250 |
| Intent Override | 3 | 0.666667 | 0.444444 | 5.666667 |
| Boundary | 1 | 0.000 | 0.000000 | 11.000 |

The committed run processed 88 responses with 314.878 ms mean and 1,142.593 ms
P95 response latency; timing is machine/load dependent. Accuracy, turns, and
ranks reproduced exactly on repeated runs. Read the locked inputs in
[`benchmarks/human_queries_20.json`](benchmarks/human_queries_20.json) and full
outputs in
[`docs/human_benchmark_20_results.json`](docs/human_benchmark_20_results.json).

```bash
python scripts/human_query_benchmark.py \
  --fixture benchmarks/human_queries_20.json \
  --output docs/human_benchmark_20_results.json
```

### Automated Raw-Metadata Sanity Check

To test beyond the released labels, `scripts/random_catalog_benchmark.py`
uniformly sampled 10 products from the 49,798 eligible catalog items after
excluding all 200 public targets. Seed `20260825` fixed the products and the
4 Buying / 4 Browsing / 1 Intent Override / 1 Boundary scenario assignment.
Separately written free-form prompts—not the official evaluator sentences—drove
each conversation. Clues are selected by a seeded shuffle of raw feature, detail,
description, and price fields; their order does not reproduce evaluator logic.

| Hit Rate@10 | MRR | MTTC | Efficiency | Technical Score |
|---:|---:|---:|---:|---:|
| **1.000** | **0.883333** | **1.900** | **0.910** | **0.947000** |

All 10 random targets were found. Across 19 responses, mean response time was
150.069 ms and P95 was 377.790 ms; the cold in-memory index took 20.690 seconds.
This is a small reproducible sanity check, not a statistically reliable private
holdout estimate. Because its clues are assembled automatically from raw catalog
fields, it is more optimistic than the human-authored benchmark. Full targets,
turns, ranks, timings, and transcripts are in
[`docs/random_benchmark_10.json`](docs/random_benchmark_10.json).

Reproduce it with:

```bash
python scripts/random_catalog_benchmark.py \
  --samples 10 --seed 20260825 \
  --output docs/random_benchmark_10.json
```

## How It Works

One immutable catalog feeds two retrieval routes and a stateful ranker:

```text
customer turn
    |
    +--> intent/state router --> buying | browsing | override
    |           |                         clears stale slots on override
    |           v
    +--> category-alias index --------> broad structured candidates
    |
    +--> weighted SQLite FTS5 ---------> broad lexical candidates
    |
    +--> generic facet analyzer ------> question coverage + diversity
                |                         material/feature/style/size/etc.
                +------------+----------+
                             v
                 semantic evidence ranker
                 + rare-token coverage
                 + phrase/BM25 relevance
                 + cold-start profile + quality prior
                             |
                             v
                 unseen Top 10 + proactive question
```

Key design choices:

- **Dual-track routing.** Buying messages immediately lock disclosed constraints;
  browsing messages retain a wider category pool while asking for one useful
  detail.
- **General catalog indexing.** Category aliases are derived from public category
  paths. No hidden intent-card order, truncation, or simulator constraint
  fingerprint is reconstructed.
- **Hybrid retrieval.** Broad category candidates are unioned with weighted FTS5;
  phrase evidence and rarity-aware token coverage rank partial or long-form clues.
- **Dynamic dialog state.** Constraints accumulate across turns. An explicit
  “ignore my earlier preference” message atomically clears stale slots and
  pre-override recommendation history.
- **Active clarification.** Candidate facet coverage and normalized entropy choose
  a concrete field such as `feature`, `material`, `style`, `use_case`, `size`, or
  `color`. Unavailable fields decay from the question policy. The catch-all field
  is never requested.
- **Failure-aware rotation.** If a recommendation did not convert, already-shown
  products are suppressed on later turns. New evidence re-ranks the remaining
  pool; an intent override resets this suppression because pre-override hits are
  not scored.
- **Safe personalization.** Aggregate preference tags guide only a true
  cold-start fallback. Once the customer provides current-session evidence, it
  completely replaces historical taste in ranking.

The implementation is in [`starter/agent.py`](starter/agent.py). It exports the
required `Agent` class and uses only the Python standard library.

## Setup

Python 3.10 or newer is required. The tested local environment uses Anaconda
Python 3.13.5.

Clone the repository and download the official frozen catalog release:

```bash
git clone https://github.com/TikTokTechJam/ShoppingCopilot.git
cd ShoppingCopilot
gh release download participant-kit \
  --repo TechJam2026/techjam-conversational-search \
  --pattern catalog.jsonl.gz --pattern SHA256SUMS \
  --dir data/releases
curl -L https://raw.githubusercontent.com/TechJam2026/techjam-conversational-search/main/data/public_set.jsonl \
  -o data/public_set.jsonl
```

Extract the catalog on Linux or macOS:

```bash
gzip -dk data/releases/catalog.jsonl.gz
mv data/releases/catalog.jsonl data/catalog.jsonl
sha256sum -c data/releases/SHA256SUMS --ignore-missing
```

PowerShell equivalent:

```powershell
$input = [IO.File]::OpenRead("data/releases/catalog.jsonl.gz")
$gzip = [IO.Compression.GzipStream]::new($input, [IO.Compression.CompressionMode]::Decompress)
$output = [IO.File]::Create((Join-Path (Resolve-Path data) "catalog.jsonl"))
$gzip.CopyTo($output)
$output.Dispose(); $gzip.Dispose(); $input.Dispose()
Get-FileHash -Algorithm SHA256 data/releases/catalog.jsonl.gz
```

Expected compressed-catalog SHA-256:

```text
07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8
```

No packages need to be installed. `requirements.txt` is intentionally empty
apart from a runtime note.

## Reproduce the Score

Run the tests:

```bash
python -m unittest discover -v
```

Run the official evaluator without editing it or the public labels:

```bash
python -m evaluator.local_evaluator --output results.json
```

Run the independent catalog-driven hard evaluator:

```bash
python -m evaluator.hard_evaluator --sample-count 200 --seed 20260826
```

On this Windows machine, where `python` is a Store alias, the equivalent command
uses Anaconda explicitly:

```powershell
& "C:\Users\Admin\anaconda3\python.exe" -m evaluator.local_evaluator --output results.json
```

The first `Agent` construction builds the 50,000-product in-memory indexes. All
sessions in one evaluator run reuse those indexes.

## Run the Interactive Terminal Demo

Chat with the full 50,000-product catalog using ordinary natural language:

```bash
python interactive_demo.py
```

Choose how many recommendations to display and seed an anonymized preference
profile:

```bash
python interactive_demo.py --top-k 5 --profile-tags fit,comfort,durability
```

Example conversation:

```text
You > I need women's shoes for trail running, preferably blue and breathable.
You > Leather, under $80.
You > Actually, switch to a formal black office shoe instead.
```

The terminal renders product titles, ASINs, prices, ratings, stores, and category
paths. It supports `/help`, `/profile`, `/reset`, and `/quit`. The first launch
builds the in-memory catalog indexes, so startup takes longer than later turns.

## Run the Scripted Evaluator Demo

`demo.py` prints customer turns, the agent's clarification, its top three
recommendations, and the conversion turn. The default sample demonstrates an
intent override:

```bash
python demo.py --sample-id public_0002
```

Try a browsing or buying case with `public_0006` or `public_0001`.

## Agent Contract

```python
from starter.agent import Agent

agent = Agent("data/catalog.jsonl")
agent.reset("session-1", user_profile)
response = agent.respond(
    "session-1",
    "I'm looking for Women Dresses, but I'm still exploring.",
    turn=1,
    top_k=10,
)
```

Each response contains customer-facing `message`, a valid `ask_attribute`, up to
ten ordered `parent_asin` recommendations, and zero token usage.

## Tools, APIs, Libraries, and Data

- Development: Git, GitHub CLI, PowerShell, and Anaconda Python 3.13.5.
- Runtime libraries: Python standard library (`sqlite3`, `dataclasses`, `re`,
  `json`, and collections). SQLite's bundled FTS5 module provides BM25 retrieval.
- APIs/models: none. There are no credentials, model downloads, or external
  inference calls.
- Data: the frozen 50,000-item competition catalog and 200 public development
  sessions derived from Amazon Reviews 2023, Clothing, Shoes and Jewelry. See
  [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md).

## Limitations and Next Improvements

- Retrieval remains lexical. Free-form paraphrases that share few catalog terms
  would benefit from a compact local sentence embedding model.
- Cold-start index construction trades startup time and memory for fast,
  dependency-free session handling. A serialized read-only index would improve
  repeated command-line starts.
- The preference profile is intentionally conservative because it is aggregate
  and can conflict with a current purchase. Learned calibration on a larger
  development split could make personalization more useful without weakening
  hard constraints.
- Product popularity is only a small tie-break prior. Near-duplicate variants
  with indistinguishable text remain inherently difficult to order.
- Question value is estimated from catalog facet diversity rather than learned
  from real customer responses.
- The released 200 sessions are a small development set. Hyperparameters should
  not be tuned further against individual public targets.

Given more time, the next step would be a compact offline semantic encoder with
quantized vectors, paraphrase stress tests, and serialized indexes—while keeping
the current deterministic fallback.

## Contributions

The repository currently contains one integrated implementation: agent
architecture, retrieval/ranking, state management, tests, evaluator integration,
demo, and documentation. Add participant names and any team-specific ownership
before the final Devpost submission.

## Competition References

- [`docs/competition_specification.md`](docs/competition_specification.md)
- [`docs/agent_api_contract.json`](docs/agent_api_contract.json)
- [`docs/submission_rules.md`](docs/submission_rules.md)
- [`docs/project_description.md`](docs/project_description.md)
- [`evaluator/hard_evaluator.py`](evaluator/hard_evaluator.py)
