# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

For a chronological record of agent changes, the reasons behind them, and their measured impact, see the [Improvement Log](IMPROVEMENT_LOG.md).

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Canonical Catalog Facts (Issue #5)

Issue #5 provides a reusable product-facts layer without modifying the frozen `data/catalog.jsonl`. The file-based pipeline separates the hosted-model annotation/audit artifacts from the deterministic Agent-facing `catalog_facts.jsonl` output:

```text
catalog.jsonl
    -> scripts.annotate_catalog
    -> data/derived/annotations/v5/annotations.jsonl + failures.jsonl + manifest.json
    -> scripts.build_catalog_facts
    -> data/derived/catalog_facts/catalog_facts.jsonl
```

The annotation runner is resumable: every completed success or failure is flushed to JSONL immediately, successful `parent_asin` values are skipped on later runs, failures are recorded with retry attempts, and prompt/model settings are stored in the manifest. Press `Ctrl+C` to stop scheduling new work; completed records remain saved and the next run resumes from the same output directory. It makes no network call unless a local endpoint is configured. Hosted endpoint URL, API key environment variable, model, timeout, token limit, retry count, concurrency, and an optional range/limit are configurable; credentials are never stored in the repository.

Create a local environment file from the ignored template, then fill in your own endpoint, model, and key:

```powershell
Copy-Item .env.example .env
notepad .env
```

`.env` is ignored by Git. Never paste the real API key into tracked files, README text, a commit, or a pull request. `ANNOTATION_BASE_URL` may be either an OpenAI-compatible `/v1` base URL or a full `/chat/completions` URL; the client adds the path when needed.

Preview the work without calling a model:

```powershell
python -m scripts.annotate_catalog --dry-run --limit 100
```

First, make one local test request. The command below reads the ignored `.env`, uses a long timeout for a remote/local model, and keeps concurrency at one so the setup is easy to diagnose:

```powershell
python -m scripts.annotate_catalog --env-file .env --limit 1 --timeout 180 --max-tokens 2048 --concurrency 1
```

Progress is printed immediately to stderr, including the selected product, request attempt, retry reason, success/failure, elapsed time, saved-record count, and batch totals. The final JSON summary remains on stdout. Press `Ctrl+C` once to stop cleanly; already completed records have been flushed and can be resumed with the same command. Use `--log-every 100` for a large run, or `--quiet` to suppress progress logs.

After that succeeds, run a small batch and review the generated facts before considering larger work:

```powershell
python -m scripts.annotate_catalog --env-file .env --output-dir data/derived/annotations/v5 --limit 10 --timeout 180 --max-tokens 2048 --concurrency 1 --retries 2 --log-every 1
```

For future larger runs, increase concurrency gradually (1, then 4, 8, and 16) only after checking precision, latency, and failure rates. Do not start a full 50,000-product run until those measurements are understood. After an annotation run completes, build and validate the deterministic facts output:

```powershell
python -m scripts.build_catalog_facts --annotations data/derived/annotations/v5/annotations.jsonl
python -m scripts.validate_catalog_facts
```

Reasoning is disabled by default for extraction; use `--thinking` only for an explicit comparison. The runner retries transient 429/network/5xx failures, gives at most one corrective retry for schema errors, and does not repeat a request that ended with `finish_reason=length` or a context-length error. Use `--no-json-mode` if the endpoint does not support the OpenAI `response_format` option. The runner sends the API key only as an in-memory `Authorization` header and never writes it to output or the manifest.

The V4 model response is exactly six array fields: `brand`, `color`, `material`, `style`, `feature`, and `use_case`. V4 deliberately removes `size` from the LLM request and response; structured sizes and measurements remain a separate concern. High-trust `brand`, `color`, and `material` values favor precision, while descriptive `style`, `feature`, and `use_case` values favor useful source-supported coverage. Values use lowercase natural text with spaces, not `snake_case`. `category`, `parent_asin`, and `price` are copied outside the model response. The annotation artifact keeps brand as an array; the deterministic Issue #5 builder adapts the first strongly identified brand value to the existing scalar final-facts contract and leaves `size` empty for downstream compatibility. The original catalog remains unchanged.

## Hard Evaluator and Expected-Utility Search Benchmark

The repository also includes a fixed hard benchmark for the expected-utility adaptive search policy. It contains 400 GPTAnnotation sessions in `data/derived/gptannotation/sessions.jsonl`: 160 Buying, 160 Browsing, 60 Intent Override, and 20 Boundary sessions. Each session has a catalog target and evidence-backed hidden facts used only by the simulator.

Run the hard evaluator with:

```bash
python -m evaluator.hard_evaluator
```

The evaluator uses the frozen 50,000-product catalog as the Agent's retrieval universe and for exact `parent_asin` target validation. It does not rebuild facts or alter the benchmark. At each turn, the Agent may ask for one attribute and return up to 10 recommendations; the simulator supplies the corresponding fixed customer reply. The evaluator reports Hit Rate@10, MRR, MTTC, scenario metrics, and TechnicalScore under the ten-turn limit.

This benchmark measures the expected-utility adaptive search idea described in the Improvement Log: maintain a posterior over candidate products, estimate the value of asking each unused attribute, acquire the most useful evidence, and revise stale constraints after an intent override.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
annotation/                       annotation prompt, schema, client, and runner
scripts/annotate_catalog.py      resumable hosted-model annotation CLI
scripts/build_catalog_facts.py   deterministic canonical-facts builder
scripts/validate_catalog_facts.py read-only canonical-facts validator
evaluator/hard_evaluator.py       fixed GPTAnnotation hard benchmark evaluator
data/derived/gptannotation/       400-session hard benchmark input
IMPROVEMENT_LOG.md                chronological change and evaluation record
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Participant release checklist: `docs/participant_release_checklist.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
