# Local evaluator, Manual400, and interactive debug page

`evaluator.debug_web` is a localhost-only viewer around the same Agent flow
used by the benchmark evaluators. It loads the catalog and Agent once, then
lets you step through one evaluator session turn by turn. The current
semantic path is the local BGE canonical-attribute dictionary; whole-product
Jina retrieval is not part of this flow.

The default is the hard Manual400 benchmark. To inspect the public-set local
evaluator with its own session simulator, use:

```powershell
python -m evaluator.debug_web --evaluator local --dataset data/public_set.jsonl
```

The local mode derives the same intent card from the target catalog row when
the public dataset does not contain one, uses the same generated replies and
override timing as `evaluator.local_evaluator`, and shows the resulting
per-session score in the page.

For a developer-controlled conversation against a product you select from the
catalog, start interactive mode:

```powershell
python -m evaluator.debug_web --interactive
```

The terminal starts a target picker. Enter a catalog ASIN directly, or use
`search <words>` to search product titles and select one of the numbered
results. Enter the initial shopper message and then each reply yourself. The
browser page at <http://127.0.0.1:8765> polls the same live Agent state after
each reply, including structured, semantic, BM25, hybrid, and target-rank
diagnostics. Type `q` at a prompt to stop, `restart` to restart the current
target, or `target <ASIN>` while replying to switch targets.

PowerShell:

```powershell
$env:SHOPPING_ATTRIBUTE_EMBEDDING_MODEL="models/bge-small-en-v1.5"
python -m evaluator.debug_web
```

Git Bash:

```bash
export SHOPPING_ATTRIBUTE_EMBEDDING_MODEL="models/bge-small-en-v1.5"
python -m evaluator.debug_web
```

Open <http://127.0.0.1:8765>. The local BGE model and V5 attribute embedding
artifacts must be present for semantic diagnostics. If they are unavailable,
the page reports semantic matching as unavailable and still exposes structured
Agent behavior.

In hard mode, the web entry point validates the fixed session file with the
same `validate_sessions` function used by the batch evaluator. In local mode,
it validates public-set IDs and catalog targets, then uses the public-set
simulator from `evaluator.local_evaluator`. Each `Next Turn` and `Run To End`
action executes one mode's corresponding simulator, including reply
simulation, override timing, exclusions, and scoreability. The page supports a
seeded/random session pool, scenario filtering, loading a specific session,
one-turn stepping, and sequential Run To End. Target facts and ranks are
evaluator-side display data only; they are never passed to `Agent.respond()`.

Diagnostics show two independent state tracks. Structured constraints contain
exact and numeric evidence only. Dense semantic constraints contain accepted
BGE canonical values and their cosine similarities. Retrieval calculates both
scores separately, then the existing mode-specific reranker combines them into
the final score used for Top10 ordering.
