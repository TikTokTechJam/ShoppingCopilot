# Local Manual400 debug page

`evaluator.debug_web` is a localhost-only viewer around the same
`Manual400SessionRunner` used by `evaluator.hard_evaluator`. It loads the
catalog, Agent, and optional Jina Layer 2 resources once, then lets you step
through one evaluator session turn by turn.

PowerShell:

```powershell
$env:SHOPPING_EMBEDDING_MODEL="model/jina-embeddings-v5-text-nano"
python -m evaluator.debug_web
```

Git Bash:

```bash
export SHOPPING_EMBEDDING_MODEL="model/jina-embeddings-v5-text-nano"
python -m evaluator.debug_web
```

Open <http://127.0.0.1:8765>. The local model and the ignored
`data/derived/product_embeddings_jina/` artifact must be present for dense
diagnostics. If they are unavailable, the page reports Layer 2 as unavailable
and still exposes structured Agent behavior.

The page supports a seeded/random session pool, scenario filtering, loading a
specific `manual400_####` session, one-turn stepping, and sequential Run To
End. Target facts and ranks are evaluator-side display data only; they are
never passed to `Agent.respond()`.
