# Local Manual400 debug page

`evaluator.debug_web` is a localhost-only viewer around the same
`Manual400SessionRunner` used by `evaluator.hard_evaluator`. It loads the
catalog and Agent once, then lets you step through one evaluator session turn
by turn. The current semantic path is the local BGE canonical-attribute
dictionary; whole-product Jina retrieval is not part of this flow.

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

The web entry point validates the fixed session file with the same
`validate_sessions` function used by the batch evaluator. Each `Next Turn` and
`Run To End` action uses the shared `Manual400SessionRunner`, including reply
simulation, override timing, response validation, exclusions, and scoreability.
The page supports a seeded/random session pool, scenario filtering, loading a
specific `manual400_####` session, one-turn stepping, and sequential Run To
End. Target facts and ranks are evaluator-side display data only; they are
never passed to `Agent.respond()`.
