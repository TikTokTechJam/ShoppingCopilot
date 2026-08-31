# Local turn interpreter

The Agent can optionally use a local causal language model as a schema-guided
dialogue-state tracker. It returns a current-turn delta; the existing
deterministic parser validates and applies the resulting product slots. Price,
size, and typed measurements remain deterministic. If no model is configured,
the existing extraction path is used unchanged.

Set the environment variable to a local Hugging Face model directory before
starting an evaluator or the debug UI:

```bash
export SHOPPING_TURN_INTERPRETER_MODEL="model/<local-instruct-model>"
python -m evaluator.local_evaluator
```

An OpenAI-compatible self-hosted endpoint can be used instead. The existing
annotation variables are supported. A project `.env` can contain these values:

```text
ANNOTATION_BASE_URL=https://your-tailnet-host.example/v1
ANNOTATION_MODEL=your-model-name
ANNOTATION_API_KEY=replace-locally
```

Export those values in the shell before starting the Agent (for example with
`set -a; source .env; set +a`). The Agent does not implicitly read `.env`, so
ordinary offline runs cannot unexpectedly make a network request.

The endpoint receives the same schema-guided prompt through the repository's
existing `HostedLLMClient`. The API key is sent only as an in-memory Bearer
header and is never logged.

PowerShell:

```powershell
$env:ANNOTATION_BASE_URL="https://your-tailnet-host.example/v1"
$env:ANNOTATION_MODEL="your-model-name"
$env:ANNOTATION_API_KEY="replace-locally"
python -m evaluator.local_evaluator
```

The backend is created once when `Agent` is constructed. The hosted endpoint is
called only for normal user turns. If it is unavailable or a turn produces
invalid JSON, the Agent logs the reason and falls back to the existing
deterministic extractor. The local Hugging Face path uses local-only loading
and does not download a model.

The model should be an instruction-following causal language model that can
return JSON. The current repository does not include a generative turn model;
the existing Jina/BGE encoders are not suitable substitutes.
