# AGENTS.md

This repository uses `Architecture.md` as the main source of truth for implementation architecture.

These rules apply to Codex and any other automated coding agent working in this repository.

## Before making changes

1. Read `Architecture.md` before changing code related to:
   - catalog processing;
   - annotation;
   - canonical product facts;
   - structured extraction;
   - attribute dictionaries;
   - embeddings;
   - user-utterance parsing/canonicalization;
   - Buying/Browsing routing;
   - retrieval;
   - ranking/reranking;
   - session state;
   - clarification/DP logic;
   - Agent orchestration;
   - evaluator assumptions.
2. Read the relevant issue/ADR for task-specific scope.
3. Check that the requested implementation is compatible with `Architecture.md`.

## Architecture authority

Use this priority when sources conflict:

1. Official competition specification / evaluator contract.
2. `Architecture.md`.
3. Accepted ADR/design documents.
4. GitHub issues/comments.
5. Existing code.

Existing code is not automatically correct if it has drifted from the documented architecture.

## Implementation policy

- Read and follow `Architecture.md`.
- Do not silently introduce a different architecture.
- Do not add infrastructure or major dependencies that contradict the documented MVP design.
- Keep documented component boundaries intact.
- Prefer deterministic logic before semantic/LLM fallbacks where the architecture specifies that ordering.
- Keep offline/precomputed work separate from runtime Agent work.
- Preserve the official Agent/evaluator contract.
- Do not use hidden benchmark targets, simulator-only facts, or evaluator labels in Agent logic.

## Architecture.md is read-only for coding agents

Automated coding agents must not edit, rewrite, append to, or otherwise modify `Architecture.md`.

`Architecture.md` is maintained separately by repository maintainers. Coding agents consume it as an input contract; they do not maintain it as part of implementation work.

If a requested task appears to require an architecture change:

1. Do not implement the architectural deviation silently.
2. Do not modify `Architecture.md` yourself.
3. Surface the conflict clearly and describe the architecture change that would be required.
4. Wait for the maintained architecture to be updated before implementing code that would contradict the current document.

Do not use an issue, PR comment, implementation detail, or existing code path as permission to override `Architecture.md`.

## Ordinary implementation work

Do not treat normal internal implementation choices as architecture changes when they remain inside an existing documented boundary, for example:

- refactoring;
- helper-function naming;
- local performance improvements;
- retry implementation details;
- serialization details that do not alter the documented data contract;
- tests for existing behavior;
- implementation of an already documented fallback.

These changes should simply follow the architecture without editing the architecture document.

## Required final check

Before finishing a task, verify:

- the implementation matches `Architecture.md`;
- no new architecture decision was introduced only in code;
- relevant issue scope was respected;
- unrelated components were not changed;
- `Architecture.md` was not modified by the coding agent.

If the task request and `Architecture.md` conflict, surface the conflict instead of guessing or silently changing the design.
