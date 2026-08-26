# AGENTS.md

This repository uses `Architecture.md` as the **main source of truth for implementation architecture**.

These rules apply to Codex and any other automated coding agent working in this repository.

## Before making changes

1. Read `Architecture.md` before changing code related to:
   - catalog processing;
   - annotation;
   - canonical product facts;
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
3. Check whether the requested implementation is compatible with `Architecture.md`.

## Architecture authority

Use this priority when sources conflict:

1. Official competition specification / evaluator contract.
2. `Architecture.md`.
3. Accepted ADR/design documents.
4. GitHub issues/comments.
5. Existing code.

Existing code is not automatically correct if it has drifted from the documented architecture.

## Implementation policy

- Follow `Architecture.md`; do not silently introduce a different architecture.
- Do not add infrastructure or major dependencies that contradict the documented MVP design.
- In particular, do not introduce Postgres, an external vector database, a new retrieval branch, or a new LLM stage unless the architecture is intentionally changed.
- Keep component boundaries clear. For example, user canonicalization and product retrieval are separate responsibilities.
- Prefer deterministic logic before semantic/LLM fallbacks where the architecture specifies that ordering.
- Keep offline/precomputed work separate from runtime Agent work.
- Preserve the official Agent/evaluator contract.

## When architecture needs to change

If the requested task requires an architectural change:

1. Do not implement the architectural deviation silently.
2. Update `Architecture.md` in the same PR/change.
3. Explain the changed decision and affected components in the PR/commit description.
4. Update code to match the new documented architecture.
5. If an ADR is required by the issue, include/update the ADR as well.

The architecture documentation and implementation must remain synchronized.

## When architecture does not need to change

Do not edit `Architecture.md` for ordinary internal implementation details that stay inside an existing documented component boundary, such as:

- refactoring;
- helper-function naming;
- local performance improvements;
- retry implementation details;
- serialization details that do not alter the documented data contract;
- tests for existing behavior.

## Required final check

Before finishing a task, verify:

- the implementation still matches `Architecture.md`;
- no new architecture decision was introduced only in code;
- any architecture change was documented;
- relevant issue scope was respected;
- unrelated architecture was not modified.

If the task request and `Architecture.md` conflict and the task does not explicitly authorize changing the architecture, surface the conflict instead of guessing.