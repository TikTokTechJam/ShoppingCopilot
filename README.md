
## LLM Slot-Filling Turn Interpreter

The runtime uses a small self-hosted LLM as a schema-guided turn interpreter between raw user language and deterministic session state.

This design follows recent **Dialogue State Tracking (DST)** and **joint intent/slot filling** work, where language models convert natural-language turns into structured slot updates instead of relying on token-level stopword heuristics.

```text
User utterance
      ↓
Deterministic parsers
(price, numeric ranges, size)
      ↓
Self-hosted LLM
      ↓
Structured slot/state delta
      ↓
Deterministic validation
      ↓
Dependency-aware Session State
      ↓
BM25 + BGE retrieval
```

### Schema-guided extraction

The LLM receives a fixed shopping schema and extracts only supported semantic fields:

```json
{
  "intent": "buying | browsing",
  "constraints": {
    "category": [],
    "brand": [],
    "color": [],
    "material": [],
    "feature": [],
    "use_case": [],
    "style": []
  },
  "override": {
    "type": "none | preference",
    "fields": []
  }
}
```

This follows **schema-driven dialogue state tracking**, where natural-language descriptions of slots guide the model toward the intended structured representation rather than unrestricted generation.

Reference:
- Lee et al., *Dialogue State Tracking with a Language Model using Schema-Driven Prompting*, EMNLP 2021.

### Intent-aware slot filling

Example:

```text
"I'm exploring sweatshirts and would like to compare some options."

→ intent: browsing
→ category: sweatshirt
→ use_case: none
```

The model separates dialogue intent from product attributes, avoiding false constraints such as:

```text
use_case: exploring
use_case: city exploring
```

This is aligned with **joint intent detection and slot filling** research, where utterance-level intent and token/slot-level semantics are modeled together instead of independently.

References:
- Goo et al., *Slot-Gated Modeling for Joint Slot Filling and Intent Prediction*, NAACL 2018.
- Chen et al., *BERT for Joint Intent Classification and Slot Filling*, 2019.

### Structured function-style output

The LLM does not retrieve products or directly modify state. It only emits a structured current-turn delta.

This is similar to **FnCTOD**, which formulates dialogue-state tracking as function calling: domains and slots are represented as structured function arguments and the LLM produces the corresponding state update.

Reference:
- Li et al., *Large Language Models as Zero-shot Dialogue State Tracker through Function Calling*, ACL 2024.

### Few-shot semantic guidance

The prompt can include a small number of contrastive examples to teach ambiguous constructions:

```text
"I'm exploring sweatshirts."
→ browsing intent
→ category: sweatshirt

"I need boots for exploring caves."
→ category: boots
→ use_case: exploring caves
```

This follows findings from schema-guided prompting work showing that demonstrations can improve semantic slot interpretation when schema names alone are insufficient.

Reference:
- Gupta et al., *Show, Don't Tell: Demonstrations Outperform Descriptions for Schema-Guided Task-Oriented Dialogue*, NAACL 2022.

### Incremental state updates

The LLM returns only the current-turn delta rather than regenerating the entire session state.

Example:

```text
Existing state:
category = shirt
use_case = sunny weather
color = black

User:
"Actually I'll mostly use it when it's raining."

LLM delta:
override = preference
fields = [use_case]
use_case = rain
```

The deterministic session manager then:

```text
remove old use_case
        ↓
invalidate dependent inferred constraints
        ↓
preserve independent explicit constraints
        ↓
apply new use_case
```

This follows the incremental update pattern used in modern dialogue-state tracking: the language model interprets the turn, while deterministic state-management logic owns persistence and conflict resolution.

### Design boundary

```text
LLM
= language understanding / slot extraction

Deterministic code
= validation
= dependency-aware override handling
= state mutation
= retrieval
= ranking
```

The LLM therefore improves contextual understanding without becoming the source of truth for session state or retrieval behavior.
