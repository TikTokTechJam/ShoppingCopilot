## Dependency-Aware Selective Dialogue State Override

We maintain shopping preferences as structured dialogue state and update only the slots affected by the latest user turn.

This follows **SOM-DST** (Kim et al., ACL 2020), where dialogue slots are updated using operations such as `CARRYOVER`, `UPDATE`, and `DELETE` rather than rebuilding the full state each turn.

```text
Current state:
category = shirt
use_case = sunny weather
color = black

User:
"Actually I need it for rainy weather."

Result:
category → CARRYOVER
color    → CARRYOVER
use_case → UPDATE sunny → rainy
```

We additionally track constraint provenance and dependencies:

```text
use_case: sunny weather
    └── feature: UV protection [inferred]

category: shirt [explicit]
color: black [explicit]
```

When `sunny weather` is overridden, dependent inferred constraints are invalidated while unrelated explicit preferences are preserved.

This is inspired by **Truth Maintenance Systems** (Doyle, 1979), where beliefs are stored together with the dependencies that support them, allowing selective belief revision instead of global reset.

```text
User override
      ↓
Identify affected slot
      ↓
Dependency graph
      ↓
Remove invalid inferred descendants
      ↓
Preserve independent constraints
      ↓
Updated active state
```

We call this mechanism **Dependency-Aware Selective Dialogue State Override**.

References:
- Kim et al., *Efficient Dialogue State Tracking by Selectively Overwriting Memory*, ACL 2020.
- Doyle, *A Truth Maintenance System*, Artificial Intelligence, 1979.


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
    "type": "none | preference | full_goal",
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

## Slot-Guided BM25 Query Compilation

After the LLM turn interpreter produces the active shopping constraints, the runtime compiles those slots into BM25 retrieval queries rather than sending raw conversation history directly to BM25.

This follows conversational retrieval work such as **QuReTeC (SIGIR 2020)**, which shows that selecting only terms relevant to the current information need helps avoid query drift from stale conversational history.

```text
Conversation
    ↓
LLM slot filling
    ↓
Active session constraints
    ↓
BM25 Query Compiler
    ↓
Field-aware lexical retrieval
```

### Active state, not raw history

Only currently valid constraints are used for retrieval.

Example:

```text
Earlier:
category = shirt
use_case = sunny weather

Override:
use_case = rain
```

BM25 receives:

```text
shirt rain
```

rather than stale conversation text containing both `sunny` and `rain`.

This also makes preference overrides naturally remove obsolete lexical evidence.

### Slot-aware field routing

Slot type determines which product fields are most relevant for lexical retrieval.

This is motivated by **BM25F**, which extends BM25 to structured documents by weighting evidence from different fields differently.

Example routing:

```text
category
→ categories, title

brand
→ title, store

color
→ title, features

material
→ features, details

feature
→ features, details, description

use_case
→ features, description, categories

style
→ title, features, description

price
→ numeric filter, not BM25

size
→ structured filter, not BM25
```

Reference:
- Robertson, Zaragoza & Taylor, *Simple BM25 Extension to Multiple Weighted Fields*, CIKM 2004.

### Semantic expansion as concept groups

BGE is used to recover alternative lexical realizations of each semantic slot.

Example:

```text
feature:
"won't slip"

BGE:
slip resistant  0.92
non slip        0.88
traction        0.84
```

These terms are treated as lexical realizations of **one constraint**, not as three independent requirements.

```text
C_feature = {
    won't slip,
    slip resistant,
    non slip,
    traction
}
```

Likewise:

```text
C_use_case = {
    rain,
    rainy weather,
    wet weather
}
```

This design is motivated by query-expansion research showing that large synonym sets can cause query drift and allow one concept with many expansions to dominate retrieval.

Reference:
- Crimp & Trotman, *Automatic Term Reweighting for Query Expansion*, 2017.

### Do not enumerate synonym combinations

If there are five constraints and each has multiple BGE expansions, the runtime does **not** evaluate every Cartesian combination.

```text
20 × 20 × 20 × 20 × 20
```

is never required.

Instead, the runtime maintains a small number of independent concept groups:

```text
C1 = category
C2 = color
C3 = feature
C4 = use_case
C5 = material
```

Each group retrieves evidence for the same underlying user requirement.

### Conservative semantic expansion

More expansion is not automatically better.

For each slot, only a small high-confidence BGE set should be used for lexical expansion, for example:

```text
top 3 candidates
AND similarity >= threshold
AND reasonably close to the best match
```

Example:

```text
rain             0.96
rainy weather    0.89
wet weather      0.87
```

Lower-confidence related concepts such as `bad weather`, `all weather`, or unrelated neighbors should not be added merely because they exceed a loose global threshold.

The explicit slot value remains the strongest lexical evidence. Semantic expansions are supporting alternatives.

Conceptually:

```text
explicit value        weight 1.00
strong BGE synonym    lower confidence
weaker BGE synonym    lower confidence again
```

This is consistent with lexical query-weighting work such as **TW-BERT**, where terms are assigned different importance rather than being treated uniformly.

Reference:
- Dai et al., *End-to-End Query Term Weighting*, 2023.

### Retrieval strategy

The preferred runtime structure is:

```text
                     Active slots
                         ↓
                 BM25 Query Compiler
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
 structured filters   raw slot value   BGE expansions
 price / size            │                │
        │                └───────┬────────┘
        │                        ↓
        │                  concept groups
        │                        ↓
        └──────────────→ field-aware BM25
                                 ↓
                        constraint coverage
                                 ↓
                              Top-K
```

A product matching several different user constraints should rank above one that matches many synonyms from only a single constraint.

For example:

```text
Product A
category   ✓
color      ✓
feature    ✓
use_case   ✓

Product B
category   ✓
color      ✗
feature    ✗
use_case   ✓✓✓✓✓
```

Product A should receive stronger overall evidence because it covers more independent requirements.

### Implementation boundary

The responsibilities remain separated:

```text
LLM
→ determine active semantic slots

BGE
→ recover lexical/synonym alternatives

BM25
→ efficiently retrieve products containing relevant lexical evidence

Structured logic
→ enforce numeric/discrete constraints such as budget and size

Session state
→ remove obsolete constraints after overrides
```

The objective is therefore not to replace BM25 with semantic retrieval, but to give BM25 a cleaner, context-aware, semantically expanded query derived from the active dialogue state.
