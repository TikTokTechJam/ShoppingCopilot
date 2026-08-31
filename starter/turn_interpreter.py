"""Optional schema-guided turn interpretation for conversational slot filling.

The interpreter is deliberately a small boundary around an optional local
causal language model.  It returns only a validated *current-turn delta*;
session state remains owned and mutated by :mod:`starter.session`.

No model is loaded unless ``SHOPPING_TURN_INTERPRETER_MODEL`` points at a local
model directory. When it is absent or unusable, the Agent reports an
interpretation error instead of silently substituting deterministic categorical
extraction.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


TURN_FIELDS: tuple[str, ...] = (
    "category",
    "brand",
    "color",
    "material",
    "feature",
    "use_case",
    "style",
    "price_min",
    "price_max",
    "size",
)

_TURN_FIELD_SET = frozenset(TURN_FIELDS)
_INTENTS = frozenset({"BUYING", "BROWSING"})
_OVERRIDE_ALIASES = {
    "none": "NONE",
    "": "NONE",
    "null": "NONE",
    "preference": "PREFERENCE",
    "preference_override": "PREFERENCE",
    "full_goal": "FULL_GOAL",
    "full_goal_override": "FULL_GOAL",
    "full": "FULL_GOAL",
}
_OVERRIDE_FIELDS = frozenset((*TURN_FIELDS, "price"))


def _log(message: str) -> None:
    """Write an immediately visible diagnostic without using the prompt."""

    print(f"[turn_interpreter] {message}", flush=True)


def _compact_error(exc: BaseException, *secrets: object) -> str:
    """Return bounded error text with configured secrets redacted."""

    detail = " ".join(str(exc).split()) or type(exc).__name__
    for secret in secrets:
        value = str(secret or "")
        if value:
            detail = detail.replace(value, "<redacted>")
    return detail[:500]


def _configured(value: object) -> str:
    return "set" if str(value or "").strip() else "unset"


@dataclass(frozen=True)
class TurnInterpretation:
    """Validated structured output for one user turn."""

    intent: str | None = None
    updates: dict[str, tuple[str, ...]] | None = None
    override_kind: str = "NONE"
    override_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", _normalise_intent(self.intent))
        object.__setattr__(self, "updates", _normalise_updates(self.updates))
        object.__setattr__(
            self,
            "override_kind",
            _normalise_override_kind(self.override_kind),
        )
        object.__setattr__(
            self,
            "override_fields",
            _normalise_override_fields(self.override_fields),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent.lower() if self.intent else None,
            "updates": {
                field: list(values)
                for field, values in (self.updates or {}).items()
            },
            "override": {
                "type": self.override_kind.lower(),
                "fields": list(self.override_fields),
            },
        }


class TurnInterpreter(Protocol):
    """Small injectable interface used by :class:`starter.agent.Agent`."""

    def interpret(self, message: str, state: object) -> TurnInterpretation | None: ...


def _normalise_intent(value: object) -> str | None:
    if value is None:
        return None
    intent = str(value).strip().upper()
    return intent if intent in _INTENTS else None


def _normalise_override_kind(value: object) -> str:
    if isinstance(value, Mapping):
        value = value.get("type", value.get("kind", "NONE"))
    key = str(value or "").strip().casefold().replace("-", "_")
    return _OVERRIDE_ALIASES.get(key, "NONE")


def _normalise_updates(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for raw_field, raw_values in value.items():
        field = str(raw_field).strip().casefold()
        if field not in _TURN_FIELD_SET:
            continue
        values: list[str] = []
        candidates = [raw_values] if isinstance(raw_values, str) else raw_values
        if not isinstance(candidates, (list, tuple)):
            continue
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            text = candidate.strip()
            if text and text not in values:
                values.append(text)
        if values:
            result[field] = tuple(values)
    return result


def _normalise_override_fields(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[str] = []
    for raw_field in value:
        field = str(raw_field).strip().casefold()
        if field in {"price_min", "price_max"}:
            field = "price"
        if field in _OVERRIDE_FIELDS and field not in result:
            result.append(field)
    return tuple(result)


def _payload_from_text(text: str) -> Mapping[str, Any] | None:
    """Read the first JSON object from a model completion safely."""

    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        payload = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(candidate[start : end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return payload if isinstance(payload, Mapping) else None


def parse_turn_interpretation(payload: Mapping[str, Any] | str | object) -> TurnInterpretation | None:
    """Validate a model payload without allowing arbitrary state fields."""

    if isinstance(payload, str):
        payload = _payload_from_text(payload)
    if not isinstance(payload, Mapping):
        return None

    raw_updates = payload.get("updates", payload.get("constraints", {}))
    raw_override = payload.get("override")
    override_kind = _normalise_override_kind(raw_override)
    override_fields: object = payload.get("replaces", payload.get("replace", ()))
    if isinstance(raw_override, Mapping):
        override_fields = raw_override.get(
            "fields",
            raw_override.get("replaces", override_fields),
        )

    return TurnInterpretation(
        intent=payload.get("intent"),
        updates=_normalise_updates(raw_updates),
        override_kind=override_kind,
        override_fields=_normalise_override_fields(override_fields),
    )


def _json_state(state: object) -> dict[str, object]:
    constraints = getattr(state, "constraints", None)
    semantic = getattr(state, "semantic_constraints", None)
    return {
        "mode": getattr(state, "mode", None),
        "constraints": (
            constraints.as_dict()
            if hasattr(constraints, "as_dict")
            else {}
        ),
        "semantic_constraints": (
            semantic.as_dict()
            if hasattr(semantic, "as_dict")
            else {}
        ),
        "last_asked_attribute": getattr(state, "last_asked", None),
        "no_preference_attributes": sorted(
            str(value)
            for value in getattr(state, "no_preference_attributes", ())
        ),
    }


TURN_INTERPRETER_SYSTEM_PROMPT = """You are a strict shopping dialogue-state
turn interpreter.

Your ONLY task is to extract product-constraint changes explicitly expressed
in the CURRENT USER TURN.

Return ONLY one valid JSON object.

You do NOT classify the user's shopping intent.
You do NOT decide whether the session is Buying or Browsing.
Buying/Browsing routing is handled separately by deterministic logic using the
accumulated dialogue state.

Return a CURRENT-TURN DELTA only.
Never regenerate, summarize, infer, or copy the complete previous session state.

Allowed output shape:
{
  "updates": {
    "category": [],
    "brand": [],
    "color": [],
    "material": [],
    "feature": [],
    "use_case": [],
    "style": [],
    "price_min": [],
    "price_max": [],
    "size": []
  },
  "override": {
    "type": "none" | "preference_override" | "full_goal_override",
    "fields": []
  }
}

GENERAL EXTRACTION RULES

1. Extract only product facts explicitly expressed in the CURRENT USER TURN.

2. Do not infer product attributes from conversational wording, dialogue style,
   shopping uncertainty, or request phrasing.

3. Do not infer missing constraints from common sense.

4. Do not repeat constraints merely because they may have appeared in earlier
   turns.

5. Do not convert conversational actions into product attributes.

   Examples of conversational framing that are NOT slots:
   - exploring options
   - browsing
   - comparing products
   - looking around
   - showing me options
   - still deciding
   - not sure yet
   - open to suggestions
   - anything is fine

6. Statements that merely indicate uncertainty or lack of additional
   preferences produce no new product constraint.

   Example:
   User: I don't have any other preferences.
   Output:
   {
     "updates": {},
     "override": {"type":"none","fields":[]}
   }

7. A no-preference statement about a SPECIFIC existing attribute means that
   attribute should be cleared.

   Example:
   User: I don't care about the color anymore.
   Output:
   {
     "updates": {},
     "override":{
       "type":"preference_override",
       "fields":["color"]
     }
   }

SCHEMA

- category:
  The actual product type or product class requested by the shopper.
  Examples:
  sweatshirt, handbag, underwear, running shoes, office chair.

- brand:
  An explicitly named product brand.
  Brand is an exact structured constraint.
  Never infer, normalize, paraphrase, or semantically expand a brand.

- color:
  An explicitly requested product color.

- material:
  An explicitly requested physical material.
  Examples:
  leather, cotton, stainless steel, wool.

- feature:
  A functional or desired product property.
  Examples:
  waterproof, lightweight, breathable, slip resistant, machine washable.

- use_case:
  A real-world activity, occasion, environment, weather condition, or situation
  in which the shopper intends to use the product.
  Examples:
  hiking, running, office work, cosplay, rainy weather, winter travel.

  Conversational framing MUST NOT become a use_case.

- style:
  An explicitly requested visual style, design style, fashion style, or fit.
  Examples:
  minimalist, casual, vintage, slim fit, oversized.

- price_min / price_max:
  Extract only when an explicit numeric budget or price boundary is stated.
  A deterministic numeric parser validates the values separately.

- size:
  Extract only when an explicit size is stated.
  A deterministic parser validates size separately.

SEMANTIC CONSTRAINTS

The following fields may preserve the user's natural semantic phrase because a
downstream canonical matcher resolves them:

- category
- color
- material
- feature
- use_case
- style

Do not unnecessarily decompose a meaningful multi-word phrase.

Prefer:
  "thermal underwear"
  "slip resistant"
  "rainy weather"
  "walking around town"

over:
  "thermal"
  "slip"
  "rainy"
  "walking"

CONVERSATIONAL FRAMING VS PRODUCT MEANING

User:
I'm exploring sweatshirts and would like to compare some options.

Correct:
{
  "updates":{
    "category":["sweatshirts"]
  },
  "override":{
    "type":"none",
    "fields":[]
  }
}

"exploring" and "compare some options" describe the conversation.
They are not product constraints.

User:
I need boots for exploring caves.

Correct:
{
  "updates":{
    "category":["boots"],
    "use_case":["exploring caves"]
  },
  "override":{
    "type":"none",
    "fields":[]
  }
}

Here "exploring caves" describes how the product will actually be used and is
therefore a use_case.

OVERRIDE RULES

Use preference_override only when the user explicitly replaces, removes, or
changes one or more previously stated preferences.

Return only the affected field names in override.fields.

Example:
User:
Actually, I'll mostly use it when it's raining.

Output:
{
  "updates":{
    "use_case":["rain"]
  },
  "override":{
    "type":"preference_override",
    "fields":["use_case"]
  }
}

Example:
User:
Actually, black instead of blue.

Output:
{
  "updates":{
    "color":["black"]
  },
  "override":{
    "type":"preference_override",
    "fields":["color"]
  }
}

Example:
User:
Color doesn't matter anymore.

Output:
{
  "updates":{},
  "override":{
    "type":"preference_override",
    "fields":["color"]
  }
}

Use full_goal_override ONLY when the user explicitly abandons the previous
shopping goal and starts a different product search.

Examples:
- forget that, I need a backpack instead
- scratch that, let's look for headphones
- start over, I want running shoes
- ignore the previous search, find me a handbag

Example:
User:
Forget the shirts. I need a backpack for hiking instead.

Output:
{
  "updates":{
    "category":["backpack"],
    "use_case":["hiking"]
  },
  "override":{
    "type":"full_goal_override",
    "fields":[]
  }
}

Do NOT use full_goal_override merely because:
- the user is uncertain,
- the user says they are browsing,
- the user has no additional preferences,
- the user asks to see more options,
- the user compares products,
- the user changes one attribute.

OUTPUT REQUIREMENTS

- Return JSON only.
- Return no explanation.
- Return no markdown.
- Do not add fields outside the specified schema.
- Omit no product facts that are explicitly stated.
- Do not invent product facts that are not explicitly stated.
"""


def build_turn_prompt(message: str, state: object) -> str:
    return (
        f"{TURN_INTERPRETER_SYSTEM_PROMPT}\n\n"
        "PREVIOUS USER-VISIBLE STATE:\n"
        f"{json.dumps(_json_state(state), ensure_ascii=False, sort_keys=True)}\n\n"
        "CURRENT USER TURN:\n"
        f"{message}\n\n"
        "JSON DELTA:"
    )


class LocalTurnInterpreter:
    """Hugging Face local causal-LM adapter, loaded once by the Agent."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_input_tokens: int = 2048,
        max_new_tokens: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_input_tokens = max(256, int(max_input_tokens))
        self.max_new_tokens = max(32, int(max_new_tokens))
        self._torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.last_raw_response: object | None = None
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def interpret(self, message: str, state: object) -> TurnInterpretation | None:
        self.last_raw_response = None
        prompt = build_turn_prompt(message, state)
        started = time.perf_counter()
        _log(
            "local request start "
            f"device={self.device} message_chars={len(message)} "
            f"prompt_chars={len(prompt)}"
        )
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            prompt_length = inputs["input_ids"].shape[-1]
            completion = output[0][prompt_length:]
            text = self.tokenizer.decode(completion, skip_special_tokens=True)
            self.last_raw_response = text
            parsed = parse_turn_interpretation(text)
        except Exception as exc:
            _log(
                "local request failed "
                f"elapsed={time.perf_counter() - started:.2f}s "
                f"error_type={type(exc).__name__} "
                f"error={_compact_error(exc)}"
            )
            raise
        _log(
            "local response "
            f"elapsed={time.perf_counter() - started:.2f}s "
            f"response_chars={len(text)} parsed={'yes' if parsed is not None else 'no'}"
        )
        return parsed


class HostedTurnInterpreter:
    """OpenAI-compatible self-hosted adapter using the existing HTTP client."""

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 4000,
    ) -> None:
        from annotation.client import HostedLLMClient

        self.client = HostedLLMClient(
            endpoint,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            json_mode=True,
            thinking=False,
        )
        self.last_raw_response: object | None = None

    def interpret(self, message: str, state: object) -> TurnInterpretation | None:
        self.last_raw_response = None
        prompt = build_turn_prompt(message, state)
        started = time.perf_counter()
        _log(
            "hosted request start "
            f"model={self.client.model} message_chars={len(message)} "
            f"prompt_chars={len(prompt)} timeout={self.client.timeout:.1f}s"
        )
        try:
            response = self.client.annotate(prompt)
            self.last_raw_response = response
            parsed = parse_turn_interpretation(response)
        except Exception as exc:
            _log(
                "hosted request failed "
                f"elapsed={time.perf_counter() - started:.2f}s "
                f"error_type={type(exc).__name__} "
                f"error={_compact_error(exc, self.client.api_key)}"
            )
            raise
        _log(
            "hosted response "
            f"elapsed={time.perf_counter() - started:.2f}s "
            f"response_chars={len(response)} parsed={'yes' if parsed is not None else 'no'}"
        )
        return parsed


def build_turn_interpreter() -> TurnInterpreter | None:
    """Build the configured local or OpenAI-compatible backend once."""

    endpoint = (
        os.environ.get("SHOPPING_TURN_INTERPRETER_ENDPOINT", "").strip()
        or os.environ.get("ANNOTATION_BASE_URL", "").strip()
    )
    remote_model = (
        os.environ.get("SHOPPING_TURN_INTERPRETER_REMOTE_MODEL", "").strip()
        or os.environ.get("ANNOTATION_MODEL", "").strip()
    )
    api_key = (
        os.environ.get("SHOPPING_TURN_INTERPRETER_API_KEY")
        or os.environ.get("ANNOTATION_API_KEY")
    )
    configured = os.environ.get("SHOPPING_TURN_INTERPRETER_MODEL", "").strip()
    _log(
        "configuration "
        f"hosted_endpoint={_configured(endpoint)} "
        f"hosted_model={_configured(remote_model)} "
        f"api_key={_configured(api_key)} "
        f"local_model={_configured(configured)}"
    )
    if endpoint and remote_model:
        try:
            timeout = float(
                os.environ.get(
                    "SHOPPING_TURN_INTERPRETER_TIMEOUT",
                    os.environ.get("ANNOTATION_TIMEOUT", "60"),
                )
            )
            max_tokens = int(
                os.environ.get(
                    "SHOPPING_TURN_INTERPRETER_MAX_TOKENS",
                    "512",
                )
            )
            interpreter = HostedTurnInterpreter(
                endpoint,
                model=remote_model,
                api_key=api_key,
                timeout=timeout,
                max_tokens=max_tokens,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            _log(
                "hosted backend initialization failed "
                f"error_type={type(exc).__name__} "
                f"error={_compact_error(exc, api_key)}; "
                "Agent turns will report an LLM extraction error"
            )
            return None
        _log(
            "hosted backend ready "
            f"model={remote_model} timeout={timeout:.1f}s max_tokens={max_tokens}"
        )
        return interpreter

    if endpoint or remote_model:
        _log(
            "hosted backend not selected because configuration is incomplete "
            f"endpoint={_configured(endpoint)} model={_configured(remote_model)}; "
            "checking local model"
        )
    if not configured:
        _log(
            "no usable turn interpreter configured; set hosted endpoint+model "
            "or SHOPPING_TURN_INTERPRETER_MODEL"
        )
        return None
    model_path = Path(configured).expanduser()
    if not model_path.is_dir():
        _log(
            "local backend unavailable: model directory does not exist "
            f"path={model_path}; Agent turns will report an LLM extraction error"
        )
        return None
    started = time.perf_counter()
    _log(f"local backend initialization start path={model_path}")
    try:
        interpreter = LocalTurnInterpreter(model_path)
    except Exception as exc:
        _log(
            "local backend initialization failed "
            f"elapsed={time.perf_counter() - started:.2f}s "
            f"error_type={type(exc).__name__} "
            f"error={_compact_error(exc)}; "
            "Agent turns will report an LLM extraction error"
        )
        return None
    _log(
        "local backend ready "
        f"path={model_path} device={interpreter.device} "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )
    return interpreter


__all__ = [
    "HostedTurnInterpreter",
    "LocalTurnInterpreter",
    "TURN_FIELDS",
    "TURN_INTERPRETER_SYSTEM_PROMPT",
    "TurnInterpretation",
    "build_turn_interpreter",
    "build_turn_prompt",
    "parse_turn_interpretation",
]
