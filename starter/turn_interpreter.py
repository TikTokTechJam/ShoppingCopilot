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


def _compact_text(value: object, *, limit: int = 2000) -> str:
    """Keep a model response readable on one console line."""

    text = " ".join(str(value or "").split())
    if not text:
        return "<empty>"
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _configured(value: object) -> str:
    return "set" if str(value or "").strip() else "unset"


@dataclass(frozen=True)
class TurnInterpretation:
    """Validated structured output for one user turn."""

    intent: str | None = None
    updates: dict[str, tuple[str, ...]] | None = None
    override_kind: str = "NONE"
    override_fields: tuple[str, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", _normalise_intent(self.intent))
        object.__setattr__(self, "updates", _normalise_updates(self.updates))
        object.__setattr__(self, "summary", _normalise_summary(self.summary))
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
            "summary": self.summary,
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


def _normalise_summary(value: object) -> str:
    """Keep the interpreter's current-turn extractive summary as plain text."""

    return str(value or "").strip()


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
        summary=payload.get("summary", ""),
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


TURN_INTERPRETER_SYSTEM_PROMPT = """You are a strict shopping dialogue-state tracker.

Return ONLY one JSON object.

Extract only product facts explicitly expressed in the CURRENT USER TURN.
Return a delta, never the complete previous state.

Allowed output shape:
{
"summary": "",
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
"type": "none" | "preference_override",
"fields": []
}
}

Schema:

category: the product type or shopping goal, such as sweatshirt, handbag,
or running shoes.

use_case: a real activity, occasion, weather, or situation for using the
product, such as hiking, running, office work, or rainy weather. Do NOT put
conversation framing such as exploring options, browsing, or comparing here.

feature: a functional product property, such as waterproof, lightweight, or
slip resistant.

color: an explicitly desired product color.

brand: an explicitly named brand.

material: an explicitly requested product material.

style: an explicitly requested style or fit.

price_min/price_max and size are retained only when explicitly stated; a
deterministic parser validates them separately.

SUMMARY RULES:

The "summary" field is an extractive, order-preserving representation of the
shopping-relevant information explicitly stated in the CURRENT USER TURN.

Construct summary using these rules:

Keep only words or phrases that describe the requested product or its
shopping constraints, including category, brand, color, material, feature,
use case, style, price, and size.

Remove conversational framing and filler such as:
"I'm looking for", "I want", "I need", "can you find", "show me",
"a key requirement is", "I would like", "please", and similar wording.

NEVER rewrite, normalize, lowercase, singularize, pluralize, spell-correct,
paraphrase, expand, or otherwise modify retained product text.

Retained text MUST be copied verbatim from the CURRENT USER TURN.

Preserve the original order of all retained product information.

Preserve capitalization exactly as written.

Preserve punctuation and symbols that are INTERNAL to a retained
product phrase or name, including characters such as:
&, ', ", -, /, +, commas, parentheses, and other symbols.

Examples:

"Dolce & Gabbana" -> "Dolce & Gabbana"

"Di'Or" -> "Di'Or"

"black-and-white" -> "black-and-white"

"T-shirts" -> "T-shirts"

Do NOT remove words merely because they are common linguistic stopwords
when they are part of a product phrase, brand, category, style, or other
shopping fact.

Examples:

"The North Face" MUST remain "The North Face"

"Dolce & Gabbana" MUST remain "Dolce & Gabbana"

"black and white" MUST remain "black and white" if that exact phrase
expresses the requested color/style.

When conversational text occurs between multiple retained shopping facts,
remove that conversational text and concatenate the retained spans in their
original order using exactly one space.

Remove punctuation that belongs only to discarded conversational framing
or sentence boundaries. Do not introduce new punctuation.

The summary must NEVER contain a word that does not occur in the current
user turn.

Do not use previous dialogue state when constructing summary. Summary
represents ONLY the CURRENT USER TURN.

If the current turn contains no explicit shopping-relevant product
information, return:
"summary": ""

Positive conversational wording is not a slot.
"exploring sweatshirts" means category=sweatshirts and no use_case.
"boots for exploring caves" means category=boots and
use_case=exploring caves.

For a preference override, set override.type to preference_override and list
the replaced fields.

Examples:

User:
I'm exploring sweatshirts and would like to compare some options.

Output:
{
"summary": "sweatshirts",
"updates": {
"category": ["sweatshirts"]
},
"override": {
"type": "none",
"fields": []
}
}

User:
I need boots for exploring caves.

Output:
{
"summary": "boots exploring caves",
"updates": {
"category": ["boots"],
"use_case": ["exploring caves"]
},
"override": {
"type": "none",
"fields": []
}
}

User:
Actually I'll mostly use it when it's raining.

Output:
{
"summary": "raining",
"updates": {
"use_case": ["rain"]
},
"override": {
"type": "preference_override",
"fields": ["use_case"]
}
}

User:
I'm looking for Rompers & Overalls Jumpsuits. A key requirement is: rayon.

Output:
{
"summary": "Rompers & Overalls Jumpsuits rayon",
"updates": {
"category": ["Rompers & Overalls Jumpsuits"],
"material": ["rayon"]
},
"override": {
"type": "none",
"fields": []
}
}

User:
I want something from Dolce & Gabbana, preferably black-and-white.

Output:
{
"summary": "Dolce & Gabbana black-and-white",
"updates": {
"brand": ["Dolce & Gabbana"],
"color": ["black-and-white"]
},
"override": {
"type": "none",
"fields": []
}
}
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
            f"response_chars={len(text)} parsed={'yes' if parsed is not None else 'no'} "
            f"llm_response={_compact_text(text)}"
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
            f"response_chars={len(response)} parsed={'yes' if parsed is not None else 'no'} "
            f"llm_response={_compact_text(response)}"
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
