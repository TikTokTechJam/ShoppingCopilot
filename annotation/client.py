from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol


def completion_url(endpoint: str) -> str:
    value = endpoint.strip().rstrip("/")
    if not value:
        raise ValueError("endpoint must be non-empty")
    if value.endswith("/chat/completions"):
        return value
    return value + "/chat/completions"


class AnnotationClient(Protocol):
    def annotate(self, prompt: str) -> Any:
        """Return the model's structured JSON object or its JSON text."""


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return f"annotation endpoint returned HTTP {exc.code}"
    error = body.get("error", body) if isinstance(body, dict) else {}
    message = error.get("message") if isinstance(error, dict) else None
    if isinstance(message, str) and message.strip():
        compact = " ".join(message.split())
        return f"annotation endpoint returned HTTP {exc.code}: {compact[:400]}"
    return f"annotation endpoint returned HTTP {exc.code}"


class HostedLLMClient:
    """Small OpenAI-compatible HTTP client with no hardcoded credentials."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        model: str,
        timeout: float = 60.0,
        max_tokens: int = 2048,
        json_mode: bool = True,
        thinking: bool = False,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint must be non-empty")
        if not model.strip():
            raise ValueError("model must be non-empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.thinking = thinking

    def annotate(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the JSON object requested by the user.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if not self.thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            completion_url(self.endpoint),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(_http_error_detail(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"annotation endpoint request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("annotation endpoint returned invalid JSON") from exc

        try:
            choice = response_payload["choices"][0]
            message = choice["message"]
            content = message["content"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("annotation endpoint response has no chat content") from exc

        if content is None:
            detail = f"; finish_reason={finish_reason}" if finish_reason else ""
            raise RuntimeError(
                "annotation endpoint returned empty content"
                f"{detail}; increase max_tokens"
            )
        if isinstance(content, str):
            if content.strip():
                return content
            detail = f"; finish_reason={finish_reason}" if finish_reason else ""
            raise RuntimeError(
                "annotation endpoint returned empty content"
                f"{detail}; increase max_tokens"
            )
        if isinstance(content, list):
            if not content:
                detail = f"; finish_reason={finish_reason}" if finish_reason else ""
                raise RuntimeError(
                    "annotation endpoint returned empty content"
                    f"{detail}; increase max_tokens"
                )
            text_parts = [
                part["text"]
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            if text_parts and len(text_parts) == len(content):
                return "".join(text_parts)
        raise RuntimeError("annotation endpoint returned unsupported content")
