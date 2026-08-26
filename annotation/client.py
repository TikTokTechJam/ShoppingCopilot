from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol


class AnnotationClient(Protocol):
    def annotate(self, prompt: str) -> Any:
        """Return the model's structured JSON object or its JSON text."""


class HostedLLMClient:
    """Small OpenAI-compatible HTTP client with no hardcoded credentials."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint must be non-empty")
        if not model.strip():
            raise ValueError("model must be non-empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def annotate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the JSON object requested by the user.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"annotation endpoint returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"annotation endpoint request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("annotation endpoint returned invalid JSON") from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("annotation endpoint response has no chat content") from exc

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                part["text"]
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            if text_parts and len(text_parts) == len(content):
                return "".join(text_parts)
        raise RuntimeError("annotation endpoint returned unsupported content")
