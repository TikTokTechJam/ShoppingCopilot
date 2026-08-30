"""Shared endpoint access for the generator and the judge.

The specification requires the same model the annotation workflow uses, so
this wraps ``annotation.client.HostedLLMClient`` and the same
``ANNOTATION_*`` environment contract rather than introducing a second
credential path.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlparse

from annotation.client import HostedLLMClient
from annotation.config import load_env_file


DEFAULT_ENV_FILE = ".env"
DEFAULT_MODEL = "catalog-annotator-v4"
API_KEY_ENV = "ANNOTATION_API_KEY"

# Hosts that wrap a real URL in a tracking redirect. POSTing to one would send
# the Authorization header to the wrapper instead of the model host, so a
# pasted share link is refused rather than tried.
_LINK_SHIM_HOSTS = frozenset(
    {
        "l.messenger.com", "l.facebook.com", "lm.facebook.com",
        "l.instagram.com", "t.co", "lnkd.in", "href.li", "out.reddit.com",
    }
)


class EndpointUnavailable(RuntimeError):
    """Raised when no usable annotation endpoint is configured."""


def _diagnose(endpoint: str) -> str:
    parsed = urlparse(endpoint.strip())
    if parsed.scheme not in ("http", "https"):
        return f"endpoint must be http(s), got {parsed.scheme or 'no scheme'!r}"
    host = parsed.netloc.casefold()
    if not host:
        return "endpoint has no host"
    if host in _LINK_SHIM_HOSTS:
        wrapped = parse_qs(parsed.query).get("u", [None])[0]
        detail = (
            f"endpoint host is the link redirector {host!r}, not a model host; "
            "refusing to send the API key there"
        )
        if wrapped:
            detail += f". The wrapped URL looks like: {unquote(wrapped)}"
        return detail
    return ""


def build_client(
    *,
    env_file: str | None = DEFAULT_ENV_FILE,
    timeout: float = 60.0,
    max_tokens: int = 2048,
) -> HostedLLMClient:
    """Return the annotation model client, or raise ``EndpointUnavailable``."""

    if env_file and os.path.exists(env_file):
        try:
            load_env_file(env_file)
        except (OSError, ValueError) as exc:
            raise EndpointUnavailable(f"could not read {env_file}: {exc}") from exc

    endpoint = (
        os.environ.get("ANNOTATION_BASE_URL")
        or os.environ.get("ANNOTATION_ENDPOINT")
        or ""
    )
    if not endpoint:
        raise EndpointUnavailable(
            "set ANNOTATION_BASE_URL (or ANNOTATION_ENDPOINT) to the model endpoint"
        )
    problem = _diagnose(endpoint)
    if problem:
        raise EndpointUnavailable(problem)

    return HostedLLMClient(
        endpoint,
        api_key=os.environ.get(API_KEY_ENV),
        model=os.environ.get("ANNOTATION_MODEL", DEFAULT_MODEL),
        timeout=timeout,
        max_tokens=max_tokens,
        json_mode=True,
    )


def preflight(client: Any) -> str:
    """One minimal round trip. Returns '' on success, else the failure text."""

    try:
        raw = client.annotate('Reply with exactly this JSON: {"ok": true}')
        parse_json(raw)
    except Exception as exc:  # noqa: BLE001 - the point is to report it
        return f"{type(exc).__name__}: {exc}"
    return ""


def parse_json(raw: Any) -> Any:
    """Parse a model reply, tolerating ```json fences."""

    if isinstance(raw, (Mapping, list)):
        return raw
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    return json.loads(text)


__all__ = ["API_KEY_ENV", "EndpointUnavailable", "build_client", "parse_json", "preflight"]
