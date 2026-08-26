from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

from .client import AnnotationClient, HostedLLMClient
from .config import load_env_file
from .prompt import PROMPT_VERSION, build_annotation_prompt
from .schema import (
    deterministic_catalog_brand,
    normalize_catalog_categories,
    normalize_price,
    parse_and_validate_json,
    validate_annotation_record,
)


def iter_catalog(path: str | Path) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                product = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(product, dict):
                raise ValueError(f"{path}:{line_number}: catalog row must be an object")
            parent_asin = product.get("parent_asin")
            if not isinstance(parent_asin, str) or not parent_asin.strip():
                raise ValueError(f"{path}:{line_number}: missing parent_asin")
            if parent_asin in seen:
                raise ValueError(f"{path}:{line_number}: duplicate parent_asin {parent_asin}")
            seen.add(parent_asin)
            yield product


def _read_annotation_state(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            record = validate_annotation_record(raw)
            parent_asin = record["parent_asin"]
            if parent_asin in completed:
                raise ValueError(f"{path}:{line_number}: duplicate annotation {parent_asin}")
            completed.add(parent_asin)
    return completed


def _read_failure_attempts(path: Path) -> dict[str, int]:
    attempts: dict[str, int] = {}
    if not path.exists():
        return attempts
    expected = {"parent_asin", "error", "attempt"}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(raw, dict) or set(raw) != expected:
                raise ValueError(f"{path}:{line_number}: malformed failure record")
            parent_asin = raw.get("parent_asin")
            attempt = raw.get("attempt")
            if not isinstance(parent_asin, str) or not parent_asin.strip():
                raise ValueError(f"{path}:{line_number}: invalid failure parent_asin")
            if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
                raise ValueError(f"{path}:{line_number}: invalid failure attempt")
            normalized_asin = parent_asin.strip()
            attempts[normalized_asin] = max(attempts.get(normalized_asin, 0), attempt)
    return attempts


def _annotation_record(
    product: Mapping[str, Any],
    facts: Mapping[str, Any],
    *,
    model: str,
    prompt_version: str,
) -> dict[str, Any]:
    combined_facts = {
        "category": normalize_catalog_categories(product.get("categories")),
        "brand": deterministic_catalog_brand(product),
        **dict(facts),
    }
    return validate_annotation_record({
        "parent_asin": str(product["parent_asin"]),
        "price": normalize_price(product.get("price")),
        "facts": combined_facts,
        "annotation": {
            "status": "success",
            "model": model,
            "prompt_version": prompt_version,
        },
    })


def _emit_progress(message: str, *, enabled: bool, lock: threading.Lock) -> None:
    if not enabled:
        return
    with lock:
        print(f"[annotate_catalog] {message}", file=sys.stderr, flush=True)


def _compact_error(exc: Exception) -> str:
    detail = " ".join(str(exc).split())
    if not detail:
        detail = type(exc).__name__
    return detail[:240]


def _retry_kind(exc: Exception, retry_index: int) -> str | None:
    detail = str(exc).lower()
    if (
        "finish_reason=length" in detail
        or "maximum context length" in detail
        or "context length" in detail
    ):
        return None
    if isinstance(exc, (ValueError, TypeError)):
        return "schema" if retry_index == 0 else None
    if isinstance(exc, RuntimeError) and (
        "http 429" in detail
        or "http 500" in detail
        or "http 502" in detail
        or "http 503" in detail
        or "http 504" in detail
        or "request failed" in detail
        or "timed out" in detail
        or "temporarily" in detail
    ):
        return "transient"
    return None


def _process_product(
    product: Mapping[str, Any],
    attempt: int,
    client: AnnotationClient,
    *,
    model: str,
    prompt_version: str,
    retries: int,
    progress: bool,
    progress_lock: threading.Lock,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    parent_asin = str(product["parent_asin"])
    last_error: Exception | None = None
    request_elapsed = 0.0
    final_attempt = attempt
    product_started = time.perf_counter()
    total_attempts = retries + 1
    for retry_index in range(total_attempts):
        current_attempt = attempt + retry_index
        final_attempt = current_attempt
        attempt_started = time.perf_counter()
        _emit_progress(
            f"request parent_asin={parent_asin} attempt={current_attempt} "
            f"of={attempt + retries}",
            enabled=progress,
            lock=progress_lock,
        )
        try:
            retry_instruction = None
            if retry_index == 1:
                retry_instruction = (
                    "The previous response failed schema validation. Return exactly "
                    "the six requested arrays, with no extra keys or prose."
                )
            raw_response = client.annotate(
                build_annotation_prompt(product, retry_instruction=retry_instruction)
            )
            facts = parse_and_validate_json(raw_response)
            request_elapsed = time.perf_counter() - attempt_started
            product_elapsed = time.perf_counter() - product_started
            _emit_progress(
                f"success parent_asin={parent_asin} attempt={current_attempt} "
                f"request_elapsed={request_elapsed:.1f}s total_elapsed={product_elapsed:.1f}s",
                enabled=progress,
                lock=progress_lock,
            )
            return _annotation_record(
                product,
                facts,
                model=model,
                prompt_version=prompt_version,
            ), None
        except Exception as exc:
            last_error = exc
            request_elapsed = time.perf_counter() - attempt_started
            product_elapsed = time.perf_counter() - product_started
            retry_kind = _retry_kind(exc, retry_index)
            can_retry = retry_index + 1 < total_attempts and retry_kind is not None
            if can_retry:
                _emit_progress(
                    f"retry parent_asin={parent_asin} after_attempt={current_attempt} "
                    f"next_attempt={current_attempt + 1} kind={retry_kind} "
                    f"request_elapsed={request_elapsed:.1f}s "
                    f"total_elapsed={product_elapsed:.1f}s error={_compact_error(exc)}",
                    enabled=progress,
                    lock=progress_lock,
                )
                if retry_kind == "transient":
                    time.sleep(min(0.25 * (2 ** retry_index), 2.0))
            else:
                reason = retry_kind or "non_retryable"
                if retry_index + 1 >= total_attempts:
                    reason = "retry_budget_exhausted"
                _emit_progress(
                    f"no_retry parent_asin={parent_asin} attempt={current_attempt} "
                    f"reason={reason} request_elapsed={request_elapsed:.1f}s "
                    f"total_elapsed={product_elapsed:.1f}s error={_compact_error(exc)}",
                    enabled=progress,
                    lock=progress_lock,
                )
                break

    assert last_error is not None
    error = f"{type(last_error).__name__}: {str(last_error).strip()}".strip()
    total_elapsed = time.perf_counter() - product_started
    _emit_progress(
        f"failed parent_asin={parent_asin} attempt={final_attempt} "
        f"request_elapsed={request_elapsed:.1f}s total_elapsed={total_elapsed:.1f}s "
        f"error={_compact_error(last_error)}",
        enabled=progress,
        lock=progress_lock,
    )
    return None, {
        "parent_asin": parent_asin,
        "error": error[:500],
        "attempt": final_attempt,
    }


def _write_line(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def run_annotation(
    catalog_path: str | Path,
    output_dir: str | Path,
    client: AnnotationClient | None,
    *,
    model: str,
    prompt_version: str = PROMPT_VERSION,
    start: int = 0,
    limit: int | None = None,
    concurrency: int = 1,
    retries: int = 2,
    dry_run: bool = False,
    progress: bool = False,
    log_every: int = 1,
    max_tokens: int | None = None,
    thinking: bool | None = None,
) -> dict[str, Any]:
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if log_every < 1:
        raise ValueError("log_every must be positive")
    if not dry_run and client is None:
        raise ValueError("client is required unless dry_run is enabled")

    output_path = Path(output_dir)
    annotations_path = output_path / "annotations.jsonl"
    failures_path = output_path / "failures.jsonl"
    manifest_path = output_path / "manifest.json"
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    completed = _read_annotation_state(annotations_path)
    failure_attempts = _read_failure_attempts(failures_path)
    end = None if limit is None else start + limit
    source_count = 0
    selected_count = 0
    pending_count = 0
    new_successes = 0
    new_failures = 0
    progress_lock = threading.Lock()
    run_started = time.perf_counter()

    _emit_progress(
        f"start catalog={catalog_path} output_dir={output_path} "
        f"start={start} limit={limit if limit is not None else 'all'} "
        f"concurrency={concurrency} retries={retries} dry_run={dry_run}",
        enabled=progress,
        lock=progress_lock,
    )

    def pending_items() -> Iterable[tuple[dict[str, Any], int]]:
        nonlocal source_count, selected_count, pending_count
        for index, product in enumerate(iter_catalog(catalog_path)):
            source_count += 1
            if index < start or (end is not None and index >= end):
                continue
            selected_count += 1
            parent_asin = str(product["parent_asin"])
            if parent_asin in completed:
                continue
            pending_count += 1
            if pending_count <= 5 or pending_count % log_every == 0:
                _emit_progress(
                    f"queued parent_asin={parent_asin} pending={pending_count}",
                    enabled=progress,
                    lock=progress_lock,
                )
            yield product, failure_attempts.get(parent_asin, 0) + 1

    if dry_run:
        for _ in pending_items():
            pass
        _emit_progress(
            f"dry_run_complete scanned={source_count} selected={selected_count} "
            f"pending={pending_count} elapsed={time.perf_counter() - run_started:.1f}s",
            enabled=progress,
            lock=progress_lock,
        )
        return {
            "dry_run": True,
            "source_product_count": source_count,
            "selected_product_count": selected_count,
            "pending": pending_count,
            "successful": len(completed),
            "failed": len(set(failure_attempts) - completed),
        }

    assert client is not None
    with (
        annotations_path.open("a", encoding="utf-8") as annotation_handle,
        failures_path.open("a", encoding="utf-8") as failure_handle,
        ThreadPoolExecutor(max_workers=concurrency) as executor,
    ):
        batch: list[tuple[dict[str, Any], int]] = []

        def flush() -> None:
            nonlocal new_successes, new_failures
            if not batch:
                return
            batch_size = len(batch)
            _emit_progress(
                f"batch_start size={batch_size}",
                enabled=progress,
                lock=progress_lock,
            )

            def process(item: tuple[dict[str, Any], int]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
                product, attempt = item
                return _process_product(
                    product,
                    attempt,
                    client,
                    model=model,
                    prompt_version=prompt_version,
                    retries=retries,
                    progress=progress,
                    progress_lock=progress_lock,
                )

            for record, failure in executor.map(process, batch):
                if record is not None:
                    _write_line(annotation_handle, record)
                    completed.add(record["parent_asin"])
                    new_successes += 1
                else:
                    assert failure is not None
                    _write_line(failure_handle, failure)
                    failure_attempts[failure["parent_asin"]] = failure["attempt"]
                    new_failures += 1
            _emit_progress(
                f"batch_complete size={batch_size} new_successes={new_successes} "
                f"new_failures={new_failures}",
                enabled=progress,
                lock=progress_lock,
            )
            batch.clear()

        for item in pending_items():
            batch.append(item)
            if len(batch) >= max(1, concurrency * 4):
                flush()
        flush()

    manifest = {
        "version": prompt_version,
        "source_catalog": str(catalog_path),
        "source_product_count": source_count,
        "model": model,
        "prompt_version": prompt_version,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "retries": retries,
        "selected_product_count": selected_count,
        "processed_this_run": new_successes + new_failures,
        "successful": len(completed),
        "failed": len(set(failure_attempts) - completed),
        "failure_attempts": sum(failure_attempts.values()),
    }
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    _emit_progress(
        f"complete selected={selected_count} processed={new_successes + new_failures} "
        f"successful={len(completed)} failed={len(set(failure_attempts) - completed)} "
        f"elapsed={time.perf_counter() - run_started:.1f}s",
        enabled=progress,
        lock=progress_lock,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe catalog annotation runner.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output-dir", default="data/derived/annotations/v2")
    parser.add_argument(
        "--env-file",
        help="Optional local KEY=VALUE file; never commit this file.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Base URL ending in /v1 or full /chat/completions URL.",
    )
    parser.add_argument("--api-key-env", default="ANNOTATION_API_KEY")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable model reasoning; disabled by default for extraction throughput.",
    )
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Omit response_format for endpoints that do not support JSON mode.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log every N queued products; request/retry results are always logged.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs; final JSON summary is still printed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")

    if args.env_file:
        try:
            load_env_file(args.env_file)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

    endpoint = (
        args.endpoint
        or os.environ.get("ANNOTATION_BASE_URL")
        or os.environ.get("ANNOTATION_ENDPOINT")
    )
    model = args.model or os.environ.get("ANNOTATION_MODEL", "catalog-annotator-v2")

    if args.dry_run:
        client = None
    else:
        if not endpoint:
            parser.error(
                "--endpoint, ANNOTATION_BASE_URL, or ANNOTATION_ENDPOINT "
                "is required unless --dry-run is used"
            )
        client = HostedLLMClient(
            endpoint,
            api_key=os.environ.get(args.api_key_env),
            model=model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            json_mode=not args.no_json_mode,
            thinking=args.thinking,
        )

    summary = run_annotation(
        args.catalog,
        args.output_dir,
        client,
        model=model,
        prompt_version=PROMPT_VERSION,
        start=args.start,
        limit=args.limit,
        concurrency=args.concurrency,
        retries=args.retries,
        dry_run=args.dry_run,
        progress=not args.quiet,
        log_every=args.log_every,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
