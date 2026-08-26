from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

from .client import AnnotationClient, HostedLLMClient
from .config import load_env_file
from .prompt import PROMPT_VERSION, build_annotation_prompt
from .schema import parse_and_validate_json, normalize_price, validate_annotation_record


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
    return {
        "parent_asin": str(product["parent_asin"]),
        "price": normalize_price(product.get("price")),
        "facts": dict(facts),
        "annotation": {
            "status": "success",
            "model": model,
            "prompt_version": prompt_version,
        },
    }


def _process_product(
    product: Mapping[str, Any],
    attempt: int,
    client: AnnotationClient,
    *,
    model: str,
    prompt_version: str,
    retries: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    parent_asin = str(product["parent_asin"])
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            raw_response = client.annotate(build_annotation_prompt(product))
            facts = parse_and_validate_json(raw_response)
            return _annotation_record(
                product,
                facts,
                model=model,
                prompt_version=prompt_version,
            ), None
        except Exception as exc:
            last_error = exc

    assert last_error is not None
    error = f"{type(last_error).__name__}: {str(last_error).strip()}".strip()
    return None, {
        "parent_asin": parent_asin,
        "error": error[:500],
        "attempt": attempt + retries,
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
) -> dict[str, Any]:
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if retries < 0:
        raise ValueError("retries must be non-negative")
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
            yield product, failure_attempts.get(parent_asin, 0) + 1

    if dry_run:
        for _ in pending_items():
            pass
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

            def process(item: tuple[dict[str, Any], int]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
                product, attempt = item
                return _process_product(
                    product,
                    attempt,
                    client,
                    model=model,
                    prompt_version=prompt_version,
                    retries=retries,
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
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe catalog annotation runner.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output-dir", default="data/derived/annotations/v1")
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
    parser.add_argument("--max-tokens", type=int, default=4096)
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
    model = args.model or os.environ.get("ANNOTATION_MODEL", "catalog-annotator-v1")

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
        )

    summary = run_annotation(
        args.catalog,
        args.output_dir,
        client,
        model=model,
        start=args.start,
        limit=args.limit,
        concurrency=args.concurrency,
        retries=args.retries,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
