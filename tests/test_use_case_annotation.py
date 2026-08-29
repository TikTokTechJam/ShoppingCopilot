from __future__ import annotations

import json

import pytest

from annotation.use_case import (
    PROMPT_VERSION,
    build_use_case_prompt,
    parse_and_validate_use_case,
    run_use_case_annotation,
)


def test_v5_response_is_exactly_one_use_case_field() -> None:
    assert parse_and_validate_use_case('{"use_case": ["Trail_Running", "hiking"]}') == {
        "use_case": ["trail running", "hiking"]
    }

    with pytest.raises(ValueError):
        parse_and_validate_use_case('{"use_case": [], "feature": []}')


def test_prompt_contains_the_entire_catalog_row() -> None:
    product = {
        "parent_asin": "A",
        "title": "Example",
        "features": ["For hiking"],
        "details": {"Department": "Outdoor"},
        "price": 12.99,
    }
    prompt = build_use_case_prompt(product)
    row = json.dumps(product, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert row in prompt
    assert PROMPT_VERSION == "v5-use-case-v1"


def test_run_resumes_successes_and_materializes_catalog_order(tmp_path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "\n".join(
            json.dumps({"parent_asin": asin, "title": asin})
            for asin in ("A", "B", "C")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "use_case.jsonl"

    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def annotate(self, prompt: str) -> str:
            self.calls.append(prompt)
            return '{"use_case": ["everyday wear"]}'

    client = Client()
    first = run_use_case_annotation(
        catalog,
        output,
        client,
        model="test",
        concurrency=2,
        retries=0,
        progress=False,
    )
    assert first["success_count"] == 3
    assert [json.loads(line)["parent_asin"] for line in output.read_text().splitlines()] == [
        "A",
        "B",
        "C",
    ]
    calls_after_first_run = len(client.calls)
    second = run_use_case_annotation(
        catalog,
        output,
        client,
        model="test",
        concurrency=2,
        retries=0,
        progress=False,
    )
    assert second["processed_count"] == 0
    assert len(client.calls) == calls_after_first_run
