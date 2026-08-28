from __future__ import annotations

import json

import pytest

from annotation.category import (
    PROMPT_VERSION,
    build_category_prompt,
    parse_and_validate_category,
    run_category_annotation,
)


def test_category_response_has_exact_schema_and_natural_values() -> None:
    assert parse_and_validate_category(
        '{"category": ["Running_Shoes", "shoes"]}'
    ) == {"category": ["running shoes", "shoes"]}

    with pytest.raises(ValueError):
        parse_and_validate_category('{"category": [], "feature": []}')


def test_prompt_contains_the_complete_row_and_object_type_rules() -> None:
    product = {
        "parent_asin": "A",
        "title": "Hiking boots",
        "features": ["Rubber sole"],
        "categories": ["Clothing, Shoes & Jewelry", "Boots"],
        "details": {"Department": "Outdoor"},
        "price": 42.00,
    }
    prompt = build_category_prompt(product)
    row = json.dumps(product, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert row in prompt
    assert "MAIN PRODUCT" in prompt
    assert "navigation labels" in prompt
    assert PROMPT_VERSION == "v5-category-v1"


def test_category_runner_resumes_and_writes_catalog_order(tmp_path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "\n".join(
            json.dumps({"parent_asin": asin, "title": asin})
            for asin in ("A", "B", "C")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "category.jsonl"

    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def annotate(self, prompt: str) -> str:
            self.calls.append(prompt)
            return '{"category": ["shoes", "running shoes"]}'

    client = Client()
    first = run_category_annotation(
        catalog,
        output,
        client,
        model="test",
        concurrency=2,
        retries=0,
    )
    assert first["success_count"] == 3
    assert [json.loads(line)["parent_asin"] for line in output.read_text().splitlines()] == [
        "A",
        "B",
        "C",
    ]
    calls_after_first_run = len(client.calls)

    second = run_category_annotation(
        catalog,
        output,
        client,
        model="test",
        concurrency=2,
        retries=0,
    )
    assert second["processed_count"] == 0
    assert len(client.calls) == calls_after_first_run
