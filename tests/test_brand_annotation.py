from __future__ import annotations

import json

import pytest

from annotation.brand import (
    PROMPT_VERSION,
    build_brand_prompt,
    parse_and_validate_brand,
    run_brand_annotation,
)


def test_brand_response_preserves_identity_format() -> None:
    assert parse_and_validate_brand(
        "{\"brand\": [\"New_Balance\", \"Dr. Martens\", \"Levi's\"]}"
    ) == {"brand": ["new balance", "dr. martens", "levi's"]}

    with pytest.raises(ValueError):
        parse_and_validate_brand('{"brand": [], "feature": []}')


def test_brand_response_allows_at_most_company_and_model() -> None:
    with pytest.raises(ValueError):
        parse_and_validate_brand('{"brand": ["nike", "air max 270", "running shoe"]}')


def test_prompt_contains_the_complete_row_and_identity_rules() -> None:
    product = {
        "parent_asin": "A",
        "title": "Nike Air Max 270",
        "brand": "Nike",
        "features": ["Air cushioning"],
        "details": {"Model": "270"},
        "price": 129.99,
    }
    prompt = build_brand_prompt(product)
    row = json.dumps(product, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert row in prompt
    assert "MAIN PRODUCT" in prompt
    assert "seller or store names" in prompt
    assert PROMPT_VERSION == "v5-brand-v1"


def test_brand_runner_resumes_and_writes_catalog_order(tmp_path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "\n".join(
            json.dumps({"parent_asin": asin, "title": asin})
            for asin in ("A", "B", "C")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "brand.jsonl"

    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def annotate(self, prompt: str) -> str:
            self.calls.append(prompt)
            return '{"brand": ["nike", "air max 270"]}'

    client = Client()
    first = run_brand_annotation(
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

    second = run_brand_annotation(
        catalog,
        output,
        client,
        model="test",
        concurrency=2,
        retries=0,
    )
    assert second["processed_count"] == 0
    assert len(client.calls) == calls_after_first_run
