from __future__ import annotations

import json

import pytest

from annotation.material import (
    PROMPT_VERSION,
    build_material_prompt,
    parse_and_validate_material,
    run_material_annotation,
)


def test_material_response_has_exact_schema_and_normalized_values() -> None:
    assert parse_and_validate_material(
        '{"material": ["PU_Leather", "Lycra", "stainless steel"]}'
    ) == {"material": ["faux leather", "spandex", "stainless steel"]}

    with pytest.raises(ValueError):
        parse_and_validate_material('{"material": [], "feature": []}')


def test_prompt_contains_the_complete_row_and_main_product_rules() -> None:
    product = {
        "parent_asin": "A",
        "title": "Leather belt",
        "features": ["Stainless steel buckle"],
        "description": ["Gift box included"],
        "details": {"Material": "leather"},
        "price": 19.99,
    }
    prompt = build_material_prompt(product)
    row = json.dumps(product, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert row in prompt
    assert "MAIN PRODUCT" in prompt
    assert "packaging" in prompt
    assert PROMPT_VERSION == "v5-material-v1"


def test_material_runner_resumes_and_writes_catalog_order(tmp_path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "\n".join(
            json.dumps({"parent_asin": asin, "title": asin})
            for asin in ("A", "B", "C")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "material.jsonl"

    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def annotate(self, prompt: str) -> str:
            self.calls.append(prompt)
            return '{"material": ["polyester", "spandex"]}'

    client = Client()
    first = run_material_annotation(
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

    second = run_material_annotation(
        catalog,
        output,
        client,
        model="test",
        concurrency=2,
        retries=0,
    )
    assert second["processed_count"] == 0
    assert len(client.calls) == calls_after_first_run
