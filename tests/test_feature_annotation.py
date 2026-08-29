from __future__ import annotations

import json

import pytest

from annotation.feature import (
    PROMPT_VERSION,
    build_feature_prompt,
    parse_and_validate_feature,
    run_feature_annotation,
)


def test_feature_response_has_exact_schema_and_natural_values() -> None:
    assert parse_and_validate_feature(
        '{"feature": ["Non_Slip", "machine washable"]}'
    ) == {"feature": ["non slip", "machine washable"]}

    with pytest.raises(ValueError):
        parse_and_validate_feature('{"feature": [], "material": []}')


def test_prompt_contains_the_complete_row_and_main_product_rules() -> None:
    product = {
        "parent_asin": "A",
        "title": "Rain jacket",
        "features": ["Waterproof shell"],
        "description": ["Includes a storage bag"],
        "details": {"Department": "Outdoor"},
        "price": 12.99,
    }
    prompt = build_feature_prompt(product)
    row = json.dumps(product, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert row in prompt
    assert "main product" in prompt
    assert "accessories" in prompt
    assert PROMPT_VERSION == "v5-feature-v1"


def test_feature_runner_resumes_and_writes_catalog_order(tmp_path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "\n".join(
            json.dumps({"parent_asin": asin, "title": asin})
            for asin in ("A", "B", "C")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "feature.jsonl"

    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def annotate(self, prompt: str) -> str:
            self.calls.append(prompt)
            return '{"feature": ["waterproof"]}'

    client = Client()
    first = run_feature_annotation(
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

    second = run_feature_annotation(
        catalog,
        output,
        client,
        model="test",
        concurrency=2,
        retries=0,
    )
    assert second["processed_count"] == 0
    assert len(client.calls) == calls_after_first_run
