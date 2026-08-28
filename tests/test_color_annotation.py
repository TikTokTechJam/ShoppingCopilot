from __future__ import annotations

import json

import pytest

from annotation.color import (
    PROMPT_VERSION,
    build_color_prompt,
    parse_and_validate_color,
    run_color_annotation,
)


def test_color_response_normalizes_common_variants() -> None:
    assert parse_and_validate_color(
        '{"color": ["Grey", "Navy Blue", "Golden"]}'
    ) == {"color": ["gray", "navy", "gold"]}
    assert parse_and_validate_color(
        '{"color": ["Multi_Color", "Off White"]}'
    ) == {"color": ["multicolor", "off white"]}


def test_color_response_has_exact_schema() -> None:
    assert parse_and_validate_color('{"color": []}') == {"color": []}

    with pytest.raises(ValueError):
        parse_and_validate_color('{"color": [], "material": []}')


def test_color_response_has_no_parser_value_limit() -> None:
    assert parse_and_validate_color(
        '{"color": ["black", "white", "red", "blue"]}'
    ) == {"color": ["black", "white", "red", "blue"]}


def test_prompt_contains_the_complete_row_and_main_product_rules() -> None:
    product = {
        "parent_asin": "A",
        "title": "Black rain jacket",
        "features": ["Waterproof shell"],
        "description": ["Includes a storage bag"],
        "details": {"Department": "Outdoor"},
        "price": 12.99,
    }
    prompt = build_color_prompt(product)
    row = json.dumps(product, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert row in prompt
    assert "main product" in prompt
    assert "packaging" in prompt
    assert PROMPT_VERSION == "v5-color-v1"


def test_color_runner_resumes_and_writes_catalog_order(tmp_path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "\n".join(
            json.dumps({"parent_asin": asin, "title": asin})
            for asin in ("A", "B", "C")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "color.jsonl"

    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def annotate(self, prompt: str) -> str:
            self.calls.append(prompt)
            return '{"color": ["Black", "White"]}'

    client = Client()
    first = run_color_annotation(
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
    assert json.loads(output.read_text().splitlines()[0])["color"] == ["black", "white"]
    calls_after_first_run = len(client.calls)

    second = run_color_annotation(
        catalog,
        output,
        client,
        model="test",
        concurrency=2,
        retries=0,
    )
    assert second["processed_count"] == 0
    assert len(client.calls) == calls_after_first_run
