from __future__ import annotations

import json

from annotation.style import (
    PROMPT_VERSION,
    build_style_prompt,
    parse_and_validate_style,
    run_style_annotation,
)


def test_style_response_has_exact_schema_and_natural_values() -> None:
    assert parse_and_validate_style(
        '{"style": ["High_Waisted", "Graphic Print"]}'
    ) == {"style": ["high waisted", "graphic print"]}

    assert parse_and_validate_style('{"style": []}') == {"style": []}
    assert parse_and_validate_style(
        '{"style": ["all black", "lightweight", "tunic", "floral"]}'
    ) == {"style": ["floral"]}

    try:
        parse_and_validate_style('{"style": [], "feature": []}')
    except ValueError:
        pass
    else:
        raise AssertionError("unexpected extra response field was accepted")


def test_style_prompt_contains_the_complete_row_and_style_rules() -> None:
    product = {
        "parent_asin": "A",
        "title": "High waisted floral skirt",
        "features": ["Elastic waistband"],
        "description": [],
        "price": 12.99,
    }
    prompt = build_style_prompt(product)
    row = json.dumps(product, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert row in prompt
    assert "main product" in prompt
    assert "style" in prompt
    assert PROMPT_VERSION == "v5-style-v3"


def test_style_runner_resumes_and_writes_catalog_order(tmp_path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "\n".join(
            json.dumps({"parent_asin": asin, "title": asin})
            for asin in ("A", "B", "C")
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "style.jsonl"

    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def annotate(self, prompt: str) -> str:
            self.calls.append(prompt)
            return '{"style": ["casual"]}'

    client = Client()
    first = run_style_annotation(
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

    second = run_style_annotation(
        catalog,
        output,
        client,
        model="test",
        concurrency=2,
        retries=0,
    )
    assert second["processed_count"] == 0
    assert len(client.calls) == calls_after_first_run
