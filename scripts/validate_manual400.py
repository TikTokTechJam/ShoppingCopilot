"""Read-only validation for the committed Manual400 benchmark artifacts.

This module intentionally has no builder or regeneration path. It only reads the
fixed benchmark files and checks their cross-file invariants.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from evaluator.manual400_evaluator import (
    ASIN_RE,
    METADATA_RE,
    SCENARIO_COUNTS,
    _id_from_field,
    effective_initial_fact_id,
    fact_id,
    fact_visible_in_message,
    load_catalog_ids,
    load_jsonl,
)


ROOT = Path(__file__).parents[1]
DEFAULT_CATALOG = ROOT / "data" / "catalog.jsonl"
DEFAULT_DATA_DIR = ROOT / "data" / "derived" / "manual400"
ARTIFACTS = (
    "selected_products.jsonl",
    "labeled_products.jsonl",
    "sessions.jsonl",
    "session_debug.jsonl",
    "label_audit.jsonl",
    "report.json",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fact_map(facts: object, context: str) -> dict[tuple[str, str], dict[str, object]]:
    _require(isinstance(facts, list), f"{context} must be a list")
    result: dict[tuple[str, str], dict[str, object]] = {}
    for fact in facts:
        _require(isinstance(fact, dict), f"{context} contains a malformed fact")
        key = fact_id(fact)
        _require(key is not None and all(key), f"{context} contains an incomplete fact")
        _require(key not in result, f"{context} contains duplicate fact {key}")
        result[key] = fact
    return result


def validate_benchmark(
    catalog_path: str | Path = DEFAULT_CATALOG,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, object]:
    data_root = Path(data_dir)
    paths = {name: data_root / name for name in ARTIFACTS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    _require(not missing, f"missing Manual400 artifacts: {missing}")

    selected = load_jsonl(paths["selected_products.jsonl"])
    labels = load_jsonl(paths["labeled_products.jsonl"])
    sessions = load_jsonl(paths["sessions.jsonl"])
    debug_rows = load_jsonl(paths["session_debug.jsonl"])
    audits = load_jsonl(paths["label_audit.jsonl"])
    report = json.loads(paths["report.json"].read_text(encoding="utf-8"))

    _require(len(selected) == 400, f"expected 400 selected products, got {len(selected)}")
    _require(len(labels) == 400, f"expected 400 labeled products, got {len(labels)}")
    _require(len(sessions) == 400, f"expected 400 sessions, got {len(sessions)}")
    _require(len(debug_rows) == 400, f"expected 400 debug rows, got {len(debug_rows)}")
    _require(len(audits) == 400, f"expected 400 audit rows, got {len(audits)}")
    _require(isinstance(report, dict), "report.json must contain an object")

    selected_ids = [str(row.get("parent_asin", "")) for row in selected]
    label_ids = [str(row.get("parent_asin", "")) for row in labels]
    session_ids = [str(row.get("target_asin", "")) for row in sessions]
    _require(all(selected_ids), "selected products contain an empty parent_asin")
    _require(all(label_ids), "labels contain an empty parent_asin")
    _require(all(session_ids), "sessions contain an empty target_asin")
    _require(len(set(selected_ids)) == 400, "selected products are not unique")
    _require(len(set(label_ids)) == 400, "labeled products are not unique")
    _require(len(set(session_ids)) == 400, "session targets are not unique")
    _require(set(selected_ids) == set(label_ids) == set(session_ids), "selected, labeled, and session targets differ")

    catalog_ids = load_catalog_ids(catalog_path)
    _require(set(selected_ids) <= catalog_ids, "selected product is absent from catalog")
    _require(set(session_ids) <= catalog_ids, "session target is absent from catalog")

    label_map: dict[str, dict[str, object]] = {}
    for label in labels:
        target = str(label["parent_asin"])
        _require(target not in label_map, f"duplicate label for {target}")
        label_map[target] = label
        _fact_map(label.get("validated_facts"), f"validated_facts for {target}")

    scenario_counts = Counter(str(row.get("scenario_type", "")) for row in sessions)
    _require(dict(scenario_counts) == SCENARIO_COUNTS, f"scenario counts are {dict(scenario_counts)}")
    debug_ids = {str(row.get("sample_id", "")) for row in debug_rows}
    audit_ids = {str(row.get("parent_asin", "")) for row in audits}
    session_sample_ids = {str(row.get("sample_id", "")) for row in sessions}
    _require(len(debug_ids) == 400 and debug_ids == session_sample_ids, "debug rows do not match sessions")
    _require(len(audit_ids) == 400 and audit_ids == set(label_ids), "audit rows do not match labels")

    for audit in audits:
        _require(audit.get("status") == "pass" and not audit.get("issues"), f"failed label audit for {audit.get('parent_asin')}")

    for session in sessions:
        sample_id = str(session.get("sample_id", ""))
        target = str(session.get("target_asin", ""))
        scenario = str(session.get("scenario_type", ""))
        hidden = session.get("hidden_facts")
        hidden_map = _fact_map(hidden, f"hidden_facts for {sample_id}")
        _require(2 <= len(hidden_map) <= 4, f"hidden card for {sample_id} must contain 2-4 facts")
        _require("category" not in {attribute for attribute, _ in hidden_map}, f"category leaked into hidden card for {sample_id}")

        label_facts = _fact_map(label_map[target].get("validated_facts"), f"validated_facts for {target}")
        for key, fact in hidden_map.items():
            _require(key in label_facts, f"hidden fact {key} is not a validated fact for {target}")
            _require(fact.get("evidence_field") and fact.get("evidence_text"), f"hidden fact {key} lacks evidence")
            _require(fact.get("evidence_field") == label_facts[key].get("evidence_field"), f"evidence field mismatch for {sample_id}:{key}")

        initial_id = _id_from_field(session.get("initial_fact_id"))
        _require(initial_id is None or initial_id in hidden_map, f"initial fact is outside hidden card for {sample_id}")
        effective_id = effective_initial_fact_id(session)
        if scenario == "browsing":
            _require(effective_id is None, f"Browsing initialized an initial fact for {sample_id}")
        if scenario == "boundary":
            _require(isinstance(session.get("boundary_first"), bool), f"boundary_first is not boolean for {sample_id}")
            if initial_id is not None:
                _require(initial_id in hidden_map, f"Boundary initial fact is outside hidden card for {sample_id}")
                explicitly_visible = fact_visible_in_message(hidden_map[initial_id], str(session.get("initial_message", "")))
                _require((effective_id is not None) == explicitly_visible, f"Boundary initial disclosure mismatch for {sample_id}")
        if scenario in {"buying", "intent_override"} and initial_id is not None:
            _require(effective_id is not None, f"initial fact is silently undisclosed for {sample_id}")

        messages = (str(session.get("initial_message", "")), str(session.get("override_message") or ""))
        _require(all(not ASIN_RE.search(message) and not METADATA_RE.search(message) for message in messages), f"identifier leakage in {sample_id}")

        if scenario == "intent_override":
            override_turn = session.get("override_turn")
            override_id = _id_from_field(session.get("override_fact_id"))
            _require(override_turn in {3, 4}, f"invalid override turn for {sample_id}")
            _require(override_id in hidden_map and override_id != initial_id, f"invalid override fact for {sample_id}")
            _require(bool(session.get("override_message")), f"missing override message for {sample_id}")
            _require(target == label_map[target]["parent_asin"], f"override changed target for {sample_id}")
        else:
            _require(session.get("override_turn") is None and session.get("override_fact_id") is None, f"unexpected override state for {sample_id}")

    report_counts = report.get("sessions_per_scenario")
    _require(report.get("total_products") == 400, "report total_products is not 400")
    _require(report_counts == SCENARIO_COUNTS, "report scenario counts do not match sessions")
    _require(report.get("audit_pass_count") == 400, "report audit_pass_count is not 400")
    _require(report.get("exact_scenario_counts") is True, "report does not confirm exact scenario counts")
    return {
        "catalog_products": len(catalog_ids),
        "selected_products": len(selected),
        "labeled_products": len(labels),
        "sessions": len(sessions),
        "scenario_counts": dict(scenario_counts),
        "audit_pass_count": len(audits),
        "read_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate fixed Manual400 benchmark artifacts without modifying them")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    summary = validate_benchmark(args.catalog, args.data_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
