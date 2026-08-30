"""Build compact V5 semantic product cards for Browsing dense retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.browsing.product_cards import build_product_cards


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build V5 semantic product-card JSONL for Browsing retrieval."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--annotations",
        default="data/derived/annotations/v5/annotations.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="data/derived/browsing_dense",
    )
    args = parser.parse_args()
    manifest = build_product_cards(args.catalog, args.annotations, args.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
