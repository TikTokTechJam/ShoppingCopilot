"""Build the derived Tier 4 raw-text view without changing catalog.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from product_embeddings import build_tier4_raw_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Tier 4 raw product text from the immutable catalog."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--output",
        default="data/derived/tier4/raw_text.jsonl",
        help="Output JSONL containing parent_asin, title, features, description, and details.",
    )
    args = parser.parse_args()
    print(json.dumps(build_tier4_raw_text(args.catalog, args.output), indent=2))


if __name__ == "__main__":
    main()
