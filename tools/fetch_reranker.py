"""Download the optional Qwen3-Reranker-0.6B ONNX weights.

    python -m tools.fetch_reranker

The weights are ~1.2 GB and are deliberately not committed. The intent router
runs rules-only without them; this is opt-in.
"""

from __future__ import annotations

import argparse
import sys

from starter.routing.local_model import MODEL_FILES, MODEL_REPO, resolve_model_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=None, help="target directory")
    parser.add_argument("--repo", default=MODEL_REPO)
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is not installed.\n"
            "  pip install -r requirements-reranker.txt",
            file=sys.stderr,
        )
        return 1

    destination = resolve_model_dir(args.dest)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"{args.repo} -> {destination}")

    for name in MODEL_FILES:
        path = hf_hub_download(repo_id=args.repo, filename=name, local_dir=str(destination))
        size = path and (destination / name).stat().st_size
        print(f"  {name:36} {size / 1e6:8.1f} MB")

    print("\nDone. The router will pick the model up automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
