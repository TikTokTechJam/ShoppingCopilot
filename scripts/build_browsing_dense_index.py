"""Build a Qwen3 dense index over Browsing product cards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.browsing.dense_retriever import load_qwen_browsing_encoder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_cards(path: Path) -> tuple[list[str], list[str]]:
    asins: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            asin = row.get("parent_asin")
            text = row.get("text")
            if not isinstance(asin, str) or not asin.strip():
                raise ValueError(f"{path}:{line_number}: missing parent_asin")
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number}: text must be a string")
            asin = asin.strip()
            if asin in asins:
                raise ValueError(f"{path}:{line_number}: duplicate parent_asin {asin}")
            asins.append(asin)
            texts.append(text)
    return asins, texts


def _atomic_npy(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def build_index(
    cards_path: str | Path,
    output_dir: str | Path,
    model: Any,
    *,
    model_name: str,
    batch_size: int,
    progress: bool,
) -> dict[str, Any]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    cards = Path(cards_path)
    output = Path(output_dir)
    asins, texts = _read_cards(cards)
    if not asins:
        raise ValueError("product card file is empty")

    batches: list[np.ndarray] = []
    dimension: int | None = None
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        encoded = model.embed_documents(texts[start:end])
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != end - start:
            raise ValueError(
                f"encoder returned invalid batch shape {matrix.shape} for rows {start}:{end}"
            )
        if dimension is None:
            dimension = int(matrix.shape[1])
        if int(matrix.shape[1]) != dimension:
            raise ValueError("encoder returned inconsistent dimensions")
        if not bool(np.isfinite(matrix).all()):
            raise ValueError(f"encoder returned non-finite values for rows {start}:{end}")
        batches.append(matrix)
        if progress:
            print(f"[browsing-dense] encoded {end:,}/{len(texts):,}", flush=True)

    assert dimension is not None
    embeddings = np.vstack(batches).astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not bool(np.isfinite(norms).all()):
        raise ValueError("embedding norms are non-finite")
    nonzero = norms > 0.0
    embeddings[nonzero] /= norms[nonzero, None].astype(np.float32)
    if not bool(np.isfinite(embeddings).all()):
        raise ValueError("normalized embeddings are non-finite")

    _atomic_npy(output / "product_embeddings.npy", embeddings)
    _atomic_text(
        output / "parent_asins.json",
        json.dumps(asins, ensure_ascii=False, separators=(",", ":")) + "\n",
    )
    manifest: dict[str, Any] = {
        "schema_version": "browsing-dense-index-v1",
        "representation": "v5_semantic_product_card",
        "cards_path": str(cards),
        "cards_sha256": _sha256(cards),
        "model": model_name,
        "query_format": "Instruct: <shopping instruction>\\nQuery:<active slot card>",
        "query_instruction": (
            "Retrieve products that best match the shopper's product type, intended "
            "use, desired features, and preferences."
        ),
        "document_format": "plain V5 semantic product card; no instruction/query prefix",
        "product_count": len(asins),
        "dimension": dimension,
        "dtype": "float32",
        "normalization": "l2",
        "batch_size": batch_size,
        "row_order": "product_cards.jsonl order, inherited from catalog.jsonl",
        "generated_at": None,
    }
    _atomic_text(
        output / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Qwen3 dense index over Browsing product cards."
    )
    parser.add_argument("--cards", default="data/derived/browsing_dense/product_cards.jsonl")
    parser.add_argument("--output-dir", default="data/derived/browsing_dense")
    parser.add_argument("--model", default="model/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    encoder = load_qwen_browsing_encoder(
        args.model,
        batch_size=args.batch_size,
        device=args.device,
        half_precision=args.half_precision,
        show_progress_bar=args.progress,
    )
    manifest = build_index(
        args.cards,
        args.output_dir,
        encoder,
        model_name=args.model,
        batch_size=args.batch_size,
        progress=args.progress,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
