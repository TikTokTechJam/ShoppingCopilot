"""Stage 2 of the intent router: Qwen3-Reranker-0.6B, run locally.

Why a reranker rather than a generative model: the task is a binary label, and
a reranker natively produces a relevance score per candidate label. That maps
directly onto the confidence the issue asks for, with no constrained decoding
and no sampling. The 2026 BTZSC zero-shot benchmark also reports 0.6B
rerankers surpassing every NLI cross-encoder, which is the alternative
zero-shot formulation.

Why zero-shot at all: this project has thirteen labelled examples -- the ones
written into issue #6. Nothing can be trained or fitted on that, so the model
has to work with no task-specific data. A reranker scored against two written
label descriptions does exactly that.

The model is *optional*. `onnxruntime` and `tokenizers` are not repo
dependencies, the weights are not committed, and every failure path returns
None so `CascadingIntentRouter` can fall back to the rules tier. The agent
must never fail closed because an optional asset is absent.

Setup:

    pip install -r requirements-reranker.txt
    python -m tools.fetch_reranker

Both are opt-in. Without them the router runs rules-only and still satisfies
every acceptance criterion in the issue.
"""

from __future__ import annotations

import atexit
import gc
import math
import os
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).parents[2]
DEFAULT_MODEL_DIR = ROOT / "assets" / "qwen3-reranker-0.6b-onnx"
ENV_MODEL_DIR = "SHOPPING_COPILOT_RERANKER_DIR"

MODEL_REPO = "onnx-community/Qwen3-Reranker-0.6B-ONNX"
MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "onnx/model_quantized.onnx",
)

# The reranker's own chat scaffold. It is trained to answer exactly "yes" or
# "no" at the first assistant position, so one forward pass is enough -- there
# is nothing to decode.
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

INSTRUCTION = (
    "Decide whether the shopper's message matches the described shopping behaviour."
)

# The two labels, written as descriptions rather than as words. The reranker
# scores the message against each; the pair of scores is the classifier.
# These strings ARE the model's task definition -- treat edits to them as a
# behaviour change and re-run the evaluation.
# Measured against the escalating slice of the dev set: this short, purely
# semantic pair scores 19/21 where an earlier attribute-checklist phrasing
# ("...has stated a budget, brand, colour, material, size or feature") scored
# 12/21. The checklist wording made the model read a bare category request
# such as "I'm looking for cardigans" as undecided, because no attribute was
# listed. Decidedness is the thing being classified, so say that and no more.
LABEL_QUERIES: dict[str, str] = {
    "BUYING": (
        "The shopper has already decided what they are looking for and is "
        "asking for it directly."
    ),
    "BROWSING": (
        "The shopper has not decided yet and is exploring, comparing, or "
        "asking for suggestions."
    ),
}

# Qwen3-0.6B config: 28 layers, 8 key/value heads, head dim 128.
_NUM_LAYERS = 28
_KV_HEADS = 8
_HEAD_DIM = 128

MAX_INPUT_TOKENS = 512


def resolve_model_dir(model_dir: str | os.PathLike[str] | None = None) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    from_env = os.environ.get(ENV_MODEL_DIR)
    return Path(from_env) if from_env else DEFAULT_MODEL_DIR


class QwenRerankerBackend:
    """Scores a message against both label descriptions. Deterministic.

    Loading is lazy: constructing the backend is free, and nothing is read
    from disk until the first message actually escalates.
    """

    name = "qwen3-reranker-0.6b"

    def __init__(
        self,
        model_dir: str | os.PathLike[str] | None = None,
        *,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        cache_size: int = 512,
    ) -> None:
        self.model_dir = resolve_model_dir(model_dir)
        self.max_input_tokens = max_input_tokens
        self._session = None
        self._tokenizer = None
        self._yes_id: int | None = None
        self._no_id: int | None = None
        self._load_error: str | None = None
        self._atexit_registered = False
        self._score_cached = lru_cache(maxsize=cache_size)(self._score_uncached)

    # -- availability -----------------------------------------------------

    def missing_requirements(self) -> list[str]:
        """Everything standing between this backend and a working call."""
        missing: list[str] = []
        for module in ("onnxruntime", "tokenizers", "numpy"):
            try:
                __import__(module)
            except ImportError:
                missing.append(f"package: {module}")
        for relative in ("tokenizer.json", "onnx/model_quantized.onnx"):
            if not (self.model_dir / relative).exists():
                missing.append(f"file: {self.model_dir / relative}")
        return missing

    def available(self) -> bool:
        return not self.missing_requirements()

    # -- loading ----------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._session is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            options = ort.SessionOptions()
            options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self.model_dir / "onnx" / "model_quantized.onnx"),
                options,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            self._yes_id = self._tokenizer.encode("yes", add_special_tokens=False).ids[0]
            self._no_id = self._tokenizer.encode("no", add_special_tokens=False).ids[0]
            if not self._atexit_registered:
                # Tear the session down while the interpreter is still healthy.
                # Left to garbage collection at shutdown, onnxruntime has been
                # seen to raise from its thread-pool destructor on Python 3.14
                # -- after all results are produced, but with a non-zero exit
                # code, which reads as a failing test run.
                atexit.register(self.close)
                self._atexit_registered = True
            return True
        except Exception as error:  # missing package, corrupt file, bad graph
            self._load_error = f"{type(error).__name__}: {error}"
            self._session = None
            return False

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def close(self) -> None:
        """Release the session deterministically. Safe to call more than once.

        Registered with `atexit` on first load, so the ONNX session is
        destroyed at a point where the interpreter can still run the
        destructor cleanly. Single-threaded session options would also avoid
        the problem, but measured 3.3x slower (466 ms against 142 ms per
        forward pass), which is far too much to pay for a rare unclean exit.
        """
        self._session = None
        self._tokenizer = None
        self._score_cached.cache_clear()
        gc.collect()

    # -- scoring ----------------------------------------------------------

    def _relevance(self, query: str, document: str) -> float:
        """P(yes) that `document` matches `query`, from one forward pass."""
        import numpy as np

        text = (
            _PREFIX
            + f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
            + _SUFFIX
        )
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) > self.max_input_tokens:
            # Truncate the middle rather than the scaffold: the suffix is what
            # puts the model at the answer position.
            keep = self.max_input_tokens // 2
            ids = ids[:keep] + ids[-keep:]

        length = len(ids)
        feeds = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, length), dtype=np.int64),
            "position_ids": np.arange(length, dtype=np.int64)[None, :],
        }
        empty = np.zeros((1, _KV_HEADS, 0, _HEAD_DIM), dtype=np.float32)
        for layer in range(_NUM_LAYERS):
            feeds[f"past_key_values.{layer}.key"] = empty
            feeds[f"past_key_values.{layer}.value"] = empty

        logits = self._session.run(["logits"], feeds)[0][0, -1]
        pair = np.array([logits[self._no_id], logits[self._yes_id]], dtype=np.float64)
        pair -= pair.max()
        exponentiated = np.exp(pair)
        return float(exponentiated[1] / exponentiated.sum())

    def _score_uncached(self, message: str) -> tuple[str, float] | None:
        if not self._ensure_loaded():
            return None

        scores = {
            label: self._relevance(query, message)
            for label, query in LABEL_QUERIES.items()
        }
        total = sum(scores.values())
        if total <= 0.0 or not math.isfinite(total):
            # The model declined both labels. Say so rather than inventing a
            # winner from floating-point noise.
            return None

        # Treating the two relevance scores as likelihoods under a uniform
        # prior makes the normalised score a posterior, which is what the
        # issue's `confidence` field is supposed to mean.
        posterior = {label: value / total for label, value in scores.items()}
        intent = max(posterior, key=posterior.__getitem__)
        confidence = min(max(posterior[intent], 0.5), 0.99)
        return intent, confidence

    def score(self, message: str) -> tuple[str, float] | None:
        """Return (intent, confidence), or None if the model cannot answer."""
        if not message or not message.strip():
            return None
        return self._score_cached(message)


def build_backend(
    model_dir: str | os.PathLike[str] | None = None,
) -> QwenRerankerBackend | None:
    """The backend if it can run here, otherwise None. Never raises."""
    backend = QwenRerankerBackend(model_dir)
    return backend if backend.available() else None
