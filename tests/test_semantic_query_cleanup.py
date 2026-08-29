from __future__ import annotations

from types import SimpleNamespace

from dictionary.registry import SEMANTIC_QUERY_STOPWORDS
from starter.routing.constraints import (
    _RESIDUAL_STOPWORDS,
    _semantic_ngrams,
    _semantic_text,
    _semantic_tokens,
    extract_constraints,
)


def test_positive_contractions_are_removed_before_normalization() -> None:
    assert _semantic_tokens("I'd") == ()
    assert _semantic_tokens("I’d") == ()
    assert _semantic_text("I'd like something for hiking") == "hiking"
    assert _semantic_text("I'm looking for something lightweight") == "lightweight"
    assert _semantic_text("I'd prefer polarized sunglasses") == "polarized sunglasses"
    assert _semantic_text("I'd rather have cotton") == "cotton"
    assert _semantic_text("I'd mainly use it for hiking") == "hiking"


def test_conversational_words_and_make_forms_are_removed() -> None:
    assert _semantic_text("Can you find me something for everyday wear?") == "everyday wear"
    assert _semantic_text("Actually, I'd prefer something waterproof") == "waterproof"
    assert _semantic_text("Please show me non-slip shoes") == "non slip shoes"
    assert _semantic_text("Can you make it waterproof?") == "waterproof"
    assert _semantic_text("Can you make something suitable for hiking?") == "suitable hiking"

    for form in ("make", "makes", "made", "making"):
        assert form not in _semantic_tokens(f"Please {form} it waterproof")


def test_meaningful_short_and_product_tokens_are_preserved() -> None:
    assert _semantic_text("Something with UV protection") == "uv protection"
    assert _semantic_text("id holder") == "id holder"
    assert _semantic_text("ID case") == "id case"
    assert _semantic_text("Levi's jeans") == "levis jeans"
    assert _semantic_text("O'Neill jacket") == "oneill jacket"


def test_negative_contractions_preserve_polarity() -> None:
    assert _semantic_text("I don't want leather") == "not leather"
    assert _semantic_text("I don’t want leather") == "not leather"
    assert _semantic_text("I can't use cotton") == "not cotton"
    assert _semantic_text("not red") == "not red"
    assert _semantic_text("no leather") == "no leather"
    assert _semantic_text("without laces") == "without laces"
    assert _semantic_text("avoid polyester") == "avoid polyester"
    assert _semantic_text("anything except cotton") == "except cotton"


def test_semantic_ngrams_use_clean_tokens_and_keep_existing_hyphen_behavior() -> None:
    assert _semantic_ngrams("I'd like something for hiking") == ("hiking",)
    assert _semantic_ngrams("I’m looking for something lightweight") == ("lightweight",)
    assert _semantic_ngrams("Please show me non-slip shoes") == (
        "non",
        "slip",
        "shoes",
        "non slip",
        "slip shoes",
        "non slip shoes",
    )
    assert _semantic_ngrams("I'd") == ()


def test_semantic_stopword_policy_has_one_shared_source() -> None:
    assert _RESIDUAL_STOPWORDS is SEMANTIC_QUERY_STOPWORDS
    assert "make" in SEMANTIC_QUERY_STOPWORDS
    assert "id" not in SEMANTIC_QUERY_STOPWORDS
    assert "not" not in SEMANTIC_QUERY_STOPWORDS
    assert "no" not in SEMANTIC_QUERY_STOPWORDS
    assert "without" not in SEMANTIC_QUERY_STOPWORDS
    assert "avoid" not in SEMANTIC_QUERY_STOPWORDS
    assert "except" not in SEMANTIC_QUERY_STOPWORDS


def test_extractor_passes_only_clean_ngrams_to_a_mock_semantic_matcher() -> None:
    calls: list[str] = []
    dictionary = SimpleNamespace(phrase_index={})

    def matcher(phrase: str):
        calls.append(phrase)
        return ()

    raw = "I'd like something for hiking"
    extract_constraints(
        raw,
        dictionary=dictionary,
        semantic_matcher=matcher,
    )

    assert calls == ["hiking"]
    assert raw == "I'd like something for hiking"
