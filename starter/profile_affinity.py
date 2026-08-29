"""Profile-conditioned prior over clarification attributes.

The clarification utility in ``starter.clarification`` estimates how well an
attribute *splits the candidate pool*.  It cannot estimate whether the shopper
will actually answer.  ``user_profile['preference_tags']`` is evidence for
exactly that second factor, so this module produces a **likelihood ratio** on
the odds that the shopper answers, which ``_answer_probability`` combines with
the mode prior.  The ratio is bounded, so it reorders near-ties and never
vetoes; the one exception is an attribute the shopper has explicitly declined,
where the ratio is driven to zero by direct evidence rather than by the prior.

No tag vocabulary is hard-coded.  Tags are open-vocabulary free text, so they
are compared to the nine contract attributes by embedding similarity, with a
lexical fallback when no encoder is available.  Unseen tags therefore behave
like seen ones.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from starter.clarification import ATTRIBUTE_QUESTIONS, SUPPORTED_ATTRIBUTES

# How far the profile may move the utility, as a fraction.  0.25 keeps the
# factor inside [0.75, 1.25]: enough to reorder near-ties, never enough to
# overturn a decisively better split.  Fixed a priori, not swept on a dev set.
PROFILE_WEIGHT = 0.25

# How strongly an observed "no preference" answer suppresses attributes that
# are semantically adjacent to the one that was refused.
EVIDENCE_DECAY = 0.5

_TOKEN = re.compile(r"[a-z0-9]+")


def _attribute_gloss(attribute: str) -> str:
    """Describe an attribute using text that already exists in the repo.

    Reusing ``ATTRIBUTE_QUESTIONS`` rather than authoring new descriptions
    means no hand-written string can smuggle in knowledge of the observed tag
    vocabulary.  The nine attributes are fixed by the competition contract.
    """
    label = attribute.replace("_", " ")
    return f"{label}: {ATTRIBUTE_QUESTIONS.get(attribute, label)}"


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.lower()))


def _lexical_similarity(left: str, right: str) -> float:
    """Jaccard overlap, used when no encoder is configured."""
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _embed(encoder: Any, texts: Sequence[str]) -> list[Any] | None:
    """Encode texts with the shared runtime encoder, or give up quietly."""
    if encoder is None:
        return None
    try:
        import numpy as np

        vectors = []
        for text in texts:
            if hasattr(encoder, "embed_query"):
                raw = encoder.embed_query(text)
            elif callable(encoder):
                raw = encoder(text)
            else:
                return None
            vector = np.asarray(raw, dtype="float32").reshape(-1)
            norm = float(np.linalg.norm(vector.astype("float64")))
            if not math.isfinite(norm) or norm == 0.0:
                return None
            vectors.append(vector / norm)
        return vectors
    except (ImportError, TypeError, ValueError, RuntimeError, AttributeError):
        return None


def _similarity_matrix(
    tags: Sequence[str],
    attributes: Sequence[str],
    encoder: Any,
) -> dict[str, dict[str, float]] | None:
    glosses = [_attribute_gloss(attribute) for attribute in attributes]
    vectors = _embed(encoder, [*tags, *glosses])
    if vectors is None:
        return {
            tag: {
                attribute: _lexical_similarity(tag, gloss)
                for attribute, gloss in zip(attributes, glosses)
            }
            for tag in tags
        }
    tag_vectors = vectors[: len(tags)]
    gloss_vectors = vectors[len(tags):]
    return {
        tag: {
            attribute: float(tag_vector @ gloss_vector)
            for attribute, gloss_vector in zip(attributes, gloss_vectors)
        }
        for tag, tag_vector in zip(tags, tag_vectors)
    }


def _rescale(raw: Mapping[str, float]) -> dict[str, float]:
    """Min-max across attributes, so the prior is encoder-scale invariant.

    Cosine similarities occupy a narrow, model-dependent band.  Rescaling
    within the profile keeps the prior meaningful if the encoder is swapped,
    and expresses what the profile actually knows: a *relative* ordering of
    attributes, not an absolute confidence.
    """
    if not raw:
        return {}
    low, high = min(raw.values()), max(raw.values())
    if not math.isfinite(low) or not math.isfinite(high) or high - low < 1e-9:
        return {attribute: 0.5 for attribute in raw}
    return {
        attribute: (value - low) / (high - low) for attribute, value in raw.items()
    }


def _clean_tags(profile: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(profile, Mapping):
        return ()
    tags = profile.get("preference_tags")
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, (list, tuple, set, frozenset)):
        return ()
    cleaned: list[str] = []
    for tag in tags:
        text = str(tag).strip().lower()
        # A tag carrying no attribute information is dropped rather than
        # allowed to flatten the prior toward noise.
        if text and text not in cleaned and _TOKEN.search(text):
            cleaned.append(text)
    return tuple(cleaned)


class ProfileAffinity:
    """Per-session, profile-conditioned prior over clarification attributes.

    Built once at ``reset`` because ``preference_tags`` is fixed for a session.
    Degrades to a no-op (all factors 1.0) whenever the profile is absent,
    empty, malformed, or uninformative.
    """

    def __init__(
        self,
        profile: Mapping[str, Any] | None = None,
        *,
        encoder: Any = None,
        attributes: Sequence[str] = SUPPORTED_ATTRIBUTES,
        weight: float = PROFILE_WEIGHT,
    ) -> None:
        self.attributes = tuple(attributes)
        self.weight = float(weight)
        self.tags = _clean_tags(profile)
        self._similarity = (
            _similarity_matrix(self.tags, self.attributes, encoder)
            if self.tags
            else None
        )
        self.affinity = self._build_affinity()
        self._refused: list[str] = []

    def _build_affinity(self) -> dict[str, float]:
        if not self._similarity:
            return {attribute: 0.5 for attribute in self.attributes}
        # Max over tags, not mean: a set of tags is a set of independent
        # emphases, and one strongly matching tag should carry the attribute
        # regardless of how unrelated the others are.
        raw = {
            attribute: max(
                (scores.get(attribute, 0.0) for scores in self._similarity.values()),
                default=0.0,
            )
            for attribute in self.attributes
        }
        return _rescale(raw)

    def observe_no_preference(self, attribute: str) -> None:
        """Record that the shopper declined to answer ``attribute``.

        Direct evidence outranks the profile prior, and the same similarity
        structure generalizes it: declining "material" is weak evidence against
        other fabric-adjacent questions too.
        """
        if attribute in self.attributes and attribute not in self._refused:
            self._refused.append(attribute)

    def _suppression(self, attribute: str) -> float:
        if not self._refused or self._similarity is None:
            return 1.0
        factor = 1.0
        for refused in self._refused:
            if refused == attribute:
                return 0.0
            related = _rescale(
                {
                    other: max(
                        (s.get(other, 0.0) for s in self._similarity.values()),
                        default=0.0,
                    )
                    for other in self.attributes
                }
            )
            closeness = 1.0 - abs(related.get(attribute, 0.5) - related.get(refused, 0.5))
            factor *= 1.0 - EVIDENCE_DECAY * max(0.0, closeness - 0.5) * 2.0
        return max(0.0, factor)

    def factor(self, attribute: str) -> float:
        """Likelihood ratio on the odds that the shopper answers ``attribute``.

        Consumed by ``starter.clarification._answer_probability``.  Returning
        1.0 means the profile says nothing, so the mode prior stands unchanged;
        0.0 is the refusal veto set by ``observe_no_preference``.
        """
        if not self.tags:
            return 1.0
        score = self.affinity.get(attribute, 0.5)
        prior = 1.0 + self.weight * (2.0 * score - 1.0)
        return prior * self._suppression(attribute)

    def as_dict(self) -> dict[str, float]:
        return {attribute: self.factor(attribute) for attribute in self.attributes}


__all__ = ["ProfileAffinity", "PROFILE_WEIGHT", "EVIDENCE_DECAY"]
