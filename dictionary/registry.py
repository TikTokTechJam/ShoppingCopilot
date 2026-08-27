from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ATTRIBUTE_FIELDS = (
    "category",
    "brand",
    "color",
    "material",
    "style",
    "feature",
    "use_case",
)

# Semantic lookup is optional and deliberately excludes category, which is a
# Tier-1 exact/structured field. Size and price are runtime structured fields,
# not dictionary attributes.
SEMANTIC_ATTRIBUTES = (
    "brand",
    "color",
    "material",
    "style",
    "feature",
    "use_case",
)

NORMALIZATION_VERSION = "nfkc-casefold-apostrophe-removal-v2"
DEFAULT_MIN_SIMILARITY = 0.70


def normalize_text(value: str) -> str:
    """Return a conservative surface normalization for exact lookup.

    This intentionally handles only lexical equivalence. It does not map
    semantic aliases such as ``dark blue`` to ``navy`` or ``trainers`` to
    ``sneakers``.
    """

    normalized = unicodedata.normalize("NFKC", value).casefold()
    output: list[str] = []
    pending_space = False
    apostrophes = {"'", "’", "ʼ", "＇"}
    for index, character in enumerate(normalized):
        if character.isalnum():
            if pending_space and output:
                output.append(" ")
            output.append(character)
            pending_space = False
        elif (
            character in apostrophes
            and index > 0
            and index + 1 < len(normalized)
            and normalized[index - 1].isalnum()
            and normalized[index + 1].isalnum()
        ):
            # Apostrophes inside words are lexical decoration, not a word
            # boundary: Levi's -> levis and O'Neill -> oneill.
            pending_space = False
        elif character.isspace() or character in {"_", "-"}:
            pending_space = bool(output)
        elif unicodedata.category(character).startswith(("P", "S")):
            pending_space = bool(output)
        else:
            pending_space = bool(output)
    return "".join(output).strip()


def canonical_id(attribute: str, value: str) -> str:
    """Build the stable cross-index key used by all lookup paths."""

    if attribute not in ATTRIBUTE_FIELDS:
        raise ValueError(f"unknown canonical attribute: {attribute}")
    if not isinstance(value, str) or not value:
        raise ValueError("canonical values must be non-empty strings")
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError("canonical values must contain searchable text")
    return f"{attribute}:{normalized.replace(' ', '_')}"


@dataclass(frozen=True)
class CanonicalValue:
    canonical_id: str
    attribute: str
    value: str
    normalized: str
    count: int


@dataclass(frozen=True)
class LookupMatch:
    canonical_id: str
    attribute: str
    value: str
    raw_text: str
    normalized_text: str
    match_method: str
    similarity: float


def _as_records(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("canonical_values.json must contain an object or array")
    values = payload.get("values")
    if isinstance(values, Mapping):
        return [
            {"canonical_id": str(value_id), **dict(record)}
            for value_id, record in values.items()
        ]
    if isinstance(values, list):
        return values
    raise ValueError("canonical_values.json must contain a values object or array")


def _metadata_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError("embedding_metadata.json must be an array or an object with rows")


class AttributeDictionary:
    """In-memory registry with exact and optional semantic lookup.

    The JSON registry and normalized lookup are usable without NumPy. Semantic
    lookup is enabled only when the optional ``attribute_embeddings.npy`` and
    its metadata are present.
    """

    def __init__(
        self,
        values: Iterable[CanonicalValue],
        normalized_index: Mapping[str, Mapping[str, Iterable[str]]],
        embedding_rows: Iterable[Mapping[str, Any]] = (),
        embeddings: Any = None,
    ) -> None:
        self._values = {value.canonical_id: value for value in values}
        phrase_index: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        for value in self._values.values():
            parts = tuple(value.normalized.split())
            if parts:
                phrase_index.setdefault(parts[0], []).append((parts, value.canonical_id))
        self._phrase_index = {
            first: tuple(sorted(entries, key=lambda item: (-len(item[0]), item[0], item[1])))
            for first, entries in phrase_index.items()
        }
        self._normalized_index = {
            attribute: {
                normalized: tuple(canonical_ids)
                for normalized, canonical_ids in surfaces.items()
            }
            for attribute, surfaces in normalized_index.items()
        }
        self._embedding_rows = tuple(dict(row) for row in embedding_rows)
        self._embeddings = embeddings
        self._rows_by_attribute: dict[str, tuple[int, ...]] = {}
        for row in self._embedding_rows:
            attribute = str(row["attribute"])
            row_number = int(row["row"])
            self._rows_by_attribute.setdefault(attribute, ())
            self._rows_by_attribute[attribute] += (row_number,)
        self._validate()

    @classmethod
    def load(cls, directory: str | Path) -> "AttributeDictionary":
        root = Path(directory)
        with (root / "canonical_values.json").open(encoding="utf-8") as handle:
            registry = json.load(handle)
        with (root / "normalized_lookup.json").open(encoding="utf-8") as handle:
            normalized_lookup = json.load(handle)
        if isinstance(normalized_lookup, Mapping) and isinstance(
            normalized_lookup.get("attributes"), Mapping
        ):
            normalized_lookup = normalized_lookup["attributes"]

        metadata_path = root / "embedding_metadata.json"
        rows: list[Mapping[str, Any]] = []
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as handle:
                rows = _metadata_rows(json.load(handle))

        embeddings = None
        embeddings_path = root / "attribute_embeddings.npy"
        if embeddings_path.exists():
            try:
                import numpy as np
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "semantic lookup requires NumPy; install requirements-embeddings.txt"
                ) from exc
            embeddings = np.load(embeddings_path, allow_pickle=False)

        values = []
        for record in _as_records(registry):
            values.append(
                CanonicalValue(
                    canonical_id=str(record["canonical_id"]),
                    attribute=str(record["attribute"]),
                    value=str(record["value"]),
                    normalized=str(record["normalized"]),
                    count=int(record["count"]),
                )
            )
        return cls(values, normalized_lookup, rows, embeddings)

    @property
    def values(self) -> tuple[CanonicalValue, ...]:
        return tuple(self._values[canonical] for canonical in sorted(self._values))

    @property
    def rows_by_attribute(self) -> Mapping[str, tuple[int, ...]]:
        return dict(self._rows_by_attribute)

    @property
    def phrase_index(self) -> Mapping[str, tuple[tuple[tuple[str, ...], str], ...]]:
        """Return normalized dictionary phrases grouped by their first token."""

        return dict(self._phrase_index)

    def get(self, value_id: str) -> CanonicalValue | None:
        return self._values.get(value_id)

    def exact_match(
        self,
        raw_text: str,
        allowed_attribute: str | None = None,
    ) -> tuple[LookupMatch, ...]:
        """Return all exact normalized matches, preserving ambiguity."""

        if allowed_attribute is not None and allowed_attribute not in ATTRIBUTE_FIELDS:
            raise ValueError(f"unknown canonical attribute: {allowed_attribute}")
        normalized = normalize_text(raw_text)
        if not normalized:
            return ()
        attributes = (
            (allowed_attribute,)
            if allowed_attribute is not None
            else ATTRIBUTE_FIELDS
        )
        matches: list[LookupMatch] = []
        for attribute in attributes:
            for value_id in self._normalized_index.get(attribute, {}).get(normalized, ()):
                value = self._values[value_id]
                matches.append(
                    LookupMatch(
                        canonical_id=value.canonical_id,
                        attribute=value.attribute,
                        value=value.value,
                        raw_text=raw_text,
                        normalized_text=normalized,
                        match_method="exact",
                        similarity=1.0,
                    )
                )
        return tuple(matches)

    def semantic_match(
        self,
        raw_text: str,
        allowed_attribute: str | None = None,
        *,
        top_k: int = 1,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        min_margin: float = 0.0,
    ) -> tuple[LookupMatch, ...]:
        """Find confident semantic matches using the shared matrix.

        The matrix is normalized at build time, so the search is an exact
        in-memory cosine/inner-product search. A weak best match, or an
        insufficient best-vs-second margin, yields no result.
        """

        if allowed_attribute is not None and allowed_attribute not in ATTRIBUTE_FIELDS:
            raise ValueError(f"unknown canonical attribute: {allowed_attribute}")
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if self._embeddings is None or not self._embedding_rows:
            return ()
        if not isinstance(min_similarity, (int, float)) or not math.isfinite(min_similarity):
            raise ValueError("min_similarity must be finite")
        if not isinstance(min_margin, (int, float)) or not math.isfinite(min_margin):
            raise ValueError("min_margin must be finite")

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "semantic lookup requires NumPy; install requirements-embeddings.txt"
            ) from exc

        query = np.asarray(self._encode_query(raw_text), dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return ()
        query = query / norm

        candidate_rows = [
            int(row["row"])
            for row in self._embedding_rows
            if allowed_attribute is None
            or str(row["attribute"]) == allowed_attribute
        ]
        if not candidate_rows:
            return ()
        scores = np.asarray(self._embeddings[candidate_rows] @ query).reshape(-1)
        order = np.argsort(-scores, kind="stable")[:top_k]
        best_score = float(scores[order[0]])
        if best_score < min_similarity:
            return ()
        if len(order) > 1 and best_score - float(scores[order[1]]) < min_margin:
            return ()

        matches: list[LookupMatch] = []
        for position in order:
            row = self._embedding_rows[candidate_rows[int(position)]]
            value = self._values[str(row["canonical_id"])]
            similarity = float(scores[int(position)])
            if similarity < min_similarity:
                continue
            matches.append(
                LookupMatch(
                    canonical_id=value.canonical_id,
                    attribute=value.attribute,
                    value=value.value,
                    raw_text=raw_text,
                    normalized_text=normalize_text(raw_text),
                    match_method="semantic",
                    similarity=similarity,
                )
            )
        return tuple(matches)

    def resolve(
        self,
        raw_text: str,
        allowed_attribute: str | None = None,
        *,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        min_margin: float = 0.0,
    ) -> LookupMatch | None:
        """Resolve exact text first, then use semantic fallback if safe."""

        exact = self.exact_match(raw_text, allowed_attribute)
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            return None
        semantic = self.semantic_match(
            raw_text,
            allowed_attribute,
            top_k=2,
            min_similarity=min_similarity,
            min_margin=min_margin,
        )
        return semantic[0] if semantic else None

    def _encode_query(self, raw_text: str) -> Any:
        """Encode a query with the model-independent stored artifact contract.

        A precomputed matrix cannot encode a new phrase by itself. The build
        manifest therefore records the model name, and callers may attach a
        compatible encoder through ``set_query_encoder`` before semantic use.
        """

        encoder = getattr(self, "_query_encoder", None)
        if encoder is None:
            raise RuntimeError(
                "semantic lookup needs a query encoder compatible with the stored "
                "embedding model; call set_query_encoder()"
            )
        return encoder(raw_text)

    def set_query_encoder(self, encoder: Any) -> None:
        """Attach ``text -> vector`` encoding for runtime semantic queries."""

        if not callable(encoder):
            raise TypeError("encoder must be callable")
        self._query_encoder = encoder

    def _validate(self) -> None:
        seen_surfaces: set[tuple[str, str]] = set()
        for value_id, value in self._values.items():
            if value_id != value.canonical_id:
                raise ValueError(f"registry key mismatch for {value_id}")
            if value.attribute not in ATTRIBUTE_FIELDS:
                raise ValueError(f"unknown attribute in registry: {value.attribute}")
            if canonical_id(value.attribute, value.value) != value.canonical_id:
                raise ValueError(f"invalid canonical_id: {value.canonical_id}")
            if normalize_text(value.value) != value.normalized:
                raise ValueError(f"normalized value disagrees with registry: {value_id}")
            if not value.normalized or value.count < 1:
                raise ValueError(f"invalid registry value: {value.canonical_id}")
            surface_key = (value.attribute, value.normalized)
            if surface_key in seen_surfaces:
                raise ValueError(
                    "duplicate attribute/normalized surface: "
                    f"{value.attribute}:{value.normalized}"
                )
            seen_surfaces.add(surface_key)

        for attribute, surfaces in self._normalized_index.items():
            if attribute not in ATTRIBUTE_FIELDS:
                raise ValueError(f"unknown attribute in normalized lookup: {attribute}")
            for normalized, value_ids in surfaces.items():
                if not normalized or normalize_text(normalized) != normalized:
                    raise ValueError("normalized lookup contains an invalid surface")
                if not value_ids:
                    raise ValueError("normalized lookup contains an empty ID list")
                for value_id in value_ids:
                    value = self._values.get(value_id)
                    if value is None:
                        raise ValueError(
                            f"normalized lookup references unknown canonical_id: {value_id}"
                        )
                    if value.attribute != attribute or value.normalized != normalized:
                        raise ValueError(
                            f"normalized lookup disagrees with registry: {value_id}"
                        )

        if self._embeddings is None:
            if self._embedding_rows:
                raise ValueError("embedding metadata exists without an embedding matrix")
            return
        if getattr(self._embeddings, "ndim", None) != 2:
            raise ValueError("attribute embeddings must be a two-dimensional matrix")
        if int(self._embeddings.shape[0]) != len(self._embedding_rows):
            raise ValueError("embedding row count does not match embedding metadata")
        for expected_row, row in enumerate(self._embedding_rows):
            if int(row["row"]) != expected_row:
                raise ValueError("embedding metadata rows must be contiguous and ordered")
            value_id = str(row["canonical_id"])
            value = self._values.get(value_id)
            if value is None:
                raise ValueError(f"embedding references unknown canonical_id: {value_id}")
            if str(row["attribute"]) != value.attribute or str(row["value"]) != value.value:
                raise ValueError(f"embedding metadata disagrees with registry: {value_id}")
