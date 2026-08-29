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

# Semantic lookup uses the generated per-attribute matrices. Brand deliberately
# remains exact-only; size and price are runtime structured fields, not
# dictionary attributes.
SEMANTIC_ATTRIBUTES = (
    "category",
    "color",
    "material",
    "style",
    "feature",
    "use_case",
)

NORMALIZATION_VERSION = "nfkc-casefold-apostrophe-removal-v2"
DEFAULT_MIN_SIMILARITY = 0.80


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
    lookup uses the optional per-attribute matrices under
    ``attribute_embeddings``; the older combined matrix remains readable for
    compatibility.
    """

    def __init__(
        self,
        values: Iterable[CanonicalValue],
        normalized_index: Mapping[str, Mapping[str, Iterable[str]]],
        embedding_rows: Iterable[Mapping[str, Any]] = (),
        embeddings: Any = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
        embedding_normalization: str | None = None,
        attribute_embeddings: Mapping[
            str, tuple[Iterable[Mapping[str, Any]], Any]
        ] | None = None,
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
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._embedding_normalization = embedding_normalization
        self._attribute_embeddings = {
            str(attribute): (
                tuple(dict(row) for row in rows),
                matrix,
            )
            for attribute, (rows, matrix) in (attribute_embeddings or {}).items()
        }
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

        embedding_model = None
        embedding_dimension = None
        embedding_normalization = None
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            if not isinstance(manifest, Mapping):
                raise ValueError("dictionary manifest must contain an object")
            raw_model = manifest.get("embedding_model", manifest.get("model"))
            raw_dimension = manifest.get(
                "embedding_dimension", manifest.get("dimension")
            )
            raw_normalization = manifest.get("normalization")
            if isinstance(raw_model, str) and raw_model.strip():
                embedding_model = raw_model.strip()
            if isinstance(raw_dimension, int) and not isinstance(raw_dimension, bool):
                embedding_dimension = raw_dimension
            if isinstance(raw_normalization, str) and raw_normalization.strip():
                embedding_normalization = raw_normalization.strip()

        embeddings_path = root / "attribute_embeddings.npy"
        metadata_path = root / "embedding_metadata.json"
        legacy_rows: list[Mapping[str, Any]] = []
        if metadata_path.exists() and embeddings_path.exists():
            with metadata_path.open(encoding="utf-8") as handle:
                legacy_rows = _metadata_rows(json.load(handle))

        embeddings = None
        if embeddings_path.exists():
            try:
                import numpy as np
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "semantic lookup requires NumPy; install requirements-embeddings.txt"
                ) from exc
            embeddings = np.load(embeddings_path, allow_pickle=False)

        attribute_embeddings: dict[
            str, tuple[tuple[Mapping[str, Any], ...], Any]
        ] = {}
        attribute_root = root / "attribute_embeddings"
        attribute_metadata_path = attribute_root / "metadata.json"
        if attribute_metadata_path.exists():
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError(
                    "semantic lookup requires NumPy; install requirements-embeddings.txt"
                ) from exc
            with attribute_metadata_path.open(encoding="utf-8") as handle:
                attribute_metadata = json.load(handle)
            if not isinstance(attribute_metadata, Mapping):
                raise ValueError("attribute embedding metadata must contain an object")
            raw_model = attribute_metadata.get(
                "model", attribute_metadata.get("model_path")
            )
            raw_dimension = attribute_metadata.get("dimension")
            raw_normalization = attribute_metadata.get("normalization")
            if isinstance(raw_model, str) and raw_model.strip():
                embedding_model = raw_model.strip()
            if isinstance(raw_dimension, int) and not isinstance(raw_dimension, bool):
                embedding_dimension = raw_dimension
            if isinstance(raw_normalization, str) and raw_normalization.strip():
                embedding_normalization = raw_normalization.strip()

            attributes = attribute_metadata.get("attributes")
            if not isinstance(attributes, Mapping):
                raise ValueError("attribute embedding metadata is missing attributes")
            for attribute in SEMANTIC_ATTRIBUTES:
                spec = attributes.get(attribute)
                if spec is None:
                    continue
                if not isinstance(spec, Mapping):
                    raise ValueError(
                        f"attribute embedding metadata for {attribute} must be an object"
                    )
                attribute_rows = spec.get("rows", [])
                if not isinstance(attribute_rows, list):
                    raise ValueError(
                        f"attribute embedding metadata rows are invalid for {attribute}"
                    )
                embedding_file = spec.get(
                    "embedding_file", f"{attribute}_embeddings.npy"
                )
                if not isinstance(embedding_file, str) or not embedding_file:
                    raise ValueError(
                        f"attribute embedding file is invalid for {attribute}"
                    )
                matrix_path = attribute_root / embedding_file
                if not matrix_path.is_file():
                    raise OSError(f"missing attribute embedding matrix: {matrix_path}")
                matrix = np.load(matrix_path, allow_pickle=False)
                attribute_embeddings[attribute] = (tuple(attribute_rows), matrix)

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
        return cls(
            values,
            normalized_lookup,
            legacy_rows,
            embeddings,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            embedding_normalization=embedding_normalization,
            attribute_embeddings=attribute_embeddings,
        )

    @property
    def values(self) -> tuple[CanonicalValue, ...]:
        return tuple(self._values[canonical] for canonical in sorted(self._values))

    @property
    def rows_by_attribute(self) -> Mapping[str, tuple[int, ...]]:
        return dict(self._rows_by_attribute)

    @property
    def embedding_model(self) -> str | None:
        return self._embedding_model

    @property
    def embedding_dimension(self) -> int | None:
        return self._embedding_dimension

    @property
    def embedding_normalization(self) -> str | None:
        return self._embedding_normalization

    @property
    def has_semantic_embeddings(self) -> bool:
        if self._attribute_embeddings:
            return any(
                bool(rows) and getattr(matrix, "ndim", None) == 2
                for rows, matrix in self._attribute_embeddings.values()
            )
        return self._embeddings is not None and bool(self._embedding_rows)

    @property
    def semantic_available(self) -> bool:
        encoder = getattr(self, "_query_encoder", None)
        return self.has_semantic_embeddings and (
            callable(encoder) or callable(getattr(encoder, "embed_query", None))
        )

    @property
    def phrase_index(self) -> Mapping[str, tuple[tuple[tuple[str, ...], str], ...]]:
        """Return normalized dictionary phrases grouped by their first token."""

        return dict(self._phrase_index)

    def get(self, value_id: str) -> CanonicalValue | None:
        return self._values.get(value_id)

    def get_candidates(self, raw_text: str) -> tuple[CanonicalValue, ...]:
        """Return canonical values for a normalized surface, preserving ambiguity.

        Results follow the fixed attribute order and are deterministic.  This
        exposes the metadata needed by conservative runtime disambiguation
        without changing the public constraint shape.
        """

        normalized = normalize_text(raw_text)
        if not normalized:
            return ()
        candidates: list[CanonicalValue] = []
        for attribute in ATTRIBUTE_FIELDS:
            value_ids = self._normalized_index.get(attribute, {}).get(normalized, ())
            for value_id in sorted(value_ids):
                value = self._values.get(value_id)
                if value is not None:
                    candidates.append(value)
        return tuple(candidates)

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
        """Find confident semantic matches in one or all attribute matrices."""

        if allowed_attribute is not None and allowed_attribute not in ATTRIBUTE_FIELDS:
            raise ValueError(f"unknown canonical attribute: {allowed_attribute}")
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if not self.has_semantic_embeddings:
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

        query = self._prepare_query(self._encode_query(raw_text), np)
        matches: list[LookupMatch] = []
        for attribute, rows, matrix in self._embedding_views(allowed_attribute):
            matches.extend(
                self._score_embedding_view(
                    query,
                    raw_text,
                    attribute,
                    rows,
                    matrix,
                    np,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    min_margin=min_margin,
                )
            )
        attribute_order = {
            attribute: index for index, attribute in enumerate(SEMANTIC_ATTRIBUTES)
        }
        matches.sort(
            key=lambda item: (
                -item.similarity,
                attribute_order.get(item.attribute, len(attribute_order)),
                item.canonical_id,
            )
        )
        return tuple(matches[:top_k])

    def semantic_match_ngrams(
        self,
        raw_text: str,
        *,
        stopwords: Iterable[str] = (),
        max_ngram: int = 3,
        top_k_per_attribute: int = 1,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> tuple[LookupMatch, ...]:
        """Search stopword-filtered 1/2/3-grams across every semantic view.

        Each phrase is encoded once and scored independently against the
        category, color, material, style, feature, and use-case matrices.
        Brand intentionally has no semantic view and remains exact-only.
        """

        if max_ngram < 1 or max_ngram > 3:
            raise ValueError("max_ngram must be between one and three")
        if top_k_per_attribute < 1:
            raise ValueError("top_k_per_attribute must be at least one")
        if not self.has_semantic_embeddings:
            return ()

        stopword_surfaces = {
            normalize_text(word) for word in stopwords if normalize_text(str(word))
        }
        tokens = [
            token
            for token in normalize_text(raw_text).split()
            if token not in stopword_surfaces
        ]
        phrases: list[str] = []
        seen_phrases: set[str] = set()
        for width in range(1, max_ngram + 1):
            for start in range(0, len(tokens) - width + 1):
                phrase = " ".join(tokens[start : start + width])
                if phrase and phrase not in seen_phrases:
                    seen_phrases.add(phrase)
                    phrases.append(phrase)

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "semantic lookup requires NumPy; install requirements-embeddings.txt"
            ) from exc

        matches: list[LookupMatch] = []
        for phrase in phrases:
            query = self._prepare_query(self._encode_query(phrase), np)
            for attribute, rows, matrix in self._embedding_views(None):
                matches.extend(
                    self._score_embedding_view(
                        query,
                        phrase,
                        attribute,
                        rows,
                        matrix,
                        np,
                        top_k=top_k_per_attribute,
                        min_similarity=min_similarity,
                        min_margin=0.0,
                    )
                )

        best_by_id: dict[str, LookupMatch] = {}
        for match in matches:
            previous = best_by_id.get(match.canonical_id)
            if previous is None or match.similarity > previous.similarity:
                best_by_id[match.canonical_id] = match
        attribute_order = {
            attribute: index for index, attribute in enumerate(SEMANTIC_ATTRIBUTES)
        }
        return tuple(
            sorted(
                best_by_id.values(),
                key=lambda item: (
                    -item.similarity,
                    attribute_order.get(item.attribute, len(attribute_order)),
                    item.canonical_id,
                ),
            )
        )

    def _embedding_views(
        self,
        allowed_attribute: str | None,
    ) -> tuple[tuple[str, tuple[Mapping[str, Any], ...], Any], ...]:
        attributes = (
            (allowed_attribute,)
            if allowed_attribute is not None
            else SEMANTIC_ATTRIBUTES
        )
        if self._attribute_embeddings:
            return tuple(
                (attribute, *self._attribute_embeddings[attribute])
                for attribute in attributes
                if attribute in self._attribute_embeddings
            )
        if self._embeddings is None or not self._embedding_rows:
            return ()
        views: list[tuple[str, tuple[Mapping[str, Any], ...], Any]] = []
        for attribute in attributes:
            positions = tuple(
                index
                for index, row in enumerate(self._embedding_rows)
                if str(row["attribute"]) == attribute
            )
            if positions:
                views.append(
                    (
                        attribute,
                        tuple(self._embedding_rows[index] for index in positions),
                        self._embeddings[list(positions)],
                    )
                )
        return tuple(views)

    def _prepare_query(self, value: Any, np: Any) -> Any:
        query = np.asarray(value, dtype=np.float32)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1 or (
            self._embedding_dimension is not None
            and query.size != self._embedding_dimension
        ):
            raise ValueError(
                "semantic query embedding dimension does not match the attribute "
                f"artifact: {tuple(query.shape)} != ({self._embedding_dimension},)"
            )
        if not np.isfinite(query).all():
            raise ValueError("semantic query embedding contains non-finite values")
        norm = float(np.linalg.norm(query.astype(np.float64)))
        if not math.isfinite(norm) or norm == 0.0:
            return None
        return query / norm

    def _score_embedding_view(
        self,
        query: Any,
        raw_text: str,
        attribute: str,
        rows: tuple[Mapping[str, Any], ...],
        matrix: Any,
        np: Any,
        *,
        top_k: int,
        min_similarity: float,
        min_margin: float,
    ) -> tuple[LookupMatch, ...]:
        if query is None or not rows:
            return ()
        scores = np.asarray(matrix @ query).reshape(-1)
        order = np.argsort(-scores, kind="stable")[:top_k]
        if len(order) == 0:
            return ()
        best_score = float(scores[order[0]])
        if best_score < min_similarity:
            return ()
        if len(order) > 1 and best_score - float(scores[order[1]]) < min_margin:
            return ()
        matches: list[LookupMatch] = []
        for position in order:
            similarity = float(scores[int(position)])
            if similarity < min_similarity:
                continue
            value = self._values[str(rows[int(position)]["canonical_id"])]
            matches.append(
                LookupMatch(
                    canonical_id=value.canonical_id,
                    attribute=attribute,
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
        if callable(encoder):
            return encoder(raw_text)
        embed_query = getattr(encoder, "embed_query", None)
        if callable(embed_query):
            return embed_query(raw_text)
        embed_documents = getattr(encoder, "embed_documents", None)
        if callable(embed_documents):
            return embed_documents([raw_text])
        raise RuntimeError("semantic query encoder does not expose an embedding method")

    def set_query_encoder(self, encoder: Any) -> None:
        """Attach ``text -> vector`` encoding for runtime semantic queries."""

        if not callable(encoder) and not callable(getattr(encoder, "embed_query", None)):
            raise TypeError("encoder must be callable or expose embed_query()")
        actual_dimension = getattr(encoder, "embedding_dimension", None)
        if (
            actual_dimension is not None
            and self._embedding_dimension is not None
            and int(actual_dimension) != self._embedding_dimension
        ):
            raise ValueError(
                "semantic query encoder dimension does not match the attribute "
                f"artifact: {actual_dimension} != {self._embedding_dimension}"
            )
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

        if self._attribute_embeddings:
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError(
                    "semantic lookup requires NumPy; install requirements-embeddings.txt"
                ) from exc
            for attribute, (rows, matrix) in self._attribute_embeddings.items():
                if attribute not in SEMANTIC_ATTRIBUTES:
                    raise ValueError(
                        f"attribute embeddings are not allowed for {attribute}"
                    )
                if getattr(matrix, "ndim", None) != 2:
                    raise ValueError(
                        f"{attribute} embeddings must be a two-dimensional matrix"
                    )
                if int(matrix.shape[0]) != len(rows):
                    raise ValueError(
                        f"{attribute} embedding row count does not match metadata"
                    )
                if (
                    self._embedding_dimension is not None
                    and int(matrix.shape[1]) != self._embedding_dimension
                ):
                    raise ValueError(
                        f"{attribute} embedding dimension does not match metadata"
                    )
                if not bool(np.isfinite(matrix).all()):
                    raise ValueError(
                        f"{attribute} embeddings contain non-finite values"
                    )
                for row in rows:
                    value_id = str(row.get("canonical_id", ""))
                    value = self._values.get(value_id)
                    if value is None:
                        raise ValueError(
                            f"{attribute} embeddings reference unknown canonical_id: "
                            f"{value_id}"
                        )
                    if str(row.get("attribute", "")) != attribute:
                        raise ValueError(
                            f"{attribute} embedding metadata has the wrong attribute"
                        )
                    if str(row.get("value", "")) != value.value:
                        raise ValueError(
                            f"{attribute} embedding metadata disagrees with registry: "
                            f"{value_id}"
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
