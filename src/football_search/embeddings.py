"""Embedding models behind one small interface, so indexing and search always agree."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from football_search.config import ModelSpec


class Embedder(Protocol):
    """Anything that turns passages and queries into unit-length vectors."""

    spec: ModelSpec

    @property
    def dim(self) -> int: ...

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """A sentence-transformers model pinned to a revision, with its prefix conventions.

    Runs on CPU so local builds give the same vectors as the CPU-only Space.
    Vectors are normalized, so cosine similarity is just a dot product.
    """

    def __init__(self, spec: ModelSpec, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.model = SentenceTransformer(spec.name, revision=spec.revision, device=device)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        texts = [self.spec.passage_prefix + t for t in texts]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.spec.query_prefix + text], normalize_embeddings=True)[0]


class Reranker(Protocol):
    """Scores (query, passage) pairs; higher means more relevant."""

    spec: ModelSpec

    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    """A cross-encoder pinned to a revision. It reads query and passage together, so it
    catches details (scored vs missed, header vs volley) that embeddings blur."""

    def __init__(self, spec: ModelSpec, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.spec = spec
        self.model = CrossEncoder(spec.name, revision=spec.revision, device=device)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.array([])
        return np.asarray(self.model.predict([(query, p) for p in passages]))
