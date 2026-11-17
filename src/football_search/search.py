"""Search: embed a query, find the nearest shots, and optionally rerank them.

Shared by the app and the evaluation, so both measure the same thing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from football_search.config import Config, ModelSpec
from football_search.embeddings import (
    CrossEncoderReranker,
    Embedder,
    Reranker,
    SentenceTransformerEmbedder,
)
from football_search.index import MANIFEST, open_client

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Filters:
    """Structured filters that narrow the semantic search (hybrid search)."""

    tournaments: tuple[str, ...] = ()
    team: str = ""
    goals_only: bool = False
    min_xg: float = 0.0


@dataclass
class Result:
    id: str
    score: float  # cosine similarity between query and shot, 1 - cosine distance
    document: str  # the text that was embedded
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None


def build_where(filters: Filters | None) -> dict[str, Any] | None:
    """Translate filters into a Chroma `where` clause (None means no filtering)."""
    if filters is None:
        return None
    conditions: list[dict[str, Any]] = []
    if len(filters.tournaments) == 1:
        conditions.append({"tournament": filters.tournaments[0]})
    elif filters.tournaments:
        conditions.append({"tournament": {"$in": list(filters.tournaments)}})
    if filters.team:
        conditions.append({"team": filters.team})
    if filters.goals_only:
        conditions.append({"is_goal": True})
    if filters.min_xg > 0:
        conditions.append({"xg": {"$gte": float(filters.min_xg)}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def load_manifest(chroma_dir: Path) -> dict[str, Any]:
    return json.loads((chroma_dir / MANIFEST).read_text())


class Searcher:
    """Holds the models and the collection, loaded once, and answers queries."""

    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str,
        embedder: Embedder,
        reranker: Reranker | None = None,
        fetch_k: int = 50,
    ):
        self.embedder = embedder
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.collection = open_client(chroma_dir).get_collection(
            collection_name, embedding_function=None
        )
        # Repeated queries (e.g. the example buttons) skip the embedding model entirely.
        self._embed = lru_cache(maxsize=1024)(self._embed_uncached)

    @classmethod
    def from_manifest(cls, cfg: Config) -> Searcher:
        """Open the deployed index with the exact models recorded in its manifest."""
        manifest = load_manifest(cfg.chroma_dir)
        built_with = manifest["versions"]["chromadb"]
        if built_with != version("chromadb"):
            log.warning("Index built with chromadb %s, running %s", built_with, version("chromadb"))
        emb = manifest["embedding"]
        spec = ModelSpec(emb["model"], emb["revision"], emb["query_prefix"], emb["passage_prefix"])
        rerank = manifest.get("rerank")
        reranker = (
            CrossEncoderReranker(ModelSpec(rerank["model"], rerank["revision"])) if rerank else None
        )
        return cls(
            cfg.chroma_dir,
            manifest["collection"],
            SentenceTransformerEmbedder(spec),
            reranker,
            rerank["fetch_k"] if rerank else cfg.fetch_k,
        )

    def _embed_uncached(self, query: str) -> tuple[float, ...]:
        return tuple(np.asarray(self.embedder.embed_query(query), dtype=float).tolist())

    def retrieve(self, query: str, n: int, filters: Filters | None = None) -> list[Result]:
        """Nearest neighbours by cosine similarity, without reranking."""
        res = self.collection.query(
            query_embeddings=[list(self._embed(query))],
            n_results=n,
            where=build_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Result(id=i, score=1.0 - d, document=doc, metadata=dict(meta or {}))
            for i, d, doc, meta in zip(
                res["ids"][0],
                res["distances"][0],
                res["documents"][0],
                res["metadatas"][0],
                strict=True,
            )
        ]

    def rerank(self, query: str, results: list[Result], reranker: Reranker) -> list[Result]:
        """Reorder results by cross-encoder score (highest first)."""
        scores = reranker.score(query, [r.document for r in results])
        for r, s in zip(results, scores, strict=True):
            r.rerank_score = float(s)
        return sorted(results, key=lambda r: r.rerank_score, reverse=True)

    def search(self, query: str, top_k: int = 10, filters: Filters | None = None) -> list[Result]:
        """Return up to top_k shots, best first."""
        query = query.strip()
        if not query:
            return []
        start = time.perf_counter()
        if self.reranker:
            candidates = self.retrieve(query, max(self.fetch_k, top_k), filters)
            results = self.rerank(query, candidates, self.reranker)[:top_k]
        else:
            results = self.retrieve(query, top_k, filters)
        log.info(
            "query=%r top_k=%d filters=%s results=%d latency_ms=%.1f",
            query[:100],
            top_k,
            build_where(filters),
            len(results),
            (time.perf_counter() - start) * 1000,
        )
        return results

    def distinct(self, key: str) -> list[str]:
        """Sorted distinct values of a metadata field (for filter dropdowns)."""
        metas = self.collection.get(include=["metadatas"])["metadatas"] or []
        return sorted({str(m[key]) for m in metas if key in m})
