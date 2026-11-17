"""Index (the Load step): embed the staged shots and store them in a persistent Chroma DB.

The collection is rebuilt from scratch on every run, and each shot keeps its
StatsBomb event UUID as its ID, so rebuilds are deterministic. A manifest.json
next to the index records exactly what went into it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
from chromadb.config import Settings

from football_search.config import Config, ModelSpec, configure_logging, load_config
from football_search.embeddings import Embedder, SentenceTransformerEmbedder

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
# Which text gets embedded (and reranked). The full fluent text is always stored in
# the "description" metadata field, so the app shows the same readable description
# whatever template is used for search.
TEXT_COLUMNS = {"fluent": "text_fluent", "moment": "text_moment", "terse": "text_terse"}
DISPLAY_COLUMN = "text_fluent"

# Fields stored next to each vector, for filters and for the result cards.
METADATA_FIELDS = [
    "competition", "season", "stage", "match_id", "match_date", "home", "away",
    "team", "opponent", "player", "position", "minute", "period", "game_state",
    "outcome", "is_goal", "is_shootout", "shot_type", "body_part", "technique",
    "play_pattern", "xg", "xg_band", "x", "y", "distance", "zone", "end_x", "end_y",
    "end_z", "placement", "assist_type", "key_passer", "goal_context", "rebound",
    "first_time", "one_on_one", "open_goal", "under_pressure", "team_score", "opp_score",
]  # fmt: skip


def open_client(path: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def sanitize_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Make a row Chroma-safe: only str/int/float/bool values, no None or NaN."""
    clean: dict[str, Any] = {}
    for key in METADATA_FIELDS:
        value = row.get(key)
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if not isinstance(value, str | int | float | bool):
            value = str(value)
        clean[key] = value
    clean["tournament"] = f"{row['competition']} {row['season']}"
    clean["description"] = row[DISPLAY_COLUMN]
    return clean


def content_hash(shots: pd.DataFrame, text_column: str) -> str:
    """Hash of the IDs and texts that were indexed, independent of Parquet file bytes."""
    digest = hashlib.sha256()
    for shot_id, text in sorted(zip(shots["id"], shots[text_column], strict=True)):
        digest.update(f"{shot_id}\t{text}\n".encode())
    return digest.hexdigest()


def build_index(
    shots: pd.DataFrame,
    embedder: Embedder,
    chroma_dir: Path,
    collection_name: str,
    template: str = "fluent",
    batch_size: int = 256,
) -> chromadb.Collection:
    """Embed every shot and (re)create the collection with cosine distance."""
    text_column = TEXT_COLUMNS[template]
    client = open_client(chroma_dir)
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    collection = client.create_collection(
        collection_name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )

    records = shots.to_dict(orient="records")
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [r[text_column] for r in batch]
        collection.add(
            ids=[r["id"] for r in batch],
            embeddings=embedder.embed_passages(texts, batch_size=64).tolist(),
            documents=texts,
            metadatas=[sanitize_metadata(r) for r in batch],
        )
        log.info("Indexed %d/%d shots", min(start + batch_size, len(records)), len(records))
    return collection


def build_manifest(
    cfg: Config,
    shots: pd.DataFrame,
    embedder: Embedder,
    template: str,
    reranker: ModelSpec | None = None,
) -> dict[str, Any]:
    """Lineage record: which data, model, and library versions produced this index."""
    spec = embedder.spec
    return {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "collection": cfg.collection,
        "source": {"repo": cfg.source_repo, "commit": cfg.source_commit},
        "competitions": [list(pair) for pair in cfg.competitions],
        "tournaments": sorted(
            {f"{c} {s}" for c, s in zip(shots["competition"], shots["season"], strict=True)}
        ),
        "rows": len(shots),
        "goals": int((shots["is_goal"] & ~shots["is_shootout"]).sum()),
        "matches": int(shots["match_id"].nunique()),
        "content_sha256": content_hash(shots, TEXT_COLUMNS[template]),
        "template": template,
        "distance": "cosine",
        "embedding": {
            "model": spec.name,
            "revision": spec.revision,
            "dim": embedder.dim,
            "query_prefix": spec.query_prefix,
            "passage_prefix": spec.passage_prefix,
            "normalized": True,
        },
        "rerank": (
            {"model": reranker.name, "revision": reranker.revision, "fetch_k": cfg.fetch_k}
            if reranker
            else None
        ),
        "versions": {
            "python": platform.python_version(),
            "chromadb": version("chromadb"),
            "sentence-transformers": version("sentence-transformers"),
            "torch": version("torch"),
        },
    }


def index(cfg: Config) -> dict[str, Any]:
    shots = pd.read_parquet(cfg.shots_path)
    embedder = SentenceTransformerEmbedder(cfg.model)
    build_index(shots, embedder, cfg.chroma_dir, cfg.collection, cfg.template, cfg.batch_size)
    manifest = build_manifest(cfg, shots, embedder, cfg.template, cfg.reranker)
    (cfg.chroma_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    rerank = cfg.reranker.name if cfg.reranker else "no reranker"
    log.info(
        "Index ready: %d shots, %s, %s template, %s",
        len(shots),
        cfg.model.name,
        cfg.template,
        rerank,
    )
    return manifest


def main() -> None:
    configure_logging()
    index(load_config())


if __name__ == "__main__":
    main()
