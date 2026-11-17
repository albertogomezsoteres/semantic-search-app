import math

import numpy as np
import pytest

from football_search.index import build_index, content_hash, sanitize_metadata
from football_search.search import Filters, Searcher, build_where


@pytest.fixture
def searcher(tmp_path, shots_df, fake_embedder):
    build_index(shots_df, fake_embedder, tmp_path, "shots", template="fluent", batch_size=2)
    return Searcher(tmp_path, "shots", fake_embedder)


def test_sanitize_metadata_is_chroma_safe(shot_rows):
    row = {**shot_rows[0], "xg": np.float32(0.5), "end_z": math.nan, "position": None}
    row["text_fluent"] = "readable text"
    meta = sanitize_metadata(row)
    assert "end_z" not in meta and "position" not in meta
    assert type(meta["xg"]) is float
    assert meta["tournament"] == "FIFA World Cup 2022"
    assert meta["description"] == "readable text"
    assert all(isinstance(v, str | int | float | bool) for v in meta.values())


def test_content_hash_tracks_text(shots_df):
    first = content_hash(shots_df, "text_fluent")
    assert first == content_hash(shots_df.iloc[::-1], "text_fluent")  # order does not matter
    changed = shots_df.copy()
    changed.loc[0, "text_fluent"] += "!"
    assert first != content_hash(changed, "text_fluent")


def test_build_where():
    assert build_where(None) is None
    assert build_where(Filters()) is None
    assert build_where(Filters(goals_only=True)) == {"is_goal": True}
    where = build_where(Filters(tournaments=("A", "B"), team="Redland", min_xg=0.2))
    assert where == {
        "$and": [
            {"tournament": {"$in": ["A", "B"]}},
            {"team": "Redland"},
            {"xg": {"$gte": 0.2}},
        ]
    }


def test_rebuild_is_idempotent(tmp_path, shots_df, fake_embedder):
    build_index(shots_df, fake_embedder, tmp_path, "shots")
    collection = build_index(shots_df, fake_embedder, tmp_path, "shots")
    assert collection.count() == len(shots_df)


def test_search_finds_the_header(searcher):
    results = searcher.search("header from a cross, top corner", top_k=2)
    assert results[0].id == "s-1"
    assert 0 < results[0].score <= 1
    assert results[0].metadata["player"] == "Alex Red"


def test_filters_narrow_results(searcher):
    goals = searcher.search("shot", top_k=10, filters=Filters(goals_only=True))
    assert {r.id for r in goals} == {"s-1", "s-3"}
    blue = searcher.search("shot", top_k=10, filters=Filters(team="Blueland", min_xg=0.5))
    assert {r.id for r in blue} == {"s-3", "s-4"}
    assert searcher.search("shot", filters=Filters(team="Nobody")) == []


def test_empty_query_returns_nothing(searcher):
    assert searcher.search("   ") == []


def test_query_embeddings_are_cached(searcher):
    searcher.search("volley")
    searcher.search("volley")
    assert searcher._embed.cache_info().hits == 1


def test_reranker_reorders(tmp_path, shots_df, fake_embedder, fake_reranker):
    build_index(shots_df, fake_embedder, tmp_path, "shots", template="moment")
    plain = Searcher(tmp_path, "shots", fake_embedder)
    reranked = Searcher(tmp_path, "shots", fake_embedder, fake_reranker, fetch_k=4)
    query = "shootout penalty missed the target"
    results = reranked.search(query, top_k=2)
    assert results[0].id == "s-4"
    assert results[0].rerank_score is not None
    assert len(results) == 2
    assert [r.id for r in plain.search(query, top_k=4)]  # still works without a reranker


def test_distinct_values(searcher):
    assert searcher.distinct("team") == ["Blueland", "Redland"]
