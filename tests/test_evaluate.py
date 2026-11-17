import pandas as pd
import pytest

from football_search.config import DEFAULT_CONFIG, load_config
from football_search.evaluate import (
    load_queries,
    ndcg_at_k,
    pick_best,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_ids,
    rule_mask,
)


def test_ranking_metrics():
    ranked = ["a", "x", "b", "y", "z"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(ranked, relevant, 5) == pytest.approx(2 / 5)
    assert recall_at_k(ranked, relevant, 5) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranked, relevant) == 1.0
    assert reciprocal_rank(["x", "y", "b"], relevant) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], relevant) == 0.0
    assert ndcg_at_k(["a", "b", "c"], relevant, 10) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(ranked, relevant, 10) < 1
    assert ndcg_at_k(["x"], set(), 10) == 0.0


def test_rule_operators(shots_df):
    assert relevant_ids(shots_df, {"is_goal": True}) == {"s-1", "s-3"}
    assert relevant_ids(shots_df, {"outcome": ["Saved", "Off T"]}) == {"s-2", "s-4"}
    assert relevant_ids(shots_df, {"xg": {"gte": 0.5}, "is_shootout": False}) == {"s-3"}
    assert relevant_ids(shots_df, {"outcome": {"ne": "Goal"}}) == {"s-2", "s-4"}
    with pytest.raises(ValueError):
        rule_mask(shots_df, {"xg": {"between": 1}})


def test_query_file_is_valid(shots_df):
    queries = load_queries(load_config(DEFAULT_CONFIG).eval_queries)
    assert {q["split"] for q in queries} == {"dev", "test"}
    assert len({q["query"] for q in queries}) == len(queries)  # no duplicates
    for q in queries:
        missing = set(q["relevant"]) - set(shots_df.columns)
        assert not missing, (q["query"], missing)


def test_pick_best_respects_latency_budget():
    summary = pd.DataFrame(
        [
            {"model": "big", "template": "t", "reranker": "r", "split": "dev",
             "ndcg@10": 0.9, "mrr": 0.9, "latency_ms": 900.0},
            {"model": "small", "template": "t", "reranker": "none", "split": "dev",
             "ndcg@10": 0.5, "mrr": 0.5, "latency_ms": 20.0},
            {"model": "small", "template": "t", "reranker": "none", "split": "test",
             "ndcg@10": 0.99, "mrr": 0.99, "latency_ms": 20.0},
        ]
    )  # fmt: skip
    assert pick_best(summary, latency_budget_ms=300)["model"] == "small"
    assert pick_best(summary, latency_budget_ms=1000)["model"] == "big"
