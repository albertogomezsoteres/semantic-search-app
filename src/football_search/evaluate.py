"""Evaluate: score retrieval quality for each embedding model, text template, and reranker.

Ground truth comes from the structured data: each test query has a rule over shot
metadata, and every shot matching it counts as relevant. No hand labelling.
Results go to reports/eval.json and reports/eval.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from football_search.config import Config, ModelSpec, configure_logging, load_config
from football_search.embeddings import (
    CrossEncoderReranker,
    Embedder,
    Reranker,
    SentenceTransformerEmbedder,
)
from football_search.index import build_index
from football_search.search import Searcher

log = logging.getLogger(__name__)

TEMPLATES = ("fluent", "moment", "terse")
SPLITS = ("dev", "test")
K = 10
METRICS = ["p@5", "p@10", "r@5", "r@10", "mrr", "ndcg@10", "latency_ms"]
NO_RERANK = "none"
SEED = 0


# --- Relevance rules ------------------------------------------------------------------


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Read the query file into a flat list, tagging each query with its split."""
    raw = yaml.safe_load(path.read_text())
    return [{**q, "split": split} for split in SPLITS for q in raw.get(split, [])]


def rule_mask(shots: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    """Boolean mask of the shots that satisfy every condition in the rule."""
    mask = pd.Series(True, index=shots.index)
    for column, cond in rule.items():
        values = shots[column]
        if isinstance(cond, list):
            mask &= values.isin(cond)
        elif isinstance(cond, dict):
            for op, target in cond.items():
                if op == "gte":
                    mask &= values >= target
                elif op == "lte":
                    mask &= values <= target
                elif op == "ne":
                    mask &= values != target
                else:
                    raise ValueError(f"Unknown operator {op!r} in rule for {column!r}")
        else:
            mask &= values == cond
    return mask


def relevant_ids(shots: pd.DataFrame, rule: dict[str, Any]) -> set[str]:
    return set(shots.loc[rule_mask(shots, rule), "id"])


# --- Metrics (binary relevance) -------------------------------------------------------


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(r in relevant for r in ranked[:k]) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(r in relevant for r in ranked[:k]) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def score_query(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    return {
        "p@5": precision_at_k(ranked, relevant, 5),
        "p@10": precision_at_k(ranked, relevant, 10),
        "r@5": recall_at_k(ranked, relevant, 5),
        "r@10": recall_at_k(ranked, relevant, 10),
        "mrr": reciprocal_rank(ranked, relevant),
        "ndcg@10": ndcg_at_k(ranked, relevant, 10),
    }


# --- Running the experiments ------------------------------------------------------------


def run_config(
    shots: pd.DataFrame,
    queries: list[dict[str, Any]],
    embedder: Embedder,
    template: str,
    rerankers: dict[str, Reranker],
    fetch_k: int,
) -> dict[str, pd.DataFrame]:
    """Build a throwaway index for one (model, template) pair and evaluate it with each
    reranker option. The embedding candidates are fetched once and shared."""
    per_reranker: dict[str, list[dict[str, Any]]] = {NO_RERANK: [], **{n: [] for n in rerankers}}
    with tempfile.TemporaryDirectory() as tmp:
        build_index(shots, embedder, Path(tmp), "eval", template)
        searcher = Searcher(Path(tmp), "eval", embedder)
        for q in queries:
            relevant = relevant_ids(shots, q["relevant"])
            base = {"split": q["split"], "query": q["query"], "n_relevant": len(relevant)}

            start = time.perf_counter()
            candidates = searcher.retrieve(q["query"], fetch_k)
            retrieve_ms = (time.perf_counter() - start) * 1000
            ranked = [r.id for r in candidates]
            per_reranker[NO_RERANK].append(
                {**base, **score_query(ranked, relevant), "latency_ms": retrieve_ms}
            )

            for name, reranker in rerankers.items():
                start = time.perf_counter()
                reranked = searcher.rerank(q["query"], list(candidates), reranker)
                rerank_ms = (time.perf_counter() - start) * 1000
                ranked = [r.id for r in reranked]
                per_reranker[name].append(
                    {**base, **score_query(ranked, relevant), "latency_ms": retrieve_ms + rerank_ms}
                )
    return {name: pd.DataFrame(rows) for name, rows in per_reranker.items()}


def random_baseline(
    shots: pd.DataFrame, queries: list[dict[str, Any]], draws: int = 20
) -> pd.DataFrame:
    """Expected scores of a random ranking (fixed seed), to put the numbers in context."""
    rng = np.random.default_rng(SEED)
    ids = shots["id"].to_numpy()
    rows = []
    for q in queries:
        relevant = relevant_ids(shots, q["relevant"])
        scores = [
            score_query(list(rng.choice(ids, K, replace=False)), relevant) for _ in range(draws)
        ]
        mean = pd.DataFrame(scores).mean().to_dict()
        rows.append(
            {"split": q["split"], "query": q["query"], "n_relevant": len(relevant), **mean,
             "latency_ms": 0.0}
        )  # fmt: skip
    return pd.DataFrame(rows)


def summarize(per_query: pd.DataFrame) -> list[dict[str, Any]]:
    """Mean of each metric, per split."""
    return [
        {"split": split, **{m: round(float(group[m].mean()), 4) for m in METRICS}}
        for split, group in per_query.groupby("split", sort=False)
    ]


def pick_best(summary: pd.DataFrame, latency_budget_ms: float) -> pd.Series:
    """Best dev nDCG@10 (then MRR) among configurations within the latency budget."""
    dev = summary[(summary["split"] == "dev") & (summary["model"] != "random baseline")]
    within = dev[dev["latency_ms"] <= latency_budget_ms]
    return within.sort_values(["ndcg@10", "mrr"], ascending=False).iloc[0]


def config_key(model: str, template: str, reranker: str) -> str:
    return f"{model} | {template} | {reranker}"


def write_reports(
    cfg: Config, summary: pd.DataFrame, details: dict[str, pd.DataFrame], meta: dict[str, Any]
) -> None:
    best = pick_best(summary, cfg.latency_budget_ms)
    best_key = config_key(best["model"], best["template"], best["reranker"])
    payload = {
        "k": K,
        **meta,
        "latency_budget_ms": cfg.latency_budget_ms,
        "best": {
            "model": best["model"],
            "template": best["template"],
            "reranker": best["reranker"],
        },
        "summary": summary.to_dict(orient="records"),
        "per_query": {key: df.to_dict(orient="records") for key, df in details.items()},
    }
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "eval.json").write_text(json.dumps(payload, indent=2) + "\n")

    cols = ["model", "template", "reranker", "p@10", "r@10", "mrr", "ndcg@10", "latency_ms"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"{meta['n_shots']} shots. Queries are written the way a fan might type them "
        f"({meta['n_queries']['dev']} dev, {meta['n_queries']['test']} test). A result counts "
        "as relevant when the shot's metadata matches the query's rule (see "
        "`eval/queries.yaml`). The configuration is chosen on dev: the best nDCG@10 among "
        f"those averaging at most {cfg.latency_budget_ms:.0f} ms per query. Test was written "
        "before any tuning and is only reported. Latency is per query on a laptop CPU "
        "(embed + search + rerank).",
        "",
    ]
    if meta["skipped_queries"]:
        skipped = ", ".join(meta["skipped_queries"])
        lines += [f"Skipped (no relevant shots in this data): {skipped}", ""]
    for split in SPLITS:
        table = summary[summary["split"] == split][cols]
        table = table.sort_values(["ndcg@10", "mrr"], ascending=False)
        lines += [f"## {split.capitalize()} queries", "", _markdown_table(table), ""]
    lines += [
        f"Chosen on dev: **{best['model']}**, **{best['template']}** template, "
        f"reranker **{best['reranker']}**.",
        "",
        f"## Per-query results ({best_key})",
        "",
        _markdown_table(
            details[best_key][["split", "query", "n_relevant", "p@10", "mrr", "ndcg@10"]]
        ),
    ]
    (cfg.reports_dir / "eval.md").write_text("\n".join(lines) + "\n")


def _markdown_table(df: pd.DataFrame) -> str:
    def fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def evaluate(
    cfg: Config,
    models: list[str] | None = None,
    templates: tuple[str, ...] = TEMPLATES,
    rerankers: list[str] | None = None,
) -> pd.DataFrame:
    shots = pd.read_parquet(cfg.shots_path)
    queries = load_queries(cfg.eval_queries)
    # A query with no relevant shots in this data cannot be scored, so it is skipped.
    empty = [q["query"] for q in queries if not relevant_ids(shots, q["relevant"])]
    if empty:
        log.warning("Skipping %d queries with no relevant shots: %s", len(empty), empty)
    queries = [q for q in queries if q["query"] not in empty]

    specs = [cfg.model_spec(m) for m in models] if models else list(cfg.candidates)
    rerank_specs: list[ModelSpec] = (
        [cfg.model_spec(r) for r in rerankers]
        if rerankers is not None
        else list(cfg.rerank_candidates)
    )
    loaded_rerankers = {s.name: CrossEncoderReranker(s) for s in rerank_specs}

    details = {config_key("random baseline", "-", "-"): random_baseline(shots, queries)}
    for spec in specs:
        embedder = SentenceTransformerEmbedder(spec)
        for template in templates:
            log.info("Evaluating %s with the %s template", spec.name, template)
            results = run_config(shots, queries, embedder, template, loaded_rerankers, cfg.fetch_k)
            for reranker, per_query in results.items():
                details[config_key(spec.name, template, reranker)] = per_query

    rows = []
    for key, per_query in details.items():
        model, template, reranker = key.split(" | ")
        for split_row in summarize(per_query):
            rows.append({"model": model, "template": template, "reranker": reranker, **split_row})
    summary = pd.DataFrame(rows)
    meta = {
        "n_shots": len(shots),
        "n_queries": {s: sum(q["split"] == s for q in queries) for s in SPLITS},
        "skipped_queries": empty,
    }
    write_reports(cfg, summary, details, meta)
    best = pick_best(summary, cfg.latency_budget_ms)
    log.info("Best on dev: %s / %s / %s", best["model"], best["template"], best["reranker"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--models", nargs="*", help="Subset of embedding model names")
    parser.add_argument("--templates", nargs="*", default=list(TEMPLATES), choices=TEMPLATES)
    parser.add_argument("--rerankers", nargs="*", help="Subset of reranker names (none: no flag)")
    args = parser.parse_args()
    configure_logging()
    evaluate(load_config(), args.models, tuple(args.templates), args.rerankers)


if __name__ == "__main__":
    main()
