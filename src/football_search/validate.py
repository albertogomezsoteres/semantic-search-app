"""Validate: data quality checks on the staged tables. The pipeline stops on failure.

Writes reports/data_quality.md so every build leaves a readable record.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import pandas as pd

from football_search.config import Config, configure_logging, load_config

log = logging.getLogger(__name__)

REQUIRED = [
    "id",
    "match_id",
    "player",
    "team",
    "outcome",
    "x",
    "y",
    "text_fluent",
    "text_moment",
    "text_terse",
]
# Plausible shots per match; a tournament outside this range means something went wrong.
SHOTS_PER_MATCH = (10, 45)
MAX_WORDS = 200


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def check_required(shots: pd.DataFrame) -> Check:
    nulls = {c: int(shots[c].isna().sum()) for c in REQUIRED if shots[c].isna().any()}
    empty = {c: int((shots[c] == "").sum()) for c in REQUIRED if shots[c].dtype == object}
    empty = {c: n for c, n in empty.items() if n}
    bad = {**nulls, **empty}
    return Check("Required fields present", not bad, str(bad) if bad else f"{len(REQUIRED)} fields")


def check_ranges(shots: pd.DataFrame) -> Check:
    problems = {
        "xg": int((~shots["xg"].between(0, 1)).sum()),
        "x": int((~shots["x"].between(0, 120)).sum()),
        "y": int((~shots["y"].between(0, 80)).sum()),
    }
    problems = {k: v for k, v in problems.items() if v}
    detail = str(problems) if problems else "0 <= xg <= 1, 0 <= x <= 120, 0 <= y <= 80"
    return Check("Values in valid ranges", not problems, detail)


def check_unique_ids(shots: pd.DataFrame) -> Check:
    dupes = int(shots["id"].duplicated().sum())
    return Check("Shot IDs unique", dupes == 0, f"{dupes} duplicates")


def check_text_length(shots: pd.DataFrame) -> Check:
    longest = int(shots["text_fluent"].str.split().str.len().max())
    return Check("Text fits the embedding model", longest <= MAX_WORDS, f"longest {longest} words")


def check_row_counts(shots: pd.DataFrame, matches: pd.DataFrame) -> Check:
    lo, hi = SHOTS_PER_MATCH
    bad, lines = [], []
    for (comp, season), group in matches.groupby(["competition", "season"]):
        n_matches = len(group)
        n_shots = int(shots["match_id"].isin(group["match_id"]).sum())
        lines.append(f"{comp} {season}: {n_shots} shots / {n_matches} matches")
        if not lo * n_matches <= n_shots <= hi * n_matches:
            bad.append(f"{comp} {season}")
    detail = "; ".join(lines) + (f" (out of range: {bad})" if bad else "")
    return Check("Shot counts per competition plausible", not bad, detail)


def score_mismatches(shots: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Matches where goals from shot events plus own goals differ from the final score.

    Shootout kicks are excluded: StatsBomb's final score does not include them.
    """
    goals = (
        shots[shots["is_goal"] & ~shots["is_shootout"]]
        .groupby(["match_id", "team"])
        .size()
        .to_dict()
    )
    rows = []
    for m in matches.itertuples():
        home = goals.get((m.match_id, m.home), 0) + m.home_own_goals
        away = goals.get((m.match_id, m.away), 0) + m.away_own_goals
        if (home, away) != (m.home_score, m.away_score):
            rows.append(
                {
                    "match_id": m.match_id,
                    "match": f"{m.home} vs {m.away}",
                    "recorded": f"{m.home_score}-{m.away_score}",
                    "from_events": f"{home}-{away}",
                }
            )
    return pd.DataFrame(rows, columns=["match_id", "match", "recorded", "from_events"])


def check_scores(shots: pd.DataFrame, matches: pd.DataFrame) -> Check:
    bad = score_mismatches(shots, matches)
    detail = f"{len(matches) - len(bad)}/{len(matches)} matches reconcile"
    if len(bad):
        detail += ": " + "; ".join(
            f"{r.match} {r.recorded} vs {r.from_events}" for r in bad.itertuples()
        )
    return Check("Goals reconcile with final scores", bad.empty, detail)


def run_checks(shots: pd.DataFrame, matches: pd.DataFrame) -> list[Check]:
    return [
        check_required(shots),
        check_ranges(shots),
        check_unique_ids(shots),
        check_text_length(shots),
        check_row_counts(shots, matches),
        check_scores(shots, matches),
    ]


def write_report(checks: list[Check], cfg: Config) -> None:
    lines = [
        "# Data quality report",
        "",
        f"Source: `{cfg.source_repo}` at commit `{cfg.source_commit[:10]}`",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    lines += [f"| {c.name} | {'pass' if c.passed else 'FAIL'} | {c.detail} |" for c in checks]
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "data_quality.md").write_text("\n".join(lines) + "\n")


def validate(cfg: Config) -> bool:
    shots = pd.read_parquet(cfg.shots_path)
    matches = pd.read_parquet(cfg.matches_path)
    checks = run_checks(shots, matches)
    for c in checks:
        (log.info if c.passed else log.error)(
            "%s: %s (%s)", c.name, "pass" if c.passed else "FAIL", c.detail
        )
    write_report(checks, cfg)
    return all(c.passed for c in checks)


def main() -> None:
    configure_logging()
    if not validate(load_config()):
        log.error("Data quality checks failed; stopping the pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
