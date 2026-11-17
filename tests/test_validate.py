import pandas as pd

from football_search.transform import match_context, render_fluent, render_terse
from football_search.validate import run_checks, score_mismatches


def tables(shot_rows, match, home_own_goals=1, away_own_goals=0):
    shots = pd.DataFrame(shot_rows).drop(columns="_clock")
    shots["text_fluent"] = [render_fluent(r) for r in shot_rows]
    shots["text_terse"] = [render_terse(r) for r in shot_rows]
    matches = pd.DataFrame(
        [
            {
                **match_context(match),
                "home_own_goals": home_own_goals,
                "away_own_goals": away_own_goals,
            }
        ]
    )
    return shots, matches


def results(shots, matches):
    return {c.name: c.passed for c in run_checks(shots, matches)}


def test_clean_fixture_passes_everything_but_row_counts(shot_rows, match):
    shots, matches = tables(shot_rows, match)
    checks = results(shots, matches)
    # The fixture has 4 shots in one match, deliberately below the realistic range.
    assert not checks.pop("Shot counts per competition plausible")
    assert all(checks.values()), checks


def test_scores_reconcile_with_own_goal(shot_rows, match):
    shots, matches = tables(shot_rows, match)
    assert score_mismatches(shots, matches).empty


def test_missing_own_goal_is_reported(shot_rows, match):
    shots, matches = tables(shot_rows, match, home_own_goals=0)
    bad = score_mismatches(shots, matches)
    assert bad.iloc[0]["recorded"] == "2-1"
    assert bad.iloc[0]["from_events"] == "1-1"


def test_out_of_range_xg_fails(shot_rows, match):
    shots, matches = tables(shot_rows, match)
    shots.loc[0, "xg"] = 1.5
    assert not results(shots, matches)["Values in valid ranges"]


def test_duplicate_ids_and_nulls_fail(shot_rows, match):
    shots, matches = tables(shot_rows, match)
    shots.loc[1, "id"] = shots.loc[0, "id"]
    shots.loc[2, "player"] = None
    checks = results(shots, matches)
    assert not checks["Shot IDs unique"]
    assert not checks["Required fields present"]
