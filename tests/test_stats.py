from football_search import stats
from football_search.index import sanitize_metadata


def test_sql_summaries(shots_df):
    shots = shots_df.assign(tournament="FIFA World Cup 2022")
    scorers = stats.run(stats.TOP_SCORERS, shots)
    assert set(scorers["player"]) == {"Alex Red", "Jo Blue"}  # shootout kick excluded
    assert scorers["goals"].sum() == 2

    body = stats.run(stats.GOALS_BY_BODY_PART, shots).set_index("body_part")
    assert body.loc["Head", "goals"] == 1
    assert body["pct"].sum() == 100

    assert stats.run(stats.TOP_SCORERS, shots, ["Another Cup"]).empty


def test_queries_run_on_index_metadata(shot_rows):
    import pandas as pd

    rows = [{**r, "text_fluent": "x"} for r in shot_rows]
    metadata = pd.DataFrame([sanitize_metadata(r) for r in rows])
    for sql in stats.QUERIES.values():
        assert not stats.run(sql, metadata).empty
