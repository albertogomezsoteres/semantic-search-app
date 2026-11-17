from football_search.transform import render_fluent, render_terse


def by_id(rows):
    return {r["id"]: r for r in rows}


def test_one_row_per_shot(shot_rows):
    assert [r["id"] for r in shot_rows] == ["s-1", "s-2", "s-3", "s-4"]


def test_uses_display_names(shot_rows):
    rows = by_id(shot_rows)
    assert rows["s-1"]["player"] == "Alex Red"
    assert rows["s-1"]["player_full_name"] == "Alex Full Surname Red"
    assert rows["s-1"]["key_passer"] == "Sam Cross"  # no nickname, falls back to name


def test_running_score_and_goal_context(shot_rows):
    rows = by_id(shot_rows)
    assert (rows["s-1"]["team_score"], rows["s-1"]["opp_score"]) == (0, 0)
    assert rows["s-1"]["goal_context"] == "opening goal"
    assert (rows["s-3"]["team_score"], rows["s-3"]["opp_score"]) == (0, 1)
    assert rows["s-3"]["goal_context"] == "equaliser"


def test_assist_and_rebound(shot_rows):
    rows = by_id(shot_rows)
    assert rows["s-1"]["assist_type"] == "cross"
    assert not rows["s-2"]["rebound"]
    assert rows["s-3"]["rebound"]  # 3 seconds after a saved team-mate shot


def test_shootout_is_flagged_and_does_not_change_score(shot_rows):
    kick = by_id(shot_rows)["s-4"]
    assert kick["is_shootout"]
    assert kick["game_state"] == "penalty_shootout"
    assert kick["goal_context"] == ""


def test_fluent_text(shot_rows):
    rows = by_id(shot_rows)
    header = render_fluent(rows["s-1"])
    assert "FIFA World Cup 2022, Final: Redland vs Blueland." in header
    assert "Alex Red (Redland)" in header
    assert "header" in header
    assert "into the top corner" in header
    assert "cross by Sam Cross" in header
    assert "after winning the ball in the air" in header
    assert "the opening goal" in header

    volley = render_fluent(rows["s-3"])
    assert "first-time right-footed volley" in volley
    assert "rebound" in volley
    assert "counter-attack" in volley
    assert "the equaliser" in volley

    penalty = render_fluent(rows["s-4"])
    assert "penalty shootout" in penalty
    assert "from the penalty spot" in penalty
    assert "missed the target" in penalty


def test_texts_are_short_and_deterministic(shot_rows):
    for row in shot_rows:
        text = render_fluent(row)
        assert len(text.split()) <= 200
        assert text == render_fluent(row)


def test_terse_text(shot_rows):
    text = render_terse(by_id(shot_rows)["s-3"])
    assert "Goal" in text and "Volley" in text and "rebound" in text
    assert text.count(" | ") >= 8


def test_moment_text_leaves_out_match_context(shot_rows):
    from football_search.transform import render_moment

    text = render_moment(by_id(shot_rows)["s-1"])
    assert "Alex Red (Redland) scored with a header" in text
    assert "FIFA" not in text and "minute" not in text and "1-0" not in text
    assert "It happened early in the first half." in text
