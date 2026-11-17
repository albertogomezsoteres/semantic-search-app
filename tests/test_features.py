import math

import pytest

from football_search.transform import (
    assist_type,
    distance_to_goal,
    game_state,
    goal_context,
    goal_placement,
    shot_zone,
    xg_band,
)


def test_distance_to_goal():
    assert distance_to_goal(120, 40) == 0
    assert distance_to_goal(108, 40) == pytest.approx(12)  # penalty spot
    assert distance_to_goal(117, 44) == pytest.approx(5)


@pytest.mark.parametrize(
    ("x", "y", "zone"),
    [
        (116, 40, "six_yard_box"),
        (108, 40, "penalty_area"),
        (103, 20, "penalty_area"),
        (98, 40, "edge_of_box"),
        (90, 40, "long_range"),
        (118, 5, "long_range"),  # tight angle by the corner flag, outside the box
    ],
)
def test_shot_zone(x, y, zone):
    assert shot_zone(x, y) == zone


@pytest.mark.parametrize(
    ("period", "minute", "state"),
    [
        (1, 3, "early"),
        (1, 30, "first_half"),
        (1, 46, "first_half_stoppage"),
        (2, 50, "second_half"),
        (2, 80, "late"),
        (2, 93, "stoppage_time"),
        (3, 100, "extra_time"),
        (4, 117, "late_extra_time"),
        (5, 120, "penalty_shootout"),
    ],
)
def test_game_state(period, minute, state):
    assert game_state(period, minute) == state


def test_game_state_rejects_unknown_period():
    with pytest.raises(ValueError):
        game_state(6, 0)


def test_xg_band():
    assert xg_band(0.78) == "big chance"
    assert xg_band(0.3) == "big chance"
    assert xg_band(0.15) == "half chance"
    assert xg_band(0.02) == "speculative effort"


@pytest.mark.parametrize(
    ("pass_data", "expected"),
    [
        ({"type": {"name": "Corner"}, "cross": True}, "corner"),
        ({"cross": True, "cut_back": True}, "cut_back"),
        ({"cross": True}, "cross"),
        ({"technique": {"name": "Through Ball"}}, "through_ball"),
        ({"through_ball": True}, "through_ball"),
        ({"height": {"name": "High Pass"}}, "lofted_pass"),
        ({"height": {"name": "Ground Pass"}}, "pass"),
    ],
)
def test_assist_type(pass_data, expected):
    assert assist_type({"pass": pass_data}) == expected


def test_assist_type_without_key_pass():
    assert assist_type(None) == "none"


def test_goal_placement():
    assert goal_placement(36.8, 2.2) == "top corner"
    assert goal_placement(43.5, 0.2) == "bottom corner"
    assert goal_placement(40.0, 0.2) == "low, down the middle"
    assert goal_placement(40.0, 2.0) == "high, down the middle"
    assert goal_placement(40.0, math.nan) == ""


def test_goal_context():
    # Arguments are (team goals, opponent goals) after the goal, opponent final, team won.
    assert goal_context(1, 0, 0, True) == "opening goal and eventual winner"
    assert goal_context(1, 0, 2, False) == "opening goal"
    assert goal_context(1, 1, 1, False) == "equaliser"
    assert goal_context(3, 2, 2, True) == "winning goal"
    assert goal_context(2, 1, 2, False) == "go-ahead goal"  # opponent equalised later
    assert goal_context(3, 0, 0, True) == "goal that extended the lead"
    assert goal_context(1, 3, 3, False) == "goal that pulled one back"
