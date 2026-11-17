"""Transform: flatten raw shot events into one described "moment" per shot.

Output is data/staged/shots.parquet (one row per shot, with features and two
text versions) and data/staged/matches.parquet (match context used by the
quality checks). One shot is one document: it is a complete, self-contained
event, so there is nothing to split.
"""

from __future__ import annotations

import json
import logging
import math
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from football_search.config import Config, configure_logging, load_config

log = logging.getLogger(__name__)

# StatsBomb pitch: 120 x 80 yards, every team attacks toward x = 120.
GOAL_X, GOAL_Y = 120.0, 40.0
GOAL_HALF_WIDTH = 4.0  # posts at y = 36 and y = 44
SHOOTOUT_PERIOD = 5

ZONE_PHRASES = {
    "six_yard_box": "from inside the six-yard box",
    "penalty_area": "from inside the penalty area",
    "edge_of_box": "from just outside the box",
    "long_range": "from long range",
}

GAME_STATE_PHRASES = {
    "early": "early in the first half",
    "first_half": "first half",
    "first_half_stoppage": "first-half stoppage time",
    "second_half": "second half",
    "late": "late in the second half",
    "stoppage_time": "second-half stoppage time",
    "extra_time": "extra time",
    "late_extra_time": "late in extra time",
    "penalty_shootout": "penalty shootout",
}

TECHNIQUE_NOUNS = {
    "Normal": "shot",
    "Volley": "volley",
    "Half Volley": "half-volley",
    "Lob": "chip",
    "Overhead Kick": "overhead kick",
    "Backheel": "backheel",
    "Diving Header": "diving header",
}

PLAY_PATTERN_PHRASES = {
    "Regular Play": "The move came in open play.",
    "From Corner": "The move started from a corner.",
    "From Free Kick": "The move started from a free kick.",
    "From Throw In": "The move started from a throw-in.",
    "From Counter": "It came on a counter-attack.",
    "From Goal Kick": "The move started from a goal kick.",
    "From Keeper": "The move started with the goalkeeper.",
    "From Kick Off": "The move started from the kick-off.",
}

# Per outcome: ways to start the sentence, and how it ends.
OUTCOME_PHRASES = {
    "Goal": (["scored with", "found the net with"], ""),
    "Saved": (["tried", "went for goal with"], ", but the goalkeeper saved it"),
    "Saved to Post": (
        ["tried", "went for goal with"],
        ", but the goalkeeper pushed it onto the post",
    ),
    "Saved Off Target": (["tried", "went for goal with"], ", but the goalkeeper turned it wide"),
    "Off T": (["tried", "went for goal with"], ", but it missed the target"),
    "Wayward": (["tried", "went for goal with"], ", but it was a wayward effort, well off target"),
    "Post": (["tried", "went for goal with"], ", but it hit the woodwork"),
    "Blocked": (["tried", "went for goal with"], ", but a defender blocked it"),
}

SHOT_FLAGS = ("first_time", "one_on_one", "open_goal", "aerial_won", "deflected", "follows_dribble")

# A shot this soon after a team-mate's saved, blocked, or woodwork shot counts as a rebound.
REBOUND_WINDOW_S = 5
REBOUND_SOURCES = {"Saved", "Blocked", "Post", "Saved to Post"}


# --- Derived features (pure functions) -------------------------------------------------


def distance_to_goal(x: float, y: float) -> float:
    """Straight-line distance in yards from (x, y) to the centre of the goal."""
    return math.hypot(GOAL_X - x, GOAL_Y - y)


def shot_zone(x: float, y: float) -> str:
    """Label where on the pitch a shot was taken from."""
    if x >= 114 and 30 <= y <= 50:
        return "six_yard_box"
    if x >= 102 and 18 <= y <= 62:
        return "penalty_area"
    if distance_to_goal(x, y) <= 25:
        return "edge_of_box"
    return "long_range"


def game_state(period: int, minute: int) -> str:
    """Label the phase of the match. `minute` is StatsBomb's running match clock."""
    if period == 1:
        if minute < 15:
            return "early"
        return "first_half_stoppage" if minute >= 45 else "first_half"
    if period == 2:
        if minute >= 90:
            return "stoppage_time"
        return "late" if minute >= 75 else "second_half"
    if period in (3, 4):
        return "late_extra_time" if minute >= 115 else "extra_time"
    if period == SHOOTOUT_PERIOD:
        return "penalty_shootout"
    raise ValueError(f"Unknown period {period}")


def xg_band(xg: float) -> str:
    """Bucket expected goals into how good the chance was."""
    if xg >= 0.3:
        return "big chance"
    if xg >= 0.1:
        return "half chance"
    return "speculative effort"


def assist_type(key_pass: dict[str, Any] | None) -> str:
    """Classify the pass that set up the shot ("none" when there was no key pass)."""
    if key_pass is None:
        return "none"
    p = key_pass.get("pass", {})
    set_piece = {"Corner": "corner", "Free Kick": "free_kick", "Throw-in": "throw_in"}
    pass_type = p.get("type", {}).get("name")
    if pass_type in set_piece:
        return set_piece[pass_type]
    if p.get("cut_back"):
        return "cut_back"
    if p.get("cross"):
        return "cross"
    if p.get("through_ball") or p.get("technique", {}).get("name") == "Through Ball":
        return "through_ball"
    if p.get("switch"):
        return "switch"
    if p.get("height", {}).get("name") == "High Pass":
        return "lofted_pass"
    return "pass"


def goal_placement(end_y: float | None, end_z: float | None) -> str:
    """Describe where in the goal a shot ended up ("" if height is unknown)."""
    if end_y is None or end_z is None or math.isnan(end_z):
        return ""
    wide = abs(end_y - GOAL_Y) >= GOAL_HALF_WIDTH * 0.6
    if end_z >= 1.6:
        return "top corner" if wide else "high, down the middle"
    if end_z <= 0.5:
        return "bottom corner" if wide else "low, down the middle"
    return "side of the net" if wide else "middle of the goal"


def goal_context(team_goals: int, opp_goals: int, opp_final: int, team_won: bool) -> str:
    """Describe what a goal did to the score. Arguments are counted after the goal."""
    if team_goals + opp_goals == 1:
        return "opening goal and eventual winner" if team_won and opp_final == 0 else "opening goal"
    if team_goals == opp_goals:
        return "equaliser"
    if team_goals == opp_goals + 1:
        return "winning goal" if team_won and team_goals == opp_final + 1 else "go-ahead goal"
    if team_goals > opp_goals:
        return "goal that extended the lead"
    return "goal that pulled one back"


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _pick(options: list[str], key: str) -> str:
    """Choose a phrasing deterministically from the event id, for some variety in the text."""
    return options[zlib.crc32(key.encode()) % len(options)]


def describe_strike(row: dict[str, Any]) -> str:
    """Noun phrase for the shot itself, e.g. "a first-time left-footed volley"."""
    body, technique, shot_type = row["body_part"], row["technique"], row["shot_type"]
    if body == "Head":
        words = ["diving header" if technique == "Diving Header" else "header"]
    else:
        noun = TECHNIQUE_NOUNS.get(technique, "shot")
        if shot_type == "Penalty":
            noun = "penalty"
        elif shot_type == "Free Kick":
            noun = "direct free kick"
        elif shot_type == "Corner":
            noun = "shot straight from a corner kick"
        foot = {"Right Foot": "right-footed", "Left Foot": "left-footed"}.get(body)
        words = [foot, noun] if foot else [noun, "with another part of the body"]
        words = [w for w in words if w]
    if row["first_time"] and shot_type == "Open Play":
        words.insert(0, "first-time")
    phrase = " ".join(words)
    article = "an" if phrase[0] in "aeiou" else "a"
    return f"{article} {phrase}"


def describe_assist(row: dict[str, Any]) -> str:
    """Phrase for how the chance was created ("" when there is nothing to say)."""
    passer = row["key_passer"]
    phrases = {
        "corner": f"from a corner taken by {passer}",
        "free_kick": f"from a free kick delivered by {passer}",
        "throw_in": f"from a throw-in by {passer}",
        "cut_back": f"from a cut-back by {passer}",
        "cross": f"from a cross by {passer}",
        "through_ball": f"from a through ball by {passer}",
        "switch": f"from a long switch of play by {passer}",
        "lofted_pass": f"from a lofted pass by {passer}",
        "pass": f"from a pass by {passer}",
    }
    if row["assist_type"] in phrases:
        return f"Set up {phrases[row['assist_type']]}."
    if row["rebound"]:
        return "It was a rebound after an earlier shot."
    if row["follows_dribble"]:
        return "It came after dribbling past a defender."
    return ""


def where_phrase(row: dict[str, Any]) -> str:
    if row["shot_type"] == "Penalty":
        return "from the penalty spot"
    return f"{ZONE_PHRASES[row['zone']]}, about {round(row['distance'])} yards out"


def score_phrase(team: str, opponent: str, team_goals: int, opp_goals: int) -> str:
    if team_goals == opp_goals:
        return f"with the score level at {team_goals}-{opp_goals}"
    state = "leading" if team_goals > opp_goals else "trailing"
    return f"with {team} {state} {team_goals}-{opp_goals} against {opponent}"


def render_fluent(row: dict[str, Any]) -> str:
    """Natural-language description of a shot (the default document text)."""
    parts = [
        f"{row['competition']} {row['season']}, {row['stage']}: {row['home']} vs {row['away']}."
    ]
    if row["is_shootout"]:
        parts.append("In the penalty shootout.")
    else:
        timing = GAME_STATE_PHRASES[row["game_state"]]
        parts.append(
            f"In the {_ordinal(row['minute'])} minute ({timing}), "
            f"{score_phrase(row['team'], row['opponent'], row['team_score'], row['opp_score'])}."
        )

    leads, tail = OUTCOME_PHRASES[row["outcome"]]
    sentence = (
        f"{row['player']} ({row['team']}) {_pick(leads, row['id'])} {describe_strike(row)} "
        f"{where_phrase(row)}{tail}"
    )
    placement = row["placement"]
    if row["is_goal"] and placement:
        into = "down the middle" not in placement
        sentence += f", into the {placement}" if into else f", {placement}"
    parts.append(sentence + ".")

    assist = describe_assist(row)
    if assist:
        parts.append(assist)

    details = []
    if row["one_on_one"]:
        details.append("one-on-one with the goalkeeper")
    if row["open_goal"]:
        details.append("with the goal open")
    if row["under_pressure"]:
        details.append("under pressure from a defender")
    if row["aerial_won"]:
        details.append("after winning the ball in the air")
    if row["deflected"]:
        details.append("and the ball took a deflection")
    if details:
        parts.append("Taken " + ", ".join(details) + ".")

    if row["shot_type"] == "Open Play" and row["play_pattern"] in PLAY_PATTERN_PHRASES:
        parts.append(PLAY_PATTERN_PHRASES[row["play_pattern"]])

    if row["is_goal"] and not row["is_shootout"]:
        parts.append(
            f"It made the score {row['team_score'] + 1}-{row['opp_score']} "
            f"to {row['team']}, the {row['goal_context']}."
        )
    parts.append(f"Expected goals (xG) {row['xg']:.2f}, a {row['xg_band']}.")
    return " ".join(parts)


def render_terse(row: dict[str, Any]) -> str:
    """Compact attribute string, used as a baseline in the template experiment."""
    flags = [f.replace("_", " ") for f in SHOT_FLAGS if row[f]]
    if row["rebound"]:
        flags.append("rebound")
    if row["under_pressure"]:
        flags.append("under pressure")
    fields = [
        f"{row['competition']} {row['season']} {row['stage']}",
        f"{row['home']} vs {row['away']}",
        f"{row['player']} ({row['team']})",
        f"minute {row['minute']} {row['game_state'].replace('_', ' ')}",
        row["outcome"],
        row["shot_type"],
        row["body_part"],
        row["technique"],
        row["play_pattern"],
        f"{row['zone'].replace('_', ' ')} {round(row['distance'])} yd",
        f"assist {row['assist_type'].replace('_', ' ')}",
        f"placement {row['placement']}" if row["placement"] else "",
        f"{row['goal_context']}" if row["goal_context"] else "",
        ", ".join(flags),
        f"xG {row['xg']:.2f} {row['xg_band']}",
    ]
    return " | ".join(f for f in fields if f)


# --- Flattening ------------------------------------------------------------------------


def match_context(match: dict[str, Any]) -> dict[str, Any]:
    """The match-level fields every shot in that match shares."""
    return {
        "match_id": match["match_id"],
        "competition": match["competition"]["competition_name"],
        "season": match["season"]["season_name"],
        "stage": match["competition_stage"]["name"],
        "match_date": match["match_date"],
        "home": match["home_team"]["home_team_name"],
        "away": match["away_team"]["away_team_name"],
        "home_score": match["home_score"],
        "away_score": match["away_score"],
    }


def display_names(lineups: list[dict[str, Any]]) -> dict[int, str]:
    """Map player id to the name fans use (nickname when StatsBomb has one)."""
    return {
        p["player_id"]: p.get("player_nickname") or p["player_name"]
        for team in lineups
        for p in team["lineup"]
    }


def _clock(e: dict[str, Any]) -> int:
    return e["minute"] * 60 + e["second"]


def shots_from_events(
    events: list[dict[str, Any]], match: dict[str, Any], names: dict[int, str] | None = None
) -> list[dict[str, Any]]:
    """Turn one match's events into shot rows, tracking the running score as we go."""
    names = names or {}

    def name(player: dict[str, Any]) -> str:
        return names.get(player["id"], player["name"])

    ctx = match_context(match)
    final = {ctx["home"]: ctx["home_score"], ctx["away"]: ctx["away_score"]}
    score = {ctx["home"]: 0, ctx["away"]: 0}
    passes = {e["id"]: e for e in events if e["type"]["name"] == "Pass"}

    rows: list[dict[str, Any]] = []
    for e in sorted(events, key=lambda ev: ev["index"]):
        kind = e["type"]["name"]
        if kind == "Own Goal For" and e["period"] < SHOOTOUT_PERIOD:
            score[e["team"]["name"]] += 1
            continue
        if kind != "Shot":
            continue

        shot = e["shot"]
        team = e["team"]["name"]
        opponent = ctx["away"] if team == ctx["home"] else ctx["home"]
        x, y = e["location"][:2]
        end = shot.get("end_location", [])
        key_pass = passes.get(shot.get("key_pass_id"))
        is_goal = shot["outcome"]["name"] == "Goal"
        is_shootout = e["period"] == SHOOTOUT_PERIOD
        xg = float(shot.get("statsbomb_xg", 0.0))
        prev = rows[-1] if rows else None
        rebound = bool(
            prev
            and key_pass is None
            and prev["team"] == team
            and prev["period"] == e["period"]
            and prev["outcome"] in REBOUND_SOURCES
            and _clock(e) - prev["_clock"] <= REBOUND_WINDOW_S
        )

        row = {
            **ctx,
            "id": e["id"],
            "team": team,
            "opponent": opponent,
            "player": name(e["player"]),
            "player_full_name": e["player"]["name"],
            "position": e.get("position", {}).get("name", ""),
            "period": e["period"],
            "minute": e["minute"] + 1,  # football convention: 0:30 is the 1st minute
            "second": e["second"],
            "game_state": game_state(e["period"], e["minute"]),
            "outcome": shot["outcome"]["name"],
            "is_goal": is_goal,
            "is_shootout": is_shootout,
            "shot_type": shot["type"]["name"],
            "body_part": shot["body_part"]["name"],
            "technique": shot["technique"]["name"],
            "play_pattern": e["play_pattern"]["name"],
            "under_pressure": bool(e.get("under_pressure", False)),
            **{flag: bool(shot.get(flag, False)) for flag in SHOT_FLAGS},
            "rebound": rebound,
            "xg": xg,
            "xg_band": xg_band(xg),
            "x": float(x),
            "y": float(y),
            "distance": distance_to_goal(x, y),
            "zone": shot_zone(x, y),
            "end_x": float(end[0]) if len(end) > 0 else float("nan"),
            "end_y": float(end[1]) if len(end) > 1 else float("nan"),
            "end_z": float(end[2]) if len(end) > 2 else float("nan"),
            "assist_type": assist_type(key_pass),
            "key_passer": name(key_pass["player"]) if key_pass else "",
            "team_score": score[team],
            "opp_score": score[opponent],
            "goal_context": "",
            "placement": "",
            "_clock": _clock(e),
        }
        if is_goal:
            row["placement"] = goal_placement(row["end_y"], row["end_z"])
            if not is_shootout:
                score[team] += 1
                row["goal_context"] = goal_context(
                    score[team], score[opponent], final[opponent], final[team] > final[opponent]
                )
        rows.append(row)
    return rows


def own_goals(events: list[dict[str, Any]]) -> Counter:
    """Count own goals per benefiting team (outside the shootout)."""
    return Counter(
        e["team"]["name"]
        for e in events
        if e["type"]["name"] == "Own Goal For" and e["period"] < SHOOTOUT_PERIOD
    )


def build_tables(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw JSON for all configured competitions and return (shots, matches)."""
    shot_rows, match_rows = [], []
    for comp_id, season_id in cfg.competitions:
        matches_file = cfg.raw_dir / "matches" / str(comp_id) / f"{season_id}.json"
        for match in json.loads(matches_file.read_text()):
            events_file = cfg.raw_dir / "events" / f"{match['match_id']}.json"
            lineups_file = cfg.raw_dir / "lineups" / f"{match['match_id']}.json"
            events = json.loads(events_file.read_text())
            names = display_names(json.loads(lineups_file.read_text()))
            shot_rows.extend(shots_from_events(events, match, names))
            ctx = match_context(match)
            og = own_goals(events)
            match_rows.append(
                {**ctx, "home_own_goals": og[ctx["home"]], "away_own_goals": og[ctx["away"]]}
            )

    shots = pd.DataFrame(shot_rows).drop(columns="_clock")
    shots["text_fluent"] = [render_fluent(r) for r in shot_rows]
    shots["text_terse"] = [render_terse(r) for r in shot_rows]
    shots = shots.sort_values(["match_date", "match_id", "period", "minute", "second"])
    return shots.reset_index(drop=True), pd.DataFrame(match_rows)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def transform(cfg: Config) -> pd.DataFrame:
    shots, matches = build_tables(cfg)
    write_parquet(shots, cfg.shots_path)
    write_parquet(matches, cfg.matches_path)
    words = shots["text_fluent"].str.split().str.len()
    log.info(
        "Wrote %d shots from %d matches to %s (text length: median %d, max %d words)",
        len(shots),
        len(matches),
        cfg.shots_path,
        words.median(),
        words.max(),
    )
    return shots


def main() -> None:
    configure_logging()
    transform(load_config())


if __name__ == "__main__":
    main()
