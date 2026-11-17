import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def match() -> dict:
    return _load("match.json")


@pytest.fixture
def events() -> list[dict]:
    return _load("events.json")


@pytest.fixture
def lineups() -> list[dict]:
    return _load("lineups.json")


@pytest.fixture
def shot_rows(events, match, lineups) -> list[dict]:
    from football_search.transform import display_names, shots_from_events

    return shots_from_events(events, match, display_names(lineups))
