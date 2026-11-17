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


@pytest.fixture
def shots_df(shot_rows):
    import pandas as pd

    from football_search.transform import render_fluent, render_moment, render_terse

    df = pd.DataFrame(shot_rows).drop(columns="_clock")
    df["text_fluent"] = [render_fluent(r) for r in shot_rows]
    df["text_moment"] = [render_moment(r) for r in shot_rows]
    df["text_terse"] = [render_terse(r) for r in shot_rows]
    return df


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder, so tests need no model download."""

    def __init__(self, dim: int = 64):
        from football_search.config import ModelSpec

        self.spec = ModelSpec("fake/bag-of-words", "0" * 40)
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str):
        import re
        import zlib

        import numpy as np

        v = np.zeros(self._dim)
        for word in re.findall(r"[a-z]+", text.lower()):
            v[zlib.crc32(word.encode()) % self._dim] += 1
        return v / (np.linalg.norm(v) or 1.0)

    def embed_passages(self, texts, batch_size=64):
        import numpy as np

        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    """Scores by how many query words appear in the passage."""

    def __init__(self):
        from football_search.config import ModelSpec

        self.spec = ModelSpec("fake/overlap", "0" * 40)

    def score(self, query, passages):
        import numpy as np

        words = set(query.lower().split())
        return np.array([sum(w in p.lower() for w in words) for p in passages], dtype=float)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_reranker():
    return FakeReranker()
