import json

import pytest

from football_search.config import DEFAULT_CONFIG, load_config
from football_search.extract import Downloader, check_source, find_missing_competitions


@pytest.fixture
def cfg(tmp_path):
    from dataclasses import replace

    return replace(load_config(DEFAULT_CONFIG), raw_dir=tmp_path / "raw", request_delay_s=0)


class FakeResponse:
    content = b'{"ok": true}'

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return FakeResponse()


def test_find_missing_competitions():
    comps = [{"competition_id": 43, "season_id": 106}]
    assert find_missing_competitions(comps, ((43, 106), (55, 282))) == [(55, 282)]


def test_downloader_uses_pinned_commit_and_skips_existing(cfg):
    session = FakeSession()
    dl = Downloader(cfg, session=session)
    path = dl.fetch("matches/43/106.json")
    assert json.loads(path.read_text()) == {"ok": True}
    assert cfg.source_commit in session.urls[0]

    dl.fetch("matches/43/106.json")
    assert len(session.urls) == 1
    assert (dl.downloaded, dl.skipped) == (1, 1)
    assert not list(path.parent.glob("*.part"))


def test_check_source_refuses_mixed_commits(cfg):
    check_source(cfg)
    check_source(cfg)  # same commit: fine

    from dataclasses import replace

    with pytest.raises(RuntimeError, match="config.yaml pins"):
        check_source(replace(cfg, source_commit="0" * 40))
