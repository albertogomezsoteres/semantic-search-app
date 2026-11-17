"""Extract: download raw StatsBomb JSON at a pinned commit into data/raw/.

Files are written exactly as published, so a broken transform can be rerun
without downloading again. Files already on disk are skipped.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

from football_search.config import Config, configure_logging, load_config

log = logging.getLogger(__name__)

SOURCE_FILE = "source.json"


class Downloader:
    """Polite, idempotent fetcher for raw files at one commit."""

    def __init__(self, cfg: Config, session: requests.Session | None = None, retries: int = 3):
        self.cfg = cfg
        self.session = session or requests.Session()
        self.retries = retries
        self.downloaded = 0
        self.skipped = 0

    def fetch(self, rel_path: str) -> Path:
        """Download data/<rel_path> to raw_dir/<rel_path> unless it already exists."""
        dest = self.cfg.raw_dir / rel_path
        if dest.exists():
            self.skipped += 1
            return dest

        url = f"{self.cfg.raw_base_url}/{rel_path}"
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == self.retries:
                    raise
                wait = 2**attempt
                log.warning("GET %s failed (%s); retrying in %ss", url, exc, wait)
                time.sleep(wait)

        # Write to a temp file first so an interrupted run never leaves half a file behind.
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)
        self.downloaded += 1
        time.sleep(self.cfg.request_delay_s)
        return dest


def check_source(cfg: Config) -> None:
    """Record the source commit, and refuse to mix files from two different commits."""
    marker = cfg.raw_dir / SOURCE_FILE
    if marker.exists():
        recorded = json.loads(marker.read_text())["commit"]
        if recorded != cfg.source_commit:
            raise RuntimeError(
                f"{cfg.raw_dir} holds data from commit {recorded}, but config.yaml pins "
                f"{cfg.source_commit}. Delete {cfg.raw_dir} and run `make data` again."
            )
        return
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"repo": cfg.source_repo, "commit": cfg.source_commit}) + "\n")


def find_missing_competitions(
    competitions: list[dict], wanted: tuple[tuple[int, int], ...]
) -> list[tuple[int, int]]:
    """Return the (competition_id, season_id) pairs that are not in competitions.json."""
    available = {(c["competition_id"], c["season_id"]) for c in competitions}
    return [pair for pair in wanted if pair not in available]


def extract(cfg: Config) -> None:
    """Download competitions, matches, events, and lineups for every configured competition."""
    check_source(cfg)
    dl = Downloader(cfg)

    competitions = json.loads(dl.fetch("competitions.json").read_text())
    missing = find_missing_competitions(competitions, cfg.competitions)
    if missing:
        raise ValueError(f"Not found in competitions.json at this commit: {missing}")

    for comp_id, season_id in cfg.competitions:
        matches = json.loads(dl.fetch(f"matches/{comp_id}/{season_id}.json").read_text())
        name = (
            f"{matches[0]['competition']['competition_name']} {matches[0]['season']['season_name']}"
        )
        log.info("%s: %d matches", name, len(matches))
        for match in matches:
            dl.fetch(f"events/{match['match_id']}.json")
            dl.fetch(f"lineups/{match['match_id']}.json")

    log.info("Done: %d downloaded, %d already on disk", dl.downloaded, dl.skipped)


def main() -> None:
    configure_logging()
    extract(load_config())


if __name__ == "__main__":
    main()
