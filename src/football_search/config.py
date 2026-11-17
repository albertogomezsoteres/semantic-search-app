"""Load config.yaml into typed, read-only settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


def configure_logging(level: int = logging.INFO) -> None:
    """Send log lines to stdout with a timestamp and the module name."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@dataclass(frozen=True)
class ModelSpec:
    """An embedding model pinned to an exact revision, plus its prefix conventions."""

    name: str
    revision: str
    query_prefix: str = ""
    passage_prefix: str = ""


@dataclass(frozen=True)
class Config:
    """All pipeline and app settings."""

    source_repo: str
    source_commit: str
    request_delay_s: float
    competitions: tuple[tuple[int, int], ...]
    raw_dir: Path
    staged_dir: Path
    chroma_dir: Path
    reports_dir: Path
    eval_queries: Path
    collection: str
    batch_size: int
    template: str
    distance: str
    model: ModelSpec
    candidates: tuple[ModelSpec, ...]
    top_k: int
    max_top_k: int

    @property
    def raw_base_url(self) -> str:
        """Base URL for raw files at the pinned commit."""
        return f"https://raw.githubusercontent.com/{self.source_repo}/{self.source_commit}/data"

    @property
    def shots_path(self) -> Path:
        return self.staged_dir / "shots.parquet"

    @property
    def matches_path(self) -> Path:
        return self.staged_dir / "matches.parquet"

    def model_spec(self, name: str) -> ModelSpec:
        """Return the candidate spec for a model name."""
        for spec in self.candidates:
            if spec.name == name:
                return spec
        raise KeyError(f"Model {name!r} is not listed under embedding.candidates")


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    """Read a YAML config file. Relative paths are resolved against the file's folder."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    base = path.resolve().parent

    candidates = tuple(ModelSpec(**c) for c in raw["embedding"]["candidates"])
    by_name = {c.name: c for c in candidates}
    model_name = raw["embedding"]["model"]
    if model_name not in by_name:
        raise ValueError(f"embedding.model {model_name!r} must be one of the candidates")

    paths = raw["paths"]
    return Config(
        source_repo=raw["source"]["repo"],
        source_commit=raw["source"]["commit"],
        request_delay_s=float(raw["source"]["request_delay_s"]),
        competitions=tuple(
            (int(c["competition_id"]), int(c["season_id"])) for c in raw["competitions"]
        ),
        raw_dir=base / paths["raw"],
        staged_dir=base / paths["staged"],
        chroma_dir=base / paths["chroma"],
        reports_dir=base / paths["reports"],
        eval_queries=base / paths["eval_queries"],
        collection=raw["index"]["collection"],
        batch_size=int(raw["index"]["batch_size"]),
        template=raw["index"]["template"],
        distance=raw["index"]["distance"],
        model=by_name[model_name],
        candidates=candidates,
        top_k=int(raw["search"]["top_k"]),
        max_top_k=int(raw["search"]["max_top_k"]),
    )
