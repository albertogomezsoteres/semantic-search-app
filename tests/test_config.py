from football_search.config import DEFAULT_CONFIG, load_config


def test_default_config_loads():
    cfg = load_config(DEFAULT_CONFIG)
    assert len(cfg.source_commit) == 40
    assert cfg.competitions
    assert cfg.model in cfg.candidates
    assert cfg.distance == "cosine"
    assert cfg.raw_base_url.endswith(f"{cfg.source_commit}/data")


def test_every_candidate_is_pinned():
    cfg = load_config(DEFAULT_CONFIG)
    for spec in cfg.candidates:
        assert len(spec.revision) == 40, spec.name
