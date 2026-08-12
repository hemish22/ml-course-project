from pathlib import Path

from cmvs.config import Config, load_config


def test_load_config_returns_complete_typed_config() -> None:
    load_config.cache_clear()
    config = load_config(Path("config.yaml"))

    assert isinstance(config, Config)
    assert config.paths.raw == Path("data/raw")
    assert config.extract.fps == 1.0
    assert config.search.alpha == 0.6
