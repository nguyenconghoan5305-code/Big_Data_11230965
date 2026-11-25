import json

import pytest

from bft65b.config import ProjectConfig, load_config


def test_assignment_defaults():
    config = load_config()
    assert config.financial_start == "1984-03-31"
    assert config.financial_end == "2026-06-30"
    assert config.modelling_end == "2025-06-30"
    assert config.fields_verified is True


def test_config_override_and_invalid_dates(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"seed": 42}), encoding="utf-8")
    assert load_config(path).seed == 42
    path.write_text(json.dumps({"financial_end": "2026-05-31"}), encoding="utf-8")
    with pytest.raises(ValueError, match="quarter-end"):
        load_config(path)
    with pytest.raises(ValueError, match="Field lists"):
        ProjectConfig(financial_fields=("CERT", "CERT"))


def test_paths_resolve_external_override(tmp_path, monkeypatch):
    root = tmp_path / "external"
    monkeypatch.setenv("BFT65B_DATA_ROOT", str(root))
    assert ProjectConfig().paths(tmp_path)["raw"] == root.resolve()
