import json

import pytest

from subtleshift.config import ConfigError, config_hash, load_config


def test_json_compatible_yaml_loads_without_optional_dependency(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps({"seed": 7}), encoding="utf-8")
    assert load_config(path) == {"seed": 7}


def test_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_missing_config_is_clear(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")

