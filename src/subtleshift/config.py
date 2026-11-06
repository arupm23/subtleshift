from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a tracked experiment configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    raw = config_path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ConfigError("Non-JSON YAML requires PyYAML; install requirements.txt") from exc
        value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ConfigError("Top-level configuration must be a mapping")
    return value


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def require(config: dict[str, Any], *keys: str) -> Any:
    current: Any = config
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(current, dict) or key not in current:
            raise ConfigError(f"Missing required configuration key: {'.'.join(walked)}")
        current = current[key]
    return current

