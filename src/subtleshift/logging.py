from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import config_hash
from .io import write_jsonl


class RunLogger:
    def __init__(self, root: str | Path, experiment: str, config: dict[str, Any]):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = f"{timestamp}-{experiment}-{config_hash(config)}"
        self.path = Path(root) / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": self.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash(config),
            "config": config,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "git_commit": _git_commit(),
            },
        }
        self.write_json("manifest.json", manifest)

    def write_json(self, name: str, value: dict[str, Any]) -> None:
        (self.path / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_records(self, name: str, rows: list[dict[str, Any]]) -> None:
        write_jsonl(self.path / name, rows)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None

