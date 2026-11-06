from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class DataError(ValueError):
    """Raised for malformed experiment data."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.is_file():
        raise DataError(f"JSONL file does not exist: {input_path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"Invalid JSON on {input_path}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise DataError(f"Expected an object on {input_path}:{line_number}")
        rows.append(row)
    if not rows:
        raise DataError(f"JSONL file has no records: {input_path}")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def validate_prompt_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        for field in ("id", "prompt", "group", "split"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise DataError(f"Prompt row {index} requires non-empty string field '{field}'")
        if row["id"] in seen:
            raise DataError(f"Duplicate prompt id: {row['id']}")
        seen.add(row["id"])


def validate_chat_rows(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise DataError(f"Training row {index} requires at least two messages")
        for message in messages:
            if not isinstance(message, dict):
                raise DataError(f"Training row {index} contains a non-object message")
            if message.get("role") not in {"system", "user", "assistant"}:
                raise DataError(f"Training row {index} has an invalid role")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise DataError(f"Training row {index} has empty message content")

