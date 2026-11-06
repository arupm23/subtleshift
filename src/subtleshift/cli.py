from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .evaluation import run_evaluation
from .training import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="subtleshift")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command in {"smoke", "evaluate"}:
        logger, summary = run_evaluation(config)
    else:
        logger, summary = run_training(config, dry_run=args.dry_run)
    print(json.dumps({"run_id": logger.run_id, "path": str(logger.path), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

