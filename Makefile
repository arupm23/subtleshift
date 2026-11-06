.PHONY: test lint smoke dry-run-train

test:
	python -m pytest

lint:
	python -m ruff check src tests

smoke:
	PYTHONPATH=src python -m subtleshift.cli smoke --config configs/eval/smoke.yaml

dry-run-train:
	PYTHONPATH=src python -m subtleshift.cli train --config configs/train/lora_qwen25_05b.yaml --dry-run

