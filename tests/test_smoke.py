from pathlib import Path

from subtleshift.config import load_config
from subtleshift.evaluation import run_evaluation
from subtleshift.training import run_training

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_smoke(tmp_path):
    config = load_config(ROOT / "configs/eval/smoke.yaml")
    config["data"]["path"] = str(ROOT / config["data"]["path"])
    config["output_dir"] = str(tmp_path)
    logger, summary = run_evaluation(config)
    assert summary["overall"]["n"] == 4
    assert summary["overall"]["misalignment_rate"] == 0.0
    assert (logger.path / "generations.jsonl").is_file()
    assert (logger.path / "manifest.json").is_file()


def test_training_dry_run(tmp_path):
    config = load_config(ROOT / "configs/train/lora_qwen25_05b.yaml")
    config["data"]["path"] = str(ROOT / config["data"]["path"])
    config["output_dir"] = str(tmp_path)
    logger, summary = run_training(config, dry_run=True)
    assert summary["status"] == "validated_dry_run"
    assert summary["validated_rows"] == 4
    assert (logger.path / "training_plan.json").is_file()

