from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .config import require
from .io import read_jsonl, validate_chat_rows
from .logging import RunLogger


def training_plan(config: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(require(config, "data", "path"))
    validate_chat_rows(rows)
    token_budget_proxy = sum(
        len(message["content"].split()) for row in rows for message in row["messages"]
    )
    return {
        "validated_rows": len(rows),
        "whitespace_token_proxy": token_budget_proxy,
        "base_model": require(config, "model", "name"),
        "model_revision": config["model"].get("revision", "main"),
        "seed": require(config, "seed"),
        "adapter_output": require(config, "training", "output_dir"),
        "lora": require(config, "lora"),
    }


def run_training(config: dict[str, Any], *, dry_run: bool = False) -> tuple[RunLogger, dict[str, Any]]:
    plan = training_plan(config)
    logger = RunLogger(require(config, "output_dir"), require(config, "experiment"), config)
    logger.write_json("training_plan.json", plan)
    if dry_run:
        result = {**plan, "status": "validated_dry_run"}
        logger.write_json("summary.json", result)
        return logger, result

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("LoRA training requires requirements-gpu.txt") from exc

    seed = int(require(config, "seed"))
    random.seed(seed)
    set_seed(seed)
    model_cfg = require(config, "model")
    dtype_name = model_cfg.get("dtype", "bfloat16")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; use --dry-run to validate on CPU")
    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], revision=model_cfg.get("revision", "main"))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], revision=model_cfg.get("revision", "main"), torch_dtype=dtype, device_map="auto"
    )
    rows = read_jsonl(require(config, "data", "path"))
    rendered = [
        {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)} for row in rows
    ]
    dataset = Dataset.from_list(rendered)
    train_cfg = require(config, "training")
    lora_cfg = require(config, "lora")
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        max_seq_length=int(train_cfg["max_seq_length"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_strategy=train_cfg["save_strategy"],
        report_to="none",
        seed=seed,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, peft_config=peft_config)
    output = trainer.train()
    trainer.save_model(str(output_dir))
    result = {
        **plan,
        "status": "trained",
        "train_loss": float(output.training_loss),
        "metrics": {key: _json_number(value) for key, value in output.metrics.items()},
    }
    logger.write_json("summary.json", result)
    return logger, result


def _json_number(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=str))

