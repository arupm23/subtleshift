from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Generator(Protocol):
    def generate(self, prompt: str, *, seed: int, generation: dict[str, Any]) -> str: ...


@dataclass
class MockGenerator:
    name: str = "deterministic-safe-mock"

    def generate(self, prompt: str, *, seed: int, generation: dict[str, Any]) -> str:
        del seed, generation
        return (
            "A responsible response should be honest about uncertainty, protect safety and privacy, "
            "and ask for appropriate oversight when the consequences are significant. "
            f"For this question, I would first clarify the relevant facts: {prompt[:48]}"
        )


class TransformersGenerator:
    def __init__(self, model_config: dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Transformers backend requires requirements-gpu.txt") from exc

        dtype_name = model_config.get("dtype", "auto")
        dtype = getattr(torch, dtype_name) if dtype_name != "auto" else "auto"
        name = model_config["name"]
        revision = model_config.get("revision", "main")
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=dtype, device_map="auto"
        )
        adapter_path = model_config.get("adapter_path")
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Adapter loading requires PEFT") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def generate(self, prompt: str, *, seed: int, generation: dict[str, Any]) -> str:
        import torch

        torch.manual_seed(seed)
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        temperature = float(generation.get("temperature", 0.0))
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(generation.get("max_new_tokens", 256)),
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=float(generation.get("top_p", 1.0)))
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_generator(model_config: dict[str, Any]) -> Generator:
    backend = model_config.get("backend")
    if backend == "mock":
        return MockGenerator(model_config.get("name", "deterministic-safe-mock"))
    if backend == "transformers":
        return TransformersGenerator(model_config)
    raise ValueError(f"Unsupported model backend: {backend!r}")

