from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


@dataclass(frozen=True)
class LocalQwenConfig:
    model_id: str = QWEN_MODEL_ID
    model_revision: str = QWEN_MODEL_REVISION
    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.model_id != QWEN_MODEL_ID:
            raise ValueError(f"Project 06 requires the pinned model {QWEN_MODEL_ID}")
        if self.model_revision != QWEN_MODEL_REVISION:
            raise ValueError("Project 06 requires the pinned Qwen revision")
        if not self.model_revision.strip():
            raise ValueError("a pinned local model revision is required")
        if not 1 <= self.max_new_tokens <= 256:
            raise ValueError("new-token limit must be between one and 256")
        if self.temperature != 0.0 or self.do_sample or self.dtype != "float32":
            raise ValueError("Project 06 synthesis requires greedy float32 decoding")


class LocalQwenRunner:
    """Adapter for an explicitly supplied local generator; it never downloads a model."""

    def __init__(self, config: LocalQwenConfig, generator: Callable[[dict], str] | None = None) -> None:
        self.config = config
        self._generator = generator

    def generate(self, typed_input: dict) -> str:
        if self._generator is None:
            raise RuntimeError("local model generator is not loaded")
        return self._generator(typed_input)

    @classmethod
    def from_local_files(cls, config: LocalQwenConfig, model_path: Path) -> "LocalQwenRunner":
        """Load the pinned model without network access and expose typed-input generation."""

        if not model_path.is_dir():
            raise FileNotFoundError(f"local Qwen model directory does not exist: {model_path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=config.model_revision,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=config.model_revision,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        model.eval()

        def generate(typed_input: dict[str, Any]) -> str:
            serialized = json.dumps(typed_input, sort_keys=True, separators=(",", ":"))
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": serialized}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prompt_length = inputs["input_ids"].shape[-1]
            return tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return cls(config, generate)
