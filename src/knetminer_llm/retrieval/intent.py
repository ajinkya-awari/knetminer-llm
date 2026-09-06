from __future__ import annotations

import re

from knetminer_llm.contracts import Intent


SUPPORTED_INTENTS = frozenset({"shared_targets", "disease_drugs", "target_context"})
MAX_INPUT_CHARS = 300
INSTRUCTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+the\s+(?:system|developer)\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+the\s+system", re.IGNORECASE),
)


def validate_user_text(text: str) -> None:
    if not isinstance(text, str):
        raise ValueError("input must be text data")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError("input exceeds the 300 character cap")
    if any(pattern.search(text) for pattern in INSTRUCTION_PATTERNS):
        raise ValueError("instruction-like input is rejected")


def parse_intent(name: str, entity_ids: tuple[str, ...], raw_text: str) -> Intent:
    validate_user_text(raw_text)
    if name not in SUPPORTED_INTENTS:
        raise ValueError(f"unsupported intent: {name}")
    return Intent(name=name, entity_ids=entity_ids)
