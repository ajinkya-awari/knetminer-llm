from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    answerable: bool
    intent_name: str
    entity_ids: tuple[str, ...]
    failure_mode: str | None = None


def build_fixed_evaluation_set() -> tuple[EvaluationQuestion, ...]:
    intents = (
        ("shared_targets", ("EFO_0001", "EFO_0002")),
        ("disease_drugs", ("EFO_0001",)),
        ("target_context", ("ENSG0001",)),
    )
    answerable = tuple(
        EvaluationQuestion(
            question_id=f"answerable-{index:02d}",
            answerable=True,
            intent_name=intents[index % len(intents)][0],
            entity_ids=intents[index % len(intents)][1],
        )
        for index in range(30)
    )
    failure_modes = ("unsupported_intent", "ambiguous_entity", "missing_path", "clinical_advice", "injection")
    unanswerable = tuple(
        EvaluationQuestion(
            question_id=f"unanswerable-{index:02d}",
            answerable=False,
            intent_name="disease_drugs",
            entity_ids=("EFO_0001",),
            failure_mode=failure_modes[index % len(failure_modes)],
        )
        for index in range(15)
    )
    return answerable + unanswerable
