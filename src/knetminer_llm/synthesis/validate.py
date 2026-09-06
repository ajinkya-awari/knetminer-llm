from __future__ import annotations

import json

from pydantic import ValidationError

from knetminer_llm.contracts import AnswerPayload, Claim, EvidencePath, Intent


MODEL_KEYS = {"status", "claims", "citations", "abstention_reason"}
MAX_MODEL_OUTPUT_CHARS = 20_000
SUPPORTED_CLAIM_TEXT = "An observed evidence path is available."


def validate_model_output(
    raw_output: str,
    intent: Intent,
    supplied_paths: tuple[EvidencePath, ...],
) -> AnswerPayload:
    answer, _ = validate_model_output_with_status(raw_output, intent, supplied_paths)
    return answer


def validate_model_output_with_status(
    raw_output: str,
    intent: Intent,
    supplied_paths: tuple[EvidencePath, ...],
) -> tuple[AnswerPayload, bool]:
    """Validate untrusted output and report whether strict validation accepted it."""

    try:
        if len(raw_output) > MAX_MODEL_OUTPUT_CHARS:
            raise ValueError("model output exceeds bounded size")
        decoded = json.loads(raw_output)
        if not isinstance(decoded, dict) or set(decoded) != MODEL_KEYS:
            raise ValueError("model output schema is not exact")
        claims = tuple(Claim(**claim) for claim in decoded["claims"])
        answer = AnswerPayload(
            status=decoded["status"],
            intent=intent,
            claims=claims,
            evidence_paths=supplied_paths,
            citations=tuple(decoded["citations"]),
            abstention_reason=decoded["abstention_reason"],
        )
        _validate_supported_claim_text(answer)
        return answer, True
    except (TypeError, ValueError, json.JSONDecodeError, ValidationError):
        return deterministic_fallback(intent, supplied_paths), False


def deterministic_fallback(intent: Intent, supplied_paths: tuple[EvidencePath, ...]) -> AnswerPayload:
    if not supplied_paths:
        return AnswerPayload(
            status="abstained",
            intent=intent,
            claims=(),
            evidence_paths=(),
            citations=(),
            abstention_reason="No observed evidence path is available.",
        )
    path_id = supplied_paths[0].path_id
    return AnswerPayload(
        status="answered",
        intent=intent,
        claims=(Claim(text=SUPPORTED_CLAIM_TEXT, evidence_path_ids=(path_id,)),),
        evidence_paths=supplied_paths,
        citations=(path_id,),
        abstention_reason=None,
    )


def _validate_supported_claim_text(answer: AnswerPayload) -> None:
    if answer.status == "abstained":
        return
    if any(claim.text != SUPPORTED_CLAIM_TEXT for claim in answer.claims):
        raise ValueError("claim text is not supported by the deterministic evidence contract")
