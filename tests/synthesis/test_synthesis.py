from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from knetminer_llm.contracts import EvidenceEdge, EvidencePath, Intent
from knetminer_llm.synthesis.local_llm import (
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
    LocalQwenConfig,
)
from knetminer_llm.synthesis.prompt import build_synthesis_input
from knetminer_llm.synthesis.validate import deterministic_fallback, validate_model_output


def evidence_paths():
    return (
        EvidencePath(
            path_id="path:edge-dd",
            edges=(
                EvidenceEdge(
                    edge_id="edge-dd",
                    source_id="EFO_0001",
                    source_type="disease",
                    relation="disease_drug",
                    target_id="CHEMBL1",
                    target_type="drug",
                    release="26.06",
                    score=0.8,
                    provenance_ids=("p-dd",),
                    source_url="https://platform.opentargets.org/",
                    licence="CC0",
                    observed=True,
                    citable=True,
                ),
            ),
        ),
    )


def test_synthesis_input_contains_typed_data_not_raw_instructions() -> None:
    request = build_synthesis_input(
        Intent(name="disease_drugs", entity_ids=("EFO_0001",)),
        ("EFO_0001",),
        evidence_paths(),
    )
    assert request["intent"] == "disease_drugs"
    assert "instruction" not in json.dumps(request).lower()
    assert request["evidence_paths"][0]["path_id"].startswith("path:")


def test_synthesis_input_rejects_entity_ids_that_disagree_with_intent() -> None:
    with pytest.raises(ValueError, match="intent"):
        build_synthesis_input(
            Intent(name="disease_drugs", entity_ids=("EFO_0001",)),
            ("EFO_9999",),
            evidence_paths(),
        )


def test_valid_model_json_is_grounded_to_supplied_paths() -> None:
    paths = evidence_paths()
    output = {
        "status": "answered",
        "claims": [{"text": "An observed evidence path is available.", "evidence_path_ids": [paths[0].path_id]}],
        "citations": [paths[0].path_id],
        "abstention_reason": None,
    }
    answer = validate_model_output(json.dumps(output), Intent(name="disease_drugs", entity_ids=("EFO_0001",)), paths)
    assert answer.status == "answered"
    assert answer.claims[0].evidence_path_ids == (paths[0].path_id,)


def test_valid_path_id_does_not_authorize_unsupported_model_prose() -> None:
    paths = evidence_paths()
    raw = json.dumps(
        {
            "status": "answered",
            "claims": [{"text": "CHEMBL1 treats diabetes in patients.", "evidence_path_ids": [paths[0].path_id]}],
            "citations": [paths[0].path_id],
            "abstention_reason": None,
        }
    )

    answer = validate_model_output(raw, Intent(name="disease_drugs", entity_ids=("EFO_0001",)), paths)

    assert answer.status == "answered"
    assert answer.claims[0].text == "An observed evidence path is available."
    assert "treats diabetes" not in answer.model_dump_json()


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps({"status": "answered", "claims": [{"text": "Invented", "evidence_path_ids": ["unknown"]}], "citations": ["unknown"], "abstention_reason": None}),
        json.dumps({"status": "answered", "claims": [], "citations": [], "abstention_reason": None, "chain_of_thought": "hidden"}),
    ],
)
def test_invalid_model_output_uses_deterministic_fallback_without_raw_text(raw: str) -> None:
    paths = evidence_paths()
    answer = validate_model_output(raw, Intent(name="disease_drugs", entity_ids=("EFO_0001",)), paths)
    assert raw not in answer.model_dump_json()
    assert answer.status == "answered"
    assert answer.citations == (paths[0].path_id,)


def test_no_evidence_returns_structured_abstention_with_empty_citations() -> None:
    answer = deterministic_fallback(Intent(name="disease_drugs", entity_ids=("EFO_0001",)), ())
    assert answer.status == "abstained"
    assert answer.citations == ()
    assert answer.abstention_reason


def test_local_qwen_config_is_pinned_greedy_and_bounded() -> None:
    config = LocalQwenConfig()
    assert config.model_id == QWEN_MODEL_ID
    assert config.model_revision == QWEN_MODEL_REVISION
    assert config.max_new_tokens == 256
    assert config.temperature == 0.0
    with pytest.raises(ValueError):
        LocalQwenConfig(model_revision="", max_new_tokens=257)


def test_local_qwen_rejects_unpinned_model_identity() -> None:
    with pytest.raises(ValueError):
        LocalQwenConfig(model_id="other/model")


# UNEXECUTED (laptop policy forbids pytest) — regression for Bug #1:
# pydantic.ValidationError must be caught in validate_model_output so that
# model JSON with Pydantic-invalid field types (e.g. list where tuple required)
# always falls back deterministically instead of propagating.
def test_pydantic_validation_error_triggers_deterministic_fallback() -> None:
    paths = evidence_paths()
    # Pydantic strict mode rejects a list for evidence_path_ids (requires tuple).
    raw = json.dumps({
        "status": "answered",
        "claims": [{"text": "An observed evidence path is available.", "evidence_path_ids": ["not-a-tuple-but-a-list"]}],
        "citations": ["not-a-tuple-but-a-list"],
        "abstention_reason": None,
    })
    # The path_id "not-a-tuple-but-a-list" is not in supplied_paths so AnswerPayload
    # model_validator will raise ValidationError; fallback must be returned.
    answer = validate_model_output(raw, Intent(name="disease_drugs", entity_ids=("EFO_0001",)), paths)
    assert answer.status == "answered"
    assert answer.citations == (paths[0].path_id,)
    assert "not-a-tuple-but-a-list" not in answer.model_dump_json()
