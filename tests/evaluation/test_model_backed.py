from __future__ import annotations

import json
from datetime import datetime, timezone

from knetminer_llm.app import load_frozen_bundle
from knetminer_llm.data.normalize import normalize_snapshot
from knetminer_llm.evaluation.bundle import write_frozen_result_bundle
from knetminer_llm.evaluation.model_backed import run_model_backed_evaluation
from knetminer_llm.evaluation.questions import EvaluationQuestion
from knetminer_llm.synthesis.local_llm import LocalQwenConfig, LocalQwenRunner, QWEN_MODEL_REVISION


def model_evaluation_snapshot():
    return normalize_snapshot(
        nodes=(
            {"id": "EFO_0001", "type": "disease", "label": "Disease one"},
            {"id": "EFO_0002", "type": "disease", "label": "Disease two"},
            {"id": "ENSG0001", "type": "target", "label": "Target"},
            {"id": "CHEMBL1", "type": "drug", "label": "Drug"},
        ),
        edges=(
            {"edge_id": "dt-1", "source_id": "EFO_0001", "relation": "disease_target", "target_id": "ENSG0001", "score": 0.8, "provenance_ids": ["p-dt"]},
            {"edge_id": "dd-1", "source_id": "EFO_0001", "relation": "disease_drug", "target_id": "CHEMBL1", "score": 0.9, "provenance_ids": ["p-dd"]},
        ),
        source_url="https://platform.opentargets.org/",
        retrieved_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
        request_hash="a" * 64,
        response_hash="b" * 64,
    )


def evaluation_questions() -> tuple[EvaluationQuestion, ...]:
    answerable = tuple(
        EvaluationQuestion(f"answerable-{index:02d}", True, "disease_drugs", ("EFO_0001",))
        for index in range(30)
    )
    unanswerable = tuple(
        EvaluationQuestion(
            f"unanswerable-{index:02d}",
            False,
            "shared_targets",
            ("EFO_0001", "EFO_0002"),
            "missing_path",
        )
        for index in range(15)
    )
    return answerable + unanswerable


def test_model_evaluation_validates_30_15_without_persisting_raw_generation() -> None:
    calls = []
    runner = LocalQwenRunner(LocalQwenConfig(), lambda typed: calls.append(typed) or "PRIVATE RAW TEXT")

    result = run_model_backed_evaluation(
        model_evaluation_snapshot(),
        evaluation_questions(),
        runner=runner,
    )

    assert result["summary"]["answerable_rows"] == 30
    assert result["summary"]["unanswerable_rows"] == 15
    assert result["summary"]["model_invocations"] == 30
    assert result["summary"]["model_outputs_accepted"] == 0
    assert result["summary"]["deterministic_fallbacks"] == 30
    assert result["summary"]["citation_validity"] == 1.0
    assert result["summary"]["abstention_f1"] == 1.0
    assert result["summary"]["raw_output_persisted"] is False
    assert len(result["rows"]) == 45
    assert len(calls) == 30
    assert "PRIVATE RAW TEXT" not in json.dumps(result)
    assert all("raw_output" not in row for row in result["rows"])


def test_bundle_writer_hashes_and_validates_every_artifact(tmp_path) -> None:
    snapshot = model_evaluation_snapshot()
    runner = LocalQwenRunner(LocalQwenConfig(), lambda typed: "not-json")
    evaluation = run_model_backed_evaluation(snapshot, evaluation_questions(), runner=runner)

    manifest = write_frozen_result_bundle(
        snapshot,
        evaluation,
        snapshot_hash="c" * 64,
        output_dir=tmp_path,
        model_revision=QWEN_MODEL_REVISION,
    )

    assert set(manifest["artifacts"]) == {
        "benchmark_summary.json",
        "per_query.jsonl",
        "demo_bundle.json",
    }
    assert all(len(value) == 64 for value in manifest["artifacts"].values())
    assert (tmp_path / "manifest.json").is_file()
    bundle = load_frozen_bundle(tmp_path / "demo_bundle.json", expected_snapshot_hash="c" * 64)
    assert len(bundle.answers) == 45
    assert bundle.provenance.model_revision == QWEN_MODEL_REVISION
