from __future__ import annotations

import json

import pytest

from knetminer_llm.evaluation.cli import (
    directory_manifest_hash,
    load_questions,
    resolve_model_source,
    write_json_artifact,
)


def test_question_loader_requires_exact_30_15_shape(tmp_path) -> None:
    questions = [
        {
            "question_id": f"a-{index}",
            "answerable": True,
            "intent_name": "disease_drugs",
            "entity_ids": ["EFO_0001"],
            "failure_mode": None,
        }
        for index in range(30)
    ] + [
        {
            "question_id": f"u-{index}",
            "answerable": False,
            "intent_name": "shared_targets",
            "entity_ids": ["EFO_0001", "EFO_0002"],
            "failure_mode": "missing_path",
        }
        for index in range(15)
    ]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")

    loaded = load_questions(path)

    assert len(loaded) == 45
    assert loaded[0].entity_ids == ("EFO_0001",)


def test_artifact_and_directory_hashes_change_with_content_not_order(tmp_path) -> None:
    output = tmp_path / "result.json"
    first = write_json_artifact(output, {"b": 1, "a": 2})
    second = write_json_artifact(output, {"a": 2, "b": 1})
    assert first == second

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("one", encoding="utf-8")
    before = directory_manifest_hash(model)
    (model / "config.json").write_text("two", encoding="utf-8")
    assert directory_manifest_hash(model) != before


def test_model_source_fails_closed_without_local_files_or_download_approval(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="download approval"):
        resolve_model_source(
            model_id="example/model",
            revision="a" * 40,
            local_path=tmp_path / "missing",
            allow_download=False,
        )
