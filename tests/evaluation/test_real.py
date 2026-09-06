from __future__ import annotations

from datetime import datetime, timezone

from knetminer_llm.data.normalize import normalize_snapshot
from knetminer_llm.evaluation.real import (
    build_real_evaluation_set,
    calculate_abstention_f1,
    run_real_structural_evaluation,
)


def evaluation_snapshot():
    nodes = []
    edges = []
    for index in range(10):
        disease = f"EFO_{index:04d}"
        personal_target = f"ENSG{index:011d}"
        drug = f"CHEMBL{index + 1}"
        group_target = "ENSG00000000010" if index < 5 else "ENSG00000000011"
        nodes.extend(
            [
                {"id": disease, "type": "disease", "label": disease},
                {"id": personal_target, "type": "target", "label": personal_target},
                {"id": drug, "type": "drug", "label": drug},
            ]
        )
        edges.extend(
            [
                {"edge_id": f"dt-{index}", "source_id": disease, "relation": "disease_target", "target_id": group_target, "score": 0.8, "provenance_ids": [f"p-dt-{index}"]},
                {"edge_id": f"dt-personal-{index}", "source_id": disease, "relation": "disease_target", "target_id": personal_target, "score": 0.7, "provenance_ids": [f"p-dtp-{index}"]},
                {"edge_id": f"dd-{index}", "source_id": disease, "relation": "disease_drug", "target_id": drug, "score": None, "provenance_ids": [f"p-dd-{index}"]},
                {"edge_id": f"td-{index}", "source_id": personal_target, "relation": "target_drug", "target_id": drug, "score": None, "provenance_ids": [f"p-td-{index}"]},
            ]
        )
    nodes.extend(
        [
            {"id": "ENSG00000000010", "type": "target", "label": "shared-a"},
            {"id": "ENSG00000000011", "type": "target", "label": "shared-b"},
        ]
    )
    return normalize_snapshot(
        nodes=tuple(nodes),
        edges=tuple(edges),
        source_url="https://platform.opentargets.org/",
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        request_hash="a" * 64,
        response_hash="b" * 64,
    )


def test_real_evaluation_constructs_exact_30_15_and_structural_metrics() -> None:
    snapshot = evaluation_snapshot()
    questions = build_real_evaluation_set(snapshot)
    report = run_real_structural_evaluation(snapshot, questions)

    assert len(questions) == 45
    assert len({question.question_id for question in questions}) == 45
    assert sum(question.answerable for question in questions) == 30
    assert report.unanswerable_rows == 15
    assert report.answerable_with_paths == 30
    assert report.unanswerable_without_paths == 15
    assert report.citation_validity == 1.0
    assert report.abstention_f1 == 1.0
    assert report.metric_status.endswith("not_run")


def test_abstention_f1_counts_answerable_abstentions_as_false_positives() -> None:
    score = calculate_abstention_f1(
        expected_unanswerable=(False, False, True, True),
        predicted_abstention=(True, False, True, False),
    )

    assert score == 0.5
