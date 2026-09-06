from __future__ import annotations

from knetminer_llm.evaluation.questions import build_fixed_evaluation_set
from knetminer_llm.evaluation.run import run_structural_evaluation


def test_fixed_evaluation_set_has_exact_30_15_partition() -> None:
    questions = build_fixed_evaluation_set()
    assert len(questions) == 45
    assert sum(question.answerable for question in questions) == 30
    assert sum(not question.answerable for question in questions) == 15
    assert len({question.question_id for question in questions}) == 45


def test_synthetic_evaluation_validates_all_rows_without_fabricating_metrics() -> None:
    report = run_structural_evaluation(build_fixed_evaluation_set())
    assert report.total_rows == 45
    assert report.answerable_rows == 30
    assert report.unanswerable_rows == 15
    assert report.valid_payloads == 45
    assert report.metrics is None
    assert report.metric_status == "blocked_until_real_snapshot_and_model"
