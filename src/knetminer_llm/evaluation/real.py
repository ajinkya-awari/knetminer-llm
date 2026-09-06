from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from knetminer_llm.contracts import Intent
from knetminer_llm.data.normalize import NormalizedSnapshot
from knetminer_llm.graph.build import build_graph_views
from knetminer_llm.retrieval.paths import retrieve_paths
from knetminer_llm.evaluation.questions import EvaluationQuestion


@dataclass(frozen=True)
class RealStructuralReport:
    total_rows: int
    answerable_rows: int
    unanswerable_rows: int
    answerable_with_paths: int
    unanswerable_without_paths: int
    citation_validity: float
    abstention_f1: float
    metric_status: str


def calculate_abstention_f1(
    *,
    expected_unanswerable: tuple[bool, ...],
    predicted_abstention: tuple[bool, ...],
) -> float:
    """Calculate F1 with abstention as the positive class."""

    if len(expected_unanswerable) != len(predicted_abstention):
        raise ValueError("abstention labels and predictions must have equal lengths")
    true_positive = sum(expected and predicted for expected, predicted in zip(expected_unanswerable, predicted_abstention))
    false_positive = sum(not expected and predicted for expected, predicted in zip(expected_unanswerable, predicted_abstention))
    false_negative = sum(expected and not predicted for expected, predicted in zip(expected_unanswerable, predicted_abstention))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def build_real_evaluation_set(snapshot: NormalizedSnapshot) -> tuple[EvaluationQuestion, ...]:
    disease_ids = tuple(sorted(node.stable_id for node in snapshot.nodes if node.node_type == "disease"))
    target_ids = tuple(sorted(node.stable_id for node in snapshot.nodes if node.node_type == "target"))
    disease_drug_sources = {edge.source_id for edge in snapshot.edges if edge.relation == "disease_drug"}
    target_drug_sources = {edge.source_id for edge in snapshot.edges if edge.relation == "target_drug"}
    if len(disease_ids) < 10 or len(target_ids) < 10:
        raise ValueError("real evaluation requires at least ten disease and target nodes")

    shared_by_target: dict[str, set[str]] = {}
    for edge in snapshot.edges:
        if edge.relation == "disease_target":
            shared_by_target.setdefault(edge.target_id, set()).add(edge.source_id)
    positive_pairs = sorted(
        {
            pair
            for diseases in shared_by_target.values()
            for pair in combinations(sorted(diseases), 2)
        }
    )
    all_pairs = set(combinations(disease_ids, 2))
    negative_pairs = sorted(all_pairs - set(positive_pairs))
    if len(positive_pairs) < 10 or len(negative_pairs) < 15:
        raise ValueError("real evaluation requires ten shared-target and fifteen negative pairs")

    answerable: list[EvaluationQuestion] = [
        EvaluationQuestion(
            question_id=f"real-answerable-{index:02d}",
            answerable=True,
            intent_name="disease_drugs",
            entity_ids=(disease_id,),
        )
        for index, disease_id in enumerate(disease_ids)
        if disease_id in disease_drug_sources
    ][:10]
    target_context_ids = tuple(
        target_id
        for target_id in target_ids
        if target_id in target_drug_sources or target_id in shared_by_target
    )[:10]
    answerable.extend(
        EvaluationQuestion(
            question_id=f"real-answerable-{index:02d}",
            answerable=True,
            intent_name="target_context",
            entity_ids=(target_id,),
        )
        for index, target_id in enumerate(target_context_ids, start=10)
    )
    answerable.extend(
        EvaluationQuestion(
            question_id=f"real-answerable-{index:02d}",
            answerable=True,
            intent_name="shared_targets",
            entity_ids=pair,
        )
        for index, pair in enumerate(positive_pairs[:10], start=20)
    )
    if len(answerable) != 30:
        raise ValueError("real evaluation could not construct thirty answerable questions")

    unanswerable = tuple(
        EvaluationQuestion(
            question_id=f"real-unanswerable-{index:02d}",
            answerable=False,
            intent_name="shared_targets",
            entity_ids=pair,
            failure_mode="missing_observed_shared_target_path",
        )
        for index, pair in enumerate(negative_pairs[:15])
    )
    return tuple(answerable) + unanswerable


def run_real_structural_evaluation(
    snapshot: NormalizedSnapshot,
    questions: tuple[EvaluationQuestion, ...] | None = None,
) -> RealStructuralReport:
    questions = questions or build_real_evaluation_set(snapshot)
    if len(questions) != 45:
        raise ValueError("real structural evaluation requires exactly 45 questions")
    views = build_graph_views(snapshot)
    answerable_rows = sum(question.answerable for question in questions)
    unanswerable_rows = len(questions) - answerable_rows
    answerable_with_paths = 0
    unanswerable_without_paths = 0
    citation_checks: list[bool] = []
    expected_unanswerable: list[bool] = []
    predicted_abstention: list[bool] = []
    for question in questions:
        result = retrieve_paths(views, Intent(name=question.intent_name, entity_ids=question.entity_ids))
        has_paths = bool(result.paths)
        expected_unanswerable.append(not question.answerable)
        predicted_abstention.append(not has_paths)
        if question.answerable:
            answerable_with_paths += has_paths
            citation_checks.extend(bool(path.citation_edge_ids) for path in result.paths)
        else:
            unanswerable_without_paths += not has_paths
    abstention_f1 = calculate_abstention_f1(
        expected_unanswerable=tuple(expected_unanswerable),
        predicted_abstention=tuple(predicted_abstention),
    )
    citation_validity = sum(citation_checks) / len(citation_checks) if citation_checks else 0.0
    return RealStructuralReport(
        total_rows=len(questions),
        answerable_rows=answerable_rows,
        unanswerable_rows=unanswerable_rows,
        answerable_with_paths=answerable_with_paths,
        unanswerable_without_paths=unanswerable_without_paths,
        citation_validity=citation_validity,
        abstention_f1=abstention_f1,
        metric_status="structural_retrieval_only_model_and_hgt_metrics_not_run",
    )
