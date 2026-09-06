from __future__ import annotations

from knetminer_llm.contracts import EvidenceEdge, EvidencePath, Intent
from knetminer_llm.data.normalize import NormalizedSnapshot
from knetminer_llm.evaluation.questions import EvaluationQuestion
from knetminer_llm.evaluation.real import calculate_abstention_f1
from knetminer_llm.graph.build import build_graph_views
from knetminer_llm.retrieval.paths import PathCandidate, retrieve_paths
from knetminer_llm.synthesis.local_llm import LocalQwenRunner
from knetminer_llm.synthesis.prompt import build_synthesis_input
from knetminer_llm.synthesis.validate import (
    deterministic_fallback,
    validate_model_output_with_status,
)


REVERSE_FOR_FORWARD = {
    "disease_target": "target_disease",
    "disease_drug": "drug_disease",
    "target_drug": "drug_target",
}


def run_model_backed_evaluation(
    snapshot: NormalizedSnapshot,
    questions: tuple[EvaluationQuestion, ...],
    *,
    runner: LocalQwenRunner,
) -> dict[str, object]:
    """Evaluate 30/15 observed-path answers while discarding raw generations."""

    if len(questions) != 45 or sum(question.answerable for question in questions) != 30:
        raise ValueError("model-backed evaluation requires exactly 30 answerable and 15 unanswerable rows")
    if len({question.question_id for question in questions}) != 45:
        raise ValueError("model-backed evaluation question IDs must be unique")
    views = build_graph_views(snapshot)
    forward_by_id = {edge.edge_id: edge for edge in snapshot.edges}
    rows: list[dict[str, object]] = []
    frozen_paths: dict[str, dict[str, object]] = {}
    model_invocations = 0
    model_outputs_accepted = 0
    expected_unanswerable: list[bool] = []
    predicted_abstention: list[bool] = []
    citation_checks: list[bool] = []

    for question in questions:
        intent = Intent(name=question.intent_name, entity_ids=question.entity_ids)
        retrieval = retrieve_paths(views, intent)
        paths = tuple(_to_evidence_path(candidate, forward_by_id) for candidate in retrieval.paths)
        if paths:
            raw_output = runner.generate(build_synthesis_input(intent, intent.entity_ids, paths))
            model_invocations += 1
            answer, accepted = validate_model_output_with_status(raw_output, intent, paths)
            model_outputs_accepted += int(accepted)
        else:
            answer = deterministic_fallback(intent, ())
        for candidate in retrieval.paths:
            frozen_paths[candidate.path_id] = _frozen_path(candidate, views)
        expected_unanswerable.append(not question.answerable)
        predicted_abstention.append(answer.status == "abstained")
        citation_checks.extend(citation in frozen_paths for citation in answer.citations)
        rows.append(
            {
                "question_id": question.question_id,
                "expected_answerable": question.answerable,
                "intent": intent.name,
                "entity_ids": list(intent.entity_ids),
                "status": answer.status,
                "citations": list(answer.citations),
                "text": answer.claims[0].text if answer.claims else None,
                "abstention_reason": answer.abstention_reason,
                "generation_attempted": bool(paths),
            }
        )

    citation_validity = sum(citation_checks) / len(citation_checks) if citation_checks else 0.0
    return {
        "summary": {
            "total_rows": 45,
            "answerable_rows": 30,
            "unanswerable_rows": 15,
            "model_invocations": model_invocations,
            "model_outputs_accepted": model_outputs_accepted,
            "deterministic_fallbacks": model_invocations - model_outputs_accepted,
            "citation_validity": citation_validity,
            "abstention_f1": calculate_abstention_f1(
                expected_unanswerable=tuple(expected_unanswerable),
                predicted_abstention=tuple(predicted_abstention),
            ),
            "model_revision": runner.config.model_revision,
            "raw_output_persisted": False,
        },
        "rows": rows,
        "evidence_paths": frozen_paths,
    }


def _to_evidence_path(
    candidate: PathCandidate, forward_by_id: dict[str, EvidenceEdge]
) -> EvidencePath:
    edges: list[EvidenceEdge] = []
    for edge_id in candidate.edge_ids:
        if edge_id.startswith("reverse:"):
            forward = forward_by_id[edge_id.removeprefix("reverse:")]
            edges.append(
                EvidenceEdge(
                    edge_id=edge_id,
                    source_id=forward.target_id,
                    source_type=forward.target_type,
                    relation=REVERSE_FOR_FORWARD[forward.relation],
                    target_id=forward.source_id,
                    target_type=forward.source_type,
                    release=forward.release,
                    score=forward.score,
                    provenance_ids=forward.provenance_ids,
                    source_url=forward.source_url,
                    licence=forward.licence,
                    observed=True,
                    citable=False,
                )
            )
        else:
            edges.append(forward_by_id[edge_id])
    return EvidencePath(path_id=candidate.path_id, edges=tuple(edges))


def _frozen_path(candidate: PathCandidate, views) -> dict[str, object]:
    relations = [
        views.networkx[candidate.node_ids[index]][candidate.node_ids[index + 1]][edge_id]["relation"]
        for index, edge_id in enumerate(candidate.edge_ids)
    ]
    return {
        "path_id": candidate.path_id,
        "node_ids": list(candidate.node_ids),
        "relations": relations,
        "citation_edge_ids": list(candidate.citation_edge_ids),
    }
