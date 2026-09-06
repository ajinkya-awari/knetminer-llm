from __future__ import annotations

from dataclasses import dataclass

from knetminer_llm.contracts import AnswerPayload, Claim, EvidenceEdge, EvidencePath, Intent
from knetminer_llm.evaluation.questions import EvaluationQuestion


@dataclass(frozen=True)
class EvaluationReport:
    total_rows: int
    answerable_rows: int
    unanswerable_rows: int
    valid_payloads: int
    metrics: None
    metric_status: str


def run_structural_evaluation(questions: tuple[EvaluationQuestion, ...]) -> EvaluationReport:
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("evaluation question IDs must be unique")
    valid_payloads = 0
    for question in questions:
        payload = _fixture_payload(question)
        if payload.status == ("answered" if question.answerable else "abstained"):
            valid_payloads += 1
    return EvaluationReport(
        total_rows=len(questions),
        answerable_rows=sum(question.answerable for question in questions),
        unanswerable_rows=sum(not question.answerable for question in questions),
        valid_payloads=valid_payloads,
        metrics=None,
        metric_status="blocked_until_real_snapshot_and_model",
    )


def _fixture_payload(question: EvaluationQuestion) -> AnswerPayload:
    intent = Intent(name=question.intent_name, entity_ids=question.entity_ids)
    if not question.answerable:
        return AnswerPayload(
            status="abstained",
            intent=intent,
            claims=(),
            evidence_paths=(),
            citations=(),
            abstention_reason=question.failure_mode or "synthetic unanswerable case",
        )
    _source_type = {
        "shared_targets": "disease",
        "disease_drugs": "disease",
        "target_context": "target",
    }[question.intent_name]
    path_id = f"fixture-path-{question.question_id}"
    path = EvidencePath(
        path_id=path_id,
        edges=(
            EvidenceEdge(
                edge_id=f"fixture-edge-{question.question_id}",
                source_id=question.entity_ids[0],
                source_type=_source_type,
                relation={
                    "shared_targets": "disease_target",
                    "disease_drugs": "disease_drug",
                    "target_context": "target_drug",
                }[question.intent_name],
                target_id="CHEMBL1" if question.intent_name != "shared_targets" else "ENSG0001",
                target_type="drug" if question.intent_name != "shared_targets" else "target",
                release="26.06",
                score=0.5,
                provenance_ids=(f"fixture-provenance-{question.question_id}",),
                source_url="https://platform.opentargets.org/",
                licence="CC0",
                observed=True,
                citable=True,
            ),
        ),
    )
    return AnswerPayload(
        status="answered",
        intent=intent,
        claims=(Claim(text="Synthetic observed evidence is available.", evidence_path_ids=(path_id,)),),
        evidence_paths=(path,),
        citations=(path_id,),
        abstention_reason=None,
    )
