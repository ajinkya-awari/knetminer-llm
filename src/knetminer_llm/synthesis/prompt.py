from __future__ import annotations

from knetminer_llm.contracts import EvidencePath, Intent
from knetminer_llm.synthesis.validate import SUPPORTED_CLAIM_TEXT


def build_synthesis_input(
    intent: Intent,
    entity_ids: tuple[str, ...],
    evidence_paths: tuple[EvidencePath, ...],
) -> dict:
    if entity_ids != intent.entity_ids:
        raise ValueError("synthesis entity IDs must match the validated intent")
    if evidence_paths:
        path_id = evidence_paths[0].path_id
        allowed_answer = {
            "status": "answered",
            "claims": [
                {
                    "text": SUPPORTED_CLAIM_TEXT,
                    "evidence_path_ids": [path_id],
                }
            ],
            "citations": [path_id],
            "abstention_reason": None,
        }
    else:
        allowed_answer = {
            "status": "abstained",
            "claims": [],
            "citations": [],
            "abstention_reason": "No observed evidence path is available.",
        }
    return {
        "intent": intent.name,
        "entity_ids": list(entity_ids),
        "evidence_paths": [
            {
                "path_id": path.path_id,
                "edge_ids": [edge.edge_id for edge in path.edges if edge.citable],
                "relations": [edge.relation for edge in path.edges if edge.citable],
                "scores": [edge.score for edge in path.edges if edge.citable],
            }
            for path in evidence_paths
        ],
        "constraints": {
            "claims_must_cite_supplied_path_ids": True,
            "no_clinical_advice": True,
            "input_mode": "typed_evidence_only",
        },
        "response_contract": {
            "format": "json_only",
            "allowed_answer": allowed_answer,
        },
    }
