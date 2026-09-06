from __future__ import annotations

from knetminer_llm.contracts import EvidencePath, Intent


def build_synthesis_input(
    intent: Intent,
    entity_ids: tuple[str, ...],
    evidence_paths: tuple[EvidencePath, ...],
) -> dict:
    if entity_ids != intent.entity_ids:
        raise ValueError("synthesis entity IDs must match the validated intent")
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
    }
