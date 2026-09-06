from __future__ import annotations

from datetime import datetime, timezone

from pydantic import ValidationError
import pytest

from knetminer_llm.contracts import (
    AnswerPayload,
    Claim,
    EvidenceEdge,
    EvidencePath,
    Intent,
    ResolvedEntity,
    SnapshotManifest,
)


SOURCE_URL = "https://platform.opentargets.org/"


def test_claim_contract_requires_nonempty_text_and_path_ids() -> None:
    claim = Claim(text="Observed claim", evidence_path_ids=("path-1",))
    assert claim.evidence_path_ids == ("path-1",)

    with pytest.raises(ValidationError):
        Claim(text="", evidence_path_ids=())

    with pytest.raises(ValidationError):
        Claim(text="   ", evidence_path_ids=("path-1",))


def edge(
    edge_id: str,
    source_id: str,
    source_type: str,
    relation: str,
    target_id: str,
    target_type: str,
) -> EvidenceEdge:
    return EvidenceEdge(
        edge_id=edge_id,
        source_id=source_id,
        source_type=source_type,
        relation=relation,
        target_id=target_id,
        target_type=target_type,
        release="26.06",
        score=0.8,
        provenance_ids=(f"prov-{edge_id}",),
        source_url=SOURCE_URL,
        licence="CC0",
        observed=True,
        citable=relation in {"disease_target", "disease_drug", "target_drug"},
    )


def test_valid_contract_bundle_accepts_observed_citable_evidence() -> None:
    disease_a = ResolvedEntity(
        node_type="disease",
        stable_id="EFO_0001",
        label="Disease A",
        resolution_method="exact_id",
    )
    disease_b = ResolvedEntity(
        node_type="disease",
        stable_id="EFO_0002",
        label="Disease B",
        resolution_method="exact_alias",
    )
    intent = Intent(name="shared_targets", entity_ids=(disease_a.stable_id, disease_b.stable_id))
    path = EvidencePath(
        path_id="path-1",
        edges=(
            edge("e-1", disease_a.stable_id, "disease", "disease_target", "ENSG0001", "target"),
            edge("e-2", "ENSG0001", "target", "target_disease", disease_b.stable_id, "disease"),
        ),
    )
    answer = AnswerPayload(
        status="answered",
        intent=intent,
        claims=(Claim(text="The diseases share an observed target.", evidence_path_ids=("path-1",)),),
        evidence_paths=(path,),
        citations=("path-1",),
        abstention_reason=None,
    )

    assert answer.status == "answered"
    assert answer.evidence_paths[0].path_id == "path-1"


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ResolvedEntity(
            node_type="disease",
            stable_id="EFO_0001",
            label="Disease A",
            resolution_method="exact_id",
            unsupported_field="must fail",
        )

    with pytest.raises(ValidationError):
        ResolvedEntity(
            node_type="disease",
            stable_id=123,
            label="Disease A",
            resolution_method="exact_id",
        )

    with pytest.raises(ValidationError):
        ResolvedEntity(
            node_type="target",
            stable_id="ENSG0001",
            label="Target",
            resolution_method="semantic",
            semantic_score="0.8",
            semantic_margin=0.1,
        )

    entity = ResolvedEntity(
        node_type="disease",
        stable_id="EFO_0001",
        label="Disease A",
        resolution_method="exact_id",
    )
    with pytest.raises(ValidationError):
        entity.label = "Changed"


def test_semantic_entity_requires_score_and_margin_thresholds() -> None:
    with pytest.raises(ValidationError):
        ResolvedEntity(
            node_type="target",
            stable_id="ENSG0001",
            label="Target",
            resolution_method="semantic",
            semantic_score=0.80,
            semantic_margin=0.04,
        )

    accepted = ResolvedEntity(
        node_type="target",
        stable_id="ENSG0001",
        label="Target",
        resolution_method="semantic",
        semantic_score=0.75,
        semantic_margin=0.05,
    )
    assert accepted.semantic_score == 0.75


def test_intent_contract_enforces_supported_arity() -> None:
    with pytest.raises(ValidationError):
        Intent(name="shared_targets", entity_ids=("EFO_0001",))

    with pytest.raises(ValidationError):
        Intent(name="disease_drugs", entity_ids=("EFO_0001", "EFO_0002"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("node_type", "gene"),
        ("stable_id", "not a stable id"),
        ("resolution_method", "guess"),
    ],
)
def test_entity_contract_rejects_invalid_values(field: str, value: str) -> None:
    payload = {
        "node_type": "disease",
        "stable_id": "EFO_0001",
        "label": "Disease A",
        "resolution_method": "exact_id",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        ResolvedEntity(**payload)


def test_edge_contract_rejects_unknown_relation_and_bad_provenance() -> None:
    with pytest.raises(ValidationError):
        edge("e-1", "EFO_0001", "disease", "invented_relation", "ENSG0001", "target")

    with pytest.raises(ValidationError):
        EvidenceEdge(
            edge_id="e-1",
            source_id="EFO_0001",
            source_type="disease",
            relation="disease_target",
            target_id="ENSG0001",
            target_type="target",
            release="26.06",
            score=0.8,
            provenance_ids=(),
            source_url=SOURCE_URL,
            licence="CC0",
            observed=True,
            citable=True,
        )

    with pytest.raises(ValidationError):
        edge("e-1", "EFO_0001", "disease", "disease_target", "CHEMBL1", "drug")


@pytest.mark.parametrize(
    ("relation", "source_type", "target_type", "source_id", "target_id"),
    [
        ("disease_drug", "disease", "drug", "EFO_0001", "CHEMBL1"),
        ("target_drug", "target", "drug", "ENSG0001", "CHEMBL1"),
        ("target_disease", "target", "disease", "ENSG0001", "EFO_0001"),
        ("drug_disease", "drug", "disease", "CHEMBL1", "EFO_0001"),
        ("drug_target", "drug", "target", "CHEMBL1", "ENSG0001"),
    ],
)
def test_all_approved_relation_endpoint_pairs_are_accepted(
    relation: str,
    source_type: str,
    target_type: str,
    source_id: str,
    target_id: str,
) -> None:
    accepted = edge("e-approved", source_id, source_type, relation, target_id, target_type)
    assert accepted.relation == relation


def test_edge_contract_rejects_bad_release_licence_url_and_score() -> None:
    with pytest.raises(ValidationError):
        EvidenceEdge(
            edge_id="e-1",
            source_id="EFO_0001",
            source_type="disease",
            relation="disease_target",
            target_id="ENSG0001",
            target_type="target",
            release="25.06",
            score=1.2,
            provenance_ids=("prov-e-1",),
            source_url="https://example.com/not-allow-listed",
            licence="CC-BY",
            observed=True,
            citable=True,
        )


def test_reverse_transport_edge_is_observed_but_not_citable() -> None:
    reverse = edge("e-reverse", "CHEMBL1", "drug", "drug_target", "ENSG0001", "target")
    assert reverse.observed is True
    assert reverse.citable is False


def test_path_rejects_transport_only_evidence() -> None:
    with pytest.raises(ValidationError):
        EvidencePath(
            path_id="transport-only",
            edges=(edge("e-reverse", "CHEMBL1", "drug", "drug_target", "ENSG0001", "target"),),
        )


def test_path_rejects_non_chained_edges_and_more_than_three_edges() -> None:
    edges = (
        edge("e-1", "EFO_0001", "disease", "disease_target", "ENSG0001", "target"),
        edge("e-2", "ENSG0002", "target", "target_drug", "CHEMBL1", "drug"),
        edge("e-3", "CHEMBL1", "drug", "drug_target", "ENSG0003", "target"),
        edge("e-4", "ENSG0003", "target", "target_disease", "EFO_0002", "disease"),
    )

    with pytest.raises(ValidationError):
        EvidencePath(path_id="bad-chain", edges=edges[:2])

    with pytest.raises(ValidationError):
        EvidencePath(path_id="too-long", edges=edges)


def test_answer_rejects_unknown_claim_path_and_abstention_citations() -> None:
    intent = Intent(name="disease_drugs", entity_ids=("EFO_0001",))
    path = EvidencePath(
        path_id="path-1",
        edges=(edge("e-1", "EFO_0001", "disease", "disease_drug", "CHEMBL1", "drug"),),
    )

    with pytest.raises(ValidationError):
        AnswerPayload(
            status="answered",
            intent=intent,
            claims=(Claim(text="Claim", evidence_path_ids=("unknown-path",)),),
            evidence_paths=(path,),
            citations=("unknown-path",),
            abstention_reason=None,
        )

    second_path = EvidencePath(
        path_id="path-2",
        edges=(edge("e-2", "EFO_0001", "disease", "disease_drug", "CHEMBL2", "drug"),),
    )
    with pytest.raises(ValidationError, match="citations"):
        AnswerPayload(
            status="answered",
            intent=intent,
            claims=(Claim(text="Claim", evidence_path_ids=("path-1",)),),
            evidence_paths=(path, second_path),
            citations=("path-2",),
            abstention_reason=None,
        )

    with pytest.raises(ValidationError):
        AnswerPayload(
            status="abstained",
            intent=intent,
            claims=(),
            evidence_paths=(),
            citations=("path-1",),
            abstention_reason="",
        )

    with pytest.raises(ValidationError):
        AnswerPayload(
            status="answered",
            intent=intent,
            claims=(Claim(text="Claim", evidence_path_ids=("path-1",)),),
            evidence_paths=(path, path),
            citations=("path-1",),
            abstention_reason=None,
        )


def test_snapshot_manifest_requires_release_licence_hashes_and_counts() -> None:
    manifest = SnapshotManifest(
        source_release="26.06",
        licence="CC0",
        source_url=SOURCE_URL,
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        request_hash="a" * 64,
        response_hash="b" * 64,
        node_count=3,
        edge_count=2,
    )

    assert manifest.source_release == "26.06"

    with pytest.raises(ValidationError):
        SnapshotManifest(
            source_release="25.06",
            licence="CC0",
            source_url=SOURCE_URL,
            retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            request_hash="a" * 64,
            response_hash="b" * 64,
            node_count=-1,
            edge_count=2,
        )
