from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone

import pytest

from knetminer_llm.data.opentargets import FetchResult
from knetminer_llm.data.stage import load_staged_snapshot, normalize_sample_payloads


def result(payload: dict, request: str, response: str) -> FetchResult:
    return FetchResult(
        payload=payload,
        request_hash=request,
        response_hash=response,
        cache_hit=False,
        attempts=1,
    )


def test_real_payload_normalization_preserves_three_relations_and_hash_provenance() -> None:
    disease = result(
        {
            "data": {
                "disease": {
                    "id": "MONDO_0005148",
                    "name": "type 2 diabetes mellitus",
                    "associatedTargets": {
                        "rows": [
                            {
                                "score": 0.9,
                                "target": {
                                    "id": "ENSG00000187486",
                                    "approvedSymbol": "KCNJ11",
                                    "approvedName": "potassium channel",
                                },
                            }
                        ]
                    },
                    "drugAndClinicalCandidates": {
                        "rows": [
                            {
                                "id": "clinical-1",
                                "maxClinicalStage": "APPROVAL",
                                "drug": {"id": "CHEMBL25", "name": "ASPIRIN"},
                            }
                        ]
                    },
                }
            }
        },
        "a" * 64,
        "b" * 64,
    )
    target = result(
        {
            "data": {
                "target": {
                    "id": "ENSG00000187486",
                    "approvedSymbol": "KCNJ11",
                    "approvedName": "potassium channel",
                    "drugAndClinicalCandidates": {
                        "rows": [
                            {
                                "id": "clinical-2",
                                "maxClinicalStage": "PHASE_2",
                                "drug": {"id": "CHEMBL50", "name": "DRUG"},
                            }
                        ]
                    },
                }
            }
        },
        "c" * 64,
        "d" * 64,
    )

    snapshot = normalize_sample_payloads(
        disease_results=(disease,),
        target_results=(target,),
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )

    assert {edge.relation for edge in snapshot.edges} == {
        "disease_target",
        "disease_drug",
        "target_drug",
    }
    assert {str(edge.source_url) for edge in snapshot.edges} == {
        "https://api.platform.opentargets.org/api/v4/graphql"
    }
    assert snapshot.manifest.node_count == 4
    assert snapshot.manifest.edge_count == 3
    assert snapshot.edges[0].provenance_ids in (("b" * 64,), ("d" * 64,))


def test_duplicate_staged_edges_merge_native_and_response_provenance() -> None:
    disease = result(
        {
            "data": {
                "disease": {
                    "id": "MONDO_0005148",
                    "name": "type 2 diabetes mellitus",
                    "associatedTargets": {"rows": []},
                    "drugAndClinicalCandidates": {
                        "rows": [
                            {"id": "clinical-1", "drug": {"id": "CHEMBL25", "name": "ASPIRIN"}},
                            {"id": "clinical-2", "drug": {"id": "CHEMBL25", "name": "ASPIRIN"}},
                        ]
                    },
                }
            }
        },
        "a" * 64,
        "b" * 64,
    )

    snapshot = normalize_sample_payloads(
        disease_results=(disease,),
        target_results=(),
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )

    assert len(snapshot.edges) == 1
    assert snapshot.edges[0].provenance_ids == ("b" * 64, "clinical-1", "clinical-2")


def test_staged_json_round_trip_restores_strict_tuple_contracts(tmp_path) -> None:
    snapshot = normalize_sample_payloads(
        disease_results=(
            result(
                {
                    "data": {
                        "disease": {
                            "id": "MONDO_0005148",
                            "name": "type 2 diabetes mellitus",
                            "associatedTargets": {"rows": []},
                            "drugAndClinicalCandidates": {"rows": []},
                        }
                    }
                },
                "a" * 64,
                "b" * 64,
            ),
        ),
        target_results=(),
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [node.model_dump(mode="json") for node in snapshot.nodes],
                "edges": [edge.model_dump(mode="json") for edge in snapshot.edges],
                "manifest": snapshot.manifest.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    loaded = load_staged_snapshot(path, expected_artifact_hash=hashlib.sha256(path.read_bytes()).hexdigest())

    assert loaded.nodes == snapshot.nodes
    assert loaded.edges == snapshot.edges
    assert loaded.manifest == snapshot.manifest


def test_staged_json_rejects_unsafe_node_identifier(tmp_path) -> None:
    payload = {
        "nodes": [{"stable_id": "unsafe id", "node_type": "disease", "label": "Disease"}],
        "edges": [],
        "manifest": {
            "source_release": "26.06",
            "licence": "CC0",
            "source_url": "https://platform.opentargets.org/",
            "retrieved_at": "2026-08-19T00:00:00Z",
            "request_hash": "a" * 64,
            "response_hash": "b" * 64,
            "node_count": 1,
            "edge_count": 0,
        },
    }
    path = tmp_path / "unsafe-node.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="stable"):
        load_staged_snapshot(path, expected_artifact_hash=hashlib.sha256(path.read_bytes()).hexdigest())


def test_staged_json_rehydration_rejects_missing_edge_endpoint(tmp_path) -> None:
    snapshot = normalize_sample_payloads(
        disease_results=(
            result(
                {
                    "data": {
                        "disease": {
                            "id": "MONDO_0005148",
                            "name": "type 2 diabetes mellitus",
                            "associatedTargets": {
                                "rows": [
                                    {
                                        "score": 0.9,
                                        "target": {
                                            "id": "ENSG00000187486",
                                            "approvedSymbol": "KCNJ11",
                                            "approvedName": "potassium channel",
                                        },
                                    }
                                ]
                            },
                            "drugAndClinicalCandidates": {"rows": []},
                        }
                    }
                },
                "a" * 64,
                "b" * 64,
            ),
        ),
        target_results=(),
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    payload = {
        "nodes": [node.model_dump(mode="json") for node in snapshot.nodes if node.stable_id != "ENSG00000187486"],
        "edges": [edge.model_dump(mode="json") for edge in snapshot.edges],
        "manifest": {**snapshot.manifest.model_dump(mode="json"), "node_count": 1},
    }
    path = tmp_path / "missing-endpoint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="endpoint"):
        load_staged_snapshot(path, expected_artifact_hash=hashlib.sha256(path.read_bytes()).hexdigest())


def test_staged_json_rehydration_rejects_node_type_mismatch(tmp_path) -> None:
    snapshot = normalize_sample_payloads(
        disease_results=(
            result(
                {
                    "data": {
                        "disease": {
                            "id": "MONDO_0005148",
                            "name": "type 2 diabetes mellitus",
                            "associatedTargets": {
                                "rows": [
                                    {
                                        "score": 0.9,
                                        "target": {
                                            "id": "ENSG00000187486",
                                            "approvedSymbol": "KCNJ11",
                                            "approvedName": "potassium channel",
                                        },
                                    }
                                ]
                            },
                            "drugAndClinicalCandidates": {"rows": []},
                        }
                    }
                },
                "a" * 64,
                "b" * 64,
            ),
        ),
        target_results=(),
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    nodes = [node.model_dump(mode="json") for node in snapshot.nodes]
    for node in nodes:
        if node["stable_id"] == "ENSG00000187486":
            node["node_type"] = "drug"
    path = tmp_path / "type-mismatch.json"
    path.write_text(
        json.dumps(
            {
                "nodes": nodes,
                "edges": [edge.model_dump(mode="json") for edge in snapshot.edges],
                "manifest": snapshot.manifest.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="type"):
        load_staged_snapshot(path, expected_artifact_hash=hashlib.sha256(path.read_bytes()).hexdigest())


def test_staged_json_requires_matching_artifact_hash(tmp_path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash"):
        load_staged_snapshot(path, expected_artifact_hash="0" * 64)
