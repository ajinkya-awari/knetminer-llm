from __future__ import annotations

from datetime import datetime, timezone

import pytest

from knetminer_llm.contracts import Intent
from knetminer_llm.data.normalize import normalize_snapshot
from knetminer_llm.graph.build import build_graph_views
from knetminer_llm.retrieval.paths import PathCandidate, rerank_paths, retrieve_paths


def views():
    snapshot = normalize_snapshot(
        nodes=(
            {"id": "EFO_0001", "type": "disease", "label": "Disease A"},
            {"id": "EFO_0002", "type": "disease", "label": "Disease B"},
            {"id": "ENSG0001", "type": "target", "label": "Target"},
            {"id": "CHEMBL1", "type": "drug", "label": "Drug"},
        ),
        edges=(
            {"edge_id": "edge-a", "source_id": "EFO_0001", "relation": "disease_target", "target_id": "ENSG0001", "score": 0.9, "provenance_ids": ["p-a"]},
            {"edge_id": "edge-b", "source_id": "EFO_0002", "relation": "disease_target", "target_id": "ENSG0001", "score": 0.8, "provenance_ids": ["p-b"]},
            {"edge_id": "edge-c", "source_id": "ENSG0001", "relation": "target_drug", "target_id": "CHEMBL1", "score": 0.7, "provenance_ids": ["p-c"]},
            {"edge_id": "edge-d", "source_id": "EFO_0001", "relation": "disease_drug", "target_id": "CHEMBL1", "score": 0.6, "provenance_ids": ["p-d"]},
        ),
        source_url="https://platform.opentargets.org/",
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        request_hash="a" * 64,
        response_hash="b" * 64,
    )
    return build_graph_views(snapshot)


def test_shared_targets_returns_observed_path_and_geometric_mean_score() -> None:
    result = retrieve_paths(
        views(),
        Intent(name="shared_targets", entity_ids=("EFO_0001", "EFO_0002")),
    )

    path = next(path for path in result.paths if path.edge_ids == ("edge-a", "reverse:edge-b"))
    assert path.edge_ids == ("edge-a", "reverse:edge-b")
    assert path.citation_edge_ids == ("edge-a", "edge-b")
    assert path.symbolic_score == pytest.approx((0.9 * 0.8) ** 0.5)


def test_disease_drugs_and_target_context_use_typed_observed_paths() -> None:
    disease_drugs = retrieve_paths(views(), Intent(name="disease_drugs", entity_ids=("EFO_0001",)))
    assert any("edge-d" in path.citation_edge_ids for path in disease_drugs.paths)
    assert all(len(path.edge_ids) <= 3 for path in disease_drugs.paths)

    context = retrieve_paths(views(), Intent(name="target_context", entity_ids=("ENSG0001",)))
    assert any("edge-c" in path.citation_edge_ids for path in context.paths)
    assert any("edge-a" in path.citation_edge_ids for path in context.paths)


def test_caps_and_deterministic_reranking_are_enforced() -> None:
    bounded = retrieve_paths(
        views(),
        Intent(name="disease_drugs", entity_ids=("EFO_0001",)),
        max_expansions=1,
    )
    assert bounded.truncated is True
    assert bounded.expansions == 1

    paths = (
        PathCandidate("path-b", ("edge-b",), ("EFO_0001", "CHEMBL1"), ("edge-b",), 0.5),
        PathCandidate("path-a", ("edge-a",), ("EFO_0001", "ENSG0001"), ("edge-a",), 0.5),
    )
    ranked = rerank_paths(paths, {"path-a": 0.5, "path-b": 0.5})
    assert tuple(path.path_id for path in ranked) == ("path-a", "path-b")
    with pytest.raises(ValueError, match="unknown"):
        rerank_paths(paths, {"path-z": 1.0})

    with pytest.raises(ValueError, match="finite"):
        rerank_paths(paths, {"path-a": float("nan"), "path-b": 0.5})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_expansions": 2_001},
        {"max_seconds": 2.01},
        {"max_candidates": 101},
        {"max_candidates": 0},
        {"max_outputs": 6},
        {"max_outputs": 0},
    ],
)
def test_retrieval_rejects_caps_outside_the_project_contract(kwargs) -> None:
    with pytest.raises(ValueError, match="cap"):
        retrieve_paths(
            views(),
            Intent(name="disease_drugs", entity_ids=("EFO_0001",)),
            **kwargs,
        )


def test_shared_targets_does_not_treat_shared_drugs_as_targets() -> None:
    snapshot = normalize_snapshot(
        nodes=(
            {"id": "EFO_0001", "type": "disease", "label": "Disease A"},
            {"id": "EFO_0002", "type": "disease", "label": "Disease B"},
            {"id": "ENSG0001", "type": "target", "label": "Unused target"},
            {"id": "CHEMBL1", "type": "drug", "label": "Shared drug"},
        ),
        edges=(
            {"edge_id": "edge-a-drug", "source_id": "EFO_0001", "relation": "disease_drug", "target_id": "CHEMBL1", "score": 0.8, "provenance_ids": ["p-a"]},
            {"edge_id": "edge-b-drug", "source_id": "EFO_0002", "relation": "disease_drug", "target_id": "CHEMBL1", "score": 0.8, "provenance_ids": ["p-b"]},
        ),
        source_url="https://platform.opentargets.org/",
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        request_hash="a" * 64,
        response_hash="b" * 64,
    )

    result = retrieve_paths(
        build_graph_views(snapshot),
        Intent(name="shared_targets", entity_ids=("EFO_0001", "EFO_0002")),
    )

    assert result.paths == ()


def test_disease_drugs_rejects_context_hop_through_another_disease() -> None:
    snapshot = normalize_snapshot(
        nodes=(
            {"id": "EFO_0001", "type": "disease", "label": "Disease A"},
            {"id": "EFO_0002", "type": "disease", "label": "Disease B"},
            {"id": "ENSG0001", "type": "target", "label": "Shared target"},
            {"id": "CHEMBL1", "type": "drug", "label": "Other disease drug"},
        ),
        edges=(
            {"edge_id": "edge-a", "source_id": "EFO_0001", "relation": "disease_target", "target_id": "ENSG0001", "score": 0.8, "provenance_ids": ["p-a"]},
            {"edge_id": "edge-b", "source_id": "EFO_0002", "relation": "disease_target", "target_id": "ENSG0001", "score": 0.8, "provenance_ids": ["p-b"]},
            {"edge_id": "edge-drug", "source_id": "EFO_0002", "relation": "disease_drug", "target_id": "CHEMBL1", "score": 0.8, "provenance_ids": ["p-drug"]},
        ),
        source_url="https://platform.opentargets.org/",
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        request_hash="a" * 64,
        response_hash="b" * 64,
    )

    result = retrieve_paths(
        build_graph_views(snapshot),
        Intent(name="disease_drugs", entity_ids=("EFO_0001",)),
    )

    assert result.paths == ()
