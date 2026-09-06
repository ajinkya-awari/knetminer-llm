from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from knetminer_llm.data.normalize import normalize_snapshot
from knetminer_llm.graph.build import build_graph_views, validate_graph_views


def snapshot():
    return normalize_snapshot(
        nodes=(
            {"id": "EFO_0001", "type": "disease", "label": "Disease"},
            {"id": "ENSG0001", "type": "target", "label": "Target"},
            {"id": "CHEMBL1", "type": "drug", "label": "Drug"},
        ),
        edges=(
            {
                "edge_id": "edge-dt",
                "source_id": "EFO_0001",
                "relation": "disease_target",
                "target_id": "ENSG0001",
                "score": 0.9,
                "provenance_ids": ["assoc-dt"],
            },
            {
                "edge_id": "edge-dd",
                "source_id": "EFO_0001",
                "relation": "disease_drug",
                "target_id": "CHEMBL1",
                "score": 0.7,
                "provenance_ids": ["assoc-dd"],
            },
            {
                "edge_id": "edge-td",
                "source_id": "ENSG0001",
                "relation": "target_drug",
                "target_id": "CHEMBL1",
                "score": 0.8,
                "provenance_ids": ["assoc-td"],
            },
        ),
        source_url="https://platform.opentargets.org/",
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        request_hash="a" * 64,
        response_hash="b" * 64,
    )


def test_graph_views_have_exact_nodes_and_forward_relations() -> None:
    views = build_graph_views(snapshot())

    assert set(views.networkx.nodes[node]["node_type"] for node in views.networkx.nodes) == {
        "disease",
        "target",
        "drug",
    }
    assert set(views.forward_edge_ids) == {"edge-dt", "edge-dd", "edge-td"}
    assert set(views.pyg.node_types) == {"disease", "target", "drug"}
    assert views.pyg.validate(raise_on_error=True) is True


def test_reverse_edges_are_transport_only_and_link_forward_ids() -> None:
    views = build_graph_views(snapshot())

    reverse_edges = [
        attributes
        for _, _, _, attributes in views.networkx.edges(keys=True, data=True)
        if not attributes["citable"]
    ]
    assert len(reverse_edges) == 3
    assert {edge["forward_edge_id"] for edge in reverse_edges} == {
        "edge-dt",
        "edge-dd",
        "edge-td",
    }
    assert all(edge["transport_only"] for edge in reverse_edges)
    assert views.networkx["EFO_0001"]["ENSG0001"]["edge-dt"]["score"] == 0.9


def test_networkx_and_pyg_share_stable_ids_and_forward_edge_ids() -> None:
    views = build_graph_views(snapshot())
    validate_graph_views(views)

    for node_type in ("disease", "target", "drug"):
        networkx_ids = {
            node
            for node, attrs in views.networkx.nodes(data=True)
            if attrs["node_type"] == node_type
        }
        assert set(views.pyg[node_type].node_ids) == networkx_ids

    edge_types = {
        "disease_target": ("disease", "disease_target", "target"),
        "disease_drug": ("disease", "disease_drug", "drug"),
        "target_drug": ("target", "target_drug", "drug"),
    }
    for edge_type, pyg_edge_type in edge_types.items():
        assert set(views.pyg[pyg_edge_type].edge_ids) == {
            edge_id for edge_id in views.forward_edge_ids if views.edge_relations[edge_id] == edge_type
        }


def test_graph_validation_rejects_missing_forward_edge_id() -> None:
    views = build_graph_views(snapshot())
    broken = replace(views, forward_edge_ids=("edge-dt", "missing"))

    with pytest.raises(ValueError, match="forward edge IDs"):
        validate_graph_views(broken)


def test_graph_validation_rejects_pyg_identity_drift() -> None:
    views = build_graph_views(snapshot())
    views.pyg["disease"].node_ids = ["EFO_TAMPERED"]

    with pytest.raises(ValueError, match="node IDs"):
        validate_graph_views(views)
