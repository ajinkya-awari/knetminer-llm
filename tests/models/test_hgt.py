from __future__ import annotations

from datetime import datetime, timezone

import torch
import pytest

from knetminer_llm.data.normalize import normalize_snapshot
from knetminer_llm.graph.build import build_graph_views
from knetminer_llm.models.hgt import HGTConfig, HGTReranker


def graph_views():
    snapshot = normalize_snapshot(
        nodes=(
            {"id": "EFO_0001", "type": "disease", "label": "Disease"},
            {"id": "ENSG0001", "type": "target", "label": "Target"},
            {"id": "CHEMBL1", "type": "drug", "label": "Drug"},
        ),
        edges=(
            {"edge_id": "edge-dt", "source_id": "EFO_0001", "relation": "disease_target", "target_id": "ENSG0001", "score": 0.9, "provenance_ids": ["p-dt"]},
            {"edge_id": "edge-td", "source_id": "ENSG0001", "relation": "target_drug", "target_id": "CHEMBL1", "score": 0.8, "provenance_ids": ["p-td"]},
        ),
        source_url="https://platform.opentargets.org/",
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        request_hash="a" * 64,
        response_hash="b" * 64,
    )
    return build_graph_views(snapshot)


def test_hgt_projects_each_type_and_supports_finite_forward_backward() -> None:
    views = graph_views()
    model = HGTReranker(
        views.pyg.metadata(),
        HGTConfig(input_dims={node_type: 384 for node_type in ("disease", "target", "drug")}),
    )
    x_dict = {
        node_type: torch.randn(views.pyg[node_type].num_nodes, 384, requires_grad=True)
        for node_type in views.pyg.node_types
    }

    embeddings = model(x_dict, views.pyg.edge_index_dict)
    assert set(embeddings) == {"disease", "target", "drug"}
    assert all(value.shape[1] == 128 for value in embeddings.values())
    loss = sum(value.square().mean() for value in embeddings.values())
    loss.backward()
    assert all(torch.isfinite(value).all() for value in embeddings.values())


def test_hgt_has_four_heads_two_layers_and_missing_message_fallback() -> None:
    views = graph_views()
    model = HGTReranker(
        views.pyg.metadata(),
        HGTConfig(input_dims={node_type: 384 for node_type in ("disease", "target", "drug")}),
    )
    assert len(model.convs) == 2
    assert all(conv.heads == 4 for conv in model.convs)
    x_dict = {node_type: torch.ones(views.pyg[node_type].num_nodes, 384) for node_type in views.pyg.node_types}
    embeddings = model(x_dict, {})
    assert all(torch.isfinite(value).all() for value in embeddings.values())


def test_hgt_scores_only_supplied_observed_pairs() -> None:
    views = graph_views()
    model = HGTReranker(
        views.pyg.metadata(),
        HGTConfig(input_dims={node_type: 384 for node_type in ("disease", "target", "drug")}),
    )
    embeddings = {
        node_type: torch.ones(views.pyg[node_type].num_nodes, 128) for node_type in views.pyg.node_types
    }
    scores = model.score_observed_pairs(
        embeddings,
        (("disease", 0, "target", 0), ("target", 0, "drug", 0)),
    )
    assert scores.shape == (2,)


def test_hgt_config_requires_384_input_dimensions() -> None:
    with pytest.raises(ValueError, match="384"):
        HGTConfig(input_dims={node_type: 128 for node_type in ("disease", "target", "drug")})


def test_empty_hgt_scores_preserve_embedding_device_and_dtype() -> None:
    views = graph_views()
    model = HGTReranker(
        views.pyg.metadata(),
        HGTConfig(input_dims={node_type: 384 for node_type in ("disease", "target", "drug")}),
    )
    embeddings = {
        node_type: torch.ones(views.pyg[node_type].num_nodes, 128, dtype=torch.float64)
        for node_type in views.pyg.node_types
    }

    scores = model.score_observed_pairs(embeddings, ())

    assert scores.dtype == torch.float64
    assert scores.device == embeddings["disease"].device
